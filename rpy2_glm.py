"""
rpy2 GLM integration for PyDLNM

This module provides interfaces to run R's GLM directly from Python, ensuring exact coefficient matches with pure R
implementations.

Every interface instance keeps its R objects (data frame, family, the fitted ``glm`` object, ...) in a PRIVATE R
environment whose parent is ``baseenv()``: two fits never interfere with each other, a user's R variables can neither
mask nor be overwritten by them, and nothing is written into R's global environment.
"""

import inspect
import re
import warnings
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# R is started by rpy2 from the R_HOME of the environment (set it before importing PyDLNM if the R found on the
# PATH is not the one with the required packages); this module never overrides it.

try:
    import rpy2.robjects as ro
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

from basis import CrossBasis


# Name of the response column in the R data frame (``death ~ cb + ...`` as in the DLNM examples)
RESPONSE = 'death'

# Prefix of the objects that carry glm() arguments (weights, offset, subset) in the private R environment: glm()
# looks them up in the data frame first, so user covariates must not be able to shadow them.
_ARG_PREFIX = '.pydlnm_'

# Families that can be named by a string: lower-case spelling -> the R function of package stats
# (R itself is case sensitive: its family is ``Gamma``, while ``gamma`` is the mathematical function)
_FAMILIES = {
    'poisson': 'poisson',
    'quasipoisson': 'quasipoisson',
    'gaussian': 'gaussian',
    'gamma': 'Gamma',
    'binomial': 'binomial',
    'quasibinomial': 'quasibinomial',
    'inverse.gaussian': 'inverse.gaussian',
    'inverse_gaussian': 'inverse.gaussian',
}

# Arguments of R's glm() that the interfaces pass on (anything else is rejected with a TypeError)
GLM_ARGUMENTS = ('weights', 'offset', 'subset', 'control')

# Arguments of stats::glm.control() accepted in ``control={...}``
_CONTROL_ARGUMENTS = ('epsilon', 'maxit', 'trace')


# --------------------------------------------------------------------------------------------------------------
# private R environment
# --------------------------------------------------------------------------------------------------------------
def new_r_environment():
    """A fresh R environment whose parent is ``baseenv()`` (functions of other packages are called as ``pkg::f``)."""
    return ro.baseenv['new.env'](parent=ro.baseenv)


def r_eval(env, code: str):
    """Evaluate R code in the private environment ``env``."""
    return ro.baseenv['eval'](ro.baseenv['parse'](text=code), env)


def r_function(name: str):
    """``pkg::name`` as a callable (evaluated in the base environment: user-defined R objects cannot mask it)."""
    return r_eval(ro.baseenv, name)


def r_data_frame(columns: Dict[str, np.ndarray]):
    """R data.frame with exactly these column names (no ``make.names``) from a dict of float vectors."""
    n_rows = len(next(iter(columns.values())))
    frame = ro.ListVector({name: ro.FloatVector(np.asarray(values, dtype=float)) for name, values in columns.items()})
    return ro.baseenv['structure'](frame, **{'class': 'data.frame',
                                             'row.names': ro.IntVector(range(1, n_rows + 1))})


def r_quote(name: str) -> str:
    """Backtick-quoted R name (valid in formulas and with ``$``)."""
    return '`' + name + '`'


# --------------------------------------------------------------------------------------------------------------
# input handling shared by the interfaces
# --------------------------------------------------------------------------------------------------------------
def as_vector(values, n: int, what: str) -> np.ndarray:
    """A numeric vector of length ``n``: list, array, column vector (n, 1), Series or one-column DataFrame."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"'{what}' must be a vector: got a DataFrame with {values.shape[1]} columns")
        values = values.iloc[:, 0]
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    arr = np.asarray(values)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"'{what}' must be a vector, got an array of shape {arr.shape}")
    try:
        arr = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{what}' must be numeric: {exc}") from exc
    if len(arr) != n:
        raise ValueError(f"'{what}' has length {len(arr)} but the cross-basis has {n} rows")
    return arr


def as_covariates(other_vars, n: int) -> Optional[np.ndarray]:
    """Additional covariates as a float matrix (n, k), or None."""
    if other_vars is None:
        return None
    if isinstance(other_vars, (pd.DataFrame, pd.Series)):
        other_vars = other_vars.to_numpy()
    arr = np.asarray(other_vars)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"'other_vars' must be a vector or a matrix, got an array of shape {arr.shape}")
    try:
        arr = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'other_vars' must be numeric: {exc}") from exc
    if arr.shape[0] != n:
        raise ValueError(f"'other_vars' has {arr.shape[0]} rows but the cross-basis has {n} rows")
    return arr


def check_covariate_names(names, k: int, taken) -> List[str]:
    """Names of the ``k`` additional covariates: ``var1 ... vark`` by default. User names become data-frame columns
    and formula terms, so they must be unique strings that do not collide with the response or the cross-basis
    terms (R would silently overwrite or capture the column otherwise)."""
    if names is None:
        return [f'var{i + 1}' for i in range(k)]
    if isinstance(names, str):
        names = [names]
    names = list(names)
    if len(names) != k:
        raise ValueError(f"'formula_vars' has {len(names)} names for {k} additional covariate column(s)")
    seen = set(taken)
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"covariate names must be non-empty strings, got {name!r}")
        if re.search(r'[`\\\x00-\x1f]', name):
            raise ValueError(f"covariate name {name!r} contains a backtick, a backslash or a control character")
        if name.startswith(_ARG_PREFIX):
            raise ValueError(f"covariate name {name!r} uses the reserved prefix '{_ARG_PREFIX}'")
        if name in seen:
            raise ValueError(f"covariate name {name!r} is duplicated or collides with the response or a "
                             f"cross-basis term")
        seen.add(name)
    return names


def resolve_family(family):
    """The R family for glm(): a name from the documented list (case-insensitive; ``'gamma'`` is R's ``Gamma``) or an
    R family object / family function obtained through rpy2. Other strings are rejected instead of being pasted into
    R code."""
    if isinstance(family, str):
        key = family.strip().lower()
        if key not in _FAMILIES:
            raise ValueError(f"unknown family {family!r}: use one of {sorted(set(_FAMILIES) - {'inverse_gaussian'})} "
                             f"or pass an R family object")
        return r_function('stats::' + _FAMILIES[key])
    if isinstance(family, ro.rinterface.Sexp):
        return family
    raise TypeError(f"'family' must be a family name or an R family object, got {type(family).__name__}")


def prepare_glm_arguments(kwargs: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Validate the keyword arguments that are passed on to R's glm(): ``weights``, ``offset``, ``subset``
    (length-n vectors; ``subset`` is a boolean mask or integer positions, zero-based) and ``control`` (a dict of
    ``epsilon``, ``maxit``, ``trace`` as in ``glm.control()``). Returns {R argument name: R object}. Any other
    keyword raises TypeError, as R rejects an unused argument."""
    unknown = sorted(set(kwargs) - set(GLM_ARGUMENTS))
    if unknown:
        raise TypeError(f"unexpected keyword argument(s) {unknown}: the arguments passed on to R's glm() are "
                        f"{list(GLM_ARGUMENTS)}")
    prepared = {}
    if kwargs.get('weights') is not None:
        prepared['weights'] = ro.FloatVector(as_vector(kwargs['weights'], n, 'weights'))
    if kwargs.get('offset') is not None:
        prepared['offset'] = ro.FloatVector(as_vector(kwargs['offset'], n, 'offset'))
    if kwargs.get('subset') is not None:
        subset = kwargs['subset']
        if isinstance(subset, (pd.Series, pd.DataFrame)):
            subset = subset.to_numpy()
        arr = np.asarray(subset)
        if arr.ndim != 1:
            raise ValueError("'subset' must be a one-dimensional boolean mask or vector of positions")
        if arr.dtype == bool:
            if len(arr) != n:
                raise ValueError(f"boolean 'subset' has length {len(arr)} but the cross-basis has {n} rows")
            prepared['subset'] = ro.BoolVector(arr.tolist())
        elif np.issubdtype(arr.dtype, np.integer):
            if arr.size and (arr.min() < 0 or arr.max() >= n):
                raise ValueError(f"'subset' positions must lie in [0, {n - 1}]")
            prepared['subset'] = ro.IntVector((arr + 1).tolist())          # R counts from one
        else:
            raise TypeError("'subset' must be a boolean mask or an array of integer positions")
    control = kwargs.get('control')
    if control is not None:
        if isinstance(control, ro.rinterface.Sexp):
            prepared['control'] = control
        elif isinstance(control, dict):
            bad = sorted(set(control) - set(_CONTROL_ARGUMENTS))
            if bad:
                raise TypeError(f"unexpected glm.control() argument(s) {bad}: use {list(_CONTROL_ARGUMENTS)}")
            prepared['control'] = r_function('stats::glm.control')(**control)
        else:
            raise TypeError("'control' must be a dict of glm.control() arguments or an R list")
    return prepared


def reduce_cross_basis(crossbasis: CrossBasis, coef, vcov, cen: Optional[float], type: str,
                       model_link: Optional[str] = None, **kwargs):
    """crossreduce() of the cross-basis block of a fitted interface. Stops like R's crossreduce() when the block has
    missing values (aliased, i.e. not estimable, coefficients). ``model_link`` (link of the fitted model) is
    forwarded when ``crossreduce()`` accepts it, so that log/logit models also get relative risks."""
    coef = np.asarray(coef, dtype=float)
    vcov = np.asarray(vcov, dtype=float)
    if np.isnan(coef).any() or np.isnan(vcov).any():
        raise ValueError(
            "coef/vcov do not consistent with basis matrix. See help(crossreduce): the cross-basis block of the "
            f"model has {int(np.isnan(coef).sum())} missing coefficient(s) (aliased, i.e. the design is rank "
            "deficient)")
    from crossreduce import crossreduce
    params = inspect.signature(crossreduce).parameters
    type_keyword = 'type' if 'type' in params else 'reduction_type'
    if model_link is not None and 'model_link' in params:
        kwargs['model_link'] = model_link
    return crossreduce(crossbasis, coef=coef, vcov=vcov, cen=cen, **{type_keyword: type}, **kwargs)


# --------------------------------------------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------------------------------------------
class Rpy2GLMInterface:
    """
    Interface to run R's GLM directly using rpy2.

    This class provides methods to fit GLMs using R's exact implementation,
    ensuring perfect coefficient and variance-covariance matrix matches.

    The fitted R ``glm`` object (``r_model``) and everything it needs live in a private R environment owned by this
    instance (``_env``).

    Parameters
    ----------
    crossbasis : CrossBasis
        The cross-basis matrix object

    Attributes
    ----------
    r_model : R object
        The fitted R ``glm`` object (None before a fit)
    cb_coef, cb_vcov : np.ndarray
        Coefficients of the cross-basis and their variance-covariance matrix (NaN for aliased coefficients)
    family, link : str
        Family and link of the fitted model
    """

    def __init__(self, crossbasis: CrossBasis):
        if not HAS_RPY2:
            raise ImportError("rpy2 is required for R GLM integration. Install with: pip install rpy2")

        self.crossbasis = crossbasis
        self.r_model = None
        self.fitted_values = None
        self.cb_coef = None
        self.cb_vcov = None
        self.family = None
        self.link = None
        self.cb_names = None

        # Initialize R environment
        self._setup_r_environment()

    def _setup_r_environment(self):
        """Set up the R interface and the private environment of this instance"""

        self.r = ro.r

        # Add user library to R library path
        self.r('if(!("~/R/library" %in% .libPaths())) .libPaths(c("~/R/library", .libPaths()))')

        # Private environment (child of baseenv): holds the data, family and fitted model of this instance
        self._env = new_r_environment()

    # ----------------------------------------------------------------------------------------------------------
    def _n_rows(self) -> int:
        return int(np.shape(self.crossbasis.basis)[0])

    def _cross_basis_terms(self, n_columns: int) -> List[str]:
        n_lag_basis = self.crossbasis.df[1]
        return [f'cb.v{(i // n_lag_basis) + 1}.l{(i % n_lag_basis) + 1}' for i in range(n_columns)]

    def _fit_r_model(self, y: np.ndarray, other: Optional[np.ndarray], other_names: Optional[List[str]],
                     family, glm_arguments: Dict[str, Any]):
        """Fit ``glm(death ~ cb + other)`` in the private environment and extract the cross-basis block.

        The rows are NOT filtered here: missing values (the lag-induced NaN rows of the cross-basis, NaN in the
        response or in the covariates) are handled by R through ``na.action = na.exclude``, so that fitted values,
        residuals and predictions are padded to the input length like R's."""
        cb_matrix = np.array(self.crossbasis.basis, dtype=float)
        n = cb_matrix.shape[0]
        cb_terms = self._cross_basis_terms(cb_matrix.shape[1])

        data = {RESPONSE: y}
        for j, term in enumerate(cb_terms):
            data[term] = cb_matrix[:, j]
        terms = list(cb_terms)
        if other is not None and other.shape[1] > 0:
            for j, name in enumerate(other_names):
                data[name] = other[:, j]
            terms += list(other_names)

        # a fresh environment per fit: a refit of this instance must not change what an earlier model refers to
        env = self._env = new_r_environment()
        env['model_data'] = r_data_frame(data)
        env['family'] = resolve_family(family)
        extras = ''
        for key, value in glm_arguments.items():
            env[key if key == 'control' else _ARG_PREFIX + key] = value
            extras += f', {key}=' + (key if key == 'control' else _ARG_PREFIX + key)

        formula = f"{r_quote(RESPONSE)} ~ " + ' + '.join(r_quote(t) for t in terms)
        r_eval(env, f"model_fit <- stats::glm({formula}, data = model_data, family = family, "
                    f"na.action = stats::na.exclude{extras})")
        self.r_model = env['model_fit']
        self.cb_names = cb_terms
        self.family = str(r_eval(env, 'model_fit$family$family')[0])
        self.link = str(r_eval(env, 'model_fit$family$link')[0])

        self._extract_cb_coefficients(cb_terms)
        return self.r_model

    def fit_glm(self,
                y: np.ndarray,
                family: str = 'quasipoisson',
                other_vars: Optional[np.ndarray] = None,
                formula_vars: Optional[list] = None,
                **kwargs) -> Any:
        """
        Fit GLM using R's glm() function directly.

        The model is ``glm(death ~ cb + other_vars, family = family, na.action = na.exclude)``: rows with missing
        values (the first ``lag`` rows of the cross-basis, NaN in ``y`` or in the covariates) are excluded by R.

        Parameters
        ----------
        y : array-like
            Response variable (e.g., mortality counts): list, array, column vector or Series, NaN allowed
        family : str or R family object, default='quasipoisson'
            GLM family: 'poisson', 'quasipoisson', 'gaussian', 'gamma' (R's ``Gamma``), 'binomial',
            'quasibinomial', 'inverse.gaussian' (case-insensitive), or an R family object
        other_vars : array-like, optional
            Additional covariates (e.g., seasonality, day of week): (n,) vector, (n, k) matrix or DataFrame
        formula_vars : list of str, optional
            Names of the columns of ``other_vars`` (default ``var1 ... vark``): unique strings that differ from
            the response name ('death') and from the cross-basis terms ('cb.v1.l1', ...)
        **kwargs
            Arguments passed to R's glm(): ``weights`` and ``offset`` (length-n vectors), ``subset`` (boolean mask
            of length n, or integer positions counted from zero) and ``control`` (dict with ``epsilon``,
            ``maxit``, ``trace`` for ``glm.control()``). Any other keyword raises TypeError.

        Returns
        -------
        r_model : R object
            Fitted R GLM model object
        """
        n = self._n_rows()
        glm_arguments = prepare_glm_arguments(kwargs, n)
        y = as_vector(y, n, 'y')
        other = as_covariates(other_vars, n)
        if other is None:
            if formula_vars is not None and not isinstance(formula_vars, str) and len(formula_vars) > 0:
                raise ValueError("'formula_vars' was given without 'other_vars'")
            names = None
        else:
            names = check_covariate_names(formula_vars, other.shape[1],
                                          [RESPONSE] + self._cross_basis_terms(int(np.shape(self.crossbasis.basis)[1])))
        return self._fit_r_model(y, other, names, family, glm_arguments)

    def _extract_cb_coefficients(self, cb_names: List[str]):
        """Cross-basis coefficients and variance-covariance matrix from the R model, selected by their exact names
        (NaN for coefficients that R could not estimate because the design is rank deficient)"""
        env = self._env
        all_coef = np.asarray(r_eval(env, 'stats::coef(model_fit)'), dtype=float)
        coef_names = [str(nm) for nm in r_eval(env, 'names(stats::coef(model_fit))')]
        position = {nm: i for i, nm in enumerate(coef_names)}
        missing = [nm for nm in cb_names if nm not in position]
        if missing:
            raise RuntimeError(f"cross-basis term(s) {missing} are not among the coefficients of the R model")
        idx = np.array([position[nm] for nm in cb_names], dtype=int)

        vcov_all = np.array(r_eval(env, 'stats::vcov(model_fit, complete = TRUE)'), dtype=float)
        if vcov_all.shape != (len(coef_names), len(coef_names)):
            raise RuntimeError("the variance-covariance matrix of the R model does not match its coefficients")
        self.cb_coef = all_coef[idx]
        self.cb_vcov = vcov_all[np.ix_(idx, idx)]

        aliased = [nm for nm, value in zip(cb_names, self.cb_coef) if np.isnan(value)]
        if aliased:
            warnings.warn(
                f"{len(aliased)} cross-basis coefficient(s) could not be estimated (aliased: the design is rank "
                f"deficient, e.g. more lag-basis columns than lags): {aliased}. They are NaN, and R's crossreduce() "
                f"and crosspred() stop on such a model.", UserWarning, stacklevel=2)

    def get_crossbasis_coefficients(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Coefficients and variance-covariance matrix of the cross-basis block (None before a fit)."""
        return self.cb_coef, self.cb_vcov

    def get_model_summary(self):
        """Get the R summary of the model fitted by this instance (``summary(glm)``: an R list)"""
        if self.r_model is None:
            raise ValueError("No model fitted yet")

        return ro.baseenv['summary'](self.r_model)

    def fitted(self) -> np.ndarray:
        """Fitted values of the model on the response scale, one per input row: NaN for the rows R excluded
        (``na.action = na.exclude`` pads them like R's ``fitted()``)."""
        if self.r_model is None:
            raise ValueError("No model fitted yet")
        return np.asarray(r_function('stats::fitted')(self.r_model), dtype=float)

    def crossreduce(self, cen: Optional[float] = None, type: str = "overall", **kwargs):
        """
        Reduce the cross-basis of the fitted model (PyDLNM's port of R's ``crossreduce()``, applied to the
        cross-basis coefficients and variance-covariance matrix of the model).

        Parameters
        ----------
        cen : float, optional
            Centering value for reduction
        type : str, default="overall"
            Type of reduction ("overall", "var", "lag"); an unknown type raises ValueError
        **kwargs
            Further arguments of ``crossreduce.crossreduce()`` (e.g. ``value`` for type "var" and "lag")

        Returns
        -------
        CrossReduce
            Reduced coefficients (``coef``) and variance-covariance matrix (``vcov``)

        Raises
        ------
        ValueError
            If no model has been fitted, the type is unknown, or the model has aliased (NaN) cross-basis
            coefficients (R: "coef/vcov do not consistent with basis matrix")
        """
        if self.r_model is None:
            raise ValueError("No model fitted yet")

        kwargs.setdefault('model_link', self.link)
        return reduce_cross_basis(self.crossbasis, self.cb_coef, self.cb_vcov, cen, type, **kwargs)
