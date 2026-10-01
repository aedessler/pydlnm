"""
Model integration utilities for PyDLNM

This module provides utilities for extracting coefficients, variance-covariance matrices,
and link functions from various statistical modeling frameworks.
"""

import re
import numpy as np
from typing import Any, Optional, Union, Dict, List, Sequence, Tuple
import warnings


_LINK_CLASS_NAMES = {'log': 'log', 'logit': 'logit', 'identity': 'identity', 'probit': 'probit',
                     'cloglog': 'cloglog', 'inversepower': 'inverse', 'inversesquared': 'inverse_squared',
                     'cauchy': 'cauchy', 'sqrt': 'sqrt'}


def _link_name(link: Any) -> Optional[str]:
    """Name of a link given as a string, an object with .name or a statsmodels link class (Log, Logit, ...)."""
    if isinstance(link, str):
        return link.lower()
    name = getattr(link, 'name', None)
    if isinstance(name, str):
        return name.lower()
    class_name = type(link).__name__.lower()
    return _LINK_CLASS_NAMES.get(class_name, class_name if link is not None else None)


# trailing punctuation is allowed (patsy names such as Q("cb_gv1.l1")), like R's unanchored grep
_NAME_PATTERNS = {'cb': re.compile(r'v[0-9]{1,2}\.l[0-9]{1,2}[^0-9A-Za-z]*$'),
                  'one': re.compile(r'b[0-9]{1,2}[^0-9A-Za-z]*$')}

# the fitted PyDLNM interfaces (duck-typed by class name, like R's getcoef/getlink dispatch on class)
_INTERFACE_CLASSES = ('DLNMGLMInterface', 'Rpy2GLMInterface', 'ImprovedGLMInterface')

# link implied by the class of a statsmodels model (the ``.model`` of a results object; R: getlink by class)
_MODEL_CLASS_LINKS = {
    'Poisson': 'log', 'GeneralizedPoisson': 'log', 'NegativeBinomial': 'log', 'NegativeBinomialP': 'log',
    'Logit': 'logit', 'Probit': 'probit', 'MNLogit': 'logit',
    'PHReg': 'log',                                   # survival::coxph
    'ConditionalLogit': 'logit',                      # survival::clogit (time-stratified case-crossover)
    'ConditionalPoisson': 'log',                      # gnm(family = poisson, eliminate = stratum)
    'OLS': 'identity', 'WLS': 'identity', 'GLS': 'identity', 'GLSAR': 'identity',      # class "lm"
}

# relative tolerance (of the largest absolute value of the basis column) for "this model column is that basis column":
# identical arrays agree to 0, a basis rebuilt in R or stored in single precision to <= 1e-7
_COLUMN_RTOL = 1e-7


def coefficient_names(model: Any) -> Optional[List[str]]:
    """Names of a fitted model's coefficients (pandas index of params, or the design column names), else None."""
    params = getattr(model, 'params', None)
    index = getattr(params, 'index', None)
    if index is not None:
        return [str(n) for n in index]
    names = getattr(getattr(model, 'model', None), 'exog_names', None)
    return [str(n) for n in names] if names is not None else None


def model_design(model: Any, n_coef: Optional[int] = None) -> Optional[np.ndarray]:
    """Design matrix (observations x coefficients) of a fitted model that exposes it (statsmodels results:
    ``model.model.exog``), else None. A matrix whose number of columns is not ``n_coef`` (e.g. variance components
    among the parameters) is not usable to locate coefficients and gives None."""
    inner = getattr(model, 'model', None)
    exog = getattr(inner, 'exog', None) if inner is not None else getattr(model, 'exog', None)
    if exog is None:
        return None
    try:
        exog = np.asarray(exog, dtype=float)
    except (TypeError, ValueError):
        return None
    if exog.ndim != 2 or (n_coef is not None and exog.shape[1] != n_coef):
        return None
    return exog


def basis_prefix(basis: Any = None, name: Optional[str] = None) -> Optional[str]:
    """Name under which the terms of a basis appear in a model: the ``name`` argument, else the ``name`` attribute
    of the basis, else None. R uses the name of the basis object (``deparse(substitute(basis))``), which Python
    cannot see."""
    if name is None:
        name = getattr(basis, 'name', None)
    if name is None:
        return None
    if not isinstance(name, str) or not name:
        raise ValueError("'name' must be a non-empty string: the prefix of the basis terms in the model")
    return name


def _named_terms(name: str, kind: str, basis_ncol: int) -> 're.Pattern':
    """R's regular expression for the coefficients of the basis called ``name``: the name alone for a one-column
    basis, else the name, any printable characters, and ``v<i>.l<j>`` (``b<i>`` for a one-basis)."""
    if basis_ncol == 1:
        return re.compile(re.escape(name))
    suffix = r'b[0-9]{1,2}' if kind == 'one' else r'v[0-9]{1,2}\.l[0-9]{1,2}'
    return re.compile(re.escape(name) + r'[^\x00-\x1f\x7f]*' + suffix)


def _columns_on_rows(basis_matrix: np.ndarray, exog: np.ndarray) -> Optional[np.ndarray]:
    """Columns of ``exog`` equal to the columns of ``basis_matrix`` when both have the same rows (entries where the
    basis is missing are ignored). One column per basis column, the first unused match; None if a column has none."""
    used, found = set(), []
    for j in range(basis_matrix.shape[1]):
        b = basis_matrix[:, j]
        rows = np.flatnonzero(np.isfinite(b))
        if rows.size == 0:
            return None
        tol = _COLUMN_RTOL * float(np.abs(b[rows]).max())
        sample = rows[np.unique(np.linspace(0, rows.size - 1, min(rows.size, 64)).astype(int))]   # cheap pre-filter
        close = np.all(np.abs(exog[sample] - b[sample, None]) <= tol, axis=0)
        match = None
        for i in np.flatnonzero(close):
            if i not in used and np.all(np.abs(exog[rows, i] - b[rows]) <= tol):
                match = int(i)
                break
        if match is None:
            return None
        used.add(match)
        found.append(match)
    return np.asarray(found, dtype=int)


def _columns_by_values(basis_matrix: np.ndarray, exog: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """Columns of ``exog`` whose values all occur in a basis column (rows selected, repeated or reordered, e.g. the
    case and control days of a conditional logit). Weaker than a row-by-row comparison, so a basis column must have
    exactly one such model column. Returns (indices, 'found'), (None, 'absent') when some basis column has no model
    column or (None, 'ambiguous') when it has several or two basis columns share one."""
    unique_values = [np.unique(exog[:, i]) for i in range(exog.shape[1])]
    found, ambiguous = [], False
    for j in range(basis_matrix.shape[1]):
        b = basis_matrix[:, j]
        b = np.sort(b[np.isfinite(b)])
        if b.size == 0:
            return None, 'absent'
        tol = _COLUMN_RTOL * float(np.abs(b).max())
        n_distinct = min(int(np.count_nonzero(np.diff(b))) + 1, 10)
        matches = []
        for i, ux in enumerate(unique_values):
            # a column with (almost) constant values, e.g. the intercept, is not a column of a varying basis
            if ux.size < n_distinct or ux[0] < b[0] - tol or ux[-1] > b[-1] + tol:
                continue
            pos = np.clip(np.searchsorted(b, ux), 1, max(b.size - 1, 1)) if b.size > 1 else np.zeros(ux.size, int)
            nearest = np.minimum(np.abs(b[pos] - ux), np.abs(b[np.maximum(pos - 1, 0)] - ux))
            if np.all(nearest <= tol):
                matches.append(i)
        if not matches:
            return None, 'absent'
        if len(matches) > 1:
            ambiguous = True
        found.append(matches[0])
    if ambiguous or len(set(found)) != len(found):
        return None, 'ambiguous'
    return np.asarray(found, dtype=int), 'found'


def find_basis_columns(basis_matrix: Any, exog: Any) -> Tuple[Optional[np.ndarray], str]:
    """Columns of a model's design matrix that are the columns of a basis matrix, in the order of the basis
    columns. Content replaces the name R greps for: the model does not have to carry informative names, and a
    basis that is not in the model is not found.

    The rows of the model are those of the basis (``exog`` has as many rows as the basis), or the rows of the basis
    without missing values (R's ``na.omit`` of the lag-induced NaN rows), or another selection of them (observations
    dropped for other reasons, repeated case-control rows). Returns ``(indices, status)`` with status ``'found'``,
    ``'absent'`` (the basis is not among the columns) or ``'ambiguous'`` (several columns fit).
    """
    B = np.asarray(basis_matrix, dtype=float)
    X = np.asarray(exog, dtype=float)
    if B.ndim != 2 or X.ndim != 2 or B.shape[1] > X.shape[1]:
        return None, 'absent'
    if not np.isfinite(X).all():
        return None, 'ambiguous'          # a design with missing values cannot be compared: leave it to the names
    complete = np.flatnonzero(np.isfinite(B).all(axis=1))
    alignments = []
    if X.shape[0] == B.shape[0]:
        alignments.append(np.arange(B.shape[0]))
    if X.shape[0] == complete.size and complete.size != B.shape[0]:
        alignments.append(complete)
    for rows in alignments:
        idx = _columns_on_rows(B[rows], X)
        if idx is not None:
            return idx, 'found'
    return _columns_by_values(B, X)


def locate_block(names: Optional[Sequence[str]], n_coef: int, basis_ncol: int, kind: str = 'cb',
                 basis_name: str = 'basis', basis: Any = None, name: Optional[str] = None,
                 exog: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Indices of the terms of a basis among the ``n_coef`` coefficients of a fitted model.

    R selects them by the NAME of the basis object (``grep`` of ``<name>[[:print:]]*v<i>.l<j>``, ``b<i>`` for a
    one-basis, the name alone for a one-column basis). Python cannot see that name, so the block is identified, in
    this order, by

    1. ``name`` (argument) or ``basis.name`` (attribute): R's regular expression on the coefficient names (a name
       that matches no coefficient at all, e.g. in a model without informative names, falls through to 2.);
    2. the columns of the design matrix ``exog`` of the model (when it exposes one) that equal the columns of the
       basis: no names needed; a basis that is not in the model raises a ValueError;
    3. the names alone: the coefficients called ``v<i>.l<j>`` (``b<i>``), when they are as many as the columns of
       the basis; or, without informative names, the whole coefficient vector when it has that size.

    Anything else raises a ValueError (never a guess).
    """
    all_idx = np.arange(n_coef)
    names = list(names) if names is not None and len(names) == n_coef else None
    advice = "pass name= (prefix of its terms in the model) or coef= and vcov= for its block"
    matrix = getattr(basis, 'basis', None)
    by_content = exog is not None and matrix is not None
    prefix = basis_prefix(basis, name)
    if prefix is not None:
        pattern = _named_terms(prefix, kind, basis_ncol)
        idx = np.array([i for i, nm in enumerate(names or ()) if pattern.search(nm)], dtype=int)
        if len(idx) == basis_ncol:
            return idx
        if len(idx) or not by_content:
            if not len(idx) and names is None and n_coef == basis_ncol:
                return all_idx                                      # nothing to select from
            raise ValueError(f"{len(idx)} model coefficients match name={prefix!r}, but the basis has {basis_ncol} "
                             f"columns: check name=, or pass coef= and vcov= for its block")
    if by_content:
        idx, status = find_basis_columns(matrix, exog)
        if status == 'found':
            return idx
        if status == 'absent':
            raise ValueError(f"{basis_name} is not in the model: none of its {basis_ncol} columns was found among "
                             f"the design columns; {advice}")
    if names is not None:
        pattern = _NAME_PATTERNS.get(kind, _NAME_PATTERNS['cb'])
        idx = np.array([i for i, nm in enumerate(names) if pattern.search(nm)], dtype=int)
        if len(idx) == basis_ncol:
            return idx
        if len(idx) > basis_ncol:
            raise ValueError(f"{len(idx)} coefficients look like cross-basis terms but {basis_name} has {basis_ncol} "
                             f"columns (several bases in the model?): {advice}")
    if n_coef == basis_ncol:
        return all_idx
    raise ValueError(f"cannot identify the {basis_ncol} coefficients of {basis_name} among the {n_coef} model "
                     f"coefficients: {advice}")


def aliased_columns(design: Any, tol: float = 1e-11) -> np.ndarray:
    """Columns of a design matrix that R's ``lm()`` / ``glm()`` report with an NA coefficient (aliased columns).

    R's LINPACK QR (``dqrdc2``, ``tol = 1e-11``) takes the columns in order and declares a column aliased when, after the
    part explained by the columns before it is removed, its norm is below ``tol`` times its own norm. The same rule is
    applied here with Gram-Schmidt (done twice for stability). statsmodels solves a rank-deficient design with a
    pseudo-inverse and returns finite numbers for such columns, so the NA has to be recovered from the design.
    """
    X = np.asarray(design, dtype=float)
    basis = np.empty((X.shape[0], 0))
    aliased = []
    for j in range(X.shape[1]):
        v = X[:, j].copy()
        norm0 = np.linalg.norm(v)
        for _ in range(2):
            if basis.shape[1]:
                v -= basis @ (basis.T @ v)
        norm = np.linalg.norm(v)
        if norm0 == 0.0 or norm < tol * norm0:
            aliased.append(j)
        else:
            basis = np.column_stack([basis, v / norm])
    return np.asarray(aliased, dtype=int)


def basis_block(model: Any, n_coef: int, basis_ncol: int, kind: str = 'cb',
                basis_name: str = 'basis', basis: Any = None, name: Optional[str] = None) -> np.ndarray:
    """
    Indices of the cross-basis (or one-basis) block in the coefficient vector of a fitted model (see
    ``locate_block`` for how the block is identified): by ``name`` as R does, else by the columns of the model's
    design matrix, else by the coefficient names ``v<i>.l<j>`` / ``b<i>``, else as the whole coefficient vector.
    """
    return locate_block(coefficient_names(model), n_coef, basis_ncol, kind, basis_name, basis, name,
                        model_design(model, n_coef) if basis is not None else None)


def getcoef(model: Any, model_class: Optional[str] = None) -> np.ndarray:
    """
    Extract coefficients from various model types.
    
    Parameters
    ----------
    model : fitted model object
        Fitted statistical model
    model_class : str, optional
        Model class name for explicit handling
        
    Returns
    -------
    np.ndarray
        Model coefficients
        
    Raises
    ------
    AttributeError
        If coefficients cannot be extracted from the model
    """
    # Determine model class if not provided
    if model_class is None:
        model_class = type(model).__name__
    
    # Handle PyDLNM's rpy2-based GLM interfaces
    if model_class == 'DLNMGLMInterface':
        coef, _ = model.get_crossbasis_coefficients()
        if coef is not None:
            return np.asarray(coef)
        else:
            raise AttributeError("No coefficients available from DLNMGLMInterface")
    
    if model_class == 'Rpy2GLMInterface':
        if hasattr(model, 'cb_coef') and model.cb_coef is not None:
            return np.asarray(model.cb_coef)
        else:
            raise AttributeError("No cross-basis coefficients available from Rpy2GLMInterface")

    if model_class == 'ImprovedGLMInterface':
        if hasattr(model, 'cb_coef') and model.cb_coef is not None:
            return np.asarray(model.cb_coef)
        else:
            raise AttributeError("No cross-basis coefficients available from ImprovedGLMInterface")

    # Try common coefficient attributes
    coef_attrs = ['params', 'coef_', 'coefficients', 'coef', 'beta']
    
    for attr in coef_attrs:
        if hasattr(model, attr):
            coef = getattr(model, attr)
            if coef is not None:
                return np.asarray(coef)
    
    # Specific handling for different model types
    if hasattr(model, 'get_params'):
        # Some sklearn-style models
        try:
            coef = model.get_params()
            if isinstance(coef, dict) and 'coef_' in coef:
                return np.asarray(coef['coef_'])
        except:
            pass
    
    # If all else fails, raise an informative error
    raise AttributeError(
        f"Cannot extract coefficients from model of type {model_class}. "
        f"Supported attributes are: {coef_attrs}"
    )


def getvcov(model: Any, model_class: Optional[str] = None) -> np.ndarray:
    """
    Extract variance-covariance matrix from various model types.
    
    Parameters
    ----------
    model : fitted model object
        Fitted statistical model
    model_class : str, optional
        Model class name for explicit handling
        
    Returns
    -------
    np.ndarray
        Variance-covariance matrix
        
    Raises
    ------
    AttributeError
        If variance-covariance matrix cannot be extracted from the model
    """
    # Determine model class if not provided
    if model_class is None:
        model_class = type(model).__name__
    
    # Handle PyDLNM's rpy2-based GLM interfaces
    if model_class == 'DLNMGLMInterface':
        _, vcov = model.get_crossbasis_coefficients()
        if vcov is not None:
            return np.asarray(vcov)
        else:
            raise AttributeError("No variance-covariance matrix available from DLNMGLMInterface")
    
    if model_class == 'Rpy2GLMInterface':
        if hasattr(model, 'cb_vcov') and model.cb_vcov is not None:
            return np.asarray(model.cb_vcov)
        else:
            raise AttributeError("No cross-basis vcov matrix available from Rpy2GLMInterface")

    if model_class == 'ImprovedGLMInterface':
        if hasattr(model, 'cb_vcov') and model.cb_vcov is not None:
            return np.asarray(model.cb_vcov)
        else:
            raise AttributeError("No cross-basis vcov matrix available from ImprovedGLMInterface")

    # Try common vcov attributes and methods
    vcov_attrs = ['cov_params', 'vcov', 'cov_', 'covariance_matrix']
    vcov_methods = ['cov_params', 'vcov']
    
    # Try attributes first (but skip if they are callable/methods)
    for attr in vcov_attrs:
        if hasattr(model, attr):
            vcov = getattr(model, attr)
            if vcov is not None and not callable(vcov):
                return np.asarray(vcov)
    
    # Try methods
    for method in vcov_methods:
        if hasattr(model, method):
            try:
                vcov = getattr(model, method)()
                if vcov is not None:
                    return np.asarray(vcov)
            except:
                continue
    
    # Special handling for specific model types
    if hasattr(model, 'summary'):
        try:
            summary = model.summary()
            if hasattr(summary, 'cov_params'):
                return np.asarray(summary.cov_params)
        except:
            pass
    
    # R stops when there is no vcov() method; a diagonal matrix built from standard errors would discard all
    # covariances (and give wrong standard errors for a cross-basis), so it is deliberately not fabricated.
    raise AttributeError(
        f"Cannot extract variance-covariance matrix from model of type {model_class}. "
        f"Tried attributes: {vcov_attrs} and methods: {vcov_methods}"
    )


def getlink(model: Any, 
            model_class: Optional[str] = None, 
            model_link: Optional[str] = None) -> Optional[str]:
    """
    Identify or extract link functions from fitted models.
    
    Parameters
    ----------
    model : fitted model object
        Fitted statistical model
    model_class : str, optional
        Model class name for explicit handling
    model_link : str, optional
        User-specified link function (takes precedence)
        
    Returns
    -------
    str or None
        Link function name ('identity', 'log', 'logit', etc.) or None if not determined
    """
    # Return user-specified link if provided
    if model_link is not None:
        return model_link
    
    # Determine model class if not provided
    if model_class is None:
        model_class = type(model).__name__
    
    # PyDLNM's rpy2-based GLM interfaces: the link of the R model they fitted (log for the default quasi-Poisson family)
    if model_class in ('DLNMGLMInterface', 'Rpy2GLMInterface', 'ImprovedGLMInterface'):
        link = getattr(model, 'link', None)
        if not isinstance(link, str):
            link = getattr(getattr(model, 'rpy2_interface', None), 'link', None)
        return link if isinstance(link, str) else 'log'

    # Family attribute (statsmodels GLM results and models): family.link is a link object; R: model$family$link
    family = getattr(model, 'family', None)
    if family is None and hasattr(model, 'model'):
        family = getattr(model.model, 'family', None)
    if family is not None and hasattr(family, 'link'):
        name = _link_name(family.link)
        if name is not None:
            return name
    
    # statsmodels models (results wrapper -> .model): the link is implied by the model class, as R's getlink does
    # for coxph (log), clogit (logit), conditional Poisson (glm/gnm: log) and lm (identity)
    inner_class = type(getattr(model, 'model', None)).__name__
    if inner_class in _MODEL_CLASS_LINKS:
        return _MODEL_CLASS_LINKS[inner_class]
    
    # Try direct link attribute
    if hasattr(model, 'link'):
        link = model.link
        if hasattr(link, 'name'):
            return link.name
        elif isinstance(link, str):
            return link
    
    # Model type-specific defaults (a GLM without a family is left undetermined rather than guessed)
    model_defaults = {
        'Poisson': 'log',
        'Logit': 'logit',
        'LogisticRegression': 'logit',
        'PoissonRegressor': 'log',
        'LinearRegression': 'identity',
        'OLS': 'identity',
    }
    
    for pattern, default_link in model_defaults.items():
        if pattern.lower() in model_class.lower():
            return default_link
    
    # If we can't determine the link, return None
    return None


def validate_model_compatibility(model: Any, 
                                basis_ncol: int,
                                basis_name: str = "basis",
                                kind: str = "cb",
                                basis: Any = None,
                                name: Optional[str] = None,
                                model_link: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate that a model is compatible with a basis matrix and extract key information.
    
    The coefficients of the basis are selected from the model as R does, by the name of the basis (``name``), and
    otherwise by the columns of the basis in the design matrix of the model (see ``locate_block``).
    
    Parameters
    ----------
    model : fitted model object
        Fitted statistical model
    basis_ncol : int
        Number of columns in the basis matrix
    basis_name : str, default="basis"
        Short label of the basis for error messages
    kind : {"cb", "one"}, default="cb"
        Basis type, which decides the coefficient name pattern (``v1.l1`` or ``b1``)
    basis : OneBasis or CrossBasis, optional
        The basis object: its columns are looked for in the design matrix of the model, and its ``name`` attribute
        (if any) is the default of ``name``
    name : str, optional
        Name of the basis in the model: prefix of the coefficient names of its terms (R: the name of the basis
        object); needed to select a basis among several ones of a model without design matrix
    model_link : str, optional
        Link function of the user (takes precedence over the link inferred from the model, like R's getlink)
        
    Returns
    -------
    dict
        Dictionary containing model information:
        - 'coef': coefficients of the basis
        - 'vcov': their variance-covariance matrix
        - 'link': link function
        - 'class': model class name
        
    Raises
    ------
    ValueError
        If model is not compatible with basis matrix
    """
    model_class = type(model).__name__
    link = getlink(model, model_class, model_link)
    
    # A fitted PyDLNM interface (R glm): the block of its own cross-basis, or another basis among its covariates
    select_block = getattr(model, 'select_block', None) if model_class in _INTERFACE_CLASSES else None
    if callable(select_block):
        coef, vcov = select_block(basis, name=name, kind=kind, ncol=basis_ncol, label=basis_name)
        return {'coef': np.asarray(coef, dtype=float), 'vcov': np.asarray(vcov, dtype=float), 'link': link,
                'class': model_class}
    
    try:
        coef = np.asarray(getcoef(model, model_class), dtype=float)
        vcov = np.asarray(getvcov(model, model_class), dtype=float)
    except Exception as e:
        raise ValueError(f"cannot extract the coefficients and vcov of a model of type {model_class}: {e}") from e
    
    if len(coef) < basis_ncol:
        raise ValueError(f"the model has {len(coef)} coefficients but {basis_name} has {basis_ncol} columns")
    if vcov.ndim != 2 or vcov.shape[0] != len(coef) or vcov.shape[1] != len(coef):
        raise ValueError(f"variance-covariance matrix has shape {vcov.shape} but the model has {len(coef)} coefficients")
    
    idx = basis_block(model, len(coef), basis_ncol, kind, basis_name, basis=basis, name=name)
    block_coef, block_vcov = coef[idx].copy(), vcov[np.ix_(idx, idx)].copy()
    
    # R gives an aliased (rank-deficient) coefficient of the block the value NA, and crosspred / crossreduce then stop;
    # a model that solves a rank-deficient design with a pseudo-inverse (statsmodels) must not hide it. An aliased
    # column outside the block does not matter, as in R.
    design = model_design(model, len(coef))
    if design is not None:
        aliased = np.flatnonzero(np.isin(idx, aliased_columns(design)))
        block_coef[aliased] = np.nan
        block_vcov[aliased, :] = np.nan
        block_vcov[:, aliased] = np.nan
    
    return {
        'coef': block_coef,
        'vcov': block_vcov,
        'link': link,
        'class': model_class
    }
