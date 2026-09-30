"""
Basis function implementations for PyDLNM

This module contains implementations of various basis functions used in distributed
lag non-linear models, including linear, polynomial, spline, and specialized functions.
"""

import numpy as np
from typing import Union, Optional, List, Tuple, Any, Dict
import warnings

from rbridge import r_locked, r_session
from utils import asfloat

# R-compatible splines implementation - REQUIRED
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    # Load R's splines package
    with r_session():
        splines = importr('splines')
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False


def _new_scratch():
    """A fresh private R environment (child of baseenv) for the temporaries of one spline call: nothing is written
    into the user's global environment, user-defined R objects or functions cannot mask the ones used here, and no
    state is shared between calls (or threads). Call it, and use the environment, inside ``r_session()``."""
    return robjects.baseenv['new.env'](parent=robjects.baseenv)


def _r_eval(code: str, env):
    """Evaluate R code in the scratch environment ``env``."""
    return robjects.baseenv['eval'](robjects.baseenv['parse'](text=code), env)


def _eval_r_spline(r_call: str, env):
    """Evaluate an R ``splines::ns()``/``splines::bs()`` call (arguments ``x``, ``ik``, ``bk`` in the scratch
    environment ``env``).

    Returns ``(basis, interior_knots, boundary_knots)`` where the knots are the ones R actually used, also when it
    derived them from the data through ``df=``. R's ``onebasis()`` keeps them as attributes of the basis and
    ``crossbasis()``/``mkXpred()`` rebuild the identical basis at new x from them. Under the numpy2ri converter the
    attributes of an R result are lost, so the R object is held in the scratch environment and read back.
    Must be called inside ``r_session(numpy=True)``.
    """
    _r_eval(f'res <- {r_call}', env)
    basis = np.array(_r_eval('unclass(res)', env))
    knots = np.atleast_1d(np.asarray(_r_eval('as.numeric(attr(res, "knots"))', env), dtype=float))
    boundary = np.atleast_1d(np.asarray(_r_eval('as.numeric(attr(res, "Boundary.knots"))', env), dtype=float))
    return basis, knots, boundary


def _as_vector(values) -> np.ndarray:
    """A 1-d float array from a scalar, a 0-d array, a list/tuple/Series or an array (R: a length-1 numeric is a
    vector, so a scalar knot / threshold / break is a valid argument). Masked / nullable cells become NaN."""
    return np.atleast_1d(asfloat(values)).ravel()


@r_locked(numpy=True)
def _r_spline_basis(fun: str, x, df, knots, degree, intercept, boundary_knots):
    """``splines::ns()`` (``fun='ns'``) or ``splines::bs()`` (``fun='bs'``) evaluated by R, with R's own defaults.

    ``df`` and ``knots`` are forwarded only when given (R: ``df = NULL, knots = NULL``, i.e. no interior knots when
    neither is supplied), and so is ``Boundary.knots`` (R: ``range(x)``, or ``x * c(7, 9) / 8`` for a single
    non-missing x). Returns ``(basis, interior_knots, boundary_knots)`` as used by R.

    The R evaluation runs under the process-wide R lock (see ``rbridge``) in a scratch environment of its own, so
    concurrent callers (threads) cannot see each other's ``x`` / knots / result.
    """
    x = asfloat(x)
    scratch = _new_scratch()
    try:
        scratch['x'] = x
        args = []
        if knots is not None:
            scratch['ik'] = _as_vector(knots)
            args.append('knots=ik')
        elif df is not None:
            args.append(f'df={int(df)}')
        if fun == 'bs':
            args.append(f'degree={int(degree)}')
        args.append(f'intercept={"TRUE" if intercept else "FALSE"}')
        if boundary_knots is not None:
            scratch['bk'] = _as_vector(boundary_knots)
            args.append('Boundary.knots=bk')
        return _eval_r_spline(f'splines::{fun}(x, {", ".join(args)})', scratch)
    finally:
        del scratch                              # released inside the R lock


@r_locked(numpy=True)
def _r_spline_design(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """R's ``splines::splineDesign(knots, x, degree + 1, x * 0, TRUE)`` (the design matrix of ``ps()``), evaluated
    under the R lock in a scratch environment of its own."""
    scratch = _new_scratch()
    try:
        scratch['x'] = x
        scratch['knots'] = knots
        return np.array(_r_eval(f'unclass(suppressWarnings(splines::splineDesign(knots, x, {degree + 1}, x * 0, '
                                f'TRUE)))', scratch))
    finally:
        del scratch


@r_locked(numpy=True)
def _r_cr_design(x: np.ndarray, knots: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Design matrix and penalty of mgcv's cubic regression spline (``cr()``): R's
    ``smooth.construct.cr.smooth.spec(s(x, bs = "cr", k = k), data = list(x = x), knots = list(x = knots))``,
    evaluated under the R lock in a scratch environment of its own."""
    scratch = _new_scratch()
    try:
        scratch['x'] = x
        scratch['knots'] = knots
        _r_eval(f'oo <- mgcv::smooth.construct.cr.smooth.spec(mgcv::s(x, bs = "cr", k = {k}), '
                f'data = list(x = x), knots = list(x = knots))', scratch)
        return np.array(_r_eval('unclass(oo$X)', scratch)), np.array(_r_eval('unclass(oo$S[[1]])', scratch))
    finally:
        del scratch


def _r_colon(start, end) -> np.ndarray:
    """R's ``start:end`` (step +1, or -1 if end < start; the last value never exceeds ``end`` beyond 1e-10)."""
    n = int(np.floor(abs(end - start) + 1e-10)) + 1
    return start + (1.0 if start <= end else -1.0) * np.arange(n)


def _r_format(values: np.ndarray) -> np.ndarray:
    """``as.character()`` of doubles as R matches them in ``factor()`` (15 significant digits)."""
    return np.array([('%.15g' % (v + 0.0)) if not np.isnan(v) else 'NA' for v in values], dtype=object)


@r_locked
def _require_r_package(package: str):
    """Raise ImportError unless the R package can be loaded (used for mgcv, needed by the 'cr' basis)."""
    if not HAS_RPY2:
        raise ImportError("rpy2 is required for this basis function in PyDLNM. Please install rpy2 with: pip install rpy2")
    ok = robjects.r(f'isTRUE(suppressWarnings(requireNamespace("{package}", quietly = TRUE)))')[0]
    if not ok:
        raise ImportError(f"the R package '{package}' is required for this basis function but is not installed")


class BaseBasisFunction:
    """
    Base class for all basis functions.
    
    This abstract base class defines the interface that all basis functions
    must implement.
    """
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.attributes = {}
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate basis matrix from input vector.
        
        Parameters
        ----------
        x : array-like
            Input vector
        **kwargs
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Basis matrix
        """
        raise NotImplementedError("Subclasses must implement __call__")
    
    def get_attributes(self) -> Dict[str, Any]:
        """Return basis function attributes."""
        return self.attributes.copy()


class LinearBasis(BaseBasisFunction):
    """
    Linear basis function.
    
    Creates a simple linear transformation of the input vector.
    
    Parameters
    ----------
    intercept : bool, default=False
        Whether to include an intercept column
    """
    
    def __init__(self, intercept: bool = False, **kwargs):
        super().__init__(intercept=intercept, **kwargs)
        self.intercept = intercept
        self.attributes['fun'] = 'lin'
        self.attributes['intercept'] = intercept
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate linear basis matrix.
        
        Parameters
        ----------
        x : array-like
            Input vector
            
        Returns
        -------
        np.ndarray
            Linear basis matrix
        """
        x = asfloat(x)
        
        if self.intercept:
            basis = np.column_stack([np.ones(len(x)), x])
        else:
            basis = x.reshape(-1, 1)
        
        return basis


class PolynomialBasis(BaseBasisFunction):
    """
    Polynomial basis function (R dlnm ``poly()``).

    The columns are ``(x / scale) ** k`` for ``k = (1 - intercept):degree``, exactly as R's
    ``outer(x/scale, (1-intercept):degree, "^")``: missing values give NaN rows (the intercept column of a missing
    x stays 1, as ``NaN ^ 0 = 1`` in R), a constant zero x gives ``0 / 0 = NaN``, and ``degree=0`` without intercept
    is R's ``1:0`` (the columns ``x / scale`` and ``1``).

    Parameters
    ----------
    degree : int, default=1
        Polynomial degree
    scale : float, optional
        Scaling factor. If None, uses ``max(abs(x))`` over the non-missing values (R: ``na.rm = TRUE``)
    intercept : bool, default=False
        Whether to include an intercept column
    """
    
    def __init__(self, degree: int = 1, scale: Optional[float] = None, 
                 intercept: bool = False, **kwargs):
        super().__init__(degree=degree, scale=scale, intercept=intercept, **kwargs)
        self.degree = degree
        self.scale = scale
        self.intercept = intercept
        self.attributes['fun'] = 'poly'
        self.attributes['degree'] = degree
        self.attributes['intercept'] = intercept
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate polynomial basis matrix.
        
        Parameters
        ----------
        x : array-like
            Input vector
            
        Returns
        -------
        np.ndarray
            Polynomial basis matrix
        """
        x = asfloat(x)
        
        # R: if(missing(scale)) scale <- max(abs(x), na.rm=TRUE)  (-Inf for an all-missing x, with a warning)
        if self.scale is None:
            observed = np.abs(x[~np.isnan(x)])
            scale = float(observed.max()) if observed.size else -np.inf
        else:
            scale = float(self.scale)
        
        self.attributes['scale'] = scale
        
        # R: outer(x/scale, (1-intercept):degree, "^")
        with np.errstate(divide='ignore', invalid='ignore'):
            x_scaled = x / scale
            powers = _r_colon(1 - int(bool(self.intercept)), float(self.degree))
            return np.power.outer(x_scaled, powers)


class SplineBasis(BaseBasisFunction):
    """
    Natural spline basis function using R's splines::ns() via rpy2.
    
    Creates natural cubic spline basis functions that exactly match
    R's implementation for guaranteed compatibility.
    
    Parameters
    ----------
    df : int, optional
        Degrees of freedom. As in R (``df = NULL``) there is no default: with neither ``df`` nor ``knots`` the basis
        has no interior knots (a single column without intercept). With ``df`` and no ``knots`` the interior knots
        are placed at quantiles of x (by R).
    knots : array-like or scalar, optional
        Interior knot positions. If given, ``df`` is ignored (as in R)
    intercept : bool, default=False
        Whether to include an intercept column
    Boundary_knots : array-like, optional
        Boundary knots (R: ``Boundary.knots``). If None, R's default: the range of x (``x * c(7, 9) / 8`` for a
        single non-missing value).
    """
    
    def __init__(self, df: Optional[int] = None, knots: Optional[np.ndarray] = None,
                 intercept: bool = False,
                 Boundary_knots: Optional[np.ndarray] = None, **kwargs):
        super().__init__(df=df, knots=knots, intercept=intercept, **kwargs)
        self.df = df
        self.knots = None if knots is None else _as_vector(knots)
        self.intercept = intercept
        self.Boundary_knots = None if Boundary_knots is None else _as_vector(Boundary_knots)
        self.attributes['fun'] = 'ns'
        self.attributes['intercept'] = intercept
        if self.knots is not None:
            self.attributes['knots'] = self.knots
        if self.Boundary_knots is not None:
            self.attributes['Boundary_knots'] = self.Boundary_knots
        self._check_rpy2()
    
    def _check_rpy2(self):
        """Check if rpy2 is available"""
        if not HAS_RPY2:
            raise ImportError(
                "rpy2 is required for spline functionality in PyDLNM. "
                "Please install rpy2 with: pip install rpy2"
            )
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate natural spline basis matrix using R's splines::ns().
        
        Parameters
        ----------
        x : array-like
            Input vector
            
        Returns
        -------
        np.ndarray
            Natural spline basis matrix
        """
        basis_matrix, knots_used, boundary_used = _r_spline_basis(
            'ns', x, self.df, self.knots, None, self.intercept, self.Boundary_knots)

        # Record the knots R used (R: attributes of the ns object), so that prediction can rebuild this basis
        self.internal_knots = knots_used
        self.attributes['df'] = basis_matrix.shape[1]
        self.attributes['knots'] = knots_used
        self.attributes['Boundary_knots'] = boundary_used
        return basis_matrix


class BSplineBasis(BaseBasisFunction):
    """
    B-spline basis function using R's splines::bs() via rpy2.

    Creates B-spline basis functions that exactly match R's implementation
    for guaranteed compatibility.

    Parameters
    ----------
    df : int, optional
        Degrees of freedom. As in R (``df = NULL``) there is no default: with neither ``df`` nor ``knots`` the basis
        has no interior knots (``degree`` columns without intercept).
    degree : int, default=3
        B-spline degree
    knots : array-like or scalar, optional
        Interior knot positions. If given, ``df`` is ignored (as in R)
    intercept : bool, default=False
        Whether to include an intercept column
    Boundary_knots : array-like, optional
        Boundary knots (min, max of training data).  Maps to R's
        Boundary.knots argument.  When supplied, predictions outside
        the training range use the same boundary knots as training,
        matching R's crossbasis / mkXpred behaviour exactly.
    """

    def __init__(self, df: Optional[int] = None, degree: int = 3,
                 knots: Optional[np.ndarray] = None,
                 intercept: bool = False,
                 Boundary_knots: Optional[np.ndarray] = None, **kwargs):
        super().__init__(df=df, degree=degree, knots=knots,
                        intercept=intercept, **kwargs)
        self.df = df
        self.degree = degree
        self.knots = None if knots is None else _as_vector(knots)
        self.intercept = intercept
        self.Boundary_knots = None if Boundary_knots is None else _as_vector(Boundary_knots)
        self.attributes['fun'] = 'bs'
        self.attributes['degree'] = degree
        self.attributes['intercept'] = intercept
        if self.Boundary_knots is not None:
            self.attributes['Boundary_knots'] = self.Boundary_knots
        self._check_rpy2()
    
    def _check_rpy2(self):
        """Check if rpy2 is available"""
        if not HAS_RPY2:
            raise ImportError(
                "rpy2 is required for spline functionality in PyDLNM. "
                "Please install rpy2 with: pip install rpy2"
            )
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate B-spline basis matrix using R's splines::bs().
        
        Parameters
        ----------
        x : array-like
            Input vector
            
        Returns
        -------
        np.ndarray
            B-spline basis matrix
        """
        return self.transform(x, **kwargs)
    
    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate B-spline basis matrix using R's splines::bs().
        
        Parameters
        ----------
        x : array-like
            Input vector
            
        Returns
        -------
        np.ndarray
            B-spline basis matrix
        """
        basis_matrix, knots_used, boundary_used = _r_spline_basis(
            'bs', x, self.df, self.knots, self.degree, self.intercept, self.Boundary_knots)

        # Record the knots R used (R: attributes of the bs object), so that prediction can rebuild this basis
        self.internal_knots = knots_used
        self.attributes['df'] = basis_matrix.shape[1]
        self.attributes['knots'] = knots_used
        self.attributes['Boundary_knots'] = boundary_used
        return basis_matrix


class StrataBasis(BaseBasisFunction):
    """
    Stratified/categorical basis function (R dlnm ``strata()``).

    Parameters
    ----------
    df : int, default=1
        Degrees of freedom. Without ``breaks``, ``df - intercept`` breaks are placed at equally spaced quantiles
        of x. ``df=1`` with ``intercept=True`` has no break: a single column of ones (R's default lag basis).
    breaks : array-like, optional
        Cut points (sorted and de-duplicated, as R does). Strata are the intervals ``[b_i, b_i+1)``.
    ref : int, default=1
        Reference stratum (1-based) dropped from the basis; 0 keeps every stratum.
    intercept : bool, default=False
        Whether to include an intercept column
    """
    
    def __init__(self, df: int = 1, breaks: Optional[np.ndarray] = None,
                 ref: int = 1, intercept: bool = False, **kwargs):
        super().__init__(df=df, breaks=breaks, ref=ref, 
                        intercept=intercept, **kwargs)
        self.df = df
        self.breaks = breaks
        self.ref = ref
        self.intercept = bool(intercept)
        self.attributes['fun'] = 'strata'
        self.attributes['df'] = df
        self.attributes['ref'] = ref
        self.attributes['intercept'] = self.intercept
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate the stratified basis matrix exactly as R's strata().

        Parameters
        ----------
        x : array-like
            Input vector (NaN gives NaN strata columns, as R's cut() gives NA)

        Returns
        -------
        np.ndarray
            Stratified basis matrix
        """
        x = np.asarray(x, dtype=float)
        nan_mask = np.isnan(x)
        x_clean = x[~nan_mask]
        
        if len(x_clean) == 0:
            raise ValueError("No valid (non-NaN) values in x")
        
        intercept = int(self.intercept)
        
        # Define breaks and df
        if self.breaks is not None:
            breaks = np.unique(np.atleast_1d(np.asarray(self.breaks, dtype=float)))
        elif self.df - intercept > 0:
            k = int(self.df) - intercept
            breaks = np.quantile(x_clean, np.arange(1, k + 1) / (k + 1))
        else:
            breaks = None
        df = (0 if breaks is None else len(breaks)) + intercept
        
        # Transformation: cut(x, c(min - 1e-4, breaks, max + 1e-4), right = FALSE)
        edges = np.sort(np.concatenate([[x_clean.min() - 0.0001],
                                        [] if breaks is None else breaks,
                                        [x_clean.max() + 0.0001]]))
        if np.any(np.diff(edges) == 0):
            raise ValueError("'breaks' are not unique")
        n_levels = len(edges) - 1
        level = np.searchsorted(edges, np.where(nan_mask, edges[0], x), side='right') - 1
        basis = np.zeros((len(x), n_levels))
        inside = ~nan_mask & (level >= 0) & (level < n_levels)
        basis[np.flatnonzero(inside), level[inside]] = 1.0
        basis[~inside] = np.nan
        
        # Define the reference
        ref = int(self.ref)
        if ref not in range(0, basis.shape[1] + 1):
            raise ValueError("wrong value in 'ref' argument. See help('strata')")
        if not self.intercept and ref == 0:
            ref = 1
        if breaks is not None:
            if ref != 0:
                basis = np.delete(basis, ref - 1, axis=1)
            if self.intercept and ref != 0:
                basis = np.column_stack([np.ones(len(x)), basis])
        
        # Attributes of the fitted basis (R: df, breaks, ref, intercept)
        self.attributes['df'] = df
        if breaks is not None:
            self.attributes['breaks'] = breaks
        else:
            self.attributes.pop('breaks', None)
        self.attributes['ref'] = ref
        self.attributes['intercept'] = self.intercept
        
        return basis


class ThresholdBasis(BaseBasisFunction):
    """
    Threshold/hockey-stick basis function (R dlnm ``thr()``).

    Creates high, low or double linear threshold transformations.

    Parameters
    ----------
    thr_value : float or array-like, optional
        Threshold value(s). If None, the median of x (R: ``median(x, na.rm = FALSE)``, i.e. NaN, and so an all-NaN
        basis, when x has missing values). Otherwise the values are sorted (missing ones dropped); only the minimum
        is used for side 'h'/'l' and the minimum and maximum for side 'd'.
    side : {'h', 'l', 'd'}, optional
        Threshold side: 'h' (higher), 'l' (lower), 'd' (double). If None (default) it is 'd' when more than one
        threshold value is supplied and 'h' otherwise.
    intercept : bool, default=False
        Whether to include an intercept column
    """
    
    def __init__(self, thr_value: Optional[Union[float, np.ndarray]] = None,
                 side: Optional[str] = None, intercept: bool = False, **kwargs):
        thr_value = kwargs.pop('thr.value', thr_value)       # R's spelling of the argument
        super().__init__(thr_value=thr_value, side=side, 
                        intercept=intercept, **kwargs)
        self.thr_value = thr_value
        self.side = side
        self.intercept = intercept
        self.attributes['fun'] = 'thr'
        self.attributes['intercept'] = intercept
        if side is not None:
            self.attributes['side'] = side
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate threshold basis matrix.
        
        Parameters
        ----------
        x : array-like
            Input vector
            
        Returns
        -------
        np.ndarray
            Threshold basis matrix
        """
        x = asfloat(x)
        
        # R: thr.value <- if(is.null(thr.value)) median(x, na.rm=FALSE) else sort(thr.value)
        if self.thr_value is None:
            with np.errstate(invalid='ignore'):
                thr = np.array([np.median(x)]) if x.size else np.array([np.nan])
        else:
            thr = _as_vector(self.thr_value)
            thr = np.sort(thr[~np.isnan(thr)])            # sort() drops missing values
        
        # R: side <- ifelse(length(thr.value) > 1, "d", "h"), then match.arg(side, c("h", "l", "d"))
        side = self.side
        if side is None:
            side = 'd' if len(thr) > 1 else 'h'
        if side not in ('h', 'l', 'd'):
            raise ValueError(f"Invalid side '{side}'. Must be 'h', 'l', or 'd'")
        
        # R: thr.value[c(1, length)] for 'd', thr.value[1] otherwise (NA if there is none)
        if len(thr) == 0:
            thr = np.array([np.nan])
        thr = thr[[0, -1]] if side == 'd' else thr[:1]
        
        self.attributes['thr.value'] = thr
        self.attributes['side'] = side
        
        # NaN propagates through np.maximum / np.minimum, as through R's pmax / pmin
        if side == 'h':
            # Higher side: max(x - threshold, 0)
            basis_cols = [np.maximum(x - thr[0], 0.0)]
        elif side == 'l':
            # Lower side: -min(x - threshold, 0)
            basis_cols = [-np.minimum(x - thr[0], 0.0)]
        else:
            # Double side: lower hinge at min(thr), higher hinge at max(thr)
            basis_cols = [-np.minimum(x - thr[0], 0.0), np.maximum(x - thr[1], 0.0)]
        
        basis = np.column_stack(basis_cols)
        
        if self.intercept:
            intercept_col = np.ones((len(x), 1))
            basis = np.column_stack([intercept_col, basis])
        
        return basis


class IntegerBasis(BaseBasisFunction):
    """
    Integer (indicator) basis function (R dlnm ``integer()``).

    One indicator column per level: ``values`` if given, the sorted distinct values of x otherwise. The first column
    is dropped unless ``intercept`` is True; with a single level the column is kept and ``intercept`` is forced to
    True. A missing x, or an x that is not one of ``values``, gives a row of NaN (R's ``NA`` from ``factor()``).
    Values are matched as R's ``factor()`` does, through their 15-significant-digit text.

    Parameters
    ----------
    values : array-like, optional
        The levels (columns, in this order). Default: ``sorted(unique(x))``
    intercept : bool, default=False
        Whether the first level is kept (a full set of indicators)
    """

    def __init__(self, values: Optional[np.ndarray] = None, intercept: bool = False, **kwargs):
        super().__init__(values=values, intercept=intercept, **kwargs)
        self.values = None if values is None else _as_vector(values)
        self.intercept = bool(intercept)
        self.attributes['fun'] = 'integer'
        self.attributes['intercept'] = self.intercept

    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate the indicator matrix exactly as R's integer().

        Parameters
        ----------
        x : array-like
            Input vector

        Returns
        -------
        np.ndarray
            Indicator matrix (NaN rows for missing or unlisted x)
        """
        x = asfloat(x)

        # R: levels <- if(!missing(values)) values else sort(unique(x))   (sort() drops the missing values)
        levels = self.values if self.values is not None else np.unique(x[~np.isnan(x)])
        labels = _r_format(levels)
        if len(set(labels)) != len(labels):
            raise ValueError("factor level is duplicated in 'values'")

        # R: factor(x, levels) and outer(xfac, levels, "==") + 0L: a row of NA when x is missing / not a level
        position = {label: j for j, label in enumerate(labels) if label != 'NA'}
        distinct, inverse = np.unique(x, return_inverse=True)
        index = np.array([position.get(label, -1) for label in _r_format(distinct)], dtype=int)[inverse.ravel()]
        basis = (index[:, None] == np.arange(len(levels))[None, :]).astype(float)
        basis[index < 0] = np.nan

        # R: if(ncol(basis) > 1L) { if(!intercept) drop the first column } else intercept <- TRUE
        intercept = self.intercept
        if basis.shape[1] > 1:
            if not intercept:
                basis = basis[:, 1:]
        else:
            intercept = True

        self.attributes['values'] = levels
        self.attributes['intercept'] = intercept
        return basis


class PSplineBasis(BaseBasisFunction):
    """
    P-spline basis function (R dlnm ``ps()``): B-splines at equally spaced knots with a difference penalty.

    Port of dlnm's ``ps()``; the B-spline design matrix is evaluated by R's ``splines::splineDesign()`` through
    rpy2. The attributes hold R's ``df``, full knot vector ``knots``, ``degree``, ``intercept``, ``fx``, the penalty
    matrix ``S`` (None if ``fx``) and ``diff``, which is what ``crosspred`` needs to rebuild the basis at new x.

    Parameters
    ----------
    df : int, default=10
        Degrees of freedom (columns)
    knots : array-like, optional
        Either the complete knot vector (then ``df`` is derived from it) or a two-element range that replaces the
        range of x when the equally spaced knots are placed
    degree : int, default=3
        B-spline degree (>= 1)
    intercept : bool, default=False
        Whether to keep the first B-spline
    fx : bool, default=False
        Fixed (unpenalised) basis: no penalty matrix
    S : array-like, optional
        Penalty matrix (default: the ``diff``-th order difference penalty)
    diff : int, default=2
        Order of the difference penalty
    """

    def __init__(self, df: int = 10, knots: Optional[np.ndarray] = None, degree: int = 3,
                 intercept: bool = False, fx: bool = False, S: Optional[np.ndarray] = None, diff: int = 2,
                 **kwargs):
        super().__init__(df=df, knots=knots, degree=degree, intercept=intercept, fx=fx, S=S, diff=diff, **kwargs)
        self.df = df
        self.knots = None if knots is None else _as_vector(knots)
        self.degree = degree
        self.intercept = bool(intercept)
        self.fx = bool(fx)
        self.S = None if S is None else np.atleast_2d(asfloat(S))
        self.diff = diff
        self.attributes['fun'] = 'ps'
        self._check_rpy2()

    def _check_rpy2(self):
        """Check if rpy2 is available"""
        if not HAS_RPY2:
            raise ImportError(
                "rpy2 is required for spline functionality in PyDLNM. "
                "Please install rpy2 with: pip install rpy2"
            )

    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x = asfloat(x)
        observed_range = (np.nanmin(x), np.nanmax(x)) if np.any(~np.isnan(x)) else (np.inf, -np.inf)
        nax = np.isnan(x)
        xx = x[~nax]
        degree = int(self.degree)                                  # R: as.integer(degree)
        if degree < 1:
            raise ValueError("'degree' must be integer >= 1")
        intercept = int(self.intercept)
        df = self.df

        # DEFINE KNOTS AND DF
        knots = self.knots
        if knots is None or len(knots) == 2:
            nik = int(df) - degree + 2 - intercept
            if nik <= 1:
                raise ValueError("basis dimension too small for b-spline degree")
            width = (observed_range[1] - observed_range[0]) * 0.001
            xl = (knots.min() if knots is not None else xx.min()) - width
            xu = (knots.max() if knots is not None else xx.max()) + width
            dx = (xu - xl) / (nik - 1)
            knots = np.linspace(xl - dx * degree, xu + dx * degree, nik + 2 * degree)
        else:
            df = len(knots) - degree - 2 + intercept
            if df - degree <= 1:
                raise ValueError("basis dimension too small for b-spline degree")
        if np.any((xx < knots[degree]) | (knots[len(knots) - degree - 1] < xx)):
            warnings.warn('all obs expected within inner df-degree+int knots')

        # TRANSFORMATION: R's splineDesign(knots, x, degree + 1, x * 0, TRUE)
        design = _r_spline_design(xx, np.asarray(knots, dtype=float), degree)
        basis = design.reshape(len(xx), -1)
        if not intercept:
            basis = basis[:, 1:]

        # RE-INSERT MISSING
        if nax.any():
            full = np.full((len(x), basis.shape[1]), np.nan)
            full[~nax] = basis
            basis = full

        # RELATED PENALTY MATRIX
        if self.diff < 1:
            raise ValueError("'diff' must be an integer >=1")
        penalty = self.S
        if self.fx:
            penalty = None
        elif penalty is None:
            difference = np.diff(np.eye(basis.shape[1] + (1 - intercept)), n=int(self.diff), axis=0)
            penalty = difference.T @ difference
            penalty = (penalty + penalty.T) / 2
            if not intercept:
                penalty = penalty[1:, 1:]
        elif penalty.shape != (basis.shape[1], basis.shape[1]):
            raise ValueError("dimensions of 'S' not compatible")

        self.attributes.update({'df': int(df), 'knots': knots, 'degree': degree, 'intercept': self.intercept,
                                'fx': self.fx, 'S': penalty, 'diff': self.diff})
        return basis


class CRSplineBasis(BaseBasisFunction):
    """
    Cubic regression spline basis function (R dlnm ``cr()``), from mgcv.

    Port of dlnm's ``cr()``: the knots are quantiles of the distinct x values, the basis and penalty come from
    mgcv's ``smooth.construct.cr.smooth.spec()`` (R package mgcv, evaluated through rpy2). The attributes hold R's
    ``df``, ``knots``, ``intercept``, ``fx`` and the penalty matrix ``S`` (None if ``fx``), which is what
    ``crosspred`` needs to rebuild the basis at new x.

    Parameters
    ----------
    df : int, default=10
        Degrees of freedom (columns), at least 3
    knots : array-like, optional
        The knots (``len(knots) - (not intercept)`` is then the df). Default: ``df + (not intercept)`` quantiles
        of the distinct values of x, from its minimum to its maximum
    intercept : bool, default=False
        Whether to keep the first column
    fx : bool, default=False
        Fixed (unpenalised) basis: no penalty matrix
    S : array-like, optional
        Penalty matrix (default: mgcv's, symmetrised)
    """

    def __init__(self, df: int = 10, knots: Optional[np.ndarray] = None, intercept: bool = False,
                 fx: bool = False, S: Optional[np.ndarray] = None, **kwargs):
        super().__init__(df=df, knots=knots, intercept=intercept, fx=fx, S=S, **kwargs)
        self.df = df
        self.knots = None if knots is None else _as_vector(knots)
        self.intercept = bool(intercept)
        self.fx = bool(fx)
        self.S = None if S is None else np.atleast_2d(asfloat(S))
        self.attributes['fun'] = 'cr'

    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        _require_r_package('mgcv')
        x = asfloat(x)
        nax = np.isnan(x)
        xx = x[~nax]
        not_intercept = int(not self.intercept)

        # DEFINE KNOTS AND DF
        if self.knots is None:
            df = int(self.df)
            if df < 3:
                raise ValueError("'df' must be >=3")
            knots = np.quantile(np.unique(xx), np.linspace(0.0, 1.0, df + not_intercept))
        else:
            knots = self.knots
            df = len(knots) - not_intercept

        # CHECK NUMBER OF UNIQUE x VALUES (ADD SOME IF NEEDED TO PREVENT AN ERROR IN mgcv)
        add = len(np.unique(xx)) < len(knots)
        if add:
            xx = np.concatenate([np.linspace(knots.min(), knots.max(), len(knots)), xx])

        # TRANSFORMATION: CALL FUNCTION FROM MGCV
        design, penalty = _r_cr_design(xx, np.asarray(knots, dtype=float), df + not_intercept)
        basis = design.reshape(len(xx), -1)
        if not self.intercept:
            basis = basis[:, 1:]

        # REMOVE ADDED VALUES AND RE-INSERT MISSING
        if add:
            basis = basis[len(knots):]
        if nax.any():
            full = np.full((len(x), basis.shape[1]), np.nan)
            full[~nax] = basis
            basis = full

        # RELATED PENALTY MATRIX
        result = self.S
        if self.fx:
            result = None
        elif result is None:
            result = (penalty + penalty.T) / 2
            if not self.intercept:
                result = result[1:, 1:]
        elif result.shape != (basis.shape[1], basis.shape[1]):
            raise ValueError("dimensions of 'S' not compatible")

        self.attributes.update({'df': int(df), 'knots': knots, 'intercept': self.intercept, 'fx': self.fx,
                                'S': result})
        return basis
