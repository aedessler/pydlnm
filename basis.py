"""
Core basis classes for PyDLNM

This module contains the main OneBasis and CrossBasis classes that form the foundation
of the distributed lag non-linear modeling framework.
"""

import inspect
import numpy as np
from typing import Union, Optional, Dict, Any, Callable, List, Tuple
import warnings

from basis_functions import (
    LinearBasis, PolynomialBasis, SplineBasis, BSplineBasis, 
    StrataBasis, ThresholdBasis, IntegerBasis, PSplineBasis, CRSplineBasis, BaseBasisFunction
)
from utils import mklag, seqlag
from model_utils import validate_model_compatibility


def _callable_formals(fun) -> Tuple[set, bool]:
    """
    Named arguments of a user-defined basis function and whether it accepts ``**kwargs``.

    Python analogue of R's ``names(formals(fun))``: R never matches ``...`` against an attribute, so the named
    (non-variadic) parameters are the ones a function "declares". ``x`` is included when present.
    """
    try:
        parameters = list(inspect.signature(fun).parameters.values())
    except (TypeError, ValueError):
        return set(), False
    names = {p.name for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return names, any(p.kind is p.VAR_KEYWORD for p in parameters)


class OneBasis:
    """
    One-dimensional basis function class.
    
    This class creates one-dimensional basis matrices for use in distributed
    lag models. It supports various basis function types and provides a
    consistent interface for basis matrix generation.
    
    Parameters
    ----------
    x : array-like
        Input vector for basis transformation
    fun : str or callable, default='ns'
        Basis function type. Can be:
        - 'lin': Linear basis
        - 'poly': Polynomial basis  
        - 'ns': Natural spline basis (no interior knots without ``df`` or ``knots``, as in R)
        - 'bs': B-spline basis (no interior knots without ``df`` or ``knots``, as in R)
        - 'strata': Stratified/categorical basis
        - 'thr': Threshold basis
        - 'integer': Indicator of each distinct value (or of ``values``)
        - 'ps': P-spline basis with difference penalty (attribute ``S``)
        - 'cr': Cubic regression spline basis from mgcv (attribute ``S``, needs the R package mgcv)
        - Custom function (a callable of ``x``; it receives ``cen`` only if it declares that argument, and the
          attributes it returns, e.g. on an ndarray subclass, are kept)
    **kwargs
        Additional arguments passed to the basis function
        
    Attributes
    ----------
    x : np.ndarray
        Original input vector
    fun : str or callable
        Basis function used
    basis : np.ndarray
        Generated basis matrix
    range : tuple
        Range of input values (min, max)
    attributes : dict
        Basis function attributes and parameters
    """
    
    # Mapping of function names to classes
    _FUNCTION_MAP = {
        'lin': LinearBasis,
        'poly': PolynomialBasis,
        'ns': SplineBasis,
        'bs': BSplineBasis,
        'strata': StrataBasis,
        'thr': ThresholdBasis,
        'integer': IntegerBasis,
        'ps': PSplineBasis,
        'cr': CRSplineBasis,
    }

    # Arguments of each built-in function that are recorded as attributes of a fitted basis:
    # {attribute name: constructor keyword}. R equivalent: names(formals(fun)) matched against
    # names(attributes(basis)) in crossbasis() and mkXpred().
    _RESOLVED_ARGS = {
        'lin':    {'intercept': 'intercept'},
        'poly':   {'degree': 'degree', 'scale': 'scale', 'intercept': 'intercept'},
        'ns':     {'knots': 'knots', 'Boundary_knots': 'Boundary_knots', 'intercept': 'intercept'},
        'bs':     {'degree': 'degree', 'knots': 'knots', 'Boundary_knots': 'Boundary_knots',
                   'intercept': 'intercept'},
        'strata': {'df': 'df', 'breaks': 'breaks', 'ref': 'ref', 'intercept': 'intercept'},
        'thr':    {'thr.value': 'thr_value', 'side': 'side', 'intercept': 'intercept'},
        'integer': {'values': 'values', 'intercept': 'intercept'},
        'ps':     {'df': 'df', 'knots': 'knots', 'degree': 'degree', 'intercept': 'intercept', 'fx': 'fx',
                   'S': 'S', 'diff': 'diff'},
        'cr':     {'df': 'df', 'knots': 'knots', 'intercept': 'intercept', 'fx': 'fx', 'S': 'S'},
    }

    # Arguments accepted by each built-in function (R: the formals of lin, poly, ns, bs, strata, thr)
    _ACCEPTED_ARGS = {
        'lin':    {'intercept'},
        'poly':   {'degree', 'scale', 'intercept'},
        'ns':     {'df', 'knots', 'intercept', 'Boundary_knots'},
        'bs':     {'df', 'degree', 'knots', 'intercept', 'Boundary_knots'},
        'strata': {'df', 'breaks', 'ref', 'intercept'},
        'thr':    {'thr_value', 'side', 'intercept'},
        'integer': {'values', 'intercept'},
        'ps':     {'df', 'knots', 'degree', 'intercept', 'fx', 'S', 'diff'},
        'cr':     {'df', 'knots', 'intercept', 'fx', 'S'},
    }

    # Attributes holding numeric vectors (returned by resolved_args() as 1-d float arrays)
    _VECTOR_ARGS = ('knots', 'Boundary_knots', 'breaks', 'thr.value', 'values')

    @classmethod
    def accepts_argument(cls, fun: Union[str, Callable], name: str) -> bool:
        """Whether ``name`` is an argument of the basis function (R: ``name %in% names(formals(fun))``)."""
        if isinstance(fun, str):
            return name in cls._ACCEPTED_ARGS.get(fun, set())
        return name in _callable_formals(fun)[0]

    @classmethod
    def _check_args(cls, fun: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """R's checkonebasis(): accept R's dotted and legacy argument names, reject unused arguments."""
        kwargs = dict(kwargs)

        def rename(old: str, new: str, message: Optional[str] = None):
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(f"specify only one of '{old}' and '{new}'")
                kwargs[new] = kwargs.pop(old)
                if message:
                    warnings.warn(message)

        rename('Boundary.knots', 'Boundary_knots')
        rename('thr.value', 'thr_value')
        if fun in ('ns', 'bs'):
            rename('bound', 'Boundary_knots', "use the default argument 'Boundary.knots' for fun 'ns'-'bs'")
        if fun == 'thr':
            rename('knots', 'thr_value', "argument 'knots' replaced by 'thr.value' in function thr")
        if fun == 'strata':
            rename('knots', 'breaks', "argument 'knots' replaced by 'breaks' in function strata")
        accepted = cls._ACCEPTED_ARGS.get(fun)
        if accepted is not None:
            unused = sorted(set(kwargs) - accepted)
            if unused:
                raise TypeError(f"OneBasis(fun='{fun}'): unused argument(s) {unused}; accepted: {sorted(accepted)}")
        return kwargs

    def __init__(self, x: Union[np.ndarray, List], fun: Union[str, Callable] = 'ns', **kwargs):
        # Store original input (R: x <- as.vector(x), i.e. a matrix is flattened column by column)
        self.x = np.asarray(x, dtype=float).flatten(order='F')
        self.fun = fun
        # R: range(x, na.rm=TRUE) (c(Inf, -Inf), with a warning, when there is no observed value)
        observed = self.x[~np.isnan(self.x)]
        self.range = (observed.min(), observed.max()) if observed.size else (np.inf, -np.inf)
        
        # Extract centering parameter
        self.cen = kwargs.pop('cen', None)
        if isinstance(fun, str):
            kwargs = self._check_args(fun, kwargs)
        
        # Generate basis matrix
        self.basis, self.attributes = self._create_basis(**kwargs)
        
        # Store centering info in attributes
        if self.cen is not None:
            self.attributes['cen'] = self.cen
        
        # Set names for basis columns
        self._set_names()
    
    def _create_basis(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create the basis matrix using the specified function.
        
        Parameters
        ----------
        **kwargs
            Arguments for the basis function
            
        Returns
        -------
        tuple
            (basis_matrix, attributes_dict)
        """
        if isinstance(self.fun, str):
            # Use built-in function
            if self.fun not in self._FUNCTION_MAP:
                raise ValueError(f"Unknown function '{self.fun}'. Available: {list(self._FUNCTION_MAP.keys())}")
            
            basis_func = self._FUNCTION_MAP[self.fun](**kwargs)
            basis = basis_func(self.x)
            attributes = basis_func.get_attributes()
            
        elif callable(self.fun):
            # Use custom function. As in R's checkonebasis(), the centering value is handed to the function only
            # if the function declares a 'cen' argument.
            call_kwargs = dict(kwargs)
            if self.cen is not None and 'cen' in _callable_formals(self.fun)[0]:
                call_kwargs['cen'] = self.cen
            raw = self.fun(self.x, **call_kwargs)
            
            # R keeps attributes(basis): read them BEFORE np.asarray, which returns a plain ndarray and so drops
            # the instance attributes of an ndarray subclass returned by the function
            function_attributes = {k: v for k, v in getattr(raw, '__dict__', {}).items() if k != 'fun'}
            basis = np.asarray(raw)
            
            # Ensure it's a matrix
            if basis.ndim == 1:
                basis = basis.reshape(-1, 1)
            
            # The arguments are recorded (R only records what the function itself stores as attributes); the
            # attributes the function returns take precedence
            attributes = {'fun': self.fun}
            attributes.update(kwargs)
            attributes.update(function_attributes)
            
        else:
            raise TypeError("fun must be a string or callable")
        
        # Ensure basis is a 2D array
        if basis.ndim == 1:
            basis = basis.reshape(-1, 1)
        
        # Store range in attributes
        attributes['range'] = self.range
        
        return basis, attributes
    
    def _set_names(self):
        """Set column names for the basis matrix."""
        n_cols = self.basis.shape[1]
        self.colnames = [f"b{i+1}" for i in range(n_cols)]

    def resolved_args(self) -> Dict[str, Any]:
        """
        Arguments that rebuild this basis at new values of x.

        Data-dependent choices made on the training x (interior knots implied by ``df``, boundary knots, strata
        breaks, the default threshold, the polynomial scale) are returned as their resolved values. This is what
        R's ``crossbasis()`` stores in ``attr(, "argvar")``/``attr(, "arglag")`` and what ``mkXpred()`` uses for a
        ``onebasis`` object, so that prediction reproduces the training basis instead of re-deriving it from the
        prediction values.
        """
        if not isinstance(self.fun, str):
            # R's mkXpred(): "fun" plus the stored attributes that are arguments of the function (range is
            # bookkeeping; cen is replayed only to a function that declares it). A function that accepts **kwargs
            # receives every recorded argument.
            formals, variadic = _callable_formals(self.fun)
            args = {'fun': self.fun}
            for name, value in self.attributes.items():
                if name in ('fun', 'range', 'x') or value is None:
                    continue
                if name in formals or (variadic and name != 'cen'):
                    args[name] = value
            return args
        args = {'fun': self.fun}
        for attr, keyword in self._RESOLVED_ARGS[self.fun].items():
            if self.attributes.get(attr) is not None:
                value = self.attributes[attr]
                if attr in self._VECTOR_ARGS:
                    value = np.atleast_1d(np.asarray(value, dtype=float))
                args[keyword] = value
        return args

    def __array__(self) -> np.ndarray:
        """Return the basis matrix when converted to array."""
        return self.basis

    def __getitem__(self, key):
        """Allow indexing of the basis matrix."""
        return self.basis[key]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the basis matrix."""
        return self.basis.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions (always 2)."""
        return 2

    def summary(self) -> str:
        """
        Return a summary of the OneBasis object.
        
        Returns
        -------
        str
            Summary string
        """
        summary_lines = [
            f"OneBasis with {self.shape[1]} basis function(s)",
            f"Function: {self.fun}",
            f"Range: ({self.range[0]:.3f}, {self.range[1]:.3f})",
            f"Dimensions: {self.shape[0]} x {self.shape[1]}"
        ]
        
        # Add key attributes
        # As R's summary.onebasis: df is ncol(object), the number of columns actually produced (the 'df' argument is
        # only a request, and is ignored when knots are given)
        summary_lines.append(f"Degrees of freedom: {self.shape[1]}")
        if 'degree' in self.attributes:
            summary_lines.append(f"Degree: {self.attributes['degree']}")
        if self.fun in ('ns', 'bs') and 'knots' in self.attributes and len(self.attributes['knots']) > 0:
            knots = self.attributes['knots']
            summary_lines.append(f"Knots: {len(knots)} interior knots")
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        return f"OneBasis(fun='{self.fun}', shape={self.shape})"
    
    def __str__(self) -> str:
        return self.summary()


class CrossBasis:
    """
    Cross-basis class for distributed lag models.
    
    This class creates cross-basis matrices by combining exposure-response and
    lag-response basis functions using tensor products. It supports both time
    series and matrix input formats.
    
    Parameters
    ----------
    x : array-like
        Input data. Can be:
        - Vector: treated as time series
        - Matrix: treated as lagged exposure matrix
    lag : int, list, or tuple, optional
        Lag specification. If x is a vector, defaults to [0, ncol(x)-1].
        If x is a matrix, must match the number of columns.
    argvar : dict, optional
        Arguments for the exposure-response basis function
    arglag : dict, optional  
        Arguments for the lag-response basis function
    group : array-like, optional
        Grouping variable for seasonal analysis
        
    Attributes
    ----------
    x : np.ndarray
        Original input data
    lag : np.ndarray
        Lag range [min_lag, max_lag]
    basis : np.ndarray
        Cross-basis matrix
    df : tuple
        Degrees of freedom for (exposure, lag) dimensions
    range : tuple
        Range of input values
    argvar : dict
        Exposure basis arguments
    arglag : dict
        Lag basis arguments
    """
    
    def __init__(self, 
                 x: Union[np.ndarray, List], 
                 lag: Optional[Union[int, List, Tuple]] = None,
                 argvar: Optional[Dict[str, Any]] = None,
                 arglag: Optional[Dict[str, Any]] = None,
                 group: Optional[np.ndarray] = None,
                 **kwargs):
        
        # Convert x to matrix
        self.x = np.asarray(x, dtype=float)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        
        # Validate and set lag
        if lag is None:
            lag = [0, self.x.shape[1] - 1]
        self.lag = mklag(lag)
        
        # Validate x dimensions with lag
        expected_cols = int(np.diff(self.lag)[0] + 1)
        if self.x.shape[1] not in [1, expected_cols]:
            raise ValueError(
                f"x has {self.x.shape[1]} columns but lag range requires "
                f"1 (time series) or {expected_cols} (lag matrix) columns"
            )
        
        # Set default arguments. Work on copies: the resolved arguments recorded below (and the defaults injected
        # here) belong to THIS x, so they must not leak into dicts the caller reuses for another series.
        self.argvar = dict(argvar) if argvar else {}
        self.arglag = dict(arglag) if arglag else {}
        
        # R checkcrossbasis(): the old argument 'type' is 'fun'; the very old usage is not allowed any more
        old_usage = {'vartype', 'vardf', 'vardegree', 'varknots', 'varbound', 'varint', 'cen', 'cenvalue', 'maxlag',
                     'lagtype', 'lagdf', 'lagdegree', 'lagknots', 'lagbound', 'lagint'}
        if kwargs:
            if set(kwargs) & old_usage:
                raise TypeError("old usage not allowed any more. See CrossBasis and OneBasis")
            raise TypeError(f"CrossBasis got unexpected keyword argument(s) {sorted(kwargs)}")
        for args in (self.argvar, self.arglag):
            if 'fun' not in args and 'type' in args:
                args['fun'] = args.pop('type')
                warnings.warn("argument 'type' replaced by 'fun'. See OneBasis")

        # As R's crossbasis(): strata(df=1, intercept=TRUE) (one unconstrained column, the sum of the exposure
        # basis over the lags) when arglag is empty or the lag period is a single lag; onebasis()'s default
        # function ("ns") when arglag has no 'fun'.
        if len(self.arglag) == 0 or np.diff(self.lag)[0] == 0:
            self.arglag = {'fun': 'strata', 'df': 1, 'intercept': True}
        if 'fun' not in self.arglag:
            self.arglag['fun'] = 'ns'

        # As R's crossbasis(): the lag basis has an intercept by default when its function has that argument (every
        # built-in one, integer included; a callable only if it declares 'intercept'), and is never centred
        if 'intercept' not in self.arglag and OneBasis.accepts_argument(self.arglag['fun'], 'intercept'):
            self.arglag['intercept'] = True
        self.arglag['cen'] = None
        
        # Groups (independent series stacked in x): lags are computed inside each group (R: checkgroup, Lag)
        self.group = None
        self._group_labels = None
        if group is not None:
            group = np.asarray(group)
            if self.x.shape[1] > 1:
                raise ValueError("'group' allowed only for time series data")
            if len(group) != self.x.shape[0]:
                raise ValueError("'group' must have one value per observation")
            counts = np.unique(group, return_counts=True)[1]
            if counts.min() <= int(np.diff(self.lag)[0]):
                raise ValueError("each group must have length > diff(lag) (see 'group')")
            self._group_labels = group
            self.group = len(counts)           # R stores length(unique(group)) as attr(, "group")
        
        # Create the cross-basis
        self._create_cross_basis()
        
        # Set attributes
        self.range = (np.nanmin(self.x), np.nanmax(self.x))
        self._set_names()
    
    def _create_cross_basis(self):
        """Create the cross-basis matrix using tensor products."""
        
        # Create exposure basis (var dimension)
        # R: basisvar <- onebasis(as.numeric(x)), column-major also for a matrix of lagged occurrences
        x_var = self.x.flatten(order='F')
        
        # Create OneBasis for exposure dimension
        self.basisvar = OneBasis(x_var, **self.argvar)

        # Redefine argvar from the attributes of the fitted basis, as R's crossbasis() does ("they might have been
        # changed by onebasis"): knots implied by df, Boundary.knots, poly scale, strata breaks and the default
        # threshold are frozen at their training values, so crosspred/crossreduce/recentering rebuild the identical
        # basis at new x (incl. out-of-range x). The centering value, if any, is kept.
        cen = self.argvar.get('cen')
        self.argvar = self.basisvar.resolved_args()
        if cen is not None:
            self.argvar['cen'] = cen

        # Create lag basis
        lag_seq = seqlag(self.lag)
        self.basislag = OneBasis(lag_seq, **self.arglag)
        self.arglag = self.basislag.resolved_args()   # same redefinition for the lag basis (integer: values, intercept)

        # Store degrees of freedom
        self.df = (self.basisvar.shape[1], self.basislag.shape[1])
        
        # Compute cross-basis using tensor product
        n_obs = self.x.shape[0]
        n_var_basis = self.basisvar.shape[1] 
        n_lag_basis = self.basislag.shape[1]
        
        # Initialize cross-basis matrix
        self.basis = np.zeros((n_obs, n_var_basis * n_lag_basis))
        
        # Create the cross-basis
        if self.x.shape[1] == 1:
            # Time series case: create lagged versions
            self._create_time_series_basis(n_var_basis, n_lag_basis)
        else:
            # Matrix case: direct tensor product
            self._create_matrix_basis(n_var_basis, n_lag_basis)
    
    def _create_time_series_basis(self, n_var_basis: int, n_lag_basis: int):
        """Create cross-basis for time series data exactly matching R's dlnm behavior.
        
        This implements the proper crossbasis calculation:
        For each observation i, sum over all lag times:
        (variable_basis_function(exposure[i-lag]) * lag_basis_function(lag))
        
        R behavior: First max_lag observations are entirely NaN because we cannot
        compute the full distributed lag effect without complete exposure history.
        """
        lag_seq = seqlag(self.lag)
        n_obs = self.x.shape[0]
        n_lags = len(lag_seq)
        if np.max(np.abs(lag_seq)) >= n_obs:
            raise ValueError("largest lag must be less than the number of observations")
        
        # As R's crossbasis(), the exposure and lag bases are the onebasis() objects built in
        # _create_cross_basis() from argvar/arglag, so every function (lin, poly, ns, bs, strata, thr, integer, a
        # callable) and every argument (df, knots, degree, intercept, ...) is honoured in both dimensions.
        r_var_basis = np.asarray(self.basisvar.basis, dtype=float)
        r_lag_basis = np.asarray(self.basislag.basis, dtype=float)
        
        # For each exposure-basis column v: the matrix of lagged occurrences (R: tsModel::Lag, computed inside each
        # group when there are groups; NaN where there is no history or future), then one column per lag-basis column.
        self.basis = np.full((n_obs, n_var_basis * n_lag_basis), np.nan)
        for v in range(n_var_basis):
            column = r_var_basis[:, v]
            if self._group_labels is None:
                lag_matrix = self._create_lagged_matrix(column, lag_seq)
            else:
                lag_matrix = np.full((n_obs, n_lags), np.nan)
                for label in np.unique(self._group_labels):
                    rows = np.flatnonzero(self._group_labels == label)
                    lag_matrix[rows] = self._create_lagged_matrix(column[rows], lag_seq)
            # NaN propagates through the product, as in R's mat %*% basislag
            self.basis[:, v * n_lag_basis:(v + 1) * n_lag_basis] = lag_matrix @ r_lag_basis
    
    def _create_matrix_basis(self, n_var_basis: int, n_lag_basis: int):
        """Cross-basis for a matrix of lagged occurrences (rows = observations, columns = lags), as R."""
        n_obs, n_lags = self.x.shape
        lag_basis = np.asarray(self.basislag.basis, dtype=float)
        self.basis = np.zeros((n_obs, n_var_basis * n_lag_basis))
        for v in range(n_var_basis):
            # the basis was evaluated on as.numeric(x): undo the column-major flatten
            mat = np.asarray(self.basisvar.basis, dtype=float)[:, v].reshape((n_obs, n_lags), order='F')
            self.basis[:, v * n_lag_basis:(v + 1) * n_lag_basis] = mat @ lag_basis
    
    def _create_lagged_matrix(self, values: np.ndarray, lag_seq: np.ndarray) -> np.ndarray:
        """
        Create a matrix of lagged values.
        
        Parameters
        ----------
        values : np.ndarray
            Time series values
        lag_seq : np.ndarray
            Sequence of lag values
            
        Returns
        -------
        np.ndarray
            Matrix with lagged values (with NaN for unavailable lag periods)
        """
        n_obs = len(values)
        n_lags = len(lag_seq)
        lagged_matrix = np.full((n_obs, n_lags), np.nan)  # Initialize with NaN
        
        for i, lag_val in enumerate(lag_seq):
            lag_int = int(lag_val)
            if lag_int == 0:
                lagged_matrix[:, i] = values
            elif lag_int > 0:
                # Positive lag: shift backwards in time
                lagged_matrix[lag_int:, i] = values[:-lag_int]
                # First lag_int observations remain NaN (no exposure history)
            else:
                # Negative lag: shift forwards in time  
                lagged_matrix[:lag_int, i] = values[-lag_int:]
                # Last abs(lag_int) observations remain NaN
        
        return lagged_matrix
    
    def _set_names(self):
        """Set column names for the cross-basis matrix."""
        n_var = self.df[0]
        n_lag = self.df[1]
        
        # Create names following R dlnm convention: v1.l1, v1.l2, v2.l1, etc.
        names = []
        for v in range(n_var):
            for l in range(n_lag):
                names.append(f"v{v+1}.l{l+1}")
        
        self.colnames = names
    
    def __array__(self) -> np.ndarray:
        """Return the cross-basis matrix when converted to array."""
        return self.basis
    
    def __getitem__(self, key):
        """Allow indexing of the cross-basis matrix."""
        return self.basis[key]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the cross-basis matrix."""
        return self.basis.shape
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions (always 2)."""
        return 2
    
    def summary(self) -> str:
        """
        Return a summary of the CrossBasis object.
        
        Returns
        -------
        str
            Summary string
        """
        summary_lines = [
            f"CrossBasis with {self.shape[1]} basis function(s)",
            f"Lag range: [{self.lag[0]}, {self.lag[1]}]",
            f"Dimensions: {self.shape[0]} x {self.shape[1]}",
            f"DF: var={self.df[0]}, lag={self.df[1]}",
            f"Range: ({self.range[0]:.3f}, {self.range[1]:.3f})"
        ]
        
        # Add basis function info
        summary_lines.append(f"Var function: {self.argvar.get('fun', 'ns')}")
        summary_lines.append(f"Lag function: {self.arglag.get('fun', 'strata')}")
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        return f"CrossBasis(lag={self.lag.tolist()}, df={self.df}, shape={self.shape})"
    
    def __str__(self) -> str:
        return self.summary()

def onebasis(x, fun: str = "ns", **kwargs) -> OneBasis:
    """Functional form of OneBasis, like R's ``onebasis(x, fun, ...)``."""
    return OneBasis(x, fun=fun, **kwargs)


def crossbasis(x, lag=None, argvar=None, arglag=None, group=None, **kwargs) -> CrossBasis:
    """Functional form of CrossBasis, like R's ``crossbasis(x, lag, argvar, arglag, group)``."""
    return CrossBasis(x, lag=lag, argvar=argvar, arglag=arglag, group=group, **kwargs)
