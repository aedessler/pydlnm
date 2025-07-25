"""
Core basis classes for PyDLNM

This module contains the main OneBasis and CrossBasis classes that form the foundation
of the distributed lag non-linear modeling framework.
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Callable, List, Tuple
import warnings

from .basis_functions import (
    LinearBasis, PolynomialBasis, SplineBasis, BSplineBasis, 
    StrataBasis, ThresholdBasis, BaseBasisFunction
)
from .utils import mklag, seqlag
from .model_utils import validate_model_compatibility


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
        - 'ns': Natural spline basis
        - 'bs': B-spline basis
        - 'strata': Stratified/categorical basis
        - 'thr': Threshold basis
        - Custom function
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
    }
    
    def __init__(self, x: Union[np.ndarray, List], fun: Union[str, Callable] = 'ns', **kwargs):
        # Store original input
        self.x = np.asarray(x, dtype=float)
        self.fun = fun
        self.range = (np.nanmin(self.x), np.nanmax(self.x))
        
        # Extract centering parameter
        self.cen = kwargs.pop('cen', None)
        
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
            # Use custom function
            basis = self.fun(self.x, **kwargs)
            basis = np.asarray(basis)
            
            # Ensure it's a matrix
            if basis.ndim == 1:
                basis = basis.reshape(-1, 1)
            
            # Try to extract attributes from function
            attributes = {'fun': self.fun}
            if hasattr(basis, '__dict__'):
                attributes.update(getattr(basis, '__dict__', {}))
            attributes.update(kwargs)
            
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
        if 'df' in self.attributes:
            summary_lines.append(f"Degrees of freedom: {self.attributes['df']}")
        if 'degree' in self.attributes:
            summary_lines.append(f"Degree: {self.attributes['degree']}")
        if 'knots' in self.attributes and len(self.attributes['knots']) > 0:
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
        
        # Set default arguments
        self.argvar = argvar or {}
        self.arglag = arglag or {}
        
        # Set default lag arguments if not specified
        if len(self.arglag) == 0 or np.diff(self.lag)[0] == 0:
            self.arglag = {'fun': 'strata', 'df': 1, 'intercept': True}
        
        # Add intercept by default for lag basis if not specified
        if 'intercept' not in self.arglag:
            self.arglag['intercept'] = True
        
        # Force uncentered transformations for lag
        self.arglag['cen'] = None
        
        # Store group for potential seasonal analysis
        self.group = group
        
        # Create the cross-basis
        self._create_cross_basis()
        
        # Set attributes
        self.range = (np.nanmin(self.x), np.nanmax(self.x))
        self._set_names()
    
    def _create_cross_basis(self):
        """Create the cross-basis matrix using tensor products."""
        
        # Create exposure basis (var dimension)
        if self.x.shape[1] == 1:
            # Time series: use the single column
            x_var = self.x.flatten()
        else:
            # Matrix: use all values for basis (flattened approach)
            x_var = self.x.flatten()
        
        # Create OneBasis for exposure dimension
        self.basisvar = OneBasis(x_var, **self.argvar)
        
        # Create lag basis
        lag_seq = seqlag(self.lag)
        self.basislag = OneBasis(lag_seq, **self.arglag)
        
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
        """Create cross-basis for time series data."""
        lag_seq = seqlag(self.lag)
        
        for v in range(n_var_basis):
            # Get the v-th variable basis function values
            var_basis_col = self.basisvar.basis[:, v]
            
            # Create lagged matrix for this basis function
            lagged_matrix = self._create_lagged_matrix(var_basis_col, lag_seq)
            
            # Apply lag basis functions
            for l in range(n_lag_basis):
                col_idx = n_lag_basis * v + l
                self.basis[:, col_idx] = lagged_matrix @ self.basislag.basis[:, l]
    
    def _create_matrix_basis(self, n_var_basis: int, n_lag_basis: int):
        """Create cross-basis for matrix data."""
        n_obs = self.x.shape[0]
        
        # For matrix input, each row of x represents lagged exposures
        for v in range(n_var_basis):
            # Create matrix where each row has the same basis function value
            var_basis_values = self.basisvar.basis[v::n_obs, v] if len(self.basisvar.basis) > n_obs else np.full(n_obs, self.basisvar.basis[0, v])
            var_matrix = np.tile(var_basis_values.reshape(-1, 1), (1, self.x.shape[1]))
            
            # Apply lag basis functions
            for l in range(n_lag_basis):
                col_idx = n_lag_basis * v + l
                self.basis[:, col_idx] = (var_matrix * self.x) @ self.basislag.basis[:, l]
    
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
            Matrix with lagged values
        """
        n_obs = len(values)
        n_lags = len(lag_seq)
        lagged_matrix = np.zeros((n_obs, n_lags))
        
        for i, lag_val in enumerate(lag_seq):
            lag_int = int(lag_val)
            if lag_int == 0:
                lagged_matrix[:, i] = values
            elif lag_int > 0:
                # Positive lag: shift backwards in time
                lagged_matrix[lag_int:, i] = values[:-lag_int]
            else:
                # Negative lag: shift forwards in time  
                lagged_matrix[:lag_int, i] = values[-lag_int:]
        
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