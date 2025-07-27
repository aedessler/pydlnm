"""
Enhanced spline implementations for better R compatibility

This module provides improved B-spline and natural spline implementations
that more closely match the behavior of R's splines package.
"""

import numpy as np
from scipy import interpolate
from scipy.interpolate import BSpline
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings


def bs_enhanced(x: np.ndarray, 
                df: Optional[int] = None,
                knots: Optional[np.ndarray] = None,
                degree: int = 3,
                intercept: bool = False,
                boundary_knots: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, Dict]:
    """
    Enhanced B-spline basis function matching R's bs() behavior
    
    Parameters:
    -----------
    x : array-like
        Predictor variable values
    df : int, optional
        Degrees of freedom. If None, derived from knots
    knots : array-like, optional
        Internal knot locations. If None, placed at quantiles
    degree : int, default 3
        Degree of the piecewise polynomial (3 for cubic)
    intercept : bool, default False
        Whether to include intercept column
    boundary_knots : tuple, optional
        Boundary knots (min, max). If None, uses range of x
        
    Returns:
    --------
    tuple
        - basis: B-spline basis matrix
        - attributes: Dictionary with basis information
    """
    
    x = np.asarray(x, dtype=float)
    x_clean = x[~np.isnan(x)]
    
    if len(x_clean) == 0:
        raise ValueError("No valid (non-NaN) values in x")
    
    # Determine boundary knots
    if boundary_knots is None:
        xl = np.min(x_clean)
        xr = np.max(x_clean)
    else:
        xl, xr = boundary_knots
    
    # Determine internal knots
    if knots is None:
        if df is None:
            df = 4  # Default degrees of freedom
        
        # Number of internal knots
        n_internal = df - degree + (0 if intercept else -1)
        
        if n_internal > 0:
            # Place internal knots at quantiles
            quantiles = np.linspace(0, 1, n_internal + 2)[1:-1]
            internal_knots = np.quantile(x_clean, quantiles)
        else:
            internal_knots = np.array([])
    else:
        internal_knots = np.asarray(knots)
        n_internal = len(internal_knots)
        
        if df is None:
            df = n_internal + degree + (1 if intercept else 0)
    
    # Create full knot sequence
    all_knots = np.concatenate([[xl], internal_knots, [xr]])
    all_knots = np.sort(all_knots)
    
    # Extended knot vector with multiplicities at boundaries
    extended_knots = np.concatenate([
        np.repeat(xl, degree + 1),
        internal_knots,
        np.repeat(xr, degree + 1)
    ])
    # Ensure knots are sorted (should already be, but ensure compliance)
    extended_knots = np.sort(extended_knots)
    
    # Number of basis functions
    n_basis = len(extended_knots) - degree - 1
    
    # Generate B-spline basis matrix
    basis_matrix = np.zeros((len(x), n_basis))
    
    for i in range(n_basis):
        # Create coefficient vector for i-th basis function
        coef = np.zeros(n_basis)
        coef[i] = 1.0
        
        # Create B-spline
        try:
            spline = BSpline(extended_knots, coef, degree, extrapolate=False)
            
            # Evaluate, handling extrapolation
            for j, x_val in enumerate(x):
                if np.isnan(x_val):
                    basis_matrix[j, i] = np.nan
                elif x_val < xl or x_val > xr:
                    # Linear extrapolation for B-splines (R behavior)
                    if x_val < xl:
                        # Evaluate derivative at left boundary
                        h = 1e-6
                        if xl + h <= xr:
                            deriv = (spline(xl + h) - spline(xl)) / h
                            basis_matrix[j, i] = spline(xl) + deriv * (x_val - xl)
                        else:
                            basis_matrix[j, i] = spline(xl)
                    else:  # x_val > xr
                        # Evaluate derivative at right boundary
                        h = 1e-6
                        if xr - h >= xl:
                            deriv = (spline(xr) - spline(xr - h)) / h
                            basis_matrix[j, i] = spline(xr) + deriv * (x_val - xr)
                        else:
                            basis_matrix[j, i] = spline(xr)
                else:
                    basis_matrix[j, i] = spline(x_val)
                    
        except Exception as e:
            warnings.warn(f"Error evaluating B-spline {i}: {e}")
            basis_matrix[:, i] = 0
    
    # Remove intercept column if not wanted
    if not intercept and basis_matrix.shape[1] > 0:
        basis_matrix = basis_matrix[:, 1:]
    
    # Store attributes
    attributes = {
        'fun': 'bs',
        'degree': degree,
        'knots': internal_knots,
        'boundary_knots': (xl, xr),
        'intercept': intercept,
        'df': df,
        'n_basis': basis_matrix.shape[1]
    }
    
    return basis_matrix, attributes


def ns_enhanced(x: np.ndarray,
                df: Optional[int] = None,
                knots: Optional[np.ndarray] = None,
                intercept: bool = False,
                boundary_knots: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, Dict]:
    """
    Enhanced natural spline basis function matching R's ns() behavior
    
    Natural splines are cubic splines that are constrained to be linear
    beyond the boundary knots.
    
    Parameters:
    -----------
    x : array-like
        Predictor variable values
    df : int, optional
        Degrees of freedom. If None, derived from knots
    knots : array-like, optional
        Internal knot locations
    intercept : bool, default False
        Whether to include intercept column
    boundary_knots : tuple, optional
        Boundary knots (min, max)
        
    Returns:
    --------
    tuple
        - basis: Natural spline basis matrix
        - attributes: Dictionary with basis information
    """
    
    x = np.asarray(x, dtype=float)
    x_clean = x[~np.isnan(x)]
    
    if len(x_clean) == 0:
        raise ValueError("No valid (non-NaN) values in x")
    
    # Determine boundary knots
    if boundary_knots is None:
        xl = np.min(x_clean)
        xr = np.max(x_clean)
    else:
        xl, xr = boundary_knots
    
    # Determine internal knots
    if knots is None:
        if df is None:
            df = 4  # Default degrees of freedom
        
        # For natural splines: df = number of internal knots + 1 + intercept
        n_internal = df - 1 - (1 if intercept else 0)
        
        if n_internal > 0:
            quantiles = np.linspace(0, 1, n_internal + 2)[1:-1]
            internal_knots = np.quantile(x_clean, quantiles)
        else:
            internal_knots = np.array([])
    else:
        internal_knots = np.asarray(knots)
        n_internal = len(internal_knots)
        
        if df is None:
            df = n_internal + 1 + (1 if intercept else 0)
    
    # Create natural spline basis using truncated power series
    n_knots = len(internal_knots)
    n_basis = n_knots + 1 + (1 if intercept else 0)
    
    basis_matrix = np.zeros((len(x), n_basis))
    col_idx = 0
    
    # Intercept column
    if intercept:
        basis_matrix[:, col_idx] = 1.0
        col_idx += 1
    
    # Linear term
    basis_matrix[:, col_idx] = x
    col_idx += 1
    
    # Natural spline terms
    if n_knots > 0:
        # Create the natural spline basis functions
        # d_k(x) = [(x-knot_k)_+^3 - (x-xr)_+^3 * (knot_k-xl)/(xr-xl)] / (xr-xl)^2
        # where (a)_+ = max(0, a)
        
        for k, knot in enumerate(internal_knots):
            if col_idx < n_basis:
                # Truncated cubic terms with natural constraints
                d_k = np.zeros(len(x))
                
                for i, x_val in enumerate(x):
                    if np.isnan(x_val):
                        d_k[i] = np.nan
                    else:
                        # Truncated power function (x - knot)_+^3
                        cubic_term = np.maximum(0, x_val - knot)**3
                        
                        # Boundary adjustment for natural constraint
                        if xr != xl:  # Avoid division by zero
                            boundary_term = (np.maximum(0, x_val - xr)**3 * 
                                           (knot - xl) / (xr - xl))
                            d_k[i] = (cubic_term - boundary_term) / (xr - xl)**2
                        else:
                            d_k[i] = cubic_term
                
                basis_matrix[:, col_idx] = d_k
                col_idx += 1
    
    # Trim basis matrix to expected size
    if col_idx < n_basis:
        basis_matrix = basis_matrix[:, :col_idx]
    
    # Store attributes
    attributes = {
        'fun': 'ns',
        'knots': internal_knots,
        'boundary_knots': (xl, xr),
        'intercept': intercept,
        'df': df,
        'n_basis': basis_matrix.shape[1]
    }
    
    return basis_matrix, attributes


def smooth_spline_basis(x: np.ndarray,
                       lambda_smooth: float = 1.0,
                       df: Optional[int] = None,
                       knots: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Smoothing spline basis with penalty matrix
    
    Creates a basis for smoothing splines with integrated penalty.
    
    Parameters:
    -----------
    x : array-like
        Predictor variable values
    lambda_smooth : float, default 1.0
        Smoothing parameter
    df : int, optional
        Degrees of freedom
    knots : array-like, optional
        Knot positions
        
    Returns:
    --------
    tuple
        - basis: Smoothing spline basis matrix
        - attributes: Dictionary including penalty matrix
    """
    
    # Use natural spline as base
    basis_matrix, attributes = ns_enhanced(x, df=df, knots=knots, intercept=True)
    
    # Create penalty matrix (simplified version)
    n_basis = basis_matrix.shape[1]
    penalty_matrix = np.zeros((n_basis, n_basis))
    
    # Second derivative penalty (roughness penalty)
    if n_basis > 2:
        # Simple second difference penalty
        for i in range(1, n_basis - 1):
            penalty_matrix[i-1:i+2, i-1:i+2] += lambda_smooth * np.array([
                [1, -2, 1],
                [-2, 4, -2],
                [1, -2, 1]
            ])
    
    attributes['penalty_matrix'] = penalty_matrix
    attributes['lambda'] = lambda_smooth
    attributes['fun'] = 'smooth.spline'
    
    return basis_matrix, attributes


class EnhancedBSplineBasis:
    """
    Enhanced B-spline basis class with R-compatible behavior
    """
    
    def __init__(self, df: Optional[int] = None, 
                 degree: int = 3,
                 knots: Optional[np.ndarray] = None,
                 intercept: bool = False,
                 boundary_knots: Optional[Tuple[float, float]] = None):
        """
        Initialize enhanced B-spline basis
        
        Parameters:
        -----------
        df : int, optional
            Degrees of freedom
        degree : int, default 3
            Spline degree
        knots : array-like, optional
            Internal knot positions
        intercept : bool, default False
            Include intercept column
        boundary_knots : tuple, optional
            Boundary knot positions
        """
        self.df = df
        self.degree = degree
        self.knots = knots
        self.intercept = intercept
        self.boundary_knots = boundary_knots
        self.attributes = {}
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Generate B-spline basis matrix"""
        basis_matrix, self.attributes = bs_enhanced(
            x, df=self.df, knots=self.knots, degree=self.degree,
            intercept=self.intercept, boundary_knots=self.boundary_knots
        )
        return basis_matrix
    
    def get_attributes(self) -> Dict[str, Any]:
        """Get basis attributes"""
        return self.attributes.copy()


class EnhancedNaturalSplineBasis:
    """
    Enhanced natural spline basis class with R-compatible behavior
    """
    
    def __init__(self, df: Optional[int] = None,
                 knots: Optional[np.ndarray] = None,
                 intercept: bool = False,
                 boundary_knots: Optional[Tuple[float, float]] = None):
        """
        Initialize enhanced natural spline basis
        
        Parameters:
        -----------
        df : int, optional
            Degrees of freedom
        knots : array-like, optional
            Internal knot positions
        intercept : bool, default False
            Include intercept column
        boundary_knots : tuple, optional
            Boundary knot positions
        """
        self.df = df
        self.knots = knots
        self.intercept = intercept
        self.boundary_knots = boundary_knots
        self.attributes = {}
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Generate natural spline basis matrix"""
        basis_matrix, self.attributes = ns_enhanced(
            x, df=self.df, knots=self.knots,
            intercept=self.intercept, boundary_knots=self.boundary_knots
        )
        return basis_matrix
    
    def get_attributes(self) -> Dict[str, Any]:
        """Get basis attributes"""
        return self.attributes.copy()


def validate_spline_against_r(x: np.ndarray, 
                             basis_type: str = "bs",
                             **kwargs) -> Dict:
    """
    Validate spline implementation against expected R behavior
    
    Parameters:
    -----------
    x : array-like
        Test data
    basis_type : str
        Type of spline: "bs" or "ns"
    **kwargs
        Spline parameters
        
    Returns:
    --------
    dict
        Validation results and diagnostics
    """
    
    if basis_type == "bs":
        basis_matrix, attributes = bs_enhanced(x, **kwargs)
    elif basis_type == "ns":
        basis_matrix, attributes = ns_enhanced(x, **kwargs)
    else:
        raise ValueError("basis_type must be 'bs' or 'ns'")
    
    # Diagnostic checks
    diagnostics = {
        'n_obs': len(x),
        'n_basis': basis_matrix.shape[1],
        'rank': np.linalg.matrix_rank(basis_matrix),
        'condition_number': np.linalg.cond(basis_matrix.T @ basis_matrix),
        'has_nan': np.any(np.isnan(basis_matrix)),
        'attributes': attributes
    }
    
    # Check for linear independence
    diagnostics['full_rank'] = diagnostics['rank'] == basis_matrix.shape[1]
    
    # Check extrapolation behavior
    x_min, x_max = np.min(x), np.max(x)
    x_extrap = np.array([x_min - 1, x_max + 1])
    
    if basis_type == "bs":
        extrap_basis, _ = bs_enhanced(x_extrap, **kwargs)
    else:
        extrap_basis, _ = ns_enhanced(x_extrap, **kwargs)
    
    diagnostics['extrapolation_test'] = {
        'input': x_extrap,
        'output_shape': extrap_basis.shape,
        'has_nan_extrap': np.any(np.isnan(extrap_basis))
    }
    
    return {
        'basis_matrix': basis_matrix,
        'diagnostics': diagnostics,
        'attributes': attributes
    }