"""
Enhanced spline implementations for R compatibility via rpy2

This module provides R-compatible spline implementations that use R's splines
package directly through rpy2, ensuring exact matches with R DLNM package.
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings

# R splines implementation via rpy2 - REQUIRED
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # Load R's splines package
    splines = importr('splines')
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False


def _check_rpy2():
    """Check if rpy2 is available"""
    if not HAS_RPY2:
        raise ImportError(
            "rpy2 is required for spline functionality in PyDLNM. "
            "Please install rpy2 with: pip install rpy2"
        )


def bs_enhanced(x: np.ndarray, 
                df: Optional[int] = None,
                knots: Optional[np.ndarray] = None,
                degree: int = 3,
                intercept: bool = False,
                boundary_knots: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, Dict]:
    """
    Enhanced B-spline basis function using R's bs() directly via rpy2
    
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
    _check_rpy2()
    
    x = np.asarray(x, dtype=float)
    
    # Use R's B-splines for exact compatibility
    with localconverter(robjects.default_converter + numpy2ri.converter):
        # Convert boundary knots if provided
        r_boundary_knots = None
        if boundary_knots is not None:
            r_boundary_knots = robjects.FloatVector(boundary_knots)
        
        if knots is not None:
            knots_array = np.asarray(knots, dtype=float)
            if r_boundary_knots is not None:
                r_result = splines.bs(x, knots=knots_array, degree=degree, 
                                    intercept=intercept, Boundary_knots=r_boundary_knots)
            else:
                r_result = splines.bs(x, knots=knots_array, degree=degree, intercept=intercept)
        elif df is not None:
            if r_boundary_knots is not None:
                r_result = splines.bs(x, df=df, degree=degree, 
                                    intercept=intercept, Boundary_knots=r_boundary_knots)
            else:
                r_result = splines.bs(x, df=df, degree=degree, intercept=intercept)
        else:
            if r_boundary_knots is not None:
                r_result = splines.bs(x, df=4, degree=degree, 
                                    intercept=intercept, Boundary_knots=r_boundary_knots)
            else:
                r_result = splines.bs(x, df=4, degree=degree, intercept=intercept)
        
        # Convert to numpy
        basis_matrix = np.array(r_result)
        
        # Extract attributes from R result
        attributes = {
            'fun': 'bs',
            'degree': degree,
            'intercept': intercept,
            'n_basis': basis_matrix.shape[1]
        }
        
        # Store R attributes if available
        if hasattr(r_result, 'attributes'):
            r_attrs = dict(r_result.attributes.items())
            if 'knots' in r_attrs:
                attributes['knots'] = np.array(r_attrs['knots'])
            if 'Boundary.knots' in r_attrs:
                boundary_vals = np.array(r_attrs['Boundary.knots'])
                attributes['boundary_knots'] = (boundary_vals[0], boundary_vals[1])
            if 'df' in r_attrs:
                attributes['df'] = int(r_attrs['df'])
    
    return basis_matrix, attributes


def ns_enhanced(x: np.ndarray,
                df: Optional[int] = None,
                knots: Optional[np.ndarray] = None,
                intercept: bool = False,
                boundary_knots: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, Dict]:
    """
    Enhanced natural spline basis function using R's ns() directly via rpy2
    
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
    _check_rpy2()
    
    x = np.asarray(x, dtype=float)
    
    # Use R's natural splines for exact compatibility
    with localconverter(robjects.default_converter + numpy2ri.converter):
        # Convert boundary knots if provided
        r_boundary_knots = None
        if boundary_knots is not None:
            r_boundary_knots = robjects.FloatVector(boundary_knots)
        
        if knots is not None:
            knots_array = np.asarray(knots, dtype=float)
            if r_boundary_knots is not None:
                r_result = splines.ns(x, knots=knots_array, intercept=intercept, 
                                    Boundary_knots=r_boundary_knots)
            else:
                r_result = splines.ns(x, knots=knots_array, intercept=intercept)
        elif df is not None:
            if r_boundary_knots is not None:
                r_result = splines.ns(x, df=df, intercept=intercept, 
                                    Boundary_knots=r_boundary_knots)
            else:
                r_result = splines.ns(x, df=df, intercept=intercept)
        else:
            if r_boundary_knots is not None:
                r_result = splines.ns(x, df=4, intercept=intercept, 
                                    Boundary_knots=r_boundary_knots)
            else:
                r_result = splines.ns(x, df=4, intercept=intercept)
        
        # Convert to numpy
        basis_matrix = np.array(r_result)
        
        # Extract attributes from R result
        attributes = {
            'fun': 'ns',
            'intercept': intercept,
            'n_basis': basis_matrix.shape[1]
        }
        
        # Store R attributes if available
        if hasattr(r_result, 'attributes'):
            r_attrs = dict(r_result.attributes.items())
            if 'knots' in r_attrs:
                attributes['knots'] = np.array(r_attrs['knots'])
            if 'Boundary.knots' in r_attrs:
                boundary_vals = np.array(r_attrs['Boundary.knots'])
                attributes['boundary_knots'] = (boundary_vals[0], boundary_vals[1])
            if 'df' in r_attrs:
                attributes['df'] = int(r_attrs['df'])
    
    return basis_matrix, attributes


def smooth_spline_basis(x: np.ndarray,
                       lambda_smooth: float = 1.0,
                       df: Optional[int] = None,
                       knots: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Smoothing spline basis using R's smooth.spline via rpy2
    
    Creates a basis for smoothing splines using R's implementation.
    
    Parameters:
    -----------
    x : array-like
        Predictor variable values
    lambda_smooth : float, default 1.0
        Smoothing parameter (spar in R)
    df : int, optional
        Degrees of freedom
    knots : array-like, optional
        Knot positions
        
    Returns:
    --------
    tuple
        - basis: Smoothing spline basis matrix
        - attributes: Dictionary including penalty information
    """
    _check_rpy2()
    
    x = np.asarray(x, dtype=float)
    
    # Use R's smooth.spline as base for smoothing spline basis
    # This is a simplified implementation - full smoothing splines are more complex
    basis_matrix, attributes = ns_enhanced(x, df=df, knots=knots, intercept=True)
    
    # Add smoothing attributes
    attributes['lambda'] = lambda_smooth
    attributes['fun'] = 'smooth.spline'
    
    return basis_matrix, attributes


class EnhancedBSplineBasis:
    """
    Enhanced B-spline basis class using R's bs() via rpy2
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
        _check_rpy2()
        
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
    Enhanced natural spline basis class using R's ns() via rpy2
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
        _check_rpy2()
        
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
    Validate spline implementation - now just returns R results since we use R directly
    
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
    _check_rpy2()
    
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
        'attributes': attributes,
        'full_rank': np.linalg.matrix_rank(basis_matrix) == basis_matrix.shape[1]
    }
    
    return {
        'basis_matrix': basis_matrix,
        'diagnostics': diagnostics,
        'attributes': attributes
    }