"""
Enhanced spline implementations for R compatibility via rpy2

This module provides R-compatible spline implementations that use R's splines
package directly through rpy2, ensuring exact matches with R DLNM package.
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings

from rbridge import r_session

# R splines implementation via rpy2 - REQUIRED
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # Load R's splines package
    with r_session():
        splines = importr('splines')
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

from basis_functions import _r_spline_basis


def _check_rpy2():
    """Check if rpy2 is available"""
    if not HAS_RPY2:
        raise ImportError(
            "rpy2 is required for spline functionality in PyDLNM. "
            "Please install rpy2 with: pip install rpy2"
        )


def _spline_attributes(fun: str, basis_matrix: np.ndarray, knots: np.ndarray, boundary: np.ndarray,
                       degree: Optional[int], intercept: bool) -> Dict:
    """Attributes of a spline basis: R's own (the interior knots it used, also when it derived them from ``df``,
    and the boundary knots), the degree and intercept, and the number of columns."""
    attributes = {'fun': fun, 'intercept': intercept, 'n_basis': basis_matrix.shape[1],
                  'knots': np.asarray(knots, dtype=float)}
    if degree is not None:
        attributes['degree'] = degree
    if len(boundary) == 2:
        attributes['boundary_knots'] = (boundary[0], boundary[1])
    return attributes


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
        Degrees of freedom. As in R there is no default: with neither ``df`` nor ``knots`` the basis has no
        interior knots (``degree`` columns without intercept).
    knots : array-like or scalar, optional
        Internal knot locations. If None and ``df`` is given, placed at quantiles of x
    degree : int, default 3
        Degree of the piecewise polynomial (3 for cubic)
    intercept : bool, default False
        Whether to include intercept column
    boundary_knots : tuple, optional
        Boundary knots (min, max). If None, R's default (the range of x)
        
    Returns:
    --------
    tuple
        - basis: B-spline basis matrix
        - attributes: Dictionary with basis information, including the interior ``knots`` and the
          ``boundary_knots`` R used
    """
    _check_rpy2()
    
    basis_matrix, knots_used, boundary_used = _r_spline_basis('bs', x, df, knots, degree, intercept, boundary_knots)
    return basis_matrix, _spline_attributes('bs', basis_matrix, knots_used, boundary_used, degree, intercept)


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
        Degrees of freedom. As in R there is no default: with neither ``df`` nor ``knots`` the basis has no
        interior knots (a single column without intercept).
    knots : array-like or scalar, optional
        Internal knot locations. If None and ``df`` is given, placed at quantiles of x
    intercept : bool, default False
        Whether to include intercept column
    boundary_knots : tuple, optional
        Boundary knots (min, max). If None, R's default (the range of x)
        
    Returns:
    --------
    tuple
        - basis: Natural spline basis matrix
        - attributes: Dictionary with basis information, including the interior ``knots`` and the
          ``boundary_knots`` R used
    """
    _check_rpy2()
    
    basis_matrix, knots_used, boundary_used = _r_spline_basis('ns', x, df, knots, None, intercept, boundary_knots)
    return basis_matrix, _spline_attributes('ns', basis_matrix, knots_used, boundary_used, None, intercept)


def smooth_spline_basis(x: np.ndarray,
                       lambda_smooth: Optional[float] = None,
                       df: Optional[int] = None,
                       knots: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Natural-spline basis with intercept, labelled as a smoothing-spline basis.

    This is NOT a smoothing spline: it returns R's ``ns(x, intercept = TRUE)`` (``df`` defaults to 4 here, unlike
    ``ns_enhanced``, when neither ``df`` nor ``knots`` is given). ``lambda_smooth`` has no effect on the basis; a
    value is accepted only for backwards compatibility and is reported with a ``UserWarning``.

    Parameters:
    -----------
    x : array-like
        Predictor variable values
    lambda_smooth : float, optional
        Ignored (a ``UserWarning`` is issued if given). Recorded in the attributes under ``'lambda'``.
    df : int, optional
        Degrees of freedom
    knots : array-like, optional
        Knot positions
        
    Returns:
    --------
    tuple
        - basis: natural spline basis matrix with intercept
        - attributes: Dictionary of the spline attributes
    """
    _check_rpy2()
    
    if lambda_smooth is not None:
        warnings.warn("smooth_spline_basis does not fit a smoothing spline: 'lambda_smooth' is ignored and the "
                      "basis is a natural spline with intercept", UserWarning, stacklevel=2)
    
    if df is None and knots is None:
        df = 4
    basis_matrix, attributes = ns_enhanced(x, df=df, knots=knots, intercept=True)
    
    # Add smoothing attributes
    attributes['fun'] = 'smooth.spline'
    if lambda_smooth is not None:
        attributes['lambda'] = lambda_smooth
    
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