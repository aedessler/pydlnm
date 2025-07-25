"""
Basis function implementations for PyDLNM

This module contains implementations of various basis functions used in distributed
lag non-linear models, including linear, polynomial, spline, and specialized functions.
"""

import numpy as np
from scipy import interpolate
from scipy.interpolate import BSpline
from sklearn.preprocessing import PolynomialFeatures
from typing import Union, Optional, List, Tuple, Any, Dict
import warnings

# Import enhanced spline implementations
from .enhanced_splines import (
    bs_enhanced, ns_enhanced, 
    EnhancedBSplineBasis, EnhancedNaturalSplineBasis
)


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
        x = np.asarray(x, dtype=float)
        
        if self.intercept:
            basis = np.column_stack([np.ones(len(x)), x])
        else:
            basis = x.reshape(-1, 1)
        
        return basis


class PolynomialBasis(BaseBasisFunction):
    """
    Polynomial basis function.
    
    Creates polynomial basis functions of specified degree.
    
    Parameters
    ----------
    degree : int, default=1
        Polynomial degree
    scale : float, optional
        Scaling factor. If None, uses max(abs(x))
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
        x = np.asarray(x, dtype=float)
        
        # Determine scale if not provided
        if self.scale is None:
            scale = np.max(np.abs(x))
            if scale == 0:
                scale = 1.0
        else:
            scale = self.scale
        
        self.attributes['scale'] = scale
        
        # Scale x
        x_scaled = x / scale
        
        # Generate polynomial features
        poly_features = PolynomialFeatures(
            degree=self.degree, 
            include_bias=self.intercept,
            interaction_only=False
        )
        
        basis = poly_features.fit_transform(x_scaled.reshape(-1, 1))
        
        return basis


class SplineBasis(BaseBasisFunction):
    """
    Natural spline basis function.
    
    Creates natural cubic spline basis functions.
    
    Parameters
    ----------
    df : int, default=4
        Degrees of freedom (number of knots + 1)
    knots : array-like, optional
        Interior knot positions. If None, uses quantiles
    intercept : bool, default=False
        Whether to include an intercept column
    """
    
    def __init__(self, df: int = 4, knots: Optional[np.ndarray] = None,
                 intercept: bool = False, **kwargs):
        super().__init__(df=df, knots=knots, intercept=intercept, **kwargs)
        self.df = df
        self.knots = knots
        self.intercept = intercept
        self.attributes['fun'] = 'ns'
        self.attributes['df'] = df
        self.attributes['intercept'] = intercept
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate natural spline basis matrix using enhanced implementation.
        
        Parameters
        ----------
        x : array-like
            Input vector
            
        Returns
        -------
        np.ndarray
            Natural spline basis matrix
        """
        # Use enhanced natural spline implementation
        basis_matrix, enhanced_attrs = ns_enhanced(
            x,
            df=self.df,
            knots=self.knots,
            intercept=self.intercept
        )
        
        # Update attributes with enhanced information
        self.attributes.update(enhanced_attrs)
        
        return basis_matrix


class BSplineBasis(BaseBasisFunction):
    """
    B-spline basis function.
    
    Creates B-spline basis functions without natural constraints.
    
    Parameters
    ----------
    df : int, default=4
        Degrees of freedom
    degree : int, default=3
        B-spline degree
    knots : array-like, optional
        Interior knot positions. If None, uses quantiles
    intercept : bool, default=False
        Whether to include an intercept column
    """
    
    def __init__(self, df: int = 4, degree: int = 3, 
                 knots: Optional[np.ndarray] = None,
                 intercept: bool = False, **kwargs):
        super().__init__(df=df, degree=degree, knots=knots, 
                        intercept=intercept, **kwargs)
        self.df = df
        self.degree = degree
        self.knots = knots
        self.intercept = intercept
        self.attributes['fun'] = 'bs'
        self.attributes['df'] = df
        self.attributes['degree'] = degree
        self.attributes['intercept'] = intercept
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate B-spline basis matrix using enhanced implementation.
        
        Parameters
        ----------
        x : array-like
            Input vector
            
        Returns
        -------
        np.ndarray
            B-spline basis matrix
        """
        # Use enhanced B-spline implementation
        basis_matrix, enhanced_attrs = bs_enhanced(
            x, 
            df=self.df,
            knots=self.knots,
            degree=self.degree,
            intercept=self.intercept
        )
        
        # Update attributes with enhanced information
        self.attributes.update(enhanced_attrs)
        
        return basis_matrix


class StrataBasis(BaseBasisFunction):
    """
    Stratified/categorical basis function.
    
    Converts continuous variables into categorical strata with indicator variables.
    
    Parameters
    ----------
    df : int, default=1
        Number of categories (strata)
    breaks : array-like, optional
        Cut points for stratification. If None, uses quantiles
    ref : int, default=1
        Reference category (1-based indexing)
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
        self.intercept = intercept
        self.attributes['fun'] = 'strata'
        self.attributes['df'] = df
        self.attributes['ref'] = ref
        self.attributes['intercept'] = intercept
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate stratified basis matrix.
        
        Parameters
        ----------
        x : array-like
            Input vector
            
        Returns
        -------
        np.ndarray
            Stratified basis matrix
        """
        x = np.asarray(x, dtype=float)
        x_clean = x[~np.isnan(x)]
        
        if len(x_clean) == 0:
            raise ValueError("No valid (non-NaN) values in x")
        
        # Determine breaks if not provided
        if self.breaks is None:
            if self.df == 1:
                breaks = [np.median(x_clean)]
            else:
                quantiles = np.linspace(0, 1, self.df + 1)[1:-1]
                breaks = np.quantile(x_clean, quantiles)
        else:
            breaks = np.asarray(self.breaks)
        
        self.attributes['breaks'] = breaks
        
        # Create strata
        strata = np.digitize(x, breaks)
        n_strata = len(breaks) + 1
        
        # Create indicator matrix
        basis = np.zeros((len(x), n_strata))
        for i in range(n_strata):
            basis[:, i] = (strata == i).astype(float)
        
        # Handle reference category
        if self.ref > 0 and self.ref <= n_strata:
            ref_idx = self.ref - 1  # Convert to 0-based
            basis = np.delete(basis, ref_idx, axis=1)
        
        if self.intercept:
            intercept_col = np.ones((len(x), 1))
            basis = np.column_stack([intercept_col, basis])
        
        return basis


class ThresholdBasis(BaseBasisFunction):
    """
    Threshold/hockey-stick basis function.
    
    Creates threshold transformations with different sides.
    
    Parameters
    ----------
    thr_value : float or array-like, optional
        Threshold value(s). If None, uses median
    side : str, default='h'
        Threshold side: 'h' (higher), 'l' (lower), 'd' (double)
    intercept : bool, default=False
        Whether to include an intercept column
    """
    
    def __init__(self, thr_value: Optional[Union[float, np.ndarray]] = None,
                 side: str = 'h', intercept: bool = False, **kwargs):
        super().__init__(thr_value=thr_value, side=side, 
                        intercept=intercept, **kwargs)
        self.thr_value = thr_value
        self.side = side
        self.intercept = intercept
        self.attributes['fun'] = 'thr'
        self.attributes['side'] = side
        self.attributes['intercept'] = intercept
    
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
        x = np.asarray(x, dtype=float)
        x_clean = x[~np.isnan(x)]
        
        if len(x_clean) == 0:
            raise ValueError("No valid (non-NaN) values in x")
        
        # Determine threshold value if not provided
        if self.thr_value is None:
            thr = np.median(x_clean)
        else:
            thr = self.thr_value
        
        if isinstance(thr, (list, tuple, np.ndarray)):
            thr = np.asarray(thr)
        else:
            thr = np.array([thr])
        
        self.attributes['thr.value'] = thr
        
        basis_cols = []
        
        if self.side == 'h':
            # Higher side: max(x - threshold, 0)
            for t in thr:
                basis_cols.append(np.maximum(x - t, 0))
        elif self.side == 'l':
            # Lower side: -min(x - threshold, 0)
            for t in thr:
                basis_cols.append(-np.minimum(x - t, 0))
        elif self.side == 'd':
            # Double side: both higher and lower
            if len(thr) == 1:
                # Use single threshold for both sides
                t = thr[0]
                basis_cols.append(-np.minimum(x - t, 0))  # Lower side
                basis_cols.append(np.maximum(x - t, 0))   # Higher side
            else:
                # Use two thresholds
                t1, t2 = thr[0], thr[1] if len(thr) > 1 else thr[0]
                basis_cols.append(-np.minimum(x - t1, 0))  # Lower side
                basis_cols.append(np.maximum(x - t2, 0))   # Higher side
        else:
            raise ValueError(f"Invalid side '{self.side}'. Must be 'h', 'l', or 'd'")
        
        basis = np.column_stack(basis_cols)
        
        if self.intercept:
            intercept_col = np.ones((len(x), 1))
            basis = np.column_stack([intercept_col, basis])
        
        return basis