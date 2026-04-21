"""
Basis function implementations for PyDLNM

This module contains implementations of various basis functions used in distributed
lag non-linear models, including linear, polynomial, spline, and specialized functions.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import Union, Optional, List, Tuple, Any, Dict
import warnings

# R-compatible splines implementation - REQUIRED
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
    Natural spline basis function using R's splines::ns() via rpy2.
    
    Creates natural cubic spline basis functions that exactly match
    R's implementation for guaranteed compatibility.
    
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
                 intercept: bool = False,
                 Boundary_knots: Optional[np.ndarray] = None, **kwargs):
        super().__init__(df=df, knots=knots, intercept=intercept, **kwargs)
        self.df = df
        self.knots = knots
        self.intercept = intercept
        self.Boundary_knots = (np.asarray(Boundary_knots, dtype=float)
                               if Boundary_knots is not None else None)
        self.attributes['fun'] = 'ns'
        self.attributes['df'] = df
        self.attributes['intercept'] = intercept
        if knots is not None:
            self.attributes['knots'] = np.asarray(knots, dtype=float)
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
        x = np.asarray(x, dtype=float)

        # Use R's natural splines for exact compatibility
        with localconverter(robjects.default_converter + numpy2ri.converter):
            bk = (self.Boundary_knots if self.Boundary_knots is not None
                  else np.array([np.nanmin(x), np.nanmax(x)]))
            robjects.globalenv['_ns_x']  = x
            robjects.globalenv['_ns_bk'] = bk
            if self.knots is not None:
                knots_array = np.asarray(self.knots, dtype=float)
                robjects.globalenv['_ns_ik'] = knots_array
                r_result = robjects.r(
                    f'splines::ns(`_ns_x`, knots=`_ns_ik`, '
                    f'intercept={"TRUE" if self.intercept else "FALSE"}, '
                    f'Boundary.knots=`_ns_bk`)'
                )
            elif self.df is not None:
                r_result = robjects.r(
                    f'splines::ns(`_ns_x`, df={int(self.df)}, '
                    f'intercept={"TRUE" if self.intercept else "FALSE"}, '
                    f'Boundary.knots=`_ns_bk`)'
                )
            else:
                r_result = robjects.r(
                    f'splines::ns(`_ns_x`, df=4, '
                    f'intercept={"TRUE" if self.intercept else "FALSE"}, '
                    f'Boundary.knots=`_ns_bk`)'
                )
            
            # Convert to numpy
            basis_matrix = np.array(r_result)
            
            # Store attributes if available
            if hasattr(r_result, 'attributes'):
                r_attrs = dict(r_result.attributes.items())
                if 'knots' in r_attrs:
                    self.internal_knots = np.array(r_attrs['knots'])
            
            return basis_matrix


class BSplineBasis(BaseBasisFunction):
    """
    B-spline basis function using R's splines::bs() via rpy2.

    Creates B-spline basis functions that exactly match R's implementation
    for guaranteed compatibility.

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
    Boundary_knots : array-like, optional
        Boundary knots (min, max of training data).  Maps to R's
        Boundary.knots argument.  When supplied, predictions outside
        the training range use the same boundary knots as training,
        matching R's crossbasis / mkXpred behaviour exactly.
    """

    def __init__(self, df: int = 4, degree: int = 3,
                 knots: Optional[np.ndarray] = None,
                 intercept: bool = False,
                 Boundary_knots: Optional[np.ndarray] = None, **kwargs):
        super().__init__(df=df, degree=degree, knots=knots,
                        intercept=intercept, **kwargs)
        self.df = df
        self.degree = degree
        self.knots = knots
        self.intercept = intercept
        self.Boundary_knots = (np.asarray(Boundary_knots, dtype=float)
                               if Boundary_knots is not None else None)
        self.attributes['fun'] = 'bs'
        self.attributes['df'] = df
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
        x = np.asarray(x, dtype=float)

        # Use R's B-splines for exact compatibility.
        # Pass Boundary.knots so predictions outside the training range use the
        # same boundary knots as training — matching R's crossbasis/mkXpred.
        boundary = (self.Boundary_knots if self.Boundary_knots is not None
                    else np.array([np.nanmin(x), np.nanmax(x)]))

        with localconverter(robjects.default_converter + numpy2ri.converter):
            robjects.globalenv['_bs_x']   = x
            robjects.globalenv['_bs_bk']  = boundary
            if self.knots is not None:
                robjects.globalenv['_bs_ik'] = np.asarray(self.knots, dtype=float)
                r_result = robjects.r(
                    f'splines::bs(`_bs_x`, knots=`_bs_ik`, degree={int(self.degree)}, '
                    f'intercept={"TRUE" if self.intercept else "FALSE"}, '
                    f'Boundary.knots=`_bs_bk`)'
                )
            elif self.df is not None:
                r_result = robjects.r(
                    f'splines::bs(`_bs_x`, df={int(self.df)}, degree={int(self.degree)}, '
                    f'intercept={"TRUE" if self.intercept else "FALSE"}, '
                    f'Boundary.knots=`_bs_bk`)'
                )
            else:
                r_result = robjects.r(
                    f'splines::bs(`_bs_x`, df=4, degree={int(self.degree)}, '
                    f'intercept={"TRUE" if self.intercept else "FALSE"}, '
                    f'Boundary.knots=`_bs_bk`)'
                )

            basis_matrix = np.array(r_result)

            # Cache boundary knots and internal knots from R's output
            if hasattr(r_result, 'attributes'):
                r_attrs = dict(r_result.attributes.items())
                if 'knots' in r_attrs:
                    self.internal_knots = np.array(r_attrs['knots'])
                if 'degree' in r_attrs:
                    self.degree = int(r_attrs['degree'])
                if 'Boundary.knots' in r_attrs and self.Boundary_knots is None:
                    self.Boundary_knots = np.array(r_attrs['Boundary.knots'])

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