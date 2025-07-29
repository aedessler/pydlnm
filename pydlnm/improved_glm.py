"""
Improved GLM implementation for PyDLNM to match R DLNM exactly

This module provides enhanced GLM functionality with proper seasonality adjustment
and day-of-week factors to address the systematic MMT bias identified in the comparison.
"""

import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple
import warnings
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

# Set R environment before importing rpy2
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'

try:
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

from .basis import CrossBasis
from .basis_functions import SplineBasis


class ImprovedGLMInterface:
    """
    Enhanced GLM interface that matches R DLNM's model specification exactly.
    
    Key improvements:
    1. Proper seasonality adjustment using natural splines on date
    2. Day-of-week factor handling
    3. Exact formula matching: death ~ cb + dow + ns(date, df=dfseas*nyears)
    
    Parameters
    ----------
    crossbasis : CrossBasis
        The cross-basis matrix object
    """
    
    def __init__(self, crossbasis: CrossBasis):
        if not HAS_RPY2:
            raise ImportError("rpy2 is required for R GLM integration. Install with: pip install rpy2")
        
        self.crossbasis = crossbasis
        self.r_model = None
        self.fitted_values = None
        self.cb_coef = None
        self.cb_vcov = None
        
        # Initialize R environment
        self._setup_r_environment()
    
    def _setup_r_environment(self):
        """Set up R environment and load required packages"""
        
        # Get R interface
        self.r = ro.r
        
        # Add user library to R library path
        self.r('if(!("~/R/library" %in% .libPaths())) .libPaths(c("~/R/library", .libPaths()))')
        
        # Load required R packages
        try:
            self.stats = importr('stats')
            self.splines = importr('splines')
            try:
                self.dlnm = importr('dlnm')
                self.has_dlnm = True
            except:
                self.dlnm = None
                self.has_dlnm = False
                warnings.warn("dlnm package not available. crossreduce functionality will be limited.")
        except Exception as e:
            raise ImportError(f"Failed to load required R packages: {str(e)}")
    
    def create_seasonality_basis(self, dates: pd.Series, dfseas: int = 8) -> np.ndarray:
        """
        Create seasonality basis using natural splines on date.
        
        Matches R formula: ns(date, df=dfseas*length(unique(year)))
        
        Parameters
        ----------
        dates : pd.Series
            Date series (datetime objects)
        dfseas : int, default=8
            Degrees of freedom per year for seasonality
            
        Returns
        -------
        np.ndarray
            Natural spline basis matrix for seasonality
        """
        
        # Convert dates to numeric (days since start)
        if hasattr(dates, 'dt'):
            # dates is a pandas Series
            date_numeric = (dates - dates.min()).dt.days.values
        else:
            # dates is a numpy datetime array or similar
            date_diffs = dates - dates.min()
            if hasattr(date_diffs, 'days'):
                date_numeric = date_diffs.days
            else:
                # Convert timedelta64 to days
                date_numeric = date_diffs / np.timedelta64(1, 'D')
        
        # Calculate degrees of freedom as in R: dfseas * number of unique years
        if hasattr(dates, 'dt'):
            years = dates.dt.year.unique()
        else:
            # Extract year from numpy datetime64 array
            if hasattr(dates, 'year'):
                years = np.unique(dates.year)
            else:
                # For numpy datetime64 arrays
                years = np.unique(dates.astype('datetime64[Y]').astype(int) + 1970)
        n_years = len(years)
        total_df = dfseas * n_years
        
        print(f"Seasonality: {n_years} years, {dfseas} df/year = {total_df} total df")
        
        # Use R to create natural splines (exact match)
        # Set up converter context
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            ro.globalenv['date_numeric'] = date_numeric
            ro.globalenv['total_df'] = total_df
        
        # Create natural splines in R
        self.r('season_basis <- splines::ns(date_numeric, df=total_df)')
        season_matrix = np.array(self.r('season_basis'))
        
        print(f"Seasonality basis shape: {season_matrix.shape}")
        
        return season_matrix
    
    def create_dow_factors(self, dates: pd.Series) -> np.ndarray:
        """
        Create day-of-week factor matrix.
        
        Matches R: as.factor(weekdays(date))
        
        Parameters
        ----------
        dates : pd.Series  
            Date series (datetime objects)
            
        Returns
        -------
        np.ndarray
            Day-of-week dummy variable matrix (excluding reference level)
        """
        
        # Get day of week names (matching R weekdays() function)
        if hasattr(dates, 'dt'):
            dow_names = dates.dt.day_name()
        else:
            # Convert numpy datetime64 to pandas Series for easier manipulation
            if isinstance(dates[0], np.datetime64):
                dates_series = pd.Series(dates)
                dow_names = dates_series.dt.day_name()
            else:
                dow_names = pd.Series([date.strftime('%A') for date in dates])
        
        # Create dummy variables (drop first category as reference)
        dow_dummies = pd.get_dummies(dow_names, drop_first=True)
        
        print(f"Day-of-week factors: {list(dow_dummies.columns)} (reference: {dow_names.iloc[0]})")
        print(f"Day-of-week matrix shape: {dow_dummies.shape}")
        
        return dow_dummies.values
    
    def fit_dlnm_model(self, 
                       y: np.ndarray,
                       dates: pd.Series,
                       dfseas: int = 8,
                       family: str = 'quasipoisson',
                       **kwargs) -> Any:
        """
        Fit the complete DLNM model matching R exactly.
        
        Formula: death ~ cb + dow + ns(date, df=dfseas*nyears)
        
        Parameters
        ----------
        y : array-like
            Response variable (mortality counts)
        dates : pd.Series
            Date series for seasonality and day-of-week
        dfseas : int, default=8
            Seasonal degrees of freedom per year
        family : str, default='quasipoisson'
            GLM family
        **kwargs
            Additional arguments
            
        Returns
        -------
        r_model : R object
            Fitted R GLM model object
        """
        
        print(f"\n=== Fitting Enhanced DLNM Model ===")
        print(f"Response variable: {len(y)} observations")
        print(f"Date range: {dates.min()} to {dates.max()}")
        
        # Prepare cross-basis matrix
        cb_matrix = np.array(self.crossbasis.basis)
        print(f"Cross-basis matrix shape: {cb_matrix.shape}")
        
        # Create seasonality basis
        season_matrix = self.create_seasonality_basis(dates, dfseas)
        
        # Create day-of-week factors  
        dow_matrix = self.create_dow_factors(dates)
        
        # Combine all predictors
        X_full = np.column_stack([cb_matrix, dow_matrix, season_matrix])
        
        print(f"Full design matrix shape: {X_full.shape}")
        print(f"  - Cross-basis: {cb_matrix.shape[1]} columns")
        print(f"  - Day-of-week: {dow_matrix.shape[1]} columns")  
        print(f"  - Seasonality: {season_matrix.shape[1]} columns")
        
        # Handle missing values (equivalent to R's na.action="na.exclude")
        nan_mask = np.isnan(X_full).any(axis=1) | np.isnan(y)
        
        if np.any(nan_mask):
            X_clean = X_full[~nan_mask]
            y_clean = y[~nan_mask]
            n_excluded = np.sum(nan_mask)
            print(f"Excluding {n_excluded} observations with missing values")
        else:
            X_clean = X_full
            y_clean = y
        
        # Create column names matching R conventions
        cb_names = [f'cb.v{(i//5)+1}.l{(i%5)+1}' for i in range(cb_matrix.shape[1])]
        
        # Day-of-week names (R uses full day names, drop Monday as reference)
        dow_names = ['dowTuesday', 'dowWednesday', 'dowThursday', 'dowFriday', 'dowSaturday', 'dowSunday']
        dow_names = dow_names[:dow_matrix.shape[1]]  # In case we have fewer
        
        # Seasonality names
        season_names = [f'ns.date..df...total_df.{i+1}' for i in range(season_matrix.shape[1])]
        
        all_names = cb_names + dow_names + season_names
        
        # Create R data frame
        data_dict = {'death': y_clean}
        for i, name in enumerate(all_names):
            data_dict[name] = X_clean[:, i]
        
        df = pd.DataFrame(data_dict)
        
        # Convert to R and fit model
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)
        
        ro.globalenv['model_data'] = r_df
        
        # Build formula string
        predictor_formula = ' + '.join(all_names)
        formula = f"death ~ {predictor_formula}"
        
        print(f"R GLM formula: death ~ cb_terms + dow_terms + seasonality_terms")
        print(f"Total predictors: {len(all_names)}")
        
        # Fit model in R
        if family == 'quasipoisson':
            r_code = f'''
            fitted_model <- glm({formula}, data=model_data, family=quasipoisson, na.action=na.exclude)
            '''
        else:
            r_code = f'''
            fitted_model <- glm({formula}, data=model_data, family={family}, na.action=na.exclude)
            '''
        
        self.r(r_code)
        
        # Get fitted model
        fitted_model = ro.globalenv['fitted_model']
        self.r_model = fitted_model
        
        # Extract cross-basis coefficients only
        self._extract_cb_coefficients(cb_names)
        
        print(f"✓ R GLM fitted successfully")
        print(f"  Observations used: {self.r('nobs(fitted_model)')[0]}")
        print(f"  Dispersion parameter: {self.r('summary(fitted_model)$dispersion')[0]:.6f}")
        print(f"  Cross-basis coefficients extracted: {len(self.cb_coef)}")
        
        return fitted_model
    
    def _extract_cb_coefficients(self, cb_names: list):
        """Extract cross-basis coefficients and variance-covariance matrix from R model"""
        
        # Get all coefficients
        all_coef = self.r('coef(fitted_model)')
        all_vcov = self.r('vcov(fitted_model)')
        
        # Get coefficient names
        coef_names = list(self.r('names(coef(fitted_model))'))
        
        # Find cross-basis coefficient indices
        cb_indices = []
        for cb_name in cb_names:
            if cb_name in coef_names:
                cb_indices.append(coef_names.index(cb_name))
        
        if len(cb_indices) != len(cb_names):
            print(f"Warning: Expected {len(cb_names)} CB coefficients, found {len(cb_indices)}")
        
        # Extract cross-basis coefficients and vcov
        self.cb_coef = np.array([all_coef[i] for i in cb_indices])
        
        # Extract cross-basis vcov matrix
        vcov_array = np.array(all_vcov)
        self.cb_vcov = vcov_array[np.ix_(cb_indices, cb_indices)]
        
        print(f"Extracted cross-basis coefficients: {len(self.cb_coef)}")
    
    def get_model_summary(self):
        """Get R model summary"""
        if self.r_model is None:
            raise ValueError("No model fitted yet")
        
        return self.r('summary(fitted_model)')
    
    def crossreduce(self, cen: Optional[float] = None, type: str = "overall"):
        """
        Perform crossreduce using R's crossreduce function directly.
        
        Parameters
        ----------
        cen : float, optional
            Centering value for reduction
        type : str, default="overall"
            Type of reduction
            
        Returns
        -------
        dict
            Reduced coefficients and variance-covariance matrix
        """
        if self.r_model is None:
            raise ValueError("No model fitted yet")
        
        # Always use Python crossreduce for better compatibility
        from .crossreduce import crossreduce
        return crossreduce(
            basis=self.crossbasis, 
            coef=self.cb_coef, 
            vcov=self.cb_vcov, 
            cen=cen,
            reduction_type=type
        )


def fit_enhanced_dlnm_model(crossbasis: CrossBasis, 
                           y: np.ndarray, 
                           dates: pd.Series,
                           dfseas: int = 8,
                           family: str = 'quasipoisson') -> Dict[str, Any]:
    """
    Convenience function to fit enhanced DLNM model.
    
    Parameters
    ----------
    crossbasis : CrossBasis
        Cross-basis matrix object
    y : array-like
        Response variable (mortality counts)  
    dates : pd.Series
        Date series for seasonality and day-of-week
    dfseas : int, default=8
        Seasonal degrees of freedom per year
    family : str, default='quasipoisson'
        GLM family
        
    Returns
    -------
    dict
        Dictionary with model results including:
        - 'model': Fitted R model object
        - 'coefficients': Cross-basis coefficients  
        - 'vcov': Cross-basis variance-covariance matrix
        - 'reduced': Cross-reduced results
    """
    
    # Create GLM interface
    glm_interface = ImprovedGLMInterface(crossbasis)
    
    # Fit model
    model = glm_interface.fit_dlnm_model(y, dates, dfseas, family)
    
    # Perform cross-reduction
    # Use mean temperature as centering value
    cen_value = np.nanmean([x for x in crossbasis.x if not np.isnan(x)])
    reduced_obj = glm_interface.crossreduce(cen=cen_value)
    
    return {
        'model': model,
        'coefficients': glm_interface.cb_coef,
        'vcov': glm_interface.cb_vcov,
        'reduced': {
            'coefficients': reduced_obj.coef,
            'vcov': reduced_obj.vcov
        },
        'glm_interface': glm_interface
    }