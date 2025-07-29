"""
rpy2 GLM integration for PyDLNM

This module provides interfaces to run R's GLM directly from Python,
ensuring exact coefficient matches with pure R implementations.
"""

import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple
import warnings

# Set R environment before importing rpy2
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'

try:
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

from .basis import CrossBasis


class Rpy2GLMInterface:
    """
    Interface to run R's GLM directly using rpy2.
    
    This class provides methods to fit GLMs using R's exact implementation,
    ensuring perfect coefficient and variance-covariance matrix matches.
    
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
        
        # Get R interface (don't use deprecated activate)
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
    
    def fit_glm(self, 
                y: np.ndarray,
                family: str = 'quasipoisson',
                other_vars: Optional[np.ndarray] = None,
                formula_vars: Optional[list] = None,
                **kwargs) -> Any:
        """
        Fit GLM using R's glm() function directly.
        
        Parameters
        ----------
        y : array-like
            Response variable (e.g., mortality counts)
        family : str, default='quasipoisson'
            GLM family: 'poisson', 'quasipoisson', 'gaussian', 'gamma', 'binomial'
        other_vars : array-like, optional
            Additional covariates (e.g., seasonality, day of week)
        formula_vars : list of str, optional
            Names for other variables
        **kwargs
            Additional arguments passed to R's glm()
            
        Returns
        -------
        r_model : R object
            Fitted R GLM model object
        """
        
        # Prepare data
        y = np.asarray(y).flatten()
        cb_matrix = np.array(self.crossbasis.basis)
        
        print(f"CrossBasis matrix shape: {cb_matrix.shape}")
        
        # Handle NaN values (equivalent to R's na.action="na.exclude")
        if other_vars is not None:
            other_vars = np.asarray(other_vars)
            if other_vars.ndim == 1:
                other_vars = other_vars.reshape(-1, 1)
            X_full = np.column_stack([cb_matrix, other_vars])
        else:
            X_full = cb_matrix
        
        # Find rows with any NaN values
        nan_mask = np.isnan(X_full).any(axis=1) | np.isnan(y)
        
        if np.any(nan_mask):
            # Exclude observations with NaN values
            cb_clean = cb_matrix[~nan_mask]
            y_clean = y[~nan_mask]
            if other_vars is not None:
                other_clean = other_vars[~nan_mask]
            else:
                other_clean = None
            n_excluded = np.sum(nan_mask)
            print(f"Note: Excluding {n_excluded} observations with missing values (following R na.exclude)")
        else:
            cb_clean = cb_matrix
            y_clean = y
            other_clean = other_vars
        
        # Create R data frame
        data_dict = {'death': y_clean}
        
        # Add cross-basis columns
        for i in range(cb_clean.shape[1]):
            data_dict[f'cb.v{(i//5)+1}.l{(i%5)+1}'] = cb_clean[:, i]
        
        # Add other variables
        if other_clean is not None:
            if formula_vars is not None and len(formula_vars) == other_clean.shape[1]:
                for i, var_name in enumerate(formula_vars):
                    data_dict[var_name] = other_clean[:, i]
            else:
                for i in range(other_clean.shape[1]):
                    data_dict[f'var{i+1}'] = other_clean[:, i]
        
        # Convert to R dataframe
        df = pd.DataFrame(data_dict)
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)
        
        # Store in R global environment
        ro.globalenv['model_data'] = r_df
        
        # Build formula
        cb_terms = [f'cb.v{(i//5)+1}.l{(i%5)+1}' for i in range(cb_clean.shape[1])]
        cb_formula = ' + '.join(cb_terms)
        
        other_terms = []
        if other_clean is not None:
            if formula_vars is not None:
                other_terms = formula_vars
            else:
                other_terms = [f'var{i+1}' for i in range(other_clean.shape[1])]
        
        if other_terms:
            formula = f"death ~ {cb_formula} + {' + '.join(other_terms)}"
        else:
            formula = f"death ~ {cb_formula}"
        
        print(f"R GLM formula: {formula}")
        
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
        
        # Extract coefficients and vcov
        self._extract_cb_coefficients()
        
        print(f"R GLM fitted successfully")
        print(f"Observations used: {self.r('nobs(fitted_model)')[0]}")
        print(f"Dispersion parameter: {self.r('summary(fitted_model)$dispersion')[0]:.6f}")
        
        return fitted_model
    
    def _extract_cb_coefficients(self):
        """Extract cross-basis coefficients and variance-covariance matrix from R model"""
        
        # Get all coefficients
        all_coef = self.r('coef(fitted_model)')
        all_vcov = self.r('vcov(fitted_model)')
        
        # Find cross-basis coefficient indices (exclude intercept)
        coef_names = list(self.r('names(coef(fitted_model))'))
        cb_indices = [i for i, name in enumerate(coef_names) if name.startswith('cb.')]
        
        # Extract cross-basis coefficients and vcov
        self.cb_coef = np.array([all_coef[i] for i in cb_indices])
        
        # Extract cross-basis vcov matrix
        vcov_array = np.array(all_vcov)
        self.cb_vcov = vcov_array[np.ix_(cb_indices, cb_indices)]
        
        print(f"Extracted {len(self.cb_coef)} cross-basis coefficients")
    
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
        
        if not self.has_dlnm:
            # Fallback to Python crossreduce
            from .crossreduce import crossreduce
            return crossreduce(self.crossbasis, self.cb_coef, self.cb_vcov, cen=cen)
        
        # Use R's crossreduce with dlnm package
        cb_matrix = np.array(self.crossbasis.basis)
        
        # Convert numpy array to R matrix directly
        ro.globalenv['cb_matrix'] = ro.r.matrix(cb_matrix.flatten(), nrow=cb_matrix.shape[0], ncol=cb_matrix.shape[1])
        
        # Set crossbasis attributes in R to match dlnm format
        lag_range = self.crossbasis.lag
        df_var, df_lag = self.crossbasis.df
        
        r_code = f'''
        # Set up crossbasis object in R format
        attributes(cb_matrix) <- list(
            dim = c({cb_matrix.shape[0]}, {cb_matrix.shape[1]}),
            class = c("crossbasis", "matrix"),
            lag = c({lag_range[0]}, {lag_range[1]}),
            df = c({df_var}, {df_lag}),
            range = c({self.crossbasis.range[0]}, {self.crossbasis.range[1]})
        )
        '''
        
        if cen is not None:
            r_code += f'''
            # Perform crossreduce
            reduced <- crossreduce(cb_matrix, fitted_model, cen={cen})
            '''
        else:
            r_code += f'''
            # Perform crossreduce
            reduced <- crossreduce(cb_matrix, fitted_model)
            '''
        
        self.r(r_code)
        
        # Extract results
        reduced_coef = np.array(self.r('coef(reduced)'))
        reduced_vcov = np.array(self.r('vcov(reduced)'))
        
        return {
            'coef': reduced_coef,
            'vcov': reduced_vcov
        }