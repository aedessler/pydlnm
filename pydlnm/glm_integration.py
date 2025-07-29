"""
GLM integration for PyDLNM

This module provides interfaces between CrossBasis matrices and statistical models,
enabling proper fitting and coefficient extraction for DLNM analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple, List
import warnings

try:
    import statsmodels.api as sm
    import statsmodels.genmod.families as families
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from sklearn.linear_model import PoissonRegressor, GammaRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .basis import CrossBasis


class DLNMGLMInterface:
    """
    Interface between CrossBasis and statistical models.
    
    This class provides methods to fit GLMs with cross-basis matrices,
    extract coefficients and variance-covariance matrices, and handle
    the integration with different statistical modeling libraries.
    
    Parameters
    ----------
    crossbasis : CrossBasis
        The cross-basis matrix object
    """
    
    def __init__(self, crossbasis: CrossBasis):
        self.crossbasis = crossbasis
        self.model = None
        self.model_type = None
        self.fitted_values = None
        self.cb_coef = None
        self.cb_vcov = None
        
    def fit_statsmodels(self, 
                       y: np.ndarray,
                       family: str = 'poisson',
                       other_vars: Optional[np.ndarray] = None,
                       formula_vars: Optional[List[str]] = None,
                       offset: Optional[np.ndarray] = None,
                       **kwargs) -> 'sm.GLM':
        """
        Fit GLM using statsmodels with cross-basis matrix.
        
        Parameters
        ----------
        y : array-like
            Response variable (e.g., mortality counts)
        family : str, default='poisson'
            GLM family: 'poisson', 'quasipoisson', 'gaussian', 'gamma', 'binomial'
        other_vars : array-like, optional
            Additional covariates (e.g., seasonality, day of week)
        formula_vars : list of str, optional
            Names for other variables (for easier interpretation)
        offset : array-like, optional
            Offset term (e.g., log population)
        **kwargs
            Additional arguments passed to GLM
            
        Returns
        -------
        fitted_model : statsmodels GLM
            Fitted GLM model object
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for GLM fitting. Install with: pip install statsmodels")
        
        # Prepare response
        y = np.asarray(y).flatten()
        
        # Get cross-basis matrix
        cb_matrix = np.array(self.crossbasis.basis)
        
        # Combine with other variables
        if other_vars is not None:
            other_vars = np.asarray(other_vars)
            if other_vars.ndim == 1:
                other_vars = other_vars.reshape(-1, 1)
            X = np.column_stack([cb_matrix, other_vars])
        else:
            X = cb_matrix
        
        # Add intercept
        X = sm.add_constant(X)
        
        # Handle NaN values (equivalent to R's na.action="na.exclude")
        # Find rows with any NaN values
        nan_mask = np.isnan(X).any(axis=1) | np.isnan(y)
        
        if np.any(nan_mask):
            # Exclude observations with NaN values (matching R behavior)
            X_clean = X[~nan_mask]
            y_clean = y[~nan_mask] 
            n_excluded = np.sum(nan_mask)
            print(f"Note: Excluding {n_excluded} observations with missing values (following R na.exclude)")
        else:
            X_clean = X
            y_clean = y
        
        # Set up family
        family_map = {
            'poisson': families.Poisson(),
            'quasipoisson': families.Poisson(),  # Handle quasi-Poisson later
            'gaussian': families.Gaussian(),
            'gamma': families.Gamma(),
            'binomial': families.Binomial()
        }
        
        if family not in family_map:
            raise ValueError(f"Unknown family: {family}. Choose from {list(family_map.keys())}")
        
        glm_family = family_map[family]
        
        # Fit model with cleaned data
        self.model = sm.GLM(y_clean, X_clean, family=glm_family, offset=offset, **kwargs)
        fitted_model = self.model.fit()
        
        # Store information about excluded observations
        self.excluded_observations = nan_mask if np.any(nan_mask) else None
        
        # IMPROVED: Handle quasi-Poisson by adjusting scale - match R's approach better
        if family == 'quasipoisson':
            # Use R-compatible quasi-Poisson handling
            # R uses scale='X2' which is closer to this approach
            try:
                # Refit with scale estimation for better R compatibility
                refitted_model = self.model.fit(scale='X2')
                fitted_model = refitted_model
                
                # Estimate dispersion parameter like R
                pearson_chi2 = np.sum(fitted_model.resid_pearson**2)
                dispersion = pearson_chi2 / fitted_model.df_resid
                
                # Apply R-style scaling
                fitted_model.scale = dispersion
                fitted_model.bse = fitted_model.bse * np.sqrt(dispersion)
                fitted_model.cov_params_default = fitted_model.cov_params() * dispersion
                
            except:
                # Fallback to original approach
                pearson_chi2 = np.sum(fitted_model.resid_pearson**2)
                dispersion = pearson_chi2 / fitted_model.df_resid
                fitted_model.scale = dispersion
                fitted_model.bse = fitted_model.bse * np.sqrt(dispersion)
                fitted_model.cov_params_default = fitted_model.cov_params() * dispersion
        
        self.model_type = 'statsmodels'
        self.fitted_values = fitted_model
        
        # Extract cross-basis coefficients and variance-covariance
        self._extract_cb_coefficients(fitted_model, cb_matrix.shape[1])
        
        return fitted_model
    
    def fit_sklearn(self, 
                   y: np.ndarray,
                   model_type: str = 'poisson',
                   other_vars: Optional[np.ndarray] = None,
                   **kwargs) -> Any:
        """
        Fit GLM using sklearn with cross-basis matrix.
        
        Parameters
        ----------
        y : array-like
            Response variable
        model_type : str, default='poisson'
            Model type: 'poisson', 'gamma'
        other_vars : array-like, optional
            Additional covariates
        **kwargs
            Additional arguments passed to sklearn model
            
        Returns
        -------
        fitted_model : sklearn model
            Fitted model object
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required. Install with: pip install scikit-learn")
        
        # Prepare data
        y = np.asarray(y).flatten()
        cb_matrix = np.array(self.crossbasis.basis)
        
        if other_vars is not None:
            other_vars = np.asarray(other_vars)
            if other_vars.ndim == 1:
                other_vars = other_vars.reshape(-1, 1)
            X = np.column_stack([cb_matrix, other_vars])
        else:
            X = cb_matrix
        
        # Choose model
        model_map = {
            'poisson': PoissonRegressor,
            'gamma': GammaRegressor
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(model_map.keys())}")
        
        # Fit model
        model_class = model_map[model_type]
        self.model = model_class(**kwargs)
        fitted_model = self.model.fit(X, y)
        
        self.model_type = 'sklearn'
        self.fitted_values = fitted_model
        
        # Extract coefficients (sklearn doesn't provide vcov directly)
        self.cb_coef = fitted_model.coef_[:cb_matrix.shape[1]]
        self.cb_vcov = None  # Not available in sklearn
        
        return fitted_model
    
    def _extract_cb_coefficients(self, fitted_model, n_cb_terms: int):
        """Extract cross-basis coefficients and variance-covariance matrix."""
        if self.model_type == 'statsmodels':
            # Skip intercept (first coefficient)
            all_coef = fitted_model.params[1:]  # Skip intercept
            all_vcov = fitted_model.cov_params()  # Could be DataFrame or numpy array
            
            # Extract cross-basis terms (first n_cb_terms after intercept)
            try:
                # Try pandas Series first
                self.cb_coef = all_coef[:n_cb_terms].values
            except AttributeError:
                # Fallback to numpy array
                self.cb_coef = all_coef[:n_cb_terms]
            
            # Handle vcov matrix - could be DataFrame or numpy array
            if hasattr(all_vcov, 'iloc'):
                # It's a DataFrame
                all_vcov_no_intercept = all_vcov.iloc[1:, 1:]  # Skip intercept row/col
                self.cb_vcov = all_vcov_no_intercept.iloc[:n_cb_terms, :n_cb_terms].values
            else:
                # It's a numpy array
                all_vcov_no_intercept = all_vcov[1:, 1:]  # Skip intercept row/col
                self.cb_vcov = all_vcov_no_intercept[:n_cb_terms, :n_cb_terms]
            
        elif self.model_type == 'sklearn':
            # sklearn models
            self.cb_coef = self.fitted_values.coef_[:n_cb_terms]
            self.cb_vcov = None  # Not available
    
    def get_crossbasis_coefficients(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract cross-basis coefficients and variance-covariance matrix.
        
        Returns
        -------
        coef : np.ndarray or None
            Cross-basis coefficients
        vcov : np.ndarray or None
            Cross-basis variance-covariance matrix
        """
        if hasattr(self, 'cb_coef') and hasattr(self, 'cb_vcov'):
            return self.cb_coef, self.cb_vcov
        elif self.cb_coef is not None and self.cb_vcov is not None:
            return self.cb_coef, self.cb_vcov
        else:
            return None, None
    
    def predict(self, newdata: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters
        ----------
        newdata : array-like, optional
            New data for prediction. If None, uses original data.
            
        Returns
        -------
        predictions : np.ndarray
            Model predictions
        """
        if self.fitted_values is None:
            raise ValueError("No model has been fitted yet")
        
        if newdata is None:
            if self.model_type == 'statsmodels':
                return self.fitted_values.fittedvalues
            else:
                return self.fitted_values.predict(np.array(self.crossbasis.basis))
        else:
            if self.model_type == 'statsmodels':
                # For statsmodels, need to add constant
                newdata = sm.add_constant(newdata)
                return self.fitted_values.predict(newdata)
            else:
                return self.fitted_values.predict(newdata)
    
    def summary(self) -> str:
        """Return a summary of the fitted model."""
        if self.fitted_values is None:
            return "No model fitted yet"
        
        if self.model_type == 'statsmodels':
            return str(self.fitted_values.summary())
        else:
            return f"Sklearn {type(self.fitted_values).__name__} model fitted"


def fit_dlnm_model(crossbasis: CrossBasis,
                   y: np.ndarray,
                   family: str = 'poisson',
                   other_vars: Optional[np.ndarray] = None,
                   backend: str = 'statsmodels',
                   **kwargs):
    """
    Convenience function to fit a DLNM model.
    
    Parameters
    ----------
    crossbasis : CrossBasis
        The cross-basis matrix
    y : array-like
        Response variable
    family : str, default='poisson'
        GLM family
    other_vars : array-like, optional
        Additional covariates
    backend : str, default='statsmodels'
        Modeling backend: 'statsmodels', 'sklearn', or 'rpy2'
    **kwargs
        Additional arguments passed to fitting method
        
    Returns
    -------
    dlnm_interface : DLNMGLMInterface or Rpy2GLMInterface
        Fitted DLNM interface object
    """
    if backend == 'rpy2':
        from .rpy2_glm import Rpy2GLMInterface
        interface = Rpy2GLMInterface(crossbasis)
        interface.fit_glm(y, family=family, other_vars=other_vars, **kwargs)
        return interface
    else:
        interface = DLNMGLMInterface(crossbasis)
        
        if backend == 'statsmodels':
            interface.fit_statsmodels(y, family=family, other_vars=other_vars, **kwargs)
        elif backend == 'sklearn':
            interface.fit_sklearn(y, model_type=family, other_vars=other_vars, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from: 'statsmodels', 'sklearn', 'rpy2'")
        
        return interface