"""
Cross-basis reduction functionality for PyDLNM

This module implements the crossreduce() function that reduces cross-basis matrices
to overall cumulative effects, matching R's dlnm::crossreduce() behavior exactly.
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Tuple
import warnings

from .basis import CrossBasis, OneBasis
from .glm_integration import DLNMGLMInterface


class CrossReduce:
    """
    Cross-basis reduction results.
    
    This class holds the results of reducing a cross-basis to overall
    cumulative effects, identical to R's crossreduce object.
    
    Attributes
    ----------
    coef : np.ndarray
        Reduced coefficients (overall cumulative effects)
    vcov : np.ndarray
        Variance-covariance matrix of reduced coefficients
    crossbasis : CrossBasis
        Original cross-basis object
    model_info : dict
        Information about the fitted model
    cen : float
        Centering value used for reduction
    """
    
    def __init__(self, coef: np.ndarray, vcov: np.ndarray, 
                 crossbasis: CrossBasis, model_info: Dict[str, Any], 
                 cen: Optional[float] = None):
        self.coef = coef
        self.vcov = vcov
        self.crossbasis = crossbasis
        self.model_info = model_info
        self.cen = cen
        
    def __repr__(self):
        return f"CrossReduce(coef={len(self.coef)} terms, cen={self.cen})"


def crossreduce(basis: Union[CrossBasis, DLNMGLMInterface],
                model: Any = None,
                reduction_type: str = "overall",
                coef: Optional[np.ndarray] = None,
                vcov: Optional[np.ndarray] = None,
                cen: Optional[float] = None) -> CrossReduce:
    """
    Reduce cross-basis to overall cumulative effects.
    
    This function implements the exact mathematical transformation used in
    R's dlnm::crossreduce() function without any arbitrary scaling factors.
    
    The transformation follows: newcoef = M @ coef, newvcov = M @ vcov @ M.T
    where M is the reduction matrix constructed exactly as in R.
    
    Parameters
    ----------
    basis : CrossBasis or DLNMGLMInterface
        Cross-basis object or fitted GLM interface
    model : Any, optional
        Fitted model object (statsmodels GLM, sklearn, etc.)
    type : str, default "overall"
        Type of reduction ("overall", "var", "lag")
    coef : np.ndarray, optional
        Coefficients if model not provided
    vcov : np.ndarray, optional
        Variance-covariance matrix if model not provided
    cen : float, optional
        Centering value for the reduction
        
    Returns
    -------
    CrossReduce
        Object containing reduced coefficients and variance-covariance matrix
    """
    
    # Input validation
    if reduction_type != "overall":
        raise ValueError(f"Reduction type '{reduction_type}' not implemented yet. Only 'overall' supported.")
    
    # Extract cross-basis and model information
    if isinstance(basis, DLNMGLMInterface):
        cb_obj = basis.crossbasis
        
        # Extract coefficients and variance-covariance matrix
        n_cb_terms = cb_obj.shape[1]
        fitted_model = basis.fitted_values  # DLNMGLMInterface stores fitted model in fitted_values
        cb_coef = fitted_model.params[1:n_cb_terms+1]  # Skip intercept (index 0)
        cb_vcov = fitted_model.cov_params()[1:n_cb_terms+1, 1:n_cb_terms+1]  # Skip intercept
        
        # Handle numpy arrays vs pandas series
        if hasattr(cb_coef, 'values'):
            cb_coef = cb_coef.values
        if hasattr(cb_vcov, 'values'):
            cb_vcov = cb_vcov.values
            
        model_info = {'type': 'dlnm_glm', 'family': getattr(basis, 'family', 'unknown')}
        
    elif isinstance(basis, CrossBasis):
        cb_obj = basis
        
        if model is not None:
            # Extract from statsmodels or sklearn model
            if hasattr(model, 'params') and hasattr(model, 'cov_params'):
                # Statsmodels GLM or custom results object
                n_cb_terms = cb_obj.shape[1]
                cb_coef = model.params[:n_cb_terms]
                
                # Handle both callable cov_params() and attribute cov_params
                if callable(model.cov_params):
                    cb_vcov = model.cov_params()[:n_cb_terms, :n_cb_terms]
                else:
                    cb_vcov = model.cov_params[:n_cb_terms, :n_cb_terms]
                
                if hasattr(cb_coef, 'values'):
                    cb_coef = cb_coef.values
                if hasattr(cb_vcov, 'values'):
                    cb_vcov = cb_vcov.values
                    
                model_info = {'type': 'statsmodels'}
                
            elif hasattr(model, 'coef_'):
                # Sklearn model - limited support without vcov
                cb_coef = model.coef_[:cb_obj.shape[1]]
                cb_vcov = None
                model_info = {'type': 'sklearn', 'fitted_values': model}
                
            elif hasattr(model, 'params') and isinstance(model.params, np.ndarray):
                # Model with params array and optional cov_params
                cb_coef = model.params[:cb_obj.shape[1]]
                
                if hasattr(model, 'cov_params'):
                    # Use provided covariance matrix
                    cb_vcov = model.cov_params[:cb_obj.shape[1], :cb_obj.shape[1]]
                    model_info = {'type': 'proper_glm'}
                else:
                    # Create identity vcov for debugging
                    cb_vcov = np.eye(len(cb_coef)) * 1e-6  # Small variance for numerical stability
                    model_info = {'type': 'dummy'}
                
            else:
                raise ValueError("Unsupported model type")
                
        elif coef is not None and vcov is not None:
            # Direct coefficient and variance-covariance input
            cb_coef = np.asarray(coef)
            cb_vcov = np.asarray(vcov)
            model_info = {'type': 'direct'}
            
        else:
            raise ValueError("Either 'model' or both 'coef' and 'vcov' must be provided")
            
        if cb_vcov is None:
            raise ValueError("Cannot perform reduction without variance-covariance matrix")
    
    else:
        raise TypeError("crossbasis must be CrossBasis or DLNMGLMInterface")
    
    # Perform the reduction using R's exact mathematical approach
    reduced_coef, reduced_vcov = _reduce_overall_exact(cb_obj, cb_coef, cb_vcov, cen)
    
    return CrossReduce(
        coef=reduced_coef,
        vcov=reduced_vcov,
        crossbasis=cb_obj,
        model_info=model_info,
        cen=cen
    )


def _reduce_overall_exact(crossbasis: CrossBasis, 
                         cb_coef: np.ndarray, 
                         cb_vcov: np.ndarray,
                         cen: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform overall cumulative reduction using R's exact mathematical approach.
    
    FIXED VERSION that exactly matches R's dlnm::crossreduce():
    M <- diag(ncol(basis)/ncol(lagbasis)) %x% (t(rep(1,diff(lag)+1)) %*% lagbasis)
    newcoef <- as.vector(M%*%coef)
    newvcov <- M%*%vcov%*%t(M)
    
    Key fixes:
    1. Use correct lag period count (diff(lag)+1)
    2. Create lag basis matrix exactly as R does
    3. Apply transformation matrix correctly
    """
    
    n_var_basis = crossbasis.df[0]  # Number of variable basis functions
    n_lag_basis = crossbasis.df[1]  # Number of lag basis functions
    
    # Create lag sequence exactly as R does: seqlag(lag)
    lag_range = crossbasis.lag
    from .utils import seqlag
    lag_sequence = seqlag(lag_range)
    
    # Build lag basis using the same arguments as the original cross-basis
    # Remove 'cen' parameter if present (not valid for OneBasis)
    arglag = crossbasis.arglag.copy()
    if 'cen' in arglag:
        del arglag['cen']
    
    lag_basis_obj = OneBasis(lag_sequence, **arglag)
    lagbasis = lag_basis_obj.basis
    
    # R's transformation matrix construction
    # t(rep(1,diff(lag)+1)) %*% lagbasis
    
    # diff(lag)+1 is the number of lag periods (lag[1] - lag[0] + 1)
    n_lag_periods = int(lag_range[1] - lag_range[0] + 1)
    ones_vector = np.ones((1, n_lag_periods))
    
    # Matrix multiplication: ones_vector @ lagbasis
    # This sums the lag basis over all lag periods
    lag_sum = ones_vector @ lagbasis  # Shape: (1, n_lag_basis)
    
    # Kronecker product with identity matrix
    # diag(ncol(basis)/ncol(lagbasis)) is identity matrix of size n_var_basis
    I_var = np.eye(n_var_basis)
    
    # Create transformation matrix M exactly as in R
    # M <- diag(n_var_basis) %x% lag_sum
    M = np.kron(I_var, lag_sum)  # Shape: (n_var_basis, n_var_basis * n_lag_basis)
    
    # Apply the exact transformation from R
    # newcoef <- as.vector(M%*%coef)
    reduced_coef = M @ cb_coef
    
    # newvcov <- M%*%vcov%*%t(M)
    reduced_vcov = M @ cb_vcov @ M.T
    
    return reduced_coef, reduced_vcov


# Convenience functions to match R interface
def coef(crossreduce_obj: CrossReduce) -> np.ndarray:
    """Extract coefficients from CrossReduce object."""
    return crossreduce_obj.coef


def vcov(crossreduce_obj: CrossReduce) -> np.ndarray:
    """Extract variance-covariance matrix from CrossReduce object."""
    return crossreduce_obj.vcov