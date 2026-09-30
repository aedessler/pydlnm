"""
GLM integration for PyDLNM

This module provides interfaces between CrossBasis matrices and R-based statistical models
via rpy2, ensuring exact compatibility with R DLNM package.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple, List
import warnings

from basis import CrossBasis
from rbridge import r_session


class DLNMGLMInterface:
    """
    Interface between CrossBasis and R-based statistical models via rpy2.
    
    This class provides methods to fit GLMs with cross-basis matrices using
    R's GLM implementation through rpy2, ensuring exact compatibility with
    the R DLNM package.
    
    Parameters
    ----------
    crossbasis : CrossBasis
        The cross-basis matrix object
    """
    
    def __init__(self, crossbasis: CrossBasis):
        self.crossbasis = crossbasis
        self.rpy2_interface = None
        self._check_rpy2()
    
    def _check_rpy2(self):
        """Check if rpy2 is available and initialize interface"""
        try:
            from rpy2_glm import Rpy2GLMInterface
            self.rpy2_interface = Rpy2GLMInterface(self.crossbasis)
        except ImportError:
            raise ImportError(
                "rpy2 is required for GLM functionality in PyDLNM. "
                "Please install rpy2 with: pip install rpy2"
            )
    
    @property
    def family(self) -> Optional[str]:
        """Family of the fitted R model (None before a fit)"""
        return None if self.rpy2_interface is None else self.rpy2_interface.family
    
    @property
    def link(self) -> Optional[str]:
        """Link function of the fitted R model (None before a fit)"""
        return None if self.rpy2_interface is None else self.rpy2_interface.link
    
    def fit_glm(self, 
                y: np.ndarray,
                family: str = 'quasipoisson',
                other_vars: Optional[np.ndarray] = None,
                formula_vars: Optional[List[str]] = None,
                **kwargs) -> Any:
        """
        Fit GLM using R's glm() function via rpy2.
        
        Parameters
        ----------
        y : array-like
            Response variable (e.g., mortality counts)
        family : str, default='quasipoisson'
            GLM family: 'poisson', 'quasipoisson', 'gaussian', 'gamma' (R's ``Gamma``), 'binomial', ...
        other_vars : array-like, optional
            Additional covariates (e.g., seasonality, day of week)
        formula_vars : list of str, optional
            Names for other variables (for easier interpretation)
        **kwargs
            Arguments passed to R's glm(): ``weights``, ``offset``, ``subset`` and ``control`` (see
            ``Rpy2GLMInterface.fit_glm``); any other keyword raises TypeError
            
        Returns
        -------
        fitted_model : R object
            Fitted R GLM model object
        """
        return self.rpy2_interface.fit_glm(
            y=y, 
            family=family, 
            other_vars=other_vars, 
            formula_vars=formula_vars, 
            **kwargs
        )
    
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
        if self.rpy2_interface is None:
            return None, None
        return self.rpy2_interface.cb_coef, self.rpy2_interface.cb_vcov
    
    def predict(self, newdata: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fitted values of the R model (R: ``fitted()``).
        
        Parameters
        ----------
        newdata : array-like, optional
            New data for prediction (not implemented). If None, the fitted values of the original data are returned.
            
        Returns
        -------
        predictions : np.ndarray
            Fitted values on the response scale, one per input row: NaN for the rows that R excluded
            (``na.action = na.exclude`` pads them, as R's ``fitted()`` does)
        """
        if self.rpy2_interface is None or self.rpy2_interface.r_model is None:
            raise ValueError("No model has been fitted yet")
        
        if newdata is None:
            return self.rpy2_interface.fitted()
        else:
            raise NotImplementedError("Prediction with new data not yet implemented")
    
    def summary(self) -> str:
        """Return a summary of the fitted R model."""
        if self.rpy2_interface is None or self.rpy2_interface.r_model is None:
            return "No model fitted yet"
        
        with r_session():                        # printing an R object calls R: serialised, explicit converter
            return str(self.rpy2_interface.get_model_summary())
    
    def crossreduce(self, cen: Optional[float] = None, type: str = "overall", **kwargs):
        """
        Reduce the cross-basis of the fitted model (PyDLNM's port of R's ``crossreduce()`` applied to the
        cross-basis coefficients and variance-covariance matrix of the model).
        
        Parameters
        ----------
        cen : float, optional
            Centering value for reduction
        type : str, default="overall"
            Type of reduction ("overall", "var", "lag"); an unknown type raises ValueError
        **kwargs
            Further arguments of ``crossreduce.crossreduce()``
            
        Returns
        -------
        CrossReduce
            Reduced coefficients (``coef``) and variance-covariance matrix (``vcov``)
        """
        if self.rpy2_interface is None:
            raise ValueError("No rpy2 interface available")
        
        return self.rpy2_interface.crossreduce(cen=cen, type=type, **kwargs)


def fit_dlnm_model(crossbasis: CrossBasis,
                   y: np.ndarray,
                   family: str = 'quasipoisson',
                   other_vars: Optional[np.ndarray] = None,
                   **kwargs):
    """
    Convenience function to fit a DLNM model using R's GLM via rpy2.
    
    Parameters
    ----------
    crossbasis : CrossBasis
        The cross-basis matrix
    y : array-like
        Response variable
    family : str, default='quasipoisson'
        GLM family
    other_vars : array-like, optional
        Additional covariates
    **kwargs
        Additional arguments passed to ``DLNMGLMInterface.fit_glm`` (``formula_vars``, ``weights``, ``offset``,
        ``subset``, ``control``)
        
    Returns
    -------
    dlnm_interface : DLNMGLMInterface
        Fitted DLNM interface object
    """
    interface = DLNMGLMInterface(crossbasis)
    interface.fit_glm(y, family=family, other_vars=other_vars, **kwargs)
    return interface