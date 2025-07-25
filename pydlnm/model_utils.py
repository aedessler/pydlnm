"""
Model integration utilities for PyDLNM

This module provides utilities for extracting coefficients, variance-covariance matrices,
and link functions from various statistical modeling frameworks.
"""

import numpy as np
from typing import Any, Optional, Union, Dict
import warnings


def getcoef(model: Any, model_class: Optional[str] = None) -> np.ndarray:
    """
    Extract coefficients from various model types.
    
    Parameters
    ----------
    model : fitted model object
        Fitted statistical model
    model_class : str, optional
        Model class name for explicit handling
        
    Returns
    -------
    np.ndarray
        Model coefficients
        
    Raises
    ------
    AttributeError
        If coefficients cannot be extracted from the model
    """
    # Determine model class if not provided
    if model_class is None:
        model_class = type(model).__name__
    
    # Try common coefficient attributes
    coef_attrs = ['params', 'coef_', 'coefficients', 'coef', 'beta']
    
    for attr in coef_attrs:
        if hasattr(model, attr):
            coef = getattr(model, attr)
            if coef is not None:
                return np.asarray(coef)
    
    # Specific handling for different model types
    if hasattr(model, 'get_params'):
        # Some sklearn-style models
        try:
            coef = model.get_params()
            if isinstance(coef, dict) and 'coef_' in coef:
                return np.asarray(coef['coef_'])
        except:
            pass
    
    # If all else fails, raise an informative error
    raise AttributeError(
        f"Cannot extract coefficients from model of type {model_class}. "
        f"Supported attributes are: {coef_attrs}"
    )


def getvcov(model: Any, model_class: Optional[str] = None) -> np.ndarray:
    """
    Extract variance-covariance matrix from various model types.
    
    Parameters
    ----------
    model : fitted model object
        Fitted statistical model
    model_class : str, optional
        Model class name for explicit handling
        
    Returns
    -------
    np.ndarray
        Variance-covariance matrix
        
    Raises
    ------
    AttributeError
        If variance-covariance matrix cannot be extracted from the model
    """
    # Determine model class if not provided
    if model_class is None:
        model_class = type(model).__name__
    
    # Try common vcov attributes and methods
    vcov_attrs = ['cov_params', 'vcov', 'cov_', 'covariance_matrix']
    vcov_methods = ['cov_params', 'vcov']
    
    # Try attributes first
    for attr in vcov_attrs:
        if hasattr(model, attr):
            vcov = getattr(model, attr)
            if vcov is not None:
                return np.asarray(vcov)
    
    # Try methods
    for method in vcov_methods:
        if hasattr(model, method):
            try:
                vcov = getattr(model, method)()
                if vcov is not None:
                    return np.asarray(vcov)
            except:
                continue
    
    # Special handling for specific model types
    if hasattr(model, 'summary'):
        try:
            summary = model.summary()
            if hasattr(summary, 'cov_params'):
                return np.asarray(summary.cov_params)
        except:
            pass
    
    # If all else fails, try to compute from standard errors
    if hasattr(model, 'bse') or hasattr(model, 'std_err'):
        se = getattr(model, 'bse', None) or getattr(model, 'std_err', None)
        if se is not None:
            warnings.warn(
                "Full variance-covariance matrix not available. "
                "Creating diagonal matrix from standard errors."
            )
            se = np.asarray(se)
            return np.diag(se ** 2)
    
    raise AttributeError(
        f"Cannot extract variance-covariance matrix from model of type {model_class}. "
        f"Tried attributes: {vcov_attrs} and methods: {vcov_methods}"
    )


def getlink(model: Any, 
            model_class: Optional[str] = None, 
            model_link: Optional[str] = None) -> Optional[str]:
    """
    Identify or extract link functions from fitted models.
    
    Parameters
    ----------
    model : fitted model object
        Fitted statistical model
    model_class : str, optional
        Model class name for explicit handling
    model_link : str, optional
        User-specified link function (takes precedence)
        
    Returns
    -------
    str or None
        Link function name ('identity', 'log', 'logit', etc.) or None if not determined
    """
    # Return user-specified link if provided
    if model_link is not None:
        return model_link
    
    # Determine model class if not provided
    if model_class is None:
        model_class = type(model).__name__
    
    # Try to extract from family attribute (common in GLMs)
    if hasattr(model, 'family'):
        family = model.family
        if hasattr(family, 'link'):
            link = family.link
            if hasattr(link, 'name'):
                return link.name
            elif isinstance(link, str):
                return link
    
    # Try direct link attribute
    if hasattr(model, 'link'):
        link = model.link
        if hasattr(link, 'name'):
            return link.name
        elif isinstance(link, str):
            return link
    
    # Model type-specific defaults
    model_defaults = {
        'GLM': 'identity',
        'Poisson': 'log',
        'Logit': 'logit',
        'LogisticRegression': 'logit',
        'PoissonRegressor': 'log',
        'LinearRegression': 'identity',
        'OLS': 'identity',
    }
    
    for pattern, default_link in model_defaults.items():
        if pattern.lower() in model_class.lower():
            return default_link
    
    # If we can't determine the link, return None
    return None


def validate_model_compatibility(model: Any, 
                                basis_ncol: int,
                                basis_name: str = "basis") -> Dict[str, Any]:
    """
    Validate that a model is compatible with a basis matrix and extract key information.
    
    Parameters
    ----------
    model : fitted model object
        Fitted statistical model
    basis_ncol : int
        Number of columns in the basis matrix
    basis_name : str, default="basis"
        Name of the basis for error messages
        
    Returns
    -------
    dict
        Dictionary containing model information:
        - 'coef': model coefficients
        - 'vcov': variance-covariance matrix
        - 'link': link function
        - 'class': model class name
        
    Raises
    ------
    ValueError
        If model is not compatible with basis matrix
    """
    model_class = type(model).__name__
    
    try:
        coef = getcoef(model, model_class)
        vcov = getvcov(model, model_class)
        link = getlink(model, model_class)
        
        # Check dimensions
        if len(coef) < basis_ncol:
            raise ValueError(
                f"Model has {len(coef)} coefficients but {basis_name} has {basis_ncol} columns. "
                f"Model may not include all {basis_name} terms."
            )
        
        if vcov.shape[0] < basis_ncol or vcov.shape[1] < basis_ncol:
            raise ValueError(
                f"Variance-covariance matrix has shape {vcov.shape} but needs at least "
                f"({basis_ncol}, {basis_ncol}) for {basis_name}."
            )
        
        return {
            'coef': coef,
            'vcov': vcov,
            'link': link,
            'class': model_class
        }
        
    except Exception as e:
        raise ValueError(
            f"Model of type {model_class} is not compatible with {basis_name}: {str(e)}"
        ) from e