"""
Model integration utilities for PyDLNM

This module provides utilities for extracting coefficients, variance-covariance matrices,
and link functions from various statistical modeling frameworks.
"""

import re
import numpy as np
from typing import Any, Optional, Union, Dict, List, Sequence
import warnings


_LINK_CLASS_NAMES = {'log': 'log', 'logit': 'logit', 'identity': 'identity', 'probit': 'probit',
                     'cloglog': 'cloglog', 'inversepower': 'inverse', 'inversesquared': 'inverse_squared',
                     'cauchy': 'cauchy', 'sqrt': 'sqrt'}


def _link_name(link: Any) -> Optional[str]:
    """Name of a link given as a string, an object with .name or a statsmodels link class (Log, Logit, ...)."""
    if isinstance(link, str):
        return link.lower()
    name = getattr(link, 'name', None)
    if isinstance(name, str):
        return name.lower()
    class_name = type(link).__name__.lower()
    return _LINK_CLASS_NAMES.get(class_name, class_name if link is not None else None)


_NAME_PATTERNS = {'cb': re.compile(r'v[0-9]{1,2}\.l[0-9]{1,2}$'), 'one': re.compile(r'b[0-9]{1,2}$')}


def coefficient_names(model: Any) -> Optional[List[str]]:
    """Names of a fitted model's coefficients (pandas index of params, or the design column names), else None."""
    params = getattr(model, 'params', None)
    index = getattr(params, 'index', None)
    if index is not None:
        return [str(n) for n in index]
    names = getattr(getattr(model, 'model', None), 'exog_names', None)
    return [str(n) for n in names] if names is not None else None


def basis_block(model: Any, n_coef: int, basis_ncol: int, kind: str = 'cb',
                basis_name: str = 'basis') -> np.ndarray:
    """
    Indices of the cross-basis (or one-basis) block in the coefficient vector of a fitted model.

    R (crosspred, attrdl) selects the block by NAME: the coefficients whose names match ``v<i>.l<j>`` (``b<i>`` for
    a one-basis). The same is done here when the model carries coefficient names. Without informative names the
    block is identified only when it is the whole coefficient vector; otherwise a ValueError is raised instead of
    guessing that the block comes first.
    """
    all_idx = np.arange(n_coef)
    names = coefficient_names(model)
    pattern = _NAME_PATTERNS.get(kind, _NAME_PATTERNS['cb'])
    if names is not None and len(names) == n_coef:
        idx = np.array([i for i, nm in enumerate(names) if pattern.search(nm)], dtype=int)
        if len(idx) == basis_ncol:
            return idx
        if len(idx) > basis_ncol:
            raise ValueError(
                f"{len(idx)} coefficients look like cross-basis terms but {basis_name} has {basis_ncol} columns "
                f"(several bases in the model?): pass coef= and vcov= for the block of {basis_name}")
    if n_coef == basis_ncol:
        return all_idx
    raise ValueError(
        f"cannot identify the {basis_ncol} coefficients of {basis_name} among the {n_coef} model coefficients "
        f"(no coefficient names like 'v1.l1'): pass coef= and vcov= for the block of {basis_name}")


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
    
    # Handle PyDLNM's rpy2-based GLM interfaces
    if model_class == 'DLNMGLMInterface':
        coef, _ = model.get_crossbasis_coefficients()
        if coef is not None:
            return np.asarray(coef)
        else:
            raise AttributeError("No coefficients available from DLNMGLMInterface")
    
    if model_class == 'Rpy2GLMInterface':
        if hasattr(model, 'cb_coef') and model.cb_coef is not None:
            return np.asarray(model.cb_coef)
        else:
            raise AttributeError("No cross-basis coefficients available from Rpy2GLMInterface")

    if model_class == 'ImprovedGLMInterface':
        if hasattr(model, 'cb_coef') and model.cb_coef is not None:
            return np.asarray(model.cb_coef)
        else:
            raise AttributeError("No cross-basis coefficients available from ImprovedGLMInterface")

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
    
    # Handle PyDLNM's rpy2-based GLM interfaces
    if model_class == 'DLNMGLMInterface':
        _, vcov = model.get_crossbasis_coefficients()
        if vcov is not None:
            return np.asarray(vcov)
        else:
            raise AttributeError("No variance-covariance matrix available from DLNMGLMInterface")
    
    if model_class == 'Rpy2GLMInterface':
        if hasattr(model, 'cb_vcov') and model.cb_vcov is not None:
            return np.asarray(model.cb_vcov)
        else:
            raise AttributeError("No cross-basis vcov matrix available from Rpy2GLMInterface")

    if model_class == 'ImprovedGLMInterface':
        if hasattr(model, 'cb_vcov') and model.cb_vcov is not None:
            return np.asarray(model.cb_vcov)
        else:
            raise AttributeError("No cross-basis vcov matrix available from ImprovedGLMInterface")

    # Try common vcov attributes and methods
    vcov_attrs = ['cov_params', 'vcov', 'cov_', 'covariance_matrix']
    vcov_methods = ['cov_params', 'vcov']
    
    # Try attributes first (but skip if they are callable/methods)
    for attr in vcov_attrs:
        if hasattr(model, attr):
            vcov = getattr(model, attr)
            if vcov is not None and not callable(vcov):
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
    
    # Handle PyDLNM's rpy2-based GLM interfaces
    if model_class == 'DLNMGLMInterface':
        # For quasi-Poisson family, the link is log
        return 'log'

    if model_class == 'Rpy2GLMInterface':
        # For quasi-Poisson family, the link is log
        return 'log'

    if model_class == 'ImprovedGLMInterface':
        # For quasi-Poisson family, the link is log
        return 'log'
    
    # Family attribute (statsmodels GLM results and models): family.link is a link object; R: model$family$link
    family = getattr(model, 'family', None)
    if family is None and hasattr(model, 'model'):
        family = getattr(model.model, 'family', None)
    if family is not None and hasattr(family, 'link'):
        name = _link_name(family.link)
        if name is not None:
            return name
    
    # statsmodels discrete-choice models (results wrapper -> .model): the link is implied by the model class
    inner_class = type(getattr(model, 'model', None)).__name__
    discrete_links = {'Poisson': 'log', 'GeneralizedPoisson': 'log', 'NegativeBinomial': 'log',
                      'NegativeBinomialP': 'log', 'Logit': 'logit', 'Probit': 'probit', 'MNLogit': 'logit'}
    if inner_class in discrete_links:
        return discrete_links[inner_class]
    
    # Try direct link attribute
    if hasattr(model, 'link'):
        link = model.link
        if hasattr(link, 'name'):
            return link.name
        elif isinstance(link, str):
            return link
    
    # Model type-specific defaults (a GLM without a family is left undetermined rather than guessed)
    model_defaults = {
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
                                basis_name: str = "basis",
                                kind: str = "cb") -> Dict[str, Any]:
    """
    Validate that a model is compatible with a basis matrix and extract key information.
    
    The coefficients of the basis are selected from the model by name, as R does (see ``basis_block``).
    
    Parameters
    ----------
    model : fitted model object
        Fitted statistical model
    basis_ncol : int
        Number of columns in the basis matrix
    basis_name : str, default="basis"
        Name of the basis for error messages
    kind : {"cb", "one"}, default="cb"
        Basis type, which decides the coefficient name pattern (``v1.l1`` or ``b1``)
        
    Returns
    -------
    dict
        Dictionary containing model information:
        - 'coef': coefficients of the basis
        - 'vcov': their variance-covariance matrix
        - 'link': link function
        - 'class': model class name
        
    Raises
    ------
    ValueError
        If model is not compatible with basis matrix
    """
    model_class = type(model).__name__
    
    try:
        coef = np.asarray(getcoef(model, model_class), dtype=float)
        vcov = np.asarray(getvcov(model, model_class), dtype=float)
        link = getlink(model, model_class)
        
        if len(coef) < basis_ncol:
            raise ValueError(
                f"Model has {len(coef)} coefficients but {basis_name} has {basis_ncol} columns. "
                f"Model may not include all {basis_name} terms."
            )
        if vcov.ndim != 2 or vcov.shape[0] != len(coef) or vcov.shape[1] != len(coef):
            raise ValueError(
                f"Variance-covariance matrix has shape {vcov.shape} but the model has {len(coef)} coefficients."
            )
        
        idx = basis_block(model, len(coef), basis_ncol, kind, basis_name)
        
        return {
            'coef': coef[idx],
            'vcov': vcov[np.ix_(idx, idx)],
            'link': link,
            'class': model_class
        }
        
    except Exception as e:
        raise ValueError(
            f"Model of type {model_class} is not compatible with {basis_name}: {str(e)}"
        ) from e
