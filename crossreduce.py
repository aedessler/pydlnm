"""
Cross-basis reduction functionality for PyDLNM

This module implements crossreduce(), a port of R's dlnm::crossreduce(): the reduction of the coefficients of a
cross-basis to the overall cumulative effect ("overall"), to the lag-response at a fixed exposure ("var") or to the
exposure-response at a fixed lag ("lag"), with the reduced basis, fit, standard errors and confidence intervals.
"""

import numpy as np
from scipy import stats
from typing import Union, Optional, Dict, Any, Tuple
import warnings

from basis import CrossBasis, OneBasis
from glm_integration import DLNMGLMInterface
from model_utils import validate_model_compatibility, getlink
from prediction import CrossPred, mkat, mkcen
from utils import asfloat, mklag, seqlag


class CrossReduce:
    """
    Cross-basis reduction results (R's crossreduce object).
    
    Attributes
    ----------
    coef, coefficients : np.ndarray
        Reduced coefficients (R: coefficients)
    vcov : np.ndarray
        Variance-covariance matrix of the reduced coefficients
    basis : np.ndarray
        Reduced (centred) basis at the prediction values
    type : str
        Reduction type: "overall", "var" or "lag"
    value : float or None
        Exposure value (type "var") or lag (type "lag") of the reduction
    predvar : np.ndarray or None
        Exposure values of the predictions (not for type "var")
    cen : float or None
        Centering value (resolved as R's mkcen)
    lag, bylag : lag range and step of the reduction
    fit, se : np.ndarray
        Predicted effects and standard errors (log scale)
    RRfit, RRlow, RRhigh : np.ndarray
        Relative risks and confidence limits (log and logit links); otherwise ``low`` and ``high`` on the linear scale
    ci_level : float
    model_link : str or None
    crossbasis : CrossBasis
        Original cross-basis object
    model_info : dict
        Information about the fitted model
    """
    
    def __init__(self, coef: np.ndarray, vcov: np.ndarray, 
                 crossbasis: CrossBasis, model_info: Dict[str, Any], 
                 cen: Optional[float] = None, **extra):
        self.coef = coef
        self.coefficients = coef
        self.vcov = vcov
        self.crossbasis = crossbasis
        self.model_info = model_info
        self.cen = cen
        self.type = "overall"
        self.value = None
        self.predvar = None
        self.basis = None
        self.lag = None
        self.bylag = 1.0
        self.fit = None
        self.se = None
        self.ci_level = 0.95
        self.model_class = model_info.get('type') if isinstance(model_info, dict) else None
        self.model_link = None
        for key, val in extra.items():
            setattr(self, key, val)
    
    def summary(self) -> str:
        """Text summary (R's summary.crossreduce): reduction type, reduced df, centering, lag and confidence level."""
        lines = [
            "CROSSREDUCE FUNCTION",
            f"Reduction type: {self.type}" + (f" (value = {self.value})" if self.value is not None else ""),
            f"Reduced DF: {len(self.coefficients)}",
            f"Centering value: {self.cen}",
            f"Lag period: [{self.lag[0]}, {self.lag[1]}] by {self.bylag}" if self.lag is not None else "Lag period: -",
            f"Confidence level: {self.ci_level:.0%}",
            f"Model link: {self.model_link}",
        ]
        text = "\n".join(lines)
        print(text)
        return text
    
    def __repr__(self):
        return f"CrossReduce(type={self.type!r}, coef={len(self.coef)} terms, cen={self.cen})"


def _check_coef_vcov(cb_coef, cb_vcov, ncol: int):
    """R: stop unless length(coef) == ncol(basis) == dim(vcov), without missing values."""
    cb_coef = asfloat(cb_coef).ravel()                 # a masked / NA entry is NaN: rejected below
    cb_vcov = np.atleast_2d(asfloat(cb_vcov))
    if (len(cb_coef) != ncol or cb_vcov.shape != (ncol, ncol) or np.isnan(cb_coef).any()
            or np.isnan(cb_vcov).any()):
        raise ValueError("coef/vcov do not consistent with basis matrix. See help(crossreduce)")
    return cb_coef, cb_vcov


def crossreduce(basis: Union[CrossBasis, DLNMGLMInterface],
                model: Any = None,
                type: str = "overall",
                value: Optional[float] = None,
                coef: Optional[np.ndarray] = None,
                vcov: Optional[np.ndarray] = None,
                model_link: Optional[str] = None,
                at: Optional[np.ndarray] = None,
                from_val: Optional[float] = None,
                to_val: Optional[float] = None,
                by: Optional[float] = None,
                lag: Optional[Union[int, list, tuple]] = None,
                bylag: float = 1.0,
                cen: Optional[float] = None,
                ci_level: float = 0.95,
                reduction_type: Optional[str] = None) -> CrossReduce:
    """
    Reduce a cross-basis (port of R's ``dlnm::crossreduce()``).
    
    The coefficients are reduced with ``newcoef = M @ coef`` and ``newvcov = M @ vcov @ M.T``, where M is built as in
    R, and the effects are predicted with the (centred) reduced basis.
    
    Parameters
    ----------
    basis : CrossBasis or DLNMGLMInterface
        Cross-basis object (or a fitted GLM interface)
    model : Any, optional
        Fitted model; the cross-basis coefficients are selected by name (``v1.l1`` ...)
    type : {"overall", "var", "lag"}, default "overall"
        Reduction type: the overall cumulative exposure-response ("overall"), the lag-response at the exposure
        ``value`` ("var"), or the exposure-response at the lag ``value`` ("lag")
    value : float, optional
        Exposure value (type "var") or lag (type "lag"); required for these types
    coef, vcov : array-like, optional
        Coefficients and covariance of the cross-basis when ``model`` is not given
    model_link : str, optional
        Link function ("log" and "logit" give relative risks) when ``model`` is not given
    at, from_val, to_val, by : optional
        Exposure values of the predictions (R's at/from/to/by; default ``pretty(range, n=50)``)
    lag : int or tuple, optional
        Lag sub-period of the reduction (default: the lag range of the cross-basis)
    bylag : float, default 1.0
        Lag step for type "var"
    cen : float, optional
        Centering value (resolved as R's mkcen: automatic mid-range for bs/ns/poly, False for none)
    ci_level : float, default 0.95
        Confidence level
    reduction_type : str, optional
        Deprecated alias of ``type``
        
    Returns
    -------
    CrossReduce
        Reduced coefficients, vcov, basis, fit, se and confidence intervals
    """
    
    if reduction_type is not None:
        warnings.warn("'reduction_type' is deprecated: use 'type'", DeprecationWarning, stacklevel=2)
        type = reduction_type
    if not isinstance(type, str) or not type:
        raise ValueError("'type' should be one of 'overall', 'var', 'lag'")
    matches = [t for t in ("overall", "var", "lag") if t == type] or \
              [t for t in ("overall", "var", "lag") if t.startswith(type)]
    if len(matches) != 1:
        raise ValueError("'type' should be one of 'overall', 'var', 'lag'")
    type = matches[0]
    
    # Extract cross-basis and model information
    if isinstance(basis, DLNMGLMInterface) or callable(getattr(basis, 'get_crossbasis_coefficients', None)):
        # A fitted GLM interface (DLNMGLMInterface, Rpy2GLMInterface, ImprovedGLMInterface): the cross-basis block
        # that the interface selected by name from its R model
        cb_obj = basis.crossbasis
        cb_coef, cb_vcov = basis.get_crossbasis_coefficients()
        if cb_coef is None or cb_vcov is None:
            raise ValueError("the GLM interface has no fitted model: call fit_glm() or fit_dlnm_model() first")
        cb_coef = asfloat(cb_coef)
        cb_vcov = asfloat(cb_vcov)
        if np.isnan(cb_coef).any() or np.isnan(cb_vcov).any():
            raise ValueError("coef/vcov do not consistent with basis matrix. See help(crossreduce)")
        
        model_info = {'type': 'dlnm_glm', 'family': getattr(basis, 'family', None) or 'unknown'}
        
    elif isinstance(basis, CrossBasis):
        cb_obj = basis
        cb_vcov = None
        
        if model is not None:
            # Cross-basis coefficients selected by name, link from the model
            info = validate_model_compatibility(model, cb_obj.shape[1], "CrossBasis", kind="cb")
            cb_coef, cb_vcov = info['coef'], info['vcov']
            model_info = {'type': info['class']}
            model_link = getlink(model, info['class'], model_link)
            
        elif coef is not None and vcov is not None:
            # Direct coefficient and variance-covariance input
            cb_coef = asfloat(coef)
            cb_vcov = asfloat(vcov)
            model_info = {'type': 'direct'}
            
        else:
            raise ValueError("Either 'model' or both 'coef' and 'vcov' must be provided")
    
    else:
        raise TypeError("crossbasis must be CrossBasis or DLNMGLMInterface")
    
    if not isinstance(basis, CrossBasis) and model_link is None:
        model_link = getlink(basis)                           # a fitted GLM interface: the link of its family
    cb_coef, cb_vcov = _check_coef_vcov(cb_coef, cb_vcov, cb_obj.shape[1])
    
    # Checks on type, value, lag and ci_level (R)
    if type != "overall":
        if value is None:
            raise ValueError("'value' must be provided for type 'var' or 'lag'")
        if np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError("'value' must be a numeric scalar")
        if type == "lag" and not (cb_obj.lag[0] <= value <= cb_obj.lag[1]):
            raise ValueError("'value' of lag-specific effects must be within the lag range")
    else:
        value = None
    lag_range = np.asarray(cb_obj.lag) if lag is None else mklag(lag)
    if cb_obj.arglag.get('fun') == 'integer' and not np.array_equal(lag_range, cb_obj.lag):
        raise ValueError("prediction for lag sub-period not allowed for type 'integer'")
    if not (0 < ci_level < 1):
        raise ValueError("'ci_level' must be numeric and between 0 and 1")
    
    # Prediction values and centering (R: mkat, mkcen)
    if at is not None and np.ndim(at) == 2:
        raise ValueError("argument 'at' must be a vector")
    at_values = mkat(at, from_val, to_val, by, cb_obj.range, lag_range, bylag)
    cen = mkcen(cen, cb_obj, cb_obj.range)
    
    # Reduction: transformation matrix M and reduced basis (R: tensor order compatible with crossbasis)
    argvar = {k: v for k, v in cb_obj.argvar.items() if k != 'cen'}
    arglag = {k: v for k, v in cb_obj.arglag.items() if k != 'cen'}
    n_basis = cb_obj.shape[1]
    
    def lag_basis_at(values) -> np.ndarray:
        values = np.atleast_1d(asfloat(values))
        return OneBasis(values, **arglag).basis        # 'integer' rebuilds its indicator rows from the fitted values
    
    def var_basis_at(values) -> np.ndarray:
        basis_var = OneBasis(np.atleast_1d(asfloat(values)), **argvar).basis
        if cen is not None:
            basis_var = basis_var - OneBasis([cen], **argvar).basis
        return basis_var
    
    if type == "overall":
        lag_basis = lag_basis_at(seqlag(lag_range))
        M = np.kron(np.eye(n_basis // lag_basis.shape[1]), np.ones((1, len(lag_basis))) @ lag_basis)
        new_basis = var_basis_at(at_values)
    elif type == "lag":
        lag_basis = lag_basis_at([value])
        M = np.kron(np.eye(n_basis // lag_basis.shape[1]), lag_basis)
        new_basis = var_basis_at(at_values)
    else:
        var_basis = var_basis_at([value])
        M = np.kron(var_basis, np.eye(n_basis // var_basis.shape[1]))
        new_basis = lag_basis_at(seqlag(lag_range, bylag))
    
    reduced_coef = M @ cb_coef
    reduced_vcov = M @ cb_vcov @ M.T
    
    # Prediction: effects, standard errors and confidence intervals
    fit = new_basis @ reduced_coef
    se = np.sqrt(np.maximum(0.0, np.sum((new_basis @ reduced_vcov) * new_basis, axis=1)))
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    
    result = CrossReduce(coef=reduced_coef, vcov=reduced_vcov, crossbasis=cb_obj, model_info=model_info, cen=cen)
    result.basis = new_basis
    result.type = type
    result.value = value
    result.predvar = None if type == "var" else at_values
    result.lag = lag_range
    result.bylag = bylag
    result.fit = fit
    result.se = se
    if model_link in ("log", "logit"):
        result.RRfit = np.exp(fit)
        result.RRlow = np.exp(fit - z * se)
        result.RRhigh = np.exp(fit + z * se)
    else:
        result.low = fit - z * se
        result.high = fit + z * se
    result.ci_level = ci_level
    result.model_link = model_link
    return result


# Convenience functions to match R interface
def coef(obj) -> np.ndarray:
    """Coefficients of a CrossReduce (R coef.crossreduce) or a CrossPred (R coef.crosspred)."""
    if isinstance(obj, CrossPred):
        return obj.coefficients
    return obj.coef


def vcov(obj) -> np.ndarray:
    """Variance-covariance matrix of a CrossReduce (R vcov.crossreduce) or a CrossPred (R vcov.crosspred)."""
    return obj.vcov
