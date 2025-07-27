"""
Prediction classes for PyDLNM

This module contains classes for making predictions from distributed lag models,
including lag-specific, overall cumulative, and predictor-specific predictions.
"""

import numpy as np
from scipy import stats
from typing import Union, Optional, List, Dict, Any, Tuple
import warnings

from .basis import OneBasis, CrossBasis
from .model_utils import validate_model_compatibility
from .utils import mklag, seqlag


class CrossPred:
    """
    Cross-prediction class for distributed lag models.
    
    This class generates predictions from distributed lag models, including
    lag-specific effects, overall cumulative effects, and confidence intervals.
    
    Parameters
    ----------
    basis : OneBasis, CrossBasis, or str
        Basis matrix or name (for GAM smoothers)
    model : fitted model object, optional
        Fitted statistical model
    coef : array-like, optional
        Model coefficients (if model not provided)
    vcov : array-like, optional
        Variance-covariance matrix (if model not provided)
    model_link : str, optional
        Link function name
    at : array-like, optional
        Values at which to make predictions
    from_val : float, optional
        Starting value for prediction range
    to_val : float, optional
        Ending value for prediction range
    by : float, optional
        Step size for prediction range
    lag : array-like, optional
        Lag sub-period for predictions
    bylag : float, default=1.0
        Step size for lag dimension
    cen : float, optional
        Centering value for predictions
    ci_level : float, default=0.95
        Confidence interval level
    cumul : bool, default=False
        Whether to compute cumulative effects
        
    Attributes
    ----------
    predvar : np.ndarray
        Prediction values for exposure dimension
    lag : np.ndarray
        Lag range used for predictions
    coefficients : np.ndarray
        Model coefficients
    vcov : np.ndarray
        Variance-covariance matrix
    matfit : np.ndarray
        Lag-specific effect estimates
    matse : np.ndarray
        Lag-specific standard errors
    allfit : np.ndarray
        Overall cumulative effect estimates
    allse : np.ndarray
        Overall cumulative standard errors
    ci_level : float
        Confidence interval level
    model_class : str
        Model class name
    model_link : str
        Link function used
    """
    
    def __init__(self,
                 basis: Union[OneBasis, CrossBasis, str],
                 model: Optional[Any] = None,
                 coef: Optional[np.ndarray] = None,
                 vcov: Optional[np.ndarray] = None,
                 model_link: Optional[str] = None,
                 at: Optional[np.ndarray] = None,
                 from_val: Optional[float] = None,
                 to_val: Optional[float] = None,
                 by: Optional[float] = None,
                 lag: Optional[Union[int, List, Tuple]] = None,
                 bylag: float = 1.0,
                 cen: Optional[float] = None,
                 ci_level: float = 0.95,
                 cumul: bool = False):
        
        # Determine basis type
        self.basis_type = self._determine_basis_type(basis)
        self.basis_name = getattr(basis, '__name__', str(basis))
        
        # Store basis
        if isinstance(basis, str):
            # GAM smoother case (not fully implemented)
            raise NotImplementedError("GAM smoother predictions not yet implemented")
        else:
            self.basis = basis
        
        # Get original lag range
        if hasattr(basis, 'lag'):
            self.orig_lag = basis.lag
        else:
            self.orig_lag = np.array([0, 0])
        
        # Set prediction lag range
        if lag is None:
            self.lag = self.orig_lag.copy()
        else:
            self.lag = mklag(lag)
        
        # Validate lag range
        if not np.array_equal(self.lag, self.orig_lag) and cumul:
            raise ValueError("Cumulative prediction not allowed for lag sub-period")
        
        # Validate bylag for integer lag functions
        if bylag != 1.0 and hasattr(basis, 'arglag'):
            if basis.arglag.get('fun') == 'integer':
                raise ValueError("Prediction for non-integer lags not allowed for type 'integer'")
        
        # Validate inputs
        if model is None and (coef is None or vcov is None):
            raise ValueError("Either 'model' or both 'coef' and 'vcov' must be provided")
        
        if not (0 < ci_level < 1):
            raise ValueError("ci_level must be between 0 and 1")
        
        # Extract model information
        if model is not None:
            model_info = validate_model_compatibility(model, basis.shape[1], self.basis_name)
            self.coefficients = model_info['coef']
            self.vcov = model_info['vcov']
            self.model_link = model_info['link'] or model_link
            self.model_class = model_info['class']
        else:
            self.coefficients = np.asarray(coef)
            self.vcov = np.asarray(vcov)
            self.model_link = model_link
            self.model_class = 'Unknown'
        
        # Validate coefficient and vcov dimensions
        basis_ncol = basis.shape[1]
        if len(self.coefficients) < basis_ncol:
            raise ValueError(f"Coefficients length ({len(self.coefficients)}) < basis columns ({basis_ncol})")
        
        if self.vcov.shape[0] < basis_ncol or self.vcov.shape[1] < basis_ncol:
            raise ValueError(f"Variance-covariance matrix shape {self.vcov.shape} too small for basis")
        
        # Trim to basis size
        self.coefficients = self.coefficients[:basis_ncol]
        self.vcov = self.vcov[:basis_ncol, :basis_ncol]
        
        # Set prediction parameters
        self.bylag = bylag
        self.ci_level = ci_level
        self.cumul = cumul
        
        # Determine prediction values and centering
        self.predvar, self.cen = self._setup_predictions(at, from_val, to_val, by, cen)
        
        # Generate predictions
        self._generate_predictions()
    
    def _determine_basis_type(self, basis) -> str:
        """Determine the type of basis object."""
        if isinstance(basis, CrossBasis):
            return 'cb'
        elif isinstance(basis, OneBasis):
            return 'one'
        elif isinstance(basis, str):
            return 'gam'
        else:
            raise ValueError("basis must be OneBasis, CrossBasis, or string")
    
    def _setup_predictions(self, at, from_val, to_val, by, cen) -> Tuple[np.ndarray, Optional[float]]:
        """Setup prediction values and centering."""
        
        # Get range from basis
        if hasattr(self.basis, 'range'):
            range_vals = self.basis.range
        else:
            range_vals = (0, 1)  # Default fallback
        
        # Determine prediction values
        if at is not None:
            predvar = np.asarray(at)
        elif from_val is not None or to_val is not None or by is not None:
            # Use from/to/by specification
            start = from_val if from_val is not None else range_vals[0]
            end = to_val if to_val is not None else range_vals[1]
            step = by if by is not None else (end - start) / 20
            predvar = np.arange(start, end + step/2, step)
        else:
            # Default: use range with reasonable number of points
            predvar = np.linspace(range_vals[0], range_vals[1], 21)
        
        # Handle centering
        if cen is None:
            # Try to get from basis attributes
            if hasattr(self.basis, 'argvar') and 'cen' in self.basis.argvar:
                cen_val = self.basis.argvar['cen']
            elif hasattr(self.basis, 'cen'):
                cen_val = self.basis.cen
            else:
                cen_val = None
        else:
            cen_val = cen
        
        return predvar, cen_val
    
    def _generate_predictions(self):
        """Generate all predictions."""
        
        # Create prediction matrix for lag-specific effects
        predlag = seqlag(self.lag, self.bylag)
        self._create_prediction_matrix(self.predvar, predlag)
        
        # Generate lag-specific predictions
        self.matfit = (self.Xpred @ self.coefficients).reshape(len(self.predvar), len(predlag))
        matvar = np.sum((self.Xpred @ self.vcov) * self.Xpred, axis=1)
        self.matse = np.sqrt(np.maximum(0, matvar)).reshape(len(self.predvar), len(predlag))
        
        # Set names
        self.predvar_names = [str(v) for v in self.predvar]
        self.lag_names = [f"lag{int(l)}" if l.is_integer() else f"lag{l:.1f}" for l in predlag]
        
        # Generate overall and cumulative predictions
        self._generate_overall_predictions()
        
        # Generate confidence intervals
        self._generate_confidence_intervals()
    
    def _create_prediction_matrix(self, predvar: np.ndarray, predlag: np.ndarray):
        """Create the design matrix for predictions."""
        
        if self.basis_type == 'cb':
            # Cross-basis prediction matrix
            self.Xpred = self._create_crossbasis_prediction_matrix(predvar, predlag)
        elif self.basis_type == 'one':
            # One-dimensional basis prediction matrix
            self.Xpred = self._create_onebasis_prediction_matrix(predvar, predlag)
        else:
            raise NotImplementedError(f"Prediction matrix for {self.basis_type} not implemented")
    
    def _create_crossbasis_prediction_matrix(self, predvar: np.ndarray, predlag: np.ndarray) -> np.ndarray:
        """Create prediction matrix for cross-basis."""
        
        # Create marginal basis matrices
        var_basis = OneBasis(predvar, **self.basis.argvar)
        lag_basis = OneBasis(predlag, **self.basis.arglag)
        
        # Apply centering if specified
        if self.cen is not None:
            cen_basis = OneBasis([self.cen], **self.basis.argvar)
            var_basis.basis = var_basis.basis - cen_basis.basis
        
        # Create tensor product
        n_var = len(predvar)
        n_lag = len(predlag)
        n_var_basis = var_basis.shape[1]
        n_lag_basis = lag_basis.shape[1]
        
        Xpred = np.zeros((n_var * n_lag, n_var_basis * n_lag_basis))
        
        for i in range(n_var):
            for j in range(n_lag):
                row_idx = i * n_lag + j
                for v in range(n_var_basis):
                    for l in range(n_lag_basis):
                        col_idx = v * n_lag_basis + l
                        Xpred[row_idx, col_idx] = var_basis.basis[i, v] * lag_basis.basis[j, l]
        
        return Xpred
    
    def _create_onebasis_prediction_matrix(self, predvar: np.ndarray, predlag: np.ndarray) -> np.ndarray:
        """Create prediction matrix for one-dimensional basis."""
        
        # For OneBasis, predlag should be ignored (or length 1)
        if len(predlag) > 1:
            warnings.warn("OneBasis prediction ignores lag dimension beyond first value")
        
        # Create basis matrix
        basis_matrix = OneBasis(predvar, **self.basis.attributes)
        
        # Apply centering if specified
        if self.cen is not None:
            cen_basis = OneBasis([self.cen], **self.basis.attributes)
            basis_matrix.basis = basis_matrix.basis - cen_basis.basis
        
        return basis_matrix.basis
    
    def _generate_overall_predictions(self):
        """Generate overall cumulative predictions."""
        
        # Use integer lags for overall predictions
        predlag_int = seqlag(self.lag)
        
        # Create prediction matrix for integer lags
        self._create_prediction_matrix(self.predvar, predlag_int)
        
        # Compute overall effects by summing across lags
        Xpred_all = np.zeros((len(self.predvar), self.coefficients.shape[0]))
        
        if self.cumul:
            # Initialize cumulative arrays
            self.cumfit = np.zeros((len(self.predvar), len(predlag_int)))
            self.cumse = np.zeros((len(self.predvar), len(predlag_int)))
        
        # Sum across lags
        for i, lag_val in enumerate(predlag_int):
            # Get indices for this lag
            lag_indices = slice(i * len(self.predvar), (i + 1) * len(self.predvar))
            Xpred_lag = self.Xpred[lag_indices, :]
            
            Xpred_all += Xpred_lag
            
            if self.cumul:
                # Cumulative effects up to this lag
                self.cumfit[:, i] = Xpred_all @ self.coefficients
                cumvar = np.sum((Xpred_all @ self.vcov) * Xpred_all, axis=1)
                self.cumse[:, i] = np.sqrt(np.maximum(0, cumvar))
        
        # Overall effects
        self.allfit = Xpred_all @ self.coefficients
        allvar = np.sum((Xpred_all @ self.vcov) * Xpred_all, axis=1)
        self.allse = np.sqrt(np.maximum(0, allvar))
    
    def _generate_confidence_intervals(self):
        """Generate confidence intervals for all predictions."""
        
        z_score = stats.norm.ppf(1 - (1 - self.ci_level) / 2)
        
        # Determine if we need to transform (log/logit links)
        transform = self.model_link in ['log', 'logit'] if self.model_link else False
        
        if transform:
            # Relative risks/odds ratios
            self.matRRfit = np.exp(self.matfit)
            self.matRRlow = np.exp(self.matfit - z_score * self.matse)
            self.matRRhigh = np.exp(self.matfit + z_score * self.matse)
            
            self.allRRfit = np.exp(self.allfit)
            self.allRRlow = np.exp(self.allfit - z_score * self.allse)
            self.allRRhigh = np.exp(self.allfit + z_score * self.allse)
            
            if self.cumul:
                self.cumRRfit = np.exp(self.cumfit)
                self.cumRRlow = np.exp(self.cumfit - z_score * self.cumse)
                self.cumRRhigh = np.exp(self.cumfit + z_score * self.cumse)
        
        else:
            # Linear scale
            self.matlow = self.matfit - z_score * self.matse
            self.mathigh = self.matfit + z_score * self.matse
            
            self.alllow = self.allfit - z_score * self.allse
            self.allhigh = self.allfit + z_score * self.allse
            
            if self.cumul:
                self.cumlow = self.cumfit - z_score * self.cumse
                self.cumhigh = self.cumfit + z_score * self.cumse
    
    def summary(self) -> str:
        """
        Return a summary of the CrossPred object.
        
        Returns
        -------
        str
            Summary string
        """
        summary_lines = [
            f"CrossPred object",
            f"Basis type: {self.basis_type}",
            f"Model class: {self.model_class}",
            f"Link function: {self.model_link or 'identity'}",
            f"Prediction values: {len(self.predvar)} points",
            f"Lag range: [{self.lag[0]}, {self.lag[1]}]",
            f"Confidence level: {self.ci_level:.0%}",
        ]
        
        if self.cen is not None:
            summary_lines.append(f"Centered at: {self.cen}")
        
        if self.cumul:
            summary_lines.append("Includes cumulative effects")
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        return f"CrossPred(basis_type='{self.basis_type}', predvar={len(self.predvar)}, lag={self.lag.tolist()})"
    
    def __str__(self) -> str:
        return self.summary()


def crosspred(basis: Union[OneBasis, CrossBasis],
              model: Any,
              at: Optional[np.ndarray] = None,
              from_val: Optional[float] = None,
              to_val: Optional[float] = None,
              by: Optional[float] = None,
              lag: Optional[Union[int, List, Tuple]] = None,
              bylag: float = 1.0,
              cen: Optional[float] = None,
              ci_level: float = 0.95,
              cumul: bool = False,
              **kwargs) -> CrossPred:
    """
    Create cross-predictions from distributed lag models.
    
    This function creates predictions from fitted distributed lag models,
    matching R's dlnm::crosspred() interface. It generates predictions
    over specified ranges of the exposure variable and lag periods.
    
    Parameters
    ----------
    basis : OneBasis or CrossBasis
        The basis object used in model fitting
    model : fitted model object
        Fitted statistical model (e.g., from statsmodels GLM)
    at : array-like, optional
        Specific values at which to make predictions
    from_val : float, optional
        Starting value for prediction range
    to_val : float, optional
        Ending value for prediction range
    by : float, optional
        Step size for prediction range
    lag : int, list, or tuple, optional
        Lag sub-period for predictions
    bylag : float, default=1.0
        Step size for lag dimension
    cen : float, optional
        Centering value for predictions
    ci_level : float, default=0.95
        Confidence interval level
    cumul : bool, default=False
        Whether to compute cumulative effects
    **kwargs
        Additional arguments passed to CrossPred
        
    Returns
    -------
    crosspred : CrossPred
        Cross-prediction object with fitted values, standard errors,
        and confidence intervals
        
    Examples
    --------
    >>> from pydlnm import CrossBasis, crosspred, fit_dlnm_model
    >>> cb = CrossBasis(temp, lag=21, argvar={'fun': 'bs'})
    >>> model = fit_dlnm_model(cb, deaths, family='poisson')
    >>> pred = crosspred(cb, model.fitted_values, cen=mean_temp)
    >>> print(pred.summary())
    """
    
    # Extract model coefficients and variance-covariance matrix
    if hasattr(model, 'params') and hasattr(model, 'cov_params'):
        # statsmodels GLM
        coef = model.params
        vcov = model.cov_params()
        model_link = getattr(model.model.family, 'link', None)
        if model_link:
            model_link = model_link.__class__.__name__.lower()
    elif hasattr(model, 'coef_') and hasattr(model, 'predict'):
        # sklearn model
        coef = getattr(model, 'coef_', None)
        vcov = None  # sklearn doesn't provide covariance matrix
        model_link = 'identity'
    else:
        raise ValueError("Model type not recognized. Must be statsmodels or sklearn model.")
    
    # Create CrossPred object
    pred = CrossPred(
        basis=basis,
        model=model,
        coef=coef,
        vcov=vcov,
        model_link=model_link,
        at=at,
        from_val=from_val,
        to_val=to_val,
        by=by,
        lag=lag,
        bylag=bylag,
        cen=cen,
        ci_level=ci_level,
        cumul=cumul,
        **kwargs
    )
    
    return pred