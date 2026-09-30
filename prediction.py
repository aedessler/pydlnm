"""
Prediction classes for PyDLNM

This module contains classes for making predictions from distributed lag models,
including lag-specific, overall cumulative, and predictor-specific predictions.
"""

import numpy as np
from scipy import stats
from typing import Union, Optional, List, Dict, Any, Tuple
import warnings

from basis import OneBasis, CrossBasis
from model_utils import validate_model_compatibility
from utils import mklag, seqlag, pretty


def mkat(at, from_val, to_val, by, range_, lag=None, bylag=1.0) -> np.ndarray:
    """
    Exposure values at which crosspred predicts (port of R's ``mkat()``).

    Without ``at``, the grid is ``pretty(c(from, to), n = 50)`` restricted to ``[from, to]`` (``from``/``to``
    default to the range of the basis); with ``by`` it is ``seq(min(pretty), to, by)``. A vector ``at`` is sorted,
    made unique and stripped of missing values. A matrix ``at`` (rows = exposure histories) is returned as is.
    """
    if at is None:
        if from_val is None:
            from_val = range_[0]
        if to_val is None:
            to_val = range_[1]
        nobs = 50 if by is None else max(1, (range_[1] - range_[0]) / by)
        grid = pretty([from_val, to_val], n=nobs)
        grid = grid[(grid >= from_val) & (grid <= to_val)]
        if grid.size == 0:
            raise ValueError("no prediction values between 'from' and 'to'")
        return grid if by is None else seqlag([grid.min(), to_val], by)
    at = np.asarray(at, dtype=float)
    if at.ndim == 2:
        n_lags = int(np.diff(mklag(lag))[0]) + 1
        if at.shape[1] != n_lags:
            raise ValueError("matrix in 'at' must have ncol=diff(lag)+1")
        if bylag != 1:
            raise ValueError("'bylag!=1 not allowed with 'at' in matrix form")
        return at
    at = at.ravel()
    return np.unique(at[~np.isnan(at)])


def mkcen(cen, basis, range_):
    """
    Centering value of crosspred (port of R's ``mkcen()``); ``None`` means no centering.

    Taken from the basis when not given. For thr/strata/integer/lin a logical ``cen`` means no centering; for the
    other functions ``None``/``True`` is the (approximate) mid-range ``median(pretty(range))`` and ``False`` is no
    centering. A basis with an intercept is never centered.
    """
    is_cb = isinstance(basis, CrossBasis)
    nocen = cen is None
    if nocen:
        cen = basis.argvar.get('cen') if is_cb else getattr(basis, 'cen', None)
    fun = basis.argvar.get('fun') if is_cb else getattr(basis, 'fun', None)
    intercept = basis.argvar.get('intercept') if is_cb else getattr(basis, 'attributes', {}).get('intercept')
    is_logical = isinstance(cen, (bool, np.bool_))
    if isinstance(fun, str) and fun in ('thr', 'strata', 'integer', 'lin'):
        if is_logical:
            cen = None
    else:
        if cen is None or (is_logical and cen):
            cen = float(np.median(pretty(range_)))
        elif is_logical and not cen:
            cen = None
    if isinstance(intercept, (bool, np.bool_)) and intercept:
        cen = None
    if nocen and cen is not None:
        warnings.warn(f"centering value unspecified. Automatically set to {cen}")
    return cen


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
        
        # Validate inputs
        if model is None and (coef is None or vcov is None):
            raise ValueError("Either 'model' or both 'coef' and 'vcov' must be provided")
        
        if not (0 < ci_level < 1):
            raise ValueError("ci_level must be between 0 and 1")
        
        # Extract model information
        if model is not None:
            model_info = validate_model_compatibility(model, basis.shape[1], self.basis_name, kind=self.basis_type)
            self.coefficients = model_info['coef']
            self.vcov = model_info['vcov']
            self.model_link = model_info['link'] or model_link
            self.model_class = model_info['class']
        else:
            self.coefficients = np.asarray(coef, dtype=float).ravel()
            self.vcov = np.atleast_2d(np.asarray(vcov, dtype=float))
            npar = len(self.coefficients)
            if (self.vcov.shape != (npar, npar) or np.isnan(self.coefficients).any() or np.isnan(self.vcov).any()
                    or npar > basis.shape[1]):
                raise ValueError("coef/vcov not consistent with basis matrix. See help(crosspred)")
            self.model_link = model_link
            self.model_class = 'Unknown'
        
        # Reduced coefficients: fewer coefficients than basis columns are the overall-effect coefficients of the
        # exposure basis (crossreduce / BLUP). This is R's crosspred(onebasis, coef, vcov): a single lag [0, 0].
        basis_ncol = basis.shape[1]
        coef_len = len(self.coefficients)
        self.reduced_coefficients = coef_len < basis_ncol
        if self.reduced_coefficients and not (isinstance(basis, CrossBasis) and hasattr(basis, 'argvar')):
            raise ValueError(f"Cannot handle reduced coefficients for basis type {type(basis)}")
        
        # Get original lag range and set the prediction lag range
        if self.reduced_coefficients:
            self.orig_lag = np.array([0, 0])
            if lag is not None and not np.array_equal(mklag(lag), self.orig_lag):
                raise ValueError("'lag' is not applicable to reduced (overall-effect) coefficients")
        elif hasattr(basis, 'lag'):
            self.orig_lag = basis.lag
        else:
            self.orig_lag = np.array([0, 0])
        self.lag = self.orig_lag.copy() if lag is None else mklag(lag)
        
        # Validate lag range
        if not np.array_equal(self.lag, self.orig_lag) and cumul:
            raise ValueError("Cumulative prediction not allowed for lag sub-period")
        
        # Validate bylag for integer lag functions
        if bylag != 1.0 and hasattr(basis, 'arglag'):
            if basis.arglag.get('fun') == 'integer':
                raise ValueError("Prediction for non-integer lags not allowed for type 'integer'")
        
        # Exposure values and centering (R: mkat, mkcen)
        range_vals = basis.range if hasattr(basis, 'range') else (0, 1)
        at_values = mkat(at, from_val, to_val, by, range_vals, self.lag, bylag)
        if at_values.ndim == 2:
            if self.basis_type != 'cb' or self.reduced_coefficients:
                raise NotImplementedError("matrix 'at' is only supported for a CrossBasis with full coefficients")
            self._at_matrix = at_values
            self.predvar = np.arange(1, at_values.shape[0] + 1)
        else:
            self._at_matrix = None
            self.predvar = at_values
        self.cen = mkcen(cen, basis, range_vals)
        
        if self.reduced_coefficients:
            # Variable basis at the prediction values, from the arguments resolved on the training data
            self.variable_basis = OneBasis(self.predvar, **basis.argvar)
            if self.variable_basis.shape[1] != coef_len:
                raise ValueError(f"Variable basis ({self.variable_basis.shape[1]}) doesn't match coefficients ({coef_len})")
            self.original_basis = basis
        else:
            # Full coefficients: the coefficients of the basis, one per column (checked above or selected by name)
            if self.vcov.shape != (basis_ncol, basis_ncol):
                raise ValueError(f"Variance-covariance matrix shape {self.vcov.shape} not consistent with basis")
        
        # Set prediction parameters
        self.bylag = bylag
        self.ci_level = ci_level
        self.cumul = cumul
        
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
        self.lag_names = [f"lag{l:.15g}" for l in predlag]   # R: paste0("lag", predlag) (15 significant digits)
        
        # Generate overall and cumulative predictions
        self._generate_overall_predictions()
        
        # Generate confidence intervals
        self._generate_confidence_intervals()
    
    def _create_prediction_matrix(self, predvar: np.ndarray, predlag: np.ndarray):
        """Create the design matrix for predictions."""
        
        if hasattr(self, 'reduced_coefficients') and self.reduced_coefficients:
            # Use variable basis only for reduced coefficients
            self.Xpred = self._create_reduced_prediction_matrix(predvar, predlag)
        elif self.basis_type == 'cb':
            # Cross-basis prediction matrix
            self.Xpred = self._create_crossbasis_prediction_matrix(predvar, predlag)
        elif self.basis_type == 'one':
            # One-dimensional basis prediction matrix
            self.Xpred = self._create_onebasis_prediction_matrix(predvar, predlag)
        else:
            raise NotImplementedError(f"Prediction matrix for {self.basis_type} not implemented")
    
    def _create_reduced_prediction_matrix(self, predvar: np.ndarray, predlag: np.ndarray) -> np.ndarray:
        """Prediction matrix for reduced coefficients: the exposure basis, repeated for every lag requested."""
        basis_matrix = self.variable_basis.basis
        
        # Apply centering if specified
        if self.cen is not None:
            cen_basis = OneBasis([self.cen], **self.original_basis.argvar)
            basis_matrix = basis_matrix - cen_basis.basis
        
        # rows ordered VAR-outer, LAG-inner
        return np.repeat(basis_matrix, len(predlag), axis=0)
    
    def _create_crossbasis_prediction_matrix(self, predvar: np.ndarray, predlag: np.ndarray) -> np.ndarray:
        """Create prediction matrix for cross-basis (rows VAR-outer, LAG-inner; columns v*n_lag_basis + l)."""
        n_var = len(predvar)
        n_lag = len(predlag)
        
        # Marginal bases, rebuilt from the arguments resolved on the training data. A matrix 'at' gives one
        # exposure value per (row, lag): R's varvec <- as.numeric(at).
        if self._at_matrix is not None:
            var_values = self._at_matrix.ravel()
        else:
            var_values = np.asarray(predvar, dtype=float)
        var_basis = OneBasis(var_values, **self.basis.argvar).basis
        
        # integer lag: identity matrix (each lag is independent)
        if self.basis.arglag.get('fun') == 'integer':
            lag_basis = np.eye(n_lag)
        else:
            lag_basis = OneBasis(predlag, **self.basis.arglag).basis
        
        # Centering is applied to the exposure dimension only
        if self.cen is not None:
            cen_basis = OneBasis([self.cen], **self.basis.argvar).basis
            var_basis = var_basis - cen_basis
        
        n_var_basis = var_basis.shape[1]
        n_lag_basis = lag_basis.shape[1]
        if self._at_matrix is not None:
            var_basis = var_basis.reshape(n_var, n_lag, n_var_basis)
            Xpred = np.einsum('ijv,jl->ijvl', var_basis, lag_basis)
        else:
            Xpred = np.einsum('iv,jl->ijvl', var_basis, lag_basis)
        return Xpred.reshape(n_var * n_lag, n_var_basis * n_lag_basis)
    
    def _create_onebasis_prediction_matrix(self, predvar: np.ndarray, predlag: np.ndarray) -> np.ndarray:
        """Create prediction matrix for one-dimensional basis."""
        
        # For OneBasis, predlag should be ignored (or length 1)
        if len(predlag) > 1:
            warnings.warn("OneBasis prediction ignores lag dimension beyond first value")
        
        # Rebuild the basis with the arguments resolved on the training data
        # (R mkXpred, type "one": attributes matched with formals(fun))
        args = self.basis.resolved_args()
        basis_matrix = OneBasis(predvar, **args)

        # Apply centering if specified
        if self.cen is not None:
            cen_basis = OneBasis([self.cen], **args)
            basis_matrix.basis = basis_matrix.basis - cen_basis.basis
        
        return basis_matrix.basis
    
    def _generate_overall_predictions(self):
        """Generate overall cumulative predictions."""
        
        if self.reduced_coefficients:
            # Overall effects of reduced coefficients: the exposure basis (single lag [0, 0]) times the coefficients
            var_basis = self.variable_basis.basis
            if self.cen is not None:
                cen_basis = OneBasis([self.cen], **self.original_basis.argvar)
                var_basis = var_basis - cen_basis.basis
            
            self.allfit = var_basis @ self.coefficients
            allvar = np.sum((var_basis @ self.vcov) * var_basis, axis=1)
            self.allse = np.sqrt(np.maximum(0, allvar))
            
            # Cumulative effects over the single lag are the overall effects
            if self.cumul:
                self.cumfit = self.allfit.reshape(-1, 1).copy()
                self.cumse = self.allse.reshape(-1, 1).copy()
                
        else:
            # Standard approach for full cross-basis coefficients
            # Use integer lags for overall predictions
            predlag_int = seqlag(self.lag)

            # Create prediction matrix for integer lags
            # Xpred shape: (n_var * n_lag, n_basis), rows ordered VAR-outer, LAG-inner
            # i.e. row = var_idx * n_lag + lag_idx
            self._create_prediction_matrix(self.predvar, predlag_int)

            n_var_pred = len(self.predvar)
            n_lag_pred = len(predlag_int)
            n_basis = self.coefficients.shape[0]

            # Reshape to (n_var, n_lag, n_basis) so we can sum/accumulate over the lag axis
            Xpred_3d = self.Xpred.reshape(n_var_pred, n_lag_pred, n_basis)

            if self.cumul:
                # Cumulative sum over lags: shape (n_var, n_lag, n_basis)
                Xpred_cum = np.cumsum(Xpred_3d, axis=1)
                self.cumfit = np.zeros((n_var_pred, n_lag_pred))
                self.cumse = np.zeros((n_var_pred, n_lag_pred))
                for i in range(n_lag_pred):
                    Xc = Xpred_cum[:, i, :]  # (n_var, n_basis)
                    self.cumfit[:, i] = Xc @ self.coefficients
                    cumvar = np.sum((Xc @ self.vcov) * Xc, axis=1)
                    self.cumse[:, i] = np.sqrt(np.maximum(0, cumvar))

            # Overall: sum over all lags for each variable value
            Xpred_all = Xpred_3d.sum(axis=1)  # (n_var, n_basis)
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
              model: Any = None,
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
    
    # Validate parameters - either model or both coef and vcov must be provided
    if model is None and (coef is None or vcov is None):
        raise ValueError("Either 'model' or both 'coef' and 'vcov' must be provided.")
    
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