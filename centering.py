"""
Centering functionality for PyDLNM

Implements minimum mortality temperature (MMT) and other centering methods
for proper interpretation of relative risks in distributed lag models.
"""

import numpy as np
from scipy import optimize
from typing import Optional, Tuple, Union, Dict, Any, List
import warnings

from basis import CrossBasis
from prediction import CrossPred
from utils import asfloat, quantile7


def find_mmt_blup(x: np.ndarray,
                  blup_coef: np.ndarray,
                  fun: str = "bs",
                  knots: Optional[np.ndarray] = None,
                  degree: int = 2,
                  percentile_range: Tuple[int, int] = (1, 99)) -> Dict:
    """
    Find the minimum mortality temperature from BLUP coefficients (Gasparrini et al. 2015, 02.secondstage.R)

    R recipe: ``predvar <- quantile(x, 1:99/100)``, ``bvar <- onebasis(predvar, fun, knots=quantile(x, c(10,75,90)/100),
    degree, Boundary.knots=range(x))``, ``minperccity <- (1:99)[which.min(bvar %*% blup)]``. The boundary knots are
    the range of ``x``, not of the percentile grid.
    
    Parameters:
    -----------
    x : array-like
        Temperature time series data (NaN are ignored)
    blup_coef : array-like 
        BLUP (reduced) coefficients from the meta-analysis, one per column of the exposure basis
    fun : {"bs", "ns"}, default "bs"
        Exposure basis function
    knots : array-like, optional
        Interior knots (default: the 10th, 75th and 90th percentiles of ``x``)
    degree : int, default 2
        Degree of the B-spline (ignored for "ns")
    percentile_range : tuple, default (1, 99)
        Range of (integer) percentiles searched for the MMT
        
    Returns:
    --------
    dict
        ``mmt`` (temperature), ``percentile``, ``min_risk``, ``risk_range``, ``predvar``, ``risk_values``,
        ``basis_matrix`` and ``method``.

    Raises
    ------
    ValueError
        For an unsupported ``fun``, coefficients that do not match the basis, or NaN risks (R: which.min gives
        integer(0) and no MMT).
    """
    from basis import OneBasis
    
    x = asfloat(x).ravel()                      # masked / nullable cells are missing values (NaN)
    x = x[~np.isnan(x)]
    blup_coef = asfloat(blup_coef).ravel()
    
    # Prediction grid: percentiles of x (R: quantile, type 7)
    percentiles = np.arange(percentile_range[0], percentile_range[1] + 1)
    predvar = quantile7(x, percentiles / 100)            # R: quantile(x, 1:99/100)
    
    if knots is None:
        knots = quantile7(x, np.array([10, 75, 90]) / 100)
    boundary = np.array([np.min(x), np.max(x)])
    
    if fun == "bs":
        args = {'knots': knots, 'degree': degree, 'Boundary_knots': boundary}
    elif fun == "ns":
        args = {'knots': knots, 'Boundary_knots': boundary}
    else:
        raise ValueError(f"fun must be 'bs' or 'ns', not {fun!r}")
    bvar = OneBasis(predvar, fun=fun, **args).basis
    
    if bvar.shape[1] != len(blup_coef):
        raise ValueError(f"{len(blup_coef)} BLUP coefficients do not match the {bvar.shape[1]} columns of the basis")
    
    # Risk values: bvar %*% blup; the minimum risk point (R: which.min)
    risk_values = bvar @ blup_coef
    if np.isnan(risk_values).all():
        raise ValueError("no MMT: the risk values are all NaN (NaN in the BLUP coefficients?)")
    min_idx = int(np.nanargmin(risk_values))
    
    return {
        'mmt': predvar[min_idx],
        'percentile': int(percentiles[min_idx]),
        'min_risk': risk_values[min_idx],
        'risk_range': (np.nanmin(risk_values), np.nanmax(risk_values)),
        'predvar': predvar,
        'risk_values': risk_values,
        'basis_matrix': bvar,
        'method': 'blup_optimization'
    }


def find_mmt(basis: CrossBasis, 
             model: Any,
             coef: Optional[np.ndarray] = None,
             vcov: Optional[np.ndarray] = None,
             at: Optional[np.ndarray] = None,
             from_val: Optional[float] = None,
             to_val: Optional[float] = None,
             by: Optional[float] = None,
             method: str = "overall",
             name: Optional[str] = None) -> Dict:
    """
    Find minimum mortality temperature (MMT) or minimum risk exposure
    
    Parameters:
    -----------
    basis : CrossBasis
        Cross-basis object used in the model
    model : fitted model object, optional
        Fitted statistical model
    coef : array-like, optional
        Model coefficients (if model not provided)
    vcov : array-like, optional
        Variance-covariance matrix (if model not provided)
    at : array-like, optional
        Values to search over for MMT
    from_val : float, optional
        Starting value for search range
    to_val : float, optional
        Ending value for search range
    by : float, optional
        Step size for search range
    method : str, default "overall"
        Method for MMT calculation: "overall" searches the overall (summed over lags) effect; "lagspecific"
        searches the lag-specific effect at the first lag of the prediction lag range (``basis.lag[0]``, column 0
        of ``matfit``; lag 0 for the usual lag range starting at 0)
    name : str, optional
        Name of the cross-basis in ``model``: prefix of the coefficient names of its terms (R: the name of the
        basis object); selects its block in a model that holds several cross-bases (see ``crosspred``)
        
    Returns:
    --------
    dict
        Dictionary containing MMT information:
        - mmt: minimum mortality temperature value
        - fit: fitted value at MMT
        - se: standard error at MMT
        - ci_low: lower confidence interval at MMT
        - ci_high: upper confidence interval at MMT
        - method: method used
    """
    
    # Create prediction object
    try:
        pred = CrossPred(
            basis=basis,
            model=model,
            coef=coef,
            vcov=vcov,
            at=at,
            from_val=from_val,
            to_val=to_val,
            by=by,
            cen=False,  # No centering for MMT search (R: cen=FALSE)
            name=name
        )
    except Exception as e:
        raise ValueError(f"Error creating prediction object: {e}")
    
    # Get prediction values and effects
    predvar = pred.predvar
    
    if method == "overall":
        # Use overall cumulative effects
        fit_values = pred.allfit
        se_values = pred.allse
    elif method == "lagspecific":
        # Lag-specific effect at the first lag of the prediction lag range (column 0 of matfit)
        if pred.matfit.shape[1] > 0:
            fit_values = pred.matfit[:, 0]  # First lag
            se_values = pred.matse[:, 0]
        else:
            raise ValueError("No lag-specific effects available for MMT calculation")
    else:
        raise ValueError("method must be 'overall' or 'lagspecific'")
    
    # Find minimum
    min_idx = int(np.nanargmin(fit_values))
    mmt_value = predvar[min_idx]
    mmt_fit = fit_values[min_idx]
    mmt_se = se_values[min_idx]
    
    # Calculate confidence intervals
    z_score = 1.96  # 95% CI
    
    # Check if model uses a log or logit link (RR / OR, as crosspred's RRfit)
    if getattr(pred, 'model_link', None) in ('log', 'logit'):
        # Relative risk scale
        mmt_rr = np.exp(mmt_fit)
        mmt_ci_low = np.exp(mmt_fit - z_score * mmt_se)
        mmt_ci_high = np.exp(mmt_fit + z_score * mmt_se)
        scale = "RR"
    else:
        # Linear scale
        mmt_rr = mmt_fit
        mmt_ci_low = mmt_fit - z_score * mmt_se
        mmt_ci_high = mmt_fit + z_score * mmt_se
        scale = "linear"
    
    result = {
        'mmt': mmt_value,
        'fit': mmt_fit,
        'se': mmt_se,
        'rr': mmt_rr,
        'ci_low': mmt_ci_low,
        'ci_high': mmt_ci_high,
        'method': method,
        'scale': scale,
        'predvar': predvar,
        'all_fit': fit_values,
        'all_se': se_values
    }
    
    return result


def recenter_basis(basis: CrossBasis, 
                   model: Any,
                   cen: Optional[float] = None,
                   find_mmt_args: Optional[Dict] = None) -> Tuple[CrossBasis, Dict]:
    """
    Re-center a cross-basis at specified value or MMT
    
    Parameters:
    -----------
    basis : CrossBasis
        Original cross-basis object
    model : fitted model object
        Fitted statistical model
    cen : float, optional
        Centering value. If None, will find MMT automatically
    find_mmt_args : dict, optional
        Arguments to pass to find_mmt function
        
    Returns:
    --------
    tuple
        - recentered_basis: New CrossBasis object centered at specified value
        - centering_info: Dictionary with centering information
    """
    
    # Find MMT if centering value not provided
    if cen is None:
        if find_mmt_args is None:
            find_mmt_args = {}
        
        mmt_result = find_mmt(basis, model, **find_mmt_args)
        cen = mmt_result['mmt']
        centering_info = {
            'method': 'mmt',
            'value': cen,
            'mmt_info': mmt_result
        }
    else:
        centering_info = {
            'method': 'manual',
            'value': cen,
            'mmt_info': None
        }
    
    # Create new basis with centering
    new_argvar = basis.argvar.copy()
    new_argvar['cen'] = cen
    
    # Create recentered basis
    recentered_basis = CrossBasis(
        x=basis.x,
        lag=basis.lag,
        argvar=new_argvar,
        arglag=basis.arglag,
        group=getattr(basis, '_group_labels', None)       # stacked series: the lags stay inside each group
    )
    
    # Store original range and centering info
    recentered_basis._original_basis = basis
    recentered_basis._centering_info = centering_info
    
    return recentered_basis, centering_info


def compare_centering(basis: CrossBasis,
                      model: Any,
                      centering_values: Union[List[float], np.ndarray],
                      at: Optional[np.ndarray] = None,
                      name: Optional[str] = None) -> Dict:
    """
    Compare effects under different centering approaches
    
    Parameters:
    -----------
    basis : CrossBasis
        Cross-basis object
    model : fitted model object
        Fitted statistical model  
    centering_values : array-like
        List of centering values to compare
    at : array-like, optional
        Prediction values
    name : str, optional
        Name of the cross-basis in ``model`` (see ``crosspred``)
        
    Returns:
    --------
    dict
        Comparison results with predictions for each centering
    """
    
    results = {}
    
    for i, cen_val in enumerate(centering_values):
        try:
            pred = CrossPred(
                basis=basis,
                model=model,
                cen=cen_val,
                at=at,
                name=name
            )
            
            # Store key results (repeated centering values keep one record each)
            key = f'cen_{cen_val}'
            dup = 2
            while key in results:
                key = f'cen_{cen_val}_{dup}'
                dup += 1
            results[key] = {
                'centering': cen_val,
                'predvar': pred.predvar,
                'allfit': pred.allfit,
                'allse': pred.allse,
                'prediction_object': pred
            }
            
            # Add RR results if available
            if hasattr(pred, 'allRRfit'):
                results[key]['allRRfit'] = pred.allRRfit
                results[key]['allRRlow'] = pred.allRRlow
                results[key]['allRRhigh'] = pred.allRRhigh
                
        except Exception as e:
            warnings.warn(f"Failed to compute predictions for centering {cen_val}: {e}")
            results.setdefault(f'cen_{cen_val}', None)
    
    # Add summary statistics
    results['summary'] = {
        'centering_values': list(centering_values),
        'successful_runs': len([r for r in results.values() if r is not None and 'centering' in r])
    }
    
    return results


class CenteringManager:
    """
    Utility class for managing centering operations in DLNM analysis
    """
    
    def __init__(self, basis: CrossBasis, model: Any, name: Optional[str] = None):
        """
        Initialize centering manager
        
        Parameters:
        -----------
        basis : CrossBasis
            Cross-basis object
        model : fitted model object
            Fitted statistical model
        name : str, optional
            Name of the cross-basis in ``model`` (see ``crosspred``)
        """
        self.basis = basis
        self.model = model
        self.name = name
        self._mmt_cache = {}
        self._centering_history = []
    
    @staticmethod
    def _freeze(value: Any) -> Any:
        """Hashable form of an argument (arrays by content)."""
        if isinstance(value, np.ndarray):
            return ('ndarray', value.shape, str(value.dtype), value.tobytes())
        if isinstance(value, (list, tuple)):
            return tuple(CenteringManager._freeze(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((k, CenteringManager._freeze(v)) for k, v in value.items()))
        return value
    
    def _with_name(self, kwargs: Dict) -> Dict:
        """Arguments with the name of the cross-basis of the manager (unless given)."""
        return kwargs if self.name is None else {'name': self.name, **kwargs}
    
    def find_mmt(self, **kwargs) -> Dict:
        """Find MMT, cached per (arguments, model, basis): a call with other arguments, or after the model or
        the basis of the manager was replaced, is computed afresh."""
        kwargs = self._with_name(kwargs)
        key = self._freeze(kwargs)
        entry = self._mmt_cache.get(key)
        if entry is None or entry[0] is not self.basis or entry[1] is not self.model:
            entry = (self.basis, self.model, find_mmt(self.basis, self.model, **kwargs))
            self._mmt_cache[key] = entry
        return entry[2]
    
    def recenter_at_mmt(self, **kwargs) -> Tuple[CrossBasis, Dict]:
        """Convenience method to recenter at MMT (recorded in the history like recenter_at_value)"""
        result = recenter_basis(self.basis, self.model, cen=None, find_mmt_args=self._with_name(kwargs))
        self._centering_history.append({
            'method': 'mmt',
            'value': result[1]['value'],
            'timestamp': None
        })
        return result
    
    def recenter_at_value(self, cen: float) -> Tuple[CrossBasis, Dict]:
        """Convenience method to recenter at specific value"""
        result = recenter_basis(self.basis, self.model, cen=cen)
        self._centering_history.append({
            'method': 'manual',
            'value': cen,
            'timestamp': None  # Could add timestamp if needed
        })
        return result
    
    def compare_centering_strategies(self, 
                                   strategies: Optional[List[str]] = None,
                                   custom_values: Optional[List[float]] = None,
                                   **kwargs) -> Dict:
        """
        Compare different centering strategies
        
        Parameters:
        -----------
        strategies : list, optional
            List of strategies: 'mmt', 'mean', 'median', 'percentile_X'
        custom_values : list, optional
            Custom centering values to include
        **kwargs
            Additional arguments for predictions
            
        Returns:
        --------
        dict
            Comparison results
        """
        
        if strategies is None:
            strategies = ['mmt', 'mean']
        
        centering_values = []
        strategy_names = []
        
        # Get data for automatic strategies
        if hasattr(self.basis, 'x'):
            x_data = self.basis.x
            if hasattr(x_data, 'ravel'):
                x_data = x_data.ravel()
            x_data = x_data[~np.isnan(x_data)]  # Remove NaN values
        else:
            x_data = None
        
        for strategy in strategies:
            if strategy == 'mmt':
                # the MMT is searched on the same exposure grid as the predictions
                mmt_info = self.find_mmt(**{k: kwargs[k] for k in ('at', 'from_val', 'to_val', 'by')
                                            if kwargs.get(k) is not None})
                centering_values.append(mmt_info['mmt'])
                strategy_names.append('MMT')
            elif strategy == 'mean' and x_data is not None:
                centering_values.append(np.mean(x_data))
                strategy_names.append('Mean')
            elif strategy == 'median' and x_data is not None:
                centering_values.append(np.median(x_data))
                strategy_names.append('Median')
            elif strategy.startswith('percentile_') and x_data is not None:
                try:
                    pct = float(strategy.split('_')[1])
                    centering_values.append(np.percentile(x_data, pct))
                    strategy_names.append(f'{pct}th percentile')
                except (IndexError, ValueError):
                    warnings.warn(f"Invalid percentile strategy: {strategy}")
        
        # Add custom values
        if custom_values:
            centering_values.extend(custom_values)
            strategy_names.extend([f'Custom {v}' for v in custom_values])
        
        # Compare centering approaches
        comparison = compare_centering(self.basis, self.model, 
                                     centering_values, **self._with_name(kwargs))
        
        # Add strategy names to results
        comparison['strategy_names'] = strategy_names
        comparison['strategies'] = strategies
        
        return comparison
    
    def get_centering_history(self) -> List[Dict]:
        """Get history of centering operations"""
        return self._centering_history.copy()