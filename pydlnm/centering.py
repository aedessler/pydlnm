"""
Centering functionality for PyDLNM

Implements minimum mortality temperature (MMT) and other centering methods
for proper interpretation of relative risks in distributed lag models.
"""

import numpy as np
from scipy import optimize
from typing import Optional, Tuple, Union, Dict, Any, List
import warnings

from .basis import CrossBasis
from .prediction import CrossPred


def find_mmt_blup(x: np.ndarray,
                  blup_coef: np.ndarray,
                  fun: str = "bs",
                  knots: Optional[np.ndarray] = None,
                  degree: int = 2,
                  percentile_range: Tuple[int, int] = (1, 99)) -> Dict:
    """
    Find minimum mortality temperature using BLUP coefficients (R-style method)
    
    This implements the same methodology as the R dlnm package's MMT calculation
    using BLUP (Best Linear Unbiased Predictors) coefficients.
    
    Parameters:
    -----------
    x : array-like
        Temperature time series data
    blup_coef : array-like 
        BLUP coefficients from meta-analysis
    fun : str, default "bs"
        Basis function type
    knots : array-like, optional
        Knot positions for basis functions
    degree : int, default 2
        Degree of basis functions
    percentile_range : tuple, default (1, 99)
        Range of percentiles to search for MMT
        
    Returns:
    --------
    dict
        Dictionary containing MMT results
    """
    from .basis_functions import BSplineBasis
    
    # Create prediction range (1st to 99th percentiles like R)
    predvar = np.percentile(x[~np.isnan(x)], 
                          np.arange(percentile_range[0], percentile_range[1] + 1))
    
    # Set up basis function arguments like R's onebasis
    if knots is None:
        # Default knots at 10%, 75%, 90% percentiles
        knots = np.percentile(x[~np.isnan(x)], [10, 75, 90])
    
    # Create basis matrix for prediction range with boundary knots
    x_range = np.array([np.min(x[~np.isnan(x)]), np.max(x[~np.isnan(x)])])
    
    try:
        basis_func = BSplineBasis(
            knots=knots,
            degree=degree,
            include_intercept=True,
            boundary_knots=x_range
        )
        
        # Generate basis matrix for prediction temperatures
        bvar = basis_func.transform(predvar.reshape(-1, 1))
        
        # Calculate risk values: bvar %*% blup_coef (like R)
        risk_values = bvar @ blup_coef
        
        # Find minimum risk point
        min_idx = np.argmin(risk_values)
        mmt_percentile = percentile_range[0] + min_idx
        mmt_temperature = predvar[min_idx]
        
        result = {
            'mmt': mmt_temperature,
            'percentile': mmt_percentile,
            'min_risk': risk_values[min_idx],
            'risk_range': (np.min(risk_values), np.max(risk_values)),
            'predvar': predvar,
            'risk_values': risk_values,
            'basis_matrix': bvar,
            'method': 'blup_optimization'
        }
        
        return result
        
    except Exception as e:
        # Fallback to simpler method if basis function fails
        warnings.warn(f"BLUP MMT calculation failed ({e}), using median fallback")
        median_temp = np.median(x[~np.isnan(x)])
        median_percentile = np.mean(x[~np.isnan(x)] <= median_temp) * 100
        
        return {
            'mmt': median_temp,
            'percentile': median_percentile,
            'method': 'median_fallback',
            'error': str(e)
        }


def find_mmt(basis: CrossBasis, 
             model: Any,
             coef: Optional[np.ndarray] = None,
             vcov: Optional[np.ndarray] = None,
             at: Optional[np.ndarray] = None,
             from_val: Optional[float] = None,
             to_val: Optional[float] = None,
             by: Optional[float] = None,
             method: str = "overall") -> Dict:
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
        Method for MMT calculation: "overall" or "lagspecific"
        
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
            cen=None  # No centering for MMT search
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
        # Use sum of lag-specific effects at lag 0 (if available)
        if pred.matfit.shape[1] > 0:
            fit_values = pred.matfit[:, 0]  # First lag
            se_values = pred.matse[:, 0]
        else:
            raise ValueError("No lag-specific effects available for MMT calculation")
    else:
        raise ValueError("method must be 'overall' or 'lagspecific'")
    
    # Find minimum
    min_idx = np.argmin(fit_values)
    mmt_value = predvar[min_idx]
    mmt_fit = fit_values[min_idx]
    mmt_se = se_values[min_idx]
    
    # Calculate confidence intervals
    z_score = 1.96  # 95% CI
    
    # Check if model uses log link (for relative risks)
    if hasattr(pred, 'model_link') and pred.model_link == 'log':
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
        arglag=basis.arglag
    )
    
    # Store original range and centering info
    recentered_basis._original_basis = basis
    recentered_basis._centering_info = centering_info
    
    return recentered_basis, centering_info


def compare_centering(basis: CrossBasis,
                      model: Any,
                      centering_values: Union[List[float], np.ndarray],
                      at: Optional[np.ndarray] = None) -> Dict:
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
                at=at
            )
            
            # Store key results
            results[f'cen_{cen_val}'] = {
                'centering': cen_val,
                'predvar': pred.predvar,
                'allfit': pred.allfit,
                'allse': pred.allse,
                'prediction_object': pred
            }
            
            # Add RR results if available
            if hasattr(pred, 'allRRfit'):
                results[f'cen_{cen_val}']['allRRfit'] = pred.allRRfit
                results[f'cen_{cen_val}']['allRRlow'] = pred.allRRlow
                results[f'cen_{cen_val}']['allRRhigh'] = pred.allRRhigh
                
        except Exception as e:
            warnings.warn(f"Failed to compute predictions for centering {cen_val}: {e}")
            results[f'cen_{cen_val}'] = None
    
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
    
    def __init__(self, basis: CrossBasis, model: Any):
        """
        Initialize centering manager
        
        Parameters:
        -----------
        basis : CrossBasis
            Cross-basis object
        model : fitted model object
            Fitted statistical model
        """
        self.basis = basis
        self.model = model
        self._mmt_cache = None
        self._centering_history = []
    
    def find_mmt(self, **kwargs) -> Dict:
        """Find MMT with caching"""
        if self._mmt_cache is None:
            self._mmt_cache = find_mmt(self.basis, self.model, **kwargs)
        return self._mmt_cache
    
    def recenter_at_mmt(self, **kwargs) -> Tuple[CrossBasis, Dict]:
        """Convenience method to recenter at MMT"""
        return recenter_basis(self.basis, self.model, cen=None, 
                            find_mmt_args=kwargs)
    
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
                mmt_info = self.find_mmt()
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
                                     centering_values, **kwargs)
        
        # Add strategy names to results
        comparison['strategy_names'] = strategy_names
        comparison['strategies'] = strategies
        
        return comparison
    
    def get_centering_history(self) -> List[Dict]:
        """Get history of centering operations"""
        return self._centering_history.copy()