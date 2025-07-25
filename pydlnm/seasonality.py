"""
Advanced seasonality modeling for PyDLNM

Implements sophisticated seasonal pattern modeling using harmonic functions,
Fourier series, and flexible seasonal splines for time series analysis
in distributed lag models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict, Any
from datetime import datetime, date
import warnings

from .basis_functions import SplineBasis, BSplineBasis
from .enhanced_splines import ns_enhanced


class SeasonalBasis:
    """
    Base class for seasonal basis functions
    """
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.attributes = {}
    
    def __call__(self, time_index: Union[np.ndarray, pd.DatetimeIndex], **kwargs) -> np.ndarray:
        """Generate seasonal basis matrix"""
        raise NotImplementedError("Subclasses must implement __call__")
    
    def get_attributes(self) -> Dict[str, Any]:
        """Get basis attributes"""
        return self.attributes.copy()


class HarmonicSeasonalBasis(SeasonalBasis):
    """
    Harmonic/Fourier seasonal basis functions
    
    Creates sine and cosine basis functions for modeling seasonal patterns
    with specified number of harmonics.
    
    Parameters:
    -----------
    n_harmonics : int, default 2
        Number of harmonic pairs (sine/cosine) to include
    period : float, default 365.25
        Period of seasonality (e.g., 365.25 for annual cycles)
    include_trend : bool, default False
        Whether to include linear trend term
    """
    
    def __init__(self, n_harmonics: int = 2, period: float = 365.25, 
                 include_trend: bool = False, **kwargs):
        super().__init__(n_harmonics=n_harmonics, period=period, 
                        include_trend=include_trend, **kwargs)
        self.n_harmonics = n_harmonics
        self.period = period
        self.include_trend = include_trend
        self.attributes['fun'] = 'harmonic'
        self.attributes['n_harmonics'] = n_harmonics
        self.attributes['period'] = period
        self.attributes['include_trend'] = include_trend
    
    def __call__(self, time_index: Union[np.ndarray, pd.DatetimeIndex], **kwargs) -> np.ndarray:
        """
        Generate harmonic seasonal basis matrix
        
        Parameters:
        -----------
        time_index : array-like or DatetimeIndex
            Time indices (can be numeric or datetime)
            
        Returns:
        --------
        np.ndarray
            Harmonic basis matrix
        """
        # Convert time index to numeric if needed
        if isinstance(time_index, pd.DatetimeIndex):
            # Convert to days since start of series
            time_numeric = (time_index - time_index[0]).days.astype(float)
        elif hasattr(time_index, 'values') and hasattr(time_index.values[0], 'timestamp'):
            # Handle pandas datetime-like objects
            time_numeric = np.array([t.timestamp() / (24 * 3600) for t in time_index])
        else:
            time_numeric = np.asarray(time_index, dtype=float)
        
        n_obs = len(time_numeric)
        
        # Calculate number of basis functions
        n_basis = 2 * self.n_harmonics + (1 if self.include_trend else 0)
        basis_matrix = np.zeros((n_obs, n_basis))
        
        col_idx = 0
        
        # Add linear trend if requested
        if self.include_trend:
            basis_matrix[:, col_idx] = time_numeric
            col_idx += 1
        
        # Add harmonic terms
        for h in range(1, self.n_harmonics + 1):
            # Angular frequency
            omega = 2 * np.pi * h / self.period
            
            # Sine component
            basis_matrix[:, col_idx] = np.sin(omega * time_numeric)
            col_idx += 1
            
            # Cosine component  
            basis_matrix[:, col_idx] = np.cos(omega * time_numeric)
            col_idx += 1
        
        self.attributes['n_basis'] = n_basis
        self.attributes['frequencies'] = [h / self.period for h in range(1, self.n_harmonics + 1)]
        
        return basis_matrix


class SeasonalSplineBasis(SeasonalBasis):
    """
    Seasonal spline basis functions
    
    Uses cyclic splines or periodic B-splines for modeling seasonal patterns
    with smooth, flexible shapes.
    
    Parameters:
    -----------
    df : int, default 4
        Degrees of freedom for seasonal spline
    period : float, default 365.25
        Period of seasonality
    cyclic : bool, default True
        Whether to enforce cyclic constraints (periodic boundary conditions)
    spline_type : str, default 'ns'
        Type of spline: 'ns' (natural), 'bs' (B-spline)
    """
    
    def __init__(self, df: int = 4, period: float = 365.25, 
                 cyclic: bool = True, spline_type: str = 'ns', **kwargs):
        super().__init__(df=df, period=period, cyclic=cyclic, 
                        spline_type=spline_type, **kwargs)
        self.df = df
        self.period = period
        self.cyclic = cyclic
        self.spline_type = spline_type
        self.attributes['fun'] = 'seasonal_spline'
        self.attributes['df'] = df
        self.attributes['period'] = period
        self.attributes['cyclic'] = cyclic
        self.attributes['spline_type'] = spline_type
    
    def __call__(self, time_index: Union[np.ndarray, pd.DatetimeIndex], **kwargs) -> np.ndarray:
        """
        Generate seasonal spline basis matrix
        
        Parameters:
        -----------
        time_index : array-like or DatetimeIndex
            Time indices
            
        Returns:
        --------
        np.ndarray
            Seasonal spline basis matrix
        """
        # Convert time index to numeric
        if isinstance(time_index, pd.DatetimeIndex):
            time_numeric = (time_index - time_index[0]).days.astype(float)
        elif hasattr(time_index, 'values') and hasattr(time_index.values[0], 'timestamp'):
            time_numeric = np.array([t.timestamp() / (24 * 3600) for t in time_index])
        else:
            time_numeric = np.asarray(time_index, dtype=float)
        
        # Convert to seasonal position (0 to period)
        seasonal_position = time_numeric % self.period
        
        # Generate spline basis
        if self.spline_type == 'ns':
            # Natural spline
            basis_matrix, spline_attrs = ns_enhanced(
                seasonal_position, 
                df=self.df,
                intercept=False
            )
        elif self.spline_type == 'bs':
            # B-spline
            from .enhanced_splines import bs_enhanced
            basis_matrix, spline_attrs = bs_enhanced(
                seasonal_position,
                df=self.df,
                degree=3,
                intercept=False
            )
        else:
            raise ValueError("spline_type must be 'ns' or 'bs'")
        
        # Apply cyclic constraints if requested
        if self.cyclic:
            basis_matrix = self._apply_cyclic_constraints(basis_matrix, seasonal_position)
        
        self.attributes.update(spline_attrs)
        self.attributes['n_basis'] = basis_matrix.shape[1]
        
        return basis_matrix
    
    def _apply_cyclic_constraints(self, basis_matrix: np.ndarray, 
                                 seasonal_position: np.ndarray) -> np.ndarray:
        """Apply cyclic constraints to ensure periodicity"""
        
        # Simple approach: modify basis at boundaries to enforce continuity
        # This is a simplified implementation - full cyclic splines are more complex
        
        # Find observations near boundaries
        boundary_width = self.period * 0.05  # 5% of period
        near_start = seasonal_position < boundary_width
        near_end = seasonal_position > (self.period - boundary_width)
        
        if np.any(near_start) and np.any(near_end):
            # Average basis values at start and end to enforce continuity
            start_mean = np.mean(basis_matrix[near_start], axis=0)
            end_mean = np.mean(basis_matrix[near_end], axis=0)
            boundary_mean = (start_mean + end_mean) / 2
            
            # Apply smooth transition
            for i in range(len(seasonal_position)):
                if near_start[i]:
                    weight = 1 - (seasonal_position[i] / boundary_width)
                    basis_matrix[i] = weight * boundary_mean + (1 - weight) * basis_matrix[i]
                elif near_end[i]:
                    weight = (seasonal_position[i] - (self.period - boundary_width)) / boundary_width
                    basis_matrix[i] = weight * boundary_mean + (1 - weight) * basis_matrix[i]
        
        return basis_matrix


class FlexibleSeasonalBasis(SeasonalBasis):
    """
    Flexible seasonal basis combining multiple approaches
    
    Allows combination of harmonic and spline components for complex
    seasonal patterns.
    
    Parameters:
    -----------
    components : list of dict
        List of component specifications, e.g.:
        [{'type': 'harmonic', 'n_harmonics': 2},
         {'type': 'spline', 'df': 6}]
    period : float, default 365.25
        Period of seasonality
    """
    
    def __init__(self, components: List[Dict], period: float = 365.25, **kwargs):
        super().__init__(components=components, period=period, **kwargs)
        self.components = components
        self.period = period
        self.attributes['fun'] = 'flexible_seasonal'
        self.attributes['components'] = components
        self.attributes['period'] = period
        
        # Initialize component basis functions
        self.basis_functions = []
        for comp in components:
            if comp['type'] == 'harmonic':
                basis_func = HarmonicSeasonalBasis(
                    n_harmonics=comp.get('n_harmonics', 2),
                    period=period,
                    include_trend=comp.get('include_trend', False)
                )
            elif comp['type'] == 'spline':
                basis_func = SeasonalSplineBasis(
                    df=comp.get('df', 4),
                    period=period,
                    cyclic=comp.get('cyclic', True),
                    spline_type=comp.get('spline_type', 'ns')
                )
            else:
                raise ValueError(f"Unknown component type: {comp['type']}")
            
            self.basis_functions.append(basis_func)
    
    def __call__(self, time_index: Union[np.ndarray, pd.DatetimeIndex], **kwargs) -> np.ndarray:
        """
        Generate flexible seasonal basis matrix
        
        Parameters:
        -----------
        time_index : array-like or DatetimeIndex
            Time indices
            
        Returns:
        --------
        np.ndarray
            Combined seasonal basis matrix
        """
        # Generate basis for each component
        component_bases = []
        total_basis_functions = 0
        
        for basis_func in self.basis_functions:
            comp_basis = basis_func(time_index)
            component_bases.append(comp_basis)
            total_basis_functions += comp_basis.shape[1]
        
        # Combine all components
        basis_matrix = np.column_stack(component_bases)
        
        self.attributes['n_basis'] = total_basis_functions
        self.attributes['component_dimensions'] = [b.shape[1] for b in component_bases]
        
        return basis_matrix


def create_seasonal_basis(time_index: Union[np.ndarray, pd.DatetimeIndex],
                         method: str = "harmonic",
                         **kwargs) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to create seasonal basis
    
    Parameters:
    -----------
    time_index : array-like or DatetimeIndex
        Time indices
    method : str, default "harmonic"
        Method: "harmonic", "spline", "flexible"
    **kwargs
        Method-specific parameters
        
    Returns:
    --------
    tuple
        - basis_matrix: Seasonal basis matrix
        - attributes: Basis attributes
    """
    
    if method == "harmonic":
        basis_func = HarmonicSeasonalBasis(**kwargs)
    elif method == "spline":
        basis_func = SeasonalSplineBasis(**kwargs)
    elif method == "flexible":
        basis_func = FlexibleSeasonalBasis(**kwargs)
    else:
        raise ValueError("method must be 'harmonic', 'spline', or 'flexible'")
    
    basis_matrix = basis_func(time_index)
    attributes = basis_func.get_attributes()
    
    return basis_matrix, attributes


def seasonal_decomposition(time_series: np.ndarray,
                          time_index: Union[np.ndarray, pd.DatetimeIndex],
                          method: str = "harmonic",
                          return_components: bool = False,
                          **kwargs) -> Dict:
    """
    Perform seasonal decomposition of time series
    
    Parameters:
    -----------
    time_series : array-like
        Time series values to decompose
    time_index : array-like or DatetimeIndex
        Time indices
    method : str, default "harmonic"
        Seasonal basis method
    return_components : bool, default False
        Whether to return individual seasonal components
    **kwargs
        Method-specific parameters
        
    Returns:
    --------
    dict
        Decomposition results including trend, seasonal, and residual components
    """
    
    from sklearn.linear_model import LinearRegression
    
    time_series = np.asarray(time_series)
    valid_mask = ~np.isnan(time_series)
    
    if not np.any(valid_mask):
        raise ValueError("No valid observations in time series")
    
    # Create seasonal basis
    seasonal_basis, attributes = create_seasonal_basis(time_index[valid_mask], method, **kwargs)
    
    # Add trend component
    if isinstance(time_index, pd.DatetimeIndex):
        time_numeric = (time_index[valid_mask] - time_index[0]).days.astype(float)
    else:
        time_numeric = np.asarray(time_index[valid_mask], dtype=float)
    
    # Design matrix: trend + seasonal
    X = np.column_stack([time_numeric, seasonal_basis])
    
    # Fit model
    model = LinearRegression()
    model.fit(X, time_series[valid_mask])
    
    # Extract components
    fitted = model.predict(X)
    trend = model.coef_[0] * time_numeric + model.intercept_
    seasonal = fitted - trend
    residual = time_series[valid_mask] - fitted
    
    results = {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual,
        'fitted': fitted,
        'model': model,
        'seasonal_basis': seasonal_basis,
        'attributes': attributes,
        'r_squared': model.score(X, time_series[valid_mask])
    }
    
    if return_components and method == "flexible":
        # Decompose seasonal component by basis type
        results['seasonal_components'] = _decompose_seasonal_components(
            model.coef_[1:], seasonal_basis, attributes
        )
    
    return results


def _decompose_seasonal_components(seasonal_coef: np.ndarray,
                                  seasonal_basis: np.ndarray,
                                  attributes: Dict) -> Dict:
    """Helper function to decompose seasonal components"""
    
    components = {}
    
    if 'component_dimensions' in attributes:
        # Flexible seasonal basis
        start_idx = 0
        for i, comp_dim in enumerate(attributes['component_dimensions']):
            end_idx = start_idx + comp_dim
            comp_coef = seasonal_coef[start_idx:end_idx]
            comp_basis = seasonal_basis[:, start_idx:end_idx]
            comp_effect = comp_basis @ comp_coef
            
            comp_type = attributes['components'][i]['type']
            components[f'{comp_type}_{i}'] = comp_effect
            
            start_idx = end_idx
    
    return components


class SeasonalityManager:
    """
    Manager class for seasonal analysis in DLNM contexts
    """
    
    def __init__(self, time_index: Union[np.ndarray, pd.DatetimeIndex]):
        """
        Initialize seasonality manager
        
        Parameters:
        -----------
        time_index : array-like or DatetimeIndex
            Time indices for the study period
        """
        self.time_index = time_index
        self._seasonal_cache = {}
    
    def create_basis(self, method: str = "harmonic", **kwargs) -> Tuple[np.ndarray, Dict]:
        """Create seasonal basis with caching"""
        cache_key = f"{method}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key not in self._seasonal_cache:
            basis_matrix, attributes = create_seasonal_basis(
                self.time_index, method, **kwargs
            )
            self._seasonal_cache[cache_key] = (basis_matrix, attributes)
        
        return self._seasonal_cache[cache_key]
    
    def compare_methods(self, time_series: np.ndarray,
                       methods: List[str] = None,
                       **method_kwargs) -> Dict:
        """
        Compare different seasonal modeling approaches
        
        Parameters:
        -----------
        time_series : array-like
            Time series to analyze
        methods : list, optional
            Methods to compare. Default: ['harmonic', 'spline']
        **method_kwargs
            Method-specific keyword arguments
            
        Returns:
        --------
        dict
            Comparison results for each method
        """
        
        if methods is None:
            methods = ['harmonic', 'spline']
        
        results = {}
        
        for method in methods:
            try:
                # Get method-specific kwargs
                method_params = method_kwargs.get(method, {})
                
                # Perform decomposition
                decomp_result = seasonal_decomposition(
                    time_series, self.time_index, method, **method_params
                )
                
                results[method] = {
                    'decomposition': decomp_result,
                    'r_squared': decomp_result['r_squared'],
                    'residual_std': np.std(decomp_result['residual']),
                    'n_parameters': decomp_result['seasonal_basis'].shape[1] + 1,  # +1 for trend
                    'method': method
                }
                
                # Add AIC/BIC if possible
                n_obs = len(decomp_result['residual'])
                n_params = results[method]['n_parameters']
                mse = np.mean(decomp_result['residual']**2)
                
                results[method]['aic'] = n_obs * np.log(mse) + 2 * n_params
                results[method]['bic'] = n_obs * np.log(mse) + np.log(n_obs) * n_params
                
            except Exception as e:
                warnings.warn(f"Failed to fit {method}: {e}")
                results[method] = None
        
        # Add comparison summary
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if valid_results:
            best_r2 = max(valid_results.keys(), key=lambda k: valid_results[k]['r_squared'])
            best_aic = min(valid_results.keys(), key=lambda k: valid_results[k]['aic'])
            
            results['summary'] = {
                'best_r_squared': best_r2,
                'best_aic': best_aic,
                'n_methods_tested': len(valid_results)
            }
        
        return results
    
    def get_optimal_seasonality(self, time_series: np.ndarray,
                               criterion: str = "aic") -> Dict:
        """
        Find optimal seasonality specification
        
        Parameters:
        -----------
        time_series : array-like
            Time series to analyze
        criterion : str, default "aic"
            Selection criterion: "aic", "bic", "r_squared"
            
        Returns:
        --------
        dict
            Optimal seasonality specification and results
        """
        
        # Test different configurations
        configurations = [
            {'method': 'harmonic', 'n_harmonics': 1},
            {'method': 'harmonic', 'n_harmonics': 2},
            {'method': 'harmonic', 'n_harmonics': 3},
            {'method': 'spline', 'df': 4},
            {'method': 'spline', 'df': 6},
            {'method': 'spline', 'df': 8},
        ]
        
        results = {}
        
        for i, config in enumerate(configurations):
            config_name = f"{config['method']}_{i}"
            
            try:
                decomp = seasonal_decomposition(time_series, self.time_index, **config)
                
                n_obs = len(decomp['residual'])
                n_params = decomp['seasonal_basis'].shape[1] + 1
                mse = np.mean(decomp['residual']**2)
                
                results[config_name] = {
                    'config': config,
                    'decomposition': decomp,
                    'r_squared': decomp['r_squared'],
                    'aic': n_obs * np.log(mse) + 2 * n_params,
                    'bic': n_obs * np.log(mse) + np.log(n_obs) * n_params,
                    'residual_std': np.std(decomp['residual'])
                }
                
            except Exception as e:
                warnings.warn(f"Failed configuration {config}: {e}")
                continue
        
        if not results:
            raise ValueError("No valid seasonality configurations found")
        
        # Select best configuration
        if criterion == "r_squared":
            best_config = max(results.keys(), key=lambda k: results[k]['r_squared'])
        elif criterion == "aic":
            best_config = min(results.keys(), key=lambda k: results[k]['aic'])
        elif criterion == "bic":
            best_config = min(results.keys(), key=lambda k: results[k]['bic'])
        else:
            raise ValueError("criterion must be 'aic', 'bic', or 'r_squared'")
        
        return {
            'optimal_config': results[best_config]['config'],
            'optimal_results': results[best_config],
            'all_results': results,
            'selection_criterion': criterion
        }