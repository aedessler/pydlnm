"""
Attribution functions for PyDLNM

Implements attributable risk calculations with confidence intervals,
equivalent to R's attrdl function for computing heat/cold-related mortality
and other attributable measures in distributed lag models.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union, List, Dict, Tuple, Any
import warnings

from .basis import CrossBasis
from .prediction import CrossPred
from .centering import find_mmt, find_mmt_blup


def attrdl_proper(x: np.ndarray,
                  basis_matrix: np.ndarray,
                  cases: np.ndarray,
                  coef: np.ndarray,
                  vcov: Optional[np.ndarray] = None,
                  cen: float = None,
                  range_vals: Optional[Tuple[float, float]] = None,
                  type: str = "an") -> float:
    """
    Calculate attributable risk using proper R-style methodology
    
    This implements the core calculation from R's attrdl function, using
    cross-basis matrix predictions and relative risk calculations.
    
    Parameters:
    -----------
    x : array-like
        Exposure time series
    basis_matrix : array-like
        Cross-basis matrix (n_obs x n_basis)
    cases : array-like
        Observed cases (deaths) time series
    coef : array-like
        Model coefficients for basis functions
    vcov : array-like, optional
        Variance-covariance matrix
    cen : float
        Centering value (MMT)
    range_vals : tuple, optional
        Range of exposure values for attribution (min, max)
    type : str
        Type: "an" (attributable numbers) or "af" (attributable fraction)
        
    Returns:
    --------
    float
        Attributable deaths/fraction
    """
    
    # Remove missing values
    valid_mask = ~(np.isnan(x) | np.isnan(cases))
    x_valid = x[valid_mask]
    cases_valid = cases[valid_mask]
    basis_valid = basis_matrix[valid_mask]
    
    if len(x_valid) == 0:
        return 0.0
    
    # Calculate predicted log-relative risks from basis matrix
    log_rr = basis_valid @ coef
    
    # Convert to relative risks
    rr = np.exp(log_rr)
    
    # Apply range filter if specified
    if range_vals is not None:
        range_mask = (x_valid >= range_vals[0]) & (x_valid <= range_vals[1])
    else:
        range_mask = np.ones(len(x_valid), dtype=bool)
    
    # Calculate attributable fraction for each observation
    # AF = (RR - 1) / RR = 1 - 1/RR
    af_obs = 1 - (1 / rr)
    
    # Apply range mask
    af_obs = af_obs * range_mask
    
    if type == "an":
        # Attributable numbers: sum of AF * observed cases
        attributable = np.sum(af_obs * cases_valid)
    elif type == "af":
        # Attributable fraction: weighted average
        total_cases = np.sum(cases_valid[range_mask])
        if total_cases > 0:
            attributable_cases = np.sum(af_obs * cases_valid)
            attributable = attributable_cases / total_cases
        else:
            attributable = 0.0
    else:
        raise ValueError("type must be 'an' or 'af'")
    
    return attributable


def attrdl(x: np.ndarray,
           basis: CrossBasis,
           cases: np.ndarray,
           model: Optional[Any] = None,
           coef: Optional[np.ndarray] = None,
           vcov: Optional[np.ndarray] = None,
           type: str = "an",
           dir: str = "forw",
           tot: bool = True,
           cen: Optional[float] = None,
           range: Optional[Tuple[float, float]] = None,
           sim: bool = False,
           nsim: int = 5000,
           sub: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate attributable risk from distributed lag models
    
    Equivalent to R's attrdl function for computing attributable deaths/cases
    with confidence intervals using empirical methods.
    
    Parameters:
    -----------
    x : array-like
        Exposure time series
    basis : CrossBasis
        Cross-basis object used in the model
    cases : array-like
        Observed cases/deaths time series
    model : fitted model object, optional
        Fitted statistical model
    coef : array-like, optional
        Model coefficients (if model not provided)
    vcov : array-like, optional
        Variance-covariance matrix (if model not provided)
    type : str, default "an"
        Type of attributable risk: "an" (attributable number), "af" (attributable fraction)
    dir : str, default "forw"
        Direction: "forw" (forward), "back" (backward)
    tot : bool, default True
        Whether to compute total attributable risk
    cen : float, optional
        Centering value for risk attribution
    range : tuple, optional
        Range of exposure values for attribution (min, max)
    sim : bool, default False
        Whether to use simulation for confidence intervals
    nsim : int, default 5000
        Number of simulations for confidence intervals
    sub : array-like, optional
        Subset of observations to use
        
    Returns:
    --------
    dict
        Dictionary containing attribution results:
        - an: attributable numbers (if type="an")
        - af: attributable fractions (if type="af")
        - totals: total attributable measures
        - ci: confidence intervals (if sim=True)
        - sim_results: simulation results (if sim=True)
    """
    
    # Input validation
    x = np.asarray(x)
    cases = np.asarray(cases)
    
    if len(x) != len(cases):
        raise ValueError("x and cases must have the same length")
    
    if sub is not None:
        sub = np.asarray(sub, dtype=bool)
        if len(sub) != len(x):
            raise ValueError("sub must have the same length as x")
        x = x[sub]
        cases = cases[sub]
    
    # Remove missing values
    valid_mask = ~(np.isnan(x) | np.isnan(cases))
    x = x[valid_mask]
    cases = cases[valid_mask]
    
    if len(x) == 0:
        raise ValueError("No valid observations after removing missing values")
    
    # Handle centering
    if cen is None:
        # Try to find MMT automatically
        try:
            mmt_result = find_mmt(basis, model, coef=coef, vcov=vcov)
            cen = mmt_result['mmt']
        except Exception:
            # Fallback to mean centering
            cen = np.mean(x)
            warnings.warn("Could not find MMT, using mean centering")
    
    # Create prediction object
    pred = CrossPred(
        basis=basis,
        model=model,
        coef=coef,
        vcov=vcov,
        at=x,
        cen=cen
    )
    
    # Get relative risks
    if hasattr(pred, 'allRRfit'):
        rr = pred.allRRfit
    else:
        # Convert from log scale if necessary
        if hasattr(pred, 'model_link') and pred.model_link == 'log':
            rr = np.exp(pred.allfit)
        else:
            # Assume already on RR scale or identity link
            rr = pred.allfit
    
    # Apply range filtering if specified
    if range is not None:
        range_mask = (x >= range[0]) & (x <= range[1])
    else:
        range_mask = np.ones(len(x), dtype=bool)
    
    # Calculate attributable measures
    results = {}
    
    if type in ["an", "both"]:
        # Attributable numbers
        an = _calculate_attributable_numbers(cases, rr, range_mask, dir)
        results['an'] = an
        
        if tot:
            results['an_total'] = np.sum(an)
    
    if type in ["af", "both"]:
        # Attributable fractions
        af = _calculate_attributable_fractions(rr, range_mask, dir)
        results['af'] = af
        
        if tot:
            # Total attributable fraction
            total_cases = np.sum(cases[range_mask])
            total_an = np.sum(_calculate_attributable_numbers(cases, rr, range_mask, dir))
            results['af_total'] = total_an / total_cases if total_cases > 0 else 0.0
    
    # Add metadata
    results['metadata'] = {
        'type': type,
        'direction': dir,
        'centering': cen,
        'range': range,
        'n_obs': len(x),
        'n_cases': np.sum(cases),
        'simulation': sim
    }
    
    # Compute confidence intervals via simulation if requested
    if sim:
        sim_results = _simulate_attribution_ci(
            x, basis, cases, model, coef, vcov, 
            type, dir, tot, cen, range, nsim
        )
        results['ci'] = sim_results['ci']
        results['sim_results'] = sim_results
    
    return results


def _calculate_attributable_numbers(cases: np.ndarray, 
                                   rr: np.ndarray, 
                                   range_mask: np.ndarray,
                                   dir: str) -> np.ndarray:
    """Calculate attributable numbers"""
    
    if dir == "forw":
        # Forward: AN = cases * (RR - 1) / RR
        an = cases * (rr - 1) / rr
    elif dir == "back":
        # Backward: AN = cases * (1 - 1/RR)
        an = cases * (1 - 1/rr)
    else:
        raise ValueError("dir must be 'forw' or 'back'")
    
    # Apply range mask
    an = an * range_mask
    
    return an


def _calculate_attributable_fractions(rr: np.ndarray,
                                     range_mask: np.ndarray,
                                     dir: str) -> np.ndarray:
    """Calculate attributable fractions"""
    
    if dir == "forw":
        # Forward: AF = (RR - 1) / RR
        af = (rr - 1) / rr
    elif dir == "back":
        # Backward: AF = 1 - 1/RR
        af = 1 - 1/rr
    else:
        raise ValueError("dir must be 'forw' or 'back'")
    
    # Apply range mask
    af = af * range_mask
    
    return af


def _simulate_attribution_ci(x: np.ndarray,
                            basis: CrossBasis,
                            cases: np.ndarray,
                            model: Optional[Any],
                            coef: Optional[np.ndarray],
                            vcov: Optional[np.ndarray],
                            type: str,
                            dir: str,
                            tot: bool,
                            cen: Optional[float],
                            range_spec: Optional[Tuple[float, float]],
                            nsim: int) -> Dict:
    """Simulate confidence intervals for attribution measures"""
    
    # Get model coefficients and covariance
    if model is not None:
        from .model_utils import validate_model_compatibility
        model_info = validate_model_compatibility(model, basis.shape[1], "CrossBasis")
        coef_vec = model_info['coef']
        vcov_mat = model_info['vcov']
    else:
        coef_vec = np.asarray(coef)
        vcov_mat = np.asarray(vcov)
    
    # Trim to basis size
    basis_ncol = basis.shape[1]
    coef_vec = coef_vec[:basis_ncol]
    vcov_mat = vcov_mat[:basis_ncol, :basis_ncol]
    
    # Generate coefficient samples
    try:
        coef_samples = np.random.multivariate_normal(coef_vec, vcov_mat, nsim)
    except np.linalg.LinAlgError:
        # If covariance matrix is singular, add small regularization
        reg_vcov = vcov_mat + np.eye(vcov_mat.shape[0]) * 1e-8
        coef_samples = np.random.multivariate_normal(coef_vec, reg_vcov, nsim)
    
    # Storage for simulation results
    sim_an = []
    sim_af = []
    sim_an_total = []
    sim_af_total = []
    
    for i in range(nsim):
        try:
            # Create prediction with sampled coefficients
            pred = CrossPred(
                basis=basis,
                coef=coef_samples[i],
                vcov=vcov_mat,  # Use original vcov for predictions
                at=x,
                cen=cen
            )
            
            # Get relative risks
            if hasattr(pred, 'allRRfit'):
                rr = pred.allRRfit
            else:
                # Convert from log scale
                rr = np.exp(pred.allfit)
            
            # Apply range filtering
            if range_spec is not None:
                range_mask = (x >= range_spec[0]) & (x <= range_spec[1])
            else:
                range_mask = np.ones(len(x), dtype=bool)
            
            # Calculate measures for this simulation
            if type in ["an", "both"]:
                an_sim = _calculate_attributable_numbers(cases, rr, range_mask, dir)
                sim_an.append(an_sim)
                if tot:
                    sim_an_total.append(np.sum(an_sim))
            
            if type in ["af", "both"]:
                af_sim = _calculate_attributable_fractions(rr, range_mask, dir)
                sim_af.append(af_sim)
                if tot:
                    total_cases = np.sum(cases[range_mask])
                    total_an_sim = np.sum(_calculate_attributable_numbers(cases, rr, range_mask, dir))
                    sim_af_total.append(total_an_sim / total_cases if total_cases > 0 else 0.0)
                    
        except Exception as e:
            warnings.warn(f"Simulation {i} failed: {e}")
            continue
    
    # Calculate confidence intervals
    ci = {}
    sim_results = {}
    
    if sim_an:
        sim_an = np.array(sim_an)
        sim_results['an'] = sim_an
        ci['an_low'] = np.percentile(sim_an, 2.5, axis=0)
        ci['an_high'] = np.percentile(sim_an, 97.5, axis=0)
        
        if sim_an_total:
            sim_results['an_total'] = np.array(sim_an_total)
            ci['an_total_low'] = np.percentile(sim_an_total, 2.5)
            ci['an_total_high'] = np.percentile(sim_an_total, 97.5)
    
    if sim_af:
        sim_af = np.array(sim_af)
        sim_results['af'] = sim_af
        ci['af_low'] = np.percentile(sim_af, 2.5, axis=0)
        ci['af_high'] = np.percentile(sim_af, 97.5, axis=0)
        
        if sim_af_total:
            sim_results['af_total'] = np.array(sim_af_total)
            ci['af_total_low'] = np.percentile(sim_af_total, 2.5)
            ci['af_total_high'] = np.percentile(sim_af_total, 97.5)
    
    return {'ci': ci, 'simulations': sim_results, 'nsim_successful': len(sim_an) if sim_an else len(sim_af)}


def attr_heat_cold(x: np.ndarray,
                   basis: CrossBasis,
                   cases: np.ndarray,
                   model: Optional[Any] = None,
                   coef: Optional[np.ndarray] = None,
                   vcov: Optional[np.ndarray] = None,
                   percentiles: Tuple[float, float] = (2.5, 97.5),
                   cen: Optional[float] = None,
                   sim: bool = False,
                   nsim: int = 5000) -> Dict:
    """
    Calculate heat and cold attributable risks
    
    Convenience function for computing temperature-related mortality
    split into heat and cold effects based on percentile thresholds.
    
    Parameters:
    -----------
    x : array-like
        Temperature time series
    basis : CrossBasis
        Cross-basis object
    cases : array-like
        Cases/deaths time series  
    model : fitted model object, optional
        Fitted statistical model
    coef : array-like, optional
        Model coefficients
    vcov : array-like, optional
        Variance-covariance matrix
    percentiles : tuple, default (2.5, 97.5)
        Percentiles defining cold and heat thresholds
    cen : float, optional
        Centering temperature (MMT if None)
    sim : bool, default False
        Use simulation for confidence intervals
    nsim : int, default 5000
        Number of simulations
        
    Returns:
    --------
    dict
        Results containing heat and cold attribution
    """
    
    x = np.asarray(x)
    
    # Calculate thresholds
    cold_threshold = np.percentile(x, percentiles[0])
    heat_threshold = np.percentile(x, percentiles[1])
    
    # Cold attribution (below percentile[0])
    cold_results = attrdl(
        x, basis, cases, model, coef, vcov,
        type="both", range=(float('-inf'), cold_threshold),
        cen=cen, sim=sim, nsim=nsim
    )
    
    # Heat attribution (above percentile[1])  
    heat_results = attrdl(
        x, basis, cases, model, coef, vcov,
        type="both", range=(heat_threshold, float('inf')),
        cen=cen, sim=sim, nsim=nsim
    )
    
    # Combine results
    results = {
        'cold': {
            'threshold': cold_threshold,
            'percentile': percentiles[0],
            'results': cold_results
        },
        'heat': {
            'threshold': heat_threshold, 
            'percentile': percentiles[1],
            'results': heat_results
        },
        'summary': {
            'cold_an_total': cold_results.get('an_total', 0),
            'heat_an_total': heat_results.get('an_total', 0),
            'cold_af_total': cold_results.get('af_total', 0),
            'heat_af_total': heat_results.get('af_total', 0),
            'total_attributable': cold_results.get('an_total', 0) + heat_results.get('an_total', 0)
        },
        'metadata': {
            'percentiles': percentiles,
            'centering': cen,
            'simulation': sim,
            'n_obs': len(x)
        }
    }
    
    return results


def attr_by_percentiles(x: np.ndarray,
                       basis: CrossBasis,
                       cases: np.ndarray,
                       model: Optional[Any] = None,
                       coef: Optional[np.ndarray] = None,
                       vcov: Optional[np.ndarray] = None,
                       percentile_ranges: List[Tuple[float, float]] = None,
                       cen: Optional[float] = None,
                       sim: bool = False,
                       nsim: int = 5000) -> Dict:
    """
    Calculate attributable risks by percentile ranges
    
    Parameters:
    -----------
    x : array-like
        Exposure time series
    basis : CrossBasis
        Cross-basis object
    cases : array-like
        Cases time series
    model : fitted model object, optional
        Fitted statistical model
    coef, vcov : array-like, optional
        Model parameters
    percentile_ranges : list of tuples, optional
        List of (low_pct, high_pct) ranges to analyze
        Default: [(0, 1), (1, 5), (5, 10), (90, 95), (95, 99), (99, 100)]
    cen : float, optional
        Centering value
    sim : bool, default False
        Use simulation for CI
    nsim : int, default 5000
        Number of simulations
        
    Returns:
    --------
    dict
        Attribution by percentile ranges
    """
    
    if percentile_ranges is None:
        percentile_ranges = [(0, 1), (1, 5), (5, 10), (90, 95), (95, 99), (99, 100)]
    
    x = np.asarray(x)
    results = {}
    
    for i, (low_pct, high_pct) in enumerate(percentile_ranges):
        # Calculate thresholds
        low_threshold = np.percentile(x, low_pct)
        high_threshold = np.percentile(x, high_pct)
        
        # Calculate attribution for this range
        range_results = attrdl(
            x, basis, cases, model, coef, vcov,
            type="both", range=(low_threshold, high_threshold),
            cen=cen, sim=sim, nsim=nsim
        )
        
        range_name = f"pct_{low_pct}_{high_pct}"
        results[range_name] = {
            'percentiles': (low_pct, high_pct),
            'thresholds': (low_threshold, high_threshold),
            'results': range_results
        }
    
    # Create summary
    summary = pd.DataFrame({
        'percentile_range': [f"{p[0]}-{p[1]}" for p in percentile_ranges],
        'low_threshold': [results[f"pct_{p[0]}_{p[1]}"]['thresholds'][0] for p in percentile_ranges],
        'high_threshold': [results[f"pct_{p[0]}_{p[1]}"]['thresholds'][1] for p in percentile_ranges],
        'an_total': [results[f"pct_{p[0]}_{p[1]}"]['results'].get('an_total', 0) for p in percentile_ranges],
        'af_total': [results[f"pct_{p[0]}_{p[1]}"]['results'].get('af_total', 0) for p in percentile_ranges]
    })
    
    if sim:
        summary['an_total_low'] = [results[f"pct_{p[0]}_{p[1]}"]['results']['ci'].get('an_total_low', np.nan) for p in percentile_ranges]
        summary['an_total_high'] = [results[f"pct_{p[0]}_{p[1]}"]['results']['ci'].get('an_total_high', np.nan) for p in percentile_ranges]
    
    results['summary_table'] = summary
    results['metadata'] = {
        'percentile_ranges': percentile_ranges,
        'centering': cen,
        'simulation': sim,
        'n_ranges': len(percentile_ranges)
    }
    
    return results


class AttributionManager:
    """
    Manager class for attribution calculations
    """
    
    def __init__(self, x: np.ndarray, basis: CrossBasis, cases: np.ndarray,
                 model: Optional[Any] = None, coef: Optional[np.ndarray] = None,
                 vcov: Optional[np.ndarray] = None):
        """Initialize attribution manager"""
        self.x = np.asarray(x)
        self.basis = basis
        self.cases = np.asarray(cases)
        self.model = model
        self.coef = coef
        self.vcov = vcov
        
        # Cache for results
        self._mmt_cache = None
        self._attribution_cache = {}
    
    def get_mmt(self) -> float:
        """Get MMT with caching"""
        if self._mmt_cache is None:
            mmt_result = find_mmt(self.basis, self.model, coef=self.coef, vcov=self.vcov)
            self._mmt_cache = mmt_result['mmt']
        return self._mmt_cache
    
    def total_attribution(self, **kwargs) -> Dict:
        """Calculate total attributable risk"""
        return attrdl(self.x, self.basis, self.cases, 
                     self.model, self.coef, self.vcov, **kwargs)
    
    def heat_cold_attribution(self, **kwargs) -> Dict:
        """Calculate heat and cold attribution"""
        return attr_heat_cold(self.x, self.basis, self.cases,
                             self.model, self.coef, self.vcov, **kwargs)
    
    def percentile_attribution(self, **kwargs) -> Dict:
        """Calculate attribution by percentiles"""
        return attr_by_percentiles(self.x, self.basis, self.cases,
                                  self.model, self.coef, self.vcov, **kwargs)
    
    def summary_report(self, sim: bool = True, nsim: int = 1000) -> Dict:
        """Generate comprehensive attribution report"""
        
        # Get MMT
        mmt = self.get_mmt()
        
        # Total attribution
        total_attr = self.total_attribution(cen=mmt, sim=sim, nsim=nsim)
        
        # Heat/cold attribution
        heat_cold_attr = self.heat_cold_attribution(cen=mmt, sim=sim, nsim=nsim)
        
        # Percentile attribution
        pct_attr = self.percentile_attribution(cen=mmt, sim=sim, nsim=nsim)
        
        report = {
            'mmt': mmt,
            'total': total_attr,
            'heat_cold': heat_cold_attr,
            'percentiles': pct_attr,
            'summary_statistics': {
                'total_cases': np.sum(self.cases),
                'mean_exposure': np.mean(self.x),
                'exposure_range': (np.min(self.x), np.max(self.x)),
                'n_observations': len(self.x)
            }
        }
        
        return report