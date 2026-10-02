"""
Multi-location DLNM analysis with meta-analysis

This module provides functions to integrate meta-analysis functionality 
with the existing enhanced DLNM workflow for multi-location studies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings

from improved_glm import fit_enhanced_dlnm_model
from meta_analysis import mvmeta, blup
from basis import CrossBasis, OneBasis
from centering import find_mmt_blup
from utils import asfloat


def _mmt_from_blup(x: np.ndarray, blup_coef: np.ndarray, argvar: Dict[str, Any],
                   percentile_range: Tuple[int, int] = (1, 99)) -> Dict[str, Any]:
    """
    Minimum mortality temperature of a region from its BLUP (02.secondstage.R, "RE-CENTERING").
    
    R rebuilds the variable basis of the first stage on the percentiles of the region's exposure,
    ``bvar <- onebasis(quantile(x, 1:99/100), fun = varfun, knots = quantile(x, varper/100), degree = vardegree,
    Boundary.knots = range(x))``, and takes ``minperccity <- (1:99)[which.min(bvar %*% blup)]``. Here the basis is
    built from the resolved specification of the first-stage variable basis (``CrossBasis.argvar``: function, knots,
    degree, boundary knots, intercept), so that the BLUP coefficients are applied to the very basis they were
    estimated for, whatever its function and number of columns.
    
    Raises
    ------
    ValueError
        If the number of BLUP coefficients differs from the number of columns of the basis (R: non-conformable
        arguments), or the risks are all NaN (R: ``which.min`` gives ``integer(0)``).
    """
    x = asfloat(x).ravel()
    x = x[~np.isnan(x)]
    blup_coef = asfloat(blup_coef).ravel()
    
    percentiles = np.arange(percentile_range[0], percentile_range[1] + 1)
    predvar = np.percentile(x, percentiles)                     # R: quantile(x, 1:99/100), type 7
    
    spec = {key: value for key, value in argvar.items() if key != 'cen'}
    bvar = OneBasis(predvar, **spec).basis
    if bvar.shape[1] != len(blup_coef):
        raise ValueError(f"{len(blup_coef)} BLUP coefficients do not match the {bvar.shape[1]} columns of the "
                         f"first-stage variable basis")
    
    risk_values = bvar @ blup_coef
    if np.isnan(risk_values).all():
        raise ValueError("no MMT: the risk values are all NaN (NaN in the BLUP coefficients?)")
    min_idx = int(np.nanargmin(risk_values))                      # R: which.min
    
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


class MultiLocationDLNM:
    """
    Multi-location DLNM analysis manager
    
    This class coordinates the full workflow:
    1. Individual region DLNM modeling  
    2. Meta-analysis of region-specific results
    3. BLUP calculation for pooled estimates
    4. Country/pooled MMT calculation
    """
    
    def __init__(self):
        self.region_results = []
        self.region_names = []
        self.meta_predictors = {}
        self.mv_model = None
        self.blup_results = None
        self.pooled_mmts = {}
        
    def add_region_analysis(self, 
                           region_name: str,
                           crossbasis: CrossBasis,
                           y: np.ndarray,
                           dates: pd.Series,
                           dfseas: int = 8,
                           family: str = 'quasipoisson') -> Dict[str, Any]:
        """
        Analyze a single region and add to multi-location study
        
        Parameters
        ----------
        region_name : str
            Name/identifier for this region; must be unique within the study (results and meta-predictors are
            reported by name)
        crossbasis : CrossBasis
            Cross-basis matrix for this region. Its variable basis (function, knots, degree) is what the BLUP
            coefficients refer to, and is reused to find the region's MMT.
        y : array-like
            Response variable (mortality counts): list, array, Series, NaN allowed
        dates : pd.Series
            Date series for seasonality (Series, DatetimeIndex, datetime64 array, list of dates; numbers such as R
            Date day numbers are rejected with a ValueError, see ``ImprovedGLMInterface.fit_dlnm_model``)
        dfseas : int, default=8
            Seasonal degrees of freedom per year
        family : str, default='quasipoisson'
            GLM family
            
        Returns
        -------
        dict
            Individual region analysis results
        
        Raises
        ------
        ValueError
            If ``region_name`` was already added
        """
        
        if region_name in self.region_names:
            raise ValueError(f"region {region_name!r} was already added: region names must be unique")
        
        # Fit enhanced DLNM model for this region
        result = fit_enhanced_dlnm_model(
            crossbasis=crossbasis,
            y=y,
            dates=dates,
            dfseas=dfseas,
            family=family
        )
        
        # Calculate meta-predictors for this region
        temp_data = crossbasis.x
        avg_temp = np.nanmean(temp_data)
        temp_range = np.nanmax(temp_data) - np.nanmin(temp_data)
        
        meta_pred = {
            'avg_temp': avg_temp,
            'temp_range': temp_range
        }
        
        # Store results
        self.region_results.append(result)
        self.region_names.append(region_name)
        self.meta_predictors[region_name] = meta_pred
        
        return result
    
    def fit_meta_analysis(self, method: str = "reml", control: Optional[Dict] = None) -> None:
        """
        Fit multivariate meta-analysis on region results
        
        This replicates R's: mvmeta(coef~avgtmean+rangetmean, vcov, data=cities)
        
        Parameters
        ----------
        method : str, default "reml"
            Meta-analysis estimation method
        control : dict, optional
            Control parameters for optimization
        """
        
        if len(self.region_results) < 2:
            raise ValueError("Need at least 2 regions for meta-analysis")
        
        # Extract coefficients and variance-covariance matrices
        coef_matrix = []
        vcov_array = []
        
        for result in self.region_results:
            coef_matrix.append(result['reduced']['coefficients'])
            vcov_array.append(result['reduced']['vcov'])
        
        coef_matrix = np.array(coef_matrix)
        vcov_array = np.array(vcov_array)
        
        # Create meta-predictor design matrix: intercept + avg_temp + temp_range
        n_regions = len(self.region_results)
        X = np.ones((n_regions, 3))  # intercept, avg_temp, temp_range
        
        for i, region_name in enumerate(self.region_names):
            meta_pred = self.meta_predictors[region_name]
            X[i, 1] = meta_pred['avg_temp']
            X[i, 2] = meta_pred['temp_range']
        
        # Fit multivariate meta-analysis
        self.mv_model = mvmeta(
            y=coef_matrix, 
            S=vcov_array, 
            X=X, 
            method=method, 
            control=control
        )
        
        # The fitted values are valid whatever the optimiser's convergence flag says (get_summary reports the flag
        # alongside)
        print(f"  - Between-study variance (trace): {np.trace(self.mv_model.psi):.6f}")
        
    def calculate_blups(self, vcov: bool = True) -> List[Dict]:
        """
        Calculate BLUPs from fitted meta-analysis
        
        This replicates R's: blup(mv, vcov=T)
        
        Parameters
        ----------
        vcov : bool, default True
            Whether to include variance-covariance matrices
            
        Returns
        -------
        List[Dict]
            BLUP results for each region
        """
        
        if self.mv_model is None:
            raise ValueError("Must fit meta-analysis first using fit_meta_analysis()")
        
        self.blup_results = blup(self.mv_model, vcov=vcov)
        
        return self.blup_results
    
    def calculate_pooled_mmts(self) -> Dict[str, Any]:
        """
        Calculate MMTs using BLUP coefficients for each region
        
        This replicates the R workflow (02.secondstage.R):
        for(i in seq(length(dlist))) {
          bvar <- onebasis(quantile(data$tmean, 1:99/100), fun=varfun, knots=quantile(data$tmean, varper/100),
                           degree=vardegree, Boundary.knots=range(data$tmean))
          minperccity[i] <- (1:99)[which.min(bvar %*% blup[[i]]$blup)]
          mintempcity[i] <- quantile(data$tmean, minperccity[i]/100, na.rm=T)
        }
        minperccountry <- median(minperccity)
        
        The basis of the search is the variable basis of the region's first-stage cross-basis (function, knots,
        degree), as in R, where the same ``varfun``, ``varper`` and ``vardegree`` define both stages.
        
        Returns
        -------
        dict
            ``region_mmts``: per region name the MMT (``mmt`` temperature, ``percentile``, ...);
            ``individual_mmts``: the regional MMT temperatures in region order;
            ``pooled_mmt_median``, ``pooled_mmt_mean``, ``pooled_mmt_std``: median, mean and standard deviation of
            the regional MMT TEMPERATURES (degrees);
            ``pooled_mmt_percentile_median``: median of the regional MMT PERCENTILES (R's ``minperccountry``, the
            country-level quantity used to re-centre every region at that percentile of its own distribution).
        
        Raises
        ------
        ValueError
            If the MMT search fails for a region (for example a NaN BLUP or a BLUP whose length differs from the
            number of columns of the first-stage basis); R stops as well.
        """
        
        if self.blup_results is None:
            raise ValueError("Must calculate BLUPs first using calculate_blups()")
        
        region_mmts = {}
        mmt_values = []
        mmt_percentiles = []
        
        for i, region_name in enumerate(self.region_names):
            # Temperature data and first-stage variable basis of this region
            crossbasis = self.region_results[i]['glm_interface'].crossbasis
            temp_data = crossbasis.x
            blup_coef = self.blup_results[i]['blup']
            argvar = getattr(crossbasis, 'argvar', None)
            
            try:
                if argvar:
                    mmt_result = _mmt_from_blup(temp_data, blup_coef, argvar)
                else:
                    # no variable-basis specification available: the documented defaults of the Lancet analysis
                    # (bs, degree 2, knots at the 10th/75th/90th percentiles)
                    mmt_result = find_mmt_blup(x=temp_data, blup_coef=blup_coef)
            except Exception as exc:
                raise ValueError(f"MMT search failed for region {region_name!r}: {exc}") from exc
            
            region_mmts[region_name] = mmt_result
            mmt_values.append(mmt_result['mmt'])
            mmt_percentiles.append(mmt_result['percentile'])
        
        # Pooled/country-wide summaries of the regional MMTs
        if mmt_values:
            self.pooled_mmts = {
                'region_mmts': region_mmts,
                'pooled_mmt_median': np.median(mmt_values),
                'pooled_mmt_mean': np.mean(mmt_values),
                'pooled_mmt_std': np.std(mmt_values),
                'pooled_mmt_percentile_median': float(np.median(mmt_percentiles)),
                'individual_mmts': mmt_values
            }
        else:
            self.pooled_mmts = {'region_mmts': {}, 'pooled_mmt_median': None}
        
        return self.pooled_mmts
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of multi-location analysis
        
        Returns
        -------
        dict
            Summary statistics and results
        """
        
        summary = {
            'n_regions': len(self.region_results),
            'region_names': self.region_names,
            'meta_analysis_converged': self.mv_model.converged if self.mv_model else False,
            'pooled_mmts': self.pooled_mmts
        }
        
        # Report the fit whenever one exists; 'meta_analysis_converged' carries the optimiser's flag
        if self.mv_model and self.mv_model.psi is not None:
            summary.update({
                'meta_analysis_loglik': self.mv_model.loglik,
                'between_study_variance': np.trace(self.mv_model.psi),
                'meta_coefficients': self.mv_model.coefficients
            })
        
        return summary


def multi_location_dlnm_analysis(region_data: List[Dict[str, Any]], 
                                 method: str = "reml",
                                 dfseas: int = 8,
                                 family: str = 'quasipoisson') -> MultiLocationDLNM:
    """
    Convenience function for complete multi-location DLNM analysis
    
    Parameters
    ----------
    region_data : List[Dict]
        List of dictionaries, each containing:
        - 'name': region name
        - 'crossbasis': CrossBasis object
        - 'y': response variable
        - 'dates': date series
    method : str, default "reml"
        Meta-analysis method
    dfseas : int, default=8
        Seasonal degrees of freedom
    family : str, default='quasipoisson'
        GLM family
        
    Returns
    -------
    MultiLocationDLNM
        Complete analysis object with all results
    """
    
    # Initialize analysis manager
    analysis = MultiLocationDLNM()
    
    # Step 1: Analyze each region individually
    for region_info in region_data:
        analysis.add_region_analysis(
            region_name=region_info['name'],
            crossbasis=region_info['crossbasis'],
            y=region_info['y'],
            dates=region_info['dates'],
            dfseas=dfseas,
            family=family
        )
    
    # Step 2: Meta-analysis
    analysis.fit_meta_analysis(method=method)
    
    # Step 3: BLUP calculation
    analysis.calculate_blups(vcov=True)
    
    # Step 4: Pooled MMT calculation
    analysis.calculate_pooled_mmts()
    
    return analysis
