"""
Multi-location DLNM analysis with meta-analysis

This module provides functions to integrate meta-analysis functionality 
with the existing enhanced DLNM workflow for multi-location studies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings

from .improved_glm import fit_enhanced_dlnm_model
from .meta_analysis import mvmeta, blup
from .basis import CrossBasis
from .centering import find_mmt_blup


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
            Name/identifier for this region
        crossbasis : CrossBasis
            Cross-basis matrix for this region
        y : array-like
            Response variable (mortality counts)
        dates : pd.Series
            Date series for seasonality
        dfseas : int, default=8
            Seasonal degrees of freedom per year
        family : str, default='quasipoisson'
            GLM family
            
        Returns
        -------
        dict
            Individual region analysis results
        """
        
        print(f"Analyzing region: {region_name}")
        
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
        
        # Store meta-predictors
        if region_name not in self.meta_predictors:
            self.meta_predictors[region_name] = meta_pred
        
        print(f"  ✅ Region {region_name}: avg_temp={avg_temp:.1f}°C, range={temp_range:.1f}°C")
        
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
        
        print(f"Fitting meta-analysis for {len(self.region_results)} regions...")
        
        # Extract coefficients and variance-covariance matrices
        coef_matrix = []
        vcov_array = []
        
        for result in self.region_results:
            coef_matrix.append(result['reduced']['coefficients'])
            vcov_array.append(result['reduced']['vcov'])
        
        coef_matrix = np.array(coef_matrix)
        vcov_array = np.array(vcov_array)
        
        print(f"  - Coefficient matrix shape: {coef_matrix.shape}")
        print(f"  - Variance-covariance array shape: {vcov_array.shape}")
        
        # Create meta-predictor design matrix: intercept + avg_temp + temp_range
        n_regions = len(self.region_results)
        X = np.ones((n_regions, 3))  # intercept, avg_temp, temp_range
        
        for i, region_name in enumerate(self.region_names):
            meta_pred = self.meta_predictors[region_name]
            X[i, 1] = meta_pred['avg_temp']
            X[i, 2] = meta_pred['temp_range']
        
        print(f"  - Meta-predictors: avg_temp range [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}]°C")
        print(f"  - Temperature ranges: [{X[:, 2].min():.1f}, {X[:, 2].max():.1f}]°C")
        
        # Fit multivariate meta-analysis
        self.mv_model = mvmeta(
            y=coef_matrix, 
            S=vcov_array, 
            X=X, 
            method=method, 
            control=control
        )
        
        print(f"  ✅ Meta-analysis converged: {self.mv_model.converged}")
        
        if self.mv_model.converged:
            # Display some results
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
        
        print("Calculating BLUPs...")
        
        self.blup_results = blup(self.mv_model, vcov=vcov)
        
        print(f"  ✅ Generated BLUPs for {len(self.blup_results)} regions")
        
        # Show shrinkage information
        for i, region_name in enumerate(self.region_names):
            original_coef = self.region_results[i]['reduced']['coefficients']
            blup_coef = self.blup_results[i]['blup']
            shrinkage = np.linalg.norm(blup_coef - original_coef)
            print(f"  - {region_name}: shrinkage = {shrinkage:.6f}")
        
        return self.blup_results
    
    def calculate_pooled_mmts(self) -> Dict[str, Any]:
        """
        Calculate MMTs using BLUP coefficients for each region
        
        This replicates the R workflow:
        for(i in seq(length(dlist))) {
          bvar %*% blup[[i]]$blup
          mintempcity[i] <- quantile(data$tmean, minperccity[i]/100, na.rm=T)
        }
        
        Returns
        -------
        dict
            Dictionary with MMT results for each region and pooled estimates
        """
        
        if self.blup_results is None:
            raise ValueError("Must calculate BLUPs first using calculate_blups()")
        
        print("Calculating pooled MMTs using BLUPs...")
        
        region_mmts = {}
        mmt_values = []
        
        for i, region_name in enumerate(self.region_names):
            # Get temperature data for this region
            temp_data = self.region_results[i]['glm_interface'].crossbasis.x
            blup_coef = self.blup_results[i]['blup']
            
            # Use find_mmt_blup function
            try:
                mmt_result = find_mmt_blup(
                    x=temp_data,
                    blup_coef=blup_coef,
                    fun="bs",  # Match the basis function used
                    degree=2
                )
                
                region_mmts[region_name] = mmt_result
                mmt_values.append(mmt_result['mmt'])
                
                print(f"  - {region_name}: MMT = {mmt_result['mmt']:.2f}°C "
                      f"({mmt_result['percentile']:.1f}th percentile)")
                
            except Exception as e:
                print(f"  ⚠️  {region_name}: MMT calculation failed ({e})")
                continue
        
        # Calculate pooled/country-wide MMT (median of regional MMTs)
        if mmt_values:
            pooled_mmt = np.median(mmt_values)
            pooled_mean = np.mean(mmt_values)
            pooled_std = np.std(mmt_values)
            
            print(f"  ✅ Pooled MMT (median): {pooled_mmt:.2f}°C")
            print(f"  ✅ Pooled MMT (mean ± std): {pooled_mean:.2f} ± {pooled_std:.2f}°C")
            
            self.pooled_mmts = {
                'region_mmts': region_mmts,
                'pooled_mmt_median': pooled_mmt,
                'pooled_mmt_mean': pooled_mean,
                'pooled_mmt_std': pooled_std,
                'individual_mmts': mmt_values
            }
        else:
            print("  ❌ No valid MMT calculations")
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
        
        if self.mv_model and self.mv_model.converged:
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
    
    print("=" * 60)
    print("Multi-Location DLNM Analysis")
    print("=" * 60)
    
    # Initialize analysis manager
    analysis = MultiLocationDLNM()
    
    # Step 1: Analyze each region individually
    print(f"\n1. Individual Region Analysis ({len(region_data)} regions)")
    print("-" * 40)
    
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
    print(f"\n2. Meta-Analysis")
    print("-" * 40)
    
    analysis.fit_meta_analysis(method=method)
    
    # Step 3: BLUP calculation
    print(f"\n3. BLUP Calculation")
    print("-" * 40)
    
    analysis.calculate_blups(vcov=True)
    
    # Step 4: Pooled MMT calculation
    print(f"\n4. Pooled MMT Calculation")
    print("-" * 40)
    
    analysis.calculate_pooled_mmts()
    
    # Summary
    print(f"\n5. Summary")
    print("-" * 40)
    summary = analysis.get_summary()
    print(f"✅ Analysis complete for {summary['n_regions']} regions")
    if summary['pooled_mmts'].get('pooled_mmt_median'):
        print(f"✅ Pooled MMT: {summary['pooled_mmts']['pooled_mmt_median']:.2f}°C")
    
    print("=" * 60)
    
    return analysis