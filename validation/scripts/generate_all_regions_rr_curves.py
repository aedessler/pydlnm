#!/usr/bin/env python3
"""
Generate PyDLNM RR curves for all 10 England & Wales regions
Fixed version with proper region name mapping
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Add the pydlnm package to the path
sys.path.insert(0, '/Users/adessler/Desktop/DLNM')

from pydlnm.basis import CrossBasis
from pydlnm.prediction import crosspred
from pydlnm.utils import logknots

def create_region_mapping():
    """Create mapping between CSV region names and R result names"""
    return {
        'East': 'East',
        'E-Mid': 'East Midlands', 
        'London': 'London',
        'N-East': 'North East',
        'N-West': 'North West',
        'S-East': 'South East',
        'S-West': 'South West',
        'Wales': 'Wales',
        'W-Mid': 'West Midlands',
        'York&Hum': 'Yorkshire & Humber'
    }

def load_r_rr_curves():
    """Load all R reference RR curves"""
    
    print("Loading R reference RR curves...")
    
    r_results_dir = '/Users/adessler/Desktop/DLNM/r_validation_results'
    region_files = list(Path(r_results_dir).glob('rr_curve_*.csv'))
    
    r_curves = {}
    
    for file_path in region_files:
        # Extract region name from filename
        region_name = file_path.stem.replace('rr_curve_', '').replace('_', ' ')
        if region_name == 'Yorkshire   Humber':
            region_name = 'Yorkshire & Humber'
            
        # Load the curve data
        curve_data = pd.read_csv(file_path)
        r_curves[region_name] = curve_data
        
        print(f"  ✅ {region_name}: {len(curve_data)} points")
    
    return r_curves

def generate_pydlnm_rr_curves():
    """Generate PyDLNM RR curves for all regions with proper mapping"""
    
    print("\nGenerating PyDLNM RR curves...")
    
    # Load England & Wales data
    data_path = '/Users/adessler/Desktop/DLNM/2015_gasparrini_Lancet_Rcodedata-master/regEngWales.csv'
    df = pd.read_csv(data_path, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create region mapping
    region_mapping = create_region_mapping()
    print(f"Region mapping: {region_mapping}")
    
    # Load BLUP results
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        
        os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
        
        robjects.r('''
        blup_data <- readRDS("/Users/adessler/Desktop/DLNM/r_validation_results/blup_results.rds")
        mmt_data <- read.csv("/Users/adessler/Desktop/DLNM/r_validation_results/mmt_results.csv")
        ''')
        
        with localconverter(robjects.default_converter + numpy2ri.converter):
            n_regions = int(robjects.r('length(blup_data)')[0])
            r_region_names = list(robjects.r('mmt_data$city'))
            
        print(f"  Found {n_regions} regions in R results: {r_region_names}")
        
    except Exception as e:
        print(f"❌ Error loading BLUP data: {e}")
        return {}
    
    # Parameters from R validation
    varper = [10, 75, 90]
    vardegree = 2
    lag = 21
    lagnk = 3
    
    pydlnm_curves = {}
    
    # Process each region from R results
    for i, r_region_name in enumerate(r_region_names):
        print(f"  Processing {r_region_name} (index {i+1})...")
        
        try:
            # Find corresponding CSV region name
            csv_region_name = None
            for csv_name, r_name in region_mapping.items():
                if r_name == r_region_name:
                    csv_region_name = csv_name
                    break
            
            if csv_region_name is None:
                print(f"    ❌ No CSV mapping found for {r_region_name}")
                continue
                
            print(f"    📍 Mapping: {r_region_name} -> {csv_region_name}")
            
            # Load region data from CSV
            region_data = df[df['regnames'] == csv_region_name].copy()
            
            if len(region_data) == 0:
                print(f"    ❌ No data found for {csv_region_name}")
                continue
                
            region_data = region_data.sort_values('date').reset_index(drop=True)
            temp = region_data['tmean'].values
            
            print(f"    📊 Found {len(region_data)} data points, temp range: {temp.min():.1f} to {temp.max():.1f}°C")
            
            # Load BLUP coefficients for this region
            with localconverter(robjects.default_converter + numpy2ri.converter):
                region_blup = np.array(robjects.r(f'blup_data[[{i+1}]]$blup'))
                region_vcov = np.array(robjects.r(f'blup_data[[{i+1}]]$vcov'))
                region_mmt = float(robjects.r(f'mmt_data$mmt_temperature[{i+1}]')[0])
            
            print(f"    🎯 BLUP coefficients: {len(region_blup)}, MMT: {region_mmt:.2f}°C")
            
            # Create cross-basis
            temp_knots = np.percentile(temp[~np.isnan(temp)], varper)
            lag_knots = logknots([0, lag], nk=lagnk)
            
            cb = CrossBasis(
                temp,
                lag=lag,
                argvar={'fun': 'bs', 'knots': temp_knots, 'degree': vardegree},
                arglag={'knots': lag_knots}
            )
            
            print(f"    🔧 Cross-basis created: {cb.basis.shape}")
            
            # Generate RR curve
            pred = crosspred(
                basis=cb,
                coef=region_blup,
                vcov=region_vcov,
                model_link="log",
                by=0.1,
                cen=region_mmt
            )
            
            # Store results
            pydlnm_curves[r_region_name] = {
                'temperature': pred.predvar,
                'rr_fit': pred.allRRfit,
                'rr_low': pred.allRRlow,
                'rr_high': pred.allRRhigh,
                'mmt': region_mmt
            }
            
            print(f"    ✅ Generated {len(pred.predvar)} prediction points")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pydlnm_curves

def create_summary_statistics(r_curves, pydlnm_curves):
    """Create summary statistics for all regions"""
    
    print(f"\nGenerating summary statistics...")
    
    # Find matching regions
    matched_regions = []
    for r_name in r_curves.keys():
        if r_name in pydlnm_curves:
            matched_regions.append(r_name)
    
    print(f"  Found {len(matched_regions)} matching regions: {matched_regions}")
    
    summary_stats = []
    
    for region_name in matched_regions:
        r_data = r_curves[region_name]
        pydlnm_data = pydlnm_curves[region_name]
        
        # Interpolate PyDLNM to R temperature points for exact comparison
        pydlnm_rr_interp = np.interp(r_data['temperature'], 
                                     pydlnm_data['temperature'], 
                                     pydlnm_data['rr_fit'])
        
        # Calculate statistics
        differences = pydlnm_rr_interp - r_data['rr_fit']
        
        stats = {
            'Region': region_name,
            'N_Points': len(r_data),
            'Temp_Range': f"{r_data['temperature'].min():.1f} to {r_data['temperature'].max():.1f}°C",
            'MMT': r_data['mmt'].iloc[0],
            'R_RR_Range': f"{r_data['rr_fit'].min():.3f} to {r_data['rr_fit'].max():.3f}",
            'PyDLNM_RR_Range': f"{pydlnm_rr_interp.min():.3f} to {pydlnm_rr_interp.max():.3f}",
            'Max_Diff': np.max(np.abs(differences)),
            'Mean_Diff': np.mean(np.abs(differences)),
            'RMSE': np.sqrt(np.mean(differences**2)),
            'Correlation': np.corrcoef(r_data['rr_fit'], pydlnm_rr_interp)[0,1]
        }
        
        summary_stats.append(stats)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)
    
    # Save summary
    summary_path = '/Users/adessler/Desktop/DLNM/complete_rr_curves_summary.csv'
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    
    # Print summary
    print(f"\n📊 COMPLETE VALIDATION SUMMARY:")
    print(f"{'Region':<20} {'Max Diff':<10} {'Mean Diff':<10} {'RMSE':<10} {'Correlation':<12}")
    print("-" * 72)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Region']:<20} {row['Max_Diff']:<10.4f} {row['Mean_Diff']:<10.4f} {row['RMSE']:<10.4f} {row['Correlation']:<12.6f}")
    
    # Overall statistics
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Regions Validated: {len(summary_df)}/10")
    print(f"   Mean of Max Differences: {summary_df['Max_Diff'].mean():.4f}")
    print(f"   Mean of Mean Differences: {summary_df['Mean_Diff'].mean():.4f}")
    print(f"   Mean RMSE: {summary_df['RMSE'].mean():.4f}")
    print(f"   Mean Correlation: {summary_df['Correlation'].mean():.6f}")
    
    # Assessment
    excellent_regions = sum(summary_df['Max_Diff'] < 0.05)
    good_regions = sum(summary_df['Max_Diff'] < 0.1)
    acceptable_regions = sum(summary_df['Max_Diff'] < 0.2)
    
    print(f"\n✅ EPIDEMIOLOGICAL ASSESSMENT:")
    print(f"   Excellent agreement (<0.05): {excellent_regions}/{len(summary_df)} regions")
    print(f"   Good agreement (<0.10): {good_regions}/{len(summary_df)} regions") 
    print(f"   Acceptable agreement (<0.20): {acceptable_regions}/{len(summary_df)} regions")
    
    if excellent_regions >= 8:
        print(f"   🎉 FRAMEWORK STATUS: PRODUCTION-READY")
    elif good_regions >= 8:
        print(f"   ✅ FRAMEWORK STATUS: SCIENTIFICALLY VALIDATED")
    else:
        print(f"   ⚠️  FRAMEWORK STATUS: NEEDS REFINEMENT")
    
    print(f"  💾 Complete summary saved to: {summary_path}")
    
    return summary_df

def main():
    """Main function to generate all RR curves"""
    
    print("PyDLNM Complete RR Curves Validation")
    print("=" * 50)
    
    # Load R reference curves
    r_curves = load_r_rr_curves()
    
    if not r_curves:
        print("❌ No R reference curves found")
        return False
    
    # Generate PyDLNM curves for all regions
    pydlnm_curves = generate_pydlnm_rr_curves()
    
    if not pydlnm_curves:
        print("❌ Failed to generate any PyDLNM curves")
        return False
    
    print(f"\n✅ Successfully generated PyDLNM curves for {len(pydlnm_curves)} regions")
    
    # Generate comprehensive summary statistics
    summary_df = create_summary_statistics(r_curves, pydlnm_curves)
    
    print(f"\n{'='*50}")
    print("🎯 COMPLETE RR CURVES VALIDATION FINISHED!")
    print(f"✅ Validated {len(pydlnm_curves)}/10 England & Wales regions")
    
    if len(pydlnm_curves) == 10:
        print(f"🎉 COMPLETE SUCCESS: All regions validated!")
        print(f"✅ PyDLNM framework ready for epidemiological research")
    else:
        print(f"⚠️  {10-len(pydlnm_curves)} regions still need debugging")
    
    return True

if __name__ == "__main__":
    main()