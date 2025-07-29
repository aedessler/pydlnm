#!/usr/bin/env python3
"""
Final test with correct London BLUP coefficients
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the pydlnm package to the path
sys.path.insert(0, '/Users/adessler/Desktop/DLNM')

from pydlnm.basis import CrossBasis
from pydlnm.prediction import crosspred
from pydlnm.utils import logknots

def final_rr_curve_test():
    """Final test with correct BLUP coefficients"""
    
    print("Final RR Curve Test with Correct London BLUP Coefficients")
    print("=" * 60)
    
    # Load London data
    data_path = '/Users/adessler/Desktop/DLNM/2015_gasparrini_Lancet_Rcodedata-master/regEngWales.csv'
    df = pd.read_csv(data_path, index_col=0)
    london_data = df[df['regnames'] == 'London'].copy()
    temp = london_data['tmean'].values
    
    print(f"✅ London data loaded: {len(temp)} observations")
    
    # Load CORRECT London BLUP data (region index 3)
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        
        os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
        
        robjects.r('''
        blup_data <- readRDS("/Users/adessler/Desktop/DLNM/r_validation_results/blup_results.rds")
        mmt_data <- read.csv("/Users/adessler/Desktop/DLNM/r_validation_results/mmt_results.csv")
        
        # Find London index
        london_idx <- which(mmt_data$city == "London")
        ''')
        
        with localconverter(robjects.default_converter + numpy2ri.converter):
            london_idx = int(robjects.r('london_idx')[0])
            # Use the CORRECT London BLUP coefficients
            london_blup = np.array(robjects.r(f'blup_data[[{london_idx}]]$blup'))
            london_vcov = np.array(robjects.r(f'blup_data[[{london_idx}]]$vcov'))
            london_mmt = float(robjects.r(f'mmt_data$mmt_temperature[{london_idx}]')[0])
        
        print(f"✅ CORRECT London BLUP data loaded:")
        print(f"   London index: {london_idx}")
        print(f"   BLUP coefficients: {london_blup}")
        print(f"   London MMT: {london_mmt:.4f}°C")
        
    except Exception as e:
        print(f"❌ Error loading correct BLUP data: {e}")
        return False
    
    # Create cross-basis
    varper = [10, 75, 90]
    vardegree = 2
    lag = 21
    lagnk = 3
    
    temp_knots = np.percentile(temp[~np.isnan(temp)], varper)
    lag_knots = logknots([0, lag], nk=lagnk)
    
    try:
        cb = CrossBasis(
            temp,
            lag=lag,
            argvar={'fun': 'bs', 'knots': temp_knots, 'degree': vardegree},
            arglag={'knots': lag_knots}
        )
        
        print(f"✅ Cross-basis created: {cb.shape}")
        
    except Exception as e:
        print(f"❌ Cross-basis creation failed: {e}")
        return False
    
    # Test PyDLNM crosspred with CORRECT coefficients
    try:
        print(f"\n📊 Testing PyDLNM crosspred with correct BLUP coefficients...")
        
        pred = crosspred(
            basis=cb,
            coef=london_blup,  # CORRECT London BLUP coefficients
            vcov=london_vcov,
            model_link="log",
            by=0.1,
            cen=london_mmt
        )
        
        print(f"✅ PyDLNM crosspred SUCCESS!")
        print(f"   Temperature range: {pred.predvar.min():.1f} to {pred.predvar.max():.1f}°C")
        print(f"   Prediction points: {len(pred.predvar)}")
        print(f"   RR range: {pred.allRRfit.min():.6f} to {pred.allRRfit.max():.6f}")
        
        # Check RR at MMT
        mmt_idx = np.argmin(np.abs(pred.predvar - london_mmt))
        rr_at_mmt = pred.allRRfit[mmt_idx]
        print(f"   RR at MMT: {rr_at_mmt:.6f}")
        
        # Compare with R reference
        r_curve_path = '/Users/adessler/Desktop/DLNM/r_validation_results/rr_curve_London.csv'
        r_curve = pd.read_csv(r_curve_path)
        
        r_temps = r_curve['temperature'].values
        r_rr_fit = r_curve['rr_fit'].values
        
        # Interpolate PyDLNM to R temperature points
        pydlnm_rr_interp = np.interp(r_temps, pred.predvar, pred.allRRfit)
        
        # Calculate differences
        differences = pydlnm_rr_interp - r_rr_fit
        max_diff = np.max(np.abs(differences))
        mean_diff = np.mean(np.abs(differences))
        rmse = np.sqrt(np.mean(differences**2))
        
        print(f"\n📈 Comparison with R reference:")
        print(f"   Max absolute difference: {max_diff:.6f}")
        print(f"   Mean absolute difference: {mean_diff:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        
        # Detailed sample comparison
        print(f"\n📋 Detailed Sample Point Comparison:")
        print(f"{'Temp':<8} {'R RR':<10} {'PyDLNM RR':<12} {'Diff':<10} {'Status'}")
        print("-" * 55)
        
        sample_indices = [0, len(r_temps)//6, len(r_temps)//3, len(r_temps)//2, 
                         2*len(r_temps)//3, 5*len(r_temps)//6, -1]
        
        excellent_matches = 0
        good_matches = 0
        
        for i in sample_indices:
            temp = r_temps[i]
            r_rr = r_rr_fit[i]
            pydlnm_rr = pydlnm_rr_interp[i]
            diff = abs(pydlnm_rr - r_rr)
            
            if diff < 0.001:
                status = "🟢 EXCELLENT"
                excellent_matches += 1
                good_matches += 1
            elif diff < 0.01:
                status = "🟡 GOOD"
                good_matches += 1
            elif diff < 0.05:
                status = "🟠 FAIR"
            else:
                status = "🔴 POOR"
                
            print(f"{temp:<8.1f} {r_rr:<10.4f} {pydlnm_rr:<12.4f} {diff:<10.6f} {status}")
        
        # Overall assessment
        print(f"\n{'='*60}")
        
        if max_diff < 0.001:
            status = "🟢 EXCELLENT MATCH"
            success = True
        elif max_diff < 0.01:
            status = "🟡 GOOD MATCH"
            success = True
        elif max_diff < 0.05:
            status = "🟠 FAIR MATCH"
            success = False
        else:
            status = "🔴 POOR MATCH"
            success = False
        
        print(f"FINAL RESULT: {status}")
        print(f"Excellent matches (<0.001): {excellent_matches}/{len(sample_indices)}")
        print(f"Good matches (<0.01): {good_matches}/{len(sample_indices)}")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        # Save final results
        comparison_df = pd.DataFrame({
            'temperature': r_temps,
            'r_rr_fit': r_rr_fit,
            'r_rr_low': r_curve['rr_low'].values,
            'r_rr_high': r_curve['rr_high'].values,
            'pydlnm_rr_fit': pydlnm_rr_interp,
            'pydlnm_rr_low': np.interp(r_temps, pred.predvar, pred.allRRlow),
            'pydlnm_rr_high': np.interp(r_temps, pred.predvar, pred.allRRhigh),
            'difference': differences,
            'abs_difference': np.abs(differences)
        })
        
        output_path = '/Users/adessler/Desktop/DLNM/final_rr_comparison_results.csv'
        comparison_df.to_csv(output_path, index=False)
        print(f"\n💾 Final results saved to: {output_path}")
        
        if success:
            print(f"\n🎯 MISSION ACCOMPLISHED!")
            print(f"PyDLNM successfully reproduces R DLNM RR curves!")
            print(f"✅ RR curve comparison: {status}")
            print(f"✅ Framework ready for epidemiological research")
        
        return success
        
    except Exception as e:
        print(f"❌ Final test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    success = final_rr_curve_test()
    
    print(f"\n{'='*60}")
    if success:
        print("🏆 SUCCESS: PyDLNM RR curve reproduction COMPLETE!")
        print("The implementation differences have been resolved.")
        print("PyDLNM can now generate RR curves matching R DLNM.")
    else:
        print("❌ Issues remain - further debugging needed")
    
    return success

if __name__ == "__main__":
    main()