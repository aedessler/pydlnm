#!/usr/bin/env python3
"""
Comprehensive PyDLNM Analysis for All Cities in data.csv
Generate RR curves for temperature-mortality relationships across 106 US cities
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set R environment before importing PyDLNM
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'

# Import PyDLNM modules
from pydlnm import CrossBasis, SplineBasis
from pydlnm.improved_glm import ImprovedGLMInterface
from pydlnm.crossreduce import CrossReduce
from pydlnm.prediction import crosspred

def prepare_city_data(df, city_abb):
    """Prepare data for a specific city"""
    city_data = df[df['cityAbb'] == city_abb].copy()
    
    if len(city_data) == 0:
        return None, None
    
    # Create date column
    city_data['date'] = pd.to_datetime(city_data['Date'].astype(str), format='%Y%m%d')
    city_data = city_data.sort_values('date')
    
    # Extract required variables
    temperature = city_data['TMean'].values
    deaths = city_data['Death'].values
    dates = city_data['date']
    dow = city_data['DOW'].values
    
    # Remove any NaN values
    valid_idx = ~(np.isnan(temperature) | np.isnan(deaths))
    temperature = temperature[valid_idx]
    deaths = deaths[valid_idx]
    dates = dates[valid_idx]
    dow = dow[valid_idx]
    
    city_info = {
        'cityAbb': city_abb,
        'cityName': city_data['cityName'].iloc[0],
        'state': city_data['State'].iloc[0],
        'n_days': len(temperature),
        'temp_range': (temperature.min(), temperature.max()),
        'total_deaths': deaths.sum()
    }
    
    return {
        'temperature': temperature,
        'deaths': deaths,
        'dates': dates,
        'dow': dow
    }, city_info

def create_crossbasis(temperature, lag_max=30):
    """Create cross-basis matrix for temperature"""
    # Temperature spline basis (3 df for non-linear effect)
    temp_basis = SplineBasis(temperature, df=3, degree=2)
    
    # Lag basis (3 df across lag_max days)
    lag_knots = np.array([1, 7, 21])  # Knots at 1, 7, and 21 days
    
    # Create cross-basis
    cb = CrossBasis(
        temp_basis,
        lag=lag_max,
        lag_knots=lag_knots,
        lag_df=3
    )
    
    return cb

def fit_dlnm_model(city_data, city_info, lag_max=30):
    """Fit DLNM model for a city"""
    try:
        print(f"  Processing {city_info['cityName']}, {city_info['state']} ({city_info['cityAbb']})...")
        
        # Create cross-basis
        cb = create_crossbasis(city_data['temperature'], lag_max)
        
        # Create GLM interface
        glm_interface = ImprovedGLMInterface(cb)
        
        # Prepare additional covariates
        n_years = len(set(city_data['dates'].dt.year))
        df_seas = 4  # Seasonal degrees of freedom per year
        
        # Fit model
        model = glm_interface.fit_poisson(
            y=city_data['deaths'],
            dates=city_data['dates'],
            dow=city_data['dow'],
            df_time=df_seas * n_years
        )
        
        # Extract model information
        coefficients = model.params
        vcov = model.cov_params()
        
        return {
            'crossbasis': cb,
            'model': model,
            'coefficients': coefficients,
            'vcov': vcov,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        print(f"    Error fitting model for {city_info['cityAbb']}: {str(e)}")
        return {
            'crossbasis': None,
            'model': None,
            'coefficients': None,
            'vcov': None,
            'success': False,
            'error': str(e)
        }

def generate_rr_curve(city_data, model_result, city_info):
    """Generate RR curve for a city"""
    if not model_result['success']:
        return None
    
    try:
        # Get temperature range for prediction
        temp_min, temp_max = city_info['temp_range']
        temp_pred = np.linspace(temp_min, temp_max, 100)
        
        # Create cross-reducer
        reducer = CrossReduce(model_result['crossbasis'], model_result['model'])
        
        # Generate predictions
        pred_result = reducer.crosspred(
            var=temp_pred,
            coef=model_result['coefficients'],
            vcov=model_result['vcov']
        )
        
        # Calculate MMT (minimum mortality temperature)
        mmt_idx = np.argmin(pred_result['rr_fit'])
        mmt = temp_pred[mmt_idx]
        
        return {
            'temperature': temp_pred,
            'rr_fit': pred_result['rr_fit'],
            'rr_low': pred_result['rr_low'],
            'rr_high': pred_result['rr_high'],
            'mmt': mmt,
            'mmt_rr': pred_result['rr_fit'][mmt_idx],
            'success': True
        }
        
    except Exception as e:
        print(f"    Error generating RR curve for {city_info['cityAbb']}: {str(e)}")
        return {'success': False, 'error': str(e)}

def analyze_all_cities():
    """Main function to analyze all cities"""
    print("🌡️ PyDLNM Analysis: Temperature-Mortality RR Curves for All US Cities")
    print("=" * 80)
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv('data.csv')
    cities = df[['cityAbb', 'cityName', 'State']].drop_duplicates().sort_values('cityName')
    
    print(f"Found {len(cities)} cities in dataset")
    
    # Initialize results storage
    results = {}
    failed_cities = []
    summary_stats = []
    
    print("\n🔬 Processing cities...")
    
    # Process each city
    for idx, (_, city_row) in enumerate(cities.iterrows(), 1):
        city_abb = city_row['cityAbb']
        print(f"[{idx:3d}/{len(cities)}] {city_abb}")
        
        # Prepare city data
        city_data, city_info = prepare_city_data(df, city_abb)
        
        if city_data is None:
            print(f"    No data available for {city_abb}")
            failed_cities.append(city_abb)
            continue
        
        # Fit DLNM model
        model_result = fit_dlnm_model(city_data, city_info)
        
        if not model_result['success']:
            failed_cities.append(city_abb)
            continue
        
        # Generate RR curve
        rr_result = generate_rr_curve(city_data, model_result, city_info)
        
        if rr_result and rr_result['success']:
            # Store results
            results[city_abb] = {
                'city_info': city_info,
                'city_data': city_data,
                'model_result': model_result,
                'rr_result': rr_result
            }
            
            # Add to summary
            summary_stats.append({
                'cityAbb': city_abb,
                'cityName': city_info['cityName'],
                'state': city_info['state'],
                'n_days': city_info['n_days'],
                'temp_min': city_info['temp_range'][0],
                'temp_max': city_info['temp_range'][1],
                'temp_range': city_info['temp_range'][1] - city_info['temp_range'][0],
                'total_deaths': city_info['total_deaths'],
                'mmt': rr_result['mmt'],
                'mmt_rr': rr_result['mmt_rr'],
                'max_rr_cold': np.max(rr_result['rr_fit'][rr_result['temperature'] < rr_result['mmt']]),
                'max_rr_hot': np.max(rr_result['rr_fit'][rr_result['temperature'] > rr_result['mmt']])
            })
            
            print(f"    ✅ Success - MMT: {rr_result['mmt']:.1f}°C")
        else:
            failed_cities.append(city_abb)
            print(f"    ❌ Failed to generate RR curve")
    
    print(f"\n📈 Analysis Complete!")
    print(f"✅ Successfully analyzed: {len(results)} cities")
    print(f"❌ Failed: {len(failed_cities)} cities")
    
    if failed_cities:
        print(f"Failed cities: {', '.join(failed_cities)}")
    
    return results, summary_stats

def create_summary_plots(results, summary_stats):
    """Create comprehensive summary plots"""
    print("\n📊 Creating summary visualizations...")
    
    # Create output directory
    output_dir = Path('pydlnm_city_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # 1. MMT Distribution Plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: MMT by State
    plt.subplot(2, 3, 1)
    summary_df = pd.DataFrame(summary_stats)
    mmt_by_state = summary_df.groupby('state')['mmt'].mean().sort_values()
    plt.barh(range(len(mmt_by_state)), mmt_by_state.values)
    plt.yticks(range(len(mmt_by_state)), mmt_by_state.index, fontsize=8)
    plt.xlabel('Mean MMT (°C)')
    plt.title('Average MMT by State')
    plt.grid(axis='x', alpha=0.3)
    
    # Subplot 2: MMT Distribution
    plt.subplot(2, 3, 2)
    plt.hist(summary_df['mmt'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('MMT (°C)')
    plt.ylabel('Number of Cities')
    plt.title('Distribution of MMT Values')
    plt.grid(alpha=0.3)
    
    # Subplot 3: Temperature Range vs MMT
    plt.subplot(2, 3, 3)
    plt.scatter(summary_df['temp_range'], summary_df['mmt'], alpha=0.6)
    plt.xlabel('Temperature Range (°C)')
    plt.ylabel('MMT (°C)')
    plt.title('MMT vs Temperature Range')
    plt.grid(alpha=0.3)
    
    # Subplot 4: Max Cold RR Distribution
    plt.subplot(2, 3, 4)
    plt.hist(summary_df['max_rr_cold'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Max Cold RR')
    plt.ylabel('Number of Cities')
    plt.title('Max Cold Temperature RR')
    plt.grid(alpha=0.3)
    
    # Subplot 5: Max Hot RR Distribution
    plt.subplot(2, 3, 5)
    plt.hist(summary_df['max_rr_hot'], bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Max Hot RR')
    plt.ylabel('Number of Cities')
    plt.title('Max Hot Temperature RR')
    plt.grid(alpha=0.3)
    
    # Subplot 6: Data Coverage
    plt.subplot(2, 3, 6)
    plt.scatter(summary_df['n_days'], summary_df['total_deaths'], alpha=0.6)
    plt.xlabel('Days of Data')
    plt.ylabel('Total Deaths')
    plt.title('Data Coverage')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sample RR Curves Plot
    plt.figure(figsize=(20, 15))
    
    # Select sample cities for visualization (top 20 by total deaths)
    sample_cities = summary_df.nlargest(20, 'total_deaths')
    
    for i, (_, city) in enumerate(sample_cities.iterrows()):
        plt.subplot(4, 5, i + 1)
        
        city_abb = city['cityAbb']
        rr_data = results[city_abb]['rr_result']
        
        # Plot RR curve
        plt.plot(rr_data['temperature'], rr_data['rr_fit'], 'b-', linewidth=2)
        plt.fill_between(rr_data['temperature'], rr_data['rr_low'], rr_data['rr_high'],
                        alpha=0.3, color='blue')
        
        # Add MMT line
        plt.axvline(x=rr_data['mmt'], color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        
        plt.title(f"{city['cityName']}, {city['state']}\nMMT: {rr_data['mmt']:.1f}°C", 
                 fontsize=10)
        plt.xlabel('Temperature (°C)', fontsize=9)
        plt.ylabel('RR', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tick_params(labelsize=8)
    
    plt.suptitle('Temperature-Mortality RR Curves: Top 20 Cities by Total Deaths', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_rr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary plots saved to {output_dir}")

def export_results(results, summary_stats):
    """Export results to CSV files"""
    print("\n💾 Exporting results...")
    
    output_dir = Path('pydlnm_city_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Export summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / 'city_summary_statistics.csv', index=False)
    
    # 2. Export detailed RR curves for each city
    all_rr_curves = []
    
    for city_abb, city_results in results.items():
        city_info = city_results['city_info']
        rr_data = city_results['rr_result']
        
        for i, temp in enumerate(rr_data['temperature']):
            all_rr_curves.append({
                'cityAbb': city_abb,
                'cityName': city_info['cityName'],
                'state': city_info['state'],
                'temperature': temp,
                'rr_fit': rr_data['rr_fit'][i],
                'rr_low': rr_data['rr_low'][i],
                'rr_high': rr_data['rr_high'][i],
                'mmt': rr_data['mmt']
            })
    
    rr_curves_df = pd.DataFrame(all_rr_curves)
    rr_curves_df.to_csv(output_dir / 'all_city_rr_curves.csv', index=False)
    
    # 3. Export individual city files
    city_files_dir = output_dir / 'individual_cities'
    city_files_dir.mkdir(exist_ok=True)
    
    for city_abb, city_results in results.items():
        city_info = city_results['city_info']
        rr_data = city_results['rr_result']
        
        city_df = pd.DataFrame({
            'temperature': rr_data['temperature'],
            'rr_fit': rr_data['rr_fit'],
            'rr_low': rr_data['rr_low'],
            'rr_high': rr_data['rr_high']
        })
        
        filename = f"rr_curve_{city_abb}_{city_info['cityName'].replace(' ', '_').replace('/', '_')}.csv"
        city_df.to_csv(city_files_dir / filename, index=False)
    
    print(f"✅ Results exported to {output_dir}")
    print(f"   - Summary statistics: city_summary_statistics.csv")
    print(f"   - All RR curves: all_city_rr_curves.csv")
    print(f"   - Individual city files: individual_cities/ directory")

def main():
    """Main analysis function"""
    start_time = datetime.now()
    
    # Run analysis
    results, summary_stats = analyze_all_cities()
    
    if results:
        # Create visualizations
        create_summary_plots(results, summary_stats)
        
        # Export results
        export_results(results, summary_stats)
        
        # Print final summary
        print(f"\n🎉 PyDLNM Analysis Complete!")
        print(f"📊 Successfully analyzed {len(results)} cities")
        print(f"⏱️  Total time: {datetime.now() - start_time}")
        
        # Key statistics
        summary_df = pd.DataFrame(summary_stats)
        print(f"\n📈 Key Results:")
        print(f"   - Mean MMT: {summary_df['mmt'].mean():.1f}°C (range: {summary_df['mmt'].min():.1f} to {summary_df['mmt'].max():.1f}°C)")
        print(f"   - Mean cold RR: {summary_df['max_rr_cold'].mean():.2f}")
        print(f"   - Mean hot RR: {summary_df['max_rr_hot'].mean():.2f}")
        print(f"   - Temperature ranges: {summary_df['temp_range'].min():.1f} to {summary_df['temp_range'].max():.1f}°C")
        
    else:
        print("❌ No successful analyses completed")
    
    return results, summary_stats

if __name__ == "__main__":
    results, summary_stats = main()