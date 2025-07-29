# PyDLNM Validation Documentation

**Date:** July 2025

## Overview

This folder contains files and documentation for comparing PyDLNM results against the R DLNM package using the 2015 Gasparrini Lancet England & Wales dataset.

## Folder Structure
Input data and R code to generate results was [obtained from github](https://github.com/gasparrini/2015_gasparrini_Lancet_Rcodedata)
Input data: `regEngWales.csv`

### `/scripts/` - Python Comparison Scripts
Contains Python scripts that perform the comparison analysis. **Scripts must be run in this specific order**:

#### Core Analysis Scripts 
Run these scripts in order:
1. **`analyze_all_cities_pydlnm.py`** - Main PyDLNM analysis script that processes all 10 regions
   - Generates PyDLNM models and meta-analysis
   - Creates intermediate results needed by subsequent scripts
2. **`generate_all_regions_rr_curves.py`** - Generates RR curves for all regions using PyDLNM
   - Loads R BLUP results and generates PyDLNM equivalents
   - Maps region names between PyDLNM and R data structures
3. **`create_complete_validation_plots.py`** - Generates comprehensive comparison visualizations
   - Creates side-by-side comparison plots for all regions
4. **`final_rr_curve_test.py`** - Final comparison test with point-by-point statistical analysis
   - Calculates detailed comparison metrics (RMSE, correlation, differences)
   - Handles region indexing issues (particularly for London)

#### Plotting Scripts (Run After Core Scripts)
- **`plot_rr_curves_comparison.py`** - Creates additional comparison plots between PyDLNM and R DLNM results
- **`plot_available_rr_curves.py`** - Utility script for plotting available RR curve data

### `/results/` 
Contains both R reference results and PyDLNM comparison outputs:

#### Comparison Analysis Results (R vs PyDLNM)
Files that compare PyDLNM against R DLNM:
- **`complete_rr_curves_summary.csv`** - Summary comparison statistics (includes R_RR_Range, PyDLNM_RR_Range, Max_Diff, RMSE, Correlation)
- **`final_rr_comparison_results.csv`** - Point-by-point comparison with columns: r_rr_fit, pydlnm_rr_fit, difference, abs_difference
- **`complete_rr_curves_validation_all_regions.png`** - Side-by-side visual comparison plots

#### R Reference Results (R DLNM Only)
Files containing only R DLNM results used as reference:
- **`all_rr_curves.rds`** - R DLNM relative risk curves for all regions (R binary format)
- **`blup_results.rds`** - Best Linear Unbiased Predictors from R meta-analysis (R binary format)
- **`meta_analysis_model.rds`** - Meta-analysis model object from R (R binary format)
- **`coefficients.rds`** - Model coefficients from R analysis (R binary format)
- **`vcov_matrices.rds`** - Variance-covariance matrices from R (R binary format)
- **`temperature_percentiles.rds`** - Temperature percentile data from R (R binary format)
- **`mmt_results.csv`** - Minimum Mortality Temperature results from R analysis

#### Regional R Reference Curves (R DLNM Only)
Individual CSV files with **only R DLNM** RR curve data (temperature, rr_fit, rr_low, rr_high, mmt):
- `rr_curve_East.csv`
- `rr_curve_East_Midlands.csv`
- `rr_curve_London.csv`
- `rr_curve_North_East.csv`
- `rr_curve_North_West.csv`
- `rr_curve_South_East.csv`
- `rr_curve_South_West.csv`
- `rr_curve_Wales.csv`
- `rr_curve_West_Midlands.csv`
- `rr_curve_Yorkshire___Humber.csv`

## Comparison Methodology

### Data Source
- **Paper**: Gasparrini et al. "Mortality risk attributable to high and low ambient temperature: a multicountry observational study" *The Lancet* 2015
- **Dataset**: England & Wales daily mortality and temperature data (1993-2006)
- **Regions**: 10 government office regions
- **Total Observations**: >51,000 daily observations

## How to Perform the Comparison

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scipy rpy2
# R packages: dlnm, mvmeta, splines
```

### Step-by-Step Execution

#### Step 1: Ensure R Reference Data Exists
Confirm the R reference analysis has been completed using scripts in `/r_reference/`. This should generate the R BLUP results that PyDLNM will be compared against.

#### Step 2: Run Core Analysis Scripts (In Order)
```bash
# 1. Generate PyDLNM analysis for all regions
python validation/scripts/analyze_all_cities_pydlnm.py

# 2. Generate and compare RR curves
python validation/scripts/generate_all_regions_rr_curves.py

# 3. Create visual comparisons
python validation/scripts/create_complete_validation_plots.py

# 4. Perform statistical comparison
python validation/scripts/final_rr_curve_test.py
```

#### Step 3: Optional Additional Plots
```bash
# Additional plotting utilities (optional)
python validation/scripts/plot_rr_curves_comparison.py
python validation/scripts/plot_available_rr_curves.py
```

**Last Updated**: July 29, 2025 