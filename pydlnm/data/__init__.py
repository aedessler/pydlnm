"""
Test datasets for PyDLNM

This module provides access to the test datasets converted from the R dlnm package.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Get the data directory path
_DATA_DIR = Path(__file__).parent


def chicago_nmmaps() -> pd.DataFrame:
    """
    Load the Chicago NMMAPS dataset.
    
    This dataset contains daily mortality and environmental data for Chicago
    from 1987-2000, derived from the National Morbidity, Mortality and Air 
    Pollution Study (NMMAPS).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 5,114 observations and 14 variables:
        - date: Date (1987-2000)
        - time: Sequential time index
        - year, month, doy, dow: Time components
        - death: Daily mortality count (all-cause, excluding accidents)
        - cvd: Cardiovascular deaths
        - resp: Respiratory deaths  
        - temp: Mean temperature (Celsius)
        - dptp: Dew point temperature
        - rhum: Mean relative humidity
        - pm10: PM10 pollution levels
        - o3: Ozone levels
        
    Examples
    --------
    >>> data = chicago_nmmaps()
    >>> print(data.shape)
    (5114, 14)
    >>> print(data.columns.tolist())
    ['date', 'time', 'year', 'month', 'doy', 'dow', 'death', 'cvd', 'resp', 'temp', 'dptp', 'rhum', 'pm10', 'o3']
    """
    data_file = _DATA_DIR / 'chicago_nmmaps.csv'
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def drug_trial() -> pd.DataFrame:
    """
    Load the drug trial dataset.
    
    This dataset contains simulated data from a randomized controlled trial
    with time-varying drug exposures, designed to test DLNM methodology in
    experimental settings.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 200 observations and 7 variables:
        - id: Subject identifier
        - out: Outcome level measured at day 28
        - sex: Subject's sex
        - day1_7: Daily dose for week 1
        - day8_14: Daily dose for week 2
        - day15_21: Daily dose for week 3
        - day22_28: Daily dose for week 4
        
    Examples
    --------
    >>> data = drug_trial()
    >>> print(data.shape)
    (200, 7)
    """
    data_file = _DATA_DIR / 'drug_trial.csv'
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    
    return pd.read_csv(data_file)


def nested_case_control() -> pd.DataFrame:
    """
    Load the nested case-control dataset.
    
    This dataset contains simulated data from a nested case-control study
    with long-term occupational exposures, designed to test DLNM methodology
    in case-control settings.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 600 observations and 14 variables:
        - id: Subject identifier
        - case: Case (1) or control (0) indicator
        - age: Subject's age
        - riskset: Risk set identifier
        - exp15 through exp60: Yearly exposure levels for age periods
          15-19, 20-24, ..., 60-64 years
          
    Examples
    --------
    >>> data = nested_case_control()
    >>> print(data.shape)
    (600, 14)
    """
    data_file = _DATA_DIR / 'nested_case_control.csv'
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    
    return pd.read_csv(data_file)


# Convenience function for loading any dataset
def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a dataset by name.
    
    Parameters
    ----------
    name : str
        Dataset name: 'chicago', 'drug', or 'nested'
        
    Returns
    -------
    pd.DataFrame
        Requested dataset
        
    Raises
    ------
    ValueError
        If dataset name is not recognized
    """
    datasets = {
        'chicago': chicago_nmmaps,
        'drug': drug_trial,
        'nested': nested_case_control
    }
    
    if name not in datasets:
        available = list(datasets.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    
    return datasets[name]()


# Make datasets available at module level
__all__ = [
    'chicago_nmmaps',
    'drug_trial', 
    'nested_case_control',
    'load_dataset'
]