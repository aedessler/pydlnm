"""
PyDLNM: Distributed Lag Non-Linear Models in Python

A Python implementation of distributed lag linear and non-linear models (DLMs/DLNMs)
for modeling exposure-lag-response associations in epidemiological studies.

This package provides a comprehensive framework for:
- Creating basis functions for exposure-response and lag-response relationships
- Constructing cross-basis matrices using tensor products
- Fitting models with various regression frameworks
- Making predictions and visualizing results
- Handling time series and non-time series study designs

Main classes:
- OneBasis: One-dimensional basis functions
- CrossBasis: Cross-basis matrices for distributed lag models
- CrossPred: Prediction results and inference

Based on the R dlnm package by Antonio Gasparrini.
"""

__version__ = "0.1.0"
__author__ = "Python DLNM Contributors"

# Core classes
from .basis import OneBasis, CrossBasis
from .prediction import CrossPred

# Utility functions
from .utils import mklag, seqlag, exphist

# Basis function implementations
from .basis_functions import (
    LinearBasis,
    PolynomialBasis,
    SplineBasis,
    BSplineBasis,
    StrataBasis,
    ThresholdBasis,
)

# Data
from . import data

__all__ = [
    "OneBasis",
    "CrossBasis", 
    "CrossPred",
    "mklag",
    "seqlag", 
    "exphist",
    "LinearBasis",
    "PolynomialBasis",
    "SplineBasis",
    "BSplineBasis", 
    "StrataBasis",
    "ThresholdBasis",
    "data",
]