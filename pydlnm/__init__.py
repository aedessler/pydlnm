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
from .prediction import CrossPred, crosspred
from .meta_analysis import MVMeta, mvmeta
from .centering import find_mmt, recenter_basis, CenteringManager
from .attribution import attrdl, attr_heat_cold, AttributionManager

# GLM integration and model fitting
from .glm_integration import DLNMGLMInterface, fit_dlnm_model
from .crossreduce import CrossReduce, crossreduce, coef, vcov

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

# Enhanced spline implementations
from .enhanced_splines import (
    EnhancedBSplineBasis,
    EnhancedNaturalSplineBasis,
    bs_enhanced,
    ns_enhanced,
)

# Seasonality modeling
from .seasonality import (
    HarmonicSeasonalBasis,
    SeasonalSplineBasis,
    FlexibleSeasonalBasis,
    create_seasonal_basis,
    SeasonalityManager,
)

# Penalized DLNM framework
from .penalized import (
    PenalizedCrossBasis,
    PenalizedDLNM,
    penalized_dlnm,
)

# Data
from . import data

__all__ = [
    "OneBasis",
    "CrossBasis", 
    "CrossPred",
    "crosspred",
    "MVMeta",
    "mvmeta",
    "find_mmt",
    "recenter_basis",
    "CenteringManager",
    "attrdl",
    "attr_heat_cold",
    "AttributionManager",
    "DLNMGLMInterface",
    "fit_dlnm_model",
    "CrossReduce",
    "crossreduce",
    "coef",
    "vcov",
    "mklag",
    "seqlag", 
    "exphist",
    "LinearBasis",
    "PolynomialBasis",
    "SplineBasis",
    "BSplineBasis", 
    "StrataBasis",
    "ThresholdBasis",
    "EnhancedBSplineBasis",
    "EnhancedNaturalSplineBasis",
    "bs_enhanced",
    "ns_enhanced",
    "HarmonicSeasonalBasis",
    "SeasonalSplineBasis", 
    "FlexibleSeasonalBasis",
    "create_seasonal_basis",
    "SeasonalityManager",
    "PenalizedCrossBasis",
    "PenalizedDLNM", 
    "penalized_dlnm",
    "data",
]