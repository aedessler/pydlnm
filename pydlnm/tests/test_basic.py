"""
Basic tests for PyDLNM functionality

This module contains tests to validate the core functionality of the PyDLNM package.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

# Import PyDLNM components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pydlnm.utils import mklag, seqlag, exphist
from pydlnm.basis_functions import LinearBasis, PolynomialBasis, SplineBasis
from pydlnm.basis import OneBasis, CrossBasis
from pydlnm.prediction import CrossPred
from pydlnm.data import chicago_nmmaps


class TestUtils:
    """Test utility functions."""
    
    def test_mklag_single_positive(self):
        """Test mklag with single positive value."""
        result = mklag(5)
        expected = np.array([0, 5])
        np.testing.assert_array_equal(result, expected)
    
    def test_mklag_single_negative(self):
        """Test mklag with single negative value."""
        result = mklag(-3)
        expected = np.array([-3, 0])
        np.testing.assert_array_equal(result, expected)
    
    def test_mklag_two_values(self):
        """Test mklag with two values."""
        result = mklag([2, 8])
        expected = np.array([2, 8])
        np.testing.assert_array_equal(result, expected)
    
    def test_mklag_invalid_order(self):
        """Test mklag with invalid order (min > max)."""
        with pytest.raises(ValueError):
            mklag([8, 2])
    
    def test_seqlag_basic(self):
        """Test seqlag basic functionality."""
        result = seqlag([0, 5])
        expected = np.array([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(result, expected)
    
    def test_seqlag_with_step(self):
        """Test seqlag with custom step size."""
        result = seqlag([0, 2], by=0.5)
        expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_exphist_basic(self):
        """Test exphist basic functionality."""
        exposure = [1, 2, 3, 4, 5]
        result = exphist(exposure, lag=[0, 2])
        
        # Should be 5x3 matrix (5 time points, 3 lags: 0, 1, 2)
        assert result.shape == (5, 3)
        
        # First row should be [1, 0, 0] (exposure 1, no history)
        np.testing.assert_array_equal(result[0], [1, 0, 0])
        
        # Third row should be [3, 2, 1] (exposure 3, lag-1=2, lag-2=1)
        np.testing.assert_array_equal(result[2], [3, 2, 1])


class TestBasisFunctions:
    """Test basis function implementations."""
    
    def test_linear_basis(self):
        """Test linear basis function."""
        x = np.array([1, 2, 3, 4, 5])
        basis = LinearBasis()
        result = basis(x)
        
        expected = x.reshape(-1, 1)
        np.testing.assert_array_equal(result, expected)
    
    def test_linear_basis_with_intercept(self):
        """Test linear basis with intercept."""
        x = np.array([1, 2, 3, 4, 5])
        basis = LinearBasis(intercept=True)
        result = basis(x)
        
        expected = np.column_stack([np.ones(5), x])
        np.testing.assert_array_equal(result, expected)
    
    def test_polynomial_basis(self):
        """Test polynomial basis function."""
        x = np.array([1, 2, 3, 4, 5])
        basis = PolynomialBasis(degree=2)
        result = basis(x)
        
        # Should have 2 columns for degree 2 (x^1, x^2)
        assert result.shape == (5, 2)
        
        # Values should be scaled versions of x and x^2
        scale = np.max(np.abs(x))  # Should be 5
        expected_col1 = x / scale
        expected_col2 = (x / scale) ** 2
        
        np.testing.assert_array_almost_equal(result[:, 0], expected_col1)
        np.testing.assert_array_almost_equal(result[:, 1], expected_col2)


class TestOneBasis:
    """Test OneBasis class."""
    
    def test_onebasis_linear(self):
        """Test OneBasis with linear function."""
        x = np.array([1, 2, 3, 4, 5])
        basis = OneBasis(x, fun='lin')
        
        assert basis.shape == (5, 1)
        np.testing.assert_array_equal(basis.basis, x.reshape(-1, 1))
    
    def test_onebasis_polynomial(self):
        """Test OneBasis with polynomial function."""
        x = np.array([1, 2, 3, 4, 5])
        basis = OneBasis(x, fun='poly', degree=2)
        
        assert basis.shape == (5, 2)
        assert basis.fun == 'poly'
        assert 'degree' in basis.attributes
    
    def test_onebasis_range(self):
        """Test OneBasis range calculation."""
        x = np.array([1, 5, 3, 2, 4])
        basis = OneBasis(x, fun='lin')
        
        assert basis.range == (1, 5)


class TestCrossBasis:
    """Test CrossBasis class."""
    
    def test_crossbasis_basic(self):
        """Test basic CrossBasis functionality."""
        x = np.array([1, 2, 3, 4, 5])
        basis = CrossBasis(x, lag=3, 
                          argvar={'fun': 'lin'}, 
                          arglag={'fun': 'lin'})
        
        # Should have 5 observations, lag range [0,3] -> 4 lags
        # Linear functions give 1 df each, so 1*1 = 1 column total
        assert basis.shape[0] == 5
        assert basis.lag.tolist() == [0, 3]
        
    def test_crossbasis_df(self):
        """Test CrossBasis degrees of freedom calculation."""
        x = np.array([1, 2, 3, 4, 5])
        basis = CrossBasis(x, lag=2,
                          argvar={'fun': 'poly', 'degree': 2},
                          arglag={'fun': 'poly', 'degree': 1})
        
        # Polynomial degree 2 -> 2 df, degree 1 -> 1 df
        # So total should be 2 * 1 = 2 columns
        assert basis.df == (2, 1)


class TestIntegration:
    """Test integration between components."""
    
    def test_data_loading(self):
        """Test loading Chicago NMMAPS data."""
        data = chicago_nmmaps()
        
        assert isinstance(data, pd.DataFrame)
        assert 'temp' in data.columns
        assert 'death' in data.columns
        assert len(data) > 0
    
    def test_basic_workflow(self):
        """Test basic DLNM workflow."""
        # Load data
        data = chicago_nmmaps()
        
        # Create cross-basis
        cb = CrossBasis(data['temp'].values[:10], lag=3,
                       argvar={'fun': 'lin'},
                       arglag={'fun': 'lin'})
        
        assert cb.shape[0] == 10
        assert cb.lag.tolist() == [0, 3]
        
        # Create mock model for prediction testing
        mock_model = Mock()
        mock_model.params = np.random.randn(cb.shape[1])
        mock_model.cov_params = lambda: np.eye(cb.shape[1]) * 0.1
        
        # This would test prediction, but we need proper model integration
        # pred = CrossPred(cb, mock_model)
        # assert hasattr(pred, 'allfit')


if __name__ == '__main__':
    pytest.main([__file__])