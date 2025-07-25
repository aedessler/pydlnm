"""
Meta-analysis functionality for PyDLNM

Implements multivariate meta-analysis methods equivalent to R's mvmeta package
for pooling results from multiple locations/studies in DLNM analysis.
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
from typing import List, Dict, Optional, Tuple, Union
import warnings


class MVMeta:
    """
    Multivariate meta-analysis for DLNM results
    
    Equivalent to R's mvmeta package for pooling location-specific results
    from distributed lag non-linear models.
    """
    
    def __init__(self, method: str = "reml", control: Optional[Dict] = None):
        """
        Initialize MVMeta object
        
        Parameters:
        -----------
        method : str, default "reml"
            Estimation method. Options: "reml", "ml", "fixed"
        control : dict, optional
            Control parameters for optimization
        """
        self.method = method
        self.control = control or {}
        self.coefficients = None
        self.vcov = None
        self.psi = None  # Between-study variance-covariance matrix
        self.loglik = None
        self.converged = False
        self.fitted_values = None
        self.residuals = None
        
    def fit(self, y: np.ndarray, S: np.ndarray, X: Optional[np.ndarray] = None) -> 'MVMeta':
        """
        Fit multivariate meta-analysis model
        
        Parameters:
        -----------
        y : array-like, shape (n_studies, n_outcomes)
            Matrix of effect estimates from each study
        S : array-like, shape (n_studies, n_outcomes, n_outcomes) or (n_studies, n_outcomes)
            Within-study variance-covariance matrices or variances
        X : array-like, shape (n_studies, n_predictors), optional
            Study-level covariates (meta-regression)
            
        Returns:
        --------
        self : MVMeta
            Fitted model object
        """
        y = np.asarray(y)
        S = np.asarray(S)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        n_studies, n_outcomes = y.shape
        
        # Handle S input format
        if S.ndim == 2 and S.shape[1] == n_outcomes:
            # Diagonal matrices from variances
            S_matrices = np.zeros((n_studies, n_outcomes, n_outcomes))
            for i in range(n_studies):
                S_matrices[i] = np.diag(S[i])
            S = S_matrices
        elif S.ndim == 3:
            # Already in matrix format
            pass
        else:
            raise ValueError("S must be either (n_studies, n_outcomes) or (n_studies, n_outcomes, n_outcomes)")
            
        # Handle meta-regression
        if X is None:
            X = np.ones((n_studies, 1))
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != n_studies:
                raise ValueError("X must have same number of rows as y")
                
        self.n_studies = n_studies
        self.n_outcomes = n_outcomes
        self.n_predictors = X.shape[1]
        self.X = X
        self.y = y
        self.S = S
        
        # Fit model based on method
        if self.method in ["reml", "ml"]:
            self._fit_random_effects()
        elif self.method == "fixed":
            self._fit_fixed_effects()
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return self
    
    def _fit_fixed_effects(self):
        """Fit fixed-effects meta-analysis"""
        # Vectorize y and create block-diagonal covariance matrix
        y_vec = self.y.ravel()
        
        # Create block-diagonal matrix from S matrices
        V = linalg.block_diag(*self.S)
        
        # Create design matrix
        X_block = np.kron(np.eye(self.n_outcomes), self.X)
        
        # Weighted least squares
        try:
            V_inv = linalg.inv(V)
            XtVinv = X_block.T @ V_inv
            self.vcov = linalg.inv(XtVinv @ X_block)
            self.coefficients = self.vcov @ XtVinv @ y_vec
            
            # Reshape coefficients
            self.coefficients = self.coefficients.reshape(self.n_predictors, self.n_outcomes)
            
            # Calculate fitted values and residuals
            self.fitted_values = (X_block @ self.coefficients.ravel()).reshape(self.n_studies, self.n_outcomes)
            self.residuals = self.y - self.fitted_values
            
            # No between-study heterogeneity in fixed effects
            self.psi = np.zeros((self.n_outcomes, self.n_outcomes))
            self.converged = True
            
        except linalg.LinAlgError:
            warnings.warn("Singular covariance matrix in fixed effects estimation")
            self.converged = False
    
    def _fit_random_effects(self):
        """Fit random-effects meta-analysis using REML or ML"""
        # Initialize Psi (between-study variance-covariance)
        psi_init = np.eye(self.n_outcomes) * 0.1
        
        # Flatten upper triangle of Psi for optimization
        psi_params = psi_init[np.triu_indices(self.n_outcomes)]
        
        # Optimization
        result = minimize(
            fun=self._neg_loglik,
            x0=psi_params,
            method='BFGS',
            options=self.control
        )
        
        if result.success:
            # Reconstruct Psi from optimized parameters
            self.psi = self._params_to_psi(result.x)
            
            # Calculate final coefficients with optimal Psi
            self._update_coefficients()
            self.loglik = -result.fun
            self.converged = True
        else:
            warnings.warn("Optimization did not converge")
            self.converged = False
    
    def _neg_loglik(self, psi_params: np.ndarray) -> float:
        """Negative log-likelihood for optimization"""
        try:
            psi = self._params_to_psi(psi_params)
            
            # Check if Psi is positive definite
            if not self._is_positive_definite(psi):
                return np.inf
            
            # Calculate total covariance matrix for each study
            loglik = 0.0
            
            for i in range(self.n_studies):
                Vi = self.S[i] + psi  # Total covariance
                
                if not self._is_positive_definite(Vi):
                    return np.inf
                
                # Calculate log-likelihood contribution
                try:
                    Vi_inv = linalg.inv(Vi)
                    sign, logdet = linalg.slogdet(Vi)
                    if sign <= 0:
                        return np.inf
                    
                    # Expected values under current model
                    expected = (self.X[i:i+1] @ self.coefficients.ravel().reshape(-1, self.n_outcomes)).ravel()
                    residual = self.y[i] - expected.reshape(1, -1).ravel()
                    
                    loglik += -0.5 * (logdet + residual @ Vi_inv @ residual)
                    
                except linalg.LinAlgError:
                    return np.inf
            
            # REML correction
            if self.method == "reml":
                # Add penalty for fixed effects
                X_block = np.kron(np.eye(self.n_outcomes), self.X)
                V_total = linalg.block_diag(*[self.S[i] + psi for i in range(self.n_studies)])
                
                try:
                    V_inv = linalg.inv(V_total)
                    XtVinvX = X_block.T @ V_inv @ X_block
                    sign, logdet_X = linalg.slogdet(XtVinvX)
                    if sign > 0:
                        loglik -= 0.5 * logdet_X
                except linalg.LinAlgError:
                    return np.inf
            
            return -loglik
            
        except Exception:
            return np.inf
    
    def _params_to_psi(self, params: np.ndarray) -> np.ndarray:
        """Convert parameter vector to Psi matrix"""
        psi = np.zeros((self.n_outcomes, self.n_outcomes))
        triu_indices = np.triu_indices(self.n_outcomes)
        psi[triu_indices] = params
        
        # Make symmetric
        psi = psi + psi.T - np.diag(np.diag(psi))
        
        return psi
    
    def _is_positive_definite(self, matrix: np.ndarray) -> bool:
        """Check if matrix is positive definite"""
        try:
            linalg.cholesky(matrix)
            return True
        except linalg.LinAlgError:
            return False
    
    def _update_coefficients(self):
        """Update coefficients given current Psi estimate"""
        # Create total covariance matrix
        V_total = linalg.block_diag(*[self.S[i] + self.psi for i in range(self.n_studies)])
        
        # Vectorized design and response
        X_block = np.kron(np.eye(self.n_outcomes), self.X)
        y_vec = self.y.ravel()
        
        try:
            V_inv = linalg.inv(V_total)
            XtVinv = X_block.T @ V_inv
            self.vcov = linalg.inv(XtVinv @ X_block)
            coef_vec = self.vcov @ XtVinv @ y_vec
            
            # Reshape coefficients
            self.coefficients = coef_vec.reshape(self.n_predictors, self.n_outcomes)
            
            # Calculate fitted values and residuals
            self.fitted_values = (X_block @ coef_vec).reshape(self.n_studies, self.n_outcomes)
            self.residuals = self.y - self.fitted_values
            
        except linalg.LinAlgError:
            warnings.warn("Singular matrix in coefficient update")
    
    def predict(self, X_new: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for new studies
        
        Parameters:
        -----------
        X_new : array-like, optional
            Covariate values for prediction. If None, uses intercept only.
            
        Returns:
        --------
        pred : ndarray
            Predicted values
        pred_se : ndarray  
            Standard errors of predictions
        """
        if not self.converged:
            raise ValueError("Model has not converged")
            
        if X_new is None:
            X_new = np.ones((1, 1))
        else:
            X_new = np.asarray(X_new)
            if X_new.ndim == 1:
                X_new = X_new.reshape(1, -1)
        
        # Predictions
        pred = X_new @ self.coefficients
        
        # Prediction variance (includes between-study heterogeneity)
        pred_var = np.zeros((X_new.shape[0], self.n_outcomes, self.n_outcomes))
        
        for i in range(X_new.shape[0]):
            x_i = X_new[i:i+1]
            # Variance from fixed effects
            X_block_i = np.kron(np.eye(self.n_outcomes), x_i)
            var_fixed = X_block_i @ self.vcov @ X_block_i.T
            var_fixed_matrix = var_fixed.reshape(self.n_outcomes, self.n_outcomes)
            
            # Add between-study heterogeneity
            pred_var[i] = var_fixed_matrix + self.psi
        
        # Standard errors
        pred_se = np.sqrt(np.array([np.diag(pred_var[i]) for i in range(X_new.shape[0])]))
        
        return pred, pred_se
    
    def summary(self) -> pd.DataFrame:
        """Return summary of meta-analysis results"""
        if not self.converged:
            raise ValueError("Model has not converged")
        
        # Create summary DataFrame
        coef_flat = self.coefficients.ravel()
        se_flat = np.sqrt(np.diag(self.vcov))
        
        t_stat = coef_flat / se_flat
        p_values = 2 * (1 - np.abs(t_stat))  # Approximate, should use t-distribution
        
        # Create parameter names
        param_names = []
        predictor_names = [f"X{i}" for i in range(self.n_predictors)]
        outcome_names = [f"Y{i}" for i in range(self.n_outcomes)]
        
        for outcome in outcome_names:
            for predictor in predictor_names:
                param_names.append(f"{predictor}:{outcome}")
        
        summary_df = pd.DataFrame({
            'Parameter': param_names,
            'Estimate': coef_flat,
            'Std.Error': se_flat,
            't.value': t_stat,
            'p.value': p_values,
            'CI.lower': coef_flat - 1.96 * se_flat,
            'CI.upper': coef_flat + 1.96 * se_flat
        })
        
        return summary_df
    
    def heterogeneity_stats(self) -> Dict:
        """Calculate heterogeneity statistics"""
        if not self.converged:
            raise ValueError("Model has not converged")
        
        # I-squared equivalent for multivariate case
        # Simplified version - full implementation would be more complex
        
        stats = {
            'psi': self.psi,
            'tau2': np.diag(self.psi),  # Outcome-specific between-study variances
            'loglik': self.loglik,
            'converged': self.converged
        }
        
        return stats


def mvmeta(y: np.ndarray, S: np.ndarray, X: Optional[np.ndarray] = None, 
          method: str = "reml", control: Optional[Dict] = None) -> MVMeta:
    """
    Convenience function for multivariate meta-analysis
    
    Parameters:
    -----------
    y : array-like
        Effect estimates from each study
    S : array-like  
        Within-study variance-covariance matrices
    X : array-like, optional
        Study-level covariates
    method : str, default "reml"
        Estimation method
    control : dict, optional
        Control parameters
        
    Returns:
    --------
    MVMeta
        Fitted meta-analysis model
    """
    model = MVMeta(method=method, control=control)
    return model.fit(y, S, X)