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
        
        # Fit the model based on method
        if self.method == "fixed":
            self._fit_fixed()
        elif self.method in ["reml", "ml"]:
            self._fit_random()
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self.converged = True
        return self
    
    def _fit_fixed(self):
        """Fit fixed-effects meta-analysis model"""
        # Fixed effects: no between-study heterogeneity
        self.psi = np.zeros((self.n_outcomes, self.n_outcomes))
        
        # Stack data and create block-diagonal covariance matrix
        y_vec = self.y.flatten(order='F')  # Column-major like R
        X_expanded = np.kron(np.eye(self.n_outcomes), self.X)
        
        # Create block-diagonal within-study covariance matrix
        V = np.zeros((self.n_studies * self.n_outcomes, self.n_studies * self.n_outcomes))
        for i in range(self.n_studies):
            start_idx = i * self.n_outcomes
            end_idx = (i + 1) * self.n_outcomes
            V[start_idx:end_idx, start_idx:end_idx] = self.S[i]
        
        # Weighted least squares estimation
        try:
            V_inv = linalg.inv(V)
            XtV_inv = X_expanded.T @ V_inv
            self.vcov = linalg.inv(XtV_inv @ X_expanded)
            self.coefficients = self.vcov @ XtV_inv @ y_vec
            
            # Reshape coefficients to (n_predictors, n_outcomes)
            self.coefficients = self.coefficients.reshape((self.n_predictors, self.n_outcomes), order='F')
            
            # Calculate fitted values and residuals
            self.fitted_values = X_expanded @ self.coefficients.flatten(order='F')
            self.residuals = y_vec - self.fitted_values
            
            # Calculate log-likelihood
            self.loglik = -0.5 * (np.log(linalg.det(V)) + self.residuals.T @ V_inv @ self.residuals)
            
        except linalg.LinAlgError:
            raise ValueError("Singular covariance matrix in fixed-effects model")
    
    def _fit_random(self):
        """Fit random-effects meta-analysis model using REML or ML"""
        
        # Initialize between-study covariance matrix
        psi_init = np.eye(self.n_outcomes) * 0.1
        
        # Optimize between-study variance parameters
        if self.method == "reml":
            objective = self._reml_objective
        else:  # ML
            objective = self._ml_objective
        
        # Vectorize psi for optimization (lower triangle + diagonal)
        psi_vec_init = self._psi_to_vec(psi_init)
        
        result = minimize(
            objective, 
            psi_vec_init,
            method='BFGS',
            options={'disp': self.control.get('showiter', False)}
        )
        
        if result.success:
            # Extract optimal psi
            self.psi = self._vec_to_psi(result.x)
            
            # Estimate fixed effects with optimal psi
            self._estimate_fixed_effects()
            
            # Store optimization results
            self.loglik = -result.fun
            self.converged = True
        else:
            # Fallback to simple estimates
            warnings.warn("Optimization failed, using simple estimates")
            self.psi = np.eye(self.n_outcomes) * 0.1
            self._estimate_fixed_effects()
            self.converged = False
    
    def _estimate_fixed_effects(self):
        """Estimate fixed effects given between-study covariance psi"""
        
        # Create total covariance matrix (within + between study)
        y_vec = self.y.flatten(order='F')
        X_expanded = np.kron(np.eye(self.n_outcomes), self.X)
        
        # Total covariance matrix
        V_total = np.zeros((self.n_studies * self.n_outcomes, self.n_studies * self.n_outcomes))
        for i in range(self.n_studies):
            start_idx = i * self.n_outcomes
            end_idx = (i + 1) * self.n_outcomes
            V_total[start_idx:end_idx, start_idx:end_idx] = self.S[i] + self.psi
        
        # Generalized least squares
        try:
            V_inv = linalg.inv(V_total)
            XtV_inv = X_expanded.T @ V_inv
            self.vcov = linalg.inv(XtV_inv @ X_expanded)
            coef_vec = self.vcov @ XtV_inv @ y_vec
            
            # Reshape coefficients
            self.coefficients = coef_vec.reshape((self.n_predictors, self.n_outcomes), order='F')
            
            # Calculate fitted values and residuals  
            self.fitted_values = X_expanded @ coef_vec
            self.residuals = y_vec - self.fitted_values
            
        except linalg.LinAlgError:
            raise ValueError("Singular total covariance matrix")
    
    def _psi_to_vec(self, psi):
        """Convert psi matrix to vector (lower triangle)"""
        indices = np.tril_indices(self.n_outcomes)
        return psi[indices]
    
    def _vec_to_psi(self, vec):
        """Convert vector to psi matrix (symmetric)"""
        psi = np.zeros((self.n_outcomes, self.n_outcomes))
        indices = np.tril_indices(self.n_outcomes)
        psi[indices] = vec
        psi = psi + psi.T - np.diag(np.diag(psi))
        return psi
    
    def _reml_objective(self, psi_vec):
        """REML objective function"""
        try:
            psi = self._vec_to_psi(psi_vec)
            
            # Ensure positive definiteness
            eigenvals = linalg.eigvals(psi)
            if np.any(eigenvals <= 0):
                return 1e10
            
            # Create total covariance matrix
            y_vec = self.y.flatten(order='F')
            X_expanded = np.kron(np.eye(self.n_outcomes), self.X)
            
            V_total = np.zeros((self.n_studies * self.n_outcomes, self.n_studies * self.n_outcomes))
            for i in range(self.n_studies):
                start_idx = i * self.n_outcomes
                end_idx = (i + 1) * self.n_outcomes
                V_total[start_idx:end_idx, start_idx:end_idx] = self.S[i] + psi
            
            V_inv = linalg.inv(V_total)
            XtV_inv = X_expanded.T @ V_inv
            
            # REML likelihood components
            logdet_V = np.log(linalg.det(V_total))
            logdet_XtVX = np.log(linalg.det(XtV_inv @ X_expanded))
            
            # Residual sum of squares
            P = V_inv - V_inv @ X_expanded @ linalg.inv(XtV_inv @ X_expanded) @ XtV_inv
            rss = y_vec.T @ P @ y_vec
            
            # REML objective (negative log-likelihood)
            reml = 0.5 * (logdet_V + logdet_XtVX + rss)
            
            return reml
            
        except (linalg.LinAlgError, ValueError):
            return 1e10
    
    def _ml_objective(self, psi_vec):
        """ML objective function"""
        try:
            psi = self._vec_to_psi(psi_vec)
            
            # Ensure positive definiteness
            eigenvals = linalg.eigvals(psi)
            if np.any(eigenvals <= 0):
                return 1e10
            
            # Create total covariance matrix and estimate effects
            y_vec = self.y.flatten(order='F')
            X_expanded = np.kron(np.eye(self.n_outcomes), self.X)
            
            V_total = np.zeros((self.n_studies * self.n_outcomes, self.n_studies * self.n_outcomes))
            for i in range(self.n_studies):
                start_idx = i * self.n_outcomes
                end_idx = (i + 1) * self.n_outcomes
                V_total[start_idx:end_idx, start_idx:end_idx] = self.S[i] + psi
            
            V_inv = linalg.inv(V_total)
            XtV_inv = X_expanded.T @ V_inv
            
            # ML estimates
            coef_vec = linalg.inv(XtV_inv @ X_expanded) @ XtV_inv @ y_vec
            residuals = y_vec - X_expanded @ coef_vec
            
            # ML objective (negative log-likelihood)
            ml = 0.5 * (np.log(linalg.det(V_total)) + residuals.T @ V_inv @ residuals)
            
            return ml
            
        except (linalg.LinAlgError, ValueError):
            return 1e10
    
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