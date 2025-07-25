"""
Penalized DLNM framework for PyDLNM

Implements penalized distributed lag non-linear models with smoothing
penalties for both exposure-response and lag-response dimensions,
following the methodology of Gasparrini & Scheipl (2017).
"""

import numpy as np
from scipy import linalg, optimize
from scipy.sparse import csc_matrix, identity, kron
from typing import Optional, Union, Dict, Tuple, Any, List
import warnings

from .basis import CrossBasis, OneBasis
from .enhanced_splines import bs_enhanced, ns_enhanced


class PenalizedCrossBasis(CrossBasis):
    """
    Penalized cross-basis for smooth DLNM models
    
    Extends CrossBasis with penalty matrices for smoothing in both
    exposure and lag dimensions.
    
    Parameters:
    -----------
    penalty_var : float or array-like, default 1.0
        Penalty parameter(s) for exposure dimension
    penalty_lag : float or array-like, default 1.0  
        Penalty parameter(s) for lag dimension
    penalty_type : str, default 'difference'
        Type of penalty: 'difference', 'ridge', 'roughness'
    diff_order : int, default 2
        Order of difference penalty
    """
    
    def __init__(self, x, lag, argvar, arglag,
                 penalty_var: Union[float, np.ndarray] = 1.0,
                 penalty_lag: Union[float, np.ndarray] = 1.0,
                 penalty_type: str = 'difference',
                 diff_order: int = 2,
                 **kwargs):
        
        # Initialize parent CrossBasis
        super().__init__(x, lag, argvar, arglag, **kwargs)
        
        # Store penalty parameters
        self.penalty_var = penalty_var
        self.penalty_lag = penalty_lag
        self.penalty_type = penalty_type
        self.diff_order = diff_order
        
        # Create penalty matrices
        self._create_penalty_matrices()
        
        # Mark as penalized
        self.penalized = True
    
    def _create_penalty_matrices(self):
        """Create penalty matrices for both dimensions"""
        
        # Get basis dimensions
        n_var_basis = self.basis_var.shape[1]
        n_lag_basis = self.basis_lag.shape[1]
        
        # Create penalty matrices for each dimension
        self.P_var = self._create_penalty_matrix(n_var_basis, self.penalty_type, self.diff_order)
        self.P_lag = self._create_penalty_matrix(n_lag_basis, self.penalty_type, self.diff_order)
        
        # Create combined penalty matrix using Kronecker products
        self._create_combined_penalty()
    
    def _create_penalty_matrix(self, n_basis: int, penalty_type: str, diff_order: int) -> np.ndarray:
        """Create penalty matrix for a single dimension"""
        
        if penalty_type == 'difference':
            # Difference penalty matrix
            if diff_order == 1:
                # First difference: P[i,j] for |i-j| = 1
                D = np.zeros((n_basis - 1, n_basis))
                for i in range(n_basis - 1):
                    D[i, i] = -1
                    D[i, i + 1] = 1
                P = D.T @ D
                
            elif diff_order == 2:
                # Second difference: more common for smoothing
                if n_basis < 3:
                    # Fallback to ridge penalty for small basis
                    P = np.eye(n_basis)
                else:
                    D = np.zeros((n_basis - 2, n_basis))
                    for i in range(n_basis - 2):
                        D[i, i] = 1
                        D[i, i + 1] = -2
                        D[i, i + 2] = 1
                    P = D.T @ D
                    
            else:
                raise ValueError("diff_order must be 1 or 2")
                
        elif penalty_type == 'ridge':
            # Ridge penalty (identity matrix)
            P = np.eye(n_basis)
            
        elif penalty_type == 'roughness':
            # Roughness penalty (approximate second derivative)
            # This is simplified - full implementation would use integrated squared second derivatives
            P = np.zeros((n_basis, n_basis))
            for i in range(1, n_basis - 1):
                P[i-1:i+2, i-1:i+2] += np.array([
                    [1, -2, 1],
                    [-2, 4, -2], 
                    [1, -2, 1]
                ])
            
        else:
            raise ValueError("penalty_type must be 'difference', 'ridge', or 'roughness'")
        
        return P
    
    def _create_combined_penalty(self):
        """Create combined penalty matrix using Kronecker products"""
        
        n_var = self.P_var.shape[0]
        n_lag = self.P_lag.shape[0]
        
        # Convert penalty parameters to arrays
        penalty_var = np.atleast_1d(self.penalty_var)
        penalty_lag = np.atleast_1d(self.penalty_lag)
        
        # Create identity matrices
        I_var = np.eye(n_var)
        I_lag = np.eye(n_lag)
        
        # Combined penalty matrix
        # P_total = λ_var * (P_var ⊗ I_lag) + λ_lag * (I_var ⊗ P_lag)
        self.P_combined = (penalty_var[0] * np.kron(self.P_var, I_lag) + 
                          penalty_lag[0] * np.kron(I_var, self.P_lag))
        
        # Store components for diagnostics
        self.P_var_expanded = np.kron(self.P_var, I_lag)
        self.P_lag_expanded = np.kron(I_var, self.P_lag)
    
    def get_penalty_matrix(self) -> np.ndarray:
        """Get the combined penalty matrix"""
        return self.P_combined
    
    def update_penalties(self, penalty_var: Optional[float] = None, 
                        penalty_lag: Optional[float] = None):
        """Update penalty parameters and recompute penalty matrix"""
        
        if penalty_var is not None:
            self.penalty_var = penalty_var
        
        if penalty_lag is not None:
            self.penalty_lag = penalty_lag
        
        # Recompute combined penalty
        self._create_combined_penalty()


class PenalizedDLNM:
    """
    Penalized DLNM model with automatic smoothing parameter selection
    
    Implements penalized regression for DLNM with various methods for
    selecting smoothing parameters (GCV, AIC, REML).
    
    Parameters:
    -----------
    basis : PenalizedCrossBasis
        Penalized cross-basis object
    selection_method : str, default 'gcv'
        Method for smoothing parameter selection: 'gcv', 'aic', 'reml'
    optimizer : str, default 'bfgs'
        Optimization method for parameter selection
    """
    
    def __init__(self, basis: PenalizedCrossBasis, 
                 selection_method: str = 'gcv',
                 optimizer: str = 'bfgs'):
        
        self.basis = basis
        self.selection_method = selection_method
        self.optimizer = optimizer
        
        # Model results
        self.coefficients = None
        self.fitted_values = None
        self.residuals = None
        self.vcov = None
        self.optimal_penalties = None
        self.converged = False
        
        # Smoothing parameter selection results
        self.selection_results = {}
    
    def fit(self, y: np.ndarray, 
            X_extra: Optional[np.ndarray] = None,
            family: str = 'gaussian',
            initial_penalties: Optional[Tuple[float, float]] = None,
            max_iter: int = 100,
            tol: float = 1e-6) -> 'PenalizedDLNM':
        """
        Fit penalized DLNM model
        
        Parameters:
        -----------
        y : array-like
            Response vector  
        X_extra : array-like, optional
            Additional covariates (not penalized)
        family : str, default 'gaussian'
            Distribution family: 'gaussian', 'poisson', 'binomial'
        initial_penalties : tuple, optional
            Initial values for (penalty_var, penalty_lag)
        max_iter : int, default 100
            Maximum iterations for optimization
        tol : float, default 1e-6
            Convergence tolerance
            
        Returns:
        --------
        self : PenalizedDLNM
            Fitted model object
        """
        
        y = np.asarray(y)
        X_basis = self.basis.basis
        
        # Combine basis with extra covariates
        if X_extra is not None:
            X_extra = np.asarray(X_extra)
            if X_extra.ndim == 1:
                X_extra = X_extra.reshape(-1, 1)
            X_full = np.column_stack([X_basis, X_extra])
            
            # Extend penalty matrix with zeros for extra covariates
            n_extra = X_extra.shape[1]
            P_extended = linalg.block_diag(
                self.basis.get_penalty_matrix(),
                np.zeros((n_extra, n_extra))
            )
        else:
            X_full = X_basis
            P_extended = self.basis.get_penalty_matrix()
        
        self.X = X_full
        self.y = y
        self.P = P_extended
        self.family = family
        
        # Select optimal smoothing parameters
        if initial_penalties is None:
            initial_penalties = (1.0, 1.0)
        
        self.optimal_penalties = self._select_smoothing_parameters(
            initial_penalties, max_iter, tol
        )
        
        # Fit final model with optimal penalties
        self._fit_with_fixed_penalties(self.optimal_penalties)
        
        return self
    
    def _select_smoothing_parameters(self, initial_penalties: Tuple[float, float],
                                   max_iter: int, tol: float) -> Tuple[float, float]:
        """Select optimal smoothing parameters"""
        
        def objective(log_penalties):
            penalties = np.exp(log_penalties)
            return self._compute_selection_criterion(penalties)
        
        # Optimize on log scale to ensure positivity
        log_initial = np.log(initial_penalties)
        
        try:
            if self.optimizer == 'bfgs':
                result = optimize.minimize(
                    objective, log_initial, method='BFGS',
                    options={'maxiter': max_iter, 'gtol': tol}
                )
            elif self.optimizer == 'nelder-mead':
                result = optimize.minimize(
                    objective, log_initial, method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': tol}
                )
            else:
                raise ValueError("optimizer must be 'bfgs' or 'nelder-mead'")
            
            if result.success:
                optimal_log_penalties = result.x
                self.converged = True
            else:
                warnings.warn("Smoothing parameter optimization did not converge")
                optimal_log_penalties = log_initial
                
        except Exception as e:
            warnings.warn(f"Smoothing parameter optimization failed: {e}")
            optimal_log_penalties = log_initial
        
        optimal_penalties = tuple(np.exp(optimal_log_penalties))
        
        # Store selection results
        self.selection_results = {
            'initial_penalties': initial_penalties,
            'optimal_penalties': optimal_penalties,
            'converged': self.converged,
            'method': self.selection_method
        }
        
        return optimal_penalties
    
    def _compute_selection_criterion(self, penalties: Tuple[float, float]) -> float:
        """Compute selection criterion (GCV, AIC, or REML)"""
        
        penalty_var, penalty_lag = penalties
        
        # Update penalty matrix
        self.basis.update_penalties(penalty_var, penalty_lag)
        P = self.basis.get_penalty_matrix()
        
        # Extend penalty matrix if needed
        if hasattr(self, 'P'):
            if P.shape[0] < self.P.shape[0]:
                n_extra = self.P.shape[0] - P.shape[0]
                P = linalg.block_diag(P, np.zeros((n_extra, n_extra)))
        
        try:
            if self.family == 'gaussian':
                # Analytical solution for Gaussian case
                XtX = self.X.T @ self.X
                Xty = self.X.T @ self.y
                
                # Penalized normal equations: (X'X + P)β = X'y
                A = XtX + P
                A_inv = linalg.inv(A)
                beta = A_inv @ Xty
                
                # Fitted values and residuals
                y_fitted = self.X @ beta
                residuals = self.y - y_fitted
                
                # Compute selection criterion
                n = len(self.y)
                rss = np.sum(residuals**2)
                
                # Effective degrees of freedom
                H = self.X @ A_inv @ self.X.T
                edf = np.trace(H)
                
                if self.selection_method == 'gcv':
                    # Generalized Cross-Validation
                    gcv = (rss / n) / ((1 - edf/n)**2)
                    return gcv
                    
                elif self.selection_method == 'aic':
                    # Akaike Information Criterion
                    sigma2 = rss / (n - edf)
                    aic = n * np.log(sigma2) + 2 * edf
                    return aic
                    
                elif self.selection_method == 'reml':
                    # Restricted Maximum Likelihood
                    sigma2 = rss / (n - edf)
                    sign, logdet_A = linalg.slogdet(A)
                    sign_XtX, logdet_XtX = linalg.slogdet(XtX)
                    
                    if sign <= 0 or sign_XtX <= 0:
                        return np.inf
                    
                    reml = ((n - edf) * np.log(sigma2) + 
                           logdet_A - logdet_XtX)
                    return reml
                
            else:
                # Iterative methods for non-Gaussian families
                return self._compute_criterion_iterative(P)
                
        except (linalg.LinAlgError, np.linalg.LinAlgError):
            # Return large value if matrix operations fail
            return np.inf
    
    def _compute_criterion_iterative(self, P: np.ndarray) -> float:
        """Compute selection criterion for non-Gaussian families using IWLS"""
        
        # This is a simplified implementation
        # Full implementation would use iteratively weighted least squares
        
        from sklearn.linear_model import PoissonRegressor
        
        if self.family == 'poisson':
            # Use sklearn's Poisson regression as approximation
            # Note: This doesn't include the penalty - simplified for demo
            try:
                model = PoissonRegressor(fit_intercept=False, max_iter=100)
                model.fit(self.X, self.y)
                
                y_pred = model.predict(self.X)
                deviance = 2 * np.sum(self.y * np.log(self.y / y_pred) - (self.y - y_pred))
                
                # Approximate AIC
                edf = np.sum(np.diag(self.X @ linalg.pinv(self.X.T @ self.X + P) @ self.X.T))
                aic = deviance + 2 * edf
                
                return aic
                
            except Exception:
                return np.inf
        
        return np.inf
    
    def _fit_with_fixed_penalties(self, penalties: Tuple[float, float]):
        """Fit model with fixed penalty parameters"""
        
        penalty_var, penalty_lag = penalties
        self.basis.update_penalties(penalty_var, penalty_lag)
        P = self.basis.get_penalty_matrix()
        
        # Extend penalty if needed
        if hasattr(self, 'P'):
            if P.shape[0] < self.P.shape[0]:
                n_extra = self.P.shape[0] - P.shape[0]
                P = linalg.block_diag(P, np.zeros((n_extra, n_extra)))
        
        if self.family == 'gaussian':
            # Analytical solution
            XtX = self.X.T @ self.X
            Xty = self.X.T @ self.y
            
            A = XtX + P
            A_inv = linalg.inv(A)
            
            self.coefficients = A_inv @ Xty
            self.vcov = A_inv @ XtX @ A_inv  # Sandwich estimator
            self.fitted_values = self.X @ self.coefficients
            self.residuals = self.y - self.fitted_values
            
        else:
            # Would implement IWLS for other families
            raise NotImplementedError("Non-Gaussian families not yet implemented")
    
    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty
        
        Parameters:
        -----------
        X_new : array-like
            New design matrix
            
        Returns:
        --------
        tuple
            - predictions: Predicted values
            - std_errors: Standard errors of predictions
        """
        
        if self.coefficients is None:
            raise ValueError("Model has not been fitted")
        
        X_new = np.asarray(X_new)
        predictions = X_new @ self.coefficients
        
        # Prediction standard errors
        pred_var = np.sum((X_new @ self.vcov) * X_new, axis=1)
        std_errors = np.sqrt(np.maximum(0, pred_var))
        
        return predictions, std_errors
    
    def get_smoothing_info(self) -> Dict:
        """Get information about smoothing parameters and effective degrees of freedom"""
        
        if self.coefficients is None:
            raise ValueError("Model has not been fitted")
        
        # Compute effective degrees of freedom
        XtX = self.X.T @ self.X
        P = self.basis.get_penalty_matrix()
        
        if hasattr(self, 'P'):
            if P.shape[0] < self.P.shape[0]:
                n_extra = self.P.shape[0] - P.shape[0]  
                P = linalg.block_diag(P, np.zeros((n_extra, n_extra)))
        
        try:
            A_inv = linalg.inv(XtX + P)
            H = self.X @ A_inv @ self.X.T
            edf_total = np.trace(H)
            
            # Component-wise effective degrees of freedom (approximate)
            edf_var = np.trace(H) * self.basis.penalty_lag / (self.basis.penalty_var + self.basis.penalty_lag)
            edf_lag = np.trace(H) * self.basis.penalty_var / (self.basis.penalty_var + self.basis.penalty_lag)
            
        except linalg.LinAlgError:
            edf_total = edf_var = edf_lag = np.nan
        
        info = {
            'penalty_var': self.basis.penalty_var,
            'penalty_lag': self.basis.penalty_lag,
            'edf_total': edf_total,
            'edf_var': edf_var,
            'edf_lag': edf_lag,
            'selection_method': self.selection_method,
            'converged': self.converged
        }
        
        return info
    
    def summary(self) -> Dict:
        """Return model summary"""
        
        if self.coefficients is None:
            raise ValueError("Model has not been fitted")
        
        smoothing_info = self.get_smoothing_info()
        
        # Model fit statistics
        n_obs = len(self.y)
        rss = np.sum(self.residuals**2)
        tss = np.sum((self.y - np.mean(self.y))**2)
        r_squared = 1 - rss/tss
        
        summary = {
            'n_observations': n_obs,
            'n_parameters': len(self.coefficients),
            'residual_sum_squares': rss,
            'r_squared': r_squared,
            'smoothing_info': smoothing_info,
            'selection_results': self.selection_results
        }
        
        return summary


def penalized_dlnm(x: np.ndarray, lag: Union[int, List, Tuple], 
                   y: np.ndarray,
                   argvar: Dict, arglag: Dict,
                   penalty_var: float = 1.0,
                   penalty_lag: float = 1.0,
                   selection_method: str = 'gcv',
                   **kwargs) -> PenalizedDLNM:
    """
    Convenience function for fitting penalized DLNM
    
    Parameters:
    -----------
    x : array-like
        Exposure variable
    lag : int or sequence
        Lag structure
    y : array-like  
        Response variable
    argvar : dict
        Arguments for exposure basis
    arglag : dict
        Arguments for lag basis
    penalty_var : float, default 1.0
        Penalty parameter for exposure dimension
    penalty_lag : float, default 1.0
        Penalty parameter for lag dimension
    selection_method : str, default 'gcv'
        Smoothing parameter selection method
    **kwargs
        Additional arguments
        
    Returns:
    --------
    PenalizedDLNM
        Fitted penalized DLNM model
    """
    
    # Create penalized cross-basis
    basis = PenalizedCrossBasis(
        x, lag, argvar, arglag,
        penalty_var=penalty_var,
        penalty_lag=penalty_lag,
        **kwargs
    )
    
    # Create and fit model
    model = PenalizedDLNM(basis, selection_method=selection_method)
    model.fit(y)
    
    return model