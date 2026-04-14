"""
Meta-analysis functionality for PyDLNM

Implements multivariate meta-analysis equivalent to R's mvmeta package.
Algorithm matches R's mvmeta exactly:
  - Psi parameterised as L @ L.T (lower-Cholesky) → always PSD
  - Per-study Cholesky whitening (no big block-diagonal V matrix)
  - REML log-det computed from per-study Cholesky diagonals
  - BLUP vcov includes fixed-effects uncertainty (X @ vcov(beta) @ X.T term)
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from typing import List, Dict, Optional
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers matching R's internal functions
# ─────────────────────────────────────────────────────────────────────────────

def _par2Psi(par: np.ndarray, k: int) -> np.ndarray:
    """
    Convert parameter vector to Psi using L @ L.T Cholesky parameterisation.
    Matches R mvmeta's par2Psi(..., bscov='unstr').

    par has k*(k+1)/2 elements (lower triangle, column-major like R's lower.tri).
    """
    L = np.zeros((k, k))
    L[np.tril_indices(k)] = par          # lower triangle including diagonal
    return L @ L.T


def _Psi2par(Psi: np.ndarray) -> np.ndarray:
    """Extract parameter vector from Psi via Cholesky factorisation."""
    try:
        L = np.linalg.cholesky(Psi)
    except np.linalg.LinAlgError:
        # Near-singular: add small ridge
        L = np.linalg.cholesky(Psi + np.eye(Psi.shape[0]) * 1e-8)
    return L[np.tril_indices(Psi.shape[0])]


def _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False):
    """
    GLS fit using per-study Cholesky whitening.
    Matches R's glsfit() in mvmeta.

    Xlist : list of (k, p) arrays   — per-study design matrices
    ylist : list of (k,) arrays     — per-study outcome vectors
    Slist : list of (k, k) arrays   — per-study within-study covariances
    Psi   : (k, k)                  — between-study covariance
    """
    k = Psi.shape[0]
    Sigma_list, invU_list, invtUX_list, invtUy_list = [], [], [], []

    for S_i, X_i, y_i in zip(Slist, Xlist, ylist):
        Sigma_i = S_i + Psi
        try:
            U_i = np.linalg.cholesky(Sigma_i).T   # upper triangular: U_i.T @ U_i = Sigma_i
        except np.linalg.LinAlgError:
            U_i = np.linalg.cholesky(Sigma_i + np.eye(k) * 1e-8).T
        invU_i = linalg.solve_triangular(U_i, np.eye(k))  # U_i^{-1}
        # whitened: invU_i.T @ X_i,  invU_i.T @ y_i
        invtUX_i = invU_i.T @ X_i
        invtUy_i = invU_i.T @ y_i
        Sigma_list.append(Sigma_i)
        invU_list.append(invU_i)
        invtUX_list.append(invtUX_i)
        invtUy_list.append(invtUy_i)

    invtUX = np.vstack(invtUX_list)       # (n*k, p)
    invtUy = np.concatenate(invtUy_list)  # (n*k,)

    coef = np.linalg.lstsq(invtUX, invtUy, rcond=None)[0]  # (p,)

    if onlycoef:
        return coef

    return dict(coef=coef, Sigma_list=Sigma_list, invU_list=invU_list,
                invtUX_list=invtUX_list, invtUX=invtUX, invtUy=invtUy)


def _reml_fn(par, k, Xlist, ylist, Slist):
    """
    REML negative log-profile-likelihood.
    Matches R's remlprof.fn in mvmeta.
    """
    Psi = _par2Psi(par, k)
    try:
        gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False)
    except Exception:
        return 1e10

    coef = gls['coef']
    n_params = len(coef)

    # pdet1: -sum of log(diag(U_i))  (= -0.5 * log|Sigma_i| per study)
    pdet1 = 0.0
    for invU_i in gls['invU_list']:
        # U_i = inv(invU_i),  diag(U_i) = 1/diag(invU_i)
        pdet1 += np.sum(np.log(np.abs(np.diag(invU_i))))  # = -sum log diag(U_i)

    # pdet2: -log|X.T W X|  (REML correction for estimating fixed effects)
    tXWX = sum(invtUX_i.T @ invtUX_i for invtUX_i in gls['invtUX_list'])
    try:
        pdet2 = -np.sum(np.log(np.diag(np.linalg.cholesky(tXWX))))
    except np.linalg.LinAlgError:
        return 1e10

    # pres: residual sum of squares (quadratic form)
    residuals = gls['invtUy'] - gls['invtUX'] @ coef
    pres = -0.5 * np.dot(residuals, residuals)

    # constant (omit: doesn't affect optimisation)
    nall = sum(len(y) for y in ylist)
    pconst = -0.5 * (nall - n_params) * np.log(2 * np.pi)

    reml = pconst + pdet1 + pdet2 + pres
    return -reml   # minimise negative log-likelihood


def _reml_gr(par, k, Xlist, ylist, Slist):
    """
    Analytical REML gradient w.r.t. par (lower-Cholesky elements of Psi).
    Matches R's gradchol.reml in mvmeta.

    Derivative of REML log-likelihood w.r.t. par[i] where par[i] = L[r,c] (r>=c):
      grad[i] = 0.5 * sum_j { r_j.T @ W_j @ dPsi @ W_j @ r_j
                               - tr(W_j @ dPsi)
                               + tr(invtXWXtot @ X_j.T @ W_j @ dPsi @ W_j @ X_j) }
    where W_j = Sigma_j^{-1} and dPsi = d(Psi)/d(par[i]) = L[:,c] e_r.T + e_r L[:,c].T
    (with e_r the r-th basis vector, L lower-Cholesky factor).
    """
    L = np.zeros((k, k))
    L[np.tril_indices(k)] = par
    Psi = L @ L.T

    try:
        gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False)
    except Exception:
        return np.zeros_like(par)

    coef = gls['coef']
    invSigma_list = [invU_i @ invU_i.T for invU_i in gls['invU_list']]

    tXWX = sum(Xi.T @ Xi for Xi in gls['invtUX_list'])
    try:
        invtXWXtot = np.linalg.inv(tXWX)
    except np.linalg.LinAlgError:
        return np.zeros_like(par)

    res_list = [y_i - X_i @ coef for X_i, y_i in zip(Xlist, ylist)]

    tril_rows, tril_cols = np.tril_indices(k)
    grad = np.zeros(len(par))

    for i, (r, c) in enumerate(zip(tril_rows, tril_cols)):
        # Psi = L @ L.T, so dPsi/d(L[r,c]) has element [a,b]:
        #   = L[b,c] if a==r, else L[a,c] if b==r, else 0
        # In matrix form: D = e_r @ L[:,c].T + L[:,c] @ e_r.T
        # Matches R's gradchol.reml formula with U = L.T (upper Cholesky).
        e_r = np.zeros(k); e_r[r] = 1.0
        L_col_c = L[:, c]   # column c of L

        D = np.outer(e_r, L_col_c) + np.outer(L_col_c, e_r)

        g = 0.0
        for j, (invSigma_j, res_j, invtUX_j, X_j) in enumerate(
                zip(invSigma_list, res_list, gls['invtUX_list'], Xlist)):
            W_D = invSigma_j @ D @ invSigma_j
            F = res_j @ W_D @ res_j
            G = np.trace(invSigma_j @ D)
            H = np.trace(invtXWXtot @ X_j.T @ W_D @ X_j)
            g += 0.5 * (F - G + H)
        grad[i] = g

    return -grad   # negative because we minimise negative log-likelihood


def _igls_init(Xlist, ylist, Slist, k, n_iter=10):
    """
    IGLS initialisation for Psi. Matches R's initpar with igls.iter iterations.
    Produces a better starting point for BFGS than a fixed small identity.
    """
    Psi = np.eye(k) * 0.001
    npar = k * (k + 1) // 2

    # indMat[a,b] = vech index of (a,b) element (0-based)
    indMat = np.zeros((k, k), dtype=int)
    idx = 0
    for col in range(k):
        for row in range(col, k):
            indMat[row, col] = idx
            indMat[col, row] = idx
            idx += 1

    for _ in range(n_iter):
        try:
            gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False)
        except Exception:
            break

        coef = gls['coef']

        # Build Z matrix (k^2 × npar) and per-study eΣ = Σ ⊗ Σ
        # f_i = vec(outer(r_i, r_i)) - vec(S_i)
        Z = np.zeros((k * k, npar))
        for a in range(k):
            for b in range(k):
                Z[a * k + b, indMat[a, b]] = 1.0

        invteUZ_accum = np.zeros((npar, npar))
        invteUf_accum = np.zeros(npar)

        for S_i, X_i, y_i, Sigma_i in zip(Slist, Xlist, ylist, gls['Sigma_list']):
            r_i = y_i - X_i @ coef
            f_i = (np.outer(r_i, r_i) - S_i).ravel()

            # eΣ = Sigma_i ⊗ Sigma_i  (k^2 × k^2)
            eSigma = np.kron(Sigma_i, Sigma_i)
            try:
                eU = np.linalg.cholesky(eSigma).T
                inveU = linalg.solve_triangular(eU, np.eye(k * k))
            except np.linalg.LinAlgError:
                continue

            invteUZ_i = inveU.T @ Z
            invteUf_i = inveU.T @ f_i
            invteUZ_accum += invteUZ_i.T @ invteUZ_i
            invteUf_accum += invteUZ_i.T @ invteUf_i

        try:
            theta = np.linalg.solve(invteUZ_accum, invteUf_accum)
        except np.linalg.LinAlgError:
            break

        # Reconstruct symmetric Psi from vech(theta)
        Psi_new = np.zeros((k, k))
        for j in range(npar):
            # find (row,col) for index j
            rows, cols = np.where(indMat == j)
            for r, c in zip(rows, cols):
                Psi_new[r, c] = theta[j]

        # Project onto PSD cone
        eigvals, eigvecs = np.linalg.eigh(Psi_new)
        eigvals = np.maximum(eigvals, np.sqrt(np.finfo(float).eps))
        Psi = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return Psi


def _ml_fn(par, k, Xlist, ylist, Slist):
    """ML negative log-likelihood."""
    Psi = _par2Psi(par, k)
    try:
        gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False)
    except Exception:
        return 1e10

    coef = gls['coef']
    pdet1 = 0.0
    for invU_i in gls['invU_list']:
        pdet1 += np.sum(np.log(np.abs(np.diag(invU_i))))

    residuals = gls['invtUy'] - gls['invtUX'] @ coef
    pres = -0.5 * np.dot(residuals, residuals)

    nall = sum(len(y) for y in ylist)
    pconst = -0.5 * nall * np.log(2 * np.pi)

    ml = pconst + pdet1 + pres
    return -ml


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class MVMeta:
    """
    Multivariate meta-analysis matching R's mvmeta package.

    Usage
    -----
    mv = MVMeta()
    mv.fit(y, S, X)        # y: (n,k), S: (n,k,k), X: (n,p)
    results = blup(mv)
    """

    def __init__(self, method: str = "reml", control: Optional[Dict] = None):
        self.method = method
        self.control = control or {}
        self.coefficients = None   # (p, k) — matches R's coef(mv) layout
        self.vcov = None           # (p*k, p*k)
        self.psi = None            # (k, k)
        self.loglik = None
        self.converged = False

    def fit(self, y: np.ndarray, S: np.ndarray,
            X: Optional[np.ndarray] = None) -> 'MVMeta':
        """
        Parameters
        ----------
        y : (n_studies, k)
        S : (n_studies, k, k)  within-study covariance matrices
        X : (n_studies, p)     study-level covariates (meta-regression)
        """
        y = np.asarray(y, dtype=float)
        S = np.asarray(S, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n, k = y.shape

        if S.ndim == 2:
            # Diagonal case: (n, k) variances → (n, k, k)
            S3 = np.zeros((n, k, k))
            for i in range(n):
                S3[i] = np.diag(S[i])
            S = S3

        if X is None:
            X = np.ones((n, 1))
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

        self.n, self.k, self.p = n, k, X.shape[1]
        self.y = y
        self.S = S
        self.X = X

        # Per-study lists used throughout (matches R's list-based approach)
        # Each X_i is (k, p) — kron(I_k, x_i.T) = I_k ⊗ x_i  shape (k, k*p)
        # But R's glsfit uses X as (k × p) per study for meta-regression beta (k*p,)
        # Actually R stacks the design as kron(I_k, X[i,]) giving (k, k*p) per study.
        # We follow R's convention: per-study design = I_k ⊗ X[i,] = (k, p*k)
        # and coef is a (p*k,) vector = vec(beta.T).
        Xlist = [np.kron(np.eye(k), X[i:i+1]) for i in range(n)]  # (k, p*k) each
        ylist = [y[i] for i in range(n)]
        Slist = [S[i] for i in range(n)]

        self._Xlist = Xlist
        self._ylist = ylist
        self._Slist = Slist

        # IGLS initialization (matches R's initpar with igls.iter=10)
        igls_iter = self.control.get('igls.iter', 10)
        psi_init = _igls_init(Xlist, ylist, Slist, k, n_iter=igls_iter)
        par_init = _Psi2par(psi_init)

        obj = _reml_fn if self.method == 'reml' else _ml_fn
        jac = _reml_gr if self.method == 'reml' else None

        result = minimize(
            obj, par_init,
            args=(k, Xlist, ylist, Slist),
            jac=jac,
            method='BFGS',
            options={'maxiter': self.control.get('maxiter', 500),
                     'gtol': 1e-8,
                     'disp': self.control.get('showiter', False)}
        )

        self.psi = _par2Psi(result.x, k)
        self.loglik = -result.fun
        self.converged = result.success

        if not result.success:
            warnings.warn(f"MVMeta optimisation did not fully converge: {result.message}")

        # Final GLS for coefficients and vcov
        gls = _glsfit(Xlist, ylist, Slist, self.psi, onlycoef=False)
        coef_vec = gls['coef']          # (p*k,)

        # vcov of beta_vec
        tXWX = sum(Xi.T @ Xi for Xi in gls['invtUX_list'])
        self._vcov_beta = np.linalg.inv(tXWX)   # (p*k, p*k)
        self.vcov = self._vcov_beta

        # Reshape coef to (k, p) then transpose to (p, k) matching R's coef(mv)
        # coef_vec is ordered: [beta_{1,1}, beta_{1,2}, ..., beta_{1,k},
        #                        beta_{2,1}, ..., beta_{p,k}]
        # because Xlist[i] = kron(I_k, X[i,]) maps (p*k,) → k outcomes
        self.coefficients = coef_vec.reshape(k, self.p).T   # (p, k)

        return self


def blup(mv_model: MVMeta, vcov: bool = True) -> List[Dict]:
    """
    BLUPs from a fitted MVMeta model.

    Matches R's blup.mvmeta:
      blup_i = pred_i + Psi @ (S_i + Psi)^{-1} @ (y_i - pred_i)
      V_i    = X_i @ vcov(beta) @ X_i.T + Psi - Psi @ (S_i + Psi)^{-1} @ Psi
    """
    if not mv_model.converged and mv_model.psi is None:
        raise ValueError("MVMeta model has not been fitted")

    Psi = mv_model.psi
    k   = mv_model.k
    results = []

    for i in range(mv_model.n):
        y_i    = mv_model.y[i]
        S_i    = mv_model.S[i]
        X_i    = mv_model._Xlist[i]   # (k, p*k)

        # Meta-regression prediction for study i
        pred_i = X_i @ mv_model.coefficients.T.ravel()   # (k,)

        Sigma_i = S_i + Psi
        try:
            Sigma_inv_i = np.linalg.inv(Sigma_i)
        except np.linalg.LinAlgError:
            Sigma_inv_i = np.linalg.pinv(Sigma_i)

        # BLUP (shrinkage toward prediction)
        blup_coef = pred_i + Psi @ Sigma_inv_i @ (y_i - pred_i)

        result = {'blup': blup_coef}

        if vcov:
            # Uncertainty from fixed effects estimation
            var_fixed = X_i @ mv_model._vcov_beta @ X_i.T
            # Residual uncertainty after shrinkage
            var_random = Psi - Psi @ Sigma_inv_i @ Psi
            result['vcov'] = var_fixed + var_random

        results.append(result)

    return results


def mvmeta(y: np.ndarray, S: np.ndarray, X: Optional[np.ndarray] = None,
           method: str = "reml", control: Optional[Dict] = None) -> MVMeta:
    """Convenience wrapper: create and fit MVMeta."""
    model = MVMeta(method=method, control=control)
    return model.fit(y, S, X)
