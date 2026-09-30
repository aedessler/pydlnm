"""
Multivariate meta-analysis for PyDLNM: a Python port of R's ``mvmeta`` (version 1.0.3).

What is ported
--------------
  * methods 'fixed', 'ml', 'reml', 'mm' and 'vc' (names are case-sensitive, as in R's ``match.arg``; any other name
    raises ValueError);
  * the estimation engine ``mvmeta.fit``: per-study Cholesky whitening (``glsfit``), the REML / ML profile
    log-likelihoods (``remlprof.fn`` / ``mlprof.fn``), their analytic gradients (``gradchol.reml`` / ``gradchol.ml``),
    the IGLS start (``initpar`` / ``iter.igls``) and ``mvmeta.control`` (see below);
  * R's input handling (``mvmeta`` / ``mkS``): S as an (n, k, k) array, a list of (k, k) matrices, (n, k(k+1)/2) vech
    rows, or (n, k) within-study variances (expanded with the correlation ``Scor``, as ``inputcov`` does); studies with
    missing covariates or with every outcome missing are dropped; partly missing outcomes are masked study by study in
    the GLS, the objective, the gradient, the IGLS start and ``blup``;
  * ``blup`` (``blup.mvmeta``).

Psi = L L' with L lower triangular (always positive semi-definite). The optimiser's parameter vector is the lower
triangle of L in ROW-major order (``np.tril_indices``); R orders it column by column. The two are a fixed permutation of
each other (``_par_to_R_order``); fits, Psi, coefficients and log-likelihoods do not depend on the choice, and
``MVMeta.hessian`` is returned in R's order.

Precision of the agreement with R (measured, see tests/test_metaanalysis_N.py)
------------------------------------------------------------------------------
  * deterministic algebra at the same Psi (REML / ML objective, analytic gradient, BLUP and BLUP variance): 1e-10 or
    better (typically 1e-13);
  * fitted optimum: agrees with R run at a tight tolerance (``control = list(maxiter = 20000, reltol = 1e-14)``) to
    <= 1e-6 relative in coefficients, vcov and Psi, and to 1e-8 in the log-likelihood; PyDLNM never stops at a lower
    log-likelihood than R;
  * against R's DEFAULT output the agreement is only about 1e-5 (1e-6 .. 1e-5 for simulated data, about 1e-5 in the
    coefficients and Psi, 6e-6 in the BLUPs for England & Wales and the US-106 second stage). This is R's own stopping
    error, not a difference in the maths: optim(method = "BFGS") stops as soon as the relative change of the objective
    over one iteration falls below ``reltol = sqrt(.Machine$double.eps)`` (1.5e-8), which leaves R's default estimates
    about 1e-5 short of the true optimum, whereas PyDLNM iterates to the round-off floor of the objective. PyDLNM does
    NOT reproduce R's optim trajectory; it is not "machine precision" agreement with R's default output, and
    "Algorithm matches R's mvmeta exactly" holds only for the algebra listed above.
  * the REML log-likelihood follows mvmeta (constant ``-0.5 * (nall - p*k) * log(2 pi)``). mixmeta's REML logLik is larger
    by ``sum(log(diag(chol(sum_i X_i' X_i))))`` (the ML logLik is identical); mixmeta's coefficient vector and vcov are
    predictor-major, mvmeta's and PyDLNM's outcome-major (index ``l * p + j`` for outcome l, predictor j).

Differences to R that callers should know about
-----------------------------------------------
  * ``maxiter`` defaults to 500 (R: 100) and counts scipy BFGS iterations; ``reltol`` is only applied as a stopping rule
    when it is supplied (PyDLNM otherwise iterates to the round-off floor, see above). The ``converged`` flag is judged
    like R, from the relative change of the objective: the fit has converged when the optimiser can make no more
    progress and the decrease that a further Newton step is predicted to give is at most
    ``reltol * (|f| + reltol)`` (``reltol`` defaults to ``sqrt(eps)`` as in R). scipy's 'precision loss' exit at the
    floor of the objective therefore counts as converged. A warning is issued only when ``maxiter`` is hit or the
    objective cannot be improved although the predicted decrease (the gradient) is still large; the results are
    reported either way.
  * ``blup`` replaces R's variance of 1e10 for a missing outcome by the exact limit (the outcome is dropped from
    Sigma_i); the two differ by O(1e-10).
  * for method 'mm' / 'vc' (no likelihood) ``loglik`` is NaN; for 'fixed' ``psi`` is the zero matrix (R: no Psi).
  * R's control option ``inputna = TRUE`` is not implemented (NotImplementedError); ``Psifix`` and ``Psicor`` are
    accepted and ignored, exactly as R ignores them for an unstructured Psi (the only ``bscov`` supported here).
"""

import math
import warnings
from collections.abc import Mapping
from typing import Dict, List, Optional

import numpy as np
from scipy import linalg
from scipy.optimize import minimize

from utils import asfloat


_EPS = float(np.finfo(float).eps)
_SQRT_EPS = math.sqrt(_EPS)
_IGLS_MIN_EIG = 1e-8        # R's iter.igls: eig$values <- pmax(eig$values, 10^-8)
_QR_TOL = 1e-7              # R's qr() default tolerance for declaring a column of the whitened design dependent

METHODS = ('fixed', 'ml', 'reml', 'mm', 'vc')


# ─────────────────────────────────────────────────────────────────────────────
# Control options (R: mvmeta.control)
# ─────────────────────────────────────────────────────────────────────────────

_CONTROL_NAMES = ('optim', 'showiter', 'maxiter', 'initPsi', 'Psifix', 'Psicor', 'Scor', 'inputna', 'inputvar',
                  'igls.iter', 'hessian', 'vc.adj', 'reltol', 'set.negeigen')
# mixmeta.control() names that mvmeta.control() does not have (the Europe-2022 second stage is a mixmeta() call):
# igls.inititer = igls.iter that may be <= 0, loglik.iter = route of the optimiser, checkPD, addSlist = S given through control
_MIXMETA_NAMES = ('igls.inititer', 'loglik.iter', 'checkPD', 'addSlist')
_LOGLIK_ITER = ('hybrid', 'newton', 'igls', 'rigls')
# entries of control['optim'] (R: a list handed to optim(method = "BFGS")) that have an equivalent here
_OPTIM_NAMES = ('reltol', 'maxit', 'trace', 'REPORT', 'ndeps')


def _check_method(method):
    """R: match.arg(method, c("fixed", "ml", "reml", "mm", "vc")) -- exact, case-sensitive names."""
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"invalid method {method!r}: 'arg' should be one of {', '.join(repr(m) for m in METHODS)} "
                         "(names are case-sensitive, as in R's mvmeta)")
    return method


def _mvmeta_control(control) -> dict:
    """Validate a control dict like R's ``mvmeta.control`` and fill in the defaults.

    Unknown names raise TypeError ("unused argument", as R does); options that are accepted by R but not implemented
    here raise NotImplementedError naming the option; nothing is silently ignored except ``Psifix`` / ``Psicor``, which
    R itself ignores for ``bscov = "unstr"``.
    """
    if control is None:
        control = {}
    if not isinstance(control, Mapping):
        raise TypeError(f"control must be a dict of mvmeta.control() options, not {type(control).__name__}")
    unknown = [key for key in control if key not in _CONTROL_NAMES and key not in _MIXMETA_NAMES]
    if unknown:
        raise TypeError("unused argument in control: " + ', '.join(repr(key) for key in unknown)
                        + " (accepted: " + ', '.join(_CONTROL_NAMES + _MIXMETA_NAMES) + ")")
    ctl = {'optim': {}, 'showiter': False, 'maxiter': 500, 'initPsi': None, 'Psifix': None, 'Psicor': 0, 'Scor': 0,
           'inputna': False, 'inputvar': 1e4, 'igls.iter': 10, 'hessian': False, 'vc.adj': True, 'reltol': None,
           'set.negeigen': _SQRT_EPS, 'igls.inititer': None, 'loglik.iter': 'hybrid', 'checkPD': None,
           'addSlist': None}
    ctl.update(control)

    if ctl['igls.inititer'] is not None:
        # mixmeta.control: `if (igls.inititer <= 0L) igls.inititer <- 0` -- zero IGLS iterations is legal (start from
        # diag(0.001)); mvmeta's igls.iter < 1 error does not apply
        if 'igls.iter' in control:
            raise ValueError("control has both 'igls.iter' (mvmeta) and 'igls.inititer' (mixmeta): give only one")
        if not np.isscalar(ctl['igls.inititer']) or not np.isfinite(ctl['igls.inititer']):
            raise ValueError("'igls.inititer' in the control list must be a number")
        ctl['igls.iter'] = max(int(ctl['igls.inititer']), 0)
    else:
        if ctl['igls.iter'] is None or not ctl['igls.iter'] >= 1:
            raise ValueError("'igls.iter' in the control list must be positive")
        ctl['igls.iter'] = int(ctl['igls.iter'])
    # R: match.arg(loglik.iter, c("hybrid", "newton", "igls", "rigls")), partial matching included. The route changes
    # where R's optimiser starts and which algorithm it runs, not the optimum it stops at, so it is validated here and
    # the optimum is always found by the BFGS from the IGLS start
    route = ctl['loglik.iter']
    hits = [name for name in _LOGLIK_ITER if isinstance(route, str) and route and name.startswith(route)]
    if route in _LOGLIK_ITER:
        hits = [route]
    if len(hits) != 1:
        raise ValueError(f"'arg' should be one of {', '.join(repr(m) for m in _LOGLIK_ITER)} (control['loglik.iter'])")
    ctl['loglik.iter'] = hits[0]
    if ctl['checkPD'] is not None:
        ctl['checkPD'] = bool(ctl['checkPD'])
    if ctl['inputna']:
        raise NotImplementedError("control['inputna'] = True is not implemented in PyDLNM; missing outcomes are "
                                  "handled exactly by masking them study by study (R's default, inputna = FALSE)")
    optim = ctl['optim'] or {}
    if not isinstance(optim, Mapping):
        raise TypeError("control['optim'] must be a dict of optim() control options")
    bad = [key for key in optim if key not in _OPTIM_NAMES]
    if bad:
        raise NotImplementedError("control['optim'] option(s) " + ', '.join(repr(key) for key in bad)
                                  + " are not supported by PyDLNM's optimiser (supported: "
                                  + ', '.join(_OPTIM_NAMES) + ")")
    ctl['optim'] = dict(optim)
    # R: optim <- modifyList(list(fnscale = -1, maxit = maxiter, reltol = reltol), optim)
    maxit = optim.get('maxit', ctl['maxiter'])
    if maxit is None or not maxit >= 1:
        raise ValueError("the maximum number of iterations ('maxiter' / optim$maxit) must be positive")
    ctl['_maxit'] = int(maxit)
    reltol = optim.get('reltol', ctl['reltol'])
    ctl['_reltol_user'] = reltol is not None
    if reltol is not None and not reltol > 0:
        raise ValueError("'reltol' must be positive")
    ctl['_reltol'] = _SQRT_EPS if reltol is None else float(reltol)
    ctl['_showiter'] = bool(ctl['showiter']) or bool(optim.get('trace', 0))
    ndeps = optim.get('ndeps', 1e-3)
    ctl['_ndeps'] = float(ndeps) if np.ndim(ndeps) == 0 else float(np.asarray(ndeps, dtype=float).ravel()[0])
    return ctl


# ─────────────────────────────────────────────────────────────────────────────
# Input handling (R: mvmeta(), mkS(), inputcov())
# ─────────────────────────────────────────────────────────────────────────────

def _col_major_lower(k: int, strict: bool = False):
    """(row, col) pairs of the lower triangle taken column by column (R's ``lower.tri`` / ``vechMat`` order)."""
    off = 1 if strict else 0
    return [(a, b) for b in range(k) for a in range(b + off, k)]


def _xpnd(vech_rows: np.ndarray, k: int) -> np.ndarray:
    """R's ``xpndMat`` for every row: (n, k(k+1)/2) vech rows -> (n, k, k) symmetric matrices."""
    out = np.empty((vech_rows.shape[0], k, k))
    for j, (a, b) in enumerate(_col_major_lower(k)):
        out[:, a, b] = vech_rows[:, j]
        out[:, b, a] = vech_rows[:, j]
    return out


def _inputcov(sd: np.ndarray, cor) -> np.ndarray:
    """R's ``mixmeta:::inputcov``: covariances D R D from standard deviations sd (n, k) and correlations ``cor``.

    ``cor`` is a scalar, a vector of length n (one per study), a vector of length k(k-1)/2 (one per outcome pair, lower
    triangle column by column) or an (n, k(k-1)/2) matrix. Returns (n, k, k); NaN standard deviations propagate to the
    corresponding row and column.
    """
    n, k = sd.shape
    if k == 1:
        return (sd ** 2)[:, :, None]
    npair = k * (k - 1) // 2
    cor = np.asarray(0.0 if cor is None else cor, dtype=float)
    if cor.ndim <= 1:
        cor = np.atleast_1d(cor)
        if cor.size in (1, n):
            cor = np.tile(cor.reshape(-1, 1), (1, npair)) if cor.size == 1 else np.repeat(cor[:, None], npair, axis=1)
        elif cor.size == npair:
            cor = np.tile(cor.reshape(1, -1), (n, 1))
        else:
            raise ValueError("Dimensions of 'sd' and 'cor' not consistent: control['Scor'] has length "
                             f"{cor.size}, expected 1, {n} (one per study) or {npair} (one per outcome pair)")
    elif cor.shape != (n, npair):
        raise ValueError("Dimensions of 'sd' and 'cor' not consistent: control['Scor'] has shape "
                         f"{cor.shape}, expected ({n}, {npair})")
    if np.any(cor ** 2 > 1):
        raise ValueError("correlations must be between -1 and 1 (control['Scor'])")
    R = np.ones((n, k, k))
    for j, (a, b) in enumerate(_col_major_lower(k, strict=True)):
        R[:, a, b] = cor[:, j]
        R[:, b, a] = cor[:, j]
    R[:, np.arange(k), np.arange(k)] = 1.0
    return (sd[:, :, None] * R) * sd[:, None, :]


def _as_array(a, name):
    try:
        return asfloat(a, copy=True)                # always a copy: the fitted model owns its data; a masked or
                                                    # <NA> cell is a missing value (NaN), as R's NA
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' could not be converted to a numeric array: {exc}") from None


def _resolve_addSlist(S, ctl):
    """mixmeta's ``control$addSlist``: the within-study covariances given through control instead of ``S``.

    R (getSlist): an error when ``S`` is also given, a list with one k x k matrix per study otherwise, no missing values.
    Returns the (n, k, k) array that stands in for ``S`` (or ``S`` itself without ``addSlist``).
    """
    add = ctl['addSlist']
    if add is None:
        return S
    if S is not None:
        raise ValueError("'addSlist' only allowed without 'S'")
    try:
        arr = np.array([np.atleast_2d(np.asarray(a, dtype=float)) for a in add])
    except (TypeError, ValueError):
        raise ValueError("'addSlist' not consistent with required format: a list with one covariance matrix per "
                         "study") from None
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError("wrong dimensions in 'addSlist': expected one square matrix per study")
    if np.isnan(arr).any():
        raise ValueError("no missing allowed in 'addSlist'")
    return arr


def _check_S_pd(S3, label):
    """mixmeta's ``checkPD(error = TRUE)``: stop when a within-study covariance has a negative eigenvalue (the observed
    outcomes of each study only; NaN rows / columns are the missing outcomes)."""
    for i, S_i in enumerate(S3):
        obs = ~np.isnan(np.diag(S_i))
        if obs.any() and np.linalg.eigvalsh(S_i[np.ix_(obs, obs)]).min() < 0:
            raise ValueError(f"Problems with positive-definiteness in '{label}' (study {i + 1}): "
                             "a within-study covariance has a negative eigenvalue (control['checkPD'])")


def _prepare(y, S, X, ctl):
    """Input handling of R's ``mvmeta()`` + ``mkS()`` + the start of ``mvmeta.fit``.

    Returns a dict with owned copies of the data of the studies that are used: y (m, k) with NaN for missing outcomes,
    S (m, k, k) symmetric within-study covariances (NaN rows / columns for missing outcomes), X (m, p), the indices of
    the used rows in the input ('kept') and the number of input rows.
    """
    y = _as_array(y, 'y')
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"'y' must be a vector or an (n, k) matrix, got {y.ndim} dimensions")
    n_in, k = y.shape

    if X is None:
        X = np.ones((n_in, 1))
    else:
        X = _as_array(X, 'X')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"'X' must be a vector or an (n, p) matrix, got {X.ndim} dimensions")
        if X.shape[0] != n_in:
            raise ValueError(f"variable lengths differ: 'X' has {X.shape[0]} rows but 'y' has {n_in}")

    mes = "incorrect dimensions for 'S'"
    S = _as_array(S, 'S')
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if S.ndim not in (2, 3):
        raise ValueError(f"{mes}: {S.ndim} dimensions")
    if S.shape[0] != n_in:
        raise ValueError(f"variable lengths differ: {mes}: S has {S.shape[0]} rows but 'y' has {n_in} studies")
    if S.ndim == 3:
        if S.shape[1:] != (k, k):
            raise ValueError(f"{mes}: expected ({n_in}, {k}, {k}), got {S.shape}")
        variances_only = False
        low = np.tril(S)                                        # R keeps the lower triangle (vechMat) and mirrors it
        S3 = low + np.swapaxes(np.tril(S, -1), 1, 2)
    elif S.shape[1] == k:
        variances_only = True                                   # R: a k-column S holds variances
        S3 = None
    elif S.shape[1] == k * (k + 1) // 2:
        variances_only = False
        S3 = _xpnd(S, k)
    else:
        raise ValueError(f"{mes}: S has {S.shape[1]} columns, expected {k} (variances) or {k * (k + 1) // 2} "
                         f"(vech rows of the {k}x{k} covariances), or an ({n_in}, {k}, {k}) array")

    for name, arr in (('y', y), ('X', X), ('S', S)):
        if np.isinf(arr).any():
            raise ValueError(f"'{name}' contains infinite values")

    # R: na.omit.data.frame.mvmeta drops studies with a missing covariate or with every outcome missing
    omit = np.isnan(X).any(axis=1) | np.isnan(y).all(axis=1)
    kept = np.flatnonzero(~omit)
    if kept.size < 2:
        raise ValueError("less than 2 valid studies after exclusion of missing")
    y, X = y[kept], X[kept]
    if variances_only:
        with np.errstate(invalid='ignore'):
            S3 = _inputcov(np.sqrt(S[kept]), ctl['Scor'])
    else:
        S3 = S3[kept]

    # R: "missing pattern in 'y' and S' is not consistent"
    nay = np.isnan(y)
    for i in range(y.shape[0]):
        obs = ~nay[i]
        if np.isnan(S3[i][np.ix_(obs, obs)]).any():
            raise ValueError("missing pattern in 'y' and S is not consistent: S has missing values for an observed "
                             f"outcome of study {kept[i]}")
    return dict(y=y, S=S3, X=X, kept=kept, n_in=n_in)


def _study_lists(y, S, X):
    """R's Xlist / ylist / Slist / nalist: per-study designs, outcomes and within-study covariances restricted to the
    observed outcomes. Xlist[i] = I_k[obs] (x) X[i] has shape (k_i, k*p)."""
    m, k = y.shape
    nalist = [np.isnan(y[i]) for i in range(m)]
    eye = np.eye(k)
    Xlist = [np.kron(eye[~na], X[i:i + 1]) for i, na in enumerate(nalist)]
    ylist = [y[i][~na] for i, na in enumerate(nalist)]
    Slist = [S[i][np.ix_(~na, ~na)] for i, na in enumerate(nalist)]
    return Xlist, ylist, Slist, nalist


# ─────────────────────────────────────────────────────────────────────────────
# Helpers matching R's internal functions
# ─────────────────────────────────────────────────────────────────────────────

def _par2Psi(par: np.ndarray, k: int) -> np.ndarray:
    """
    Convert the parameter vector to Psi = L @ L.T (R mvmeta's ``par2Psi(..., bscov = "unstr")``).

    ``par`` has k*(k+1)/2 elements: the lower triangle of the Cholesky factor L in ROW-major order
    (``np.tril_indices``: L[0,0], L[1,0], L[1,1], L[2,0], ...). R fills ``lower.tri(L, diag = TRUE)`` COLUMN by column
    (L[0,0], L[1,0], L[2,0], ..., L[1,1], ...); the two orders coincide for k <= 2 and are a permutation of each other
    for k >= 3 (``_par_to_R_order``). The parameter vector is internal to the optimiser: Psi, the coefficients and the
    log-likelihood do not depend on the order.
    """
    L = _par2L(par, k)
    return L @ L.T


def _par2L(par: np.ndarray, k: int) -> np.ndarray:
    L = np.zeros((k, k))
    L[np.tril_indices(k)] = par                   # lower triangle including the diagonal, row-major
    return L


def _Psi2par(Psi: np.ndarray) -> np.ndarray:
    """Parameter vector (lower Cholesky factor, row-major like ``_par2Psi``) of a positive definite Psi; R's
    ``initpar`` uses ``vechMat(t(chol(Psi)))``, the same numbers in column-major order."""
    try:
        L = np.linalg.cholesky(Psi)
    except np.linalg.LinAlgError:
        raise ValueError("Psi is not positive definite: the Cholesky factor needed to start the optimiser "
                         "does not exist") from None
    return L[np.tril_indices(Psi.shape[0])]


def _par_to_R_order(par: np.ndarray, k: int) -> np.ndarray:
    """Reorder a parameter vector from this module's row-major lower-triangle order to R's column-major order."""
    perm = _R_order_permutation(k)
    return np.asarray(par)[perm]


def _R_order_permutation(k: int) -> np.ndarray:
    """perm[j] = position, in the row-major lower triangle, of the j-th element of R's column-major order."""
    pos = {ab: i for i, ab in enumerate(zip(*np.tril_indices(k)))}
    return np.array([pos[ab] for ab in _col_major_lower(k)], dtype=int)


def _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False, nalist=None):
    """
    GLS fit using per-study Cholesky whitening (R's ``glsfit``).

    Xlist : list of (k_i, k*p) arrays   — per-study design matrices (observed outcomes only)
    ylist : list of (k_i,) arrays       — per-study outcome vectors
    Slist : list of (k_i, k_i) arrays   — per-study within-study covariances
    Psi   : (k, k)                      — between-study covariance (masked per study with ``nalist``)
    nalist: optional list of (k,) bool arrays, True for a missing outcome (default: nothing missing)

    Raises numpy.linalg.LinAlgError (a ValueError) when Sigma_i = S_i + Psi is not positive definite and ValueError when
    the design is rank deficient (R: error in chol / qr.solve).
    """
    k = Psi.shape[0]
    Sigma_list, U_list, invU_list, invtUX_list, invtUy_list = [], [], [], [], []

    for i, (S_i, X_i, y_i) in enumerate(zip(Slist, Xlist, ylist)):
        if nalist is None or not nalist[i].any():
            Psi_i = Psi
        else:
            obs = ~nalist[i]
            Psi_i = Psi[np.ix_(obs, obs)]
        Sigma_i = S_i + Psi_i
        try:
            U_i = np.linalg.cholesky(Sigma_i).T                   # upper triangular: U_i.T @ U_i = Sigma_i
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                f"within-study covariance plus Psi is not positive definite for study {i} "
                "(R: error in chol())") from None
        invU_i = linalg.solve_triangular(U_i, np.eye(U_i.shape[0]))   # U_i^{-1}
        Sigma_list.append(Sigma_i)
        U_list.append(U_i)
        invU_list.append(invU_i)
        invtUX_list.append(invU_i.T @ X_i)                        # whitened design
        invtUy_list.append(invU_i.T @ y_i)                        # whitened outcomes

    invtUX = np.vstack(invtUX_list)           # (sum_i k_i, p*k)
    invtUy = np.concatenate(invtUy_list)      # (sum_i k_i,)

    nrow, ncol = invtUX.shape
    if nrow < ncol:
        raise ValueError(f"rank-deficient meta-regression: {ncol} coefficients (k * p) but only {nrow} observed "
                         "outcomes; too few studies for this many meta-predictors (R stops in chol / qr.solve)")
    Q, R = np.linalg.qr(invtUX)
    diagR = np.abs(np.diag(R))
    if np.any(diagR < _QR_TOL * np.linalg.norm(invtUX, axis=0)):
        raise ValueError("rank-deficient meta-regression: the (whitened) design matrix is singular or collinear "
                         "(R stops in chol / qr.solve); check the meta-predictors")
    coef = linalg.solve_triangular(R, Q.T @ invtUy)               # R: qr.solve(invtUX, invtUy)

    if onlycoef:
        return coef

    return dict(coef=coef, Sigma_list=Sigma_list, Ulist=U_list, invU_list=invU_list,
                invtUX_list=invtUX_list, invtUX=invtUX, invtUy=invtUy, R=R)


def _profile(par, k, Xlist, ylist, Slist, nalist, method, need_grad):
    """Negative profile log-likelihood (REML or ML) and, if asked, its analytic gradient with respect to ``par``.

    Objective: R's ``remlprof.fn`` / ``mlprof.fn`` (negated, to be minimised). Gradient: R's ``gradchol.reml`` /
    ``gradchol.ml``. With Q = sum_i [ a_i a_i' - W_i (+ B_i inv(X'WX) B_i' for REML) ] (embedded in k x k), where
    W_i = Sigma_i^{-1}, a_i = W_i r_i, B_i = W_i X_i, the derivative of the log-likelihood with respect to L[r, c] is
    (Q L)[r, c]; the loops over parameters and studies of the R code collapse to this one product.
    """
    L = _par2L(par, k)
    Psi = L @ L.T
    gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False, nalist=nalist)
    coef = gls['coef']
    nall = gls['invtUy'].shape[0]

    residuals = gls['invtUy'] - gls['invtUX'] @ coef
    pres = -0.5 * np.dot(residuals, residuals)
    pdet1 = -sum(np.sum(np.log(np.diag(U))) for U in gls['Ulist'])        # -sum_i log|U_i|

    if method == 'reml':
        tXWX = sum(Xi.T @ Xi for Xi in gls['invtUX_list'])
        try:
            chol_tXWX = np.linalg.cholesky(tXWX)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("X'WX is not positive definite (R: error in chol(tXWXtot)); the "
                                        "meta-regression is rank deficient") from None
        pdet2 = -np.sum(np.log(np.diag(chol_tXWX)))
        pconst = -0.5 * (nall - coef.shape[0]) * np.log(2 * np.pi)
        value = -(pconst + pdet1 + pdet2 + pres)
    else:
        pconst = -0.5 * nall * np.log(2 * np.pi)
        value = -(pconst + pdet1 + pres)

    if not need_grad:
        return value, None

    if method == 'reml':
        Linv = linalg.solve_triangular(chol_tXWX, np.eye(chol_tXWX.shape[0]), lower=True)
        invtXWXtot = Linv.T @ Linv                                        # R: chol2inv(chol(tXWXtot))
    Q = np.zeros((k, k))
    for i, (invU, X_i, y_i) in enumerate(zip(gls['invU_list'], Xlist, ylist)):
        W = invU @ invU.T                                                 # Sigma_i^{-1} (observed outcomes only)
        a = W @ (y_i - X_i @ coef)
        Qi = np.outer(a, a) - W
        if method == 'reml':
            B = W @ X_i
            Qi = Qi + B @ invtXWXtot @ B.T
        if nalist is None or not nalist[i].any():
            Q += Qi
        else:
            obs = ~nalist[i]
            Q[np.ix_(obs, obs)] += Qi
    Q = 0.5 * (Q + Q.T)
    grad_loglik = (Q @ L)[np.tril_indices(k)]
    return value, -grad_loglik


def _reml_fn(par, k, Xlist, ylist, Slist, nalist=None):
    """REML negative log-profile-likelihood (R's ``remlprof.fn``, sign flipped because PyDLNM minimises)."""
    return _profile(par, k, Xlist, ylist, Slist, nalist, 'reml', False)[0]


def _reml_gr(par, k, Xlist, ylist, Slist, nalist=None):
    """Analytic gradient of ``_reml_fn`` with respect to ``par`` (R's ``gradchol.reml``, sign flipped).

    For par[i] = L[r, c] (r >= c, Psi = L L') and dPsi = e_r L[:,c]' + L[:,c] e_r':
      d loglik / d par[i] = 0.5 * sum_j { r_j' W_j dPsi W_j r_j - tr(W_j dPsi)
                                          + tr(inv(X'WX) X_j' W_j dPsi W_j X_j) }.
    """
    return _profile(par, k, Xlist, ylist, Slist, nalist, 'reml', True)[1]


def _ml_fn(par, k, Xlist, ylist, Slist, nalist=None):
    """ML negative log-likelihood (R's ``mlprof.fn``, sign flipped)."""
    return _profile(par, k, Xlist, ylist, Slist, nalist, 'ml', False)[0]


def _ml_gr(par, k, Xlist, ylist, Slist, nalist=None):
    """Analytic gradient of ``_ml_fn``: ``_reml_gr`` without the tr(inv(X'WX) X_j' W dPsi W X_j) term (R's
    ``gradchol.ml``)."""
    return _profile(par, k, Xlist, ylist, Slist, nalist, 'ml', True)[1]


def _igls_init(Xlist, ylist, Slist, k, n_iter=10, nalist=None):
    """
    IGLS start for Psi (R's ``initpar`` with ``igls.iter`` calls of ``iter.igls``), missing outcomes masked per study.
    """
    Psi = np.eye(k) * 0.001
    npar = k * (k + 1) // 2

    # indMat = xpndMat(seq(npar)) (0-based): vech index of element (a, b), column-major lower-triangle numbering
    indMat = np.zeros((k, k), dtype=int)
    for j, (a, b) in enumerate(_col_major_lower(k)):
        indMat[a, b] = indMat[b, a] = j

    for _ in range(n_iter):
        gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False, nalist=nalist)
        coef = gls['coef']

        invteUZ_blocks, invteUf_blocks = [], []
        for i, (S_i, X_i, y_i, U_i) in enumerate(zip(Slist, Xlist, ylist, gls['Ulist'])):
            obs = slice(None) if nalist is None else ~nalist[i]
            r_i = y_i - X_i @ coef
            f_i = (np.outer(r_i, r_i) - S_i).ravel()
            ind_i = indMat[np.ix_(np.arange(k)[obs], np.arange(k)[obs])].ravel()
            Z_i = (ind_i[:, None] == np.arange(npar)[None, :]).astype(float)       # (k_i^2, npar)
            # R: chol(Sigma (x) Sigma) = chol(Sigma) (x) chol(Sigma)
            invU_i = gls['invU_list'][i]
            inveU = np.kron(invU_i, invU_i)
            invteUZ_blocks.append(inveU.T @ Z_i)
            invteUf_blocks.append(inveU.T @ f_i)

        theta = np.linalg.lstsq(np.vstack(invteUZ_blocks), np.concatenate(invteUf_blocks), rcond=None)[0]
        Psi_new = np.zeros((k, k))
        for j, (a, b) in enumerate(_col_major_lower(k)):
            Psi_new[a, b] = Psi_new[b, a] = theta[j]

        eigvals, eigvecs = np.linalg.eigh(Psi_new)
        eigvals = np.maximum(eigvals, _IGLS_MIN_EIG)
        Psi = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return Psi


def _newton_decrement(fg, x, g):
    """Objective decrease a further Newton step from ``x`` is predicted to achieve, 0.5 * g' H^+ g.

    H is the Hessian of the objective, by central differences of the analytic gradient (step cbrt(eps) * max(|x_i|, 1));
    H^+ is its pseudo-inverse over the directions of positive curvature (numerical rank tolerance max(w) * npar * eps,
    as in ``numpy.linalg.matrix_rank``): flat directions and boundary directions of a rank-deficient Psi carry no
    predicted decrease. Returns inf when no direction has positive curvature or the Hessian cannot be evaluated.
    """
    n = x.shape[0]
    H = np.empty((n, n))
    try:
        for i in range(n):
            h = _EPS ** (1.0 / 3.0) * max(abs(x[i]), 1.0)
            e = np.zeros(n)
            e[i] = h
            H[i] = (fg(x + e)[1] - fg(x - e)[1]) / (2.0 * h)
    except (np.linalg.LinAlgError, ValueError):
        return math.inf
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        return math.inf
    w, V = np.linalg.eigh(H)
    keep = w > w.max() * n * _EPS
    if not keep.any():
        return math.inf
    c = V.T @ g
    return float(0.5 * np.sum(c[keep] ** 2 / w[keep]))


class _StopOptimiser(Exception):
    """Raised from the BFGS callback when R's relative-objective-change rule is met."""


class _EvalCache:
    """value-and-gradient evaluator that remembers recent points (the optimiser's callback needs them again)."""

    def __init__(self, fg, size=16):
        self.fg, self.size, self.store = fg, size, {}

    def __call__(self, x):
        key = np.asarray(x, dtype=float).tobytes()
        if key not in self.store:
            if len(self.store) >= self.size:
                self.store.pop(next(iter(self.store)))
            f, g = self.fg(np.array(x, dtype=float))
            self.store[key] = (float(f), np.array(g, dtype=float))
        f, g = self.store[key]
        return f, g.copy()


def _minimise(fg, x0, maxit, reltol, reltol_user, showiter):
    """Minimise f with scipy's BFGS, given value-and-gradient ``fg``; restarts from the exit point when the line search
    breaks down although the objective can still be improved.

    scipy stops with 'precision loss' (status 2) when its strong-Wolfe line search cannot find a point because the
    objective differences are at round-off level. That is how a fit normally ends at the floor of the objective (the
    gradient cannot be driven below ~sqrt(eps * |f| * H), about 1e-8 of |f|, by comparing function values), and it is
    also how BFGS gives up early on extremely ill-conditioned problems (gradient ~1e8 at a log-likelihood 0.5 below
    the optimum). The two are told apart like R tells convergence from non-convergence, by the relative change of the
    objective: the fit has converged when the decrease a further Newton step is predicted to give is at most
    reltol * (|f| + reltol) (``_newton_decrement``); otherwise BFGS is restarted from the exit point (fresh
    inverse-Hessian) for as long as the last run improved the objective by more than that, and the fit is reported
    as not converged when it cannot be improved any more.

    If ``reltol_user`` the rule of R's optim is applied after every iteration as well: stop when the objective
    changes by no more than reltol * (|f| + reltol).
    """
    cache = _EvalCache(fg)
    x = np.array(x0, dtype=float)
    f, g = cache(x)
    if showiter:
        print(f'initial  value {-f:.10g}')
    total, reason, predicted = 0, None, math.nan
    converged = False
    while True:
        budget = maxit - total
        if budget <= 0:
            reason = 'maxiter'
            break
        seg = {'n': 0, 'prev': f, 'stop': None}

        def callback(xk, seg=seg, base=total):
            seg['n'] += 1
            fk, gk = cache(xk)
            if showiter:
                print(f'iter {base + seg["n"]:4d} value {-fk:.10g}')
            if reltol_user and abs(seg['prev'] - fk) <= reltol * (abs(fk) + reltol):
                seg['stop'] = (np.array(xk, dtype=float), fk, gk)
                raise _StopOptimiser()
            seg['prev'] = fk

        try:
            res = minimize(cache, x, jac=True, method='BFGS', callback=callback,
                           options={'maxiter': budget, 'gtol': 0.0})
            x_new, f_new, g_new = np.array(res.x, dtype=float), float(res.fun), np.array(res.jac, dtype=float)
            nit, status = int(res.nit), int(res.status)
        except _StopOptimiser:
            x_new, f_new, g_new = seg['stop']
            nit, status = seg['n'], 'reltol'
        total += nit
        if not np.isfinite(f_new):
            reason = 'nan'
            break
        gain = f - f_new
        x, f, g = x_new, f_new, g_new
        if status == 'reltol' or status == 0:
            converged = True
            break
        if status == 1:
            reason = 'maxiter'
            break
        # status 2 (line search 'precision loss') or 3 (NaN)
        tol = reltol * (abs(f) + reltol)
        predicted = _newton_decrement(fg, x, g)
        if predicted <= tol:
            converged = True
            break
        if status == 2 and gain > tol:
            continue                                        # still improving: restart from the exit point
        reason = 'stagnated'
        break

    if showiter:
        print(f'final  value {-f:.10g} ' + ('converged' if converged else f'NOT converged ({reason})'))
    return dict(x=x, f=f, g=g, nit=total, converged=converged, reason=reason, predicted=predicted)


def _hessian_R(fg, par, k, ndeps):
    """Hessian of the log-likelihood at ``par`` by central differences of the analytic gradient (R's optimHess with
    ``ndeps = 1e-3``), returned in R's parameter order and symmetrised, as R's ``fit$hessian``."""
    npar = par.shape[0]
    H = np.empty((npar, npar))
    for i in range(npar):
        step = np.zeros(npar)
        step[i] = ndeps
        H[i] = -(fg(par + step)[1] - fg(par - step)[1]) / (2 * ndeps)      # d(grad loglik) / d par_i
    H = 0.5 * (H + H.T)
    perm = _R_order_permutation(k)
    return H[np.ix_(perm, perm)]


def _finish_gls(gls):
    """Coefficients, vcov and rank from the QR of the whitened design, as R's mvmeta.* (vcov = R^-1 R^-T)."""
    Rinv = linalg.solve_triangular(gls['R'], np.eye(gls['R'].shape[0]))
    return gls['coef'], Rinv @ Rinv.T


def _posdef_psd(Psi, floor):
    """R: eigen(Psi); values <- pmax(values, floor); back-transform."""
    eigvals, eigvecs = np.linalg.eigh(Psi)
    negeigen = int(np.sum(eigvals < 0))
    return eigvecs @ np.diag(np.maximum(eigvals, floor)) @ eigvecs.T, negeigen


def _fit_mm(Xlist, ylist, Slist, nalist, k, m, p, ctl):
    """Method of moments (R's ``mvmeta.mm``; Jackson, White & Riley 2013). Returns (Psi, negeigen)."""
    Psi0 = np.zeros((k, k))
    gls = _glsfit(Xlist, ylist, Slist, Psi0, onlycoef=False, nalist=nalist)
    na_all = np.concatenate(nalist)                         # (m*k,), True where missing
    W = np.zeros((m * k, m * k))
    for i, invU in enumerate(gls['invU_list']):
        obs = ~nalist[i]
        W[i * k:(i + 1) * k, i * k:(i + 1) * k][np.ix_(obs, obs)] = invU @ invU.T
    X = np.zeros((m * k, k * p))
    X[~na_all] = np.vstack(Xlist)
    y = np.zeros(m * k)
    y[~na_all] = np.concatenate(ylist)
    tXWXtot = sum(Xi.T @ Xi for Xi in gls['invtUX_list'])
    Lc = np.linalg.cholesky(tXWXtot)
    Lcinv = linalg.solve_triangular(Lc, np.eye(Lc.shape[0]), lower=True)
    invtXWXtot = Lcinv.T @ Lcinv

    def fbtr(A):                                            # sum of the k x k diagonal blocks
        return sum(A[i * k:(i + 1) * k, i * k:(i + 1) * k] for i in range(A.shape[0] // k))

    H = X @ invtXWXtot @ X.T @ W
    IminusH = np.eye(m * k) - H
    v = IminusH @ y
    Q = fbtr(W @ np.outer(v, v))
    A = IminusH.T @ W
    B = IminusH.T @ np.diag((~na_all).astype(float))
    btrB = fbtr(B)
    # R: tBA <- sum over all pairs (r, c) of blocks of t(B_rc %x% A_rc), with B_rc = B[block r, block c] (k x k), so
    # K[(a1, a2), (b1, b2)] = sum_rc B_rc[a1, b1] * A_rc[a2, b2]  (kron index: first factor slowest, as in R and numpy)
    Bb = B.reshape(m, k, m, k)                              # Bb[r, a, c, b] = B[r*k + a, c*k + b]
    Ab = A.reshape(m, k, m, k)
    K = np.einsum('racb,rdce->adbe', Bb, Ab).reshape(k * k, k * k)
    tBA = K.T
    try:
        Psi1 = np.linalg.solve(tBA, (Q - btrB).ravel(order='F'))
    except np.linalg.LinAlgError:
        raise ValueError("method 'mm': the moment equations are singular (R: error in qr.solve)") from None
    Psi1 = Psi1.reshape((k, k), order='F')
    return _posdef_psd(0.5 * (Psi1 + Psi1.T), ctl['set.negeigen'])


def _fit_vc(Xlist, ylist, Slist, nalist, k, m, p, ctl):
    """Variance components (R's ``mvmeta.vc``; Chen, Manning & Dupuis 2012). Returns (Psi, gls, negeigen, converged,
    niter). As in R, the returned GLS (coefficients, vcov) is the one of the last iteration, i.e. evaluated at the
    Psi of the iteration before."""
    Psi = np.zeros((k, k))
    niter, converged = 0, False
    reltol = _SQRT_EPS if ctl['reltol'] is None else float(ctl['reltol'])      # R: control$reltol (not optim$reltol)
    maxiter = int(ctl['maxiter'])                                              # R: control$maxiter (not optim$maxit)
    negeigen = 0
    gls = None
    nmiss = np.sum(np.array(nalist), axis=0)                # number of studies missing each outcome
    ind = m - nmiss
    Nmat = np.minimum(ind[:, None], ind[None, :]).astype(float)
    df_corr = 0 if ctl['vc.adj'] else p
    while not converged and niter < maxiter:
        old_Psi = Psi
        gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False, nalist=nalist)
        coef = gls['coef']
        tXWXtot = sum(Xi.T @ Xi for Xi in gls['invtUX_list'])
        Lc = np.linalg.cholesky(tXWXtot)
        Lcinv = linalg.solve_triangular(Lc, np.eye(Lc.shape[0]), lower=True)
        invtXWXtot = Lcinv.T @ Lcinv
        reslist = [y_i - X_i @ coef for y_i, X_i in zip(ylist, Xlist)]
        if ctl['vc.adj']:
            adj = []
            for res, X_i, invU in zip(reslist, Xlist, gls['invU_list']):
                IH = np.eye(res.shape[0]) - X_i @ invtXWXtot @ X_i.T @ (invU @ invU.T)
                values, vectors = np.linalg.eig(IH)
                inv_sqrt = vectors @ np.diag(1.0 / np.sqrt(values.astype(complex))) @ np.linalg.inv(vectors)
                if np.max(np.abs(inv_sqrt.imag)) > 1e-8 * max(np.max(np.abs(inv_sqrt.real)), 1.0):
                    raise ValueError("method 'vc': complex eigenvalues in the variance-components adjustment")
                adj.append(inv_sqrt.real @ res)
            reslist = adj
        M = np.zeros((k, k))
        S0 = np.zeros((k, k))
        for res, S_i, na in zip(reslist, Slist, nalist):
            obs = ~na
            M[np.ix_(obs, obs)] += np.outer(res, res)
            S0[np.ix_(obs, obs)] += S_i
        Psi, negeigen = _posdef_psd(M / (Nmat - df_corr) - S0 / Nmat, ctl['set.negeigen'])
        niter += 1
        value = np.abs(Psi - old_Psi)
        converged = bool(np.all(value < reltol * np.abs(Psi + reltol)))
        if ctl['_showiter']:
            print(f'iter {niter}: value {np.max(value)}')
            if converged:
                print('converged')
    return Psi, gls, negeigen, converged, niter


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class MVMeta:
    """
    Multivariate (meta-regression) meta-analysis following R's mvmeta package.

    Usage
    -----
    mv = MVMeta(method='reml', control=None)
    mv.fit(y, S, X)        # y: (n,k), S: (n,k,k) | (n,k(k+1)/2) | (n,k), X: (n,p)
    results = blup(mv)

    Parameters
    ----------
    method : one of 'reml' (default), 'ml', 'fixed', 'mm', 'vc' (case-sensitive; anything else raises ValueError)
    control : dict of R ``mvmeta.control`` options: ``optim``, ``showiter``, ``maxiter``, ``initPsi``, ``Scor``,
        ``igls.iter``, ``hessian``, ``reltol``, ``vc.adj``, ``set.negeigen`` (``Psifix`` / ``Psicor`` are accepted and
        ignored as in R for an unstructured Psi), plus the ``mixmeta.control`` names ``igls.inititer`` (like
        ``igls.iter`` but <= 0 means no IGLS iterations), ``loglik.iter`` (validated; the optimum does not depend on
        the route), ``checkPD`` (True: ValueError on a within-study covariance with a negative eigenvalue) and
        ``addSlist`` (list of the k x k within-study covariances, instead of ``S``; ValueError if ``S`` is also given).
        Unknown names raise TypeError, ``inputna=True`` raises NotImplementedError. See the module docstring for the
        differences in defaults.

    Attributes after ``fit``
    ------------------------
    coefficients : (p, k) array; coefficient of predictor j for outcome l
    vcov         : (p*k, p*k) covariance of ``coefficients`` in outcome-major order (index l*p + j)
    psi          : (k, k) between-study covariance (zero matrix for method 'fixed')
    loglik       : maximised REML / ML log-likelihood in mvmeta's convention (NaN for 'mm' / 'vc'; 'fixed': the GLS
                   log-likelihood with Psi = 0)
    converged, niter, message, hessian (control hessian=True), na_action (input rows that were dropped, or None)
    y, S, X      : owned copies of the data of the studies used (y with NaN for missing outcomes; S (n,k,k))
    """

    def __init__(self, method: str = "reml", control: Optional[Dict] = None):
        self.method = _check_method(method)
        _mvmeta_control(control)                 # fail early on unsupported / unknown options
        self.control = dict(control) if control else {}
        self.coefficients = None   # (p, k) — matches R's coef(mv) layout
        self.vcov = None           # (p*k, p*k), outcome-major
        self.psi = None            # (k, k)
        self.loglik = None
        self.converged = False
        self.niter = 0
        self.message = None
        self.hessian = None
        self.na_action = None
        self.negeigen = None       # methods 'mm' / 'vc': number of negative eigenvalues of the unconstrained Psi

    def fit(self, y: np.ndarray, S: np.ndarray,
            X: Optional[np.ndarray] = None) -> 'MVMeta':
        """
        Parameters
        ----------
        y : (n_studies, k) estimates; NaN marks a missing outcome (studies with every outcome missing are dropped)
        S : within-study covariances (None with control['addSlist']): (n_studies, k, k) array or list of (k, k) matrices, (n_studies, k(k+1)/2) vech rows
            (lower triangle of each S_i column by column, R's ``xpndMat`` layout), or (n_studies, k) variances
            (expanded with correlation control['Scor'], default 0)
        X : (n_studies, p) study-level covariates (meta-regression); default intercept only. Studies with a missing
            covariate are dropped.

        Raises ValueError for inconsistent lengths / shapes, non-finite values, fewer than 2 usable studies and
        rank-deficient meta-regressions (as R stops), numpy.linalg.LinAlgError (a ValueError) when S_i + Psi is not
        positive definite.
        """
        method = _check_method(self.method)
        ctl = _mvmeta_control(self.control)
        S = _resolve_addSlist(S, ctl)
        data = _prepare(y, S, X, ctl)
        if ctl['checkPD'] or (ctl['checkPD'] is None and ctl['addSlist'] is not None):
            _check_S_pd(data['S'], 'S' if ctl['addSlist'] is None else 'Slist')
        # the model owns copies of the data and starts from a clean state (a failed re-fit leaves no stale results)
        self.coefficients = self.vcov = self.psi = self.loglik = self.hessian = self.negeigen = None
        self.converged = False
        self.y, self.S, self.X = data['y'], data['S'], data['X']
        self.n_input = data['n_in']
        self._kept = data['kept']
        self.na_action = (np.setdiff1d(np.arange(self.n_input), self._kept) if self._kept.size < self.n_input
                          else None)
        self.n, self.k = self.y.shape
        self.p = self.X.shape[1]
        k, p, m = self.k, self.p, self.n

        # Per-study lists restricted to the observed outcomes (R's Xlist / ylist / Slist / nalist); each X_i = I_k[obs] (x) x_i
        Xlist, ylist, Slist, nalist = _study_lists(self.y, self.S, self.X)
        self._Xlist, self._ylist, self._Slist, self._nalist = Xlist, ylist, Slist, nalist
        self.nall = int(sum(v.shape[0] for v in ylist))
        self.niter = 0
        self.message = None

        if method == 'fixed':
            Psi = np.zeros((k, k))
            gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False, nalist=nalist)
            res = gls['invtUy'] - gls['invtUX'] @ gls['coef']
            loglik = (-0.5 * self.nall * np.log(2 * np.pi) - sum(np.sum(np.log(np.diag(U))) for U in gls['Ulist'])
                      - 0.5 * np.dot(res, res))
            converged, message = True, 'fixed-effects GLS (closed form)'
        elif method == 'mm':
            Psi, self.negeigen = _fit_mm(Xlist, ylist, Slist, nalist, k, m, p, ctl)
            gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False, nalist=nalist)
            loglik, converged, message = float('nan'), True, 'method of moments (closed form)'
        elif method == 'vc':
            Psi, gls, self.negeigen, converged, self.niter = _fit_vc(Xlist, ylist, Slist, nalist, k, m, p, ctl)
            loglik, message = float('nan'), 'variance components'
            if not converged:
                message = f'variance components did not converge after {ctl["_maxit"]} iterations'
                warnings.warn("convergence not reached after maximum number of iterations (method 'vc')")
        else:
            Psi, gls, loglik, converged, message = self._fit_likelihood(method, ctl, Xlist, ylist, Slist, nalist)

        self.psi = Psi
        self.loglik = float(loglik)
        self.converged = bool(converged)
        self.message = message
        coef_vec, vcov = _finish_gls(gls)
        self.rank = int(coef_vec.shape[0])

        # coef_vec is outcome-major: [beta_{1,1}, ..., beta_{p,1}, beta_{1,2}, ..., beta_{p,k}] (outcome l owns entries
        # l*p .. l*p+p-1, because Xlist[i] = kron(I_k, x_i)); reshape to (k, p) and transpose to R's coef(mv) (p, k).
        self.coefficients = coef_vec.reshape(k, p).T
        self.vcov = vcov
        self._vcov_beta = self.vcov        # name used by earlier versions of blup()
        return self

    def _fit_likelihood(self, method, ctl, Xlist, ylist, Slist, nalist):
        k = self.k
        if ctl['initPsi'] is not None:
            initPsi = np.array(ctl['initPsi'], dtype=float)
            if initPsi.ndim == 1:
                if initPsi.shape[0] != k * (k + 1) // 2:
                    raise ValueError(f"control['initPsi'] must be a ({k}, {k}) matrix or a vector of length "
                                     f"{k * (k + 1) // 2} (vech, column by column)")
                initPsi = _xpnd(initPsi.reshape(1, -1), k)[0]
            if initPsi.shape != (k, k):
                raise ValueError(f"control['initPsi'] must be a ({k}, {k}) matrix or a vector of length "
                                 f"{k * (k + 1) // 2} (vech, column by column)")
            initPsi = 0.5 * (initPsi + initPsi.T)
            try:
                np.linalg.cholesky(initPsi)
            except np.linalg.LinAlgError:
                raise ValueError("control['initPsi'] is not positive definite") from None
        else:
            initPsi = _igls_init(Xlist, ylist, Slist, k, n_iter=ctl['igls.iter'], nalist=nalist)
        par_init = _Psi2par(initPsi)

        def fg(par):
            return _profile(par, k, Xlist, ylist, Slist, nalist, method, True)

        opt = _minimise(fg, par_init, ctl['_maxit'], ctl['_reltol'], ctl['_reltol_user'], ctl['_showiter'])
        self.niter = opt['nit']
        Psi = _par2Psi(opt['x'], k)
        gls = _glsfit(Xlist, ylist, Slist, Psi, onlycoef=False, nalist=nalist)
        if ctl['hessian']:
            self.hessian = _hessian_R(fg, opt['x'], k, ctl['_ndeps'])

        if opt['converged']:
            message = 'converged' if opt['reason'] is None else str(opt['reason'])
        elif opt['reason'] == 'maxiter':
            message = f"maximum number of iterations ({ctl['_maxit']}) reached"
            warnings.warn("MVMeta: convergence not reached after maximum number of iterations "
                          f"(maxiter = {ctl['_maxit']}); the estimates are those of the last iteration")
        else:
            message = (f"the objective cannot be improved but a Newton step is still predicted to lower it by "
                       f"{opt['predicted']:.2e} (more than reltol * |f| = {ctl['_reltol'] * abs(opt['f']):.2e}); "
                       f"the gradient is large (max |g| = {np.max(np.abs(opt['g'])):.2e})")
            warnings.warn("MVMeta: optimisation stopped without convergence: " + message)
        return Psi, gls, -opt['f'], opt['converged'], message


def blup(mv_model: MVMeta, vcov: bool = True, drop_omitted: bool = False) -> List[Dict]:
    """
    BLUPs from a fitted MVMeta model (R's blup.mvmeta).

      blup_i = pred_i + Psi @ Sigma_i^{-1} @ (y_i - pred_i)
      V_i    = X_i @ vcov(beta) @ X_i.T + Psi - Psi @ Sigma_i^{-1} @ Psi

    with Sigma_i = S_i + Psi restricted to the observed outcomes of study i (missing outcomes contribute nothing to the
    shrinkage; R approximates this with a variance of 1e10). The list has one entry per input row of ``fit`` so that it
    stays aligned with the caller's studies: studies that were dropped because of a missing covariate or all outcomes
    missing get NaN entries. R's default output (``na.action = na.omit``) has no entries for them; pass
    ``drop_omitted=True`` to get that shorter list (the used studies in input order, see ``mv_model.na_action``).
    """
    if mv_model.psi is None or mv_model.coefficients is None:
        raise ValueError("MVMeta model has not been fitted")

    Psi = mv_model.psi
    k = mv_model.k
    beta = mv_model.coefficients.T.ravel()
    results: List[Optional[Dict]] = [None] * mv_model.n_input
    eye = np.eye(k)

    for j in range(mv_model.n):
        y_i = mv_model.y[j]
        S_i = mv_model.S[j]
        X_i = np.kron(eye, mv_model.X[j:j + 1])            # (k, p*k), all outcomes (also the missing ones)
        obs = ~np.isnan(y_i)

        pred_i = X_i @ beta                                # meta-regression prediction for study j

        U = np.linalg.cholesky(S_i[np.ix_(obs, obs)] + Psi[np.ix_(obs, obs)]).T
        invU = linalg.solve_triangular(U, np.eye(U.shape[0]))
        W = np.zeros((k, k))
        W[np.ix_(obs, obs)] = invU @ invU.T                # Sigma_i^{-1}, zero rows / columns for missing outcomes
        res = np.where(obs, y_i - pred_i, 0.0)

        result = {'blup': pred_i + Psi @ W @ res}           # shrinkage toward the prediction
        if vcov:
            # uncertainty from the fixed effects + residual uncertainty after shrinkage
            result['vcov'] = X_i @ mv_model.vcov @ X_i.T + Psi - Psi @ W @ Psi
        results[mv_model._kept[j]] = result

    if drop_omitted:
        return [results[i] for i in mv_model._kept]
    for i in range(mv_model.n_input):
        if results[i] is None:
            results[i] = {'blup': np.full(k, np.nan)}
            if vcov:
                results[i]['vcov'] = np.full((k, k), np.nan)
    return results


def mvmeta(y: np.ndarray, S: np.ndarray, X: Optional[np.ndarray] = None,
           method: str = "reml", control: Optional[Dict] = None) -> MVMeta:
    """Convenience wrapper: create and fit MVMeta."""
    model = MVMeta(method=method, control=control)
    return model.fit(y, S, X)
