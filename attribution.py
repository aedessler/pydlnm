"""
Attribution functions for PyDLNM

Attributable numbers and fractions from a distributed lag non-linear model: a port of R's ``attrdl``
(Gasparrini et al., Lancet 2015, ``attrdl.R``) plus convenience wrappers for heat/cold splits and percentile
bins. Conventions follow R:

* the counterfactual is the centering value ``cen`` (null risk): exposures outside ``range`` are set to ``cen``;
* ``dir='back'`` attributes the cases of day t to the exposure history of days t-lag..t; ``dir='forw'`` attributes
  the cases of days t..t+lag (their forward moving average) to the exposure of day t, held constant along the lags;
* the attributable fraction of a total is ``sum(AN)/sum(cases)`` over the rows with a complete window, and the
  total attributable number rescales it to all observed cases;
* the coefficients are those of the cross-basis (selected by name from a model, or given as ``coef``/``vcov``,
  which are always on the log scale, e.g. the BLUPs of a multi-location analysis).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple, Any
import warnings

from basis import CrossBasis, OneBasis
from prediction import mkxpred
from centering import find_mmt
from model_utils import validate_model_compatibility
from utils import asfloat, lagmatrix, seqlag


def _match_arg(value: Any, choices: Tuple[str, ...], name: str) -> str:
    """R's match.arg: an exact or a unique partial match."""
    if isinstance(value, str) and value:
        if value in choices:
            return value
        partial = [c for c in choices if c.startswith(value)]
        if len(partial) == 1:
            return partial[0]
    raise ValueError(f"'{name}' should be one of {', '.join(repr(c) for c in choices)}")


def _resolve_coef_vcov(basis: CrossBasis, model: Optional[Any], coef, vcov,
                       name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Coefficients/vcov of the cross-basis: from the model (its block by ``name`` or by its columns, log or logit
    link required) or given."""
    if model is not None:
        info = validate_model_compatibility(model, basis.shape[1], name or "CrossBasis", kind="cb",
                                            basis=basis, name=name)
        if info['link'] not in ('log', 'logit'):
            raise ValueError("'model' must have a log or logit link function")
        return info['coef'], info['vcov']
    if coef is None or vcov is None:
        raise ValueError("arguments 'basis' do not match 'model' or 'coef'-'vcov'")
    return asfloat(coef).ravel(), np.atleast_2d(asfloat(vcov))


def _resolve_cen(cen: Optional[float], basis: CrossBasis) -> float:
    """R: cen must be given, or be stored in the basis (argvar$cen)."""
    if cen is None:
        cen = basis.argvar.get('cen')
    if cen is None:
        raise ValueError("'cen' must be provided")
    if np.ndim(cen) != 0:
        raise ValueError("'cen' must be a numeric scalar")
    return float(cen)


def attrdl(x: np.ndarray,
           basis: CrossBasis,
           cases: np.ndarray,
           model: Optional[Any] = None,
           coef: Optional[np.ndarray] = None,
           vcov: Optional[np.ndarray] = None,
           type: str = "an",
           dir: str = "forw",
           tot: bool = True,
           cen: Optional[float] = None,
           range: Optional[Tuple[float, float]] = None,
           sim: bool = False,
           nsim: int = 5000,
           sub: Optional[np.ndarray] = None,
           name: Optional[str] = None) -> Dict:
    """
    Attributable numbers and fractions from a distributed lag model (port of R's ``attrdl``).

    Parameters
    ----------
    x : array-like
        Exposure series, or (only ``dir='back'``) a matrix of lagged exposures.
    basis : CrossBasis
        Cross-basis computed from ``x``.
    cases : array-like
        Cases series, or (only ``dir='forw'``) the matrix of future cases.
    model : fitted model, optional
        Its cross-basis coefficients are selected as in ``crosspred`` (by ``name``, else by the columns of the
        cross-basis among the design columns of the model, else by the names ``v1.l1`` ...); the link must be log
        or logit (inferred as R's getlink does, including Cox, conditional logit and conditional Poisson models).
    coef, vcov : array-like, optional
        Coefficients and covariance of the basis when ``model`` is not given (log scale). Fewer coefficients
        than basis columns are the reduced (overall-effect) coefficients; only ``dir='forw'`` is possible then.
    type : {'an', 'af', 'both'}, default 'an'
        Attributable number or fraction (R has 'an' and 'af'); abbreviations are accepted.
    dir : {'forw', 'back'}, default 'forw'
        Forward or backward perspective (R's default is 'back'); abbreviations are accepted.
    tot : bool, default True
        Also compute the total attributable number/fraction.
    cen : float, optional
        Reference (counterfactual) exposure. Required unless stored in ``basis.argvar['cen']``.
    range : tuple, optional
        Exposure range counted in the attribution (closed interval); values outside it are set to ``cen``.
    sim : bool, default False
        Also return ``nsim`` simulated totals and their 2.5%-97.5% interval (needs ``tot=True``).
    nsim : int, default 5000
        Number of simulations.
    sub : array-like of bool, optional
        Observations (rows) to attribute, e.g. the days of one summer. The lag windows (lagged exposures for
        ``dir='back'``, forward moving average of the cases for ``dir='forw'``) are built on the FULL series and
        ``sub`` only selects rows afterwards, as the Europe-2022 script does for its sub-periods; the per-observation
        results have one entry per kept row, and the totals use the kept rows with a complete window
        (``den`` = observed cases of the kept rows). R's ``attrdl`` has no ``sub``: this equals its ``tot=FALSE``
        result restricted to the rows.
    name : str, optional
        Name of the cross-basis in ``model``: prefix of the coefficient names of its terms (R: the name of the
        basis object); selects its block in a model that holds several cross-bases.

    Returns
    -------
    dict
        ``'an'``, ``'af'``: per-observation attributable numbers/fractions, aligned with ``x`` (NaN where the
        exposure, the cases or the lag window are incomplete); with ``tot``: ``'an_total'``, ``'af_total'``;
        ``'metadata'``; with ``sim``: ``'ci'`` and ``'sim_results'`` (``['simulations']['an_total']`` ...).
    """
    type = _match_arg(type, ("an", "af", "both"), "type")
    dir = _match_arg(dir, ("back", "forw"), "dir")

    x = asfloat(x, copy=True)               # copies: the inputs are never modified; masked cells are NaN (R's NA)
    cases = asfloat(cases, copy=True)
    if sub is not None:
        sub = np.asarray(sub, dtype=bool)
        if sub.shape != (len(x),):
            raise ValueError("sub must be a logical vector with one value per observation of x")

    cen = _resolve_cen(cen, basis)

    # Exposure outside the range is set to the centering value (null risk)
    if range is not None:
        x = np.where((x < range[0]) | (x > range[1]), cen, x)

    # Matrix of lagged exposures (dir='back'), or the exposure held constant along the lags (dir='forw')
    lag = np.asarray(basis.lag)
    lags = seqlag(lag)
    n_lag = len(lags)
    if x.ndim == 1 or x.shape[1] == 1:
        x = x.ravel()
        at = lagmatrix(x, lags) if dir == "back" else x        # forw: the same exposure at every lag
    else:
        if dir == "forw":
            raise ValueError("'x' must be a vector when dir='forw'")
        if x.shape[1] != n_lag:
            raise ValueError("dimension of 'x' not compatible with 'basis'")
        at = x

    # Cases attributed to each row; 'den' is the observed total used to rescale totals
    if cases.shape[0] != at.shape[0]:
        raise ValueError("'x' and 'cases' not consistent")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if cases.ndim == 2 and cases.shape[1] > 1:
            if dir == "back":
                raise ValueError("'cases' must be a vector if dir='back'")
            if cases.shape[1] != n_lag:
                raise ValueError("dimension of 'cases' not compatible")
            den = np.nansum(np.nanmean(cases if sub is None else cases[sub], axis=1))
            cases = cases.mean(axis=1)
        else:
            cases = cases.ravel()
            den = np.nansum(cases if sub is None else cases[sub])
            if dir == "forw":
                cases = lagmatrix(cases, -lags).mean(axis=1)

    # Sub-period: the lag windows above come from the FULL series (a window may reach outside the sub-period);
    # only the rows of 'sub' are attributed, and 'den' is the observed total of those rows
    if sub is not None:
        at, cases = at[sub], cases[sub]
        if x.ndim == 1:
            x = x[sub]

    # Coefficients and the design matrix summed over the lags
    coef_vec, vcov_mat = _resolve_coef_vcov(basis, model, coef, vcov, name)
    reduced = len(coef_vec) != basis.shape[1]
    if reduced:
        if dir == "back":
            raise ValueError("only dir='forw' allowed for reduced estimates")
        x_all = mkxpred(OneBasis(x, **{k: v for k, v in basis.argvar.items() if k != 'cen'}), x, None, cen)[:, 0, :]
    else:
        x_all = mkxpred(basis, at, lags, cen).sum(axis=1)
    if len(coef_vec) != x_all.shape[1]:
        raise ValueError("arguments 'basis' do not match 'model' or 'coef'-'vcov'")
    if vcov_mat.shape != (len(coef_vec), len(coef_vec)):
        raise ValueError("arguments 'coef' and 'vcov' do no match")

    # Attributable fraction and number per observation (log or logit link: RR = exp(eta))
    af_obs = 1.0 - np.exp(-(x_all @ coef_vec))
    an_obs = af_obs * cases

    results: Dict[str, Any] = {'an': an_obs, 'af': af_obs}

    # Totals: rows with a complete window, AF = sum(AN)/sum(cases), total AN rescaled to all observed cases
    valid = ~np.isnan(an_obs)
    if tot:
        with np.errstate(invalid="ignore", divide="ignore"):
            af_total = np.sum(an_obs[valid]) / np.sum(cases[valid])
        results['af_total'] = af_total
        results['an_total'] = af_total * den

    results['metadata'] = {
        'type': type,
        'direction': dir,
        'centering': cen,
        'range': range,
        'n_obs': len(at),
        'n_cases': den,
        'simulation': bool(sim and tot),
    }

    # Empirical confidence intervals from coefficients sampled from their sampling distribution
    if sim and not tot:
        warnings.warn("simulation samples only returned for tot=True")
    elif sim:
        draws = _simulate_totals(x_all, cases, den, coef_vec, vcov_mat, int(nsim))
        ci = {}
        for key, values in draws.items():
            ci[key + '_low'] = np.percentile(values, 2.5)
            ci[key + '_high'] = np.percentile(values, 97.5)
        results['ci'] = ci
        results['sim_results'] = {'simulations': draws, 'nsim_successful': int(nsim)}

    return results


def _simulate_totals(x_all: np.ndarray, cases: np.ndarray, den: float, coef: np.ndarray, vcov: np.ndarray,
                     nsim: int, chunk: int = 500) -> Dict[str, np.ndarray]:
    """Simulated total AF/AN: coefficients drawn with the eigen decomposition of vcov (as R's attrdl); the design
    matrix is built once and each draw costs one matrix product."""
    k = len(coef)
    values, vectors = np.linalg.eigh(asfloat(vcov))
    root = vectors * np.sqrt(np.clip(values, 0.0, None))          # vectors %*% diag(sqrt(values))
    z = np.random.standard_normal((nsim, k))
    coef_sim = coef[:, None] + root @ z.T                         # k x nsim
    af_sim = np.empty(nsim)
    for start in np.arange(0, nsim, chunk):
        block = coef_sim[:, start:start + chunk]
        an_block = (1.0 - np.exp(-(x_all @ block))) * cases[:, None]
        ok = ~np.isnan(an_block)
        with np.errstate(invalid="ignore", divide="ignore"):
            af_sim[start:start + chunk] = (np.where(ok, an_block, 0.0).sum(axis=0) /
                                           np.where(ok, cases[:, None], 0.0).sum(axis=0))
    return {'af_total': af_sim, 'an_total': af_sim * den}


def _resolve_wrapper_cen(cen, basis, model, coef, vcov, name=None) -> float:
    """Centering of the wrappers: given, stored in the basis, else the minimum-risk exposure (MMT)."""
    if cen is None:
        cen = basis.argvar.get('cen')
    if cen is None:
        cen = find_mmt(basis, model, coef=coef, vcov=vcov, name=name)['mmt']
    return float(cen)


def attr_heat_cold(x: np.ndarray,
                   basis: CrossBasis,
                   cases: np.ndarray,
                   model: Optional[Any] = None,
                   coef: Optional[np.ndarray] = None,
                   vcov: Optional[np.ndarray] = None,
                   percentiles: Tuple[float, float] = (2.5, 97.5),
                   cen: Optional[float] = None,
                   sim: bool = False,
                   nsim: int = 5000,
                   split: str = "cen",
                   name: Optional[str] = None) -> Dict:
    """
    Heat and cold attributable risks.

    With ``split='cen'`` (default) cold is the exposure below the centering value and heat the exposure above it
    (R's ``range=c(-Inf, cen)`` and ``range=c(cen, Inf)``), so that cold + heat is the total. With
    ``split='percentile'`` they are the tails below/above the ``percentiles`` of ``x`` (computed ignoring NaN).
    ``cen`` defaults to the value stored in the basis, else the minimum-risk exposure. ``name`` is the name of the
    cross-basis in ``model`` (see ``attrdl``).
    """
    split = _match_arg(split, ("cen", "percentile"), "split")
    x = asfloat(x)                      # masked / nullable cells are NaN (R's NA)
    cen = _resolve_wrapper_cen(cen, basis, model, coef, vcov, name)

    if split == "cen":
        cold_threshold = heat_threshold = cen
    else:
        cold_threshold, heat_threshold = np.nanpercentile(x, percentiles)

    cold_results = attrdl(x, basis, cases, model, coef, vcov, type="both",
                          range=(-np.inf, cold_threshold), cen=cen, sim=sim, nsim=nsim, name=name)
    heat_results = attrdl(x, basis, cases, model, coef, vcov, type="both",
                          range=(heat_threshold, np.inf), cen=cen, sim=sim, nsim=nsim, name=name)

    return {
        'cold': {'threshold': cold_threshold, 'percentile': percentiles[0] if split == "percentile" else None,
                 'results': cold_results},
        'heat': {'threshold': heat_threshold, 'percentile': percentiles[1] if split == "percentile" else None,
                 'results': heat_results},
        'summary': {
            'cold_an_total': cold_results['an_total'],
            'heat_an_total': heat_results['an_total'],
            'cold_af_total': cold_results['af_total'],
            'heat_af_total': heat_results['af_total'],
            'total_attributable': cold_results['an_total'] + heat_results['an_total'],
        },
        'metadata': {
            'split': split,
            'percentiles': percentiles,
            'centering': cen,
            'simulation': sim,
            'n_obs': len(x),
        },
    }


def attr_by_percentiles(x: np.ndarray,
                        basis: CrossBasis,
                        cases: np.ndarray,
                        model: Optional[Any] = None,
                        coef: Optional[np.ndarray] = None,
                        vcov: Optional[np.ndarray] = None,
                        percentile_ranges: List[Tuple[float, float]] = None,
                        cen: Optional[float] = None,
                        sim: bool = False,
                        nsim: int = 5000,
                        name: Optional[str] = None) -> Dict:
    """
    Attributable risks by percentile bins of the exposure.

    Each bin is ``[p_low, p_high)`` of the percentiles of ``x`` (computed ignoring NaN), closed on the right for
    a bin ending at the 100th percentile, so that contiguous bins count every observation once.
    ``cen`` defaults to the value stored in the basis, else the minimum-risk exposure. ``name`` is the name of the
    cross-basis in ``model`` (see ``attrdl``).
    """
    if percentile_ranges is None:
        percentile_ranges = [(0, 1), (1, 5), (5, 10), (90, 95), (95, 99), (99, 100)]

    x = asfloat(x)                      # masked / nullable cells are NaN (R's NA)
    cen = _resolve_wrapper_cen(cen, basis, model, coef, vcov, name)
    results = {}

    for low_pct, high_pct in percentile_ranges:
        low_threshold, high_threshold = np.nanpercentile(x, [low_pct, high_pct])
        upper = high_threshold if high_pct >= 100 else np.nextafter(high_threshold, -np.inf)
        range_results = attrdl(x, basis, cases, model, coef, vcov, type="both",
                               range=(low_threshold, upper), cen=cen, sim=sim, nsim=nsim, name=name)
        results[f"pct_{low_pct}_{high_pct}"] = {
            'percentiles': (low_pct, high_pct),
            'thresholds': (low_threshold, high_threshold),
            'results': range_results,
        }

    keys = [f"pct_{p[0]}_{p[1]}" for p in percentile_ranges]
    summary = pd.DataFrame({
        'percentile_range': [f"{p[0]}-{p[1]}" for p in percentile_ranges],
        'low_threshold': [results[k]['thresholds'][0] for k in keys],
        'high_threshold': [results[k]['thresholds'][1] for k in keys],
        'an_total': [results[k]['results']['an_total'] for k in keys],
        'af_total': [results[k]['results']['af_total'] for k in keys],
    })
    if sim:
        summary['an_total_low'] = [results[k]['results']['ci']['an_total_low'] for k in keys]
        summary['an_total_high'] = [results[k]['results']['ci']['an_total_high'] for k in keys]

    results['summary_table'] = summary
    results['metadata'] = {
        'percentile_ranges': percentile_ranges,
        'centering': cen,
        'simulation': sim,
        'n_ranges': len(percentile_ranges),
    }
    return results


class AttributionManager:
    """
    Manager class for attribution calculations
    """

    def __init__(self, x: np.ndarray, basis: CrossBasis, cases: np.ndarray,
                 model: Optional[Any] = None, coef: Optional[np.ndarray] = None,
                 vcov: Optional[np.ndarray] = None, name: Optional[str] = None):
        """Initialize attribution manager (``name``: name of the cross-basis in ``model``, see ``attrdl``)"""
        self.x = asfloat(x)                      # masked / nullable cells are NaN (R's NA)
        self.basis = basis
        self.cases = asfloat(cases)
        self.model = model
        self.coef = coef
        self.vcov = vcov
        self.name = name

        # Cache for results
        self._mmt_cache = None
        self._attribution_cache = {}

    def get_mmt(self) -> float:
        """Get MMT with caching"""
        if self._mmt_cache is None:
            mmt_result = find_mmt(self.basis, self.model, coef=self.coef, vcov=self.vcov, name=self.name)
            self._mmt_cache = mmt_result['mmt']
        return self._mmt_cache

    def total_attribution(self, **kwargs) -> Dict:
        """Calculate total attributable risk"""
        kwargs.setdefault('name', self.name)
        return attrdl(self.x, self.basis, self.cases,
                     self.model, self.coef, self.vcov, **kwargs)

    def heat_cold_attribution(self, **kwargs) -> Dict:
        """Calculate heat and cold attribution"""
        kwargs.setdefault('name', self.name)
        return attr_heat_cold(self.x, self.basis, self.cases,
                             self.model, self.coef, self.vcov, **kwargs)

    def percentile_attribution(self, **kwargs) -> Dict:
        """Calculate attribution by percentiles"""
        kwargs.setdefault('name', self.name)
        return attr_by_percentiles(self.x, self.basis, self.cases,
                                  self.model, self.coef, self.vcov, **kwargs)

    def summary_report(self, sim: bool = True, nsim: int = 1000) -> Dict:
        """Generate comprehensive attribution report"""

        # Get MMT
        mmt = self.get_mmt()

        # Total attribution
        total_attr = self.total_attribution(cen=mmt, sim=sim, nsim=nsim)

        # Heat/cold attribution
        heat_cold_attr = self.heat_cold_attribution(cen=mmt, sim=sim, nsim=nsim)

        # Percentile attribution
        pct_attr = self.percentile_attribution(cen=mmt, sim=sim, nsim=nsim)

        return {
            'mmt': mmt,
            'total': total_attr,
            'heat_cold': heat_cold_attr,
            'percentiles': pct_attr,
            'summary_statistics': {
                'total_cases': np.nansum(self.cases),
                'mean_exposure': np.nanmean(self.x),
                'exposure_range': (np.nanmin(self.x), np.nanmax(self.x)),
                'n_observations': len(self.x)
            }
        }
