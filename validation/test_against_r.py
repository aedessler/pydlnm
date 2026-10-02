#!/usr/bin/env python3
"""
End-to-end check of PyDLNM against the R dlnm reference (2015 Gasparrini Lancet dataset, England & Wales).

Run from anywhere:  python validation/test_against_r.py     (exit status 1 if any check fails)

R must be startable by rpy2 with the packages dlnm and splines (set R_HOME before running if the R on the PATH is not
the right one; this script never overrides it).

Three comparison stages, each asserted at the level the agreement is expected to reach:
  Stage 1 — First stage (crossbasis + GLM + crossreduce): reduced coefficients AND vcov per region against R's
            saved coefficients.rds / vcov_matrices.rds                               (machine precision, < 1e-10)
  Stage 2 — Prediction with R's own BLUPs: PyDLNM crosspred on blup_results.rds against R's RR curves, and the MMT
            PyDLNM finds on the same grid against R's                                (machine precision, < 1e-10)
  Stage 3 — Full pipeline (first stage + PyDLNM MVMeta + BLUP + prediction)
            The meta-analysis is optimiser-limited: R's mvmeta stops its optim at reltol = sqrt(eps), so the BLUPs
            and RR curves agree with R's default output to ~1e-5, not to machine precision   (BLUP, RR < 1e-4)
Regenerate the reference with validation/generate_r_reference.R.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import rpy2.robjects as ro                      # noqa: E402  (R is started here, before any PyDLNM import)
from rpy2.robjects import numpy2ri              # noqa: E402
from rpy2.robjects.conversion import localconverter   # noqa: E402

from basis import CrossBasis                    # noqa: E402
from prediction import crosspred                # noqa: E402
from improved_glm import ImprovedGLMInterface   # noqa: E402
from utils import logknots                      # noqa: E402
from meta_analysis import MVMeta, blup          # noqa: E402

# ── Parameters (matching R 00.prepdata.R) ─────────────────────────────────────
VARFUN = "bs"; VARDEGREE = 2; VARPER = [10, 75, 90]
LAG = 21; LAGNK = 3; DFSEAS = 8
DATA_PATH = HERE.parent / '2015_gasparrini_Lancet_Rcodedata-master' / 'regEngWales.csv'
RESULTS_DIR = HERE / 'reference_data'

# Tolerances (absolute, on coefficients / relative risks), see the module docstring
TOL_EXACT = 1e-10          # stages 1 and 2
TOL_META = 1e-4            # stage 3 (measured ~6e-6 for the BLUPs, ~1.4e-5 for the RR curves)

CODE_TO_NAME = {
    'N-East': 'North East', 'N-West': 'North West', 'York&Hum': 'Yorkshire & Humber',
    'E-Mid': 'East Midlands', 'W-Mid': 'West Midlands', 'East': 'East',
    'London': 'London', 'S-East': 'South East', 'S-West': 'South West', 'Wales': 'Wales',
}
SORTED_CODES = sorted(CODE_TO_NAME, key=lambda k: CODE_TO_NAME[k])
SORTED_NAMES = [CODE_TO_NAME[c] for c in SORTED_CODES]

failures = []


def check(ok: bool, message: str):
    print(('  ✓ ' if ok else '  ✗ ') + message)
    if not ok:
        failures.append(message)


def r_array(expr: str) -> np.ndarray:
    with localconverter(ro.default_converter + numpy2ri.converter):
        return np.array(ro.r(expr))


# ── Load data & R reference ───────────────────────────────────────────────────
df_all = pd.read_csv(DATA_PATH, index_col=0)
df_all['date'] = pd.to_datetime(df_all['date'])

ro.r(f'''
suppressMessages({{ library(dlnm); library(splines) }})
r_coef <- readRDS("{RESULTS_DIR / 'coefficients.rds'}")
r_blup <- readRDS("{RESULTS_DIR / 'blup_results.rds'}")
r_vcov <- readRDS("{RESULTS_DIR / 'vcov_matrices.rds'}")
''')
r_coef_mat = r_array('r_coef')
r_region_codes = list(ro.r('rownames(r_coef)'))
r_code_idx = {c: i for i, c in enumerate(r_region_codes)}

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: First-stage per-region analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("STAGE 1 — First stage: CrossBasis + GLM + CrossReduce (coef and vcov vs R)")
print("═" * 70)

py_coef_list, py_vcov_list, region_data = [], [], {}

for code in SORTED_CODES:
    name = CODE_TO_NAME[code]
    df = df_all[df_all['regnames'] == code].copy().reset_index(drop=True)
    knots_var = np.quantile(df['tmean'].dropna(), [p / 100 for p in VARPER])
    lag_knots = logknots([0, LAG], nk=LAGNK)

    cb = CrossBasis(x=df['tmean'].values, lag=LAG,
                    argvar={'fun': VARFUN, 'knots': knots_var, 'degree': VARDEGREE},
                    arglag={'fun': 'ns', 'knots': lag_knots})

    glm = ImprovedGLMInterface(cb)
    glm.fit_dlnm_model(y=df['death'].values, dates=df['date'],
                       dfseas=DFSEAS, family='quasipoisson')

    cen = float(df['tmean'].mean())
    red = glm.crossreduce(cen=cen)
    py_coef_list.append(red.coef)
    py_vcov_list.append(red.vcov)
    region_data[name] = {'cb': cb, 'cen': cen, 'df': df}

py_coef_mat = np.vstack(py_coef_list)

coef_diffs, vcov_diffs = [], []
for py_idx, code in enumerate(SORTED_CODES):
    r_idx = r_code_idx[code]
    d_coef = np.abs(py_coef_mat[py_idx] - r_coef_mat[r_idx]).max()
    d_vcov = np.abs(py_vcov_list[py_idx] - r_array(f'r_vcov[[{r_idx + 1}]]')).max()
    coef_diffs.append(d_coef)
    vcov_diffs.append(d_vcov)
    check(d_coef < TOL_EXACT and d_vcov < TOL_EXACT,
          f"{CODE_TO_NAME[code]:20s} max|Δcoef| = {d_coef:.2e}  max|Δvcov| = {d_vcov:.2e}")
print(f"\n  → worst region: max|Δcoef| = {max(coef_diffs):.2e}, max|Δvcov| = {max(vcov_diffs):.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: RR curves using R's own BLUPs → tests prediction code in isolation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("STAGE 2 — Prediction using R's BLUPs (crosspred in isolation)")
print("═" * 70)

stage2_max = []
for py_idx, code in enumerate(SORTED_CODES):
    name = CODE_TO_NAME[code]
    r_idx = r_code_idx[code]
    r_blup_coef = r_array(f'r_blup[[{r_idx + 1}]]$blup')
    r_blup_vcov = r_array(f'r_blup[[{r_idx + 1}]]$vcov')

    csv_path = RESULTS_DIR / f"rr_curve_{code.replace(' & ', '___').replace(' ', '_')}.csv"
    r_curve = pd.read_csv(csv_path)
    grid = r_curve['temperature'].values
    mmt = float(r_curve['mmt'].iloc[0])

    pred = crosspred(basis=region_data[name]['cb'], coef=r_blup_coef, vcov=r_blup_vcov,
                     model_link='log', at=grid, cen=mmt)
    d_rr = np.abs(r_curve['rr_fit'].values - pred.allRRfit).max()
    stage2_max.append(d_rr)

    # MMT of the reference rule (minimum of the BLUP curve on the reference grid) found by PyDLNM on the same grid
    uncentred = crosspred(basis=region_data[name]['cb'], coef=r_blup_coef, vcov=r_blup_vcov,
                          model_link='log', at=grid, cen=False)
    mmt_py = float(grid[np.argmin(uncentred.allfit)])
    check(d_rr < TOL_EXACT and mmt_py == mmt,
          f"{name:20s} max|ΔRR| = {d_rr:.2e}   MMT R {mmt:.2f} / PyDLNM {mmt_py:.2f}")
print(f"\n  → worst region: max|ΔRR| = {max(stage2_max):.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Full pipeline (PyDLNM coef → PyDLNM MVMeta → BLUP → RR curves)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("STAGE 3 — Full pipeline with PyDLNM MVMeta (optimiser-limited, tolerance %.0e)" % TOL_META)
print("═" * 70)

avg_t = np.array([region_data[n]['df']['tmean'].mean() for n in SORTED_NAMES])
range_t = np.array([region_data[n]['df']['tmean'].max() - region_data[n]['df']['tmean'].min()
                    for n in SORTED_NAMES])
# R's formula ~avgtmean+rangetmean auto-adds an intercept → 3 columns: [1, avg, range]
S_meta = np.column_stack([np.ones(len(SORTED_NAMES)), avg_t, range_t])
vcov_3d = np.stack(py_vcov_list, axis=0)

mv = MVMeta()
mv.fit(y=py_coef_mat, S=vcov_3d, X=S_meta)
blup_results = blup(mv)
check(bool(mv.converged), f"MVMeta converged (loglik = {mv.loglik:.6f})")

blup_diffs, stage3_max = [], []
for py_idx, code in enumerate(SORTED_CODES):
    name = CODE_TO_NAME[code]
    r_idx = r_code_idx[code]
    d_blup = np.abs(r_array(f'r_blup[[{r_idx + 1}]]$blup') - blup_results[py_idx]['blup']).max()
    blup_diffs.append(d_blup)

    r_curve = pd.read_csv(RESULTS_DIR / f"rr_curve_{code.replace(' & ', '___').replace(' ', '_')}.csv")
    pred = crosspred(basis=region_data[name]['cb'],
                     coef=blup_results[py_idx]['blup'], vcov=blup_results[py_idx]['vcov'],
                     model_link='log', at=r_curve['temperature'].values, cen=float(r_curve['mmt'].iloc[0]))
    d_rr = np.abs(r_curve['rr_fit'].values - pred.allRRfit).max()
    stage3_max.append(d_rr)
    check(d_blup < TOL_META and d_rr < TOL_META,
          f"{name:20s} max|ΔBLUP| = {d_blup:.2e}   max|ΔRR| = {d_rr:.2e}")
print(f"\n  → worst region: max|ΔBLUP| = {max(blup_diffs):.2e}, max|ΔRR| = {max(stage3_max):.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SUMMARY")
print("═" * 70)
print(f"  Stage 1 (first stage):    worst max|Δcoef| = {max(coef_diffs):.2e}, max|Δvcov| = {max(vcov_diffs):.2e}")
print(f"  Stage 2 (R's BLUPs):      worst max|ΔRR|   = {max(stage2_max):.2e}")
print(f"  Stage 3 (full pipeline):  worst max|ΔRR|   = {max(stage3_max):.2e}  (optimiser-limited)")
if failures:
    print(f"\n{len(failures)} CHECK(S) FAILED")
    sys.exit(1)
print("\nALL CHECKS PASSED")
