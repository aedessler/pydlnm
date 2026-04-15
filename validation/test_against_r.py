#!/usr/bin/env python3
"""
End-to-end test: run PyDLNM against R DLNM reference (2015 Gasparrini Lancet dataset).

Three comparison stages:
  Stage 1 — First-stage (crossbasis + GLM + crossreduce)
             Python reduced coefficients vs R's saved coefficients.rds
  Stage 2 — Prediction using R's own BLUPs
             Feeds R's blup_results.rds directly into PyDLNM crosspred()
             to isolate whether the prediction code is correct
  Stage 3 — Full pipeline (first-stage + PyDLNM MVMeta + BLUP + prediction)
             Shows remaining MVMeta differences
"""

import os, sys
import numpy as np
import pandas as pd

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
sys.path.insert(0, '/Users/adessler/Desktop/DLNM')

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

from basis import CrossBasis
from prediction import crosspred
from improved_glm import ImprovedGLMInterface
from utils import logknots
from meta_analysis import MVMeta, blup

# ── Parameters (matching R 00.prepdata.R) ─────────────────────────────────────
VARFUN    = "bs";  VARDEGREE = 2;  VARPER = [10, 75, 90]
LAG = 21;  LAGNK = 3;  DFSEAS = 8
DATA_PATH   = '2015_gasparrini_Lancet_Rcodedata-master/regEngWales.csv'
RESULTS_DIR = 'reference_data'

CODE_TO_NAME = {
    'N-East':'North East','N-West':'North West','York&Hum':'Yorkshire & Humber',
    'E-Mid':'East Midlands','W-Mid':'West Midlands','East':'East',
    'London':'London','S-East':'South East','S-West':'South West','Wales':'Wales',
}
SORTED_CODES = sorted(CODE_TO_NAME, key=lambda k: CODE_TO_NAME[k])
SORTED_NAMES = [CODE_TO_NAME[c] for c in SORTED_CODES]

# ── Load data & R reference ───────────────────────────────────────────────────
df_all = pd.read_csv(DATA_PATH, index_col=0)
df_all['date'] = pd.to_datetime(df_all['date'])

ro.r(f'''
suppressMessages({{ library(dlnm); library(splines) }})
r_coef <- readRDS("{RESULTS_DIR}/coefficients.rds")
r_blup <- readRDS("{RESULTS_DIR}/blup_results.rds")
r_vcov <- readRDS("{RESULTS_DIR}/vcov_matrices.rds")
''')
with localconverter(ro.default_converter + numpy2ri.converter):
    r_coef_mat    = np.array(ro.r('r_coef'))
    r_region_codes = list(ro.r('rownames(r_coef)'))
r_code_idx = {c: i for i, c in enumerate(r_region_codes)}

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: First-stage per-region analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("STAGE 1 — First-stage: CrossBasis + GLM + CrossReduce")
print("═"*70)

py_coef_list, py_vcov_list, region_data = [], [], {}

for code in SORTED_CODES:
    name = CODE_TO_NAME[code]
    df   = df_all[df_all['regnames'] == code].copy().reset_index(drop=True)
    knots_var = np.quantile(df['tmean'].dropna(), [p/100 for p in VARPER])
    lag_knots = logknots([0, LAG], nk=LAGNK)

    cb = CrossBasis(x=df['tmean'].values, lag=LAG,
                    argvar={'fun': VARFUN, 'knots': knots_var, 'degree': VARDEGREE},
                    arglag={'fun': 'ns',   'knots': lag_knots})

    glm = ImprovedGLMInterface(cb)
    glm.fit_dlnm_model(y=df['death'].values, dates=df['date'],
                       dfseas=DFSEAS, family='quasipoisson')

    cen = float(df['tmean'].mean())
    red = glm.crossreduce(cen=cen)
    py_coef_list.append(red.coef)
    py_vcov_list.append(red.vcov)
    region_data[name] = {'cb': cb, 'cen': cen, 'df': df}

py_coef_mat = np.vstack(py_coef_list)

print("\nFirst-stage coefficient comparison (Python vs R):")
coef_diffs = []
for py_idx, code in enumerate(SORTED_CODES):
    name  = CODE_TO_NAME[code]
    r_idx = r_code_idx.get(code)
    if r_idx is None: continue
    diff  = np.abs(py_coef_mat[py_idx] - r_coef_mat[r_idx])
    coef_diffs.append(diff.max())
    ok = "✓" if diff.max() < 1e-8 else ("~" if diff.max() < 0.01 else "✗")
    print(f"  {ok} {name:25s}  max|Δcoef|={diff.max():.2e}")
print(f"\n  → All 10 regions: mean max|Δcoef| = {np.mean(coef_diffs):.2e}"
      f"  ({'EXACT MATCH' if np.mean(coef_diffs) < 1e-8 else 'MISMATCH'})")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: RR curves using R's own BLUPs → tests prediction code in isolation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("STAGE 2 — Prediction using R's BLUPs (tests crosspred in isolation)")
print("═"*70)

rr_rows_stage2 = []
for py_idx, code in enumerate(SORTED_CODES):
    name  = CODE_TO_NAME[code]
    r_idx = r_code_idx.get(code)
    if r_idx is None: continue

    # Load R's BLUP coef/vcov directly
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_blup_coef = np.array(ro.r(f'r_blup[[{r_idx+1}]]$blup'))
        r_blup_vcov = np.array(ro.r(f'r_blup[[{r_idx+1}]]$vcov'))

    # Find reference CSV
    csv_stem = name.replace(' & ', '___').replace(' ', '_')
    csv_path = f"{RESULTS_DIR}/rr_curve_{csv_stem}.csv"
    if not os.path.exists(csv_path): continue
    r_curve = pd.read_csv(csv_path)
    mmt = float(r_curve['mmt'].iloc[0])

    pred = crosspred(basis=region_data[name]['cb'],
                     coef=r_blup_coef, vcov=r_blup_vcov,
                     model_link='log',
                     at=r_curve['temperature'].values, cen=mmt)

    diff  = np.abs(r_curve['rr_fit'].values - pred.allRRfit)
    rr_rows_stage2.append({'region': name, 'max': diff.max(), 'mean': diff.mean(),
                            'corr': float(np.corrcoef(r_curve['rr_fit'].values, pred.allRRfit)[0,1])})

print("\nRR curve comparison using R's BLUPs (R BLUP → PyDLNM crosspred → compare vs R curve):")
for r in rr_rows_stage2:
    ok = "✓" if r['max'] < 0.005 else ("~" if r['max'] < 0.05 else "✗")
    print(f"  {ok} {r['region']:25s}  max|ΔRR|={r['max']:.5f}  mean|ΔRR|={r['mean']:.5f}  corr={r['corr']:.7f}")
print(f"\n  → Mean max|ΔRR| = {np.mean([r['max'] for r in rr_rows_stage2]):.6f}"
      f"  ({'PASS' if np.mean([r['max'] for r in rr_rows_stage2]) < 0.005 else 'REVIEW'})")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Full pipeline (PyDLNM coef → PyDLNM MVMeta → BLUP → RR curves)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("STAGE 3 — Full pipeline with PyDLNM MVMeta")
print("═"*70)

avg_t   = np.array([region_data[n]['df']['tmean'].mean() for n in SORTED_NAMES])
range_t = np.array([region_data[n]['df']['tmean'].max() - region_data[n]['df']['tmean'].min()
                    for n in SORTED_NAMES])
# R's formula ~avgtmean+rangetmean auto-adds intercept → 3 columns: [1, avg, range]
S_meta  = np.column_stack([np.ones(len(SORTED_NAMES)), avg_t, range_t])
vcov_3d = np.stack(py_vcov_list, axis=0)

mv = MVMeta()
mv.fit(y=py_coef_mat, S=vcov_3d, X=S_meta)
blup_results = blup(mv)

print("\nBLUP comparison (PyDLNM MVMeta vs R mvmeta):")
blup_diffs = []
for py_idx, code in enumerate(SORTED_CODES):
    name  = CODE_TO_NAME[code]
    r_idx = r_code_idx.get(code)
    if r_idx is None: continue
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_blup_i = np.array(ro.r(f'r_blup[[{r_idx+1}]]$blup'))
    diff = np.abs(r_blup_i - blup_results[py_idx]['blup'])
    blup_diffs.append(diff.max())
    ok = "✓" if diff.max() < 0.01 else ("~" if diff.max() < 0.1 else "✗")
    print(f"  {ok} {name:25s}  max|ΔBLUP|={diff.max():.5f}")
print(f"\n  → Mean max|ΔBLUP| = {np.mean(blup_diffs):.5f}"
      f"  (MVMeta differences are a known separate issue)")

rr_rows_stage3 = []
for py_idx, code in enumerate(SORTED_CODES):
    name  = CODE_TO_NAME[code]
    csv_stem = name.replace(' & ', '___').replace(' ', '_')
    csv_path = f"{RESULTS_DIR}/rr_curve_{csv_stem}.csv"
    if not os.path.exists(csv_path): continue
    r_curve = pd.read_csv(csv_path)
    mmt = float(r_curve['mmt'].iloc[0])

    pred = crosspred(basis=region_data[name]['cb'],
                     coef=blup_results[py_idx]['blup'], vcov=blup_results[py_idx]['vcov'],
                     model_link='log',
                     at=r_curve['temperature'].values, cen=mmt)

    diff = np.abs(r_curve['rr_fit'].values - pred.allRRfit)
    rr_rows_stage3.append({'region': name, 'max': diff.max(), 'mean': diff.mean(),
                           'corr': float(np.corrcoef(r_curve['rr_fit'].values, pred.allRRfit)[0,1])})

print("\nRR curves from full PyDLNM pipeline:")
for r in rr_rows_stage3:
    ok = "✓" if r['max'] < 0.05 else ("~" if r['max'] < 0.15 else "✗")
    print(f"  {ok} {r['region']:25s}  max|ΔRR|={r['max']:.4f}  corr={r['corr']:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("SUMMARY")
print("═"*70)
print(f"  Stage 1 (coef):         mean max|Δcoef| = {np.mean(coef_diffs):.2e}  → {'EXACT' if np.mean(coef_diffs)<1e-8 else 'DIFFERS'}")
print(f"  Stage 2 (pred w/R BLUPs): mean max|ΔRR| = {np.mean([r['max'] for r in rr_rows_stage2]):.6f}  → {'PASS' if np.mean([r['max'] for r in rr_rows_stage2])<0.005 else 'REVIEW'}")
print(f"  Stage 3 (full pipeline): mean max|ΔRR|  = {np.mean([r['max'] for r in rr_rows_stage3]):.4f}  (driven by MVMeta differences)")
