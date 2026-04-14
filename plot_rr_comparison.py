#!/usr/bin/env python3
"""
Plot RR curve comparisons: R (reference) vs Python (two variants)
  - Stage 2: Python crossbasis + R BLUPs → crosspred  (tests prediction only)
  - Stage 3: Full Python pipeline (PyDLNM GLM + MVMeta + BLUP + crosspred)

Produces a 2×5 grid (one panel per region) with:
  - R curve (shaded CI)
  - Python Stage 2 curve (should overlay R exactly)
  - Python Stage 3 curve (shows MVMeta residual differences)
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# ── Parameters ────────────────────────────────────────────────────────────────
VARFUN = "bs"; VARDEGREE = 2; VARPER = [10, 75, 90]
LAG = 21; LAGNK = 3; DFSEAS = 8
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
r_blup <- readRDS("{RESULTS_DIR}/blup_results.rds")
r_coef <- readRDS("{RESULTS_DIR}/coefficients.rds")
''')
with localconverter(ro.default_converter + numpy2ri.converter):
    r_region_codes = list(ro.r('rownames(r_coef)'))
r_code_idx = {c: i for i, c in enumerate(r_region_codes)}

# ── Stage 1: fit per-region GLMs ──────────────────────────────────────────────
print("Running Stage 1 (GLM fits)...")
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
    print(f"  fitted {name}")

# ── Stage 3: PyDLNM MVMeta ────────────────────────────────────────────────────
print("Running Stage 3 (MVMeta)...")
py_coef_mat = np.vstack(py_coef_list)
avg_t   = np.array([region_data[n]['df']['tmean'].mean() for n in SORTED_NAMES])
range_t = np.array([region_data[n]['df']['tmean'].max() - region_data[n]['df']['tmean'].min()
                    for n in SORTED_NAMES])
S_meta  = np.column_stack([np.ones(len(SORTED_NAMES)), avg_t, range_t])
vcov_3d = np.stack(py_vcov_list, axis=0)

mv = MVMeta()
mv.fit(y=py_coef_mat, S=vcov_3d, X=S_meta)
blup_results = blup(mv)

# ── Build predictions for all regions ─────────────────────────────────────────
print("Generating predictions...")

# colour / style constants
C_R    = '#2166ac'   # R reference: blue
C_S2   = '#d73027'   # Stage 2 (Python pred w/ R BLUPs): red
C_S3   = '#1a9641'   # Stage 3 (full Python pipeline): green
ALPHA_CI = 0.15

fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharey=False)
axes_flat = axes.flatten()

for py_idx, code in enumerate(SORTED_CODES):
    name  = CODE_TO_NAME[code]
    r_idx = r_code_idx.get(code)
    ax    = axes_flat[py_idx]

    # Load R reference curve
    csv_path = f"{RESULTS_DIR}/rr_curve_{code}.csv"
    r_curve  = pd.read_csv(csv_path)
    temps    = r_curve['temperature'].values
    mmt      = float(r_curve['mmt'].iloc[0])

    # ── R reference ──────────────────────────────────────────────────────────
    ax.fill_between(temps, r_curve['rr_low'], r_curve['rr_high'],
                    color=C_R, alpha=ALPHA_CI)
    ax.plot(temps, r_curve['rr_fit'], color=C_R, lw=1.8, label='R reference')

    # ── Stage 2: R BLUPs → Python crosspred ──────────────────────────────────
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_blup_coef = np.array(ro.r(f'r_blup[[{r_idx+1}]]$blup'))
        r_blup_vcov = np.array(ro.r(f'r_blup[[{r_idx+1}]]$vcov'))

    pred_s2 = crosspred(basis=region_data[name]['cb'],
                        coef=r_blup_coef, vcov=r_blup_vcov,
                        model_link='log', at=temps, cen=mmt)
    ax.fill_between(temps, pred_s2.allRRlow, pred_s2.allRRhigh,
                    color=C_S2, alpha=ALPHA_CI)
    ax.plot(temps, pred_s2.allRRfit, color=C_S2, lw=1.4,
            linestyle='--', label='Py pred (R BLUPs)')

    # ── Stage 3: Full Python pipeline ────────────────────────────────────────
    pred_s3 = crosspred(basis=region_data[name]['cb'],
                        coef=blup_results[py_idx]['blup'],
                        vcov=blup_results[py_idx]['vcov'],
                        model_link='log', at=temps, cen=mmt)
    ax.fill_between(temps, pred_s3.allRRlow, pred_s3.allRRhigh,
                    color=C_S3, alpha=ALPHA_CI)
    ax.plot(temps, pred_s3.allRRfit, color=C_S3, lw=1.4,
            linestyle=':', label='Py full pipeline')

    # ── Decorations ──────────────────────────────────────────────────────────
    ax.axhline(1.0, color='gray', lw=0.7, ls='-')
    ax.axvline(mmt, color='gray', lw=0.7, ls='--', alpha=0.6)
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.set_xlabel('Temperature (°C)', fontsize=7)
    ax.set_ylabel('RR', fontsize=7)
    ax.tick_params(labelsize=7)

    # Annotate max|ΔRR| for Stage 2 and Stage 3
    diff_s2 = np.abs(r_curve['rr_fit'].values - pred_s2.allRRfit).max()
    diff_s3 = np.abs(r_curve['rr_fit'].values - pred_s3.allRRfit).max()
    ax.text(0.03, 0.97,
            f'S2 Δ={diff_s2:.4f}\nS3 Δ={diff_s3:.3f}',
            transform=ax.transAxes, fontsize=6.5, va='top',
            color='#444444', family='monospace')

# ── Legend & title ────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=C_R,  label='R reference (shaded CI)'),
    mpatches.Patch(color=C_S2, label='Python: R BLUPs → crosspred (Stage 2)'),
    mpatches.Patch(color=C_S3, label='Python: full pipeline (Stage 3)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Temperature–Mortality RR Curves: R vs Python\n'
             '(England & Wales 10 regions, 2015 Gasparrini Lancet)',
             fontsize=11, fontweight='bold', y=1.01)

plt.tight_layout()
out_path = "rr_comparison_R_vs_Python.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
