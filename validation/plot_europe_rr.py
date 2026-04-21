"""
Plot RR curves using pydlnm — mirrors plot_RR.pdf from code.R.

Uses saved R BLUPs and MMTs as inputs, runs pydlnm's OneBasis + crosspred
for each region, then renders a multi-page PDF (18 panels per page,
3 rows × 6 cols) matching R's layout.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
sys.path.insert(0, '/Users/adessler/Desktop/DLNM')

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

from basis import OneBasis
from prediction import crosspred

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = '/Users/adessler/Desktop/europe_summer_2022_heat-main'
REF_DIR  = '/Users/adessler/Desktop/DLNM/validation/europe_ref'
OUT_PDF  = '/Users/adessler/Desktop/DLNM/validation/plot_europe_rr_pydlnm.pdf'

# ── parameters ────────────────────────────────────────────────────────────────
DATE1_CALI = pd.Timestamp('2015-01-01')
DATE2_CALI = pd.Timestamp('2019-12-26')
VAR_PRC    = np.array([10, 50, 90]) / 100
MAX_LAG    = 3
MIN_PMMT   = 5  / 100
MAX_PMMT   = 100 / 100
PRED_PRC   = np.unique(np.concatenate([
    np.arange(0.0,   1.1,  0.1),
    np.arange(1.5,   5.1,  0.5),
    np.arange(6.0,  94.1,  1.0),
    np.arange(95.0, 98.6,  0.5),
    np.arange(99.0, 100.1, 0.1),
])) / 100
PRED_PRC = np.clip(PRED_PRC, 0, 1)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data and reference BLUPs...")
datatable = pd.read_csv(os.path.join(DATA_DIR, 'data.csv'))
datatable['date'] = pd.to_datetime(
    datatable['year'].astype(str) + '-W' + datatable['woy'].astype(str).str.zfill(2) + '-4',
    format='%G-W%V-%u')
metatable = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
vREG = metatable['location'].tolist()
nREG = len(vREG)

cali_mask = ((DATE1_CALI <= datatable['date']) &
             (datatable['date'] <= DATE2_CALI + pd.Timedelta(weeks=MAX_LAG)))
datalist_cali = {r: datatable.loc[cali_mask & (datatable['location'] == r)].reset_index(drop=True)
                 for r in vREG}

# Load saved BLUPs and MMTs
blup_coef = pd.read_csv(os.path.join(REF_DIR, 'blup_coefs.csv'), index_col=0).loc[vREG].values
mmt_reg   = pd.read_csv(os.path.join(REF_DIR, 'mmt_reg.csv')).set_index('location').loc[vREG, 'mmt'].values

# Load BLUP vcov matrices from RDS via rpy2
with localconverter(robjects.default_converter + numpy2ri.converter):
    robjects.r(f'blup_vcov_list <- readRDS("{os.path.join(REF_DIR, "blup_vcov.rds")}")')
blup_vcov = []
for i in range(nREG):
    with localconverter(robjects.default_converter + numpy2ri.converter):
        mat = np.array(robjects.r(f'blup_vcov_list[[{i+1}]]'))
    blup_vcov.append(mat)

# ── compute RR curves with pydlnm ─────────────────────────────────────────────
print("Computing RR curves with pydlnm...")
rr_results = []

for i, reg in enumerate(vREG):
    temp = datalist_cali[reg]['temp'].values
    knots_var = np.nanpercentile(temp, VAR_PRC * 100)
    bk_var    = np.array([np.nanmin(temp), np.nanmax(temp)])
    pred_temps = np.nanpercentile(temp, PRED_PRC * 100)
    mmt        = mmt_reg[i]

    ob = OneBasis(pred_temps, fun='ns', knots=knots_var, Boundary_knots=bk_var)
    pred = crosspred(ob, coef=blup_coef[i], vcov=blup_vcov[i],
                     model_link='log', at=pred_temps, cen=mmt)
    rr_results.append({
        'reg': reg, 'temps': pred_temps, 'mmt': mmt,
        'fit': pred.allRRfit, 'low': pred.allRRlow, 'high': pred.allRRhigh,
    })

    if (i + 1) % 20 == 0 or i == nREG - 1:
        print(f"  {i+1}/{nREG} done")

# ── plot ──────────────────────────────────────────────────────────────────────
print(f"Writing PDF to {OUT_PDF} ...")
NCOLS = 6
NROWS = 3
PER_PAGE = NCOLS * NROWS

n_pages = int(np.ceil(nREG / PER_PAGE))

with PdfPages(OUT_PDF) as pdf:
    for page in range(n_pages):
        start = page * PER_PAGE
        end   = min(start + PER_PAGE, nREG)
        n_this_page = end - start

        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(18, 9))
        axes_flat = axes.flatten()

        for k in range(PER_PAGE):
            ax = axes_flat[k]
            if k < n_this_page:
                r = rr_results[start + k]
                temps, fit, low, high, mmt_val, reg = (
                    r['temps'], r['fit'], r['low'], r['high'], r['mmt'], r['reg'])

                ax.fill_between(temps, low, high,
                                color='black', alpha=0.2, linewidth=0)
                ax.plot(temps, fit, color='black', linewidth=2)
                ax.axvline(mmt_val, color='black', linewidth=0.8, linestyle='-')
                ax.axhline(1.0,     color='black', linewidth=0.8, linestyle='-')
                ax.set_title(reg, fontsize=8)
                ax.set_xlabel('Temperature (°C)', fontsize=7)
                ax.set_ylabel('Relative Risk', fontsize=7)
                ax.tick_params(labelsize=6)
            else:
                ax.set_visible(False)

        fig.suptitle('Cumulative Exposure–Response Curves (pydlnm)',
                     fontsize=11, y=1.01)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"Done. {n_pages} page(s), {nREG} regions total.")
