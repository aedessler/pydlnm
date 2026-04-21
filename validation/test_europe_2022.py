"""
Validate pydlnm against R's dlnm for the Europe Summer 2022 heat analysis.

Mirrors code.R from europe_summer_2022_heat-main/, comparing at four stages:
  Stage 1: Per-region GLM → reduced coefficients
  Stage 2: MVMeta → BLUPs
  Stage 3: OneBasis predictions → RR curves
  Stage 4: Attribution → attributable numbers for Summer 2022
"""

import os
import sys
import numpy as np
import pandas as pd

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
sys.path.insert(0, '/Users/adessler/Desktop/DLNM')

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from basis import CrossBasis, OneBasis
from crossreduce import crossreduce
from prediction import crosspred
from meta_analysis import MVMeta, blup

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = '/Users/adessler/Desktop/europe_summer_2022_heat-main'
REF_DIR  = '/Users/adessler/Desktop/DLNM/validation/europe_ref'

# ── parameters (identical to code.R) ────────────────────────────────────────
DATE1_CALI = pd.Timestamp('2015-01-01')
DATE2_CALI = pd.Timestamp('2019-12-26')
DATE1_PRED = pd.Timestamp('2015-01-01')
DATE2_PRED = pd.Timestamp('2022-11-03')
DATE1_SU22 = pd.Timestamp('2022-06-02')
DATE2_SU22 = pd.Timestamp('2022-09-01')

VAR_FUN  = 'ns'
VAR_PRC  = np.array([10, 50, 90]) / 100
MIN_LAG  = 0
MAX_LAG  = 3
DF_SEAS  = 8
MIN_PMMT = 5  / 100
MAX_PMMT = 100 / 100

PRED_PRC = np.unique(np.concatenate([
    np.arange(0.0,   1.1,  0.1),
    np.arange(1.5,   5.1,  0.5),
    np.arange(6.0,  94.1,  1.0),
    np.arange(95.0, 98.6,  0.5),
    np.arange(99.0, 100.1, 0.1),
])) / 100
PRED_PRC = np.clip(PRED_PRC, 0, 1)

# ── load R packages ──────────────────────────────────────────────────────────
splines_r  = importr('splines')
dlnm_r     = importr('dlnm')
stats_r    = importr('stats')

# ── helper: build ISO-week date string → Thursday ────────────────────────────
def isoweek_to_date(year, woy):
    return pd.to_datetime(
        year.astype(str) + '-W' + woy.astype(str).str.zfill(2) + '-4',
        format='%G-W%V-%u'
    )

# ── load data ────────────────────────────────────────────────────────────────
print("Loading data...")
datatable = pd.read_csv(os.path.join(DATA_DIR, 'data.csv'))
datatable['date'] = isoweek_to_date(datatable['year'], datatable['woy'])
metatable = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

vREG = metatable['location'].tolist()
nREG = len(vREG)

cali_mask = (DATE1_CALI <= datatable['date']) & (datatable['date'] <= DATE2_CALI + pd.Timedelta(weeks=MAX_LAG))
pred_mask = (DATE1_PRED <= datatable['date']) & (datatable['date'] <= DATE2_PRED + pd.Timedelta(weeks=MAX_LAG))

datalist_cali = {r: datatable.loc[cali_mask & (datatable['location'] == r)].copy().reset_index(drop=True)
                 for r in vREG}
datalist_pred = {r: datatable.loc[pred_mask & (datatable['location'] == r)].copy().reset_index(drop=True)
                 for r in vREG}

for r in vREG:
    datalist_cali[r]['wop'] = np.arange(1, len(datalist_cali[r]) + 1)

print(f"  {nREG} regions, calibration rows per region ≈ {len(datalist_cali[vREG[0]])}")

# ── Stage 1: Per-region GLMs ─────────────────────────────────────────────────
print("\n=== Stage 1: Per-region GLMs ===")

n_coef    = len(VAR_PRC) + 1   # ns with 3 knots → 4 basis functions
coef_py   = np.full((nREG, n_coef), np.nan)
vcov_py   = [None] * nREG
mmt_stage1_py = np.full(nREG, np.nan)

ref_coef = pd.read_csv(os.path.join(REF_DIR, 'coef_model.csv'), index_col=0)
ref_mmt1 = pd.read_csv(os.path.join(REF_DIR, 'mmt_stage1.csv')).set_index('location')

def fit_glm_r(df_cali, cb_matrix):
    """Fit quasi-Poisson GLM using R, return (coef_cb, vcov_cb).

    NaN values in cb_matrix (first MAX_LAG rows) are passed as-is; R's
    na.action=na.exclude drops those rows from the fit, matching code.R.
    """
    with localconverter(robjects.default_converter + numpy2ri.converter + pandas2ri.converter):
        robjects.globalenv['py_mort'] = df_cali['mort'].values.astype(float)
        robjects.globalenv['py_wop']  = df_cali['wop'].values.astype(float)
        robjects.globalenv['py_cb']   = cb_matrix  # NaN stays NaN → R NA
        n_weeks = len(df_cali)
        df_seas_val = int(round(DF_SEAS * n_weeks * 7 / 365.25))
        robjects.r(f'''
            py_glm <- glm(py_mort ~ ns(py_wop, df={df_seas_val}) + py_cb,
                          family    = quasipoisson,
                          na.action = na.exclude)
            cb_idx    <- grep("^py_cb", names(coef(py_glm)))
            coef_cb_r <- coef(py_glm)[cb_idx]
            vcov_full <- vcov(py_glm)
            vcov_cb_r <- vcov_full[cb_idx, cb_idx]
        ''')
        coef_cb = np.array(robjects.r('coef_cb_r'))
        vcov_cb = np.array(robjects.r('vcov_cb_r'))
    return coef_cb, vcov_cb

def find_mmt_from_pred(pred_temps, all_rr_fit, min_pmmt, max_pmmt, pred_prc):
    """Find MMT: lowest RR in [MIN_PMMT, MAX_PMMT] percentile range."""
    idx_lo = np.searchsorted(pred_prc, min_pmmt)
    idx_hi = np.searchsorted(pred_prc, max_pmmt)
    sub_rr  = all_rr_fit[idx_lo:idx_hi + 1]
    return pred_temps[idx_lo + np.argmin(sub_rr)]

for i, reg in enumerate(vREG):
    df_c = datalist_cali[reg]
    temp = df_c['temp'].values

    knots_var = np.nanpercentile(temp, VAR_PRC * 100)
    bk_var    = np.array([np.nanmin(temp), np.nanmax(temp)])

    cb = CrossBasis(
        temp,
        lag=np.array([MIN_LAG, MAX_LAG]),
        argvar={'fun': 'ns', 'knots': knots_var, 'Boundary_knots': bk_var},
        arglag={'fun': 'integer'},
    )

    glm_coef, glm_vcov = fit_glm_r(df_c, cb.basis)

    # Find MMT with preliminary prediction (no centering)
    pred_temps_cali = np.nanpercentile(temp, PRED_PRC * 100)
    pred_prc_idx_lo = np.searchsorted(PRED_PRC, MIN_PMMT)
    pred_prc_idx_hi = np.searchsorted(PRED_PRC, MAX_PMMT)

    pred_tmp = crosspred(cb, coef=glm_coef, vcov=glm_vcov,
                         model_link='log', at=pred_temps_cali, cen=pred_temps_cali[0])
    sub_rr = pred_tmp.allRRfit[pred_prc_idx_lo:pred_prc_idx_hi + 1]
    mmt = pred_temps_cali[pred_prc_idx_lo + np.argmin(sub_rr)]
    mmt_stage1_py[i] = mmt

    # Reduce to overall effect
    red = crossreduce(cb, coef=glm_coef, vcov=glm_vcov, cen=mmt)
    coef_py[i] = red.coef
    vcov_py[i] = red.vcov

    if (i + 1) % 20 == 0 or i == nREG - 1:
        print(f"  Region {i+1}/{nREG}: {reg}  MMT={mmt:.2f}°C")

# Compare Stage 1 coefficients
ref_coef_arr = ref_coef.values  # shape (nREG, n_coef)
delta_coef = np.abs(coef_py - ref_coef_arr)
print(f"\n  Stage 1 coefficient comparison:")
print(f"    max|Δcoef| = {delta_coef.max():.2e}")
print(f"    mean|Δcoef| = {delta_coef.mean():.2e}")

ref_mmt1_arr = ref_mmt1.loc[vREG, 'mmt'].values
delta_mmt1 = np.abs(mmt_stage1_py - ref_mmt1_arr)
print(f"    max|ΔMMT| stage1 = {delta_mmt1.max():.4f}°C")
stage1_pass = delta_coef.max() < 1e-6
print(f"  Stage 1: {'PASS' if stage1_pass else 'FAIL (check logs)'}")

# ── Stage 2: Meta-analysis and BLUPs ────────────────────────────────────────
print("\n=== Stage 2: MVMeta / BLUPs ===")

temp_avg = np.array([np.nanmean(datalist_cali[r]['temp']) for r in vREG])
temp_iqr = np.array([np.nanpercentile(datalist_cali[r]['temp'], 75) -
                     np.nanpercentile(datalist_cali[r]['temp'], 25) for r in vREG])

X_meta = np.column_stack([np.ones(nREG), temp_avg, temp_iqr])
vcov_stack = np.stack(vcov_py)  # (nREG, n_coef, n_coef)

mv = MVMeta()
mv.fit(coef_py, vcov_stack, X=X_meta)
blup_results = blup(mv)

blup_coef_py = np.array([blup_results[i]['blup'] for i in range(nREG)])
blup_vcov_py = [blup_results[i]['vcov'] for i in range(nREG)]

ref_blup_coef = pd.read_csv(os.path.join(REF_DIR, 'blup_coefs.csv'), index_col=0)
ref_blup_arr  = ref_blup_coef.loc[vREG].values

delta_blup = np.abs(blup_coef_py - ref_blup_arr)
print(f"  max|ΔBLUP coef| = {delta_blup.max():.2e}")
print(f"  mean|ΔBLUP coef| = {delta_blup.mean():.2e}")
stage2_pass = delta_blup.max() < 1e-4
print(f"  Stage 2: {'PASS' if stage2_pass else 'FAIL (check logs)'}")

# ── Stage 3: RR curves ───────────────────────────────────────────────────────
print("\n=== Stage 3: RR Curves ===")

mmt_reg_py = np.full(nREG, np.nan)
rr_diffs   = []

for i, reg in enumerate(vREG):
    df_c = datalist_cali[reg]
    temp = df_c['temp'].values
    knots_var = np.nanpercentile(temp, VAR_PRC * 100)
    bk_var    = np.array([np.nanmin(temp), np.nanmax(temp)])
    pred_temps = np.nanpercentile(temp, PRED_PRC * 100)

    # OneBasis (no lag dimension)
    ob = OneBasis(pred_temps, fun='ns', knots=knots_var, Boundary_knots=bk_var)

    # Preliminary prediction (no cen) to find MMT after meta-analysis
    pred_prc_idx_lo = np.searchsorted(PRED_PRC, MIN_PMMT)
    pred_prc_idx_hi = np.searchsorted(PRED_PRC, MAX_PMMT)

    pred_tmp = crosspred(ob, coef=blup_coef_py[i], vcov=blup_vcov_py[i],
                         model_link='log', at=pred_temps, cen=pred_temps[0])
    sub_rr = pred_tmp.allRRfit[pred_prc_idx_lo:pred_prc_idx_hi + 1]
    mmt = pred_temps[pred_prc_idx_lo + np.argmin(sub_rr)]
    mmt_reg_py[i] = mmt

    # Final prediction centered at MMT
    pred_final = crosspred(ob, coef=blup_coef_py[i], vcov=blup_vcov_py[i],
                           model_link='log', at=pred_temps, cen=mmt)

    ref_rr = pd.read_csv(os.path.join(REF_DIR, 'rr_curves', f'rr_{reg}.csv'))
    diff = np.abs(pred_final.allRRfit - ref_rr['allRRfit'].values)
    rr_diffs.append(diff.max())

ref_mmt_reg = pd.read_csv(os.path.join(REF_DIR, 'mmt_reg.csv')).set_index('location')
delta_mmt = np.abs(mmt_reg_py - ref_mmt_reg.loc[vREG, 'mmt'].values)

print(f"  max|ΔMMT| = {delta_mmt.max():.4f}°C  (mean {delta_mmt.mean():.4f}°C)")
print(f"  max|ΔRR| across all regions = {max(rr_diffs):.2e}")
print(f"  median max|ΔRR| per region  = {np.median(rr_diffs):.2e}")
stage3_pass = max(rr_diffs) < 1e-3 and delta_mmt.max() < 0.1
print(f"  Stage 3: {'PASS' if stage3_pass else 'FAIL (check logs)'}")

# ── Stage 4: Attribution ─────────────────────────────────────────────────────
print("\n=== Stage 4: Attribution (Summer 2022) ===")

ref_an = pd.read_csv(os.path.join(REF_DIR, 'an_summer2022.csv'))

an_rows = []

for i, reg in enumerate(vREG):
    df_c = datalist_cali[reg]
    df_p = datalist_pred[reg]
    temp_cali = df_c['temp'].values
    knots_var = np.nanpercentile(temp_cali, VAR_PRC * 100)
    bk_var    = np.array([np.nanmin(temp_cali), np.nanmax(temp_cali)])

    mmt = mmt_reg_py[i]

    # OneBasis for prediction period temperatures (centered at MMT)
    basis_var = OneBasis(df_p['temp'].values, fun='ns', knots=knots_var, Boundary_knots=bk_var).basis
    basis_mmt = OneBasis(np.array([mmt]),      fun='ns', knots=knots_var, Boundary_knots=bk_var).basis
    basis_cen = basis_var - basis_mmt  # (n_pred_rows, n_basis)

    # Lagged mortality matrix (lags 0..MAX_LAG)
    mort_pred = df_p['mort'].values.astype(float)
    n_rows = len(mort_pred)
    lag_mort_mat = np.full((n_rows, MAX_LAG - MIN_LAG + 1), np.nan)
    for l_idx, lag in enumerate(range(MIN_LAG, MAX_LAG + 1)):
        if lag == 0:
            lag_mort_mat[:, l_idx] = mort_pred
        else:
            lag_mort_mat[lag:, l_idx] = mort_pred[:-lag]
    lag_mort_vec = np.nanmean(lag_mort_mat, axis=1)

    # Point estimate of attributable numbers
    att_ts_ref = (1 - np.exp(-basis_cen @ blup_coef_py[i])) * lag_mort_vec

    # Monte Carlo uncertainty
    rng = np.random.default_rng(5634654)
    coef_sim = rng.multivariate_normal(blup_coef_py[i], blup_vcov_py[i], size=1000).T  # (n_coef, 1000)
    att_ts_sim = (1 - np.exp(-basis_cen @ coef_sim)) * lag_mort_vec[:, None]

    # Summer 2022 indices (exclude final MAX_LAG rows)
    n_excl = MAX_LAG
    pred_dates = df_p['date'].values
    n_valid = len(pred_dates) - n_excl
    su22_mask = (pred_dates[:n_valid] >= np.datetime64(DATE1_SU22)) & \
                (pred_dates[:n_valid] <= np.datetime64(DATE2_SU22))
    vTIM = np.where(su22_mask)[0]

    temp_pred = df_p['temp'].values

    def compute_an(vTIM, range_name):
        if range_name == 'Total':
            vRNG = np.arange(len(vTIM))
        elif range_name == 'Total Cold':
            vRNG = np.where(temp_pred[vTIM] < mmt)[0]
        elif range_name == 'Total Heat':
            vRNG = np.where(temp_pred[vTIM] > mmt)[0]
        else:
            raise ValueError(range_name)

        if len(vRNG) == 0 or np.nansum(lag_mort_vec[vTIM]) == 0:
            return 0.0, 0.0, 0.0

        rows_all  = lag_mort_mat[vTIM, :]
        correction = np.nansum(np.nanmean(rows_all, axis=1)) / np.nansum(lag_mort_vec[vTIM])

        sel = vTIM[vRNG]
        an_val  = np.nansum(att_ts_ref[sel]) * correction
        an_sims = np.nansum(att_ts_sim[sel, :], axis=0) * correction
        an_low, an_high = np.nanpercentile(an_sims, [2.5, 97.5])
        return an_val, an_low, an_high

    for rng_name in ['Total', 'Total Cold', 'Total Heat']:
        val, lo, hi = compute_an(vTIM, rng_name)
        an_rows.append({'location': reg, 'period': 'Summer 2022',
                        'range': rng_name, 'att_val': val, 'att_low': lo, 'att_upp': hi})

an_py = pd.DataFrame(an_rows)

# Compare Total Heat for Summer 2022
ref_heat = ref_an[(ref_an['period'] == 'Summer 2022') & (ref_an['range'] == 'Total Heat')].set_index('location')
py_heat  = an_py[(an_py['period'] == 'Summer 2022') & (an_py['range'] == 'Total Heat')].set_index('location')

heat_rel_err = np.abs((py_heat.loc[vREG, 'att_val'].values - ref_heat.loc[vREG, 'att_val'].values) /
                      (ref_heat.loc[vREG, 'att_val'].values + 1e-9))
total_heat_py  = py_heat['att_val'].sum()
total_heat_ref = ref_heat['att_val'].sum()

print(f"  Total Heat AN (R)  = {total_heat_ref:,.0f}")
print(f"  Total Heat AN (Py) = {total_heat_py:,.0f}")
print(f"  Relative error (overall) = {abs(total_heat_py-total_heat_ref)/abs(total_heat_ref)*100:.2f}%")
print(f"  max per-region rel error = {heat_rel_err.max()*100:.2f}%")
stage4_pass = abs(total_heat_py - total_heat_ref) / abs(total_heat_ref) < 0.05
print(f"  Stage 4: {'PASS' if stage4_pass else 'FAIL (check logs)'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"  Stage 1 (GLM coefficients): {'PASS' if stage1_pass else 'FAIL'}")
print(f"  Stage 2 (BLUPs):            {'PASS' if stage2_pass else 'FAIL'}")
print(f"  Stage 3 (RR curves):        {'PASS' if stage3_pass else 'FAIL'}")
print(f"  Stage 4 (Attribution):      {'PASS' if stage4_pass else 'FAIL'}")

all_pass = stage1_pass and stage2_pass and stage3_pass and stage4_pass
print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES — see details above'}")
sys.exit(0 if all_pass else 1)
