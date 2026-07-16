# =============================================================================
# PUBLICATION 1 — STEP 3b: SUBGROUP ANALYSES + INTERACTION + MCC
# Run AFTER step3.
# Output: subgroup_mortality_by_diagnosis.csv, regression_interaction_model.csv,
#         corrected_pvalues.csv, disposition_by_race_pct.csv,
#         disposition_by_insurance_pct.csv, regression_home_discharge.csv
# =============================================================================

import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "numpy", "scipy", "statsmodels"], check=False)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import os
import warnings
warnings.filterwarnings('ignore')

BASE = '/content/drive/MyDrive/mimic 4/'
OUT  = os.path.join(BASE, 'outputs/')

cohort   = pd.read_csv(OUT + 'neurosurg_cohort_enriched.csv', low_memory=False)
has_elix = 'elixhauser_score' in cohort.columns
has_gcs  = 'gcs_admission'    in cohort.columns
has_drg  = 'drg_severity_class' in cohort.columns

cohort['age_centered']  = cohort['age'] - cohort['age'].mean()
cohort['gender_female'] = (cohort['gender'] == 'F').astype(int)
cohort['non_english']   = cohort['non_english'].fillna(0).astype(int)
cohort['primary_cat']   = cohort['neurosurg_category'].apply(
    lambda x: x.split(' | ')[0] if isinstance(x, str) else 'Other Neurosurgical')
cohort['year_group'] = cohort['anchor_year_group'].astype(str) \
    if 'anchor_year_group' in cohort.columns else 'unknown'

base_covs = "gender_female + age_centered + non_english + C(year_group)"
if has_elix: base_covs += " + elixhauser_score"
if has_gcs:  base_covs += " + gcs_admission"
if has_drg:  base_covs += " + C(drg_severity_class)"

race_ref = "C(race_group, Treatment('White'))"
ins_ref  = "C(insurance_group, Treatment('Private'))"

reg = cohort.dropna(subset=['race_group','insurance_group','age_centered',
                             'in_hospital_mortality','primary_cat']).copy()

def run_logit(formula, data, label=""):
    try:
        m = smf.logit(formula, data=data).fit(maxiter=300, disp=False)
        df = pd.DataFrame({
            'Variable': m.params.index,
            'OR':       np.exp(m.params.values),
            'CI_low':   np.exp(m.conf_int()[0].values),
            'CI_high':  np.exp(m.conf_int()[1].values),
            'p_value':  m.pvalues.values,
            'n': len(data), 'subgroup': label
        })
        df['p_fmt'] = df['p_value'].apply(lambda p: "<0.001" if p<0.001 else f"{p:.3f}")
        df['OR_CI'] = df.apply(lambda r: f"{r['OR']:.2f} ({r['CI_low']:.2f}–{r['CI_high']:.2f})", axis=1)
        return df, m
    except Exception as e:
        print(f"  Error ({label}): {e}"); return pd.DataFrame(), None

# -----------------------------------------------------------------------------
# 1. STRATIFIED MODELS BY DIAGNOSIS CATEGORY
# -----------------------------------------------------------------------------
print("\n=== STRATIFIED MODELS BY DIAGNOSIS CATEGORY ===")
formula_strat = f"in_hospital_mortality ~ {race_ref} + {ins_ref} + {base_covs}"
strat_results = []

for cat in ['TBI','Hemorrhagic Stroke','Craniotomy','Spine Surgery','Brain Tumor']:
    sub      = reg[reg['primary_cat'] == cat].copy()
    n_events = sub['in_hospital_mortality'].sum() if len(sub) > 0 else 0
    if len(sub) < 100 or n_events < 10:
        print(f"  {cat}: n={len(sub)}, events={n_events:.0f} — skipping"); continue
    print(f"  {cat}: n={len(sub):,}, events={n_events:.0f}")
    res, _ = run_logit(formula_strat, sub, label=cat)
    if len(res) > 0: strat_results.append(res)

if strat_results:
    strat_df  = pd.concat(strat_results, ignore_index=True)
    race_rows = strat_df[strat_df['Variable'].str.contains('race_group|insurance_group')]
    race_rows.to_csv(OUT + 'subgroup_mortality_by_diagnosis.csv', index=False)
    print(f"Stratified results saved: {len(race_rows)} rows")

# -----------------------------------------------------------------------------
# 2. RACE × INSURANCE INTERACTION
# -----------------------------------------------------------------------------
print("\n=== RACE × INSURANCE INTERACTION ===")
f_int  = (f"in_hospital_mortality ~ {race_ref} + {ins_ref} + "
          f"C(race_group, Treatment('White')):C(insurance_group, Treatment('Private')) + "
          f"{base_covs} + C(primary_cat)")
f_main = f"in_hospital_mortality ~ {race_ref} + {ins_ref} + {base_covs} + C(primary_cat)"

reg_int    = reg.dropna(subset=['race_group','insurance_group','primary_cat'])
int_res, int_model  = run_logit(f_int,  reg_int, "Interaction")
_,       main_model = run_logit(f_main, reg_int, "Main")

if len(int_res) > 0:
    int_res.to_csv(OUT + 'regression_interaction_model.csv', index=False)
    print("Interaction terms:")
    print(int_res[int_res['Variable'].str.contains(':')][['Variable','OR_CI','p_fmt']].to_string())

if int_model and main_model:
    lrt_stat = -2*(main_model.llf - int_model.llf)
    lrt_df   = int_model.df_model - main_model.df_model
    lrt_p    = 1 - stats.chi2.cdf(lrt_stat, df=max(1,lrt_df))
    print(f"\nLRT: χ²={lrt_stat:.2f}, df={lrt_df:.0f}, p={lrt_p:.4f}")
    with open(OUT + 'interaction_lrt.txt','w') as f:
        f.write(f"LRT statistic: {lrt_stat:.2f}\ndf: {lrt_df:.0f}\np-value: {lrt_p:.4f}\n")

# -----------------------------------------------------------------------------
# 3. MULTIPLE COMPARISON CORRECTION
# -----------------------------------------------------------------------------
print("\n=== MULTIPLE COMPARISON CORRECTION ===")
all_pvals = []
for fname, outcome in [
    ('regression_mortality_OR.csv',  'In-Hospital Mortality'),
    ('regression_readmit_OR.csv',    '30-Day Readmission'),
    ('regression_icu_los.csv',       'ICU LOS'),
    ('regression_time_to_icu.csv',   'Time to ICU'),
]:
    try:
        df  = pd.read_csv(OUT + fname)
        sub = df[df['Variable'].str.contains('race_group|insurance_group')].copy()
        sub['outcome'] = outcome; sub['source'] = 'Main Model'
        all_pvals.append(sub[['Variable','p_value','outcome','source']])
    except FileNotFoundError:
        pass

if strat_results:
    s = race_rows.copy()
    s['outcome'] = 'Mortality'; s['source'] = 'Subgroup: ' + s['subgroup']
    all_pvals.append(s[['Variable','p_value','outcome','source']])

if all_pvals:
    pval_df = pd.concat(all_pvals, ignore_index=True).dropna(subset=['p_value'])
    _, bonf_p, _, _ = multipletests(pval_df['p_value'].values, method='bonferroni')
    _, fdr_p,  _, _ = multipletests(pval_df['p_value'].values, method='fdr_bh')
    pval_df['p_bonferroni'] = bonf_p
    pval_df['p_fdr_bh']     = fdr_p
    pval_df['sig_bonferroni'] = pval_df['p_bonferroni'] < 0.05
    pval_df['sig_fdr']        = pval_df['p_fdr_bh']     < 0.05
    pval_df.to_csv(OUT + 'corrected_pvalues.csv', index=False)
    print(f"Tests: {len(pval_df)} | Sig Bonferroni: {pval_df['sig_bonferroni'].sum()} | Sig FDR: {pval_df['sig_fdr'].sum()}")

# -----------------------------------------------------------------------------
# 4. DISCHARGE DISPOSITION
# -----------------------------------------------------------------------------
print("\n=== DISCHARGE DISPOSITION ===")
if 'discharge_disposition' in cohort.columns:
    surv = cohort[cohort['in_hospital_mortality']==0].copy()
    pd.crosstab(surv['discharge_disposition'], surv['race_group'],    normalize='columns').mul(100).to_csv(OUT+'disposition_by_race_pct.csv')
    pd.crosstab(surv['discharge_disposition'], surv['insurance_group'],normalize='columns').mul(100).to_csv(OUT+'disposition_by_insurance_pct.csv')
    ct = pd.crosstab(surv['discharge_disposition'], surv['race_group'])
    if ct.shape[0]>1 and ct.shape[1]>1:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        print(f"Chi-square (disposition × race): χ²={chi2:.2f}, df={dof}, p={'<0.001' if p<0.001 else f'{p:.3f}'}")

    surv['home_discharge'] = surv['discharge_disposition'].apply(
        lambda x: 1 if x in ['Home','Home with Services'] else 0)
    surv['primary_cat'] = surv['neurosurg_category'].apply(
        lambda x: x.split(' | ')[0] if isinstance(x, str) else 'Other Neurosurgical')
    f_disp = (f"home_discharge ~ {race_ref} + {ins_ref} + "
              f"gender_female + age_centered + non_english + C(primary_cat)")
    if has_elix: f_disp += " + elixhauser_score"
    if has_gcs:  f_disp += " + gcs_admission"
    reg_disp = surv.dropna(subset=['race_group','insurance_group','home_discharge','primary_cat'])
    disp_res, _ = run_logit(f_disp, reg_disp, "Home Discharge")
    if len(disp_res):
        disp_res[disp_res['Variable'].str.contains('race_group|insurance_group')].to_csv(
            OUT+'regression_home_discharge.csv', index=False)
        print("Home discharge regression saved.")
else:
    print("discharge_disposition missing — run Step 1b.")

print("\n=== STEP 3b COMPLETE ===")
