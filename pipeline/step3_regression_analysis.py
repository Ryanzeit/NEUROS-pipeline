# =============================================================================
# PUBLICATION 1 — STEP 3: MAIN REGRESSION MODELS
# Run AFTER step2.
# Output: regression_mortality_OR.csv, regression_icu_los.csv,
#         regression_readmit_OR.csv, regression_time_to_icu.csv,
#         regression_insurance_sensitivity.csv
# =============================================================================

import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "numpy", "scipy", "statsmodels"], check=False)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import warnings
warnings.filterwarnings('ignore')

BASE = '/content/drive/MyDrive/mimic 4/'
OUT  = os.path.join(BASE, 'outputs/')

cohort = pd.read_csv(OUT + 'neurosurg_cohort_enriched.csv', low_memory=False)
has_elix = 'elixhauser_score' in cohort.columns
has_gcs  = 'gcs_admission'    in cohort.columns
has_tti  = 'time_to_icu_hours' in cohort.columns
has_drg  = 'drg_severity_class' in cohort.columns
print(f"Cohort: {len(cohort):,} | Elixhauser: {has_elix} | GCS: {has_gcs} | TTI: {has_tti}")

# -----------------------------------------------------------------------------
# PREP
# -----------------------------------------------------------------------------
cohort['age_centered']  = cohort['age'] - cohort['age'].mean()
cohort['gender_female'] = (cohort['gender'] == 'F').astype(int)
cohort['non_english']   = cohort['non_english'].fillna(0).astype(int)
cohort['log_icu_los']   = np.log1p(cohort['icu_los_days'])
cohort['log_tti']       = np.log1p(cohort['time_to_icu_hours']) if has_tti else np.nan
cohort['primary_cat']   = cohort['neurosurg_category'].apply(
    lambda x: x.split(' | ')[0] if isinstance(x, str) else 'Other Neurosurgical'
)
cohort['year_group'] = cohort['anchor_year_group'].astype(str) \
    if 'anchor_year_group' in cohort.columns else 'unknown'

base_covs = "gender_female + age_centered + non_english + C(primary_cat) + C(year_group)"
if has_elix: base_covs += " + elixhauser_score"
if has_gcs:  base_covs += " + gcs_admission"
if has_drg:  base_covs += " + C(drg_severity_class)"

race_ref = "C(race_group, Treatment('White'))"
ins_ref  = "C(insurance_group, Treatment('Private'))"

reg = cohort.dropna(subset=['race_group','insurance_group','age_centered',
                             'in_hospital_mortality','primary_cat']).copy()
print(f"Regression N: {len(reg):,}")

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def logit_results(formula, data, label):
    try:
        m   = smf.logit(formula, data=data).fit(maxiter=300, disp=False)
        df  = pd.DataFrame({
            'Variable':   m.params.index,
            'OR':         np.exp(m.params.values),
            'CI_95_low':  np.exp(m.conf_int()[0].values),
            'CI_95_high': np.exp(m.conf_int()[1].values),
            'p_value':    m.pvalues.values,
        })
        df['p_fmt']        = df['p_value'].apply(lambda p: "<0.001" if p<0.001 else f"{p:.3f}")
        df['OR_CI_string'] = df.apply(
            lambda r: f"{r['OR']:.2f} ({r['CI_95_low']:.2f}–{r['CI_95_high']:.2f})", axis=1)
        df['model'] = label; df['n'] = int(m.nobs); df['pseudo_r2'] = round(m.prsquared,4)
        print(f"\n{label}: N={int(m.nobs):,}, Pseudo-R²={m.prsquared:.3f}, AIC={m.aic:.1f}")
        return df, m
    except Exception as e:
        print(f"  {label} error: {e}"); return pd.DataFrame(), None

def ols_results(formula, data, label):
    try:
        m   = smf.ols(formula, data=data).fit()
        df  = pd.DataFrame({
            'Variable':   m.params.index,
            'Beta':       m.params.values,
            'CI_95_low':  m.conf_int()[0].values,
            'CI_95_high': m.conf_int()[1].values,
            'p_value':    m.pvalues.values,
            'exp_Beta':   np.exp(m.params.values),
        })
        df['p_fmt']  = df['p_value'].apply(lambda p: "<0.001" if p<0.001 else f"{p:.3f}")
        df['model']  = label; df['n'] = int(m.nobs); df['r2'] = round(m.rsquared,4)
        print(f"\n{label}: N={int(m.nobs):,}, R²={m.rsquared:.3f}, AIC={m.aic:.1f}")
        return df, m
    except Exception as e:
        print(f"  {label} error: {e}"); return pd.DataFrame(), None

def show_key(df, label):
    if len(df) == 0: return
    sub = df[df['Variable'].str.contains('race_group|insurance_group')]
    if 'OR_CI_string' in df.columns:
        print(sub[['Variable','OR_CI_string','p_fmt']].to_string())
    else:
        print(sub[['Variable','Beta','exp_Beta','CI_95_low','CI_95_high','p_fmt']].to_string())

# -----------------------------------------------------------------------------
# MODEL 1: IN-HOSPITAL MORTALITY
# -----------------------------------------------------------------------------
print("\n=== MODEL 1: In-Hospital Mortality ===")
r1, _ = logit_results(
    f"in_hospital_mortality ~ {race_ref} + {ins_ref} + {base_covs}", reg, "Mortality")
if len(r1): r1.to_csv(OUT + 'regression_mortality_OR.csv', index=False); show_key(r1,"Mortality")

# -----------------------------------------------------------------------------
# MODEL 2: ICU LOS
# -----------------------------------------------------------------------------
print("\n=== MODEL 2: ICU LOS (log-transformed) ===")
r2, _ = ols_results(
    f"log_icu_los ~ {race_ref} + {ins_ref} + {base_covs}",
    reg.dropna(subset=['log_icu_los']), "ICU LOS")
if len(r2): r2.to_csv(OUT + 'regression_icu_los.csv', index=False); show_key(r2,"ICU LOS")

# -----------------------------------------------------------------------------
# MODEL 3: 30-DAY READMISSION (survivors only)
# -----------------------------------------------------------------------------
print("\n=== MODEL 3: 30-Day Readmission ===")
reg_r = reg[(reg['in_hospital_mortality']==0) & reg['readmit_30day'].notna()].copy()
print(f"Survivors N: {len(reg_r):,}, events: {reg_r['readmit_30day'].sum():.0f}")
r3, _ = logit_results(
    f"readmit_30day ~ {race_ref} + {ins_ref} + {base_covs}", reg_r, "Readmission")
if len(r3): r3.to_csv(OUT + 'regression_readmit_OR.csv', index=False); show_key(r3,"Readmission")

# -----------------------------------------------------------------------------
# MODEL 4: TIME TO ICU (log-transformed linear regression)
# Tests whether certain groups wait longer before ICU escalation
# -----------------------------------------------------------------------------
if has_tti:
    print("\n=== MODEL 4: Time to ICU ===")
    reg_tti = reg[reg['log_tti'].notna() & (reg['direct_icu_admission']==0)].copy()
    print(f"Non-direct admissions N: {len(reg_tti):,}")
    r4, _ = ols_results(
        f"log_tti ~ {race_ref} + {ins_ref} + {base_covs}", reg_tti, "Time to ICU")
    if len(r4): r4.to_csv(OUT + 'regression_time_to_icu.csv', index=False); show_key(r4,"TTI")

# -----------------------------------------------------------------------------
# SENSITIVITY: Insurance as primary exposure
# -----------------------------------------------------------------------------
print("\n=== SENSITIVITY: Insurance Primary Exposure ===")
r5, _ = logit_results(
    f"in_hospital_mortality ~ {ins_ref} + {race_ref} + {base_covs}", reg, "Insurance Primary")
if len(r5): r5.to_csv(OUT + 'regression_insurance_sensitivity.csv', index=False)

print("\n=== STEP 3 COMPLETE ===")
