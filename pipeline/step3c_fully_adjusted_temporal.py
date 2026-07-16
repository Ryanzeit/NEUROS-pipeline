# =============================================================================
# PUBLICATION 1 — STEP 3c: FULLY ADJUSTED MODELS + TEMPORAL TRENDS
# Run AFTER step3b.
# Output: fully_adjusted_mortality.csv, fully_adjusted_los.csv,
#         fully_adjusted_readmit.csv, fully_adjusted_time_to_icu.csv,
#         temporal_trend_raw.csv, temporal_interaction_test.csv,
#         missing_data_summary.csv, table2_regression_results.csv
# =============================================================================

import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "numpy", "scipy", "statsmodels"], check=False)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import os
import warnings
warnings.filterwarnings('ignore')

BASE = '/content/drive/MyDrive/mimic 4/'
OUT  = os.path.join(BASE, 'outputs/')

cohort   = pd.read_csv(OUT + 'neurosurg_cohort_enriched.csv', low_memory=False)
has_elix = 'elixhauser_score'  in cohort.columns
has_gcs  = 'gcs_admission'     in cohort.columns
has_tti  = 'time_to_icu_hours' in cohort.columns
has_drg  = 'drg_severity_class' in cohort.columns
print(f"Cohort: {len(cohort):,} | Elix: {has_elix} | GCS: {has_gcs} | TTI: {has_tti} | DRG: {has_drg}")

cohort['age_centered']  = cohort['age'] - cohort['age'].mean()
cohort['gender_female'] = (cohort['gender'] == 'F').astype(int)
cohort['non_english']   = cohort['non_english'].fillna(0).astype(int)
cohort['log_icu_los']   = np.log1p(cohort['icu_los_days'])
cohort['log_tti']       = np.log1p(cohort['time_to_icu_hours']) if has_tti else np.nan
cohort['primary_cat']   = cohort['neurosurg_category'].apply(
    lambda x: x.split(' | ')[0] if isinstance(x, str) else 'Other Neurosurgical')
cohort['year_group'] = cohort['anchor_year_group'].astype(str) \
    if 'anchor_year_group' in cohort.columns else 'unknown'

# FULL covariate set — this is what goes in the paper
full_covs = "gender_female + age_centered + non_english + C(primary_cat) + C(year_group)"
if has_elix: full_covs += " + elixhauser_score"
if has_gcs:  full_covs += " + gcs_admission"
if has_drg:  full_covs += " + C(drg_severity_class)"

race_ref = "C(race_group, Treatment('White'))"
ins_ref  = "C(insurance_group, Treatment('Private'))"

reg = cohort.dropna(subset=['race_group','insurance_group','age_centered',
                             'in_hospital_mortality','primary_cat']).copy()
print(f"Regression N: {len(reg):,}")

def logit_table(formula, data, label):
    try:
        m  = smf.logit(formula, data=data).fit(maxiter=300, disp=False)
        df = pd.DataFrame({
            'Variable':   m.params.index,
            'OR':         np.exp(m.params.values),
            'CI_95_low':  np.exp(m.conf_int()[0].values),
            'CI_95_high': np.exp(m.conf_int()[1].values),
            'p_value':    m.pvalues.values,
        })
        df['p_fmt']    = df['p_value'].apply(lambda p: "<0.001" if p<0.001 else f"{p:.3f}")
        df['OR_CI']    = df.apply(lambda r: f"{r['OR']:.2f} ({r['CI_95_low']:.2f}–{r['CI_95_high']:.2f})", axis=1)
        df['model']    = label; df['n'] = int(m.nobs); df['pseudo_r2'] = round(m.prsquared,4)
        print(f"\n{label}: N={int(m.nobs):,}, Pseudo-R²={m.prsquared:.3f}")
        return df, m
    except Exception as e:
        print(f"  {label} error: {e}"); return pd.DataFrame(), None

def ols_table(formula, data, label):
    try:
        m  = smf.ols(formula, data=data).fit()
        df = pd.DataFrame({
            'Variable':   m.params.index,
            'Beta':       m.params.values,
            'CI_95_low':  m.conf_int()[0].values,
            'CI_95_high': m.conf_int()[1].values,
            'p_value':    m.pvalues.values,
            'exp_Beta':   np.exp(m.params.values),
        })
        df['p_fmt'] = df['p_value'].apply(lambda p: "<0.001" if p<0.001 else f"{p:.3f}")
        df['Beta_CI'] = df.apply(lambda r: f"{r['Beta']:.3f} ({r['CI_95_low']:.3f}–{r['CI_95_high']:.3f})", axis=1)
        df['model'] = label; df['n'] = int(m.nobs); df['r2'] = round(m.rsquared,4)
        print(f"\n{label}: N={int(m.nobs):,}, R²={m.rsquared:.3f}")
        return df, m
    except Exception as e:
        print(f"  {label} error: {e}"); return pd.DataFrame(), None

# -----------------------------------------------------------------------------
# FULLY ADJUSTED MODELS
# -----------------------------------------------------------------------------
print("\n=== FULLY ADJUSTED MODEL 1: In-Hospital Mortality ===")
r_mort, m_mort = logit_table(f"in_hospital_mortality ~ {race_ref} + {ins_ref} + {full_covs}", reg, "Mortality")
if len(r_mort): r_mort.to_csv(OUT+'fully_adjusted_mortality.csv', index=False)

print("\n=== FULLY ADJUSTED MODEL 2: ICU LOS ===")
r_los, _ = ols_table(f"log_icu_los ~ {race_ref} + {ins_ref} + {full_covs}",
                      reg.dropna(subset=['log_icu_los']), "ICU LOS")
if len(r_los): r_los.to_csv(OUT+'fully_adjusted_los.csv', index=False)

print("\n=== FULLY ADJUSTED MODEL 3: 30-Day Readmission ===")
reg_r = reg[(reg['in_hospital_mortality']==0)&reg['readmit_30day'].notna()].copy()
r_readmit, _ = logit_table(f"readmit_30day ~ {race_ref} + {ins_ref} + {full_covs}", reg_r, "Readmission")
if len(r_readmit): r_readmit.to_csv(OUT+'fully_adjusted_readmit.csv', index=False)

if has_tti:
    print("\n=== FULLY ADJUSTED MODEL 4: Time to ICU ===")
    reg_tti = reg[reg['log_tti'].notna()&(reg['direct_icu_admission']==0)].copy()
    r_tti, _ = ols_table(f"log_tti ~ {race_ref} + {ins_ref} + {full_covs}", reg_tti, "Time to ICU")
    if len(r_tti): r_tti.to_csv(OUT+'fully_adjusted_time_to_icu.csv', index=False)

# -----------------------------------------------------------------------------
# TEMPORAL TREND ANALYSIS
# -----------------------------------------------------------------------------
print("\n=== TEMPORAL TREND ANALYSIS ===")
if 'anchor_year_group' in cohort.columns:
    trend = cohort.groupby(['anchor_year_group','race_group']).agg(
        n=('in_hospital_mortality','count'),
        deaths=('in_hospital_mortality','sum')
    ).reset_index()
    trend['mortality_pct'] = trend['deaths']/trend['n']*100
    trend.to_csv(OUT+'temporal_trend_raw.csv', index=False)
    print("Crude trend:\n", trend.pivot(index='anchor_year_group',columns='race_group',values='mortality_pct').round(1).to_string())

    # Race × year interaction test
    f_int  = (f"in_hospital_mortality ~ {race_ref} + C(year_group) + "
              f"C(race_group, Treatment('White')):C(year_group) + "
              f"gender_female + age_centered + non_english + C(primary_cat)")
    if has_elix: f_int += " + elixhauser_score"
    if has_gcs:  f_int += " + gcs_admission"
    f_main = f"in_hospital_mortality ~ {race_ref} + C(year_group) + {full_covs}"

    reg_t  = reg.dropna(subset=['anchor_year_group'])
    _, m_int  = logit_table(f_int,  reg_t, "Race×Year Interaction")
    _, m_main = logit_table(f_main, reg_t, "Main Effects")

    if m_int and m_main:
        lrt_stat = -2*(m_main.llf - m_int.llf)
        lrt_df   = m_int.df_model - m_main.df_model
        lrt_p    = 1 - stats.chi2.cdf(lrt_stat, df=max(1,lrt_df))
        interp   = ("Disparities CHANGING over time (p<0.05)" if lrt_p<0.05
                    else "No significant change in disparities over time")
        print(f"\nTemporal LRT: χ²={lrt_stat:.2f}, df={lrt_df:.0f}, p={lrt_p:.4f}")
        print(f"Interpretation: {interp}")
        pd.DataFrame([{'lrt':lrt_stat,'df':lrt_df,'p':lrt_p,'interpretation':interp}]).to_csv(
            OUT+'temporal_interaction_test.csv', index=False)
else:
    print("anchor_year_group not found — temporal analysis skipped.")

# -----------------------------------------------------------------------------
# MISSING DATA SUMMARY
# -----------------------------------------------------------------------------
print("\n=== MISSING DATA SUMMARY ===")
miss_vars = ['age','gender','race_group','insurance_group','in_hospital_mortality',
             'icu_los_days','hospital_los_days','readmit_30day','time_to_icu_hours',
             'gcs_admission','elixhauser_score','discharge_disposition','drg_severity_class',
             'non_english','marital_status']
miss_rows = []
for v in miss_vars:
    if v in cohort.columns:
        n = cohort[v].isna().sum(); pct = n/len(cohort)*100
        miss_rows.append({'Variable':v,'N_missing':n,'Pct_missing':round(pct,1),
                          'Action': 'Document in limitations' if pct<5 else 'Note as limitation; future MICE in Paper 2'})
miss_df = pd.DataFrame(miss_rows).sort_values('Pct_missing', ascending=False)
miss_df.to_csv(OUT+'missing_data_summary.csv', index=False)
print(miss_df.to_string(index=False))

# -----------------------------------------------------------------------------
# FORMATTED TABLE 2 FOR PUBLICATION (all models combined)
# -----------------------------------------------------------------------------
print("\n=== BUILDING TABLE 2 ===")
pub_rows = []
for fname, outcome, is_ols in [
    ('fully_adjusted_mortality.csv', 'In-Hospital Mortality (aOR, 95% CI)', False),
    ('fully_adjusted_los.csv',       'ICU LOS (β, 95% CI, log-days)',       True),
    ('fully_adjusted_readmit.csv',   '30-Day Readmission (aOR, 95% CI)',    False),
    ('fully_adjusted_time_to_icu.csv','Time to ICU (β, 95% CI, log-hours)', True),
]:
    try:
        df  = pd.read_csv(OUT+fname)
        sub = df[df['Variable'].str.contains('race_group|insurance_group')].copy()
        sub['Clean'] = (sub['Variable']
            .str.replace(r"C\(race_group, Treatment\('White'\)\)\[T\.", '', regex=True)
            .str.replace(r"C\(insurance_group, Treatment\('Private'\)\)\[T\.", '', regex=True)
            .str.replace(r"\]",'',regex=True))
        if is_ols:
            sub['Effect'] = sub.apply(lambda r: f"β={r['Beta']:.3f} ({r['CI_95_low']:.3f}–{r['CI_95_high']:.3f})",axis=1)
        else:
            sub['Effect'] = sub.apply(lambda r: f"{r['OR']:.2f} ({r['CI_95_low']:.2f}–{r['CI_95_high']:.2f})",axis=1)
        sub['p_format'] = sub['p_value'].apply(lambda p: "<0.001" if p<0.001 else f"{p:.3f}")
        sub['Outcome']  = outcome
        pub_rows.append(sub[['Outcome','Clean','Effect','p_format']])
    except FileNotFoundError:
        pass

if pub_rows:
    table2 = pd.concat(pub_rows, ignore_index=True)
    table2.columns = ['Outcome','Variable','Effect (95% CI)','p-value']
    table2.to_csv(OUT+'table2_regression_results.csv', index=False)
    print("\nTable 2 preview:")
    print(table2.to_string(index=False))

print("\n=== STEP 3c COMPLETE ===")
