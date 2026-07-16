# =============================================================================
# PUBLICATION 1 — STEP 2: DESCRIPTIVE STATISTICS + TABLE 1
# Run AFTER step1c.
# Output: table1_by_race.csv, table1_by_insurance.csv, descriptive_by_race.csv
# =============================================================================

import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "numpy", "scipy"], check=False)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from scipy import stats
import os

BASE = '/content/drive/MyDrive/mimic 4/'
OUT  = os.path.join(BASE, 'outputs/')

cohort = pd.read_csv(OUT + 'neurosurg_cohort_enriched.csv', low_memory=False)
print(f"Cohort loaded: {len(cohort):,}")

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def summarize_continuous(var, group_var, df):
    groups  = sorted(df[group_var].dropna().unique())
    results = {}
    for g in groups:
        vals = df[df[group_var] == g][var].dropna()
        results[g] = f"{vals.mean():.1f} ({vals.std():.1f})"
    overall = df[var].dropna()
    results['Overall'] = f"{overall.mean():.1f} ({overall.std():.1f})"
    group_vals = [df[df[group_var] == g][var].dropna() for g in groups]
    if len(group_vals) >= 2:
        _, p = stats.f_oneway(*group_vals)
        results['p_value'] = "<0.001" if p < 0.001 else f"{p:.3f}"
    return results

def summarize_categorical(var, group_var, df):
    groups  = sorted(df[group_var].dropna().unique())
    cats    = sorted(df[var].dropna().unique())
    results = {}
    for cat in cats:
        row = {}
        for g in groups:
            sub = df[df[group_var] == g]
            n   = (sub[var] == cat).sum()
            row[g] = f"{n:,} ({n/len(sub)*100:.1f}%)" if len(sub) > 0 else "0 (0.0%)"
        n_all = (df[var] == cat).sum()
        row['Overall'] = f"{n_all:,} ({n_all/len(df)*100:.1f}%)"
        results[cat] = row
    ct    = pd.crosstab(df[var], df[group_var])
    p_str = "N/A"
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        _, p, _, _ = stats.chi2_contingency(ct)
        p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
    return results, p_str

def build_table(cohort, group_var, groups, extra_vars=None):
    rows = []

    def add_n():
        row = {'Variable': 'N', 'p_value': ''}
        for g in groups: row[g] = f"{(cohort[group_var]==g).sum():,}"
        row['Overall'] = f"{len(cohort):,}"
        return [row]

    def add_cont(label, var):
        if var not in cohort.columns: return []
        res = summarize_continuous(var, group_var, cohort)
        row = {'Variable': label, 'p_value': res.get('p_value','')}
        for g in groups: row[g] = res.get(g,'')
        row['Overall'] = res.get('Overall','')
        return [row]

    def add_cat(label, var, label_map=None):
        if var not in cohort.columns: return []
        cats, p = summarize_categorical(var, group_var, cohort)
        rows_out = [{'Variable': label, 'p_value': p, **{g:'' for g in groups}, 'Overall':''}]
        for cat, vals in cats.items():
            display = label_map.get(cat, str(cat)) if label_map else str(cat)
            row = {'Variable': f"  {display}", 'p_value': ''}
            for g in groups: row[g] = vals.get(g,'')
            row['Overall'] = vals.get('Overall','')
            rows_out.append(row)
        return rows_out

    rows += add_n()
    rows += add_cont('Age, mean (SD)', 'age')
    rows += add_cat('Gender', 'gender', {'F':'Female','M':'Male'})
    rows += add_cat('Insurance', 'insurance_group')
    rows += add_cat('Marital Status', 'marital_status')
    rows += add_cat('Non-English Language', 'non_english', {0:'No',1:'Yes'})
    rows += add_cat('Neurosurgical Category', 'neurosurg_category')
    rows += add_cont('Elixhauser Score, mean (SD)', 'elixhauser_score')
    rows += add_cont('GCS on Admission, mean (SD)', 'gcs_admission')
    rows += add_cont('Time to ICU, hours, mean (SD)', 'time_to_icu_hours')
    rows += add_cat('Direct ICU Admission (≤2h)', 'direct_icu_admission', {0:'No',1:'Yes'})
    rows += add_cat('DRG Severity Class', 'drg_severity_class')
    rows += add_cont('ICU LOS, days, mean (SD)', 'icu_los_days')
    rows += add_cont('Hospital LOS, days, mean (SD)', 'hospital_los_days')
    rows += add_cat('In-Hospital Mortality', 'in_hospital_mortality', {0:'No',1:'Yes'})

    survivors = cohort[cohort['in_hospital_mortality'] == 0].copy()
    cats, p   = summarize_categorical('readmit_30day', group_var, survivors)
    rows.append({'Variable':'30-Day Readmission (survivors only)', 'p_value':p,
                 **{g:'' for g in groups}, 'Overall':''})
    for cat, vals in cats.items():
        label = 'Yes' if cat == 1 else 'No'
        row   = {'Variable': f"  {label}", 'p_value': ''}
        for g in groups: row[g] = vals.get(g,'')
        row['Overall'] = vals.get('Overall','')
        rows.append(row)

    rows += add_cat('Discharge Disposition', 'discharge_disposition')

    col_order = ['Variable'] + groups + ['Overall', 'p_value']
    df = pd.DataFrame(rows)
    return df.reindex(columns=[c for c in col_order if c in df.columns])

# -----------------------------------------------------------------------------
# TABLE 1 BY RACE
# -----------------------------------------------------------------------------
race_groups = sorted(cohort['race_group'].dropna().unique())
table1_race = build_table(cohort, 'race_group', race_groups)
table1_race.to_csv(OUT + 'table1_by_race.csv', index=False)
print("Table 1 (by race) saved.")

# -----------------------------------------------------------------------------
# TABLE 1 BY INSURANCE
# -----------------------------------------------------------------------------
ins_groups = sorted(cohort['insurance_group'].dropna().unique())
table1_ins = build_table(cohort, 'insurance_group', ins_groups)
table1_ins.to_csv(OUT + 'table1_by_insurance.csv', index=False)
print("Table 1 (by insurance) saved.")

# -----------------------------------------------------------------------------
# SUMMARY DESCRIPTIVE TABLE
# -----------------------------------------------------------------------------
summary = cohort.groupby('race_group').agg(
    N=('subject_id','count'),
    age_mean=('age','mean'), age_sd=('age','std'),
    pct_female=('gender', lambda x: (x=='F').mean()*100),
    elixhauser_mean=('elixhauser_score','mean'),
    gcs_mean=('gcs_admission','mean'),
    time_to_icu_mean=('time_to_icu_hours','mean'),
    icu_los_mean=('icu_los_days','mean'),
    hospital_los_mean=('hospital_los_days','mean'),
    mortality_pct=('in_hospital_mortality','mean'),
    readmit30_pct=('readmit_30day','mean')
).reset_index()
summary['mortality_pct'] *= 100
summary['readmit30_pct'] *= 100
summary.to_csv(OUT + 'descriptive_by_race.csv', index=False)
print("Descriptive summary saved.")
print(summary.to_string(index=False))

print(f"\n=== STEP 2 COMPLETE ===")
