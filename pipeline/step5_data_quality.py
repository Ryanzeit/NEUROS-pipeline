# =============================================================================
# PUBLICATION 1 — STEP 5: DATA QUALITY + CONSORT NUMBERS
# Run AFTER step4b.
# Output: data_quality_report.txt
# =============================================================================

import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "numpy"], check=False)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os

BASE = '/content/drive/MyDrive/mimic 4/'
OUT  = os.path.join(BASE, 'outputs/')

try:
    cohort = pd.read_csv(OUT+'neurosurg_cohort_enriched.csv', low_memory=False)
except FileNotFoundError:
    cohort = pd.read_csv(OUT+'neurosurg_cohort_raw.csv', low_memory=False)

lines = []
def log(msg=''):
    print(msg); lines.append(msg)

log("="*60)
log("DATA QUALITY & CONSORT FLOW REPORT")
log("Publication 1 — Neurosurgical ICU Cohort (MIMIC-IV v3.0)")
log("="*60)

log(f"\nFinal analytic cohort: N = {len(cohort):,}")

log("\n--- MISSING DATA ---")
for v in ['age','gender','race_group','insurance_group','in_hospital_mortality',
          'icu_los_days','hospital_los_days','readmit_30day','time_to_icu_hours',
          'gcs_admission','elixhauser_score','discharge_disposition','drg_severity_class',
          'non_english','marital_status']:
    if v in cohort.columns:
        n=cohort[v].isna().sum(); pct=n/len(cohort)*100
        log(f"  {v:<35} {n:,} ({pct:.1f}%) missing")

log("\n--- OUTCOMES ---")
log(f"In-hospital mortality:   {cohort['in_hospital_mortality'].sum():,} ({cohort['in_hospital_mortality'].mean():.1%})")
log(f"ICU LOS mean (SD):       {cohort['icu_los_days'].mean():.1f} ({cohort['icu_los_days'].std():.1f}) days")
log(f"ICU LOS median [IQR]:    {cohort['icu_los_days'].median():.1f} [{cohort['icu_los_days'].quantile(0.25):.1f}–{cohort['icu_los_days'].quantile(0.75):.1f}]")
log(f"Hospital LOS mean (SD):  {cohort['hospital_los_days'].mean():.1f} ({cohort['hospital_los_days'].std():.1f}) days")
r30 = cohort[cohort['in_hospital_mortality']==0]['readmit_30day'].dropna()
log(f"30-day readmission:      {r30.sum():.0f}/{len(r30):,} survivors ({r30.mean():.1%})")
if 'time_to_icu_hours' in cohort.columns:
    log(f"Time-to-ICU mean (SD):   {cohort['time_to_icu_hours'].mean():.1f} ({cohort['time_to_icu_hours'].std():.1f}) hours")
    log(f"Direct ICU (≤2h):        {cohort['direct_icu_admission'].mean():.1%}")

log("\n--- RACE/ETHNICITY ---")
for r,n in cohort['race_group'].value_counts().items():
    log(f"  {r:<40} {n:,} ({n/len(cohort):.1%})")

log("\n--- INSURANCE ---")
for i,n in cohort['insurance_group'].value_counts().items():
    log(f"  {i:<25} {n:,} ({n/len(cohort):.1%})")

log("\n--- NEUROSURGICAL CATEGORIES ---")
for cat,n in cohort['neurosurg_category'].value_counts().items():
    log(f"  {str(cat)[:60]:<60} {n:,} ({n/len(cohort):.1%})")

log("\n--- DEMOGRAPHICS ---")
log(f"Age mean (SD):   {cohort['age'].mean():.1f} ({cohort['age'].std():.1f})")
log(f"Age median:      {cohort['age'].median():.1f}")
log(f"Age range:       {cohort['age'].min():.0f}–{cohort['age'].max():.0f}")
log(f"Female:          {(cohort['gender']=='F').sum():,} ({(cohort['gender']=='F').mean():.1%})")
if 'elixhauser_score' in cohort.columns:
    log(f"Elixhauser mean (SD): {cohort['elixhauser_score'].mean():.1f} ({cohort['elixhauser_score'].std():.1f})")
if 'gcs_admission' in cohort.columns:
    log(f"GCS mean (SD):        {cohort['gcs_admission'].mean():.1f} ({cohort['gcs_admission'].std():.1f})")
    log(f"GCS available:        {cohort['gcs_admission'].notna().sum():,} ({cohort['gcs_admission'].notna().mean():.1%})")

if 'anchor_year_group' in cohort.columns:
    log("\n--- TEMPORAL DISTRIBUTION ---")
    for y,n in cohort['anchor_year_group'].value_counts().sort_index().items():
        log(f"  {y}: {n:,} ({n/len(cohort):.1%})")

log("\n--- INTEGRITY CHECKS ---")
dups = cohort['hadm_id'].duplicated().sum()
log(f"Duplicate hadm_id:    {dups} ({'OK' if dups==0 else 'WARNING — investigate'})")
neg_los = (cohort['icu_los_days']<0).sum()
log(f"Negative ICU LOS:     {neg_los} ({'OK' if neg_los==0 else 'WARNING'})")
if 'hospital_los_days' in cohort.columns:
    log(f"Negative hosp LOS:    {(cohort['hospital_los_days']<0).sum()}")
log(f"Age > 110:            {(cohort['age']>110).sum()} (expected — MIMIC caps at 91+)")

log("\n"+"="*60)
log("REPORT COMPLETE")
log("="*60)

with open(OUT+'data_quality_report.txt','w') as f:
    f.write('\n'.join(lines))
print(f"\nSaved: {OUT}data_quality_report.txt")
print("\n=== STEP 5 COMPLETE ===")
