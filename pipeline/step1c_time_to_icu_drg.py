# =============================================================================
# PUBLICATION 1 — STEP 1c: TIME-TO-ICU + DRG CODES
# Run AFTER step1b.
# Adds:
#   - time_to_icu_hours: hours from hospital admission to ICU transfer
#     (delays in ICU escalation by race/insurance = key disparity variable)
#   - drg_type: Medicare Severity DRG vs standard (SES proxy)
#   - drg_severity: severity subclass from DRG
# Output: outputs/neurosurg_cohort_enriched.csv (updated in place)
# =============================================================================

import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "numpy"], check=False)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os

BASE = '/content/drive/MyDrive/mimic 4/'
HOSP = os.path.join(BASE, 'hosp/')
OUT  = os.path.join(BASE, 'outputs/')

cohort   = pd.read_csv(OUT + 'neurosurg_cohort_enriched.csv', low_memory=False)
hadm_ids = cohort['hadm_id'].unique()
print(f"Cohort loaded: {len(cohort):,}")

# -----------------------------------------------------------------------------
# TIME-TO-ICU
# Hours between hospital admission and first ICU transfer
# Disparities in this variable = delays in escalation of care
# -----------------------------------------------------------------------------
cohort['admittime'] = pd.to_datetime(cohort['admittime'])
cohort['intime']    = pd.to_datetime(cohort['intime'])

cohort['time_to_icu_hours'] = (
    cohort['intime'] - cohort['admittime']
).dt.total_seconds() / 3600

# Cap at 0 (negative = direct ICU admission, treat as 0)
cohort['time_to_icu_hours'] = cohort['time_to_icu_hours'].clip(lower=0)

# Flag direct ICU admissions (within 2 hours)
cohort['direct_icu_admission'] = (cohort['time_to_icu_hours'] <= 2).astype(int)

print(f"Time-to-ICU mean (SD): {cohort['time_to_icu_hours'].mean():.1f} ({cohort['time_to_icu_hours'].std():.1f}) hours")
print(f"Direct ICU admission (≤2h): {cohort['direct_icu_admission'].mean():.1%}")

# Quick look at disparity
print("\nTime-to-ICU by race (hours, mean):")
print(cohort.groupby('race_group')['time_to_icu_hours'].mean().round(1).to_string())
print("\nTime-to-ICU by insurance (hours, mean):")
print(cohort.groupby('insurance_group')['time_to_icu_hours'].mean().round(1).to_string())

# -----------------------------------------------------------------------------
# DRG CODES — SES proxy
# MS-DRG (Medicare Severity) vs APR-DRG; severity subclass
# -----------------------------------------------------------------------------
print("\nLoading DRG codes...")
try:
    drgcodes = pd.read_csv(HOSP + 'drgcodes.csv.gz', compression='gzip', low_memory=False)
    drg_sub  = drgcodes[drgcodes['hadm_id'].isin(hadm_ids)].copy()

    # Keep MS-DRG rows (most complete)
    msdrg = drg_sub[drg_sub['drg_type'] == 'MS-DRG'].copy()

    # Take first DRG per admission if multiple
    msdrg_first = msdrg.sort_values('drg_code').groupby('hadm_id').first().reset_index()

    # Severity subclass: 1=no complication, 2=complication, 3=major complication, 4=catastrophic
    msdrg_first = msdrg_first.rename(columns={
        'drg_type':      'drg_type',
        'drg_code':      'drg_code',
        'description':   'drg_description',
    })

    # Extract severity from description if available
    def extract_drg_severity(desc):
        if pd.isnull(desc): return np.nan
        desc = desc.upper()
        if 'WITHOUT' in desc and 'MCC' not in desc and 'CC' not in desc:
            return 'No Complication'
        elif 'W CC' in desc or 'WITH CC' in desc:
            return 'With Complication'
        elif 'W MCC' in desc or 'WITH MCC' in desc:
            return 'With Major Complication'
        else:
            return 'Other'

    msdrg_first['drg_severity_class'] = msdrg_first['drg_description'].apply(extract_drg_severity)

    # Merge into cohort
    cohort = cohort.merge(
        msdrg_first[['hadm_id','drg_type','drg_code','drg_description','drg_severity_class']],
        on='hadm_id', how='left'
    )

    print(f"DRG codes merged for {cohort['drg_code'].notna().sum():,} patients")
    print(f"\nDRG severity distribution:\n{cohort['drg_severity_class'].value_counts()}")

except FileNotFoundError:
    print("drgcodes.csv.gz not found — skipping DRG merge.")
    cohort['drg_type']           = np.nan
    cohort['drg_code']           = np.nan
    cohort['drg_description']    = np.nan
    cohort['drg_severity_class'] = np.nan

# -----------------------------------------------------------------------------
# SAVE UPDATED ENRICHED COHORT
# -----------------------------------------------------------------------------
cohort.to_csv(OUT + 'neurosurg_cohort_enriched.csv', index=False)

print(f"\n=== STEP 1c COMPLETE ===")
print(f"Added: time_to_icu_hours, direct_icu_admission, drg_type, drg_severity_class")
print(f"Saved: {OUT}neurosurg_cohort_enriched.csv")
