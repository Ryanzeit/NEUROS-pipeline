# =============================================================================
# PUBLICATION 1 — STEP 1: COHORT EXTRACTION
# "Racial, socioeconomic, and insurance-based disparities in neurosurgical
#  ICU outcomes: a MIMIC-IV cohort analysis"
#
# MIMIC-IV v3.0 | ICD-10 only | Google Colab
# Output: outputs/neurosurg_cohort_raw.csv
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
ICU  = os.path.join(BASE, 'icu/')
OUT  = os.path.join(BASE, 'outputs/')
os.makedirs(OUT, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD CORE TABLES
# -----------------------------------------------------------------------------
print("Loading tables...")
admissions = pd.read_csv(HOSP + 'admissions.csv.gz',     compression='gzip', low_memory=False)
patients   = pd.read_csv(HOSP + 'patients.csv.gz',       compression='gzip', low_memory=False)
icustays   = pd.read_csv(ICU  + 'icustays.csv.gz',       compression='gzip', low_memory=False)
diagnoses  = pd.read_csv(HOSP + 'diagnoses_icd.csv.gz',  compression='gzip', low_memory=False)
procedures = pd.read_csv(HOSP + 'procedures_icd.csv.gz', compression='gzip', low_memory=False)
print("Tables loaded.")

# -----------------------------------------------------------------------------
# ICD-10 CODE LISTS
# -----------------------------------------------------------------------------
TBI_CODES = [
    'S06','S060','S061','S062','S063','S064','S065','S066','S068','S069',
    'S090','S091','S092','S093','S094','S095','S096','S097','S098','S099'
]
HEMORRHAGIC_CODES = [
    'I60','I600','I601','I602','I603','I604','I605','I606','I607','I608','I609',
    'I61','I610','I611','I612','I613','I614','I615','I616','I618','I619',
    'I62','I620','I621','I629'
]
ISCHEMIC_CODES = [
    'I63','I630','I631','I632','I633','I634','I635','I636','I638','I639'
]
CRANIOTOMY_PROC_CODES = [
    '00N00ZZ','00N10ZZ','00N20ZZ','00N30ZZ','00N40ZZ',
    '00B00ZZ','00B10ZZ','00B20ZZ','00B30ZZ','00B40ZZ',
    '00C00ZZ','00C10ZZ','00C20ZZ','00C30ZZ','00C40ZZ',
    '009000Z','009030Z','009040Z',
    '00T00ZZ','00T10ZZ','00T20ZZ',
    '0NN00ZZ','0NN10ZZ',
]
SPINE_PROC_CODES = [
    '0RG','0SG','0RB','0SB','0RC','0SC','0RN','0SN',
    '0RT','0ST','0RP','0SP','0RQ','0SQ','0RU','0SU','0RW','0SW',
]
BRAIN_TUMOR_CODES = [
    'C71','C710','C711','C712','C713','C714','C715','C716','C717','C718','C719',
    'D33','D330','D331','D332','D333','D334','D337','D339',
    'D43','D430','D431','D432','D433','D434','D437','D439'
]

# -----------------------------------------------------------------------------
# FILTER ICD-10 ONLY
# -----------------------------------------------------------------------------
dx_10 = diagnoses[diagnoses['icd_version'] == 10].copy()
px_10 = procedures[procedures['icd_version'] == 10].copy()

def prefix_match(series, code_list):
    pattern = '|'.join(['^' + c for c in code_list])
    return series.str.match(pattern, na=False)

# -----------------------------------------------------------------------------
# IDENTIFY NEUROSURGICAL HADM_IDs
# -----------------------------------------------------------------------------
tbi_hadm               = dx_10[prefix_match(dx_10['icd_code'], TBI_CODES)]['hadm_id'].unique()
hemorrhagic_hadm       = dx_10[prefix_match(dx_10['icd_code'], HEMORRHAGIC_CODES)]['hadm_id'].unique()
ischemic_hadm          = dx_10[prefix_match(dx_10['icd_code'], ISCHEMIC_CODES)]['hadm_id'].unique()
tumor_hadm             = dx_10[prefix_match(dx_10['icd_code'], BRAIN_TUMOR_CODES)]['hadm_id'].unique()
craniotomy_hadm        = px_10[prefix_match(px_10['icd_code'], CRANIOTOMY_PROC_CODES)]['hadm_id'].unique()
spine_hadm             = px_10[prefix_match(px_10['icd_code'], SPINE_PROC_CODES)]['hadm_id'].unique()
ischemic_surgical_hadm = np.intersect1d(ischemic_hadm, craniotomy_hadm)

all_neurosurg_hadm = np.union1d(
    np.union1d(tbi_hadm, hemorrhagic_hadm),
    np.union1d(
        np.union1d(ischemic_surgical_hadm, craniotomy_hadm),
        np.union1d(spine_hadm, tumor_hadm)
    )
)
print(f"Total neurosurgical admissions identified: {len(all_neurosurg_hadm):,}")

# -----------------------------------------------------------------------------
# FILTER ICU STAYS — first stay per admission only
# -----------------------------------------------------------------------------
neuro_icu = icustays[icustays['hadm_id'].isin(all_neurosurg_hadm)].copy()
neuro_icu = neuro_icu.sort_values('intime').groupby('hadm_id').first().reset_index()
print(f"First ICU stay per admission: {len(neuro_icu):,}")

# -----------------------------------------------------------------------------
# MERGE ADMISSIONS
# -----------------------------------------------------------------------------
for col in ['admittime','dischtime','deathtime']:
    admissions[col] = pd.to_datetime(admissions[col])
neuro_icu['intime']  = pd.to_datetime(neuro_icu['intime'])
neuro_icu['outtime'] = pd.to_datetime(neuro_icu['outtime'])

cohort = neuro_icu.merge(
    admissions[['hadm_id','subject_id','admittime','dischtime','deathtime',
                'admission_type','admission_location','discharge_location',
                'insurance','language','marital_status','race','hospital_expire_flag']],
    on='hadm_id', how='left'
)

# -----------------------------------------------------------------------------
# MERGE PATIENTS
# -----------------------------------------------------------------------------
patients['anchor_year'] = patients['anchor_year'].astype(int)
cohort = cohort.merge(
    patients[['subject_id','gender','anchor_age','anchor_year','anchor_year_group','dod']],
    on='subject_id', how='left'
)
cohort['admit_year'] = cohort['admittime'].dt.year
cohort['age']        = cohort['anchor_age'] + (cohort['admit_year'] - cohort['anchor_year'])

# -----------------------------------------------------------------------------
# INCLUSION / EXCLUSION
# -----------------------------------------------------------------------------
pre = len(cohort)
cohort = cohort[cohort['age'] >= 18]
print(f"After age ≥18:    {len(cohort):,} (removed {pre - len(cohort):,})")

pre = len(cohort)
cohort['los_hours'] = cohort['los'] * 24
cohort = cohort[cohort['los_hours'] >= 24]
print(f"After ICU ≥24h:   {len(cohort):,} (removed {pre - len(cohort):,})")

pre = len(cohort)
cohort = cohort[cohort['race'].notna() & (~cohort['race'].str.upper().isin(['UNKNOWN','']))]
print(f"After known race: {len(cohort):,} (removed {pre - len(cohort):,})")

# -----------------------------------------------------------------------------
# OUTCOMES
# -----------------------------------------------------------------------------
cohort['in_hospital_mortality'] = cohort['hospital_expire_flag'].astype(int)
cohort['icu_los_days']          = cohort['los']
cohort['hospital_los_days']     = (cohort['dischtime'] - cohort['admittime']).dt.total_seconds() / 86400

# 30-day readmission
cohort_subjects = cohort['subject_id'].unique()
all_adm         = admissions[admissions['subject_id'].isin(cohort_subjects)][
    ['subject_id','hadm_id','admittime','dischtime']].copy()

readmit_map = {}
for _, row in cohort[['subject_id','hadm_id','dischtime']].iterrows():
    disch = row['dischtime']
    if pd.isnull(disch):
        readmit_map[row['hadm_id']] = np.nan
        continue
    subsequent = all_adm[
        (all_adm['subject_id'] == row['subject_id']) &
        (all_adm['hadm_id']    != row['hadm_id']) &
        (all_adm['admittime']  >  disch) &
        (all_adm['admittime']  <= disch + pd.Timedelta(days=30))
    ]
    readmit_map[row['hadm_id']] = 1 if len(subsequent) > 0 else 0
cohort['readmit_30day'] = cohort['hadm_id'].map(readmit_map)

# -----------------------------------------------------------------------------
# HARMONIZE RACE, INSURANCE, LANGUAGE
# -----------------------------------------------------------------------------
def harmonize_race(r):
    if pd.isnull(r): return np.nan
    r = r.upper()
    if 'WHITE' in r:                              return 'White'
    elif 'BLACK' in r or 'AFRICAN' in r:          return 'Black'
    elif 'HISPANIC' in r or 'LATINO' in r:        return 'Hispanic'
    elif 'ASIAN' in r:                            return 'Asian'
    elif 'AMERICAN INDIAN' in r or 'ALASKA' in r: return 'Native American/Alaska Native'
    elif 'PACIFIC' in r or 'HAWAIIAN' in r:       return 'Pacific Islander'
    else:                                          return 'Other'

def harmonize_insurance(i):
    if pd.isnull(i): return np.nan
    i = i.upper()
    if 'MEDICARE' in i:   return 'Medicare'
    elif 'MEDICAID' in i: return 'Medicaid'
    elif 'SELF' in i:     return 'Self-Pay'
    elif 'OTHER' in i:    return 'Other'
    else:                 return 'Private'

cohort['race_group']      = cohort['race'].apply(harmonize_race)
cohort['insurance_group'] = cohort['insurance'].apply(harmonize_insurance)
cohort['non_english']     = cohort['language'].apply(
    lambda x: 0 if pd.isnull(x) or x.upper() in ['ENGLISH','ENGL'] else 1
)

# -----------------------------------------------------------------------------
# NEUROSURGICAL CATEGORY LABELS
# -----------------------------------------------------------------------------
def label_cat(hadm_id):
    cats = []
    if hadm_id in tbi_hadm:               cats.append('TBI')
    if hadm_id in hemorrhagic_hadm:       cats.append('Hemorrhagic Stroke')
    if hadm_id in ischemic_surgical_hadm: cats.append('Ischemic Stroke (Surgical)')
    if hadm_id in craniotomy_hadm:        cats.append('Craniotomy')
    if hadm_id in spine_hadm:             cats.append('Spine Surgery')
    if hadm_id in tumor_hadm:             cats.append('Brain Tumor')
    return ' | '.join(cats) if cats else 'Other Neurosurgical'

cohort['neurosurg_category'] = cohort['hadm_id'].apply(label_cat)

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
final_cols = [
    'subject_id','hadm_id','stay_id',
    'age','gender','race','race_group',
    'insurance','insurance_group','marital_status','language','non_english',
    'neurosurg_category','admission_type','admission_location','discharge_location',
    'anchor_year_group','intime','outtime','icu_los_days','los_hours',
    'admittime','dischtime','deathtime','hospital_los_days',
    'in_hospital_mortality','readmit_30day'
]
cohort[[c for c in final_cols if c in cohort.columns]].to_csv(
    OUT + 'neurosurg_cohort_raw.csv', index=False
)
print(f"\n=== STEP 1 COMPLETE ===")
print(f"Final cohort:          {len(cohort):,}")
print(f"In-hospital mortality: {cohort['in_hospital_mortality'].mean():.1%}")
print(f"30-day readmission:    {cohort['readmit_30day'].mean():.1%}")
print(f"Saved: {OUT}neurosurg_cohort_raw.csv")
