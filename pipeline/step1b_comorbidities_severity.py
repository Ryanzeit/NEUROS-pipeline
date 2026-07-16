# =============================================================================
# PUBLICATION 1 — STEP 1b: COMORBIDITIES, SEVERITY, DISPOSITION
# Run AFTER step1.
# Output: outputs/neurosurg_cohort_enriched.csv
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

cohort   = pd.read_csv(OUT + 'neurosurg_cohort_raw.csv', low_memory=False)
hadm_ids = cohort['hadm_id'].unique()
print(f"Cohort loaded: {len(cohort):,}")

# -----------------------------------------------------------------------------
# LOAD TABLES
# -----------------------------------------------------------------------------
print("Loading diagnoses...")
diagnoses = pd.read_csv(HOSP + 'diagnoses_icd.csv.gz', compression='gzip', low_memory=False)
dx_10     = diagnoses[(diagnoses['icd_version'] == 10) &
                       (diagnoses['hadm_id'].isin(hadm_ids))].copy()

print("Loading chartevents for GCS (large file — 10-20 min)...")
GCS_ITEMS   = [220739, 223900, 223901, 228112]
chart_chunks = []
try:
    for chunk in pd.read_csv(ICU + 'chartevents.csv.gz', compression='gzip',
                              chunksize=1_000_000, low_memory=False,
                              usecols=['subject_id','hadm_id','stay_id','itemid','charttime','valuenum']):
        sub = chunk[chunk['hadm_id'].isin(hadm_ids) &
                    chunk['itemid'].isin(GCS_ITEMS) &
                    chunk['valuenum'].notna()]
        if len(sub) > 0: chart_chunks.append(sub)
    chartevents = pd.concat(chart_chunks, ignore_index=True) if chart_chunks else pd.DataFrame()
    print(f"GCS rows: {len(chartevents):,}")
except Exception as e:
    print(f"Chartevents error: {e}")
    chartevents = pd.DataFrame()

# -----------------------------------------------------------------------------
# ELIXHAUSER COMORBIDITY INDEX (van Walraven weights)
# -----------------------------------------------------------------------------
ELIX_MAP = {
    'congestive_heart_failure':   ['I099','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','I43','I50','P290'],
    'cardiac_arrhythmia':         ['I441','I442','I443','I456','I459','I47','I48','I49','R000','R001','R008','T821','Z450','Z950'],
    'valvular_disease':           ['A520','I05','I06','I07','I08','I091','I098','I34','I35','I36','I37','I38','I39','Q230','Q231','Q232','Q233','Z952','Z953','Z954'],
    'pulmonary_circulation':      ['I26','I27','I280','I288','I289'],
    'peripheral_vascular':        ['I70','I71','I731','I738','I739','I771','I790','I792','K551','K558','K559','Z958','Z959'],
    'hypertension_uncomplicated': ['I10'],
    'hypertension_complicated':   ['I11','I12','I13','I15'],
    'paralysis':                  ['G041','G114','G801','G802','G81','G82','G830','G831','G832','G833','G834','G839','G83'],
    'other_neurological':         ['G10','G11','G12','G13','G20','G21','G22','G254','G255','G312','G318','G319','G32','G35','G36','G37','G40','G41','G931','G934','R470','R56'],
    'chronic_pulmonary':          ['I278','I279','J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67','J684','J701','J703'],
    'diabetes_uncomplicated':     ['E100','E101','E109','E110','E111','E119','E120','E121','E129','E130','E131','E139','E140','E141','E149'],
    'diabetes_complicated':       ['E102','E103','E104','E105','E106','E107','E108','E112','E113','E114','E115','E116','E117','E118','E132','E133','E134','E135','E136','E137','E138','E142','E143','E144','E145','E146','E147','E148'],
    'hypothyroidism':             ['E00','E01','E02','E03','E890'],
    'renal_failure':              ['I120','I131','N18','N19','N250','Z490','Z491','Z492','Z940','Z992'],
    'liver_disease':              ['B18','I85','I864','I982','K70','K711','K713','K714','K715','K717','K72','K73','K74','K760','K762','K763','K764','K765','K766','K767','K768','K769','Z944'],
    'peptic_ulcer':               ['K257','K259','K267','K269','K277','K279','K287','K289'],
    'aids_hiv':                   ['B20','B21','B22','B24'],
    'lymphoma':                   ['C81','C82','C83','C84','C85','C88','C96','C900','C902'],
    'metastatic_cancer':          ['C77','C78','C79','C80'],
    'solid_tumor':                ['C00','C01','C02','C03','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C30','C31','C32','C33','C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','C60','C61','C62','C63','C64','C65','C66','C67','C68','C69','C70','C71','C72','C73','C74','C75','C76','C97'],
    'rheumatoid_arthritis':       ['L400','M05','M06','M08','M120','M123','M15','M16','M17','M18','M19','M190','M191','M192','M1990','M45','M46'],
    'coagulopathy':               ['D65','D66','D67','D68','D691','D693','D694','D695','D696'],
    'obesity':                    ['E66'],
    'weight_loss':                ['E40','E41','E42','E43','E44','E45','E46','R634','R64'],
    'fluid_electrolyte':          ['E222','E86','E87'],
    'blood_loss_anemia':          ['D500'],
    'deficiency_anemia':          ['D508','D509','D51','D52','D53'],
    'alcohol_abuse':              ['F10','E52','G621','I426','K292','K700','K703','K709','T51','Z502','Z714','Z721'],
    'drug_abuse':                 ['F11','F12','F13','F14','F15','F16','F18','F19','Z715','Z722'],
    'psychoses':                  ['F20','F22','F23','F24','F25','F28','F29','F302','F312','F315'],
    'depression':                 ['F204','F313','F314','F315','F32','F33','F341','F412','F432'],
}
VW_WEIGHTS = {
    'congestive_heart_failure':7,'cardiac_arrhythmia':5,'valvular_disease':-1,
    'pulmonary_circulation':4,'peripheral_vascular':2,'hypertension_uncomplicated':0,
    'hypertension_complicated':0,'paralysis':7,'other_neurological':6,
    'chronic_pulmonary':3,'diabetes_uncomplicated':0,'diabetes_complicated':0,
    'hypothyroidism':0,'renal_failure':5,'liver_disease':11,'peptic_ulcer':0,
    'aids_hiv':0,'lymphoma':9,'metastatic_cancer':12,'solid_tumor':4,
    'rheumatoid_arthritis':0,'coagulopathy':3,'obesity':-4,'weight_loss':6,
    'fluid_electrolyte':5,'blood_loss_anemia':-2,'deficiency_anemia':-2,
    'alcohol_abuse':0,'drug_abuse':-7,'psychoses':0,'depression':-3,
}

print("Computing Elixhauser scores...")
dx_lookup    = dx_10.groupby('hadm_id')['icd_code'].apply(set).to_dict()
elix_records = []
for hadm_id in cohort['hadm_id']:
    codes = dx_lookup.get(hadm_id, set())
    flags = {'hadm_id': hadm_id}
    score = 0
    for condition, prefixes in ELIX_MAP.items():
        present = any(any(str(c).startswith(p) for p in prefixes) for c in codes)
        flags[condition] = int(present)
        if present: score += VW_WEIGHTS.get(condition, 0)
    flags['elixhauser_score'] = score
    elix_records.append(flags)
elix_df = pd.DataFrame(elix_records)
print(f"Done. Mean Elixhauser score: {elix_df['elixhauser_score'].mean():.1f}")

# -----------------------------------------------------------------------------
# GCS — first value within 24h of ICU admission
# -----------------------------------------------------------------------------
if len(chartevents) > 0:
    chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
    ct = cohort[['hadm_id','stay_id','intime']].copy()
    ct['intime']      = pd.to_datetime(ct['intime'])
    ct['window_end']  = ct['intime'] + pd.Timedelta(hours=24)
    ce = chartevents.merge(ct, on=['hadm_id','stay_id'], how='inner')
    ce = ce[(ce['charttime'] >= ce['intime']) & (ce['charttime'] <= ce['window_end'])]

    gcs_total    = ce[ce['itemid'] == 228112].groupby('hadm_id')['valuenum'].min()
    gcs_eye      = ce[ce['itemid'] == 220739].groupby('hadm_id')['valuenum'].min()
    gcs_verbal   = ce[ce['itemid'] == 223900].groupby('hadm_id')['valuenum'].min()
    gcs_motor    = ce[ce['itemid'] == 223901].groupby('hadm_id')['valuenum'].min()
    gcs_computed = (gcs_eye + gcs_verbal + gcs_motor)

    gcs_df = pd.DataFrame({'hadm_id': cohort['hadm_id']})
    gcs_df['gcs_total']     = gcs_df['hadm_id'].map(gcs_total)
    gcs_df['gcs_computed']  = gcs_df['hadm_id'].map(gcs_computed)
    gcs_df['gcs_admission'] = gcs_df['gcs_total'].fillna(gcs_df['gcs_computed'])
    gcs_df = gcs_df[['hadm_id','gcs_admission']]
    print(f"GCS available: {gcs_df['gcs_admission'].notna().sum():,} ({gcs_df['gcs_admission'].notna().mean():.1%})")
else:
    gcs_df = pd.DataFrame({'hadm_id': cohort['hadm_id'], 'gcs_admission': np.nan})
    print("GCS unavailable — missing for all patients.")

# -----------------------------------------------------------------------------
# DISCHARGE DISPOSITION
# -----------------------------------------------------------------------------
admissions = pd.read_csv(HOSP + 'admissions.csv.gz', compression='gzip', low_memory=False)
adm_sub    = admissions[admissions['hadm_id'].isin(hadm_ids)][['hadm_id','discharge_location']].copy()

def harmonize_disposition(d):
    if pd.isnull(d): return np.nan
    d = d.upper()
    if 'HOME' in d and 'HEALTH' not in d:   return 'Home'
    elif 'HOME HEALTH' in d:                return 'Home with Services'
    elif 'REHAB' in d:                      return 'Rehabilitation Facility'
    elif 'SKILLED' in d or 'SNF' in d:      return 'Skilled Nursing Facility'
    elif 'LONG TERM' in d or 'LTACH' in d:  return 'Long-Term Acute Care'
    elif 'HOSPICE' in d:                    return 'Hospice'
    elif 'DIED' in d or 'DEAD' in d:        return 'Died'
    elif 'TRANSFER' in d or 'ACUTE' in d:   return 'Transfer to Another Facility'
    elif 'AGAINST' in d or 'AMA' in d:      return 'Against Medical Advice'
    else:                                    return 'Other'

adm_sub['discharge_disposition'] = adm_sub['discharge_location'].apply(harmonize_disposition)

# -----------------------------------------------------------------------------
# MERGE AND SAVE
# -----------------------------------------------------------------------------
cohort_enriched = cohort.copy()
cohort_enriched = cohort_enriched.merge(elix_df, on='hadm_id', how='left')
cohort_enriched = cohort_enriched.merge(gcs_df,  on='hadm_id', how='left')
cohort_enriched = cohort_enriched.merge(adm_sub[['hadm_id','discharge_disposition']], on='hadm_id', how='left')

cohort_enriched.to_csv(OUT + 'neurosurg_cohort_enriched.csv', index=False)
print(f"\n=== STEP 1b COMPLETE ===")
print(f"Patients:              {len(cohort_enriched):,}")
print(f"Elixhauser mean (SD):  {cohort_enriched['elixhauser_score'].mean():.1f} ({cohort_enriched['elixhauser_score'].std():.1f})")
print(f"GCS mean (SD):         {cohort_enriched['gcs_admission'].mean():.1f} ({cohort_enriched['gcs_admission'].std():.1f})")
print(f"Saved: {OUT}neurosurg_cohort_enriched.csv")
