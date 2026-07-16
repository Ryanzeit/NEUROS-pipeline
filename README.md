# NEUROS Pipeline

Machine learning pipeline for predicting in-hospital mortality and 30-day readmission
in neurosurgical ICU patients, derived from the MIMIC-IV database.

## Contents

- `pipeline/` — cohort extraction and descriptive/statistical analysis scripts (`step1`–`step6`).
  Run in order against raw MIMIC-IV tables to produce the enriched cohort CSV
  (`neurosurg_cohort_enriched.csv`) that the modeling notebook expects, plus the
  descriptive statistics, regression tables, and figures used in the manuscripts.
- `notebooks/Neuros2_pub2_model_clean.ipynb` — **canonical** Publication 2 modeling
  notebook: takes the enriched cohort CSV from `pipeline/` and does feature engineering,
  model training (logistic regression and XGBoost), evaluation, fairness/subgroup
  analysis, and SHAP-based interpretability. Ends by saving its own trained model files
  with `joblib.dump(...)`. All configurable paths (`DRIVE`, `MIMIC`, `OUT`, `COHORT`) live
  in a single early cell — update those four variables to point at your own data before
  running.
- `notebooks/Neuros2.ipynb` — the original, unedited working notebook, kept for
  provenance. It includes abandoned file-search attempts, duplicate path definitions, and
  Colab-session-specific packaging/download cells that only work inside a live Colab
  runtime. Use the `_clean` version above to actually run or read the pipeline.
- `models/` — trained model artifacts (`.joblib`) for both outcomes, provided as a
  reference/reproducibility snapshot:
  - `lr_model_*` / `xgb_model_*` — logistic regression and XGBoost classifiers
  - `scaler_*` — corresponding feature scalers

  **These are not required to run the notebook.** They exist only to reproduce the exact
  predictions reported in the paper for the original MIMIC-IV cohort (4,281 admissions).
  If you want to train on a different cohort or dataset, ignore `models/` entirely — run
  `pipeline/` then `notebooks/Neuros2.ipynb` on your own data, and it will fit and save new
  model files from scratch.

## Data

This project uses the [MIMIC-IV](https://physionet.org/content/mimiciv/) critical care
database. Access requires completing PhysioNet's credentialing process and a data use
agreement. **No patient-level data is included in this repository** — only code and
trained model artifacts.

The `pipeline/` scripts are written against MIMIC-IV's own table structure (`icustays`,
`chartevents`, `sofa`, `kdigo_stages`, `diagnoses_icd`, `admissions`, `patients`, etc.), so
"your own data" in practice means your own credentialed MIMIC-IV extract or cohort
(e.g. a different admission window, or a similarly-structured dataset) — not an
arbitrary hospital export. Adapting the scripts to a genuinely different schema would
require editing the column/table references in `pipeline/step1_cohort_extraction.py`.

## Expected input schema (`notebooks/Neuros2_pub2_model_clean.ipynb`)

The modeling notebook doesn't care where its input CSV comes from, but it does require
these columns to already exist, with these names, in the cohort file you point it at
(this is exactly what `pipeline/` produces from MIMIC-IV):

- **Identifiers / splitting**: `subject_id`, `stay_id`, `split`, `anchor_year_group`, `year_numeric`
- **Outcomes**: `in_hospital_mortality`, `readmit_30day`
- **Demographics**: `age`, `male`, `gender`, `race_group`, `insurance_group`, `non_english`
- **Clinical**: `gcs_admission`, `sofa_score`, `elixhauser_score`, `ventilated`/`ventilation`, `neurosurg_category`, `primary_dx`
- **Comorbidities** (Elixhauser categories, one column per flag): `congestive_heart_failure`,
  `cardiac_arrhythmia`, `valvular_disease`, `pulmonary_circulation`, `peripheral_vascular`,
  `hypertension_uncomplicated`, `hypertension_complicated`, `paralysis`, `other_neurological`,
  `chronic_pulmonary`, `diabetes_uncomplicated`, `diabetes_complicated`, `hypothyroidism`,
  `renal_failure`, `liver_disease`, `coagulopathy`, `obesity`, `weight_loss`,
  `fluid_electrolyte`, `blood_loss_anemia`, `deficiency_anemia`, `alcohol_abuse`,
  `drug_abuse`, `depression`

To use a non-MIMIC-IV dataset, you'd need to derive all of the above from your own data
(most directly by writing your own version of `pipeline/step1_cohort_extraction.py` that
outputs this same schema) — the notebook itself is otherwise data-source-agnostic.

## Status

Code reflects the current pipeline as developed in Google Colab. Model artifacts in
`models/` are from preliminary runs on the original MIMIC-IV cohort; full training/
evaluation, and the debiasing work planned for Publication 3, are still in progress.

## Setup

```bash
pip install -r requirements.txt
```

## Authors

- Ryan Zeitouny (co-first author, equal contribution)
- Zane Salman (co-first author, equal contribution)
- Samer Salman
- Anita Bhansali

See [CITATION.cff](CITATION.cff) for citation metadata.

## License

See [LICENSE](LICENSE).
