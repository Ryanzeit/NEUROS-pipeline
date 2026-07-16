# NEUROS Pipeline

Machine learning pipeline for predicting in-hospital mortality and 30-day readmission
in neurosurgical ICU patients, derived from the MIMIC-IV database.

## Contents

- `pipeline/` — cohort extraction and descriptive/statistical analysis scripts (`step1`–`step6`).
  Run in order against raw MIMIC-IV tables to produce the enriched cohort CSV
  (`neurosurg_cohort_enriched.csv`) that the modeling notebook expects, plus the
  descriptive statistics, regression tables, and figures used in the manuscripts.
- `notebooks/Neuros2.ipynb` — modeling pipeline: takes the enriched cohort CSV from
  `pipeline/` and does feature engineering, model training (logistic regression and
  XGBoost), evaluation, and SHAP-based interpretability. Ends by saving its own trained
  model files with `joblib.dump(...)`.
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
