# NEUROS Pipeline

Machine learning pipeline for predicting in-hospital mortality and 30-day readmission
in neurosurgical ICU patients, derived from the MIMIC-IV database.

## Contents

- `notebooks/Neuros2.ipynb` — end-to-end pipeline: cohort extraction, feature
  engineering, model training (logistic regression and XGBoost), evaluation, and SHAP-based
  interpretability.
- `models/` — trained model artifacts (`.joblib`) for both outcomes:
  - `lr_model_*` / `xgb_model_*` — logistic regression and XGBoost classifiers
  - `scaler_*` — corresponding feature scalers

## Data

This project uses the [MIMIC-IV](https://physionet.org/content/mimiciv/) critical care
database. Access requires completing PhysioNet's credentialing process and a data use
agreement. **No patient-level data is included in this repository** — only code and
trained model artifacts. To reproduce the pipeline, obtain your own credentialed access
to MIMIC-IV and point the notebook at your local copy of the data.

## Status

Code reflects the current pipeline as developed in Google Colab. Model artifacts in
`models/` are from preliminary runs; full training/evaluation is still in progress.

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
