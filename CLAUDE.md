# NEUROS Project — Claude Code Context

## Program Overview
**"Who Does Neurosurgical AI Leave Behind?"**
A multi-publication research program investigating disparities in neurosurgical ICU outcomes and algorithmic fairness in predictive modeling. Data: MIMIC-IV v3.0 via PhysioNet. Co-authors: Ryan and Zane Salman.

## Publication Sequence
- **Pub 1a** — Descriptive cohort study of disparities (4,281 admissions). Complete, citation-ready.
- **Pub 1b** — Scoping review of AI fairness in neurosurgical prediction (Zane's paper). Complete, submitted.
- **Pub 2** — Predictive modeling and fairness audit. Pipeline complete, manuscript drafted, near submission. Target: JAMIA.
- **Pub 3** — Debiasing experiments. **Current active work.**
- **Pub 4** — External validation.
- **Pub 5** — Synthesis/policy paper.

---

## Data
- **Source:** MIMIC-IV v3.0
- **Cohort file:** `Copy of neurosurg_cohort_enriched.csv`
- **Google Drive layout:** `DRIVE/MIMIC 4/` for raw MIMIC files; `DRIVE/Project Neuros/` for outputs
- **Cohort:** 4,281 adult neurosurgical ICU admissions (TBI, hemorrhagic stroke, surgically treated ischemic stroke, craniotomy, spine surgery, brain tumor)
- **Access:** Both Ryan and Zane are PhysioNet-credentialed

---

## Pub 2 Pipeline Specs

### Temporal Split
- Train: anchor year groups 2008–2010, 2011–2013, 2014–2016, 2017–2019 → year_numeric ≤ 2018 (n=3,066)
- Test: anchor year group 2020–2022 → year_numeric > 2018 (n=1,215)

### Features
- 43 features after dropping `ventilated` (all zeros in MIMIC-IV extract)
- Base: age, elixhauser_score, gcs_admission, male, non_english
- 24 comorbidity dummies
- Race dummies (White = reference, dropped)
- Insurance dummies (Private = reference, dropped)
- Primary dx dummies (first label before pipe separator, drop_first=True)
- SOFA included only if >200 non-null values (was excluded in final run)

### Models
- Logistic Regression (primary): max_iter=2000, class_weight='balanced', C=1.0, random_state=42
- XGBoost (comparator): n_estimators=300, max_depth=4, lr=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=class ratio

### Key Results
- Mortality AUROC: LR=0.852 (0.816–0.882), XGB=0.850 (0.819–0.879)
- Readmission AUROC: LR=0.603 (0.558–0.648), XGB=0.600
- Mortality Brier: LR=0.156, XGB=0.120
- Readmission Brier: LR=0.227, XGB=0.187

### Fairness Audit Results (EOD = Equal Opportunity Difference vs reference group)
**Mortality:** All EOD CIs cross zero — reassuring.
**Readmission:**
- Medicaid EOD = +0.444 (CI +0.272 to +0.638): model over-predicts readmission for Medicaid
- Other race EOD = −0.294 (CI −0.521 to −0.049): model misses ~29% more readmissions in Other race vs White

### Additional Findings
- Race-blind vs race-aware: removing demographics worsens calibration for all subgroups (+0.33–0.42 gaps), not fairer
- SHAP mortality: GCS dominates; no race/insurance in top 20
- SHAP readmission: race_Other is 3rd most important overall; SHAP=0.799 for Other race patients vs 0.027–0.029 for all others (proxy prediction concern)
- Decision curves: mortality adds net benefit (threshold 0.10–0.30); readmission provides no net benefit over treat-all
- Cluster-robust SE sensitivity: all findings stable
- Intersectional: White×Medicaid readmission EOD=+0.405 — Medicaid gap is insurance-driven

---

## Pub 3 Goals

### Objective
Test whether targeted debiasing narrows the two readmission fairness gaps without meaningfully sacrificing overall AUROC.

### Primary Targets
1. Medicaid readmission EOD = +0.444
2. Other race readmission EOD = −0.294

### Planned Methods (run all three, compare)
1. **Reweighting** — upweight Medicaid and Other race patients in training
2. **Resampling** — oversample underrepresented/mispredicted groups (SMOTE)
3. **Fairness-constrained optimization** — penalize EOD violations during training

### Notebook Structure
Single notebook: clean Pub 2 pipeline reconstruction first (clearly sectioned), then Pub 3 debiasing experiments layered on top. Pub 2 code stays logically intact and reproducible.

---

## Key Interpretive Principles
- Race attenuation after severity adjustment ≠ no disparity — may reflect upstream embedding in severity by ICU arrival
- Removing demographic variables worsens calibration for all subgroups — "race-blind" ≠ "fair"
- Equivalent LR/XGBoost AUROC is the primary argument for preferring transparent LR
- Proxy prediction (race label driving predictions for a subgroup) is a distinct fairness concern from aggregate EOD — SHAP is essential
- Weak readmission performance is expected and consistent across Pub 1a and Pub 2

---

## Tools & Libraries
- **Environment:** Google Colab (cloud) or local Jupyter
- **Key packages:** pandas, numpy, sklearn, statsmodels, xgboost, shap, imbalanced-learn (SMOTE for Pub 3), matplotlib, scipy
- **Manuscript format:** TRIPOD+AI

---

## Communication Preferences
- Direct and concise, no over-explanation
- Complete deliverables — nothing held back for follow-up
- Delegate technical/formatting judgment calls; stay engaged on scope and sequencing
- Verify outputs rigorously when asked — actual re-verification, not reassurance

---

## Pub 2 Clean Pipeline Code

The messy Colab notebook (Neuros2.ipynb) had many duplicate data-loading and path-hunting cells from debugging. Below is the clean reconstruction — same logic, no errors, with a single config block at the top to set paths. In Colab set COHORT/OUT/MIMIC to Drive paths. Locally set to wherever files live.

```python
# ── NEUROS Publication 2 — Clean Pipeline ────────────────────────────────────
# Development, Validation, and Fairness Audit of a Neurosurgical
# ICU Outcome Prediction Model Using MIMIC-IV
# ─────────────────────────────────────────────────────────────────────────────

# ── SECTION 0: CONFIG (edit these paths only) ────────────────────────────────
# In Colab:
#   COHORT = '/content/drive/MyDrive/Project Neuros/Copy of neurosurg_cohort_enriched.csv'
#   OUT    = '/content/drive/MyDrive/Project Neuros/'
#   MIMIC  = '/content/drive/MyDrive/MIMIC 4/'
# Locally, set to wherever the files live on disk.

COHORT = 'Copy of neurosurg_cohort_enriched.csv'   # <-- update path
OUT    = './'                                        # <-- update path
MIMIC  = './MIMIC 4/'                               # <-- update path

# ── SECTION 1: IMPORTS ───────────────────────────────────────────────────────
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from xgboost import XGBClassifier
import shap
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.sandwich_covariance import cov_cluster

# ── SECTION 2: LOAD COHORT ───────────────────────────────────────────────────
df = pd.read_csv(COHORT)
print(f"Cohort: {df.shape[0]:,} admissions, {df.shape[1]} variables")
print(f"In-hospital mortality : {df['in_hospital_mortality'].sum():,} "
      f"({df['in_hospital_mortality'].mean()*100:.1f}%)")
print(f"30-day readmission    : {df['readmit_30day'].sum():,} "
      f"({df['readmit_30day'].mean()*100:.1f}%)")
print(f"Race groups           : {df['race_group'].value_counts().to_dict()}")
print(f"Insurance groups      : {df['insurance_group'].value_counts().to_dict()}")

# ── SECTION 3: VENTILATION + SOFA (graceful fallback) ────────────────────────
df['ventilated'] = 0   # All-zero in MIMIC-IV extract; flagged in Methods
df['sofa_score'] = np.nan

# Attempt SOFA from MIMIC files if available
sofa_loaded = False
for fname in ['sofa.csv', 'kdigo_stages.csv']:
    fpath = os.path.join(MIMIC, fname)
    if os.path.exists(fpath):
        try:
            tmp = pd.read_csv(fpath)
            sofa_col = [c for c in tmp.columns if 'sofa' in c.lower()]
            if sofa_col and 'stay_id' in tmp.columns:
                agg = tmp.groupby('stay_id')[sofa_col[0]].max().reset_index()
                agg.columns = ['stay_id', 'sofa_score']
                df = df.merge(agg, on='stay_id', how='left')
                print(f"SOFA from {fname}: {df['sofa_score'].notna().sum():,} records")
                sofa_loaded = True
                break
        except Exception as e:
            print(f"  {fname} error: {e}")

if not sofa_loaded:
    print("SOFA unavailable — GCS + Elixhauser serve as severity proxies.")

print(f"\nShape after merge: {df.shape}")

# ── SECTION 4: MISSINGNESS REPORT ────────────────────────────────────────────
KEY_VARS = ['in_hospital_mortality', 'readmit_30day', 'gcs_admission',
            'elixhauser_score', 'insurance_group', 'sofa_score', 'ventilated']

print("=" * 60)
print("MISSINGNESS REPORT (TRIPOD+AI requirement)")
print("=" * 60)
for v in KEY_VARS:
    if v in df.columns:
        n_miss = df[v].isna().sum()
        print(f"  {v:<30} missing={n_miss:>4} ({n_miss/len(df)*100:.1f}%)")

# ── SECTION 5: TEMPORAL SPLIT + FEATURE ENGINEERING ──────────────────────────
year_map = {
    '2008 - 2010': 2009, '2011 - 2013': 2012, '2014 - 2016': 2015,
    '2017 - 2019': 2018, '2020 - 2022': 2021
}
df['year_numeric'] = df['anchor_year_group'].map(year_map)
df['split'] = np.where(df['year_numeric'] <= 2018, 'train', 'test')
print(f"\nTemporal split — Train: {(df['split']=='train').sum():,} | "
      f"Test: {(df['split']=='test').sum():,}")

# Encode categoricals
df['male']        = (df['gender'] == 'M').astype(int)
df['non_english'] = df['non_english'].fillna(0).astype(int)

# Race dummies — White is reference
race_dummies = pd.get_dummies(df['race_group'], prefix='race')
for c in [x for x in race_dummies.columns if 'White' in x]:
    race_dummies.drop(columns=c, inplace=True)

# Insurance dummies — Private is reference
ins_dummies = pd.get_dummies(df['insurance_group'], prefix='ins')
for c in [x for x in ins_dummies.columns if 'Private' in x]:
    ins_dummies.drop(columns=c, inplace=True)

# Primary diagnosis (first label before pipe separator)
df['primary_dx'] = df['neurosurg_category'].str.split(' | ').str[0]
dx_dummies = pd.get_dummies(df['primary_dx'], prefix='dx', drop_first=True)
print("Primary dx categories:", df['primary_dx'].value_counts().to_dict())

# Comorbidities
comorbidity_cols = [
    'congestive_heart_failure','cardiac_arrhythmia','valvular_disease',
    'pulmonary_circulation','peripheral_vascular','hypertension_uncomplicated',
    'hypertension_complicated','paralysis','other_neurological',
    'chronic_pulmonary','diabetes_uncomplicated','diabetes_complicated',
    'hypothyroidism','renal_failure','liver_disease','coagulopathy',
    'obesity','weight_loss','fluid_electrolyte','blood_loss_anemia',
    'deficiency_anemia','alcohol_abuse','drug_abuse','depression'
]

base_cols = ['age', 'elixhauser_score', 'gcs_admission', 'male', 'non_english']

X_df = pd.concat([
    df[base_cols].reset_index(drop=True),
    df[comorbidity_cols].reset_index(drop=True),
    race_dummies.reset_index(drop=True),
    ins_dummies.reset_index(drop=True),
    dx_dummies.reset_index(drop=True),
], axis=1)

# Add SOFA if sufficient data
if df['sofa_score'].notna().sum() > 200:
    X_df.insert(3, 'sofa_score',
                df['sofa_score'].fillna(df['sofa_score'].median()).values)
    print("SOFA included in feature matrix.")
else:
    print("SOFA excluded — GCS + Elixhauser as severity proxies.")

# Fill GCS median for rare missingness
X_df['gcs_admission'] = X_df['gcs_admission'].fillna(X_df['gcs_admission'].median())

FEATURE_COLS = X_df.columns.tolist()
print(f"\nFeature matrix: {X_df.shape[0]:,} rows x {len(FEATURE_COLS)} features")
print(FEATURE_COLS)

# ── SECTION 6: MODEL TRAINING ─────────────────────────────────────────────────
OUTCOMES = ['in_hospital_mortality', 'readmit_30day']
MODELS   = {}
RESULTS  = {}
MIN_EVENTS = 10

def metrics(y_true, y_prob, threshold=0.5):
    y_bin = (y_prob >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
    except:
        return {}
    return {
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'sens' : tp/(tp+fn) if (tp+fn) > 0 else np.nan,
        'spec' : tn/(tn+fp) if (tn+fp) > 0 else np.nan,
        'ppv'  : tp/(tp+fp) if (tp+fp) > 0 else np.nan,
        'npv'  : tn/(tn+fn) if (tn+fn) > 0 else np.nan,
    }

for outcome in OUTCOMES:
    print(f"\n{'='*60}\n  OUTCOME: {outcome}\n{'='*60}")

    y     = df[outcome].values
    split = df['split'].values
    X     = X_df.values

    X_tr, X_te = X[split=='train'], X[split=='test']
    y_tr, y_te = y[split=='train'], y[split=='test']
    print(f"  Train: n={len(y_tr):,}  events={y_tr.sum():,} ({y_tr.mean()*100:.1f}%)")
    print(f"  Test : n={len(y_te):,}  events={y_te.sum():,} ({y_te.mean()*100:.1f}%)")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=42,
                            class_weight='balanced', C=1.0)
    lr.fit(X_tr_s, y_tr)
    lr_prob_te = lr.predict_proba(X_te_s)[:, 1]

    # XGBoost
    scale_pos = (y_tr==0).sum() / (y_tr==1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric='logloss', use_label_encoder=False,
        random_state=42, verbosity=0
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_prob_te = xgb.predict_proba(X_te)[:, 1]

    lr_met  = metrics(y_te, lr_prob_te)
    xgb_met = metrics(y_te, xgb_prob_te)

    print(f"\n  {'Metric':<12} {'LogReg':>10} {'XGBoost':>10}")
    for k in ['auroc','auprc','brier','sens','spec','ppv','npv']:
        print(f"  {k:<12} {lr_met[k]:>10.3f} {xgb_met[k]:>10.3f}")

    MODELS[outcome] = {
        'lr': lr, 'xgb': xgb, 'scaler': scaler,
        'X_tr': X_tr, 'X_tr_s': X_tr_s,
        'X_te': X_te, 'X_te_s': X_te_s,
        'y_tr': y_tr, 'y_te': y_te,
        'lr_prob': lr_prob_te, 'xgb_prob': xgb_prob_te
    }
    RESULTS[outcome] = {'lr': lr_met, 'xgb': xgb_met}

print("\n✓ Models trained.")

# ── SECTION 7: BOOTSTRAP CIs ──────────────────────────────────────────────────
N_BOOT = 1000
np.random.seed(42)

def bootstrap_metrics(y_true, y_prob, n_boot=N_BOOT, threshold=0.5):
    records = []
    for _ in range(n_boot):
        idx = resample(np.arange(len(y_true)), replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            yb = (yp >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, yb).ravel()
            records.append({
                'auroc': roc_auc_score(yt, yp),
                'auprc': average_precision_score(yt, yp),
                'brier': brier_score_loss(yt, yp),
                'sens' : tp/(tp+fn) if (tp+fn) > 0 else np.nan,
                'spec' : tn/(tn+fp) if (tn+fp) > 0 else np.nan,
                'ppv'  : tp/(tp+fp) if (tp+fp) > 0 else np.nan,
                'npv'  : tn/(tn+fn) if (tn+fn) > 0 else np.nan,
            })
        except:
            pass
    return pd.DataFrame(records).quantile([0.025, 0.975])

print("Computing bootstrap CIs (1000 iterations)...")
BOOT_CI = {}
for outcome in OUTCOMES:
    BOOT_CI[outcome] = {}
    y_te = MODELS[outcome]['y_te']
    for mname in ['lr', 'xgb']:
        prob = MODELS[outcome][f'{mname}_prob']
        ci   = bootstrap_metrics(y_te, prob)
        BOOT_CI[outcome][mname] = ci
        print(f"\n{outcome} | {mname.upper()}")
        print(f"  {'Metric':<10} {'Point':>8} {'2.5%':>10} {'97.5%':>10}")
        for k in ['auroc','auprc','brier','sens','spec','ppv','npv']:
            pt = RESULTS[outcome][mname][k]
            print(f"  {k:<10} {pt:>8.3f} {ci.loc[0.025,k]:>10.3f} {ci.loc[0.975,k]:>10.3f}")

print("\n✓ Bootstrap CIs complete.")

# ── SECTION 8: LR COEFFICIENT TABLE ──────────────────────────────────────────
print("=" * 70)
print("LOGISTIC REGRESSION COEFFICIENT TABLE (TRIPOD+AI requirement)")
print("=" * 70)

for outcome in OUTCOMES:
    X_tr_s = MODELS[outcome]['X_tr_s']
    y_tr   = MODELS[outcome]['y_tr']
    X_sm   = sm.add_constant(X_tr_s)
    glm    = sm.GLM(y_tr, X_sm, family=sm.families.Binomial()).fit()

    params = glm.params[1:]
    ses    = glm.bse[1:]
    pvals  = glm.pvalues[1:]
    or_est = np.exp(params)
    or_lo  = np.exp(params - 1.96 * ses)
    or_hi  = np.exp(params + 1.96 * ses)

    rows = [{'Feature': feat, 'OR': or_est[i], 'CI_lower': or_lo[i],
             'CI_upper': or_hi[i], 'P_value': pvals[i]}
            for i, feat in enumerate(FEATURE_COLS)]
    coef_df = pd.DataFrame(rows).sort_values('P_value')
    coef_df.to_csv(os.path.join(OUT, f'pub2_coef_table_{outcome}.csv'), index=False)

    print(f"\nOutcome: {outcome}")
    for _, row in coef_df.iterrows():
        ci_str = f"({row['CI_lower']:.3f}–{row['CI_upper']:.3f})"
        p_str  = f"{row['P_value']:.4f}" if row['P_value'] >= 0.0001 else "<0.0001"
        sig    = " *" if row['P_value'] < 0.05 else ""
        print(f"  {row['Feature']:<42} OR={row['OR']:>7.3f} {ci_str:>20} p={p_str}{sig}")

# ── SECTION 9: SUBGROUP PERFORMANCE ──────────────────────────────────────────
SUBGROUPS = {
    'race_group'      : df['race_group'],
    'insurance_group' : df['insurance_group'],
    'gender'          : df['gender'],
    'primary_dx'      : df['primary_dx'],
}

sg_records = []

for outcome in OUTCOMES:
    test_mask = df['split'] == 'test'
    df_test   = df[test_mask].copy().reset_index(drop=True)
    df_test['lr_prob']  = MODELS[outcome]['lr_prob']
    df_test['xgb_prob'] = MODELS[outcome]['xgb_prob']
    df_test['y_true']   = MODELS[outcome]['y_te']

    print(f"\n{'='*70}\nSUBGROUP PERFORMANCE: {outcome}\n{'='*70}")

    for sg_name, sg_series in SUBGROUPS.items():
        sg_vals = sg_series[test_mask].values
        df_test[sg_name] = sg_vals
        print(f"\n  [{sg_name}]")

        for grp in sorted(df_test[sg_name].dropna().unique()):
            sub    = df_test[df_test[sg_name] == grp]
            n      = len(sub)
            events = int(sub['y_true'].sum())

            if events < MIN_EVENTS:
                print(f"    {grp:<38} n={n:>4}  events={events:>3}  → SUPPRESSED")
                continue
            try:
                lr_au  = roc_auc_score(sub['y_true'], sub['lr_prob'])
                xgb_au = roc_auc_score(sub['y_true'], sub['xgb_prob'])
                lr_br  = brier_score_loss(sub['y_true'], sub['lr_prob'])
                lr_bin = (sub['lr_prob'] >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(sub['y_true'], lr_bin).ravel()
                lr_sens = tp/(tp+fn) if (tp+fn) > 0 else np.nan
                lr_ppv  = tp/(tp+fp) if (tp+fp) > 0 else np.nan
                print(f"    {grp:<38} n={n:>4}  events={events:>3}  "
                      f"LR-AUROC={lr_au:.3f}  XGB-AUROC={xgb_au:.3f}  "
                      f"Sens={lr_sens:.3f}  PPV={lr_ppv:.3f}  Brier={lr_br:.3f}")
                sg_records.append({
                    'outcome': outcome, 'subgroup': sg_name, 'group': grp,
                    'n': n, 'events': events, 'lr_auroc': lr_au,
                    'xgb_auroc': xgb_au, 'lr_brier': lr_br,
                    'lr_sens': lr_sens, 'lr_ppv': lr_ppv
                })
            except Exception as e:
                print(f"    {grp:<38} → Error: {e}")

pd.DataFrame(sg_records).to_csv(os.path.join(OUT, 'pub2_subgroup_performance.csv'), index=False)
print("\n✓ Subgroup performance saved.")

# ── SECTION 10: FAIRNESS AUDIT ────────────────────────────────────────────────
def fairness_metrics(sub, ref, y_col='y_true', prob_col='lr_prob', threshold=0.5):
    def tpr_fpr(d):
        yb = (d[prob_col] >= threshold).astype(int)
        tp = ((yb==1) & (d[y_col]==1)).sum()
        fn = ((yb==0) & (d[y_col]==1)).sum()
        fp = ((yb==1) & (d[y_col]==0)).sum()
        tn = ((yb==0) & (d[y_col]==0)).sum()
        tpr = tp/(tp+fn) if (tp+fn) > 0 else np.nan
        fpr = fp/(fp+tn) if (fp+tn) > 0 else np.nan
        cal = d[prob_col].mean() - d[y_col].mean()
        return tpr, fpr, cal
    s_tpr, s_fpr, s_cal = tpr_fpr(sub)
    r_tpr, r_fpr, r_cal = tpr_fpr(ref)
    return {
        'tpr': s_tpr, 'fpr': s_fpr,
        'eod': s_tpr - r_tpr,
        'eq_odds': max(abs(s_tpr - r_tpr), abs(s_fpr - r_fpr)),
        'cal_gap': s_cal - r_cal
    }

def bootstrap_fairness(sub, ref, n_boot=500):
    records = []
    for _ in range(n_boot):
        si = resample(np.arange(len(sub)), replace=True)
        ri = resample(np.arange(len(ref)),  replace=True)
        sb, rb = sub.iloc[si], ref.iloc[ri]
        if sb['y_true'].sum() < 3 or rb['y_true'].sum() < 3:
            continue
        try:
            records.append(fairness_metrics(sb, rb))
        except:
            pass
    if not records:
        return None
    return pd.DataFrame(records).quantile([0.025, 0.975])

AUDIT_GROUPS = {
    'race_group'      : 'White',
    'insurance_group' : 'Private',
    'gender'          : 'M',
}

audit_records = []

for outcome in OUTCOMES:
    test_mask = df['split'] == 'test'
    df_test   = df[test_mask].copy().reset_index(drop=True)
    df_test['lr_prob']  = MODELS[outcome]['lr_prob']
    df_test['xgb_prob'] = MODELS[outcome]['xgb_prob']
    df_test['y_true']   = MODELS[outcome]['y_te']

    print(f"\n{'='*70}\nFAIRNESS AUDIT: {outcome}\n{'='*70}")

    for sg_col, ref_val in AUDIT_GROUPS.items():
        ref = df_test[df_test[sg_col] == ref_val]
        if ref['y_true'].sum() < MIN_EVENTS:
            print(f"  Reference {ref_val} too small — skipped.")
            continue
        print(f"\n  [{sg_col}]  Reference={ref_val} (n={len(ref)}, events={int(ref['y_true'].sum())})")
        print(f"  {'Group':<30} {'TPR':>6} {'FPR':>6} {'EOD':>8} {'EqOdds':>8} {'CalGap':>8}  95% CI (EOD)")

        for grp in sorted(df_test[sg_col].dropna().unique()):
            if grp == ref_val:
                continue
            sub = df_test[df_test[sg_col] == grp]
            if sub['y_true'].sum() < MIN_EVENTS:
                print(f"  {grp:<30} → SUPPRESSED")
                continue
            try:
                m  = fairness_metrics(sub, ref)
                ci = bootstrap_fairness(sub, ref)
                eod_lo = ci.loc[0.025,'eod'] if ci is not None else np.nan
                eod_hi = ci.loc[0.975,'eod'] if ci is not None else np.nan
                print(f"  {grp:<30} TPR={m['tpr']:>5.3f}  FPR={m['fpr']:>5.3f}  "
                      f"EOD={m['eod']:>+7.3f}  EqOdds={m['eq_odds']:>6.3f}  "
                      f"CalGap={m['cal_gap']:>+7.3f}  [{eod_lo:+.3f}, {eod_hi:+.3f}]")
                audit_records.append({
                    'outcome': outcome, 'subgroup': sg_col, 'group': grp,
                    'reference': ref_val, 'n': len(sub),
                    'events': int(sub['y_true'].sum()),
                    **m, 'eod_ci_lo': eod_lo, 'eod_ci_hi': eod_hi
                })
            except Exception as e:
                print(f"  {grp:<30} → Error: {e}")

pd.DataFrame(audit_records).to_csv(os.path.join(OUT, 'pub2_fairness_audit.csv'), index=False)
print("\n✓ Fairness audit saved.")

# ── SECTION 11: SENSITIVITY — CLUSTER-ROBUST SEs ─────────────────────────────
print("=" * 60)
print("SENSITIVITY: Cluster-robust SEs (clustered on subject_id)")
print("=" * 60)

for outcome in OUTCOMES:
    X_full = MODELS[outcome]['scaler'].transform(X_df.values)
    y_full = df[outcome].values
    ids    = df['subject_id'].values
    X_sm   = sm.add_constant(X_full)
    try:
        glm    = sm.GLM(y_full, X_sm, family=sm.families.Binomial()).fit()
        cov_cl = cov_cluster(glm, ids)
        se_cl  = np.sqrt(np.diag(cov_cl))[1:]
        coefs  = glm.params[1:]
        or_cl  = np.exp(coefs)
        or_lo  = np.exp(coefs - 1.96 * se_cl)
        or_hi  = np.exp(coefs + 1.96 * se_cl)
        p_cl   = 2 * (1 - stats.norm.cdf(np.abs(coefs / (se_cl + 1e-10))))

        orig_coef_df = pd.read_csv(os.path.join(OUT, f'pub2_coef_table_{outcome}.csv'))
        print(f"\nOutcome: {outcome}")
        for i, feat in enumerate(FEATURE_COLS):
            orig_row = orig_coef_df[orig_coef_df['Feature'] == feat]
            orig_p   = float(orig_row['P_value'].values[0]) if len(orig_row) else np.nan
            changed  = '⚠ FLIPPED' if (orig_p < 0.05) != (p_cl[i] < 0.05) else 'stable'
            if changed != 'stable' or p_cl[i] < 0.10:
                print(f"  {feat:<40} OR={or_cl[i]:.3f} "
                      f"({or_lo[i]:.2f}–{or_hi[i]:.2f}) p={p_cl[i]:.4f}  {changed}")
    except Exception as e:
        print(f"  Cluster-robust SE failed for {outcome}: {e}")

print("\n✓ Cluster-robust sensitivity done.")

# ── SECTION 12: SENSITIVITY — RACE-BLIND ──────────────────────────────────────
print("=" * 60)
print("SENSITIVITY: Race-blind vs Race-aware")
print("=" * 60)

race_cols = [c for c in FEATURE_COLS if c.startswith('race_')]
ins_cols  = [c for c in FEATURE_COLS if c.startswith('ins_')]
demo_cols = race_cols + ins_cols
feat_idx_blind = [i for i, c in enumerate(FEATURE_COLS) if c not in demo_cols]

for outcome in OUTCOMES:
    y_tr   = MODELS[outcome]['y_tr']
    y_te   = MODELS[outcome]['y_te']
    X_tr_s = MODELS[outcome]['X_tr_s']
    X_te_s = MODELS[outcome]['X_te_s']

    aware_auroc = roc_auc_score(y_te, MODELS[outcome]['lr_prob'])

    lr_blind = LogisticRegression(max_iter=2000, random_state=42,
                                  class_weight='balanced', C=1.0)
    lr_blind.fit(X_tr_s[:, feat_idx_blind], y_tr)
    lr_blind_prob = lr_blind.predict_proba(X_te_s[:, feat_idx_blind])[:, 1]
    blind_auroc   = roc_auc_score(y_te, lr_blind_prob)

    print(f"\nOutcome: {outcome}")
    print(f"  Race-aware AUROC: {aware_auroc:.3f}")
    print(f"  Race-blind AUROC: {blind_auroc:.3f}  (diff={aware_auroc-blind_auroc:+.4f})")

    test_mask = df['split'] == 'test'
    df_test   = df[test_mask].copy().reset_index(drop=True)
    df_test['blind_prob'] = lr_blind_prob
    df_test['y_true']     = y_te
    print(f"  Calibration gaps (race-blind):")
    for grp in sorted(df_test['race_group'].dropna().unique()):
        sub = df_test[df_test['race_group'] == grp]
        if sub['y_true'].sum() < MIN_EVENTS:
            continue
        cal_gap = sub['blind_prob'].mean() - sub['y_true'].mean()
        print(f"    {grp:<30} CalGap={cal_gap:+.4f}")

print("\n✓ Race-blind sensitivity done.")

# ── SECTION 13: INTERSECTIONAL SUBGROUP ──────────────────────────────────────
print("=" * 60)
print("INTERSECTIONAL SUBGROUP: Race × Insurance")
print("=" * 60)

INTERSECTIONS = [
    ('Black',    'Medicaid'),
    ('Black',    'Medicare'),
    ('Hispanic', 'Medicaid'),
    ('White',    'Private'),
    ('White',    'Medicare'),
    ('White',    'Medicaid'),
    ('Other',    'Medicaid'),
    ('Other',    'Private'),
]

intersect_records = []

for outcome in OUTCOMES:
    test_mask = df['split'] == 'test'
    df_test   = df[test_mask].copy().reset_index(drop=True)
    df_test['lr_prob'] = MODELS[outcome]['lr_prob']
    df_test['y_true']  = MODELS[outcome]['y_te']

    ref = df_test[(df_test['race_group']=='White') & (df_test['insurance_group']=='Private')]
    print(f"\nOutcome: {outcome}")
    print(f"  {'Race × Insurance':<32} {'N':>5} {'Events':>7} {'LR-AUROC':>10} {'EOD':>8} {'CalGap':>8}")

    for race_val, ins_val in INTERSECTIONS:
        sub    = df_test[(df_test['race_group']==race_val) & (df_test['insurance_group']==ins_val)]
        n      = len(sub)
        events = int(sub['y_true'].sum())
        label  = f"{race_val} × {ins_val}"
        if events < MIN_EVENTS:
            print(f"  {label:<32} {n:>5} {events:>7}  SUPPRESSED")
            continue
        try:
            auroc = roc_auc_score(sub['y_true'], sub['lr_prob'])
            m = fairness_metrics(sub, ref) if len(ref) > 0 and ref['y_true'].sum() >= MIN_EVENTS else {}
            eod_str    = f"{m['eod']:+.3f}" if 'eod' in m else 'N/A'
            calgap_str = f"{m['cal_gap']:+.3f}" if 'cal_gap' in m else 'N/A'
            ref_note   = '← reference' if (race_val=='White' and ins_val=='Private') else ''
            print(f"  {label:<32} {n:>5} {events:>7} {auroc:>10.3f} {eod_str:>8} {calgap_str:>8}  {ref_note}")
            intersect_records.append({
                'outcome': outcome, 'race': race_val, 'insurance': ins_val,
                'n': n, 'events': events, 'lr_auroc': auroc,
                'eod': eod_str, 'cal_gap': calgap_str
            })
        except Exception as e:
            print(f"  {label:<32} → Error: {e}")

pd.DataFrame(intersect_records).to_csv(
    os.path.join(OUT, 'pub2_intersectional_subgroup.csv'), index=False)
print("\n✓ Intersectional subgroup saved.")

# ── SECTION 14: PLOTS ─────────────────────────────────────────────────────────
# Calibration
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for row, outcome in enumerate(OUTCOMES):
    y_te    = MODELS[outcome]['y_te']
    lr_prob = MODELS[outcome]['lr_prob']
    xgb_prob= MODELS[outcome]['xgb_prob']
    test_mask = df['split'] == 'test'
    df_test = df[test_mask].copy().reset_index(drop=True)
    df_test['y_true']  = y_te
    df_test['lr_prob'] = lr_prob

    ax = axes[row, 0]
    for prob, label, color in [(lr_prob,'LogReg','steelblue'),(xgb_prob,'XGBoost','darkorange')]:
        fp, mp = calibration_curve(y_te, prob, n_bins=10)
        ax.plot(mp, fp, 'o-', color=color, label=label)
    ax.plot([0,1],[0,1],'k--', alpha=0.5)
    ax.set_title(f'{outcome.replace("_"," ").title()}\nOverall Calibration')
    ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction Positive')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    colors = plt.cm.tab10.colors
    for col_idx, (sg_col, ax) in enumerate([('race_group', axes[row,1]),
                                              ('insurance_group', axes[row,2])]):
        for ci, grp in enumerate(sorted(df_test[sg_col].dropna().unique())):
            sub = df_test[df_test[sg_col]==grp]
            if sub['y_true'].sum() < MIN_EVENTS or len(sub) < 20:
                continue
            try:
                fp, mp = calibration_curve(sub['y_true'], sub['lr_prob'], n_bins=5)
                ax.plot(mp, fp, 'o-', color=colors[ci % 10], label=grp, alpha=0.8)
            except:
                pass
        ax.plot([0,1],[0,1],'k--', alpha=0.5)
        ax.set_title(f'By {"Race" if "race" in sg_col else "Insurance"}\n(LogReg)')
        ax.set_xlabel('Mean Predicted Probability')
        ax.legend(fontsize=6); ax.grid(alpha=0.3)

plt.suptitle('Calibration Plots — NEUROS Pub 2', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'pub2_calibration.png'), dpi=300, bbox_inches='tight')
plt.show()

# ROC + PR
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for row, outcome in enumerate(OUTCOMES):
    y_te     = MODELS[outcome]['y_te']
    lr_prob  = MODELS[outcome]['lr_prob']
    xgb_prob = MODELS[outcome]['xgb_prob']

    ax = axes[row, 0]
    for prob, label, color in [(lr_prob,'LogReg','steelblue'),(xgb_prob,'XGBoost','darkorange')]:
        fpr, tpr, _ = roc_curve(y_te, prob)
        au = roc_auc_score(y_te, prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUROC={au:.3f})')
    ax.plot([0,1],[0,1],'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {outcome.replace("_"," ").title()}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[row, 1]
    for prob, label, color in [(lr_prob,'LogReg','steelblue'),(xgb_prob,'XGBoost','darkorange')]:
        prec, rec, _ = precision_recall_curve(y_te, prob)
        au = average_precision_score(y_te, prob)
        ax.plot(rec, prec, color=color, lw=2, label=f'{label} (AUPRC={au:.3f})')
    ax.axhline(y_te.mean(), color='k', ls='--', alpha=0.5, label=f'Baseline ({y_te.mean():.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'PR Curve — {outcome.replace("_"," ").title()}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'pub2_roc_pr_curves.png'), dpi=300, bbox_inches='tight')
plt.show()

# Decision curves
def decision_curve(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    n    = len(y_true)
    prev = y_true.mean()
    nb_model, nb_all = [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred==1) & (y_true==1)).sum()
        fp = ((y_pred==1) & (y_true==0)).sum()
        nb_model.append(tp/n - fp/n * (t/(1-t)))
        nb_all.append(max(prev - (1-prev)*(t/(1-t)), 0))
    return thresholds, np.array(nb_model), np.array(nb_all)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, outcome in enumerate(OUTCOMES):
    ax       = axes[i]
    y_te     = MODELS[outcome]['y_te']
    lr_prob  = MODELS[outcome]['lr_prob']
    xgb_prob = MODELS[outcome]['xgb_prob']
    t, nb_lr, nb_all = decision_curve(y_te, lr_prob)
    _, nb_xgb, _     = decision_curve(y_te, xgb_prob)
    ax.plot(t, nb_lr,  color='steelblue',  lw=2, label='LogReg')
    ax.plot(t, nb_xgb, color='darkorange', lw=2, label='XGBoost')
    ax.plot(t, nb_all, color='gray', lw=1.5, ls='--', label='Treat All')
    ax.axhline(0, color='black', lw=1, label='Treat None')
    ax.set_xlim(0, 0.5)
    ax.set_xlabel('Threshold Probability'); ax.set_ylabel('Net Benefit')
    ax.set_title(f'Decision Curve — {outcome.replace("_"," ").title()}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'pub2_decision_curves.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ All plots saved.")

# ── SECTION 15: SHAP ──────────────────────────────────────────────────────────
print("Computing SHAP values (XGBoost)...")

for outcome in OUTCOMES:
    xgb      = MODELS[outcome]['xgb']
    X_te     = MODELS[outcome]['X_te']
    test_mask= df['split'] == 'test'
    df_test  = df[test_mask].copy().reset_index(drop=True)

    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(X_te)
    shap_df   = pd.DataFrame(shap_vals, columns=FEATURE_COLS)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_te, feature_names=FEATURE_COLS,
                      show=False, max_display=20)
    plt.title(f'SHAP Summary — {outcome.replace("_"," ").title()}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'pub2_shap_{outcome}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    top_feats = shap_df.abs().mean().nlargest(10).index.tolist()
    race_sg   = df_test['race_group'].values
    races     = [r for r in sorted(pd.Series(race_sg).dropna().unique())
                 if (race_sg == r).sum() >= 20]
    print(f"\nMean |SHAP| by race — {outcome} (top 10)")
    print(f"  {'Feature':<35}", end='')
    for r in races:
        print(f" {r[:10]:>12}", end='')
    print()
    for feat in top_feats:
        fidx = FEATURE_COLS.index(feat)
        print(f"  {feat:<35}", end='')
        for r in races:
            mask = race_sg == r
            val  = np.abs(shap_vals[mask, fidx]).mean() if mask.sum() >= 5 else float('nan')
            print(f" {val:>12.4f}", end='')
        print()

print("\n✓ SHAP complete.")

# ── SECTION 16: FINAL SUMMARY ─────────────────────────────────────────────────
print("=" * 65)
print("NEUROS PUBLICATION 2 — FINAL RESULTS SUMMARY")
print("=" * 65)
for outcome in OUTCOMES:
    print(f"\n{outcome}")
    print(f"  {'Metric':<10} {'LogReg':>22} {'XGBoost':>22}")
    for k in ['auroc','auprc','brier','sens','spec','ppv','npv']:
        lr_pt  = RESULTS[outcome]['lr'][k]
        xgb_pt = RESULTS[outcome]['xgb'][k]
        lr_lo  = float(BOOT_CI[outcome]['lr'].loc[0.025,k])
        lr_hi  = float(BOOT_CI[outcome]['lr'].loc[0.975,k])
        xgb_lo = float(BOOT_CI[outcome]['xgb'].loc[0.025,k])
        xgb_hi = float(BOOT_CI[outcome]['xgb'].loc[0.975,k])
        print(f"  {k:<10} {lr_pt:.3f} ({lr_lo:.3f}–{lr_hi:.3f})  "
              f"{xgb_pt:.3f} ({xgb_lo:.3f}–{xgb_hi:.3f})")

print(f"\n✓ All outputs saved to: {OUT}")
```
