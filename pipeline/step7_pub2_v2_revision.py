# =============================================================================
# NEUROS Publication 2 - v2 re-run (one cell, Google Colab)
# Adds: calibration slope/intercept, threshold sweep, age/language subgroups,
#       survivors-only readmission, patient-level (cluster) bootstrap,
#       baseline/benchmark models, missing-data + repeated-admission sensitivity.
# Reads : the cohort CSV already used for Pub 2 (neurosurg_cohort_enriched.csv)
# Writes: OUT_DIR/pub2_v2_results.json  (everything the manuscript needs)
#         OUT_DIR/*.csv, *.png, pub2_model_card_v2.txt
# Nothing patient-level is written to the JSON (aggregates only).
# =============================================================================
import os, sys, json, glob, time, platform, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ------------------------------- CONFIG -------------------------------------
BASE        = "/content/drive/MyDrive/"
OUT_DIR     = BASE + "Project Neuros/pub2_v2/"
COHORT_PATH = None          # None = auto-search Drive; or paste the exact path
N_BOOT      = 1000          # patient-level bootstrap replicates (main CIs)
N_BOOT_SW   = 500           # replicates for threshold-sweep CIs
SEED        = 42
MIN_EVENTS  = 10            # suppression rule (unchanged from the manuscript)
THRESHOLDS  = np.round(np.arange(0.05, 0.951, 0.05), 2)
AGE_CUT     = 65            # same cut as Pub 1a (<65 vs >=65)
RUN_SHAP    = True
# ----------------------------------------------------------------------------

try:                                         # mount Drive if running in Colab
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception:
    pass
os.makedirs(OUT_DIR, exist_ok=True)

from scipy.stats import rankdata, norm
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150

R = {"meta": {}, "errors": [], "checks": []}      # master results dict
T0 = time.time()
def log(*a): print(*a, flush=True)
def safe(name):
    def deco(fn):
        def wrap(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:               # keep going, record the failure
                R["errors"].append(f"{name}: {type(e).__name__}: {e}")
                log(f"  !! {name} failed: {e}")
                return None
        return wrap
    return deco

# ------------------------------ LOAD COHORT ---------------------------------
def _valid(path):
    try:
        h = pd.read_csv(path, nrows=5)
        need = {"in_hospital_mortality", "readmit_30day", "gcs_admission", "race_group",
                "insurance_group", "anchor_year_group", "subject_id", "age", "elixhauser_score"}
        if not need.issubset(h.columns):
            return False
        return len(pd.read_csv(path, usecols=["subject_id"])) == 4281
    except Exception:
        return False

paths = [COHORT_PATH] if COHORT_PATH else (
    glob.glob("/content/*neurosurg_cohort_enriched*.csv") +
    glob.glob(BASE + "**/*neurosurg_cohort_enriched*.csv", recursive=True))
cohort = next((p for p in paths if p and _valid(p)), None)
if cohort is None:
    raise FileNotFoundError("No cohort CSV with 4,281 rows + gcs_admission found. Candidates tried: "
                            f"{paths}. Set COHORT_PATH to the file used for Pub 2.")
df = pd.read_csv(cohort)
R["meta"]["cohort_file"] = os.path.basename(cohort)
log(f"Cohort file: {cohort}  shape={df.shape}")

# ------------------------------ FEATURES ------------------------------------
year_map = {"2008 - 2010": 2009, "2011 - 2013": 2012, "2014 - 2016": 2015,
            "2017 - 2019": 2018, "2020 - 2022": 2021}
df["year_numeric"] = df["anchor_year_group"].map(year_map)
df["split"] = np.where(df["year_numeric"] <= 2018, "train", "test")   # train = 2008-2019
df["male"] = (df["gender"] == "M").astype(int)
df["non_english"] = df["non_english"].fillna(0).astype(int)
df["primary_dx"] = df["neurosurg_category"].str.split(r" \| ").str[0]
df["age_group"] = np.where(df["age"] >= AGE_CUT, f">={AGE_CUT}", f"<{AGE_CUT}")
df["language"] = np.where(df["non_english"] == 1, "Non-English", "English")

race_d = pd.get_dummies(df["race_group"], prefix="race")
race_d = race_d.drop(columns=[c for c in race_d.columns if "White" in c])
ins_d = pd.get_dummies(df["insurance_group"], prefix="ins")
ins_d = ins_d.drop(columns=[c for c in ins_d.columns if "Private" in c])
dx_d = pd.get_dummies(df["primary_dx"], prefix="dx", drop_first=True)
COMORB = ["congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease", "pulmonary_circulation",
          "peripheral_vascular", "hypertension_uncomplicated", "hypertension_complicated", "paralysis",
          "other_neurological", "chronic_pulmonary", "diabetes_uncomplicated", "diabetes_complicated",
          "hypothyroidism", "renal_failure", "liver_disease", "coagulopathy", "obesity", "weight_loss",
          "fluid_electrolyte", "blood_loss_anemia", "deficiency_anemia", "alcohol_abuse", "drug_abuse",
          "depression"]
X_df = pd.concat([df[["age", "elixhauser_score", "gcs_admission", "male", "non_english"]].reset_index(drop=True),
                  df[COMORB].reset_index(drop=True), race_d.reset_index(drop=True),
                  ins_d.reset_index(drop=True), dx_d.reset_index(drop=True)], axis=1)
n_gcs_missing = int(X_df["gcs_admission"].isna().sum())
X_df["gcs_admission"] = X_df["gcs_admission"].fillna(X_df["gcs_admission"].median())
FEATS = X_df.columns.tolist()
X_all = X_df.values.astype(float)
n_ins_missing = int(df["insurance_group"].isna().sum())
split = df["split"].values
subj = df["subject_id"].values
assert len(FEATS) == 43, f"expected 43 features, got {len(FEATS)}"

# ---- reproduction checks against the manuscript -----------------------------
def check(name, got, want, tol=0):
    ok = abs(got - want) <= tol
    R["checks"].append({"name": name, "got": float(got), "expected": float(want), "ok": bool(ok)})
    log(f"  [{'OK ' if ok else 'XX '}] {name}: got {got} expected {want}")
    return ok
log("Reproduction checks:")
check("n admissions", len(df), 4281)
check("n patients", df.subject_id.nunique(), 3993)
check("train n", int((split == "train").sum()), 3066)
check("test n", int((split == "test").sum()), 1215)
check("train deaths", int(df.loc[split == "train", "in_hospital_mortality"].sum()), 366)
check("test deaths", int(df.loc[split == "test", "in_hospital_mortality"].sum()), 122)
check("deaths total", int(df.in_hospital_mortality.sum()), 488)
check("readmissions total", int(df.readmit_30day.sum()), 670)
check("GCS missing", n_gcs_missing, 2)
check("insurance missing", n_ins_missing, 74)
check("readmits among deaths (should be 0)", int(df.loc[df.in_hospital_mortality == 1, "readmit_30day"].sum()), 0)
if not all(c["ok"] for c in R["checks"]):
    log("\n*** WARNING: some reproduction checks failed. This is not the cohort used for the manuscript. ***\n")

# ------------------------------ STATS HELPERS -------------------------------
EPS = 1e-6
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def expit(z): return 1 / (1 + np.exp(-z))
def auc(y, p):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def brier(y, p): return float(np.mean((p - y) ** 2))
def auprc(y, p): return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else np.nan
def conf(y, p, thr):
    yb = p >= thr
    tp = int((yb & (y == 1)).sum()); fp = int((yb & (y == 0)).sum())
    fn = int((~yb & (y == 1)).sum()); tn = int((~yb & (y == 0)).sum())
    return tp, fp, fn, tn
def rates(y, p, thr):
    tp, fp, fn, tn = conf(y, p, thr)
    d = lambda a, b: a / b if b > 0 else np.nan
    return dict(sens=d(tp, tp + fn), spec=d(tn, tn + fp), ppv=d(tp, tp + fp), npv=d(tn, tn + fn),
                flag=d(tp + fp, len(y)))
def _irls(X, y, offset=0.0, iters=50):
    b = np.zeros(X.shape[1]); off = offset
    for _ in range(iters):
        eta = X @ b + off; p = expit(eta); W = np.clip(p * (1 - p), 1e-9, None)
        z = eta - off + (y - p) / W
        XtW = X.T * W
        try: bn = np.linalg.solve(XtW @ X + 1e-9 * np.eye(X.shape[1]), XtW @ z)
        except Exception: return np.full(X.shape[1], np.nan)
        if np.max(np.abs(bn - b)) < 1e-9: b = bn; break
        b = bn
    return b
def calib(y, p):
    """slope (logistic recalibration), CITL (intercept with slope fixed at 1), O:E."""
    if y.sum() == 0 or y.sum() == len(y): return dict(slope=np.nan, citl=np.nan, oe=np.nan, gap=np.nan)
    lp = logit(p)
    b = _irls(np.c_[np.ones_like(lp), lp], y.astype(float))
    c = _irls(np.ones((len(lp), 1)), y.astype(float), offset=lp)
    return dict(slope=float(b[1]), citl=float(c[0]), oe=float(p.mean() / y.mean()), gap=float(p.mean() - y.mean()))
def ci(a, lo=2.5, hi=97.5):
    a = np.asarray([x for x in a if x is not None and not np.isnan(x)])
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if len(a) > 20 else (np.nan, np.nan)
def pack(point, boots):
    lo, hi = ci(boots)
    return {"point": None if point is None or np.isnan(point) else float(point), "lo": lo, "hi": hi}
def cluster_boot_indices(ids, n_boot, seed):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(ids, return_inverse=True)
    rows = [[] for _ in range(len(uniq))]
    for i, k in enumerate(inv): rows[k].append(i)
    rows = [np.array(r) for r in rows]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([rows[k] for k in pick]))
    return out
def smd(a, b):
    sa, sb = np.nanstd(a, ddof=1), np.nanstd(b, ddof=1)
    return float((np.nanmean(a) - np.nanmean(b)) / np.sqrt((sa ** 2 + sb ** 2) / 2)) if (sa + sb) > 0 else np.nan

# ------------------------------ MODEL FITTING -------------------------------
def idx_of(cols): return [FEATS.index(c) for c in cols]
RACE_C = [c for c in FEATS if c.startswith("race_")]; INS_C = [c for c in FEATS if c.startswith("ins_")]
DX_C = [c for c in FEATS if c.startswith("dx_")]
VARIANTS = {   # name: (feature columns, class_weight)
    "lr_unw":    (FEATS, None),
    "lr_blind":  ([c for c in FEATS if c not in RACE_C + INS_C], "balanced"),
    "lr_core":   (["age", "male", "gcs_admission", "non_english"] + DX_C + RACE_C + INS_C, "balanced"),
    "lr_parsim": (["age", "gcs_admission", "elixhauser_score"] + DX_C, "balanced"),
}
def fit_lr(Xtr, ytr, cw="balanced"):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, random_state=42, class_weight=cw, C=1.0).fit(sc.transform(Xtr), ytr)
    return sc, m
def fit_xgb(Xtr, ytr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=42, verbosity=0, n_jobs=1, tree_method="hist").fit(Xtr, ytr)
def oof_recal(Xtr, ytr, kind, seed=SEED):
    """Logistic recalibration (intercept+slope) learned ONLY from out-of-fold training predictions."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); oof = np.zeros(len(ytr))
    for a, b in skf.split(Xtr, ytr):
        if kind == "lr":
            sc, m = fit_lr(Xtr[a], ytr[a]); oof[b] = m.predict_proba(sc.transform(Xtr[b]))[:, 1]
        else:
            m = fit_xgb(Xtr[a], ytr[a]); oof[b] = m.predict_proba(Xtr[b])[:, 1]
    cal = _irls(np.c_[np.ones(len(oof)), logit(oof)], ytr.astype(float))
    return float(cal[0]), float(cal[1])

OUTCOMES = {   # key: (outcome column, mask of admissions in the model population, label)
    "mortality":       ("in_hospital_mortality", np.ones(len(df), bool), "In-hospital mortality"),
    "readmission":     ("readmit_30day", (df.in_hospital_mortality == 0).values, "30-day readmission (survivors)"),
    "readmission_all": ("readmit_30day", np.ones(len(df), bool), "30-day readmission (all admissions, original run)"),
}
M = {}      # fitted objects + test-set arrays per outcome
for key, (col, pop, lab) in OUTCOMES.items():
    tr = (split == "train") & pop; te = (split == "test") & pop
    Xtr, Xte = X_all[tr], X_all[te]; ytr = df.loc[tr, col].values.astype(int); yte = df.loc[te, col].values.astype(int)
    sc, lr = fit_lr(Xtr, ytr); p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
    xgb = fit_xgb(Xtr, ytr); p_xgb = xgb.predict_proba(Xte)[:, 1]
    probs = {"lr": p_lr, "xgb": p_xgb}
    a_l, b_l = oof_recal(Xtr, ytr, "lr"); probs["lr_recal"] = expit(a_l + b_l * logit(p_lr))
    a_x, b_x = oof_recal(Xtr, ytr, "xgb"); probs["xgb_recal"] = expit(a_x + b_x * logit(p_xgb))
    for name, (cols, cw) in VARIANTS.items():
        ix = idx_of(cols); s2, m2 = fit_lr(Xtr[:, ix], ytr, cw)
        probs[name] = m2.predict_proba(s2.transform(Xte[:, ix]))[:, 1]
    M[key] = dict(tr=tr, te=te, ytr=ytr, yte=yte, Xtr=Xtr, Xte=Xte, sc=sc, lr=lr, xgb=xgb, probs=probs,
                  recal={"lr": (a_l, b_l), "xgb": (a_x, b_x)}, dte=df[te].reset_index(drop=True))
    log(f"{key}: train n={len(ytr)} ev={ytr.sum()} | test n={len(yte)} ev={yte.sum()} "
        f"| LR AUROC={auc(yte, p_lr):.4f} XGB AUROC={auc(yte, p_xgb):.4f}")

check("LR mortality test AUROC vs manuscript 0.852", round(auc(M["mortality"]["yte"], M["mortality"]["probs"]["lr"]), 3), 0.852, 0.001)
check("LR readmission(all) test AUROC vs manuscript 0.603", round(auc(M["readmission_all"]["yte"], M["readmission_all"]["probs"]["lr"]), 3), 0.603, 0.001)

# --------------------------- 1. HEADLINE METRICS ----------------------------
MODEL_LABELS = {"lr": "Logistic regression", "xgb": "XGBoost", "lr_recal": "LR, recalibrated", "xgb_recal": "XGBoost, recalibrated",
                "lr_unw": "LR, unweighted", "lr_blind": "LR, race/insurance-blind", "lr_core": "LR, no comorbidity flags",
                "lr_parsim": "LR, parsimonious (age, GCS, Elixhauser, diagnosis)"}
R["model_labels"] = MODEL_LABELS
R["metrics"] = {}; R["boot_idx_n"] = {}
BOOT = {}
for key in OUTCOMES:
    m = M[key]; dte = m["dte"]
    BOOT[key] = cluster_boot_indices(dte["subject_id"].values, N_BOOT, SEED)
    y = m["yte"]; R["metrics"][key] = {}
    for name, p in m["probs"].items():
        pt = {"auroc": auc(y, p), "auprc": auprc(y, p), "brier": brier(y, p)}
        pt.update(rates(y, p, 0.5)); cc = calib(y, p); pt.update(cc)
        bs = {k: [] for k in ["auroc", "auprc", "brier", "sens", "spec", "ppv", "npv", "slope", "citl", "oe"]}
        for idx in BOOT[key]:
            yb, pb = y[idx], p[idx]
            if yb.sum() < 3: continue
            bs["auroc"].append(auc(yb, pb)); bs["brier"].append(brier(yb, pb))
            bs["auprc"].append(auprc(yb, pb)) if name in ("lr", "xgb") else None
            r = rates(yb, pb, 0.5)
            for k in ("sens", "spec", "ppv", "npv"): bs[k].append(r[k])
            c = calib(yb, pb); bs["slope"].append(c["slope"]); bs["citl"].append(c["citl"]); bs["oe"].append(c["oe"])
        R["metrics"][key][name] = {k: pack(pt[k], bs[k]) for k in bs if k in pt}
        R["metrics"][key][name]["n_flagged_at_0.5"] = rates(y, p, 0.5)["flag"]
        R["metrics"][key][name]["mean_pred"] = float(p.mean()); R["metrics"][key][name]["obs"] = float(y.mean())
    R["metrics"][key]["_n"] = int(len(y)); R["metrics"][key]["_events"] = int(y.sum())
    R["metrics"][key]["_train_n"] = int(len(m["ytr"])); R["metrics"][key]["_train_events"] = int(m["ytr"].sum())
    R["metrics"][key]["_recal_params"] = {k: {"intercept": v[0], "slope": v[1]} for k, v in m["recal"].items()}
    # paired differences vs primary LR
    R["metrics"][key]["_delta_auroc_vs_lr"] = {}
    for name in ("xgb", "lr_blind", "lr_core", "lr_parsim", "lr_unw"):
        d0 = auc(y, m["probs"][name]) - auc(y, m["probs"]["lr"])
        ds = [auc(y[i], m["probs"][name][i]) - auc(y[i], m["probs"]["lr"][i]) for i in BOOT[key] if y[i].sum() >= 3]
        R["metrics"][key]["_delta_auroc_vs_lr"][name] = pack(d0, ds)
    log(f"  metrics done: {key}")

# --------------------------- 2. COHORT TABLE (train vs test) ----------------
@safe("cohort_table")
def cohort_table():
    tr, te = df[df.split == "train"], df[df.split == "test"]
    rows = []
    def cont(label, col):
        rows.append({"var": label, "train": f"{tr[col].mean():.1f} ({tr[col].std():.1f})", "test": f"{te[col].mean():.1f} ({te[col].std():.1f})",
                     "smd": smd(tr[col].values, te[col].values)})
    def cat(label, col, val):
        a, b = (tr[col] == val), (te[col] == val)
        rows.append({"var": label, "train": f"{a.sum()} ({a.mean()*100:.1f})", "test": f"{b.sum()} ({b.mean()*100:.1f})",
                     "smd": smd(a.astype(float).values, b.astype(float).values)})
    cont("Age, mean (SD), y", "age"); cat("Female", "gender", "F")
    cont("Admission GCS, mean (SD)", "gcs_admission"); cont("Elixhauser score, mean (SD)", "elixhauser_score")
    cat("Non-English primary language", "language", "Non-English")
    for r_ in ["White", "Black", "Other", "Asian", "Hispanic", "Pacific Islander", "Native American/Alaska Native"]: cat(f"Race: {r_}", "race_group", r_)
    for i_ in ["Medicare", "Private", "Medicaid", "Other"]: cat(f"Insurance: {i_}", "insurance_group", i_)
    for d_ in ["Hemorrhagic Stroke", "TBI", "Spine Surgery", "Brain Tumor", "Craniotomy", "Ischemic Stroke (Surgical)"]: cat(f"Diagnosis: {d_}", "primary_dx", d_)
    rows.append({"var": "In-hospital death", "train": f"{int(tr.in_hospital_mortality.sum())} ({tr.in_hospital_mortality.mean()*100:.1f})",
                 "test": f"{int(te.in_hospital_mortality.sum())} ({te.in_hospital_mortality.mean()*100:.1f})", "smd": smd(tr.in_hospital_mortality.values.astype(float), te.in_hospital_mortality.values.astype(float))})
    trs, tes = tr[tr.in_hospital_mortality == 0], te[te.in_hospital_mortality == 0]
    rows.append({"var": "30-day readmission, among survivors", "train": f"{int(trs.readmit_30day.sum())}/{len(trs)} ({trs.readmit_30day.mean()*100:.1f})",
                 "test": f"{int(tes.readmit_30day.sum())}/{len(tes)} ({tes.readmit_30day.mean()*100:.1f})", "smd": smd(trs.readmit_30day.values.astype(float), tes.readmit_30day.values.astype(float))})
    R["cohort_table"] = {"n_train": int(len(tr)), "n_test": int(len(te)), "patients_train": int(tr.subject_id.nunique()),
                         "patients_test": int(te.subject_id.nunique()),
                         "patients_in_both": int(len(set(tr.subject_id) & set(te.subject_id))), "rows": rows}
    log(f"  patients shared between train and test: {R['cohort_table']['patients_in_both']} (expected 0)")
cohort_table()

# --------------------------- 3. SUBGROUP PERFORMANCE ------------------------
SUBGROUPS = {"race_group": "Race/ethnicity", "insurance_group": "Insurance", "gender": "Sex", "primary_dx": "Primary diagnosis",
             "age_group": "Age", "language": "Language"}
@safe("subgroups")
def subgroup_perf():
    R["subgroups"] = {}
    for key in ("mortality", "readmission"):
        m = M[key]; d = m["dte"]; y = m["yte"]; pl = m["probs"]["lr"]; px = m["probs"]["xgb"]; rows = []
        for sg, sglab in SUBGROUPS.items():
            for g in sorted(d[sg].dropna().unique()):
                msk = (d[sg] == g).values; n = int(msk.sum()); ev = int(y[msk].sum())
                row = {"subgroup": sglab, "group": g, "n": n, "events": ev, "suppressed": ev < MIN_EVENTS}
                if ev >= MIN_EVENTS and (n - ev) > 0:
                    yg, pg = y[msk], pl[msk]; r = rates(yg, pg, 0.5); c = calib(yg, pg)
                    bs_auc, bs_slope = [], []
                    for idx in BOOT[key]:
                        mb = msk[idx]; yb, pb = y[idx][mb], pl[idx][mb]
                        if yb.sum() >= 3 and (len(yb) - yb.sum()) >= 3:
                            bs_auc.append(auc(yb, pb))
                    row.update(lr_auroc=pack(auc(yg, pg), bs_auc), xgb_auroc=float(auc(yg, px[msk])), brier=brier(yg, pg),
                               sens=r["sens"], spec=r["spec"], ppv=r["ppv"], mean_pred=float(pg.mean()), obs=float(yg.mean()),
                               citl=c["citl"], oe=c["oe"], slope=c["slope"] if ev >= 20 else None)
                    # recalibrated-LR calibration
                    c2 = calib(yg, m["probs"]["lr_recal"][msk]); row.update(citl_recal=c2["citl"], oe_recal=c2["oe"])
                rows.append(row)
        R["subgroups"][key] = rows
    log("  subgroup performance done")
subgroup_perf()

# --------------------------- 4. FAIRNESS AUDIT (+ sweep) --------------------
COMPARISONS = [   # (column, group, reference, label)
    ("race_group", "Other", "White", "Other race vs White"),
    ("race_group", "Black", "White", "Black vs White"),
    ("insurance_group", "Medicaid", "Private", "Medicaid vs Private"),
    ("insurance_group", "Medicare", "Private", "Medicare vs Private"),
    ("gender", "F", "M", "Female vs Male"),
    ("age_group", f">={AGE_CUT}", f"<{AGE_CUT}", f"Age >={AGE_CUT} vs <{AGE_CUT}"),
    ("language", "Non-English", "English", "Non-English vs English"),
]
def tpr_fpr(y, p, msk, thr):
    ys, ps = y[msk], p[msk]; r = rates(ys, ps, thr); return r["sens"], 1 - r["spec"] if not np.isnan(r["spec"]) else np.nan
def sweep_rates(y, p, msk, thrs):
    ys, ps = y[msk], p[msk]
    pred = ps[:, None] >= thrs[None, :]
    tp = (pred & (ys[:, None] == 1)).sum(0); fp = (pred & (ys[:, None] == 0)).sum(0)
    P, Nn = (ys == 1).sum(), (ys == 0).sum()
    return (tp / P if P else np.full(len(thrs), np.nan)), (fp / Nn if Nn else np.full(len(thrs), np.nan))

@safe("fairness")
def fairness():
    R["fairness"] = {}; R["sweep"] = {}
    for key in ("mortality", "readmission"):
        m = M[key]; d = m["dte"]; y = m["yte"]; rows = []; sw = {}
        for model in ("lr", "xgb"):
            pm = m["probs"][model]
            for col, g, ref, lab in COMPARISONS:
                mg = (d[col] == g).values; mr = (d[col] == ref).values
                evg, evr = int(y[mg].sum()), int(y[mr].sum())
                row = {"model": model, "comparison": lab, "group": g, "reference": ref, "n": int(mg.sum()), "events": evg,
                       "ref_n": int(mr.sum()), "ref_events": evr, "suppressed": (evg < MIN_EVENTS or evr < MIN_EVENTS)}
                if not row["suppressed"]:
                    tg, fg = tpr_fpr(y, pm, mg, 0.5); tr_, fr_ = tpr_fpr(y, pm, mr, 0.5)
                    calg = (pm[mg].mean() - y[mg].mean()); calr = (pm[mr].mean() - y[mr].mean())
                    eod_b, fpr_b, cal_b = [], [], []
                    for idx in BOOT[key]:
                        yb, pb, mgb, mrb = y[idx], pm[idx], mg[idx], mr[idx]
                        if yb[mgb].sum() < 3 or yb[mrb].sum() < 3: continue
                        a, b_ = tpr_fpr(yb, pb, mgb, 0.5); c_, d_ = tpr_fpr(yb, pb, mrb, 0.5)
                        eod_b.append(a - c_); fpr_b.append(b_ - d_)
                        cal_b.append((pb[mgb].mean() - yb[mgb].mean()) - (pb[mrb].mean() - yb[mrb].mean()))
                    eb = np.array(eod_b)
                    pval = float(min(1.0, 2 * min((eb <= 0).mean(), (eb >= 0).mean()))) if len(eb) else np.nan
                    row.update(tpr=tg, fpr=fg, ref_tpr=tr_, ref_fpr=fr_, eod=pack(tg - tr_, eod_b), fpr_diff=pack(fg - fr_, fpr_b),
                               eq_odds=float(max(abs(tg - tr_), abs(fg - fr_))), calgap_vs_ref=pack(calg - calr, cal_b),
                               abs_gap=float(calg), ref_abs_gap=float(calr), p_boot=pval)
                    # threshold sweep (EOD vs threshold) with cluster-bootstrap band
                    tg_s, fg_s = sweep_rates(y, pm, mg, THRESHOLDS); tr_s, fr_s = sweep_rates(y, pm, mr, THRESHOLDS)
                    E = []; F = []
                    for idx in BOOT[key][:N_BOOT_SW]:
                        yb, pb, mgb, mrb = y[idx], pm[idx], mg[idx], mr[idx]
                        if yb[mgb].sum() < 3 or yb[mrb].sum() < 3: continue
                        a, b_ = sweep_rates(yb, pb, mgb, THRESHOLDS); c_, d_ = sweep_rates(yb, pb, mrb, THRESHOLDS)
                        E.append(a - c_); F.append(b_ - d_)
                    E = np.array(E); F = np.array(F)
                    sw[f"{model}|{lab}"] = {"thr": THRESHOLDS.tolist(), "tpr": tg_s.tolist(), "ref_tpr": tr_s.tolist(),
                                            "fpr": fg_s.tolist(), "ref_fpr": fr_s.tolist(),
                                            "eod": (tg_s - tr_s).tolist(), "eod_lo": np.nanpercentile(E, 2.5, axis=0).tolist(),
                                            "eod_hi": np.nanpercentile(E, 97.5, axis=0).tolist(),
                                            "fprd": (fg_s - fr_s).tolist(), "fprd_lo": np.nanpercentile(F, 2.5, axis=0).tolist(),
                                            "fprd_hi": np.nanpercentile(F, 97.5, axis=0).tolist()}
                rows.append(row)
        # Benjamini-Hochberg across primary (LR, 0.5-threshold) comparisons that were estimable
        est = [r for r in rows if r["model"] == "lr" and not r["suppressed"]]
        ps = np.array([r["p_boot"] for r in est]); order = np.argsort(ps); mm = len(ps)
        adj = np.empty(mm); prev = 1.0
        for rank, i in zip(range(mm, 0, -1), order[::-1]):
            prev = min(prev, ps[i] * mm / rank); adj[i] = prev
        for r, a in zip(est, adj): r["p_fdr"] = float(a)
        R["fairness"][key] = rows; R["sweep"][key] = sw
        log(f"  fairness done: {key}")
fairness()

# ---- threshold sweep: overall operating characteristics + net benefit -------
@safe("sweep_overall")
def sweep_overall():
    R["sweep_overall"] = {}
    for key in ("mortality", "readmission"):
        y = M[key]["yte"]; n = len(y); prev = y.mean(); out = {}
        for model in ("lr", "xgb", "lr_recal"):
            p = M[key]["probs"][model]; rows = []
            for t in THRESHOLDS:
                r = rates(y, p, t); tp, fp, fn, tn = conf(y, p, t)
                nb = tp / n - fp / n * (t / (1 - t)); nb_all = prev - (1 - prev) * (t / (1 - t))
                rows.append(dict(thr=float(t), sens=r["sens"], spec=r["spec"], ppv=r["ppv"], npv=r["npv"], flag=r["flag"],
                                 nb=float(nb), nb_all=float(max(nb_all, -1))))
            out[model] = rows
        R["sweep_overall"][key] = out
        # range of thresholds where the model beats treat-all and treat-none (primary LR, recalibrated LR)
        for model in ("lr", "xgb", "lr_recal"):
            tt = np.arange(0.01, 0.99, 0.01); nbm, nba = [], []
            for t in tt:
                tp, fp, fn, tn = conf(y, M[key]["probs"][model], t); nbm.append(tp / n - fp / n * (t / (1 - t)))
                nba.append(prev - (1 - prev) * (t / (1 - t)))
            nbm, nba = np.array(nbm), np.array(nba); ok = (nbm > nba) & (nbm > 0)
            R["sweep_overall"][key][f"{model}_useful_range"] = ([float(tt[ok].min()), float(tt[ok].max())] if ok.any() else None)
            R["sweep_overall"][key][f"{model}_useful_n"] = int(ok.sum())
    log("  overall sweep done")
sweep_overall()

# --------------------------- 5. AGE-STRATIFIED CHECKS -----------------------
@safe("age")
def age_checks():
    R["age"] = {}
    for key in ("mortality", "readmission"):
        m = M[key]; d = m["dte"]; y = m["yte"]; pl = m["probs"]["lr"]; out = {}
        for grp in (f"<{AGE_CUT}", f">={AGE_CUT}"):
            mg = (d["age_group"] == grp).values; yg, pg = y[mg], pl[mg]
            out[grp] = dict(n=int(mg.sum()), events=int(yg.sum()), prevalence=float(yg.mean()))
            if yg.sum() >= MIN_EVENTS:
                bs = [auc(y[i][mg[i]], pl[i][mg[i]]) for i in BOOT[key] if y[i][mg[i]].sum() >= 3]
                c = calib(yg, pg); cr = calib(yg, m["probs"]["lr_recal"][mg]); r = rates(yg, pg, 0.5)
                out[grp].update(auroc=pack(auc(yg, pg), bs), brier=brier(yg, pg), sens=r["sens"], spec=r["spec"], ppv=r["ppv"],
                                slope=c["slope"] if yg.sum() >= 20 else None, citl=c["citl"], oe=c["oe"], citl_recal=cr["citl"], oe_recal=cr["oe"])
        # within-age-stratum fairness (insurance / race / sex)
        wa = {}
        for grp in (f"<{AGE_CUT}", f">={AGE_CUT}"):
            ag = (d["age_group"] == grp).values
            for col, g, ref, lab in COMPARISONS:
                if col == "age_group": continue
                mg = ((d[col] == g).values & ag); mr = ((d[col] == ref).values & ag)
                evg, evr = int(y[mg].sum()), int(y[mr].sum())
                if evg < MIN_EVENTS or evr < MIN_EVENTS:
                    wa[f"{grp}|{lab}"] = dict(suppressed=True, n=int(mg.sum()), events=evg, ref_events=evr); continue
                tg, _ = tpr_fpr(y, pl, mg, 0.5); tr_, _ = tpr_fpr(y, pl, mr, 0.5); E = []
                for idx in BOOT[key]:
                    yb, pb, a_, b_ = y[idx], pl[idx], mg[idx], mr[idx]
                    if yb[a_].sum() < 3 or yb[b_].sum() < 3: continue
                    x1, _ = tpr_fpr(yb, pb, a_, 0.5); x2, _ = tpr_fpr(yb, pb, b_, 0.5); E.append(x1 - x2)
                wa[f"{grp}|{lab}"] = dict(suppressed=False, n=int(mg.sum()), events=evg, ref_events=evr, tpr=tg, ref_tpr=tr_, eod=pack(tg - tr_, E))
        out["within_age"] = wa
        # Bands (descriptive, exploratory)
        bands = pd.cut(d["age"], [17, 49, 64, 79, 200], labels=["18-49", "50-64", "65-79", ">=80"])
        out["bands"] = {}
        for b in bands.cat.categories:
            mg = (bands == b).values; yg, pg = y[mg], pl[mg]
            row = dict(n=int(mg.sum()), events=int(yg.sum()))
            if yg.sum() >= MIN_EVENTS and (len(yg) - yg.sum()) >= MIN_EVENTS: row.update(auroc=float(auc(yg, pg)), oe=float(pg.mean() / yg.mean()))
            out["bands"][str(b)] = row
        R["age"][key] = out
    log("  age checks done")
age_checks()

# --------------------------- 6. RACE-BLIND VS RACE-AWARE --------------------
@safe("raceblind")
def raceblind():
    R["raceblind"] = {}
    for key in ("mortality", "readmission"):
        m = M[key]; d = m["dte"]; y = m["yte"]; out = {"groups": {}}
        for g in ["White", "Black", "Other"]:
            mg = (d["race_group"] == g).values
            if y[mg].sum() < MIN_EVENTS: continue
            gaps = {}
            for name in ("lr", "lr_blind", "lr_recal"):
                p = m["probs"][name][mg]; gaps[name] = dict(mean_pred=float(p.mean()), obs=float(y[mg].mean()), gap=float(p.mean() - y[mg].mean()), oe=float(p.mean() / y[mg].mean()))
            out["groups"][g] = gaps
        out["delta_auroc"] = R["metrics"][key]["_delta_auroc_vs_lr"]["lr_blind"]
        R["raceblind"][key] = out
    log("  race-blind done")
raceblind()

# --------------------------- 7. INTERSECTIONAL ------------------------------
@safe("intersect")
def intersectional():
    pairs = [("White", "Private"), ("White", "Medicare"), ("White", "Medicaid"), ("Other", "Medicaid"), ("Other", "Private"),
             ("Black", "Medicaid"), ("Black", "Medicare"), ("Hispanic", "Medicaid")]
    R["intersect"] = {}
    for key in ("mortality", "readmission"):
        m = M[key]; d = m["dte"]; y = m["yte"]; p = m["probs"]["lr"]
        ref = ((d.race_group == "White") & (d.insurance_group == "Private")).values; rows = []
        for r_, i_ in pairs:
            mg = ((d.race_group == r_) & (d.insurance_group == i_)).values; ev = int(y[mg].sum()); row = dict(group=f"{r_} x {i_}", n=int(mg.sum()), events=ev)
            if ev < MIN_EVENTS: row["suppressed"] = True
            else:
                tg, _ = tpr_fpr(y, p, mg, 0.5); tr_, _ = tpr_fpr(y, p, ref, 0.5)
                E = []
                for idx in BOOT[key]:
                    yb, pb, a_, b_ = y[idx], p[idx], mg[idx], ref[idx]
                    if yb[a_].sum() < 3 or yb[b_].sum() < 3: continue
                    x1, _ = tpr_fpr(yb, pb, a_, 0.5); x2, _ = tpr_fpr(yb, pb, b_, 0.5); E.append(x1 - x2)
                row.update(suppressed=False, auroc=float(auc(y[mg], p[mg])), eod=pack(tg - tr_, E), calgap=float((p[mg].mean() - y[mg].mean()) - (p[ref].mean() - y[ref].mean())))
            rows.append(row)
        R["intersect"][key] = rows
    log("  intersectional done")
intersectional()

# --------------------------- 8. COEFFICIENTS (GLM + cluster-robust + deployed model) ----------
@safe("coefs")
def coefficients():
    R["coefs"] = {}
    for key in ("mortality", "readmission"):
        m = M[key]; Xtr, ytr = m["Xtr"], m["ytr"]; ids = df.loc[m["tr"], "subject_id"].values
        Xs = m["sc"].transform(Xtr)
        # drop columns that cannot be estimated (no events among exposed, or variance 0)
        keep, notest = [], []
        for j, f in enumerate(FEATS):
            binary = set(np.unique(Xtr[:, j])) <= {0.0, 1.0}
            if np.std(Xtr[:, j]) == 0 or (binary and (ytr[Xtr[:, j] == 1].sum() == 0 or ytr[Xtr[:, j] == 0].sum() == 0 or (Xtr[:, j] == 1).sum() == 0)):
                notest.append(f)
            else: keep.append(j)
        Xk = sm.add_constant(Xs[:, keep])
        g1 = sm.GLM(ytr, Xk, family=sm.families.Binomial()).fit()
        g2 = sm.GLM(ytr, Xk, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": ids})
        rows = []
        for a, j in enumerate(keep, start=1):
            rows.append(dict(feature=FEATS[j], OR=float(np.exp(g1.params[a])), lo=float(np.exp(g1.params[a] - 1.96 * g1.bse[a])), hi=float(np.exp(g1.params[a] + 1.96 * g1.bse[a])),
                             p=float(g1.pvalues[a]), cl_lo=float(np.exp(g2.params[a] - 1.96 * g2.bse[a])), cl_hi=float(np.exp(g2.params[a] + 1.96 * g2.bse[a])), cl_p=float(g2.pvalues[a])))
        for f in notest: rows.append(dict(feature=f, OR=None, note="not estimable (no events or no variation among exposed)"))
        # deployed (L2, class-weighted) model on the raw scale = the full model for reuse (TRIPOD+AI)
        coef_std = m["lr"].coef_[0]; sc = m["sc"]
        raw = coef_std / sc.scale_; icpt = float(m["lr"].intercept_[0] - np.sum(coef_std * sc.mean_ / sc.scale_))
        R["coefs"][key] = dict(glm=rows, n_train=int(len(ytr)), events_train=int(ytr.sum()), cluster_flips=[r["feature"] for r in rows if r.get("OR") and ((r["p"] < .05) != (r["cl_p"] < .05))],
                               deployed={"intercept_raw_scale": icpt, "coef_raw_scale": {f: float(c) for f, c in zip(FEATS, raw)}},
                               recal=R["metrics"][key]["_recal_params"]["lr"])
        pd.DataFrame(rows).to_csv(OUT_DIR + f"pub2_v2_coef_{key}.csv", index=False)
        # Riley-type sample-size criteria, using apparent Cox-Snell R2 (optimistic; confirm with pmsampsize)
        n = len(ytr); ll1 = g1.llf; ll0 = ytr.sum() * np.log(ytr.mean()) + (n - ytr.sum()) * np.log(1 - ytr.mean())
        r2cs = float(1 - np.exp(2 * (ll0 - ll1) / n)); S = 0.9; P = len(keep)
        n_i = float(P / ((S - 1) * np.log(1 - r2cs / S))) if r2cs > 0 else np.nan
        phi = ytr.mean(); n_iii = float((1.96 / 0.05) ** 2 * phi * (1 - phi))
        R["coefs"][key]["sample_size"] = dict(
            params=P, r2cs_apparent=r2cs, n_required_shrinkage=n_i, n_required_prevalence_margin=n_iii, n_train=n, events_per_param=float(ytr.sum() / P))
    log("  coefficients done")
coefficients()

# --------------------------- 9. SENSITIVITY: missing data & repeated admissions ----
@safe("sens_missing")
def sens_missing():
    R["sens"] = {}
    for key in ("mortality", "readmission"):
        m = M[key]; d = m["dte"]; y = m["yte"]; p = m["probs"]["lr"]; out = {}
        # (a) exclude admissions with missing insurance from test evaluation
        ok = d["insurance_group"].notna().values
        out["insurance_complete_case"] = dict(n=int(ok.sum()), auroc=float(auc(y[ok], p[ok])), brier=brier(y[ok], p[ok]))
        # (b) refit with a 'missing insurance' indicator instead of folding missing into the Private reference
        Xa = np.c_[X_all, df["insurance_group"].isna().values.astype(float)]
        tr, te = m["tr"], m["te"]; ytr = m["ytr"]
        sc, lr = fit_lr(Xa[tr], ytr); pp = lr.predict_proba(sc.transform(Xa[te]))[:, 1]
        out["missing_indicator_refit"] = dict(auroc=float(auc(y, pp)), brier=brier(y, pp), d_auroc=float(auc(y, pp) - auc(y, p)))
        # (c) first admission per patient only (needs a time column)
        tcol = next((c for c in ["intime", "admittime", "icu_intime", "admit_time"] if c in df.columns), None)
        if tcol:
            dd = d.copy(); dd["_t"] = pd.to_datetime(dd[tcol], errors="coerce"); first = dd.groupby("subject_id")["_t"].transform("min") == dd["_t"]
            f = first.values; out["first_admission_only"] = dict(n=int(f.sum()), events=int(y[f].sum()), auroc=float(auc(y[f], p[f])), time_col=tcol)
        else:
            out["first_admission_only"] = dict(note="no admission-time column in cohort file; skipped")
        # (d) patient-level vs admission-level bootstrap width (primary AUROC)
        rng = np.random.default_rng(SEED); adm = [auc(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
        out["admission_level_boot_auroc_ci"] = list(ci(adm))
        R["sens"][key] = out
    R["sens"]["missingness"] = dict(gcs_missing=n_gcs_missing, insurance_missing=n_ins_missing,
                                    note="in the original code missing insurance (all-zero dummies) was folded into the Private reference group")
    log("  sensitivity done")
sens_missing()

@safe("missingness")
def missingness():
    rows = []
    for col, labs in (("race_group", sorted(df.race_group.dropna().unique())), ("insurance_group", sorted(df.insurance_group.dropna().unique()))):
        for g in labs:
            s = df[df[col] == g]
            rows.append(dict(variable=col, group=g, n=int(len(s)), gcs_missing=int(s.gcs_admission.isna().sum()), insurance_missing=int(s.insurance_group.isna().sum())))
    s = df[df.insurance_group.isna()]
    rows.append(dict(variable="insurance_group", group="Missing", n=int(len(s)), gcs_missing=int(s.gcs_admission.isna().sum()), insurance_missing=int(len(s))))
    R["missingness"] = dict(overall=dict(n=int(len(df)), gcs_missing=n_gcs_missing, insurance_missing=n_ins_missing), rows=rows)
missingness()

# --------------------------- 10. SHAP (XGBoost) ----------------------------
@safe("shap")
def shap_block():
    import shap
    R["shap"] = {}
    for key in ("mortality", "readmission"):
        m = M[key]; d = m["dte"]; Xte = m["Xte"]
        sv = shap.TreeExplainer(m["xgb"]).shap_values(Xte); S = pd.DataFrame(sv, columns=FEATS)
        top = S.abs().mean().nlargest(10).index.tolist(); races = [r for r in sorted(d.race_group.dropna().unique()) if (d.race_group == r).sum() >= 20]
        tab = {f: {r: float(np.abs(sv[(d.race_group == r).values, FEATS.index(f)]).mean()) for r in races} for f in top}
        overall = S.abs().mean().sort_values(ascending=False)
        R["shap"][key] = dict(top=top, by_race=tab, overall_rank_race_Other=int(list(overall.index).index("race_Other") + 1) if "race_Other" in overall.index else None,
                              race_Other_overall=float(overall.get("race_Other", np.nan)),
                              non_english_by_race={r: float(np.abs(sv[(d.race_group == r).values, FEATS.index("non_english")]).mean()) for r in races})
        plt.figure(figsize=(10, 8)); shap.summary_plot(sv, Xte, feature_names=FEATS, show=False, max_display=20)
        plt.tight_layout(); plt.savefig(OUT_DIR + f"pub2_v2_shap_{key}.png", dpi=300, bbox_inches="tight"); plt.close()
    log("  shap done")
if RUN_SHAP: shap_block()

# --------------------------- 11. FIGURES ----------------------------------
@safe("figures")
def figures():
    from sklearn.metrics import roc_curve, precision_recall_curve
    col = {"lr": "steelblue", "xgb": "darkorange", "lr_recal": "seagreen"}
    keys = [("mortality", "In-hospital mortality"), ("readmission", "30-day readmission (survivors)")]
    # Fig 1 ROC/PR
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for r_, (k, lab) in enumerate(keys):
        y = M[k]["yte"]
        for mdl, nm in (("lr", "Logistic regression"), ("xgb", "XGBoost")):
            fpr, tpr, _ = roc_curve(y, M[k]["probs"][mdl]); ax[r_, 0].plot(fpr, tpr, color=col[mdl], label=f"{nm} ({auc(y, M[k]['probs'][mdl]):.3f})")
            pr, rc, _ = precision_recall_curve(y, M[k]["probs"][mdl]); ax[r_, 1].plot(rc, pr, color=col[mdl], label=nm)
        ax[r_, 0].plot([0, 1], [0, 1], "k--", alpha=.5); ax[r_, 1].axhline(y.mean(), color="k", ls="--", alpha=.5)
        ax[r_, 0].set_title(f"{lab}: ROC"); ax[r_, 1].set_title(f"{lab}: precision-recall"); ax[r_, 0].legend(fontsize=8); ax[r_, 1].legend(fontsize=8)
        ax[r_, 0].set_xlabel("False-positive rate"); ax[r_, 0].set_ylabel("True-positive rate"); ax[r_, 1].set_xlabel("Recall"); ax[r_, 1].set_ylabel("Precision")
    plt.tight_layout(); plt.savefig(OUT_DIR + "pub2_v2_fig1_roc_pr.png", dpi=300); plt.close()
    # Fig 2 DCA
    fig, ax = plt.subplots(1, 2, figsize=(13, 5)); tt = np.linspace(0.01, 0.6, 100)
    for i, (k, lab) in enumerate(keys):
        y = M[k]["yte"]; n = len(y); prev = y.mean()
        for mdl, nm in (("lr", "Logistic regression"), ("xgb", "XGBoost"), ("lr_recal", "LR, recalibrated")):
            nb = [conf(y, M[k]["probs"][mdl], t)[0] / n - conf(y, M[k]["probs"][mdl], t)[1] / n * t / (1 - t) for t in tt]
            ax[i].plot(tt, nb, color=col[mdl], label=nm, lw=2)
        ax[i].plot(tt, prev - (1 - prev) * tt / (1 - tt), "gray", ls="--", label="Treat all"); ax[i].axhline(0, color="k", label="Treat none")
        ax[i].set_ylim(-0.03, prev + 0.01); ax[i].set_xlabel("Threshold probability"); ax[i].set_ylabel("Net benefit"); ax[i].set_title(lab); ax[i].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_DIR + "pub2_v2_fig2_dca.png", dpi=300); plt.close()
    # Fig 3 calibration: overall (LR, XGB, LR recal) + LR by race + by insurance (deciles/5 bins) + lowess
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    for r_, (k, lab) in enumerate(keys):
        y = M[k]["yte"]; d = M[k]["dte"]
        for mdl, nm in (("lr", "LR"), ("xgb", "XGBoost"), ("lr_recal", "LR recalibrated")):
            fp, mp = calibration_curve(y, M[k]["probs"][mdl], n_bins=10, strategy="quantile"); c = R["metrics"][k][mdl]
            ax[r_, 0].plot(mp, fp, "o-", color=col[mdl], label=f"{nm}: slope {c['slope']['point']:.2f}, CITL {c['citl']['point']:+.2f}")
        ax[r_, 0].plot([0, 1], [0, 1], "k--", alpha=.5); ax[r_, 0].legend(fontsize=7); ax[r_, 0].set_title(f"{lab}: overall"); ax[r_, 0].set_xlabel("Mean predicted probability"); ax[r_, 0].set_ylabel("Observed fraction")
        for j, (sg, ttl) in enumerate((("race_group", "By race/ethnicity (LR)"), ("insurance_group", "By insurance (LR)")), start=1):
            for ci_, g in enumerate(sorted(d[sg].dropna().unique())):
                msk = (d[sg] == g).values
                if y[msk].sum() < MIN_EVENTS or msk.sum() < 20: continue
                try:
                    fp, mp = calibration_curve(y[msk], M[k]["probs"]["lr"][msk], n_bins=5, strategy="quantile"); ax[r_, j].plot(mp, fp, "o-", color=plt.cm.tab10.colors[ci_ % 10], label=g, alpha=.85)
                except Exception: pass
            ax[r_, j].plot([0, 1], [0, 1], "k--", alpha=.5); ax[r_, j].legend(fontsize=7); ax[r_, j].set_title(ttl); ax[r_, j].set_xlabel("Mean predicted probability")
    plt.tight_layout(); plt.savefig(OUT_DIR + "pub2_v2_fig3_calibration.png", dpi=300); plt.close()
    # Fig 5 sweep: EOD vs threshold (LR) for selected comparisons
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, (k, lab) in enumerate(keys):
        for ci_, (_, _, _, cl) in enumerate(COMPARISONS):
            s = R["sweep"].get(k, {}).get(f"lr|{cl}")
            if not s: continue
            c = plt.cm.tab10.colors[ci_ % 10]; ax[i].plot(s["thr"], s["eod"], color=c, label=cl, lw=1.8)
            if cl in ("Medicaid vs Private", "Other race vs White", f"Age >={AGE_CUT} vs <{AGE_CUT}"):
                ax[i].fill_between(s["thr"], s["eod_lo"], s["eod_hi"], color=c, alpha=.15)
        ax[i].axhline(0, color="k", lw=.8); ax[i].set_xlabel("Decision threshold"); ax[i].set_ylabel("Equal opportunity difference (TPR difference)"); ax[i].set_title(lab); ax[i].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT_DIR + "pub2_v2_fig5_threshold_sweep.png", dpi=300); plt.close()
    log("  figures saved")
figures()

# --------------------------- 12. VERSIONS, MODEL CARD, SAVE ----------------
import sklearn, xgboost, statsmodels, scipy
R["meta"].update(dict(date=time.strftime("%Y-%m-%d %H:%M"), python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__, statsmodels=statsmodels.__version__,
                      n_boot=N_BOOT, n_boot_sweep=N_BOOT_SW, seed=SEED, thresholds=THRESHOLDS.tolist(), age_cut=AGE_CUT, runtime_s=round(time.time() - T0)))
try:
    import shap as _s; R["meta"]["shap"] = _s.__version__
except Exception: pass
R["meta"]["splits"] = {"train": "anchor-year groups 2008-2010 ... 2017-2019 (i.e., 2008-2019)", "test": "anchor-year group 2020-2022"}

def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
with open(OUT_DIR + "pub2_v2_results.json", "w") as f: json.dump(_clean(R), f, indent=1)

mm = R["metrics"]["mortality"]["lr"]; rr = R["metrics"]["readmission"]["lr"]
card = f"""NEUROS MODEL CARD (v2) - generated {R['meta']['date']}
Source: MIMIC-IV (see manuscript for version); single centre (BIDMC), 2008-2022.
Cohort: 4,281 admissions / 3,993 patients. Temporal split by patient anchor-year group: train 2008-2019 (n=3,066), test 2020-2022 (n=1,215).
Intended use: research and audit only; prediction at ICU admission; NOT for individual triage, treatment limitation or discharge decisions.
Outcomes: in-hospital mortality (all admissions); 30-day readmission among hospital survivors.
Test performance (logistic regression): mortality AUROC {mm['auroc']['point']:.3f} ({mm['auroc']['lo']:.3f}-{mm['auroc']['hi']:.3f}), calibration slope {mm['slope']['point']:.2f}, CITL {mm['citl']['point']:+.2f};
 readmission AUROC {rr['auroc']['point']:.3f} ({rr['auroc']['lo']:.3f}-{rr['auroc']['hi']:.3f}), calibration slope {rr['slope']['point']:.2f}, CITL {rr['citl']['point']:+.2f}.
Class-weighted training inflates predicted probabilities; use the recalibration parameters in pub2_v2_results.json (metrics.<outcome>._recal_params) before interpreting probabilities as absolute risk.
Known limitations: single centre; readmission captured only at BIDMC; Other-race category is heterogeneous; predictors include hospital-coded diagnoses; fairness metrics are threshold-dependent.
Retirement criteria: retrain if overall AUROC falls by >0.05 or any subgroup EOD exceeds 0.10 (groups with >=10 events).
Software: python {R['meta']['python']}, scikit-learn {R['meta']['sklearn']}, xgboost {R['meta']['xgboost']}, statsmodels {R['meta']['statsmodels']}.
"""
open(OUT_DIR + "pub2_model_card_v2.txt", "w").write(card)

# --------------------------- 13. PRINT SUMMARY -----------------------------
log("\n" + "=" * 70 + "\nSUMMARY (copy nothing - the JSON has it all)\n" + "=" * 70)
for k in ("mortality", "readmission", "readmission_all"):
    for mdl in ("lr", "xgb", "lr_recal"):
        c = R["metrics"][k][mdl]
        log(f"{k:16s} {mdl:9s} AUROC {c['auroc']['point']:.3f} ({c['auroc']['lo']:.3f}-{c['auroc']['hi']:.3f}) "
            f"Brier {c['brier']['point']:.3f} slope {c['slope']['point']:.2f} CITL {c['citl']['point']:+.2f} O:E {c['oe']['point']:.2f}")
log("\nChecks failed: " + (", ".join(c["name"] for c in R["checks"] if not c["ok"]) or "none"))
log("Errors: " + ("; ".join(R["errors"]) or "none"))
log(f"\nDone in {round(time.time()-T0)} s. Results: {OUT_DIR}pub2_v2_results.json")
