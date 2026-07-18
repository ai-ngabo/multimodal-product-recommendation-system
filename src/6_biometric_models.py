"""Stage 6: Biometric identity models.

Trains and evaluates the two identity models on REAL member captures:
  * Facial Recognition      - which member, from image features
  * Voiceprint Verification - which member, from audio features


Model choice - RandomForest: handles the modest, high-dimensional and
correlated feature vectors here without feature scaling, resists overfitting on
few samples through bagging, and exposes class probabilities used both for
Log-Loss and for the simulation's rejection threshold. We report Accuracy,
Macro-F1 and Log-Loss (mean across folds), a pooled confusion matrix over the
out-of-fold predictions, and an honestly-derived rejection threshold.
"""
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             classification_report, confusion_matrix)

from config import FEATURES, PLOTS, MODELS

sns.set_theme(style="whitegrid")

META_COLS = ["member", "expression", "variant", "source",   # image meta
             "phrase", "sample_id", "augmentation"] # audio meta (any that exist)
RF_KW = dict(n_estimators=400, max_depth=None, min_samples_leaf=1,
             random_state=42, n_jobs=-1)

metrics = {}


def _feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def evaluate(df, group_col, tag, title):
    """Leave-one-group-out CV; returns per-fold scores and pooled OOF predictions."""
    feat_cols = _feature_cols(df)
    le = LabelEncoder()
    y = le.fit_transform(df["member"])
    groups = df[group_col].values
    X = df[feat_cols].values
    n_classes = len(le.classes_)
    labels = list(range(n_classes))

    logo = LeaveOneGroupOut()
    accs, f1s, lls = [], [], []
    oof_true, oof_pred = [], []

    for tr, te in logo.split(X, y, groups):
        held = df.iloc[te][group_col].iloc[0]
        clf = RandomForestClassifier(**RF_KW).fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        proba = clf.predict_proba(X[te])

        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        lls.append(log_loss(y[te], proba, labels=labels))
        oof_true.extend(y[te]); oof_pred.extend(pred)
        print(f"    fold held-out {group_col}='{held}': "
              f"acc={accs[-1]:.2f} f1={f1s[-1]:.2f} n_test={len(te)}")

    acc_m, acc_s = float(np.mean(accs)), float(np.std(accs))
    f1_m = float(np.mean(f1s))
    ll_m = float(np.mean(lls))

    print(f"\n=== {title} ===")
    print(f"  folds: {len(accs)}  (leave-one-{group_col}-out)")
    print(f"  Accuracy {acc_m:.3f} +/- {acc_s:.3f} | "
          f"Macro-F1 {f1_m:.3f} | Log-Loss {ll_m:.3f}")
    print(classification_report(
        oof_true, oof_pred, labels=labels,
        target_names=[c.split('_')[0] for c in le.classes_], zero_division=0))

    # pooled out-of-fold confusion matrix - every capture tested exactly once
    cm = confusion_matrix(oof_true, oof_pred, labels=labels)
    names = [c.split('_')[0] for c in le.classes_]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=names, yticklabels=names)
    plt.title(f"{title}\nmean acc={acc_m:.2f} +/- {acc_s:.2f} (leave-one-out)")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(PLOTS / "biometric_model" / f"{tag}_confusion_matrix.png", dpi=130)
    plt.close()

    return le, feat_cols, dict(
        accuracy_mean=acc_m, accuracy_std=acc_s, macro_f1=f1_m, log_loss=ll_m,
        n_folds=len(accs), n_samples=int(len(df)), n_features=len(feat_cols))


def fit_deployment(df, feat_cols, le, group_col):
    """Refit on all data for deployment, and derive an HONEST rejection threshold.
    """
    X = df[feat_cols].values
    y = le.transform(df["member"])
    groups = df[group_col].values

    oof_conf = np.zeros(len(df))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        clf = RandomForestClassifier(**RF_KW).fit(X[tr], y[tr])
        oof_conf[te] = clf.predict_proba(X[te]).max(axis=1)
    thr = float(np.percentile(oof_conf, 5))

    clf_full = RandomForestClassifier(**RF_KW).fit(X, y)
    return clf_full, thr


def run_modality(csv_name, group_col, tag, title):
    df = pd.read_csv(FEATURES / csv_name)
    if group_col not in df.columns:
        raise KeyError(f"{csv_name} has no '{group_col}' column; got {list(df.columns)[:8]}")
    print(f"\n>>> {title}  ({len(df)} rows, "
          f"{df[group_col].nunique()} {group_col} groups)")

    le, feat_cols, m = evaluate(df, group_col, tag, title)
    clf_full, thr = fit_deployment(df, feat_cols, le, group_col)
    m["reject_threshold"] = thr

    joblib.dump({"model": clf_full, "label_encoder": le,
                 "feat_cols": feat_cols, "reject_threshold": thr,
                 "meta_cols": META_COLS},
                MODELS / f"{tag}_model.joblib")
    metrics[tag] = m
    print(f"  saved {tag}_model.joblib  (reject_threshold={thr:.3f})")


def main():
    run_modality("image_features.csv", "expression",
                 "face", "Facial Recognition (leave-one-expression-out)")
    run_modality("audio_features.csv", "phrase",
                 "voice", "Voiceprint Verification (leave-one-phrase-out)")

    out = MODELS.parent / "biometric_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\n[Done] metrics -> {out}")
    print("[Done] face + voice models, confusion matrices, thresholds saved.")


if __name__ == "__main__":
    main()