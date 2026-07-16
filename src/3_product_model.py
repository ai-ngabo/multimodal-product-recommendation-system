"""Stage 3: Product Recommendation Model.

Predicts `product_category` (5 classes) from the merged customer + basket
features. 

  1. Establish a BASELINE (DummyClassifier = predict majority class).
     No model is worth anything unless it beats this one.
  2. Benchmark 3 candidate families (5-fold CV):
        - Logistic Regression  (linear, interpretable, needs scaling)
        - Random Forest        (non-linear, handles mixed features, robust)
        - XGBoost              (gradient boosting, usually strongest on tabular)
  3. Pick the winner on macro-F1 (fair across imbalanced classes), then TUNE it
     with GridSearchCV
  4. Evaluate on a held-out test set: Accuracy, macro-F1, and Log-Loss.

Outputs: confusion matrix + feature-importance plots, saved model, metrics json.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from config import PROC, PLOTS, MODELS, FEATURES, SEED

sns.set_theme(style="whitegrid")

NUMERIC = ["purchase_amount", "customer_rating", "rating_was_missing",
           "purchase_month", "purchase_dow", "engagement_mean", "engagement_max",
           "interest_mean", "interest_max", "sentiment_mean", "num_platforms",
           "num_social_rows", "has_social_profile"]
CATEG = ["dominant_platform"]
TARGET = "product_category"


def make_preprocessor(scale):
    """Scaling only matters for the linear model; trees are scale-invariant."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEG),
    ])


def main():
    df = pd.read_csv(PROC / "customer_features.csv")
    X = df[NUMERIC + CATEG]
    y_raw = df[TARGET]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1. baseline 
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    base_acc = accuracy_score(y_te, dummy.predict(X_te))
    print(f"[baseline] majority-class accuracy = {base_acc:.3f}")

    # 2. benchmark
    candidates = {
        "LogisticRegression": Pipeline([("prep", make_preprocessor(scale=True)),
                            ("clf", LogisticRegression(max_iter=1000,
                                                       class_weight="balanced",
                                                       random_state=SEED))]),
        "RandomForest": Pipeline([("prep", make_preprocessor(scale=False)),
                                  ("clf", RandomForestClassifier(
                                      n_estimators=300, class_weight="balanced",
                                      random_state=SEED))]),
        "XGBoost": Pipeline([("prep", make_preprocessor(scale=False)),
                             ("clf", XGBClassifier(
                                 n_estimators=300, max_depth=4, learning_rate=0.1,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric="mlogloss", random_state=SEED))]),
    }
    bench = {}
    for name, pipe in candidates.items():
        f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro").mean()
        bench[name] = f1
        print(f"[cv] {name:13s} macro-F1 = {f1:.3f}")

    winner = max(bench, key=bench.get)
    print(f"\n[select] winner by CV macro-F1: {winner}")

    # 3. tune the winner model
    if winner == "RandomForest":
        base = Pipeline([("prep", make_preprocessor(scale=False)),
                         ("clf", RandomForestClassifier(class_weight="balanced",
                                                        random_state=SEED))])
        grid = {"clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 6, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", 0.5]}
    elif winner == "XGBoost":
        base = Pipeline([("prep", make_preprocessor(scale=False)),
                         ("clf", XGBClassifier(eval_metric="mlogloss",
                                               random_state=SEED))])
        grid = {"clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 4, 6],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8, 1.0]}
    else:
        base = Pipeline([("prep", make_preprocessor(scale=True)),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced",
                                                    random_state=SEED))])
        # L2 penalty is the LogisticRegression default in sklearn; we only
        # sweep the inverse-regularisation strength C. Smaller C => stronger
        # regularisation (simpler, less over-fit model), larger C => the model
        # trusts the training data more.
        grid = {"clf__C": [0.01, 0.1, 1, 10]}

    gs = GridSearchCV(base, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    print(f"[tune] best params: {gs.best_params_}")
    print(f"[tune] best CV macro-F1: {gs.best_score_:.3f}")

    # 4. held-out evaluation 
    y_pred = best.predict(X_te)
    y_proba = best.predict_proba(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    ll = log_loss(y_te, y_proba, labels=np.arange(len(le.classes_)))
    print(f"\n[test] accuracy = {acc:.3f}  (baseline {base_acc:.3f})")
    print(f"[test] macro-F1 = {f1m:.3f}")
    print(f"[test] log-loss = {ll:.3f}")
    print("\n[test] classification report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0))

    # confusion matrix plot
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Product model - {winner} (tuned)\nacc={acc:.2f} macroF1={f1m:.2f}")
    fig.tight_layout()
    fig.savefig(PLOTS / "product_model" / "product_confusion_matrix.png", dpi=130)
    plt.close(fig)

    # feature importance 
    try:
        clf = best.named_steps["clf"]
        ohe = best.named_steps["prep"].named_transformers_["cat"]
        feat_names = NUMERIC + list(ohe.get_feature_names_out(CATEG))
        if hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values()
            fig, ax = plt.subplots(figsize=(8, 7))
            imp.tail(15).plot.barh(ax=ax, color="#55A868")
            ax.set_title(f"Top feature importances - {winner}")
            fig.tight_layout()
            fig.savefig(PLOTS / "product_model" /"product_feature_importance.png", dpi=130)
            plt.close(fig)
    except Exception as e:
        print("[warn] feature importance skipped:", e)

    # 5. signal diagnostic 
    # Mutual information quantifies how much each feature tells us about the
    # target. Near-zero MI across the board => the provided data is close to
    # random wrt product_category, which explains the baseline-level scores.
    from sklearn.feature_selection import mutual_info_classif
    Xt = best.named_steps["prep"].transform(X_tr)
    ohe = best.named_steps["prep"].named_transformers_["cat"]
    feat_names = NUMERIC + list(ohe.get_feature_names_out(CATEG))
    mi = mutual_info_classif(Xt, y_tr, random_state=SEED)
    mi_s = pd.Series(mi, index=feat_names).sort_values(ascending=False)
    print("\n[diagnostic] mutual information (feature -> target), top 8:")
    print(mi_s.head(8).round(4).to_string())
    print(f"[diagnostic] mean MI across all features = {mi.mean():.4f} "
          f"(values near 0 => little learnable signal)")
    fig, ax = plt.subplots(figsize=(8, 6))
    mi_s.head(12).iloc[::-1].plot.barh(ax=ax, color="#C44E52")
    ax.set_title("Feature <-> target mutual information (product model)")
    ax.set_xlabel("MI (nats)")
    fig.tight_layout()
    fig.savefig(PLOTS / "product_model" / "product_mutual_information.png", dpi=130)
    plt.close(fig)

    # persist
    joblib.dump({"model": best, "label_encoder": le,
                 "numeric": NUMERIC, "categ": CATEG},
                MODELS / "product_model.joblib")
    metrics = {"baseline_acc": base_acc, "cv_bench": bench, "winner": winner,
               "best_params": gs.best_params_, "test_acc": acc,
               "test_macro_f1": f1m, "test_log_loss": ll,
               "mean_mutual_information": float(mi.mean())}
    (FEATURES / "product_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print("\n[Done] product model + metrics saved")


if __name__ == "__main__":
    main()
