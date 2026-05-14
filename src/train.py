"""
train.py — Model training, hyperparameter tuning, evaluation, and persistence.

WHY a dedicated training module:
  Separating training logic from the Streamlit app means you can retrain
  from the CLI on a server (cron job, CI/CD) without launching a web UI.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import (
    confusion_matrix, roc_curve, auc, classification_report
)
import xgboost as xgb

from src.utils        import get_logger, save_model, MODEL_DIR, REPORT_DIR, classification_summary
from src.preprocessing import run_preprocessing

logger = get_logger("train")


# ──────────────────────────────────────────────
# MODEL DEFINITIONS  (with sensible defaults)
# ──────────────────────────────────────────────
def build_models() -> dict:
    """
    Instantiate the three models with production-ready default settings.

    WHY these hyperparameters:
      LR  — max_iter=1000: default 100 often fails to converge on this dataset.
      RF  — class_weight='balanced': handles the ~73/27 imbalance automatically.
      XGB — scale_pos_weight: explicit imbalance correction (ratio ≈ 2.7).
            use_label_encoder=False / eval_metric='logloss': suppress warnings.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
            C=1.0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            scale_pos_weight=2.7,   # roughly majority/minority ratio
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


# ──────────────────────────────────────────────
# HYPERPARAMETER GRIDS
# ──────────────────────────────────────────────
PARAM_GRIDS = {
    "Logistic Regression": {
        "C"      : [0.01, 0.1, 1, 10],
        "solver" : ["lbfgs", "liblinear"],
    },
    "Random Forest": {
        "n_estimators"     : [100, 200],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators"    : [100, 200],
        "max_depth"       : [3, 5, 7],
        "learning_rate"   : [0.05, 0.1, 0.2],
    },
}


# ──────────────────────────────────────────────
# TRAINING WITH OPTIONAL HYPERPARAMETER TUNING
# ──────────────────────────────────────────────
def train_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = False,
) -> object:
    """
    Train a single model, optionally with GridSearchCV.

    WHY StratifiedKFold:
      Preserves the class ratio in every fold — mandatory for imbalanced data.
      Regular KFold might put all churners in one fold and produce misleading
      cross-validation scores.

    WHY scoring='f1':
      Accuracy is misleading on imbalanced data. F1 balances precision and
      recall — equally punishes false positives and false negatives.

    Common beginner mistake: running GridSearchCV with cv=5 on a small dataset
    without stratification → some folds may have zero minority-class samples.
    """
    if tune and name in PARAM_GRIDS:
        logger.info("Tuning %s with GridSearchCV …", name)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(
            estimator=model,
            param_grid=PARAM_GRIDS[name],
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)
        logger.info("Best params for %s: %s", name, gs.best_params_)
        logger.info("Best CV F1: %.4f", gs.best_score_)
        return gs.best_estimator_
    else:
        logger.info("Training %s (no tuning) …", name)
        model.fit(X_train, y_train)
        return model


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────
def evaluate_model(
    name: str,
    model,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> dict:
    """
    Compute and log a full set of evaluation metrics.

    HOW predict_proba works:
      Returns a (n_samples, 2) array. Column [1] is the probability that the
      sample belongs to class 1 (churned). We use this for ROC-AUC and the
      Streamlit probability gauge.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = classification_summary(y_test, y_pred, y_prob)
    logger.info("── %s ──", name)
    for k, v in metrics.items():
        logger.info("  %-12s: %.4f", k, v)

    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    return metrics


# ──────────────────────────────────────────────
# VISUALISATIONS
# ──────────────────────────────────────────────
def plot_confusion_matrix(name: str, model, X_test, y_test) -> None:
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    path = os.path.join(REPORT_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    logger.info("Confusion matrix saved → %s", path)


def plot_roc_curves(results: dict, X_test, y_test) -> None:
    """
    Overlay ROC curves for all models on one figure.

    WHY AUC matters more than accuracy:
      AUC measures discrimination ability at *all* thresholds, not just 0.5.
      A model that scores 0.84 AUC gives the business flexibility to tune the
      decision threshold for cost-optimal operations.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (model, _) in results.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    path = os.path.join(REPORT_DIR, "roc_curves.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    logger.info("ROC curves saved → %s", path)


def plot_feature_importance(name: str, model, feature_names: list, top_n: int = 15) -> None:
    """
    Plot feature importances for tree-based models.

    HOW it works:
      RandomForest & XGBoost expose .feature_importances_ (mean decrease in
      impurity / gain). Logistic Regression exposes .coef_, which we convert
      to absolute values (magnitude = importance).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model %s has no importances attribute.", name)
        return

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances — {name}")
    ax.set_xlabel("Importance")
    path = os.path.join(REPORT_DIR, f"fi_{name.replace(' ', '_').lower()}.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    logger.info("Feature importance plot saved → %s", path)


# ──────────────────────────────────────────────
# MASTER TRAINING RUNNER
# ──────────────────────────────────────────────
def run_training(tune: bool = False) -> dict:
    """
    Full training pipeline.

    Returns a summary dict with metrics for every model plus the best model name.
    Also saves:
      • Each trained model   → models/<name>.pkl
      • The scaler           → models/scaler.pkl
      • Feature names        → models/feature_names.json
      • Report plots         → reports/
    """
    # 1. Preprocess
    data = run_preprocessing()
    X_train      = data["X_train"]
    X_test       = data["X_test"]
    y_train      = data["y_train"]
    y_test       = data["y_test"]
    scaler       = data["scaler"]
    feature_names = data["feature_names"]

    # Save scaler and feature names for inference
    save_model(scaler, "scaler.pkl")
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # 2. Train & evaluate all models
    models  = build_models()
    results = {}
    metrics_summary = {}

    for name, base_model in models.items():
        trained = train_model(name, base_model, X_train, y_train, tune=tune)
        metrics = evaluate_model(name, trained, X_test, y_test)

        results[name]         = (trained, metrics)
        metrics_summary[name] = metrics

        # Visualisations
        plot_confusion_matrix(name, trained, X_test, y_test)
        plot_feature_importance(name, trained, feature_names)

        # Save model
        slug = name.replace(" ", "_").lower()
        save_model(trained, f"{slug}_model.pkl")

    # 3. ROC comparison
    plot_roc_curves(results, X_test, y_test)

    # 4. Select best model by ROC-AUC
    best_name = max(metrics_summary, key=lambda n: metrics_summary[n].get("roc_auc", 0))
    logger.info("🏆 Best model: %s (AUC = %.4f)", best_name, metrics_summary[best_name]["roc_auc"])

    # Save best-model pointer
    with open(os.path.join(MODEL_DIR, "best_model.json"), "w") as f:
        json.dump({"best_model": best_name}, f)

    return {
        "metrics" : metrics_summary,
        "best"    : best_name,
    }


if __name__ == "__main__":
    run_training(tune=False)
