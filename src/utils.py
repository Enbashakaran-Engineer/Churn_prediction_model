"""
utils.py — Reusable helper functions for the Customer Churn Prediction System.

WHY this file exists:
  Centralising utilities avoids copy-pasting the same logic across scripts and
  makes the whole project easier to maintain, test, and extend.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────
def get_logger(name: str = "churn_project") -> logging.Logger:
    """
    Return a configured logger.

    WHY: Logging beats print() in production code because you can control
    verbosity at runtime, write to files, and grep structured output.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:                          # avoid duplicate handlers on re-import
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for _d in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)


# ──────────────────────────────────────────────
# MODEL SERIALISATION
# ──────────────────────────────────────────────
def save_model(model, filename: str) -> str:
    """
    Persist a trained model to disk with joblib.

    WHY joblib over pickle:
      joblib is optimised for large NumPy arrays (which live inside sklearn
      estimators). It is faster and produces smaller files for those objects.

    Args:
        model    : any sklearn-compatible estimator or pipeline
        filename : basename, e.g. "xgboost_model.pkl"

    Returns:
        Full path to the saved file.
    """
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(filename: str):
    """
    Load a previously saved model from disk.

    Common beginner mistake: forgetting that the Python + library versions
    used to *load* must match those used to *save*. Pin your requirements.txt!
    """
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at {path}. Train first.")
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model


# ──────────────────────────────────────────────
# METRIC HELPERS
# ──────────────────────────────────────────────
def classification_summary(y_true, y_pred, y_prob=None) -> dict:
    """
    Return a dict of the most important classification metrics.

    WHY a dict instead of printing: callers can store results, compare
    models programmatically, and render them in the Streamlit dashboard.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )

    metrics = {
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score" : round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)

    return metrics


# ──────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────
def load_raw_data(filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """
    Load the raw CSV from the data/ directory.

    Common beginner mistake: hard-coding an absolute path that breaks on
    every other machine. Use BASE_DIR-relative paths instead.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "and place the CSV inside the data/ folder."
        )
    df = pd.read_csv(path)
    logger.info("Raw data loaded — shape: %s", df.shape)
    return df


def display_class_balance(y: pd.Series) -> None:
    """Print class distribution and imbalance ratio."""
    counts = y.value_counts()
    ratio  = counts[0] / counts[1]
    logger.info("Class distribution:\n%s", counts.to_string())
    logger.info("Imbalance ratio (majority/minority): %.2f", ratio)
