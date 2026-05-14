"""
predict.py — Inference utilities used by the Streamlit app and CLI.

WHY a separate predict module:
  The Streamlit app should ONLY call high-level predict functions, not
  manipulate model internals directly. This keeps concerns separated and
  makes swapping the model trivial (change one line here, not in the UI).
"""

import json
import os
import numpy as np
import pandas as pd

from src.utils        import load_model, MODEL_DIR, get_logger
from src.preprocessing import preprocess_single_record, BINARY_COLS, MULTI_COLS

logger = get_logger("predict")


# ──────────────────────────────────────────────
# LAZY-LOAD ARTEFACTS
# ──────────────────────────────────────────────
_cache: dict = {}

def _load_artefacts() -> None:
    """
    Load model, scaler, and feature names once and cache in module-level dict.

    WHY lazy loading + caching:
      In Streamlit, the script re-runs on every widget interaction. Without
      caching, we would reload the model from disk dozens of times per session.
      Streamlit's @st.cache_resource is the preferred solution there, but this
      module-level cache works for CLI usage too.
    """
    if _cache:
        return   # already loaded

    # Determine which model to use
    best_path = os.path.join(MODEL_DIR, "best_model.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            best_name = json.load(f)["best_model"]
    else:
        best_name = "XGBoost"   # sensible fallback
        logger.warning("best_model.json not found — defaulting to XGBoost")

    slug = best_name.replace(" ", "_").lower()
    _cache["model"]        = load_model(f"{slug}_model.pkl")
    _cache["scaler"]       = load_model("scaler.pkl")
    _cache["best_name"]    = best_name

    feat_path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(feat_path) as f:
        _cache["feature_names"] = json.load(f)

    logger.info("Artefacts loaded — using model: %s", best_name)


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────
def predict_single(customer_dict: dict) -> dict:
    """
    Predict churn probability for a single customer.

    Args:
        customer_dict: raw feature values exactly as received from the form
                       (strings for categoricals, numbers for numerics).

    Returns:
        {
          "prediction"  : int   (1 = churn, 0 = no churn),
          "probability" : float (churn probability 0-1),
          "model_used"  : str,
        }

    HOW it works:
      1. _load_artefacts() ensures model/scaler are in memory.
      2. preprocess_single_record applies the SAME transformations used during
         training — this is critical to avoid feature mismatch errors.
      3. model.predict_proba returns [[P(no churn), P(churn)]]; we take index 1.
      4. The binary prediction uses threshold 0.5 (default); real deployments
         often tune this threshold based on business cost of FP vs FN.
    """
    _load_artefacts()

    X = preprocess_single_record(
        record       = customer_dict,
        scaler       = _cache["scaler"],
        feature_names= _cache["feature_names"],
    )

    prob       = float(_cache["model"].predict_proba(X)[0, 1])
    prediction = int(prob >= 0.5)

    return {
        "prediction" : prediction,
        "probability": round(prob, 4),
        "model_used" : _cache["best_name"],
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn for a DataFrame of customers (e.g., uploaded CSV in the app).

    Returns the original DataFrame with two extra columns appended:
        churn_prediction  — 0 or 1
        churn_probability — float
    """
    _load_artefacts()
    from src.preprocessing import run_preprocessing  # avoid circular import at module load

    # We need to apply the full preprocessing pipeline to the batch
    # but without refitting — use the saved scaler
    results = []
    for _, row in df.iterrows():
        r = predict_single(row.to_dict())
        results.append(r)

    out = df.copy()
    out["churn_prediction"]  = [r["prediction"]  for r in results]
    out["churn_probability"] = [r["probability"] for r in results]
    return out


def get_model_info() -> dict:
    """Return cached model name and feature list (used by dashboard UI)."""
    _load_artefacts()
    return {
        "model_name"   : _cache["best_name"],
        "feature_names": _cache["feature_names"],
        "n_features"   : len(_cache["feature_names"]),
    }
