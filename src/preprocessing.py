"""
preprocessing.py — End-to-end data cleaning, encoding, and feature engineering.

WHY a dedicated module:
  Keeping all data-transformation logic in one place means the *same*
  transformations are applied during training AND when the Streamlit app
  processes a live customer record — preventing training/serving skew.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger, load_raw_data

logger = get_logger("preprocessing")


# ──────────────────────────────────────────────
# STEP 1 — LOAD & INITIAL CLEAN
# ──────────────────────────────────────────────
def load_and_clean(
    filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
) -> pd.DataFrame:
    """
    Load raw CSV and perform initial cleaning.

    Steps:
      1. Drop customerID
      2. Convert TotalCharges to numeric
      3. Fill missing TotalCharges
      4. Strip whitespace from categorical columns
    """

    df = load_raw_data(filename)

    # Drop identifier column
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    missing_tc = df["TotalCharges"].isna().sum()

    logger.info(
        "TotalCharges missing values found: %d",
        missing_tc
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Strip whitespace safely
    str_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    logger.info("After initial clean — shape: %s", df.shape)

    return df


# ──────────────────────────────────────────────
# STEP 2 — ENCODE TARGET
# ──────────────────────────────────────────────
def encode_target(
    df: pd.DataFrame,
    target: str = "Churn"
) -> pd.DataFrame:
    """
    Encode target column:
        Yes -> 1
        No  -> 0
    """

    df = df.copy()

    df[target] = (
        df[target]
        .map({"Yes": 1, "No": 0})
        .astype(int)
    )

    logger.info(
        "Target encoded — churn rate: %.2f%%",
        df[target].mean() * 100
    )

    return df


# ──────────────────────────────────────────────
# STEP 3 — ENCODE CATEGORICAL FEATURES
# ──────────────────────────────────────────────
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

MULTI_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features.

    Binary:
        Yes/No -> 1/0
        Male/Female -> 1/0

    Multi-class:
        One-hot encoding
    """

    df = df.copy()

    # Binary encoding
    binary_mapping = {
        "gender": {
            "Male": 1,
            "Female": 0
        },
        "Partner": {
            "Yes": 1,
            "No": 0
        },
        "Dependents": {
            "Yes": 1,
            "No": 0
        },
        "PhoneService": {
            "Yes": 1,
            "No": 0
        },
        "PaperlessBilling": {
            "Yes": 1,
            "No": 0
        },
    }

    for col, mapping in binary_mapping.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(int)

    # One-hot encoding
    cols_present = [c for c in MULTI_COLS if c in df.columns]

    df = pd.get_dummies(
        df,
        columns=cols_present,
        drop_first=True,
        dtype=int
    )

    # Safety check
    object_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if object_cols:
        logger.warning(
            "Remaining object columns: %s",
            object_cols
        )

    logger.info(
        "After encoding — shape: %s",
        df.shape
    )

    return df


# ──────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features.
    """

    df = df.copy()

    # ──────────────────────────────────────────
    # Tenure groups
    # ──────────────────────────────────────────
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = [1, 2, 3, 4, 5]

    df["tenure_group"] = (
        pd.cut(
            df["tenure"],
            bins=bins,
            labels=labels,
            right=True
        )
        .astype("float")
        .fillna(0)
        .astype(int)
    )

    # ──────────────────────────────────────────
    # Charges per month
    # ──────────────────────────────────────────
    df["charges_per_month"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # ──────────────────────────────────────────
    # Online services flag
    # ──────────────────────────────────────────
    online_service_cols = [
        "OnlineSecurity_Yes",
        "OnlineBackup_Yes",
        "DeviceProtection_Yes",
        "TechSupport_Yes",
    ]

    present_cols = [
        c for c in online_service_cols
        if c in df.columns
    ]

    if present_cols:
        df["has_online_services"] = (
            df[present_cols]
            .any(axis=1)
            .astype(int)
        )
    else:
        df["has_online_services"] = 0

    logger.info(
        "Feature engineering complete — shape: %s",
        df.shape
    )

    return df


# ──────────────────────────────────────────────
# STEP 5 — SCALE FEATURES
# ──────────────────────────────────────────────
NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_month",
]


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numeric columns using StandardScaler.
    """

    scaler = StandardScaler()

    cols_present = [
        c for c in NUMERIC_COLS
        if c in X_train.columns
    ]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[cols_present] = scaler.fit_transform(
        X_train[cols_present]
    )

    X_test[cols_present] = scaler.transform(
        X_test[cols_present]
    )

    logger.info(
        "Scaling applied to: %s",
        cols_present
    )

    return X_train, X_test, scaler


# ──────────────────────────────────────────────
# MASTER PREPROCESSING PIPELINE
# ──────────────────────────────────────────────
def run_preprocessing(
    filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    Run full preprocessing pipeline.
    """

    df = load_and_clean(filename)

    df = encode_target(df)

    df = encode_categoricals(df)

    df = engineer_features(df)

    # Separate features / target
    y = df["Churn"]

    X = df.drop(columns=["Churn"])

    feature_names = list(X.columns)

    # Final object column safety check
    object_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if object_cols:
        logger.warning(
            "Object columns remaining: %s",
            object_cols
        )
    else:
        logger.info("No object columns remaining")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(
        "Split — train: %d rows | test: %d rows | churn in test: %.2f%%",
        len(X_train),
        len(X_test),
        y_test.mean() * 100,
    )

    # Scale features
    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
    }


# ──────────────────────────────────────────────
# SINGLE RECORD PREPROCESSOR
# (Used in Streamlit inference)
# ──────────────────────────────────────────────
def preprocess_single_record(
    record: dict,
    scaler: StandardScaler,
    feature_names: list
) -> pd.DataFrame:
    """
    Transform one customer record into
    model-ready format.
    """

    row = pd.DataFrame([record])

    # Binary encoding
    binary_mapping = {
        "gender": {"Male": 1, "Female": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
    }

    for col, mapping in binary_mapping.items():
        if col in row.columns:
            row[col] = row[col].map(mapping).astype(int)

    # One-hot encoding
    cols_present = [
        c for c in MULTI_COLS
        if c in row.columns
    ]

    row = pd.get_dummies(
        row,
        columns=cols_present,
        drop_first=True,
        dtype=int
    )

    # Feature engineering
    row = engineer_features(row)

    # Remove target if present
    row.drop(
        columns=["Churn"],
        inplace=True,
        errors="ignore"
    )

    # Align schema with training data
    row = row.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Scale numeric columns
    cols_present = [
        c for c in NUMERIC_COLS
        if c in row.columns
    ]

    row[cols_present] = scaler.transform(
        row[cols_present]
    )

    return row