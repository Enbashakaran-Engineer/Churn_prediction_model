"""
streamlit_app.py — Production-quality Streamlit dashboard for the
Customer Churn Prediction System.

Run locally with:
    streamlit run app/streamlit_app.py

Pages:
  📊 Dashboard    — dataset overview & EDA charts
  🔍 Predict      — single-customer prediction form
  📈 Model Report — model comparison & feature importance
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Allow imports from project root when running via `streamlit run app/streamlit_app.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import DATA_DIR, MODEL_DIR, REPORT_DIR, load_raw_data
from src.preprocessing import load_and_clean, encode_target

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS  (minimal, polished, dark-accent)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar header */
    [data-testid="stSidebar"] h1 { color: #4A90E2; }
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 12px 18px;
    }
    /* Section dividers */
    hr { border-color: #2e3250; }
    /* Big prediction banner */
    .churn-banner {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .churn-yes { background: #4a1010; color: #ff6b6b; }
    .churn-no  { background: #0f3020; color: #6bffb8; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CACHED LOADERS
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_raw_df():
    try:
        return load_and_clean()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def get_processed_df():
    try:
        df = load_and_clean()
        return encode_target(df)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all saved model artifacts. Returns None if not trained yet."""
    import joblib
    artefacts = {}
    try:
        # Best model name
        best_path = os.path.join(MODEL_DIR, "best_model.json")
        with open(best_path) as f:
            best_name = json.load(f)["best_model"]
        artefacts["best_name"] = best_name

        # Model
        slug = best_name.replace(" ", "_").lower()
        artefacts["model"]  = joblib.load(os.path.join(MODEL_DIR, f"{slug}_model.pkl"))
        artefacts["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

        # Feature names
        with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
            artefacts["feature_names"] = json.load(f)

        # All model metrics (if exists)
        metrics_path = os.path.join(MODEL_DIR, "metrics_summary.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                artefacts["metrics"] = json.load(f)

        return artefacts
    except Exception:
        return None


# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Churn Predictor")
    st.caption("Telco Customer Churn · ML Dashboard")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Dataset Dashboard", "🔍 Predict Churn", "📈 Model Report"],
        label_visibility="collapsed",
    )
    st.divider()

    df_raw = get_raw_df()
    if df_raw is not None:
        st.metric("Total Customers", f"{len(df_raw):,}")
        churn_rate = (df_raw["Churn"].str.lower() == "yes").mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
        st.metric("Features", str(df_raw.shape[1] - 1))
    else:
        st.warning("⚠ Dataset not found.\nPlace the CSV in `data/`.")


# ══════════════════════════════════════════════
# PAGE 1 — DATASET DASHBOARD
# ══════════════════════════════════════════════
if page == "📊 Dataset Dashboard":
    st.title("📊 Dataset Overview — Telco Customer Churn")
    st.markdown(
        "The **Telco Customer Churn** dataset contains 7,043 customers with "
        "20 features covering demographics, account info, and services subscribed. "
        "The target variable **Churn** indicates whether the customer left within the last month."
    )

    df = get_processed_df()

    if df is None:
        st.error("Dataset not found. Download it from Kaggle and place it in `data/`.")
        st.code(
            "Kaggle URL:\n"
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n\n"
            "Then move the CSV to:  customer-churn-prediction/data/"
        )
        st.stop()

    # ── KPI Row ──────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",          f"{df.shape[0]:,}")
    c2.metric("Columns",       str(df.shape[1]))
    c3.metric("Churned",       f"{df['Churn'].sum():,}")
    c4.metric("Churn Rate",    f"{df['Churn'].mean():.1%}")

    st.divider()

    # ── Class Balance ─────────────────────────
    st.subheader("Target Distribution")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        counts = df["Churn"].value_counts().rename({0: "No Churn", 1: "Churn"})
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=["#4A90E2", "#E24A4A"],
            startangle=90,
        )
        ax.set_title("Churn Split")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        # Churn by Contract Type
        raw = get_raw_df()
        ct = raw.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).reset_index()
        ct.columns = ["Contract", "Churn Rate"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(
    data=ct,
    x="Contract",
    y="Churn Rate",
    hue="Contract",
    palette="Reds_r",
    legend=False,
    ax=ax
)
        ax.set_title("Churn Rate by Contract Type")
        ax.set_ylabel("Churn Rate")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ── Numeric Distributions ─────────────────
    st.subheader("Numeric Features vs Churn")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True,
                     palette={0: "#4A90E2", 1: "#E24A4A"}, ax=ax)
        ax.set_title("Tenure Distribution")
        ax.legend(["No Churn", "Churn"])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette={0: "#4CAF50", 1: "#F44336"},
    legend=False,
    ax=ax
)
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_title("Monthly Charges by Churn")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ── Correlation Heatmap ───────────────────
    st.subheader("Correlation Heatmap (Numeric Features)")
    num_df = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax, linewidths=0.4)
    ax.set_title("Feature Correlations")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Raw Data Preview ──────────────────────
    st.divider()
    st.subheader("Raw Data Sample")
    st.dataframe(get_raw_df().head(50), use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 — PREDICT CHURN
# ══════════════════════════════════════════════
elif page == "🔍 Predict Churn":
    st.title("🔍 Customer Churn Predictor")
    st.markdown("Fill in the customer profile below to get an **instant churn prediction**.")

    artifacts = load_artifacts()
    if artifacts is None:
        st.error("No trained model found. Run `python main.py --train` first.")
        st.stop()

    st.info(f"Active model: **{artifacts['best_name']}**", icon="🤖")

    # ── Input Form ───────────────────────────
    with st.form("customer_form"):
        st.subheader("👤 Demographics")
        col1, col2, col3 = st.columns(3)
        gender         = col1.selectbox("Gender",            ["Male", "Female"])
        senior         = col2.selectbox("Senior Citizen",    ["No", "Yes"])
        partner        = col3.selectbox("Partner",           ["Yes", "No"])
        dependents     = col1.selectbox("Dependents",        ["No", "Yes"])

        st.subheader("📋 Account Info")
        col4, col5, col6 = st.columns(3)
        tenure         = col4.slider("Tenure (months)", 0, 72, 12)
        contract       = col5.selectbox("Contract",      ["Month-to-month", "One year", "Two year"])
        paperless      = col6.selectbox("Paperless Billing", ["Yes", "No"])
        payment        = col4.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_chg    = col5.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_chg      = col6.number_input("Total Charges ($)", 0.0, 9000.0,
                                            monthly_chg * tenure, step=1.0)

        st.subheader("📶 Services")
        col7, col8, col9 = st.columns(3)
        phone_svc      = col7.selectbox("Phone Service",     ["Yes", "No"])
        multi_lines    = col8.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
        internet_svc   = col9.selectbox("Internet Service",  ["Fiber optic", "DSL", "No"])
        online_sec     = col7.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        online_bkp     = col8.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_prot    = col9.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_sup       = col7.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        streaming_tv   = col8.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        streaming_mov  = col9.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True, type="primary")

    if submitted:
        customer = {
            "gender"           : gender,
            "SeniorCitizen"    : 1 if senior == "Yes" else 0,
            "Partner"          : partner,
            "Dependents"       : dependents,
            "tenure"           : tenure,
            "PhoneService"     : phone_svc,
            "MultipleLines"    : multi_lines,
            "InternetService"  : internet_svc,
            "OnlineSecurity"   : online_sec,
            "OnlineBackup"     : online_bkp,
            "DeviceProtection" : device_prot,
            "TechSupport"      : tech_sup,
            "StreamingTV"      : streaming_tv,
            "StreamingMovies"  : streaming_mov,
            "Contract"         : contract,
            "PaperlessBilling" : paperless,
            "PaymentMethod"    : payment,
            "MonthlyCharges"   : monthly_chg,
            "TotalCharges"     : total_chg,
        }

        from src.predict import predict_single
        result = predict_single(customer)
        prob   = result["probability"]

        # ── Prediction Banner ─────────────────
        if result["prediction"] == 1:
            st.markdown(
                f'<div class="churn-banner churn-yes">⚠️ HIGH CHURN RISK &nbsp;·&nbsp; {prob:.1%} probability</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="churn-banner churn-no">✅ LIKELY TO STAY &nbsp;·&nbsp; {prob:.1%} churn probability</div>',
                unsafe_allow_html=True,
            )

        # ── Probability Gauge ─────────────────
        st.subheader("Churn Probability")
        col_g1, col_g2 = st.columns([1, 2])
        with col_g1:
            st.metric("Probability", f"{prob:.1%}")
            st.metric("Prediction",  "Churn" if result["prediction"] else "No Churn")
            st.metric("Model",       result["model_used"])

        with col_g2:
            fig, ax = plt.subplots(figsize=(5, 1.2))
            ax.barh([""], [prob], color="#E24A4A" if prob > 0.5 else "#4A90E2", height=0.5)
            ax.barh([""], [1 - prob], left=[prob], color="#2e3250", height=0.5)
            ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.set_title("Churn Probability Gauge")
            ax.axvline(0.5, color="white", linestyle="--", linewidth=1.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Business Interpretation ───────────
        st.divider()
        st.subheader("💡 Business Interpretation")
        if prob > 0.7:
            st.error(
                "🔴 **Critical risk.** Immediate intervention recommended: "
                "offer a loyalty discount or contract upgrade."
            )
        elif prob > 0.5:
            st.warning(
                "🟠 **Moderate risk.** Consider a proactive outreach campaign "
                "and review service quality issues."
            )
        else:
            st.success("🟢 **Low risk.** Customer is likely satisfied. Monitor normally.")


# ══════════════════════════════════════════════
# PAGE 3 — MODEL REPORT
# ══════════════════════════════════════════════
elif page == "📈 Model Report":
    st.title("📈 Model Performance Report")

    artifacts = load_artifacts()
    if artifacts is None:
        st.warning("No trained models found. Run `python main.py --train` first.")
        st.stop()

    # ── Metrics Comparison Table ──────────────
    metrics_path = os.path.join(MODEL_DIR, "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        st.subheader("Model Comparison")
        rows = []
        for model_name, m in metrics.items():
            rows.append({
                "Model"    : model_name,
                "Accuracy" : f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall"   : f"{m['recall']:.4f}",
                "F1 Score" : f"{m['f1_score']:.4f}",
                "ROC-AUC"  : f"{m.get('roc_auc', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        # Bar chart comparison
        df_metrics = pd.DataFrame(rows).set_index("Model").astype(float)
        fig, ax = plt.subplots(figsize=(9, 4))
        df_metrics.plot(kind="bar", ax=ax, colormap="tab10", rot=15)
        ax.set_title("Model Metrics Comparison")
        ax.set_ylabel("Score"); ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Detailed metrics not available. Ensure `models/metrics_summary.json` exists.")

    st.divider()

    # ── Saved Report Images ───────────────────
    st.subheader("Saved Report Visualisations")
    report_images = {
        "ROC Curves"       : os.path.join(REPORT_DIR, "roc_curves.png"),
        "Correlation Map"  : os.path.join(REPORT_DIR, "correlation_heatmap.png"),
        "Tenure vs Churn"  : os.path.join(REPORT_DIR, "tenure_by_churn.png"),
        "Monthly Charges"  : os.path.join(REPORT_DIR, "monthly_charges_by_churn.png"),
    }

    available = {k: v for k, v in report_images.items() if os.path.exists(v)}
    if available:
        cols = st.columns(2)
        for i, (title, path) in enumerate(available.items()):
            cols[i % 2].image(path, caption=title, use_column_width=True)
    else:
        st.info("No report images found yet. Run `python main.py --train --eda` to generate them.")

    # ── Feature Importance per Model ─────────
    st.divider()
    st.subheader("Feature Importance Plots")
    fi_images = {
        "Logistic Regression" : os.path.join(REPORT_DIR, "fi_logistic_regression.png"),
        "Random Forest"       : os.path.join(REPORT_DIR, "fi_random_forest.png"),
        "XGBoost"             : os.path.join(REPORT_DIR, "fi_xgboost.png"),
    }
    for title, path in fi_images.items():
        if os.path.exists(path):
            st.image(path, caption=f"Feature Importance — {title}", use_column_width=True)

    # ── Confusion Matrices ────────────────────
    st.divider()
    st.subheader("Confusion Matrices")
    cm_images = {
        "Logistic Regression" : os.path.join(REPORT_DIR, "cm_logistic_regression.png"),
        "Random Forest"       : os.path.join(REPORT_DIR, "cm_random_forest.png"),
        "XGBoost"             : os.path.join(REPORT_DIR, "cm_xgboost.png"),
    }
    cm_cols = st.columns(3)
    for i, (title, path) in enumerate(cm_images.items()):
        if os.path.exists(path):
            cm_cols[i].image(path, caption=title, use_column_width=True)
