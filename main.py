"""
main.py — CLI entry point for the Customer Churn Prediction System.

Usage:
    python main.py --train            # train all models (no hyperparameter tuning)
    python main.py --train --tune     # train with GridSearchCV tuning
    python main.py --predict          # run a demo single-customer prediction
    python main.py --eda              # generate EDA plots only
"""

import argparse
import sys


def run_eda():
    """Quick EDA visualisation run (saves plots to reports/)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.utils         import load_raw_data, REPORT_DIR, get_logger
    from src.preprocessing import load_and_clean, encode_target

    logger = get_logger("eda")
    df_raw = load_and_clean()
    df     = encode_target(df_raw)

    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Churn distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Churn"].value_counts().rename({1: "Churn", 0: "No Churn"}).plot(
        kind="bar", ax=ax, color=["steelblue", "tomato"], rot=0
    )
    ax.set_title("Overall Churn Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "churn_distribution.png"), dpi=120)
    plt.close(fig)
    logger.info("Churn distribution plot saved.")

    # 2. Numeric correlation heatmap
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)
    logger.info("Correlation heatmap saved.")

    # 3. Tenure vs Churn
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=ax, palette={0: "steelblue", 1: "tomato"})
    ax.set_title("Tenure Distribution by Churn Status")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "tenure_by_churn.png"), dpi=120)
    plt.close(fig)
    logger.info("Tenure plot saved.")

    # 4. Monthly charges box plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    legend=False,
    ax=ax,
    palette={0: "steelblue", 1: "tomato"}
)
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_title("Monthly Charges by Churn Status")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "monthly_charges_by_churn.png"), dpi=120)
    plt.close(fig)
    logger.info("Monthly charges plot saved.")

    logger.info("EDA complete — check the reports/ directory.")


def run_demo_prediction():
    """Demonstrate single-customer prediction via CLI."""
    from src.predict import predict_single

    demo_customer = {
        "gender"           : "Male",
        "SeniorCitizen"    : 0,
        "Partner"          : "No",
        "Dependents"       : "No",
        "tenure"           : 2,
        "PhoneService"     : "Yes",
        "MultipleLines"    : "No",
        "InternetService"  : "Fiber optic",
        "OnlineSecurity"   : "No",
        "OnlineBackup"     : "No",
        "DeviceProtection" : "No",
        "TechSupport"      : "No",
        "StreamingTV"      : "Yes",
        "StreamingMovies"  : "Yes",
        "Contract"         : "Month-to-month",
        "PaperlessBilling" : "Yes",
        "PaymentMethod"    : "Electronic check",
        "MonthlyCharges"   : 85.50,
        "TotalCharges"     : 171.0,
    }

    result = predict_single(demo_customer)
    label  = "⚠  CHURN RISK" if result["prediction"] == 1 else "✅ LIKELY TO STAY"
    print("\n" + "=" * 50)
    print(f"  {label}")
    print(f"  Churn Probability : {result['probability']:.1%}")
    print(f"  Model Used        : {result['model_used']}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction — CLI")
    parser.add_argument("--train",   action="store_true", help="Train all models")
    parser.add_argument("--tune",    action="store_true", help="Enable GridSearchCV tuning (slow)")
    parser.add_argument("--predict", action="store_true", help="Run demo prediction")
    parser.add_argument("--eda",     action="store_true", help="Generate EDA plots")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if args.eda:
        run_eda()

    if args.train:
        from src.train import run_training
        summary = run_training(tune=args.tune)
        print("\nModel Performance Summary")
        print("-" * 55)
        for model, metrics in summary["metrics"].items():
            flag = " ← BEST" if model == summary["best"] else ""
            print(f"  {model:<25} AUC={metrics.get('roc_auc', 0):.4f}  F1={metrics['f1_score']:.4f}{flag}")
        print("-" * 55)

    if args.predict:
        run_demo_prediction()


if __name__ == "__main__":
    main()
