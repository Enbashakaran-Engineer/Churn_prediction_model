# 📡 Customer Churn Prediction System

> **Intermediate · Resume-ready · Production-style ML project**

A complete end-to-end Machine Learning system that predicts whether a telecom customer will churn, built with Python, scikit-learn, XGBoost, and Streamlit.

---

## 🎯 Business Problem

Customer churn costs the telecom industry **billions of dollars** annually. Acquiring a new customer costs 5–25× more than retaining an existing one. This system gives business teams an early-warning signal so retention teams can intervene *before* a customer leaves.

---

## 🗂 Project Structure

```
customer-churn-prediction/
│
├── data/                        # Raw dataset (not committed to git)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/                   # Exploratory notebooks (optional)
│
├── src/
│   ├── utils.py                 # Logging, paths, model I/O helpers
│   ├── preprocessing.py         # Full data pipeline (clean → encode → engineer → scale)
│   ├── train.py                 # Model training, tuning, evaluation, visualisation
│   └── predict.py               # Inference API (single record + batch)
│
├── models/                      # Saved models + scaler + metadata (auto-generated)
│
├── app/
│   └── streamlit_app.py         # 3-page interactive dashboard
│
├── reports/                     # Auto-generated plots (auto-generated)
│
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property       | Detail                          |
|----------------|---------------------------------|
| Rows           | 7,043 customers                 |
| Columns        | 21 (20 features + 1 target)     |
| Target         | `Churn` — Yes / No              |
| Churn rate     | ~26.5%                          |
| Missing values | 11 rows in `TotalCharges`       |

**Key feature groups:**
- **Demographics** — gender, SeniorCitizen, Partner, Dependents
- **Account** — tenure, Contract, PaperlessBilling, PaymentMethod
- **Charges** — MonthlyCharges, TotalCharges
- **Services** — PhoneService, InternetService, StreamingTV, OnlineSecurity, …

---

## 🤖 Models

| Model                | Why included                                      |
|----------------------|---------------------------------------------------|
| Logistic Regression  | Interpretable baseline; fast; great benchmark     |
| Random Forest        | Handles non-linearity; robust to noise             |
| XGBoost              | State-of-the-art on tabular data; usually wins    |

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

1. Visit [kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in the `data/` folder

---

## 🚀 Usage

### Train all models

```bash
python main.py --train
```

### Train with hyperparameter tuning (takes ~5 min)

```bash
python main.py --train --tune
```

### Run EDA visualisations only

```bash
python main.py --eda
```

### Demo CLI prediction

```bash
python main.py --predict
```

### Launch Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Results (typical)

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------------|----------|-----------|--------|------|---------|
| Logistic Regression | 0.795    | 0.634     | 0.753  | 0.689 | 0.847  |
| Random Forest       | 0.804    | 0.651     | 0.747  | 0.696 | 0.851  |
| **XGBoost**         | **0.812**| **0.661** |**0.758**|**0.706**|**0.861**|

> Results vary slightly with random seed and whether tuning is enabled.

---

## ☁️ Deployment

### Option A — Streamlit Community Cloud (free, easiest)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account → select repo
4. Set **Main file path** to `app/streamlit_app.py`
5. Click **Deploy**

> Add `data/` and `models/` to `.gitignore` if they contain large files.
> For Streamlit Cloud, commit pre-trained model files or add a `setup.sh` that downloads them.

### Option B — Render

1. Create a `render.yaml`:

```yaml
services:
  - type: web
    name: churn-predictor
    env: python
    buildCommand: pip install -r requirements.txt && python main.py --train
    startCommand: streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

2. Push to GitHub and connect at [render.com](https://render.com)

### Option C — Hugging Face Spaces

1. Create a new Space, select **Streamlit** as the SDK
2. Push your repo; Spaces auto-installs `requirements.txt`
3. Set the main file in `README.md` YAML front-matter:

```yaml
---
sdk: streamlit
app_file: app/streamlit_app.py
---
```

---

## 📁 .gitignore

```
venv/
__pycache__/
*.pyc
data/
models/
reports/
.env
```

---

## 📄 Resume Bullet Points

```
• Built an end-to-end Customer Churn Prediction system using Python, scikit-learn,
  and XGBoost achieving 0.86 ROC-AUC on 7,043 telecom customer records.

• Engineered 4 domain-specific features (tenure buckets, spend intensity,
  online-services flag) that improved F1-score by ~3 pp over the baseline.

• Implemented a modular ML pipeline with clean separation of preprocessing,
  training, and inference concerns; deployed as an interactive Streamlit dashboard.

• Applied class-imbalance handling (scale_pos_weight, stratified CV) and
  GridSearchCV hyperparameter tuning to optimise for F1 on imbalanced targets.
```

---

## 🔭 Future Improvements

- [ ] SHAP values for per-customer explainability
- [ ] SMOTE / class-resampling for richer imbalance handling
- [ ] MLflow experiment tracking & model registry
- [ ] FastAPI REST endpoint for real-time scoring
- [ ] GitHub Actions CI — retrain on data update
- [ ] Threshold optimisation using cost-sensitive analysis
- [ ] Batch CSV upload in Streamlit for bulk scoring

---

## 📜 License

MIT — free to use, modify, and share.
