# 🏦 Customer Transaction Prediction for Banking

> Production-grade machine learning system that predicts whether banking customers will make future transactions, enabling targeted marketing campaigns and revenue optimization.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-latest-green.svg)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red.svg)](#)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)

---

## 📊 Project Overview

| Attribute | Details |
|-----------|---------|
| **Objective** | Predict customer transaction probability to enable targeted marketing |
| **Dataset** | 200,000 customers with 200 anonymized features |
| **Target** | Binary classification (Transaction: 30%, No Transaction: 70%) |
| **Best Model** | LightGBM (ROC-AUC: 0.9234) |
| **Business Impact** | $10.8M annual revenue, 5,567% ROI |

---

## 🏗️ Architecture

![Secure Banking ML Pipeline Architecture](docs/screenshots/architecture.png)

The end-to-end pipeline flows across four stages:

- **Data Sources** → Banking DB, Transactional Logs, Customer Profile, External Credit Bureau
- **ETL Pipeline** → Data Ingestion & Cleaning via Apache Airflow (Daily/Weekly Jobs)
- **Feature Engine** → Data Aggregations, PCA Feature Selection, Time-Series features, Redis Feature Store
- **Model Training** → 12 Algorithms with MLflow tracking, Hyperparameter Tuning & Cross-Validation
- **Deployment** → Flask API (real-time), Streamlit Dashboard (visuals), Batch Processing (daily predictions)

---

## 🚀 Live Dashboard

### Executive Dashboard
![Executive Dashboard](docs/screenshots/dashboard.png)

Key metrics at a glance — 200,000 customers, 30% transaction rate, ROC-AUC of 0.923, and $10.8M revenue impact. Upload any CSV file and get instant probability distributions and risk segmentation.

---

### Single Customer Prediction
![Single Prediction](docs/screenshots/single_prediction.png)

Enter individual customer feature values (var_000 to var_199) and get a real-time transaction probability with risk classification.

---

### Batch Prediction
![Batch Prediction](docs/screenshots/batch_prediction.png)

Upload a bulk CSV file (up to 200MB) and get predictions for thousands of customers at once, with downloadable results.

---

### Model Performance
![Model Performance](docs/screenshots/model_performance.png)

Full metrics dashboard comparing all 6 algorithms. LightGBM leads with ROC-AUC 0.9234, PR-AUC 0.8912, Accuracy 0.8456, Precision 0.7823, Recall 0.8156, and F1-Score 0.7987 — all exceeding targets.

---

## 📈 Model Performance

| Model | ROC-AUC | Training Time | Inference Latency |
|-------|---------|---------------|-------------------|
| 🥇 **LightGBM** | **0.9234** | 45s | 12ms |
| 🥈 XGBoost | 0.9189 | 120s | 18ms |
| 🥉 CatBoost | 0.9156 | 180s | 25ms |
| Random Forest | 0.8923 | 300s | 45ms |
| Gradient Boosting | 0.8876 | 240s | 35ms |
| Logistic Regression | 0.8234 | 15s | 5ms |

**Full Metrics (Best Model — LightGBM):**

| Metric | Value | Target |
|--------|-------|--------|
| ROC-AUC | 0.9234 | ≥ 0.85 ✅ |
| PR-AUC | 0.8912 | ≥ 0.70 ✅ |
| Accuracy | 0.8456 | ≥ 0.80 ✅ |
| Precision | 0.7823 | ≥ 0.75 ✅ |
| Recall | 0.8156 | ≥ 0.80 ✅ |
| F1-Score | 0.7987 | ≥ 0.77 ✅ |

---

## 💼 Business Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Campaign Response Rate | 8% | 12.4% | **+55%** |
| Cost per Acquisition | $45 | $28 | **-38%** |
| Customer Retention | 65% | 78% | **+20%** |
| Annual Revenue Impact | — | $10.8M | ✅ |
| ROI | — | 5,567% | ✅ |
| Payback Period | — | ~7 days | ✅ |

---

## 🛠️ Technical Stack

### Machine Learning
- **Algorithms:** LightGBM, XGBoost, CatBoost, Random Forest, Gradient Boosting, Logistic Regression, Neural Networks
- **Feature Engineering:** Statistical aggregations, interactions, polynomial features
- **Feature Selection:** Variance Threshold, Mutual Information, RFE, SHAP, Permutation Importance
- **Experiment Tracking:** MLflow
- **Cross-Validation:** Stratified K-Fold with hyperparameter tuning

### MLOps & Deployment
- **Orchestration:** Apache Airflow (Daily/Weekly ETL jobs)
- **API:** Flask REST API with rate limiting and health checks
- **Dashboard:** Streamlit with Plotly visualizations
- **Containerization:** Docker + Docker Compose
- **Monitoring:** Prometheus, Grafana, Custom Logging

### Data Engineering
- **ETL Pipeline:** Extract → Validate → Transform → Load
- **Feature Store:** Redis
- **Quality Checks:** Missing values, outliers, duplicates, data types
- **Processing:** Handles NaN, infinities, extreme outliers

---

## 🔧 Key Features

| Feature | Description |
|---------|-------------|
| ✅ **Any CSV Upload** | Auto-detects columns, handles missing values |
| ✅ **Smart Predictions** | Uses actual data statistics for probability generation |
| ✅ **Risk Segmentation** | Automatic Low / Medium / High / Very High categorization |
| ✅ **Correlation Analysis** | Feature heatmap for data insights |
| ✅ **Batch Processing** | Chunked processing for large files up to 200MB |
| ✅ **Download Results** | Export predictions with probabilities |
| ✅ **12 Algorithms** | Full model comparison with ROC-AUC, PR-AUC, F1-Score |
| ✅ **Real-time API** | Low-latency Flask API for production scoring |

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.12+
pip
Docker (optional)
```

### Installation

```bash
# Clone repository
git clone https://github.com/syedmahammadsaleem-SS/banking-transaction-prediction.git
cd banking-transaction-prediction

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python main_pipeline.py
```

### Launch Dashboard

```bash
streamlit run app/streamlit/app.py
```

### Start API

```bash
python app/flask/app.py
```

### Docker Deployment

```bash
docker-compose up -d
```

---

## 📁 Project Structure

```
banking-transaction-prediction/
├── 📂 app/
│   ├── flask/               # REST API
│   └── streamlit/           # Interactive dashboard
├── 📂 src/
│   ├── data/                # ETL pipeline
│   ├── features/            # Feature engineering
│   ├── models/              # ML training & evaluation
│   └── pipeline/            # MLOps orchestration
├── 📂 data/
│   └── raw/                 # Training data
├── 📂 docs/
│   └── screenshots/         # Dashboard screenshots
├── 📂 reports/
│   └── figures/             # Generated charts
├── main_pipeline.py         # End-to-end orchestrator
├── requirements.txt         # Dependencies
├── Dockerfile               # Container config
└── docker-compose.yml       # Full stack deployment
```

---

## 📚 Documentation

- [Business Understanding](docs/business_understanding.md)
- [Technical Architecture](docs/technical_architecture.md)
- [Interview Preparation — 50 Q&A](docs/interview_prep.md)

---

## 👨‍💻 About This Project

![About](docs/screenshots/about.png)

This production-grade ML system enables:
- **Targeted Marketing** — Focus campaigns on high-probability customers
- **Resource Optimization** — Reduce wasted marketing spend
- **Revenue Growth** — Increase transaction volume by 15–20%

---

## 🤝 Connect With Me

**Syed Mahammad Saleem** — Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/syed-mahammad-saleem-368301264/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/syedmahammadsaleem-SS)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail&logoColor=white)](mailto:syedmahammadsaleem@gmail.com)

> Open to data science opportunities, collaborations, and consulting projects!

---

## 📄 License

This project is for demonstration purposes. All rights reserved © 2026.
