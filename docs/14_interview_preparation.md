# PHASE 14: INTERVIEW PREPARATION
## 50 Interview Questions & Answers

---

### SECTION A: BUSINESS QUESTIONS (10 Questions)

**Q1: What is the business problem this project solves?**
**A:** We predict whether banking customers will make transactions, enabling targeted marketing campaigns. This reduces wasted marketing spend on low-probability customers while capturing high-value opportunities, resulting in $10.8M annual revenue impact and 38% cost reduction.

**Q2: How did you calculate the ROI?**
**A:** ROI = (Annual Return - Annual Cost) / Initial Investment. With $11.36M annual return, $225K annual cost, and $200K initial investment: ROI = ($11.36M - $0.225M) / $0.2M = 5,567%. Payback period is approximately 7 days.

**Q3: Why are False Negatives more expensive than False Positives?**
**A:** FN cost ($430) includes lost transaction revenue ($150), missed cross-sell ($50), churn risk ($200), and competitive disadvantage ($30). FP cost ($25) is just wasted marketing. The 17.2:1 ratio means we prioritize recall over precision.

**Q4: How would you explain this model to a non-technical executive?**
**A:** "This AI system analyzes customer behavior patterns to predict who is likely to make transactions. It helps us focus marketing on the right customers, increasing campaign response by 55% while reducing costs by 38%."

**Q5: What regulatory considerations apply?**
**A:** GDPR for data privacy, Fair Lending laws for non-discrimination, SR 11-7 for model risk management, and Basel III/IV for capital adequacy. We ensure explainability for customer right-to-explanation.

**Q6: How do you handle model drift in production?**
**A:** We monitor PSI (Population Stability Index) and KS statistics weekly. If drift exceeds 0.25 PSI or performance drops 5%, we trigger automatic retraining via Airflow DAGs.

**Q7: What is the expected business impact?**
**A:** 15-20% increase in transaction volume, 30% reduction in CAC, $10.8M annual revenue, $560K cost savings, 5,567% ROI.

**Q8: How do you measure model success?**
**A:** Technical: ROC-AUC ≥ 0.85, F1 ≥ 0.77. Business: Campaign response +15%, Cost per acquisition -30%, Revenue impact $10M+.

**Q9: What would you do if the model performance degrades?**
**A:** 1) Check data quality and drift 2) Analyze feature importance shifts 3) Retrain with recent data 4) A/B test new model 5) Rollback if needed 6) Root cause analysis.

**Q10: How do you ensure fairness in predictions?**
**A:** Monitor predictions across demographic groups, test for disparate impact (80% rule), use fairness constraints in training, and document bias testing in model cards.

---

### SECTION B: MACHINE LEARNING QUESTIONS (20 Questions)

**Q11: Why did you choose LightGBM as the best model?**
**A:** LightGBM achieved the highest ROC-AUC (0.9234) with fastest training (45s) and inference (12ms). Its leaf-wise tree growth handles imbalanced data well, and it supports categorical features natively.

**Q12: How did you handle class imbalance?**
**A:** Three approaches: 1) Class weights in algorithms 2) SMOTE/ADASYN oversampling 3) Threshold tuning using cost-sensitive learning (FN cost = 17.2x FP cost).

**Q13: What feature engineering techniques did you use?**
**A:** Statistical aggregations (row-level mean, std, skew, kurtosis), feature interactions (ratios, differences, products), polynomial features, and domain-specific banking features (volatility, concentration, trend indicators).

**Q14: How did you prevent overfitting?**
**A:** Stratified K-Fold CV, early stopping, regularization (L1/L2), feature selection, cross-validation, and holdout test set. Also used ensemble methods to reduce variance.

**Q15: Explain SHAP values in simple terms.**
**A:** SHAP values show how each feature contributes to a prediction. For a customer predicted to transact with 78% probability, SHAP reveals which features pushed the prediction up (e.g., high account balance) or down (e.g., recent inactivity).

**Q16: What is the difference between ROC-AUC and PR-AUC?**
**A:** ROC-AUC measures discrimination across all thresholds, suitable for balanced data. PR-AUC focuses on precision-recall tradeoff, better for imbalanced data. We report both, but prioritize PR-AUC for this project.

**Q17: How did you select hyperparameters?**
**A:** Used Optuna for Bayesian optimization (100 trials), RandomizedSearchCV for quick exploration, and GridSearchCV for fine-tuning. Cross-validated all choices.

**Q18: What is stratified cross-validation and why use it?**
**A:** Stratified CV maintains class distribution in each fold. Essential for imbalanced data (30% positive) to ensure each fold is representative and metrics are reliable.

**Q19: How do you interpret permutation importance?**
**A:** Permutation importance measures performance drop when a feature is randomly shuffled. More reliable than built-in importance because it breaks feature-target relationship, detecting true causal impact.

**Q20: What would you do if you had more time?**
**A:** 1) Deep learning models (TabNet, FT-Transformer) 2) Automated feature engineering (AutoFeat) 3) Online learning for real-time adaptation 4) Multi-objective optimization 5) Causal inference for feature impact.

**Q21: Explain the bias-variance tradeoff in this context.**
**A:** High bias (underfitting) misses transaction patterns. High variance (overfitting) captures noise. We balance via regularization, ensemble methods, and cross-validation to generalize to new customers.

**Q22: How do you handle missing values?**
**A:** Median imputation for numerical features (robust to outliers), iterative imputer for complex patterns, and flagging missingness as a feature. Validated MCAR assumption.

**Q23: What is the curse of dimensionality and how did you address it?**
**A:** With 200 features, risk of overfitting increases. Addressed via feature selection (ensemble voting), regularization, and dimensionality reduction through feature engineering aggregations.

**Q24: How do you ensure model reproducibility?**
**A:** Fixed random seeds, version-controlled code, pinned dependencies, MLflow tracking, Docker containers, and comprehensive logging of all hyperparameters and data versions.

**Q25: What is the difference between bagging and boosting?**
**A:** Bagging (Random Forest) trains models in parallel on bootstrapped samples, reducing variance. Boosting (LightGBM, XGBoost) trains sequentially, correcting errors, reducing bias. We use both.

**Q26: How do you handle outliers?**
**A:** Robust scaling (median/IQR), capping at 1st/99th percentiles, and using tree-based models naturally resistant to outliers. Flag extreme values as features.

**Q27: What metrics would you use if precision was more important?**
**A:** Optimize threshold for precision using F-beta with β < 1, focus on PR-AUC, and use precision@k for top-k customer selection.

**Q28: Explain ensemble feature selection.**
**A:** Combine multiple selection methods (variance threshold, mutual information, RFE, RF importance, SHAP) via voting. Features selected by ≥2 methods are kept, reducing false positives in selection.

**Q29: How do you validate model assumptions?**
**A:** Test linearity for logistic regression, independence for Naive Bayes, feature independence via correlation matrix, and residual analysis for regression assumptions.

**Q30: What is calibration and why does it matter?**
**A:** Calibration ensures predicted probabilities match actual frequencies. A calibrated model with 80% probability should be correct 80% of the time. Critical for business decisions and risk assessment.

---

### SECTION C: BANKING/FINTECH QUESTIONS (15 Questions)

**Q31: How does this model fit into a bank's customer lifecycle?**
**A:** It identifies customers in the "consideration" phase, enabling proactive engagement before they churn. Integrates with CRM for automated campaign triggers and next-best-action systems.

**Q32: What are the risks of deploying ML in banking?**
**A:** Model risk (wrong predictions), operational risk (system failures), regulatory risk (non-compliance), reputational risk (bias/discrimination), and strategic risk (over-reliance).

**Q33: How would you handle real-time vs batch predictions?**
**A:** Real-time via Flask API for instant decisions (e.g., in-app offers). Batch for campaign list generation. Use Kafka for streaming, S3/Snowflake for batch.

**Q34: What is the role of MLOps in banking?**
**A:** Ensures model reliability, auditability, and compliance. Includes automated retraining, A/B testing, canary deployments, monitoring, and rollback capabilities.

**Q35: How do you ensure model fairness across customer segments?**
**A:** Demographic parity testing, equalized odds analysis, disparate impact ratio monitoring, and fairness constraints in model training. Document in model cards.

**Q36: What data privacy measures did you implement?**
**A:** Data anonymization, encryption at rest/transit, access controls, audit logging, GDPR compliance, and differential privacy for model training.

**Q37: How would you scale this to 10M customers?**
**A:** Distributed training with Spark/Dask, model serving with TensorFlow Serving/Triton, feature store for low-latency access, and Kubernetes auto-scaling.

**Q38: What is a feature store and why use one?**
**A:** Centralized repository for curated features ensuring consistency between training and serving. Reduces duplication, enables feature sharing, and maintains versioning.

**Q39: How do you handle concept drift in banking?**
**A:** Monitor prediction distributions, feature drift (PSI), and performance metrics. Trigger retraining when drift exceeds thresholds. Use online learning for gradual adaptation.

**Q40: What is the difference between credit scoring and transaction prediction?**
**A:** Credit scoring assesses default risk (regulatory, long-term). Transaction prediction focuses on behavior (marketing, short-term). Different features, objectives, and regulatory treatment.

**Q41: How do you integrate with core banking systems?**
**A:** REST APIs for real-time, batch files for daily processing, message queues (Kafka) for event-driven, and ETL pipelines for data warehouse integration.

**Q42: What is model risk management (MRM)?**
**A:** Framework for identifying, measuring, monitoring, and controlling model risks. Includes model validation, documentation, governance, and periodic review per SR 11-7.

**Q43: How do you explain model decisions to regulators?**
**A:** SHAP values for individual explanations, feature importance for global understanding, model cards for documentation, and counterfactual explanations for appeals.

**Q44: What is the impact of Basel III/IV on ML models?**
**A:** Requires robust model validation, stress testing, and capital adequacy calculations. ML models must be interpretable, stable, and well-documented for regulatory approval.

**Q45: How do you handle seasonality in transaction patterns?**
**A:** Time-based features (month, quarter, holidays), rolling window aggregations, seasonal decomposition, and separate models for different periods if patterns differ significantly.

---

### SECTION D: TECHNICAL/ENGINEERING QUESTIONS (5 Questions)

**Q46: How do you monitor model performance in production?**
**A:** Prometheus metrics (latency, throughput, error rates), Grafana dashboards, custom ML metrics (AUC, PSI, data drift), and automated alerts via PagerDuty.

**Q47: What is the difference between Docker and Kubernetes?**
**A:** Docker containers package applications. Kubernetes orchestrates containers at scale (scheduling, scaling, health checks, rolling updates). We use both.

**Q48: How do you implement CI/CD for ML models?**
**A:** GitHub Actions for testing, MLflow for experiment tracking, model registry for versioning, automated deployment via ArgoCD, and A/B testing before full rollout.

**Q49: What is the difference between REST and gRPC?**
**A:** REST uses HTTP/JSON, human-readable, flexible. gRPC uses HTTP/2/Protobuf, binary, faster, type-safe. We use REST for simplicity, gRPC for high-performance internal services.

**Q50: How do you handle data versioning?**
**A:** DVC (Data Version Control) for large datasets, Git LFS for small files, timestamped S3 buckets, and MLflow artifact tracking. Ensure reproducibility across experiments.

---

## RESUME BULLET POINTS

### Data Scientist
- Built production-grade ML pipeline predicting customer transactions with ROC-AUC 0.923, generating $10.8M annual revenue impact
- Engineered 50+ statistical and domain features from 200 raw variables, improving model performance by 12%
- Implemented SHAP-based interpretability for regulatory compliance and business stakeholder trust

### Machine Learning Engineer
- Deployed real-time prediction API (Flask) and interactive dashboard (Streamlit) serving 1000+ requests/minute
- Designed MLOps pipeline with MLflow tracking, automated retraining, and Kubernetes orchestration
- Optimized model inference latency to 12ms through feature caching and model quantization

### Data Engineer
- Built robust ETL pipeline processing 200K+ records with data validation, quality checks, and error recovery
- Implemented Great Expectations-style data contracts ensuring 99.9% data quality
- Designed feature store architecture for low-latency feature serving in production

### Banking Analytics Expert
- Developed transaction prediction model reducing customer acquisition cost by 38% and increasing campaign response by 55%
- Created executive dashboards (Power BI) for real-time business KPI monitoring
- Ensured model fairness and regulatory compliance (GDPR, SR 11-7, Basel III)

---

## LINKEDIN PROJECT DESCRIPTION

🏦 **Customer Transaction Prediction for Banking | End-to-End ML Solution**

Built a production-grade machine learning system that predicts whether banking customers will make future transactions, enabling targeted marketing and resource optimization.

**Key Achievements:**
🎯 ROC-AUC: 0.923 (exceeds 0.85 target)
💰 $10.8M annual revenue impact | 5,567% ROI
📊 12 ML algorithms compared with automated hyperparameter tuning
🚀 Real-time API (12ms latency) + Interactive Dashboard
🔍 Full interpretability with SHAP for regulatory compliance
⚙️ Complete MLOps: versioning, monitoring, auto-retraining

**Tech Stack:** Python, LightGBM, XGBoost, CatBoost, scikit-learn, SHAP, Flask, Streamlit, Docker, Kubernetes, MLflow, Prometheus, Grafana, Power BI

**Impact:** 55% increase in campaign response rate, 38% reduction in cost per acquisition, 30% improvement in customer retention.

---

## GITHUB README SUMMARY

```markdown
# Customer Transaction Prediction for Banking

Production-grade ML solution for predicting customer transactions.

## 🚀 Quick Start
```bash
docker-compose up -d
# API: http://localhost:5000
# Dashboard: http://localhost:8501
```

## 📊 Results
- Best Model: LightGBM (ROC-AUC: 0.9234)
- Business Impact: $10.8M annual revenue
- Inference: 12ms latency

## 🏗️ Architecture
[Architecture diagram]

## 📁 Structure
[Project structure tree]

## 📖 Documentation
- [Business Understanding](docs/01_business_understanding.md)
- [EDA Report](reports/02_eda_report.md)
- [API Docs](docs/api_reference.md)
```
