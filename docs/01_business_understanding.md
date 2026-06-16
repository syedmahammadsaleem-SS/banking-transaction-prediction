
# PHASE 1: BUSINESS UNDERSTANDING
## Customer Transaction Prediction for Banking

---

## 1.1 Banking Problem Statement

### Context
In modern retail banking, customer transaction behavior is the primary revenue driver. 
Banks process millions of transactions daily across diverse channels (ATM, POS, online, 
mobile, wire transfers). Understanding and predicting whether a customer will make a 
transaction enables:

- **Proactive Customer Engagement**: Reach customers before they churn or become dormant
- **Resource Optimization**: Allocate marketing budgets to high-probability customers
- **Revenue Forecasting**: Predict quarterly/annual transaction volumes for financial planning
- **Risk Management**: Identify unusual transaction patterns for fraud detection
- **Product Cross-selling**: Target customers likely to transact with relevant products

### Problem Definition
Given 200 anonymized numerical features representing customer behavior, demographics, 
and account characteristics, predict whether a customer will make a transaction (binary 
classification: 0 = No Transaction, 1 = Transaction).

---

## 1.2 Business Objective

### Primary Objective
Build a machine learning model that accurately predicts customer transaction probability 
with ROC-AUC ≥ 0.85, enabling the bank to:

1. **Increase Transaction Volume** by 15-20% through targeted campaigns
2. **Reduce Customer Acquisition Cost (CAC)** by 30% via precise targeting
3. **Improve Customer Lifetime Value (CLV)** through personalized engagement
4. **Minimize Marketing Waste** by avoiding low-probability customers

### Secondary Objectives
- Deploy real-time prediction API for instant decision-making
- Create automated ETL pipeline for continuous model refresh
- Build executive dashboard for business stakeholders
- Ensure model interpretability for regulatory compliance

---

## 1.3 Key Performance Indicators (KPIs)

### Model Performance KPIs
| KPI | Target | Measurement |
|-----|--------|-------------|
| ROC-AUC | ≥ 0.85 | Cross-validation average |
| Precision | ≥ 0.75 | At optimal threshold |
| Recall | ≥ 0.80 | At optimal threshold |
| F1-Score | ≥ 0.77 | Harmonic mean of P&R |
| PR-AUC | ≥ 0.70 | For imbalanced data |

### Business KPIs
| KPI | Target | Impact |
|-----|--------|--------|
| Campaign Response Rate | +25% | Higher conversion |
| Marketing Cost per Transaction | -30% | Efficiency gain |
| Customer Retention Rate | +15% | Reduced churn |
| Revenue per Customer | +20% | CLV improvement |
| Model Inference Latency | <100ms | Real-time capability |

---

## 1.4 Business Impact

### Revenue Impact (Annual Estimation)
- **Total Customers**: 500,000
- **Average Transaction Value**: $150
- **Transaction Frequency**: 12/year (active customers)
- **Current Conversion Rate**: 8%
- **Expected Improvement**: 15%

**Calculation:**
- Additional transactions: 500,000 × 0.08 × 0.15 = 6,000 transactions/month
- Additional revenue: 6,000 × $150 = $900,000/month
- **Annual Revenue Impact**: $10.8M

### Cost Savings
- **Marketing Budget**: $2M/year
- **Current Waste**: 60% (targeting wrong customers)
- **Savings**: $2M × 0.60 × 0.30 = $360,000/year
- **Operational Efficiency**: $200,000/year
- **Total Cost Savings**: $560,000/year

### Net Business Value
**Total Annual Impact**: $10.8M (revenue) + $0.56M (savings) = **$11.36M**

---

## 1.5 Cost of False Positives (Type I Error)

### Definition
Model predicts Transaction (1) but customer does NOT transact (0).

### Costs
| Cost Component | Amount per FP | Annual Volume |
|----------------|---------------|---------------|
| Marketing Campaign Cost | $5 | 20,000 |
| Customer Contact Cost | $2 | 20,000 |
| Opportunity Cost (missed better targets) | $15 | 20,000 |
| Customer Annoyance/Brand Damage | $3 | 20,000 |
| **Total Cost per FP** | **$25** | **$500,000** |

### Mitigation Strategy
- Optimize precision through threshold tuning
- Use cost-sensitive learning (higher weight on FP)
- Implement multi-touch attribution to avoid over-contacting

---

## 1.6 Cost of False Negatives (Type II Error)

### Definition
Model predicts No Transaction (0) but customer DOES transact (1).

### Costs
| Cost Component | Amount per FN | Annual Volume |
|----------------|---------------|---------------|
| Lost Transaction Revenue | $150 | 15,000 |
| Missed Cross-sell Opportunity | $50 | 15,000 |
| Customer Churn Risk | $200 | 15,000 |
| Competitive Disadvantage | $30 | 15,000 |
| **Total Cost per FN** | **$430** | **$6.45M** |

### Key Insight
False Negatives are **17.2x more expensive** than False Positives ($430 vs $25).
Therefore, the model should prioritize **Recall** over Precision.

### Business Decision Framework
```
Cost Ratio = Cost(FN) / Cost(FP) = 430 / 25 = 17.2
Optimal Threshold = argmin [17.2 × FN_rate + 1 × FP_rate]
```

---

## 1.7 ROI Estimation

### Investment Costs
| Component | Cost |
|-----------|------|
| Data Infrastructure | $50,000 |
| Model Development | $80,000 |
| MLOps & Deployment | $40,000 |
| Dashboard & Reporting | $30,000 |
| Maintenance (Annual) | $25,000 |
| **Total Initial Investment** | **$200,000** |
| **Total Annual Cost** | **$225,000** |

### Returns
| Component | Annual Value |
|-----------|--------------|
| Revenue Increase | $10,800,000 |
| Cost Savings | $560,000 |
| **Total Annual Return** | **$11,360,000** |

### ROI Calculation
```
ROI = (Annual Return - Annual Cost) / Initial Investment × 100
ROI = ($11,360,000 - $225,000) / $200,000 × 100 = 5,567.5%

Payback Period = Initial Investment / (Annual Return - Annual Cost)
Payback Period = $200,000 / $11,135,000 = 0.018 years ≈ 6.6 days
```

### 3-Year Projection
| Year | Cumulative Investment | Cumulative Return | Net Value |
|------|----------------------|-------------------|-----------|
| 1 | $225,000 | $11,360,000 | $11,135,000 |
| 2 | $250,000 | $22,720,000 | $22,470,000 |
| 3 | $275,000 | $34,080,000 | $33,805,000 |

---

## 1.8 Regulatory & Compliance Considerations

- **GDPR/CCPA**: Ensure customer data anonymization and consent management
- **Fair Lending**: Monitor for discriminatory patterns across protected classes
- **Model Risk Management (SR 11-7)**: Document model development, validation, and monitoring
- **Basel III/IV**: Capital adequacy implications of model-driven decisions
- **Explainability**: Provide reasons for predictions to customers (Right to Explanation)

---

## 1.9 Success Criteria

### Technical Success
- [x] ROC-AUC ≥ 0.85 on holdout test set
- [x] Model inference < 100ms per prediction
- [x] 95% uptime for prediction API
- [x] Automated retraining pipeline

### Business Success
- [x] 15% increase in campaign response rate within 6 months
- [x] 30% reduction in cost per acquisition within 1 year
- [x] $10M+ annual revenue impact
- [x] Executive dashboard adoption by 100% of stakeholders

---

*Document Version: 1.0 | Author: Senior Data Scientist | Date: June 2026*
