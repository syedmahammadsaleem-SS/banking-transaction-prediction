# System Architecture

## Production Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │  Web App     │  │  Mobile App  │  │  CRM System  │                │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │
└─────────┼─────────────────┼─────────────────┼──────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY (Kong/AWS)                           │
│                    Rate Limiting | Authentication | SSL                  │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐ │
│  │  Flask API          │  │  Streamlit          │  │  Batch Worker    │ │
│  │  (Real-time)        │  │  Dashboard          │  │  (Airflow)       │ │
│  │  Port: 5000          │  │  Port: 8501          │  │  Scheduled       │ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └────────┬─────────┘ │
└─────────────┼──────────────────────┼────────────────────┼──────────────┘
              │                      │                    │
              ▼                      ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MODEL SERVING LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Model Registry (MLflow)                                       │    │
│  │  ├── Production Model: LightGBM v1.0.0 (ROC-AUC: 0.9234)        │    │
│  │  ├── Staging Model: XGBoost v1.1.0 (ROC-AUC: 0.9189)           │    │
│  │  └── Archived Models: ...                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Feature Store (Feast/DynamoDB)                                  │    │
│  │  ├── Real-time features (last 24h transactions)                 │    │
│  │  ├── Batch features (monthly aggregations)                      │    │
│  │  └── Static features (customer demographics)                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Raw Data    │  │  Feature     │  │  Model       │  │  Prediction│ │
│  │  (S3/Data    │  │  Store       │  │  Artifacts   │  │  Logs      │ │
│  │   Lake)      │  │  (Redis/     │  │  (S3/EFS)    │  │  (PostgreSQL│ │
│  │              │  │   DynamoDB)  │  │              │  │   /S3)     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MONITORING & OBSERVABILITY                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Prometheus  │  │  Grafana     │  │  PagerDuty   │  │  MLflow    │ │
│  │  (Metrics)   │  │  (Dashboards)│  │  (Alerts)    │  │  (Tracking)│ │
│  │  Port: 9090  │  │  Port: 3000  │  │  (Webhooks)  │  │  Port: 5001│ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | Kubernetes, Docker | Container management |
| **API** | Flask, Gunicorn | REST API serving |
| **Dashboard** | Streamlit, Plotly | Interactive visualization |
| **ML Framework** | LightGBM, XGBoost, CatBoost | Model training |
| **Feature Store** | Feast, Redis | Feature serving |
| **Data Warehouse** | Snowflake, BigQuery | Analytics storage |
| **MLflow** | MLflow | Experiment tracking |
| **Monitoring** | Prometheus, Grafana | Metrics & alerting |
| **CI/CD** | GitHub Actions, ArgoCD | Deployment automation |
| **Data Quality** | Great Expectations | Validation framework |

## Deployment Strategy

### Blue-Green Deployment
1. Train new model (Green)
2. A/B test against current (Blue)
3. Gradually shift traffic (10% → 50% → 100%)
4. Monitor for 24-48 hours
5. Rollback if metrics degrade

### Canary Deployment
- Route 5% traffic to new model
- Monitor error rates and latency
- Increase to 25%, 50%, 100% if stable
- Automatic rollback if error rate > threshold

## Security Architecture

```
┌─────────────────────────────────────────┐
│           WAF / DDoS Protection         │
├─────────────────────────────────────────┤
│           API Gateway (Kong)              │
│  - Rate limiting: 100 req/min per key    │
│  - Authentication: API Key + JWT           │
│  - SSL/TLS termination                   │
├─────────────────────────────────────────┤
│           Application (Flask)            │
│  - Input validation (Pydantic schemas)   │
│  - SQL injection prevention              │
│  - XSS protection                        │
├─────────────────────────────────────────┤
│           Model Serving                  │
│  - Sandboxed execution                   │
│  - Resource limits (CPU/Memory)          │
├─────────────────────────────────────────┤
│           Data Layer                     │
│  - Encryption at rest (AES-256)          │
│  - Encryption in transit (TLS 1.3)       │
│  - Role-based access control             │
│  - Audit logging                         │
└─────────────────────────────────────────┘
```
