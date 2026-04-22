# Crypto Risk & AML Monitoring Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Crash_Detection-00897B?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**Ontario Tech University · Winter 2026**
*Nan Chen · Student ID: 101021912*

**Live App:** [crypto-aml-dashboard.streamlit.app](https://chennan7909-cmd-crypto-aml-risk-dashboard-app.streamlit.app)

</div>

---

## Overview

An interactive web dashboard combining two machine learning systems for cryptocurrency risk management:

- **Price Risk**: LSTM next-day price forecasting and XGBoost tail-risk detection for BTC and SOL, with SHAP feature attribution
- **AML Scoring**: Rule-based transaction pattern scoring engine aligned with FATF typology indicators, producing risk tiers and triggered rule lists
- **Regulatory Context**: FATF, GDPR, PIPEDA, and EU AI Act compliance framing drawn from academic research on federated learning in AML

Built for compliance officers and risk analysts who need interpretable, scenario-based risk outputs — not just raw model predictions.

---

## Live Demo

The app is deployed on Streamlit Cloud. No installation required — open the link and interact immediately.

```
https://chennan7909-cmd-crypto-aml-risk-dashboard-app.streamlit.app
```

---

## Features

### Tab 1 — Price Risk Dashboard

- Real-time BTC / SOL price chart with crash alert markers
- XGBoost 5%-drawdown crash probability time series (adjustable threshold)
- LSTM next-day price forecast vs. actual with RMSE and MAPE metrics
- SHAP feature importance bar chart for model interpretability

### Tab 2 — AML Risk Scoring

- 5 pre-configured transaction scenarios (normal retail, structuring, cross-border, rapid layering, mixer usage)
- Manual parameter override for custom scenarios
- Rule-based scoring engine across 8 FATF typology indicators
- Risk tier output: Low / Medium / High / Critical
- Gauge chart and score decomposition by rule

### Tab 3 — Regulatory Context

- FATF Recommendation 15 compliance summary
- GDPR / Canadian PIPEDA data minimisation requirements
- EU AI Act high-risk AI classification implications
- Model limitations and ethical risk disclosure

---

## Architecture

```
yfinance (BTC-USD + SOL-USD, Jan 2021 – Mar 2026)
    |
    +-- Feature Engineering
    |     Standard: MA7, MA21, RSI, Bollinger Band, MACD, Volatility
    |     Cross-asset: BTC signals lagged [1, 3, 7 days] -> SOL features
    |
    +-- XGBoost Classifier
    |     Target: 1 if next-day return < -5%  |  Threshold: 0.30
    |     scale_pos_weight for class imbalance  |  SHAP explainability
    |
    +-- LSTM Regressor
    |     3-layer: LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense
    |     Separate MinMaxScaler for X and y  |  EarlyStopping (patience=5)
    |
    +-- AML Rule Engine
    |     8 FATF typology rules  |  Weighted score 0-100
    |     Risk tiers: Low / Medium / High / Critical
    |
    +-- Streamlit Dashboard
          @st.cache_resource: models trained once, cached for session
          Plotly interactive charts  |  Sidebar controls
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Dashboard | Streamlit |
| Charts | Plotly |
| ML — Classification | XGBoost |
| ML — Regression | LSTM (TensorFlow/Keras) |
| Explainability | SHAP |
| Data | yfinance (Yahoo Finance) |
| Feature Engineering | Pandas, NumPy, Scikit-Learn |
| Deployment | Streamlit Cloud |

---

## Run Locally

```bash
git clone https://github.com/chennan7909-cmd/crypto-aml-risk-dashboard.git
cd crypto-aml-risk-dashboard
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`. On first load, models are trained and cached automatically (~3 minutes for LSTM).

---

## Model Performance

| Asset | Model | Metric | Value |
|-------|-------|--------|-------|
| BTC | LSTM | MAPE | ~2.8% |
| BTC | XGBoost | ROC-AUC | ~0.77 |
| SOL | LSTM | MAPE | ~2.5% |
| SOL | XGBoost | ROC-AUC | ~0.74 |

Results vary slightly across runs due to non-stationarity of financial time series.

---

## Ethical Disclaimer

This dashboard is strictly academic. Model outputs should not be interpreted as financial or legal advice. Key limitations are documented in Tab 3 of the app, including training window bias, threshold sensitivity, AML rule engine simplification, and fairness risks of algorithmic compliance tools.

---

## Related Projects

- [Real-Time SME Credit Risk Pipeline](https://github.com/chennan7909-cmd/sme-credit-risk-pipeline) — PySpark · Kafka · XGBoost · Lambda Architecture
- [AI-Driven Crypto Risk Forecasting](https://github.com/chennan7909-cmd/crypto-risk-forecasting) — LSTM · XGBoost · BTC cross-asset signals

---

## References

- FATF (2021). *Opportunities and challenges of new technologies for AML/CFT.*
- McMahan et al. (2017). *Communication-efficient learning of deep networks.* PMLR.
- Regulation (EU) 2016/679 (GDPR), Articles 5 and 25.
- Regulation (EU) 2024/1689 (EU AI Act).
- Chen, N. (2026). *Federated Learning in AML: A Socio-technical Analysis.* Ontario Tech University.

---

<div align="center">

*Built with Streamlit · XGBoost · TensorFlow · SHAP · Plotly*
*Ontario Tech University · Winter 2026*

</div>
