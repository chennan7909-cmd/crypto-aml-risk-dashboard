"""
Crypto Risk & AML Monitoring Dashboard
Nan Chen | Ontario Tech University
Streamlit Cloud deployment — XGBoost only, no tensorflow/keras
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import shap

st.set_page_config(page_title="Crypto Risk & AML Dashboard", page_icon="📊", layout="wide")
st.title("Crypto Risk & AML Monitoring Dashboard")
st.caption("Nan Chen · Ontario Tech University · XGBoost Price Forecasting + Crash Detection + AML Pattern Scoring")
st.divider()

@st.cache_data(show_spinner="Downloading market data...")
def load_raw(ticker):
    df = yf.download(ticker, start="2021-01-01", end="2026-03-31", auto_adjust=True, progress=False)
    df.columns = ["Close", "High", "Low", "Open", "Volume"]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

@st.cache_data(show_spinner="Engineering features...")
def add_features(_df):
    df = _df.copy()
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA21"] = df["Close"].rolling(21).mean()
    df["MA_Ratio"] = df["MA7"] / df["MA21"]
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Position"] = (df["Close"] - mid) / (2 * std + 1e-9)
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    df["MACD_Hist"] = macd - macd.ewm(span=9).mean()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(7).std()
    for lag in [1, 2, 3, 5, 7]:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
        df[f"Return_lag{lag}"] = df["Return"].shift(lag)
    df["Next_Return"] = df["Close"].pct_change().shift(-1)
    df["Risk_Event"] = (df["Next_Return"] < -0.05).astype(int)
    df["Target_Price"] = df["Close"].shift(-1)
    return df.dropna()

@st.cache_data(show_spinner="Adding BTC cross-asset signals...")
def add_cross_asset(_sol_df, _btc_df):
    sol = _sol_df.copy()
    for lag in [1, 3, 7]:
        for col in ["Return", "Volatility", "RSI", "MACD_Hist", "MA_Ratio"]:
            sol[f"BTC_{col}_lag{lag}"] = _btc_df[col].shift(lag)
    return sol.dropna()

FEATURES_REG = ["Close_lag1","Close_lag2","Close_lag3","Close_lag5","Close_lag7",
                "Return_lag1","Return_lag2","Return_lag3",
                "Volume","Volatility","MA7","MA21","MA_Ratio","RSI","BB_Position","MACD_Hist"]
FEATURES_CLF = ["Volume","Volatility","MA7","RSI","BB_Position","MACD_Hist","MA_Ratio",
                "Return_lag1","Return_lag2","Return_lag3"]

@st.cache_resource(show_spinner="Training price forecast model...")
def train_regressor(key, _df, features):
    X = _df[features].values
    y = _df["Target_Price"].values
    split = int(len(X) * 0.8)
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X[:split])
    X_te = scaler.transform(X[split:])
    y_tr, y_te = y[:split], y[split:]
    reg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method="hist")
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mape = float(np.mean(np.abs((y_te - pred) / (y_te + 1e-9))) * 100)
    return reg, scaler, round(mape, 2), round(rmse, 0)

@st.cache_resource(show_spinner="Training crash detection model...")
def train_classifier(key, _df, features):
    X = _df[features].values
    y = _df["Risk_Event"].values
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    ratio = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1)
    clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                        scale_pos_weight=ratio, eval_metric="logloss",
                        random_state=42, tree_method="hist")
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    f1 = f1_score(y_te, (proba > 0.30).astype(int), zero_division=0)
    return clf, round(auc, 4), round(f1, 4)

with st.sidebar:
    st.header("Controls")
    asset = st.selectbox("Select Asset", ["BTC", "SOL"])
    window = st.slider("Lookback window (days)", 60, 365, 180)
    threshold = st.slider("Crash alert threshold", 0.10, 0.50, 0.30, 0.05)
    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- Price forecast: XGBoost Regressor")
    st.markdown("- Crash detection: XGBoost Classifier + SHAP")
    st.markdown("- Threshold: 0.30 (recall-prioritised)")
    st.markdown("- SOL: +15 cross-asset BTC lag features")

btc_raw = load_raw("BTC-USD")
sol_raw = load_raw("SOL-USD")
btc_full = add_features(btc_raw)
sol_full = add_cross_asset(add_features(sol_raw), btc_full)

if asset == "BTC":
    df_full = btc_full
    f_reg, f_clf = FEATURES_REG, FEATURES_CLF
else:
    btc_lags = [c for c in sol_full.columns if c.startswith("BTC_")]
    df_full = sol_full
    f_reg = FEATURES_REG + btc_lags
    f_clf = FEATURES_CLF + btc_lags

df = df_full.tail(window).copy()
reg, reg_scaler, reg_mape, reg_rmse = train_regressor(asset, df_full, f_reg)
clf, xgb_auc, xgb_f1 = train_classifier(asset, df_full, f_clf)

tab1, tab2, tab3 = st.tabs(["Price Risk Dashboard", "AML Risk Scoring", "Regulatory Context"])

with tab1:
    st.subheader(f"{asset} — Price Risk Dashboard")
    crash_proba = clf.predict_proba(df[f_clf].values)[:, 1]
    df = df.copy()
    df["Crash_Prob"] = crash_proba

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${float(df['Close'].iloc[-1]):,.0f}")
    c2.metric("Crash Probability", f"{float(crash_proba[-1]):.1%}")
    c3.metric("Risk Tier", "Critical" if crash_proba[-1]>0.55 else "High" if crash_proba[-1]>0.35 else "Medium" if crash_proba[-1]>0.20 else "Low")
    c4.metric(f"High-Risk Days (>{threshold:.0%})", f"{int((crash_proba>threshold).sum())} / {len(df)}")
    st.divider()

    mask = crash_proba > threshold
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price", line=dict(color="#2196F3", width=1.5)))
    fig_price.add_trace(go.Scatter(x=df.index[mask], y=df["Close"].values[mask], mode="markers",
                                    name=f"Crash Alert (>{threshold:.0%})", marker=dict(color="red", size=7, symbol="x")))
    fig_price.update_layout(title=f"{asset} Price with Crash Alerts", height=380, template="plotly_dark")
    st.plotly_chart(fig_price, use_container_width=True)

    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(x=df.index, y=crash_proba, fill="tozeroy", name="Crash Probability", line=dict(color="#FF5722")))
    fig_prob.add_hline(y=threshold, line_dash="dash", line_color="yellow", annotation_text=f"Threshold ({threshold:.0%})")
    fig_prob.update_layout(title=f"Crash Probability  |  AUC {xgb_auc}  F1 {xgb_f1}", yaxis=dict(tickformat=".0%"), height=280, template="plotly_dark")
    st.plotly_chart(fig_prob, use_container_width=True)

    st.subheader("XGBoost Next-Day Price Forecast")
    X_all = reg_scaler.transform(df_full[f_reg].values)
    pred_all = reg.predict(X_all)
    show_n = min(120, len(pred_all))
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=df_full.index[-show_n:], y=df_full["Target_Price"].values[-show_n:], name="Actual", line=dict(color="#2196F3")))
    fig_reg.add_trace(go.Scatter(x=df_full.index[-show_n:], y=pred_all[-show_n:], name="XGBoost Forecast", line=dict(color="#4CAF50", dash="dash")))
    fig_reg.update_layout(title=f"Price Forecast  |  MAPE {reg_mape:.2f}%  RMSE ${reg_rmse:,.0f}", height=360, template="plotly_dark")
    st.plotly_chart(fig_reg, use_container_width=True)

    st.subheader("Feature Importance (SHAP)")
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(df_full[f_clf].tail(200))
    shap_df = pd.DataFrame({"Feature": f_clf, "Mean |SHAP|": np.abs(shap_vals).mean(axis=0)}).sort_values("Mean |SHAP|", ascending=True)
    fig_shap = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h", title="Crash Risk Drivers",
                      color="Mean |SHAP|", color_continuous_scale="Oranges")
    fig_shap.update_layout(height=420, template="plotly_dark")
    st.plotly_chart(fig_shap, use_container_width=True)

with tab2:
    st.subheader("AML Transaction Pattern Risk Scoring")
    st.info("Select a transaction scenario or adjust parameters manually.")

    SCENARIOS = {
        "Normal retail purchase": {"amount_usd":500,"frequency_7d":2,"cross_border":False,"round_amount":False,"rapid_layering":False,"smurfing":False,"mixer_usage":False},
        "High-frequency small transfers (structuring)": {"amount_usd":900,"frequency_7d":18,"cross_border":False,"round_amount":False,"rapid_layering":False,"smurfing":True,"mixer_usage":False},
        "Large single cross-border transfer": {"amount_usd":85000,"frequency_7d":1,"cross_border":True,"round_amount":True,"rapid_layering":False,"smurfing":False,"mixer_usage":False},
        "Rapid layering — multiple hops in 24h": {"amount_usd":22000,"frequency_7d":12,"cross_border":True,"round_amount":False,"rapid_layering":True,"smurfing":False,"mixer_usage":False},
        "Mixer + cross-border + round amount": {"amount_usd":50000,"frequency_7d":5,"cross_border":True,"round_amount":True,"rapid_layering":False,"smurfing":False,"mixer_usage":True},
    }

    selected = st.selectbox("Transaction Scenario", list(SCENARIOS.keys()))
    p = SCENARIOS[selected].copy()

    with st.expander("Override parameters manually"):
        col_a, col_b = st.columns(2)
        with col_a:
            p["amount_usd"] = st.number_input("Amount (USD)", 0, 1_000_000, int(p["amount_usd"]))
            p["frequency_7d"] = st.slider("Transactions / 7 days", 1, 50, p["frequency_7d"])
            p["cross_border"] = st.checkbox("Cross-border transfer", p["cross_border"])
            p["round_amount"] = st.checkbox("Round-number amount", p["round_amount"])
        with col_b:
            p["rapid_layering"] = st.checkbox("Rapid layering pattern", p["rapid_layering"])
            p["smurfing"] = st.checkbox("Smurfing / structuring", p["smurfing"])
            p["mixer_usage"] = st.checkbox("Crypto mixer usage", p["mixer_usage"])

    RULES = [
        ("R1 — Amount exceeds $10,000 (FINTRAC reporting threshold)", 20, p["amount_usd"]>10000),
        ("R2 — Amount exceeds $50,000 (large cash indicator)", 15, p["amount_usd"]>50000),
        ("R3 — High transaction frequency (>10 in 7 days)", 20, p["frequency_7d"]>10),
        ("R4 — Cross-border transfer (elevated jurisdiction risk)", 15, p["cross_border"]),
        ("R5 — Round-number amount (classic structuring indicator)", 10, p["round_amount"]),
        ("R6 — Rapid layering: multiple hops within 24 hours", 25, p["rapid_layering"]),
        ("R7 — Smurfing: high-frequency sub-threshold amounts", 25, p["smurfing"]),
        ("R8 — Crypto mixer usage detected (FATF red flag)", 30, p["mixer_usage"]),
    ]

    score = min(sum(pts for _, pts, hit in RULES if hit), 100)
    triggered = [(label, pts) for label, pts, hit in RULES if hit]
    tier, color = (("Low","#4CAF50") if score<20 else ("Medium","#FFC107") if score<45 else ("High","#FF5722") if score<70 else ("Critical","#B71C1C"))

    c1, c2, c3 = st.columns(3)
    c1.metric("AML Risk Score", f"{score} / 100")
    c2.metric("Risk Tier", tier)
    c3.metric("Rules Triggered", len(triggered))

    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=score, title={"text": "AML Risk Score"},
        gauge={"axis":{"range":[0,100]},"bar":{"color":color},
               "steps":[{"range":[0,20],"color":"#1B5E20"},{"range":[20,45],"color":"#F57F17"},
                        {"range":[45,70],"color":"#BF360C"},{"range":[70,100],"color":"#7B1FA2"}],
               "threshold":{"line":{"color":"white","width":3},"value":score}}))
    fig_gauge.update_layout(height=300, template="plotly_dark")
    st.plotly_chart(fig_gauge, use_container_width=True)

    if triggered:
        st.subheader("Triggered Risk Rules")
        for label, _ in triggered:
            st.error(label)
    else:
        st.success("No risk rules triggered — transaction pattern appears normal.")

    fig_rules = px.bar(x=[r[1] if r[2] else 0 for r in RULES], y=[r[0].split(" — ")[0] for r in RULES],
                       orientation="h", title="Score Contribution by Rule",
                       labels={"x":"Score points","y":""}, color=[r[1] if r[2] else 0 for r in RULES],
                       color_continuous_scale="Reds")
    fig_rules.update_layout(height=350, template="plotly_dark")
    st.plotly_chart(fig_rules, use_container_width=True)

with tab3:
    st.subheader("Regulatory & Ethical Framework")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### FATF Key Compliance Points")
        st.markdown("""
- **Recommendation 15**: Countries must assess ML/TF risks from virtual assets proportionately.
- **Explainability**: AI-based AML tools must produce outputs compliance officers can justify.
- **De-risking**: FATF warns against blanket exclusions of customer segments.
- **Collaborative analytics**: FATF (2021) identifies cross-institutional sharing as promising but legally constrained.
        """)
        st.markdown("### GDPR / Canadian PIPEDA")
        st.markdown("""
- Raw transaction data must not leave institutions without legal basis (GDPR Articles 5 & 25).
- Federated Learning shares only model updates, not underlying data.
- Canada's PIPEDA requires purpose limitation and data minimisation.
        """)
    with col2:
        st.markdown("### Model Limitations")
        st.warning("This dashboard is strictly academic. Outputs should not be interpreted as financial or legal advice.")
        st.markdown("""
- Training window (2021–2026) reflects specific market regimes.
- The 5% drawdown threshold affects recall/precision tradeoffs.
- AML scoring uses a simplified rule engine; production systems require graph analysis.
- Fairness audits are required before deployment of algorithmic compliance tools.
        """)
        st.markdown("### EU AI Act (2024)")
        st.markdown("""
AML systems are likely classified as **high-risk AI**, requiring:
- Transparency and logging of all decisions
- Human oversight before enforcement action
- Regular drift monitoring
- Right of individuals to seek explanation and redress
        """)
    st.divider()
    st.markdown("### References")
    st.markdown("""
- FATF (2021). *Opportunities and challenges of new technologies for AML/CFT.*
- McMahan et al. (2017). *Communication-efficient learning of deep networks.* PMLR.
- Regulation (EU) 2016/679 (GDPR); Regulation (EU) 2024/1689 (EU AI Act).
- Chen, N. (2026). *Federated Learning in AML: A Socio-technical Analysis.* Ontario Tech University.
    """)

