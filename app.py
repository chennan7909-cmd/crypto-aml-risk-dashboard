"""
Crypto Risk & AML Monitoring Dashboard
Nan Chen | Ontario Tech University
Streamlit Cloud deployment version
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
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
import shap

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Risk & AML Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.title("Crypto Risk & AML Monitoring Dashboard")
st.caption(
    "Nan Chen · Ontario Tech University · "
    "LSTM Price Forecasting + XGBoost Crash Detection + AML Pattern Scoring"
)
st.divider()

# ═════════════════════════════════════════════════════════════
# DATA & MODEL — cached so they only run once per session
# ═════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Downloading market data...")
def load_raw(ticker):
    df = yf.download(ticker, start="2021-01-01", end="2026-03-31",
                     auto_adjust=True, progress=False)
    df.columns = ["Close", "High", "Low", "Open", "Volume"]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


@st.cache_data(show_spinner="Engineering features...")
def add_features(_df):
    df = _df.copy()
    df["MA7"]       = df["Close"].rolling(7).mean()
    df["MA21"]      = df["Close"].rolling(21).mean()
    df["MA_Ratio"]  = df["MA7"] / df["MA21"]
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]        = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Position"] = (df["Close"] - mid) / (2 * std + 1e-9)
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd  = ema12 - ema26
    df["MACD_Hist"]  = macd - macd.ewm(span=9).mean()
    df["Return"]     = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(7).std()
    df["Next_Return"]  = df["Close"].pct_change().shift(-1)
    df["Risk_Event"]   = (df["Next_Return"] < -0.05).astype(int)
    df["Target_Price"] = df["Close"].shift(-1)
    return df.dropna()


@st.cache_data(show_spinner="Adding cross-asset BTC signals...")
def add_cross_asset(_sol_df, _btc_df):
    sol = _sol_df.copy()
    for lag in [1, 3, 7]:
        for col in ["Return", "Volatility", "RSI", "MACD_Hist", "MA_Ratio"]:
            sol[f"BTC_{col}_lag{lag}"] = _btc_df[col].shift(lag)
    return sol.dropna()


FEATURES_BASE = ["Close", "Volume", "Volatility", "MA7",
                 "RSI", "BB_Position", "MACD_Hist", "MA_Ratio"]
SEQ_LEN = 30


@st.cache_resource(show_spinner="Training XGBoost model...")
def train_xgb(df_key, features, df):
    X = df[features].values
    y = df["Risk_Event"].values
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    ratio = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1)
    clf = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        scale_pos_weight=ratio, eval_metric="logloss",
        random_state=42, tree_method="hist"
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba > 0.30).astype(int)
    auc = roc_auc_score(y_te, proba)
    f1  = f1_score(y_te, preds, zero_division=0)
    return clf, round(auc, 4), round(f1, 4)


@st.cache_resource(show_spinner="Training LSTM model (this takes ~3 min)...")
def train_lstm(df_key, df):
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping

    features = FEATURES_BASE
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(df[features])
    y_sc = scaler_y.fit_transform(df[["Target_Price"]])

    Xs, ys = [], []
    for i in range(SEQ_LEN, len(X_sc)):
        Xs.append(X_sc[i - SEQ_LEN:i])
        ys.append(y_sc[i])
    Xs, ys = np.array(Xs), np.array(ys)

    split = int(len(Xs) * 0.8)
    X_tr, X_te = Xs[:split], Xs[split:]
    y_tr, y_te = ys[:split], ys[split:]

    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(SEQ_LEN, len(features))),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(X_tr, y_tr, epochs=30, batch_size=32,
              callbacks=[es], verbose=0)

    pred_sc = model.predict(X_te, verbose=0)
    pred    = scaler_y.inverse_transform(pred_sc).flatten()
    actual  = scaler_y.inverse_transform(y_te).flatten()
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-9))) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))

    return model, scaler_X, scaler_y, round(mape, 2), round(rmse, 0)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    asset     = st.selectbox("Select Asset", ["BTC", "SOL"])
    window    = st.slider("Lookback window (days)", 60, 365, 180)
    threshold = st.slider("Crash alert threshold", 0.10, 0.50, 0.30, 0.05)
    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- LSTM: 3-layer, EarlyStopping (patience=5)")
    st.markdown("- XGBoost: walk-forward, threshold = 0.30")
    st.markdown("- SOL: +15 cross-asset BTC lag features")
    st.markdown("- Data: Jan 2021 – Mar 2026 (yfinance)")

# ── Load & prepare data ───────────────────────────────────────
btc_raw = load_raw("BTC-USD")
sol_raw = load_raw("SOL-USD")
btc_full = add_features(btc_raw)
sol_base = add_features(sol_raw)
sol_full = add_cross_asset(sol_base, btc_full)

if asset == "BTC":
    df_full   = btc_full
    features_clf = FEATURES_BASE
else:
    df_full   = sol_full
    features_clf = FEATURES_BASE + [c for c in sol_full.columns
                                     if c.startswith("BTC_")]

df = df_full.tail(window).copy()

# ── Train models (cached) ─────────────────────────────────────
clf, xgb_auc, xgb_f1 = train_xgb(asset, features_clf, df_full)
lstm_m, scaler_X, scaler_y, lstm_mape, lstm_rmse = train_lstm(asset, df_full)

# ═════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "Price Risk Dashboard",
    "AML Risk Scoring",
    "Regulatory Context"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Price Risk Dashboard
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"{asset} — Price Risk Dashboard")

    # XGBoost probabilities
    crash_proba = clf.predict_proba(df[features_clf].values)[:, 1]
    df = df.copy()
    df["Crash_Prob"] = crash_proba
    df["Risk_Tier"]  = pd.cut(
        crash_proba,
        bins=[0, 0.20, 0.35, 0.55, 1.0],
        labels=["Low", "Medium", "High", "Critical"]
    )

    # KPIs
    latest_prob  = float(crash_proba[-1])
    latest_tier  = str(df["Risk_Tier"].iloc[-1])
    latest_price = float(df["Close"].iloc[-1])
    high_days    = int((crash_proba > threshold).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price",      f"${latest_price:,.0f}")
    c2.metric("Crash Probability",  f"{latest_prob:.1%}")
    c3.metric("Risk Tier",          latest_tier)
    c4.metric(f"High-Risk Days (>{threshold:.0%})",
              f"{high_days} / {len(df)}")

    st.divider()

    # Price chart with crash alert markers
    mask = crash_proba > threshold
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close Price", line=dict(color="#2196F3", width=1.5)
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index[mask], y=df["Close"].values[mask],
        mode="markers",
        name=f"Crash Alert (>{threshold:.0%})",
        marker=dict(color="red", size=7, symbol="x")
    ))
    fig_price.update_layout(
        title=f"{asset} Price with Crash Alerts",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=380, template="plotly_dark"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Crash probability time series
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(
        x=df.index, y=crash_proba,
        fill="tozeroy", name="Crash Probability",
        line=dict(color="#FF5722")
    ))
    fig_prob.add_hline(
        y=threshold, line_dash="dash", line_color="yellow",
        annotation_text=f"Alert threshold ({threshold:.0%})"
    )
    fig_prob.update_layout(
        title=f"XGBoost Crash Probability  |  AUC {xgb_auc}  F1 {xgb_f1}",
        yaxis=dict(tickformat=".0%"),
        height=280, template="plotly_dark"
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # LSTM forecast
    st.subheader("LSTM Next-Day Price Forecast")
    X_sc_full = scaler_X.transform(df_full[FEATURES_BASE])
    n = min(120, len(X_sc_full) - SEQ_LEN)
    Xs = np.array([X_sc_full[i - SEQ_LEN:i]
                   for i in range(SEQ_LEN, SEQ_LEN + n)])
    pred_sc  = lstm_m.predict(Xs, verbose=0)
    pred_usd = scaler_y.inverse_transform(pred_sc).flatten()
    actual   = df_full["Close"].values[SEQ_LEN:SEQ_LEN + n]
    idx      = df_full.index[SEQ_LEN:SEQ_LEN + n]

    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(
        x=idx, y=actual, name="Actual",
        line=dict(color="#2196F3")
    ))
    fig_lstm.add_trace(go.Scatter(
        x=idx, y=pred_usd, name="LSTM Forecast",
        line=dict(color="#4CAF50", dash="dash")
    ))
    fig_lstm.update_layout(
        title=f"LSTM Forecast  |  MAPE {lstm_mape:.2f}%  RMSE ${lstm_rmse:,.0f}",
        height=360, template="plotly_dark"
    )
    st.plotly_chart(fig_lstm, use_container_width=True)

    # SHAP
    st.subheader("Feature Importance (SHAP — XGBoost)")
    sample = df_full[features_clf].tail(300)
    explainer = shap.TreeExplainer(clf)
    shap_vals  = explainer.shap_values(sample)
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature":    features_clf,
        "Mean |SHAP|": mean_shap
    }).sort_values("Mean |SHAP|", ascending=True)

    fig_shap = px.bar(
        shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
        title="Default Risk Drivers",
        color="Mean |SHAP|", color_continuous_scale="Oranges"
    )
    fig_shap.update_layout(height=380, template="plotly_dark")
    st.plotly_chart(fig_shap, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — AML Risk Scoring
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("AML Transaction Pattern Risk Scoring")
    st.info(
        "Select a transaction scenario or adjust parameters manually. "
        "The scoring engine applies FATF typology indicators and outputs "
        "a risk tier, score, and triggered rule list."
    )

    SCENARIOS = {
        "Normal retail purchase": {
            "amount_usd": 500, "frequency_7d": 2,
            "cross_border": False, "round_amount": False,
            "rapid_layering": False, "smurfing": False, "mixer_usage": False
        },
        "High-frequency small transfers (structuring)": {
            "amount_usd": 900, "frequency_7d": 18,
            "cross_border": False, "round_amount": False,
            "rapid_layering": False, "smurfing": True, "mixer_usage": False
        },
        "Large single cross-border transfer": {
            "amount_usd": 85000, "frequency_7d": 1,
            "cross_border": True, "round_amount": True,
            "rapid_layering": False, "smurfing": False, "mixer_usage": False
        },
        "Rapid layering — multiple hops in 24h": {
            "amount_usd": 22000, "frequency_7d": 12,
            "cross_border": True, "round_amount": False,
            "rapid_layering": True, "smurfing": False, "mixer_usage": False
        },
        "Mixer + cross-border + round amount": {
            "amount_usd": 50000, "frequency_7d": 5,
            "cross_border": True, "round_amount": True,
            "rapid_layering": False, "smurfing": False, "mixer_usage": True
        },
    }

    selected = st.selectbox("Transaction Scenario", list(SCENARIOS.keys()))
    p = SCENARIOS[selected].copy()

    with st.expander("Override parameters manually"):
        col_a, col_b = st.columns(2)
        with col_a:
            p["amount_usd"]    = st.number_input("Amount (USD)", 0, 1_000_000,
                                                  int(p["amount_usd"]))
            p["frequency_7d"] = st.slider("Transactions / 7 days",
                                           1, 50, p["frequency_7d"])
            p["cross_border"] = st.checkbox("Cross-border transfer",
                                             p["cross_border"])
            p["round_amount"] = st.checkbox("Round-number amount",
                                             p["round_amount"])
        with col_b:
            p["rapid_layering"] = st.checkbox("Rapid layering pattern",
                                               p["rapid_layering"])
            p["smurfing"]       = st.checkbox("Smurfing / structuring",
                                               p["smurfing"])
            p["mixer_usage"]    = st.checkbox("Crypto mixer usage",
                                               p["mixer_usage"])

    # Scoring engine
    RULES = [
        ("R1 — Amount exceeds $10,000 (FINTRAC reporting threshold)",
         20, p["amount_usd"] > 10000),
        ("R2 — Amount exceeds $50,000 (large cash indicator)",
         15, p["amount_usd"] > 50000),
        ("R3 — High transaction frequency (>10 in 7 days)",
         20, p["frequency_7d"] > 10),
        ("R4 — Cross-border transfer (elevated jurisdiction risk)",
         15, p["cross_border"]),
        ("R5 — Round-number amount (classic structuring indicator)",
         10, p["round_amount"]),
        ("R6 — Rapid layering: multiple hops within 24 hours",
         25, p["rapid_layering"]),
        ("R7 — Smurfing: high-frequency sub-threshold amounts",
         25, p["smurfing"]),
        ("R8 — Crypto mixer usage detected (FATF red flag)",
         30, p["mixer_usage"]),
    ]

    score = min(sum(pts for _, pts, hit in RULES if hit), 100)
    triggered = [(label, pts) for label, pts, hit in RULES if hit]

    tier, color = (
        ("Low",      "#4CAF50") if score < 20 else
        ("Medium",   "#FFC107") if score < 45 else
        ("High",     "#FF5722") if score < 70 else
        ("Critical", "#B71C1C")
    )

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("AML Risk Score",  f"{score} / 100")
    c2.metric("Risk Tier",       tier)
    c3.metric("Rules Triggered", len(triggered))

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "AML Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  20], "color": "#1B5E20"},
                {"range": [20, 45], "color": "#F57F17"},
                {"range": [45, 70], "color": "#BF360C"},
                {"range": [70,100], "color": "#7B1FA2"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "value": score}
        }
    ))
    fig_gauge.update_layout(height=300, template="plotly_dark")
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Triggered rules
    if triggered:
        st.subheader("Triggered Risk Rules")
        for label, _ in triggered:
            st.error(label)
    else:
        st.success("No risk rules triggered — transaction pattern appears normal.")

    # Score decomposition
    rule_labels = [r[0].split(" — ")[0] for r in RULES]
    rule_scores = [r[1] if r[2] else 0 for r in RULES]
    fig_rules = px.bar(
        x=rule_scores, y=rule_labels, orientation="h",
        title="Score Contribution by Rule",
        labels={"x": "Score points", "y": ""},
        color=rule_scores, color_continuous_scale="Reds"
    )
    fig_rules.update_layout(height=350, template="plotly_dark")
    st.plotly_chart(fig_rules, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — Regulatory Context
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Regulatory & Ethical Framework")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### FATF Key Compliance Points")
        st.markdown("""
        - **Recommendation 15**: Countries must assess ML/TF risks from
          virtual assets and apply AML/CFT measures proportionately.
        - **Explainability**: AI-based AML tools must produce outputs
          that compliance officers can interpret and justify to regulators.
        - **De-risking risk**: Overly sensitive models may flag legitimate
          users — FATF warns against blanket exclusions of customer segments.
        - **Collaborative analytics**: FATF (2021) identifies cross-
          institutional data sharing as promising but legally constrained.
        """)

        st.markdown("### GDPR / Canadian PIPEDA")
        st.markdown("""
        - Raw transaction data must not leave any institution without
          explicit legal basis (GDPR Articles 5 & 25).
        - Federated Learning is a privacy-preserving alternative:
          only model updates are shared, not underlying data.
        - Canada's PIPEDA requires purpose limitation and data
          minimisation in all financial data processing workflows.
        """)

    with col2:
        st.markdown("### Model Limitations")
        st.warning("This dashboard is strictly academic. "
                   "Outputs should not be interpreted as financial or legal advice.")
        st.markdown("""
        - Training window (2021–2026) covers specific market regimes;
          performance may degrade in structurally different conditions.
        - The 5% drawdown threshold for crash classification is
          arbitrary — different thresholds produce different recall/precision tradeoffs.
        - AML scoring uses a simplified rule engine; a production system
          requires transaction graph analysis and network-level features.
        - Algorithmic tools may disproportionately flag certain user
          segments — fairness audits are required before deployment.
        """)

        st.markdown("### EU AI Act (2024)")
        st.markdown("""
        AML systems powered by ML are likely classified as
        **high-risk AI** under the EU AI Act, requiring:
        - Transparency and logging of all model decisions
        - Human oversight before any enforcement action
        - Regular drift monitoring and performance review
        - Right of affected individuals to seek explanation and redress
        """)

    st.divider()
    st.markdown("### References")
    st.markdown("""
    - FATF (2021). *Opportunities and challenges of new technologies for AML/CFT.*
    - McMahan et al. (2017). *Communication-efficient learning of deep networks.* PMLR.
    - Regulation (EU) 2016/679 (GDPR), Articles 5 and 25.
    - Regulation (EU) 2024/1689 (EU AI Act).
    - Chen, N. (2026). *Federated Learning in AML: A Socio-technical Analysis.*
      Ontario Tech University.
    - Winner, L. (1980). Do artifacts have politics? *Daedalus, 109*(1), 121–136.
    """)

