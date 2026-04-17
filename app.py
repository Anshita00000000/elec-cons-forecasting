"""
app.py — US Electricity Consumption Forecast Dashboard
=======================================================
Single-page Streamlit dashboard. No form inputs. No predict button.
All forecasts are pre-computed on startup from saved model files.

Sections:
  1. Header + KPI metrics
  2. Main interactive forecast chart (Plotly)
  3. Model comparison table

Run:
  streamlit run app.py
"""

import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Electricity Consumption Forecast",
    page_icon="⚡",
    layout="wide",
)

# ── Import shared utilities from models.py ─────────────────────────────────────
# Re-define the minimal helpers needed (avoids re-importing heavy model training)
TARGET   = "sales_total"
SEQ_LEN  = 12
TRAIN_END = "2023-12"
TEST_END  = "2026-03"


def _next_month_str(month_str: str, n: int = 1) -> str:
    dt = pd.to_datetime(month_str + "-01") + pd.DateOffset(months=n)
    return dt.strftime("%Y-%m")


def _build_next_row(history_df, next_month, temp_baselines, include_temp):
    h     = history_df.sort_values("month").reset_index(drop=True)
    total = h[TARGET].values
    n     = len(total)
    row: dict = {}
    for lag in [1, 2, 3, 6, 12, 24]:
        row[f"{TARGET}_lag_{lag}"] = total[-lag] if n >= lag else np.nan
    row[f"{TARGET}_roll3"]     = float(np.mean(total[-3:]))          if n >= 3  else np.nan
    row[f"{TARGET}_roll12"]    = float(np.mean(total[-12:]))         if n >= 12 else np.nan
    row[f"{TARGET}_roll3_std"] = float(np.std(total[-3:], ddof=1))   if n >= 3  else np.nan
    for sector in ["sales_COM", "sales_IND", "sales_RES"]:
        sv = h[sector].values
        ns = len(sv)
        for lag in [1, 2, 3, 6, 12, 24]:
            row[f"{sector}_lag_{lag}"] = sv[-lag] if ns >= lag else np.nan
        row[f"{sector}_roll3"]     = float(np.mean(sv[-3:]))         if ns >= 3  else np.nan
        row[f"{sector}_roll12"]    = float(np.mean(sv[-12:]))        if ns >= 12 else np.nan
        row[f"{sector}_roll3_std"] = float(np.std(sv[-3:], ddof=1)) if ns >= 3  else np.nan
    month_num = int(next_month[5:7])
    year      = int(next_month[:4])
    row["month_num"] = month_num
    row["year"]      = year
    row["quarter"]   = (month_num - 1) // 3 + 1
    if include_temp and temp_baselines:
        b = temp_baselines[str(month_num)]
        row["temp_avg_f"]          = b["temp_avg_f"]
        row["heating_degree_days"] = b["hdd"]
        row["cooling_degree_days"] = b["cdd"]
    return row


def _project_sectors(pred_total, recent_df):
    shares = {}
    for s in ["sales_COM", "sales_IND", "sales_RES"]:
        ratio = (recent_df[s] / recent_df[TARGET]).mean()
        shares[s] = float(pred_total * ratio)
    return shares


def recursive_forecast_ml(model, history_df, n_steps, feat_cols,
                           temp_baselines, scaler=None, include_temp=True):
    h = history_df[["month", TARGET, "sales_COM", "sales_IND", "sales_RES"]].copy()
    h = h.sort_values("month").reset_index(drop=True)
    last_month = h["month"].iloc[-1]
    preds = []
    for step in range(1, n_steps + 1):
        nxt = _next_month_str(last_month, step)
        row = _build_next_row(h, nxt, temp_baselines, include_temp)
        X_new = np.array([[row.get(c, 0.0) for c in feat_cols]], dtype=float)
        if scaler is not None:
            X_new = scaler.transform(X_new)
        pred = float(model.predict(X_new)[0])
        preds.append(pred)
        sectors = _project_sectors(pred, h.tail(6))
        h = pd.concat([h, pd.DataFrame([{"month": nxt, TARGET: pred, **sectors}])],
                      ignore_index=True)
    return np.array(preds)


def recursive_forecast_rnn(model, history_df, n_steps, feat_cols, X_scaler, y_scaler):
    h = history_df[["month", TARGET, "sales_COM", "sales_IND", "sales_RES"]].copy()
    h = h.sort_values("month").reset_index(drop=True)
    tail = h.tail(SEQ_LEN + 24).copy()
    window_rows = []
    for i in range(len(tail) - SEQ_LEN, len(tail)):
        ctx = tail.iloc[:i + 1]
        next_m = tail["month"].iloc[i]
        row = _build_next_row(ctx.iloc[:-1], next_m, None, False)
        window_rows.append([row.get(c, 0.0) for c in feat_cols])
    window = list(X_scaler.transform(np.array(window_rows[-SEQ_LEN:], dtype=float)))
    last_month = h["month"].iloc[-1]
    preds = []
    for step in range(1, n_steps + 1):
        nxt = _next_month_str(last_month, step)
        X_seq = np.array(window, dtype=float).reshape(1, SEQ_LEN, len(feat_cols))
        pred_scaled = float(model.predict(X_seq, verbose=0)[0, 0])
        pred_actual = float(y_scaler.inverse_transform([[pred_scaled]])[0, 0])
        preds.append(pred_actual)
        sectors = _project_sectors(pred_actual, h.tail(6))
        new_row_d = {"month": nxt, TARGET: pred_actual, **sectors}
        h = pd.concat([h, pd.DataFrame([new_row_d])], ignore_index=True)
        feat_row = _build_next_row(h.iloc[:-1], nxt, None, False)
        fv_scaled = X_scaler.transform(
            np.array([[feat_row.get(c, 0.0) for c in feat_cols]], dtype=float)
        )[0]
        window.pop(0)
        window.append(fv_scaled)
    return np.array(preds)


# ===========================================================================
# Data loading (cached)
# ===========================================================================

@st.cache_resource
def load_all():
    """Load all model artefacts and data once. Results are cached."""
    master   = pd.read_csv("data/master_df.csv").sort_values("month").reset_index(drop=True)
    results  = pd.read_csv("outputs/model_results.csv").set_index("Model")

    with open("outputs/monthly_temp_baselines.json") as f:
        temp_baselines = json.load(f)

    with open("models/sarima_model.pkl", "rb") as f:
        sarima_model = pickle.load(f)

    with open("models/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    with open("models/linear_model.pkl", "rb") as f:
        lr_model = pickle.load(f)

    with open("models/feature_scaler.pkl", "rb") as f:
        lr_scaler = pickle.load(f)

    with open("models/lstm_scaler.pkl", "rb") as f:
        lstm_scalers = pickle.load(f)
    X_scaler_lstm = lstm_scalers["X"]
    y_scaler_lstm = lstm_scalers["y"]

    lstm_model = tf.keras.models.load_model("models/lstm_model.keras")
    gru_model  = tf.keras.models.load_model("models/gru_model.keras")

    feat_wt = pd.read_csv("data/features_with_temp.csv")
    feat_lo = pd.read_csv("data/features_lag_only.csv")
    feat_cols_wt = [c for c in feat_wt.columns if c not in ("month", TARGET)]
    feat_cols_lo = [c for c in feat_lo.columns if c not in ("month", TARGET)]

    return dict(
        master=master, results=results, temp_baselines=temp_baselines,
        sarima_model=sarima_model,
        xgb_model=xgb_model, lr_model=lr_model, lr_scaler=lr_scaler,
        lstm_model=lstm_model, gru_model=gru_model,
        X_scaler_lstm=X_scaler_lstm, y_scaler_lstm=y_scaler_lstm,
        feat_cols_wt=feat_cols_wt, feat_cols_lo=feat_cols_lo,
    )


@st.cache_resource
def compute_forward_forecasts(_artefacts: dict, n_steps: int = 12):
    """
    Generate forward forecasts from last actual data point (2026-03)
    for the best short-term and best long-term model.
    Uses all available data (train + test) as history.
    """
    ar   = _artefacts
    res  = ar["results"]
    mst  = ar["master"]

    best_short = res["MAPE_h3"].idxmin()
    best_long  = res["MAPE_h12"].idxmin()

    def _forecast(name):
        if name == "SARIMA":
            # Update SARIMA with test-period actuals then forecast forward
            test_master = mst[(mst["month"] >= "2024-01") & (mst["month"] <= TEST_END)]
            updated = ar["sarima_model"].update(test_master[TARGET].values)
            fp, fp_ci = updated.predict(n_steps, return_conf_int=True, alpha=0.05)
            rmse = float(res.loc[name, "RMSE_h12"])
            ci = np.column_stack([fp - 1.96 * rmse, fp + 1.96 * rmse])
            return np.array(fp), ci
        rnn = {"LSTM": ar["lstm_model"], "GRU": ar["gru_model"]}
        if name in rnn:
            fp = recursive_forecast_rnn(
                rnn[name], mst, n_steps,
                ar["feat_cols_lo"], ar["X_scaler_lstm"], ar["y_scaler_lstm"],
            )
        else:
            sk = {"LinearRegression": ar["lr_scaler"]}.get(name)
            models_sk = {"XGBoost": ar["xgb_model"], "LinearRegression": ar["lr_model"]}
            fp = recursive_forecast_ml(
                models_sk[name], mst, n_steps, ar["feat_cols_wt"],
                ar["temp_baselines"], scaler=sk, include_temp=True,
            )
        rmse = float(res.loc[name, "RMSE_h12"])
        ci   = np.array([[fp[i] - 1.96 * rmse * np.sqrt(i + 1),
                          fp[i] + 1.96 * rmse * np.sqrt(i + 1)]
                         for i in range(n_steps)])
        return fp, ci

    fp_short, ci_short = _forecast(best_short)
    fp_long,  ci_long  = _forecast(best_long)

    return dict(
        best_short=best_short, best_long=best_long,
        fp_short=fp_short, ci_short=ci_short,
        fp_long=fp_long,   ci_long=ci_long,
    )


# ===========================================================================
# App
# ===========================================================================

artefacts = load_all()
forecasts = compute_forward_forecasts(artefacts)

master  = artefacts["master"]
results = artefacts["results"]

# ── Derived values for KPI cards ───────────────────────────────────────────────
latest_row = master.iloc[-1]
latest_month = latest_row["month"]
latest_total = latest_row[TARGET]

year_ago_month = _next_month_str(latest_month, -12)
year_ago_row   = master[master["month"] == year_ago_month]
year_ago_val   = float(year_ago_row[TARGET].values[0]) if len(year_ago_row) else None
yoy_pct        = ((latest_total - year_ago_val) / year_ago_val * 100
                  if year_ago_val else None)

best_mape_h3  = float(results["MAPE_h3"].min())
best_mape_h12 = float(results["MAPE_h12"].min())

# Future dates
future_dates = pd.DatetimeIndex([
    pd.to_datetime(TEST_END + "-01") + pd.DateOffset(months=i + 1)
    for i in range(12)
])

# ===========================================================================
# SECTION 1: Header + KPI cards
# ===========================================================================

st.title("US Electricity Consumption Forecast")
st.markdown("**National monthly sales — Residential, Commercial, Industrial**")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label=f"Latest Total Sales ({latest_month})",
        value=f"{latest_total:,.0f} million kWh",
    )
with col2:
    if yoy_pct is not None:
        sign = "+" if yoy_pct >= 0 else ""
        month_lbl = pd.to_datetime(year_ago_month + "-01").strftime("%b %Y")
        st.metric(
            label="YoY Change",
            value=f"{sign}{yoy_pct:.1f}%",
            delta=f"vs {month_lbl}",
        )
    else:
        st.metric(label="YoY Change", value="N/A")
with col3:
    st.metric(
        label="Best MAPE (3-month)",
        value=f"{best_mape_h3:.1f}%",
        delta=f"{forecasts['best_short']}",
        delta_color="off",
    )
with col4:
    st.metric(
        label="Best MAPE (12-month)",
        value=f"{best_mape_h12:.1f}%",
        delta=f"{forecasts['best_long']}",
        delta_color="off",
    )

# ===========================================================================
# SECTION 2: Main forecast chart (Plotly)
# ===========================================================================

st.markdown("---")
st.subheader("Forecast Chart")

view_col, check_col = st.columns([3, 1])
with view_col:
    view = st.radio(
        "View range",
        ["Last 5 years + forecast", "Full history + forecast"],
        horizontal=True, label_visibility="collapsed",
    )
with check_col:
    show_sectors = st.checkbox("Show sector breakdown", value=False)

hist_start = (
    _next_month_str(TEST_END, -60) if view == "Last 5 years + forecast"
    else master["month"].min()
)
plot_hist = master[master["month"] >= hist_start].copy()
hist_dates = pd.to_datetime(plot_hist["month"] + "-01")

fig = go.Figure()

# Historical total
fig.add_trace(go.Scatter(
    x=hist_dates, y=plot_hist[TARGET],
    mode="lines", name="Historical Total",
    line=dict(color="#1565C0", width=2.5),
))

# Sector lines (optional)
if show_sectors:
    sector_cfg = [
        ("sales_RES", "Residential",  "#43A047"),
        ("sales_COM", "Commercial",   "#1E88E5"),
        ("sales_IND", "Industrial",   "#FB8C00"),
    ]
    for col, label, color in sector_cfg:
        fig.add_trace(go.Scatter(
            x=hist_dates, y=plot_hist[col],
            mode="lines", name=label,
            line=dict(color=color, width=1.3, dash="dot"),
            opacity=0.75,
        ))

# Vertical line: last actual data point
fig.add_vline(
    x=pd.to_datetime(TEST_END + "-01").timestamp() * 1000,
    line_dash="dash", line_color="gray", line_width=1.5,
    annotation_text=f"Last actual: {TEST_END}",
    annotation_position="top left",
    annotation_font_color="gray",
)

# Short-term forecast (h1–h3)
fp_s = forecasts["fp_short"]
ci_s = forecasts["ci_short"]
fig.add_trace(go.Scatter(
    x=future_dates[:3], y=fp_s[:3],
    mode="lines+markers", name=f"Short-term ({forecasts['best_short']}, h1–3)",
    line=dict(color="#E53935", width=2),
    marker=dict(size=6),
))
fig.add_trace(go.Scatter(
    x=list(future_dates[:3]) + list(future_dates[:3])[::-1],
    y=list(ci_s[:3, 1]) + list(ci_s[:3, 0])[::-1],
    fill="toself", fillcolor="rgba(229,57,53,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip", showlegend=False,
))

# Long-term forecast (h6–h12)
fp_l = forecasts["fp_long"]
ci_l = forecasts["ci_long"]
fig.add_trace(go.Scatter(
    x=future_dates[:6], y=fp_l[:6],
    mode="lines+markers", name=f"Long-term ({forecasts['best_long']}, h6)",
    line=dict(color="#7B1FA2", width=2, dash="dash"),
    marker=dict(size=5),
))
fig.add_trace(go.Scatter(
    x=future_dates[:12], y=fp_l[:12],
    mode="lines", name=f"Long-term ({forecasts['best_long']}, h12)",
    line=dict(color="#7B1FA2", width=1.5, dash="dot"),
    opacity=0.8,
))
fig.add_trace(go.Scatter(
    x=list(future_dates[:12]) + list(future_dates[:12])[::-1],
    y=list(ci_l[:12, 1]) + list(ci_l[:12, 0])[::-1],
    fill="toself", fillcolor="rgba(123,31,162,0.10)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip", showlegend=False,
))

fig.update_layout(
    xaxis_title="Month",
    yaxis_title="million kWh",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=480,
    margin=dict(l=50, r=20, t=40, b=50),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
    yaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
)

st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# SECTION 3: Model comparison table
# ===========================================================================

st.markdown("---")
st.subheader("Model Performance — Holdout Period (Jan 2024 – Mar 2026)")

display_cols = {
    "MAPE_h1":   "MAPE 1m (%)",
    "MAPE_h3":   "MAPE 3m (%)",
    "MAPE_h6":   "MAPE 6m (%)",
    "MAPE_h12":  "MAPE 12m (%)",
    "R2_h12":    "R² 12m",
    "DirAcc_h12":"Dir.Acc 12m (%)",
}

table_df = results[list(display_cols.keys())].rename(columns=display_cols).reset_index()

def highlight_best(col_data):
    """Green background for best (min MAPE, max R2) in each column."""
    styles = [""] * len(col_data)
    label = col_data.name
    if "MAPE" in label:
        best_idx = col_data.values[1:].argmin() + 1  # skip header; 0-indexed in data rows
        # We need to handle this differently since it's for styler
        best_val = col_data.values[1:].min()
        return [
            "background-color: #c8e6c9; font-weight: bold"
            if (i > 0 and col_data.iloc[i] == best_val) else ""
            for i in range(len(col_data))
        ]
    elif label == "R² 12m":
        best_val = col_data.values[1:].max()
        return [
            "background-color: #c8e6c9; font-weight: bold"
            if (i > 0 and col_data.iloc[i] == best_val) else ""
            for i in range(len(col_data))
        ]
    return styles


# Build styler
numeric_cols = [c for c in table_df.columns if c != "Model"]

styled = (
    table_df.style
    .hide(axis="index")
    .format({c: "{:.3f}" for c in numeric_cols})
    .apply(
        lambda col: [
            "background-color: #c8e6c9; font-weight: bold"
            if (col.name != "Model"
                and col.notna().all()
                and (
                    ("MAPE" in col.name and val == col.min())
                    or ("R²" in col.name and val == col.max())
                    or ("Dir.Acc" in col.name and val == col.max())
                )
            )
            else ""
            for val in col
        ],
        axis=0,
    )
)

st.dataframe(styled, use_container_width=True, height=220)

st.caption(
    "Lower MAPE = better. Higher R² = better. Direction accuracy >65% is good."
)

best_short = forecasts["best_short"]
best_long  = forecasts["best_long"]
mape_short = float(results.loc[best_short, "MAPE_h3"])
mape_long  = float(results.loc[best_long,  "MAPE_h12"])

st.markdown(
    f"**Best short-term model:** {best_short} (MAPE: {mape_short:.1f}% at 3 months)  \n"
    f"**Best long-term model:** {best_long} (MAPE: {mape_long:.1f}% at 12 months)"
)
