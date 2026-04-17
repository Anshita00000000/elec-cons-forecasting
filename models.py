"""
models.py — US Electricity Consumption Forecasting
====================================================
Trains and evaluates 5 models:
  1. SARIMA (pmdarima auto_arima)
  2. XGBoost (GridSearchCV with TimeSeriesSplit)
  3. Linear Regression (StandardScaler baseline)
  4. LSTM (Keras, lag-only features)
  5. GRU  (Keras, lag-only features)

Evaluation:
  - Holdout: 2024-01 to 2026-03 (27 months, never touched during training)
  - Horizons: h1, h3, h6, h12 — metrics over first 1/3/6/12 test months
  - MAPE (primary), RMSE, MAE, R², Direction Accuracy

Outputs:
  sarima_model.pkl, xgboost_model.pkl, linear_model.pkl,
  feature_scaler.pkl, lstm_scaler.pkl, lstm_model.keras, gru_model.keras,
  monthly_temp_baselines.json, model_results.csv,
  evaluation_chart.png, forecast_chart.png

Constraints:
  - TimeSeriesSplit(n_splits=5) for all CV — never KFold or shuffle
  - Test set touched only during final evaluation
  - Temperature for future months from monthly_temp_baselines.json (train data only)
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score

import xgboost as xgb
import pmdarima as pm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ── Global config ──────────────────────────────────────────────────────────────
TRAIN_END  = "2023-12"
TEST_START = "2024-01"
TEST_END   = "2026-03"
TARGET     = "sales_total"
SEQ_LEN    = 12
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ===========================================================================
# Utility: metrics
# ===========================================================================

def compute_metrics(actual: np.ndarray, pred: np.ndarray,
                    prior: np.ndarray) -> dict:
    """
    actual, pred, prior : 1-D arrays of equal length.
    prior[i] = actual[i-1] (the month before actual[i]) for direction accuracy.
    """
    actual = np.asarray(actual, float)
    pred   = np.asarray(pred,   float)
    prior  = np.asarray(prior,  float)

    mape     = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    rmse     = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae      = float(np.mean(np.abs(actual - pred)))
    r2       = float(r2_score(actual, pred))
    dir_act  = np.sign(actual - prior)
    dir_pred = np.sign(pred   - prior)
    dir_acc  = float(np.mean(dir_act == dir_pred) * 100)

    return {"MAPE": mape, "RMSE": rmse, "MAE": mae, "R2": r2, "DirAcc": dir_acc}


def horizon_metrics(actual_test, pred_all, prior_last_train,
                    h: int) -> dict:
    """
    Compute metrics using the first h test months.
    prior_last_train : actual value at the last training month (for direction at t=0).
    """
    a = actual_test[:h]
    p = pred_all[:h]
    # prior[0] = value at last training month; prior[i>0] = actual[i-1]
    pr = np.concatenate([[prior_last_train], actual_test[:h - 1]])
    return compute_metrics(a, p, pr)


# ===========================================================================
# Utility: month arithmetic
# ===========================================================================

def _next_month_str(month_str: str, n: int = 1) -> str:
    dt = pd.to_datetime(month_str + "-01") + pd.DateOffset(months=n)
    return dt.strftime("%Y-%m")


# ===========================================================================
# Utility: monthly temperature baselines
# ===========================================================================

def build_temp_baselines(train_df: pd.DataFrame) -> dict:
    """
    Monthly averages of temp_avg_f, HDD, CDD from TRAINING data only.
    Returns dict keyed by str month number 1–12.
    """
    baselines = {}
    for m in range(1, 13):
        sub = train_df[train_df["month_num"] == m]
        baselines[str(m)] = {
            "temp_avg_f": round(float(sub["temp_avg_f"].mean()), 4),
            "hdd":        round(float(sub["heating_degree_days"].mean()), 4),
            "cdd":        round(float(sub["cooling_degree_days"].mean()), 4),
        }
    return baselines


# ===========================================================================
# Utility: build a single feature row for the NEXT month
# ===========================================================================

def build_next_row(history_df: pd.DataFrame, next_month: str,
                   temp_baselines: dict | None, include_temp: bool) -> dict:
    """
    Compute all lag/rolling/calendar/temperature features for next_month,
    using history_df (sorted ascending, contains month, sales_total,
    sales_COM, sales_IND, sales_RES).
    """
    h = history_df.sort_values("month").reset_index(drop=True)

    total = h[TARGET].values
    n     = len(total)

    row: dict = {}

    # Lags and rolling for sales_total
    for lag in [1, 2, 3, 6, 12, 24]:
        row[f"{TARGET}_lag_{lag}"] = total[-lag] if n >= lag else np.nan

    row[f"{TARGET}_roll3"]     = float(np.mean(total[-3:]))  if n >= 3  else np.nan
    row[f"{TARGET}_roll12"]    = float(np.mean(total[-12:])) if n >= 12 else np.nan
    row[f"{TARGET}_roll3_std"] = float(np.std(total[-3:], ddof=1)) if n >= 3 else np.nan

    # Lags and rolling for sector sub-series
    for sector in ["sales_COM", "sales_IND", "sales_RES"]:
        sv = h[sector].values
        ns = len(sv)
        for lag in [1, 2, 3, 6, 12, 24]:
            row[f"{sector}_lag_{lag}"] = sv[-lag] if ns >= lag else np.nan
        row[f"{sector}_roll3"]     = float(np.mean(sv[-3:]))         if ns >= 3  else np.nan
        row[f"{sector}_roll12"]    = float(np.mean(sv[-12:]))        if ns >= 12 else np.nan
        row[f"{sector}_roll3_std"] = float(np.std(sv[-3:], ddof=1)) if ns >= 3  else np.nan

    # Calendar
    month_num = int(next_month[5:7])
    year      = int(next_month[:4])
    row["month_num"] = month_num
    row["year"]      = year
    row["quarter"]   = (month_num - 1) // 3 + 1

    # Temperature (with_temp only)
    if include_temp and temp_baselines:
        b = temp_baselines[str(month_num)]
        row["temp_avg_f"]            = b["temp_avg_f"]
        row["heating_degree_days"]   = b["hdd"]
        row["cooling_degree_days"]   = b["cdd"]

    return row


def _project_sectors(pred_total: float, recent_df: pd.DataFrame) -> dict:
    """Estimate sector split for a predicted total using recent share averages."""
    shares = {}
    for s in ["sales_COM", "sales_IND", "sales_RES"]:
        ratio = (recent_df[s] / recent_df[TARGET]).mean()
        shares[s] = float(pred_total * ratio)
    return shares


# ===========================================================================
# Recursive ML forecaster (XGBoost, Linear Regression)
# ===========================================================================

def recursive_forecast_ml(model, history_df: pd.DataFrame, n_steps: int,
                           feat_cols: list[str], temp_baselines: dict | None,
                           scaler=None, include_temp: bool = True) -> np.ndarray:
    """
    Recursive forecast: predict one step → append → repeat.
    Returns array of length n_steps.
    """
    h = history_df[["month", TARGET, "sales_COM", "sales_IND", "sales_RES"]].copy()
    h = h.sort_values("month").reset_index(drop=True)
    last_month = h["month"].iloc[-1]
    preds = []

    for step in range(1, n_steps + 1):
        nxt = _next_month_str(last_month, step)
        row = build_next_row(h, nxt, temp_baselines, include_temp)
        X_new = np.array([[row.get(c, 0.0) for c in feat_cols]], dtype=float)
        if scaler is not None:
            X_new = scaler.transform(X_new)
        pred = float(model.predict(X_new)[0])
        preds.append(pred)

        sectors = _project_sectors(pred, h.tail(6))
        new_row = {"month": nxt, TARGET: pred, **sectors}
        h = pd.concat([h, pd.DataFrame([new_row])], ignore_index=True)

    return np.array(preds)


# ===========================================================================
# Recursive LSTM / GRU forecaster
# ===========================================================================

def recursive_forecast_rnn(model, history_df: pd.DataFrame,
                            n_steps: int, feat_cols: list[str],
                            X_scaler: MinMaxScaler,
                            y_scaler: MinMaxScaler) -> np.ndarray:
    """
    Recursive forecast using LSTM or GRU with a sliding 12-month window.
    """
    h = history_df[["month", TARGET, "sales_COM", "sales_IND", "sales_RES"]].copy()
    h = h.sort_values("month").reset_index(drop=True)

    # Build initial window from last SEQ_LEN rows of history (scaled)
    tail = h.tail(SEQ_LEN + 24).copy()   # extra rows to build lag features
    window_rows = []
    for i in range(len(tail) - SEQ_LEN, len(tail)):
        ctx = tail.iloc[:i + 1]
        next_m = tail["month"].iloc[i]
        row = build_next_row(ctx.iloc[:-1], next_m,
                             temp_baselines=None, include_temp=False)
        window_rows.append([row.get(c, 0.0) for c in feat_cols])

    window_scaled = X_scaler.transform(np.array(window_rows[-SEQ_LEN:], dtype=float))
    window = list(window_scaled)          # list of SEQ_LEN feature vectors

    last_month = h["month"].iloc[-1]
    preds = []

    for step in range(1, n_steps + 1):
        nxt = _next_month_str(last_month, step)
        X_seq = np.array(window, dtype=float).reshape(1, SEQ_LEN, len(feat_cols))
        pred_scaled = float(model.predict(X_seq, verbose=0)[0, 0])
        pred_actual = float(y_scaler.inverse_transform([[pred_scaled]])[0, 0])
        preds.append(pred_actual)

        sectors = _project_sectors(pred_actual, h.tail(6))
        new_row = {"month": nxt, TARGET: pred_actual, **sectors}
        h = pd.concat([h, pd.DataFrame([new_row])], ignore_index=True)

        # Build new feature row for the predicted month
        feat_row = build_next_row(h.iloc[:-1], nxt,
                                  temp_baselines=None, include_temp=False)
        fv = np.array([[feat_row.get(c, 0.0) for c in feat_cols]], dtype=float)
        fv_scaled = X_scaler.transform(fv)[0]
        window.pop(0)
        window.append(fv_scaled)

    return np.array(preds)


# ===========================================================================
# Model builders
# ===========================================================================

def _build_rnn_model(cell_type, n_features: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(SEQ_LEN, n_features)),
        cell_type(64, return_sequences=True),
        Dropout(0.2),
        cell_type(32, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def make_sequences(X: np.ndarray, y: np.ndarray,
                   seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    master   = pd.read_csv("master_df.csv")
    feat_wt  = pd.read_csv("features_with_temp.csv")
    feat_lo  = pd.read_csv("features_lag_only.csv")

    master = master.sort_values("month").reset_index(drop=True)

    # Train / test data frames
    train_master = master[master["month"] <= TRAIN_END].copy()
    test_master  = master[(master["month"] >= TEST_START) &
                          (master["month"] <= TEST_END)].copy()

    train_wt = feat_wt[feat_wt["month"] <= TRAIN_END].copy()
    test_wt  = feat_wt[(feat_wt["month"] >= TEST_START) &
                        (feat_wt["month"] <= TEST_END)].copy()
    train_lo = feat_lo[feat_lo["month"] <= TRAIN_END].copy()
    test_lo  = feat_lo[(feat_lo["month"] >= TEST_START) &
                        (feat_lo["month"] <= TEST_END)].copy()

    print(f"Train: {train_wt.shape[0]} rows  |  Test (holdout): {test_wt.shape[0]} rows")

    feat_cols_wt = [c for c in feat_wt.columns if c not in ("month", TARGET)]
    feat_cols_lo = [c for c in feat_lo.columns if c not in ("month", TARGET)]

    X_train_wt = train_wt[feat_cols_wt].values.astype(float)
    y_train_wt = train_wt[TARGET].values.astype(float)
    X_test_wt  = test_wt[feat_cols_wt].values.astype(float)
    y_test     = test_wt[TARGET].values.astype(float)

    X_train_lo = train_lo[feat_cols_lo].values.astype(float)
    y_train_lo = train_lo[TARGET].values.astype(float)

    actual_test = y_test
    prior_last_train = float(train_master[TARGET].iloc[-1])
    test_months = test_wt["month"].tolist()

    # ── Temperature baselines (training data only) ────────────────────────────
    temp_baselines = build_temp_baselines(train_master)
    with open("monthly_temp_baselines.json", "w") as f:
        json.dump(temp_baselines, f, indent=2)
    print("  Saved monthly_temp_baselines.json")

    # ── MODEL 1: SARIMA ────────────────────────────────────────────────────────
    print("\n── SARIMA ──────────────────────────────────────────────────────────")
    train_series = train_master.set_index("month")[TARGET]

    sarima_model = pm.auto_arima(
        train_series,
        seasonal=True, m=12,
        stepwise=True,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
        max_p=3, max_q=3, max_P=2, max_Q=2,
    )
    print(f"  Selected order: {sarima_model.order}  seasonal: {sarima_model.seasonal_order}")

    # 27-step forecast with confidence intervals
    sarima_preds, sarima_ci = sarima_model.predict(n_periods=27, return_conf_int=True, alpha=0.05)

    with open("sarima_model.pkl", "wb") as f:
        pickle.dump(sarima_model, f)
    print("  Saved sarima_model.pkl")

    # ── MODEL 2: XGBoost ───────────────────────────────────────────────────────
    print("\n── XGBoost ─────────────────────────────────────────────────────────")
    param_grid = {
        "n_estimators":    [100, 300, 500],
        "learning_rate":   [0.01, 0.05, 0.1],
        "max_depth":       [3, 5, 7],
        "subsample":       [0.8, 1.0],
        "colsample_bytree":[0.8, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_base = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_SEED, tree_method="hist"
    )
    xgb_search = GridSearchCV(
        xgb_base, param_grid, cv=tscv,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1, verbose=0,
    )
    xgb_search.fit(X_train_wt, y_train_wt)
    xgb_model = xgb_search.best_estimator_
    print(f"  Best params: {xgb_search.best_params_}")

    # Feature importance plot (top 15)
    importances = xgb_model.feature_importances_
    imp_df = pd.DataFrame({"feature": feat_cols_wt, "importance": importances})
    imp_df = imp_df.nlargest(15, "importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#1565C0")
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance (Top 15)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("xgboost_feature_importance.png", bbox_inches="tight")
    plt.close()

    # XGBoost recursive forecast from end of training
    xgb_preds = recursive_forecast_ml(
        xgb_model, train_master, 27, feat_cols_wt,
        temp_baselines, scaler=None, include_temp=True,
    )

    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    print("  Saved xgboost_model.pkl")

    # ── MODEL 3: Linear Regression ─────────────────────────────────────────────
    print("\n── Linear Regression ───────────────────────────────────────────────")
    lr_scaler = StandardScaler()
    X_train_lr = lr_scaler.fit_transform(X_train_wt)
    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train_wt)

    lr_preds = recursive_forecast_ml(
        lr_model, train_master, 27, feat_cols_wt,
        temp_baselines, scaler=lr_scaler, include_temp=True,
    )

    with open("linear_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open("feature_scaler.pkl", "wb") as f:
        pickle.dump(lr_scaler, f)
    print("  Saved linear_model.pkl  +  feature_scaler.pkl")

    # ── MODEL 4: LSTM ──────────────────────────────────────────────────────────
    print("\n── LSTM ────────────────────────────────────────────────────────────")
    X_scaler_lstm = MinMaxScaler()
    y_scaler_lstm = MinMaxScaler()

    X_train_lo_scaled = X_scaler_lstm.fit_transform(X_train_lo)
    y_train_lo_scaled = y_scaler_lstm.fit_transform(
        y_train_lo.reshape(-1, 1)
    ).flatten()

    X_seq, y_seq = make_sequences(X_train_lo_scaled, y_train_lo_scaled, SEQ_LEN)

    val_size = int(0.2 * len(X_seq))
    X_tr_s, X_val_s = X_seq[:-val_size], X_seq[-val_size:]
    y_tr_s, y_val_s = y_seq[:-val_size], y_seq[-val_size:]

    lstm_model = _build_rnn_model(LSTM, len(feat_cols_lo))
    early_stop = EarlyStopping(monitor="val_loss", patience=10,
                               restore_best_weights=True, verbose=0)
    history_lstm = lstm_model.fit(
        X_tr_s, y_tr_s,
        validation_data=(X_val_s, y_val_s),
        epochs=100, batch_size=16,
        callbacks=[early_stop], verbose=0,
    )
    print(f"  Trained {len(history_lstm.epoch)} epochs "
          f"(early stopped at {len(history_lstm.epoch)})")

    lstm_preds = recursive_forecast_rnn(
        lstm_model, train_master, 27, feat_cols_lo,
        X_scaler_lstm, y_scaler_lstm,
    )

    lstm_model.save("lstm_model.keras")
    with open("lstm_scaler.pkl", "wb") as f:
        pickle.dump({"X": X_scaler_lstm, "y": y_scaler_lstm}, f)
    print("  Saved lstm_model.keras  +  lstm_scaler.pkl")

    # ── MODEL 5: GRU ───────────────────────────────────────────────────────────
    print("\n── GRU ─────────────────────────────────────────────────────────────")
    # Same scalers and data as LSTM
    gru_model = _build_rnn_model(GRU, len(feat_cols_lo))
    early_stop_gru = EarlyStopping(monitor="val_loss", patience=10,
                                   restore_best_weights=True, verbose=0)
    history_gru = gru_model.fit(
        X_tr_s, y_tr_s,
        validation_data=(X_val_s, y_val_s),
        epochs=100, batch_size=16,
        callbacks=[early_stop_gru], verbose=0,
    )
    print(f"  Trained {len(history_gru.epoch)} epochs "
          f"(early stopped at {len(history_gru.epoch)})")

    gru_preds = recursive_forecast_rnn(
        gru_model, train_master, 27, feat_cols_lo,
        X_scaler_lstm, y_scaler_lstm,
    )

    gru_model.save("gru_model.keras")
    print("  Saved gru_model.keras")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    print("\n── Evaluation ──────────────────────────────────────────────────────")
    models_preds = {
        "SARIMA":           np.array(sarima_preds),
        "XGBoost":          xgb_preds,
        "LinearRegression": lr_preds,
        "LSTM":             lstm_preds,
        "GRU":              gru_preds,
    }

    rows = []
    for name, preds in models_preds.items():
        r: dict = {"Model": name}
        for h in [1, 3, 6, 12]:
            m = horizon_metrics(actual_test, preds, prior_last_train, h)
            r[f"MAPE_h{h}"] = round(m["MAPE"], 3)
            if h == 12:
                r["RMSE_h12"]   = round(m["RMSE"], 1)
                r["MAE_h12"]    = round(m["MAE"], 1)
                r["R2_h12"]     = round(m["R2"], 4)
                r["DirAcc_h12"] = round(m["DirAcc"], 1)
        rows.append(r)

    results = pd.DataFrame(rows).set_index("Model")
    print("\n" + results.to_string())
    results.to_csv("model_results.csv")
    print("\n  Saved model_results.csv")

    # ── CHART 1: Actual vs Predicted (5 subplots) ─────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    test_dates = pd.to_datetime([m + "-01" for m in test_months])

    for ax, (name, preds) in zip(axes, models_preds.items()):
        mape12 = results.loc[name, "MAPE_h12"]
        ax.plot(test_dates, actual_test, "b-", lw=2, label="Actual")
        ax.plot(test_dates, preds[:27],  "r--", lw=1.5, label="Predicted")
        ax.set_ylabel("million kWh")
        ax.set_title(f"{name}  —  MAPE(h12): {mape12:.2f}%", fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    axes[-1].set_xlabel("Month")
    fig.suptitle("Actual vs Predicted — Holdout Period (Jan 2024 – Mar 2026)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("evaluation_chart.png", bbox_inches="tight")
    plt.close()
    print("  Saved evaluation_chart.png")

    # ── CHART 2: Forward forecast from last data point ────────────────────────
    # Best model per horizon
    best_short = results["MAPE_h3"].idxmin()
    best_long  = results["MAPE_h12"].idxmin()
    print(f"\n  Best short-term model (h3): {best_short}  "
          f"MAPE={results.loc[best_short,'MAPE_h3']:.2f}%")
    print(f"  Best long-term  model (h12): {best_long}  "
          f"MAPE={results.loc[best_long,'MAPE_h12']:.2f}%")

    # Generate forward forecasts from end of FULL dataset (2026-03)
    def _forward_forecast(model_name, n_steps):
        if model_name == "SARIMA":
            # Re-fit SARIMA on full data (train + test) for forward forecast
            full_series = master.set_index("month")[TARGET]
            full_sarima = sarima_model.update(test_master[TARGET].values)
            fp, fp_ci = full_sarima.predict(n_steps, return_conf_int=True, alpha=0.05)
            return np.array(fp), np.array(fp_ci)
        else:
            rnn_models = {"LSTM": lstm_model, "GRU": gru_model}
            if model_name in rnn_models:
                fp = recursive_forecast_rnn(
                    rnn_models[model_name], master, n_steps,
                    feat_cols_lo, X_scaler_lstm, y_scaler_lstm,
                )
            else:
                ml_models = {"XGBoost": xgb_model, "LinearRegression": lr_model}
                sc = {"LinearRegression": lr_scaler}.get(model_name)
                fp = recursive_forecast_ml(
                    ml_models[model_name], master, n_steps, feat_cols_wt,
                    temp_baselines, scaler=sc, include_temp=True,
                )
            # Approximate CI: ±1.96 × RMSE_h12 × sqrt(step)
            rmse = results.loc[model_name, "RMSE_h12"]
            ci = np.array([[fp[i] - 1.96 * rmse * np.sqrt(i + 1),
                            fp[i] + 1.96 * rmse * np.sqrt(i + 1)]
                           for i in range(n_steps)])
            return fp, ci

    fp_short, ci_short = _forward_forecast(best_short, 12)
    fp_long,  ci_long  = _forward_forecast(best_long, 12)

    # Historical: last 3 years
    hist_start = "2023-04"
    hist_df = master[(master["month"] >= hist_start) & (master["month"] <= TEST_END)]
    hist_dates = pd.to_datetime(hist_df["month"] + "-01")
    last_actual_date = pd.to_datetime(TEST_END + "-01")

    future_dates = pd.DatetimeIndex([
        pd.to_datetime(TEST_END + "-01") + pd.DateOffset(months=i + 1)
        for i in range(12)
    ])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist_dates, hist_df[TARGET].values, "b-", lw=2, label="Historical (actual)")
    ax.axvline(last_actual_date, color="gray", linestyle="--", lw=1.5,
               label=f"Last actual: {TEST_END}")

    short_dates_3 = future_dates[:3]
    ax.plot(short_dates_3, fp_short[:3], color="#E53935", lw=2,
            label=f"Short-term forecast ({best_short}, h1–h3)")
    ax.fill_between(short_dates_3,
                    ci_short[:3, 0], ci_short[:3, 1],
                    color="#E53935", alpha=0.2)

    ax.plot(future_dates[:6], fp_long[:6], color="#1565C0", lw=1.5,
            linestyle="--", label=f"Long-term forecast ({best_long}, h6)")
    ax.plot(future_dates[:12], fp_long[:12], color="#1565C0", lw=1.5,
            linestyle=":", alpha=0.8, label=f"Long-term forecast ({best_long}, h12)")
    ax.fill_between(future_dates[:12],
                    ci_long[:12, 0], ci_long[:12, 1],
                    color="#1565C0", alpha=0.15)

    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales (million kWh)")
    ax.set_title("US Electricity Sales — Historical (Last 3 Years) + Forward Forecast",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("forecast_chart.png", bbox_inches="tight")
    plt.close()
    print("  Saved forecast_chart.png")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL MODEL RESULTS")
    print("=" * 70)
    print(results.to_string())
    print("\nPart 4 complete. All 5 models trained and evaluated on holdout "
          "(Jan 2024 – Mar 2026). model_results.csv, evaluation_chart.png, "
          f"forecast_chart.png saved. Best short-term: {best_short} "
          f"(MAPE_h3={results.loc[best_short,'MAPE_h3']:.2f}%). "
          f"Best long-term: {best_long} "
          f"(MAPE_h12={results.loc[best_long,'MAPE_h12']:.2f}%).")


if __name__ == "__main__":
    main()
