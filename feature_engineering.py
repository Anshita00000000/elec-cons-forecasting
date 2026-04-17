"""
feature_engineering.py — US Electricity Consumption Forecasting
================================================================
Builds lag, calendar, and temperature features from master_df.csv.

Outputs:
  features_with_temp.csv  — lag + calendar + temperature (XGBoost, LinearReg)
  features_lag_only.csv   — lag + calendar only           (LSTM, GRU)

Train/test split (time-based, never shuffled):
  Training : 2001-01 → 2023-12
  Holdout  : 2024-01 → 2026-03  (27 months, never touched during training)

Constraints:
  - No revenue_musd, no holidays, no price, no customers
  - Never KFold or shuffle=True
  - Always drop first 24 rows after creating lag features
"""

import pandas as pd
import numpy as np

TARGET = "sales_total"

TRAIN_END  = "2023-12"
TEST_START = "2024-01"
TEST_END   = "2026-03"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all lag, rolling, and calendar features to df in-place copy.
    Returns the augmented dataframe (rows with NaN lag values still present).
    """
    out = df.copy()

    # ── Lag features for sales_total ─────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24]:
        out[f"sales_total_lag_{lag}"] = out[TARGET].shift(lag)

    # ── Rolling statistics on sales_total ────────────────────────────────────
    out["sales_total_roll3"]     = out[TARGET].shift(1).rolling(3).mean()
    out["sales_total_roll12"]    = out[TARGET].shift(1).rolling(12).mean()
    out["sales_total_roll3_std"] = out[TARGET].shift(1).rolling(3).std()

    # ── Lag features for sector sub-series ───────────────────────────────────
    for sector in ["sales_COM", "sales_IND", "sales_RES"]:
        for lag in [1, 2, 3, 6, 12, 24]:
            out[f"{sector}_lag_{lag}"] = out[sector].shift(lag)
        out[f"{sector}_roll3"]     = out[sector].shift(1).rolling(3).mean()
        out[f"{sector}_roll12"]    = out[sector].shift(1).rolling(12).mean()
        out[f"{sector}_roll3_std"] = out[sector].shift(1).rolling(3).std()

    # ── Calendar features ─────────────────────────────────────────────────────
    out["quarter"] = ((out["month_num"] - 1) // 3) + 1

    return out


def main() -> None:
    df = pd.read_csv("master_df.csv")
    df = df.sort_values("month").reset_index(drop=True)
    print(f"Loaded master_df: {df.shape}")

    # Build features
    featured = build_features(df)

    # ── Drop first 24 rows (NaN from lag_24) ─────────────────────────────────
    featured = featured.iloc[24:].reset_index(drop=True)
    assert featured[f"{TARGET}_lag_24"].notna().all(), \
        "Lag-24 still contains NaN after dropping first 24 rows."
    print(f"After dropping first 24 rows: {featured.shape[0]} rows remaining — OK")

    # ── Define feature sets ───────────────────────────────────────────────────
    lag_cols = [c for c in featured.columns if (
        "_lag_" in c or "_roll" in c
    )]
    calendar_cols = ["month_num", "year", "quarter"]
    temp_cols     = ["temp_avg_f", "heating_degree_days", "cooling_degree_days"]

    # features_with_temp: lag + calendar + temperature (for XGBoost, LinearReg)
    with_temp_cols  = ["month", TARGET] + lag_cols + calendar_cols + temp_cols
    features_with_temp = featured[with_temp_cols].copy()

    # features_lag_only: lag + calendar only (for LSTM, GRU — no temp projection needed)
    lag_only_cols  = ["month", TARGET] + lag_cols + calendar_cols
    features_lag_only = featured[lag_only_cols].copy()

    # ── Assert no missing values in either feature set ────────────────────────
    for name, fset in [("features_with_temp", features_with_temp),
                       ("features_lag_only",  features_lag_only)]:
        missing = fset.isnull().sum()
        bad = missing[missing > 0]
        if len(bad):
            raise AssertionError(f"{name} has missing values:\n{bad}")
        print(f"  {name}: {fset.shape}, no missing values — OK")

    # ── Train / test split ────────────────────────────────────────────────────
    def split(fset):
        train = fset[fset["month"] <= TRAIN_END]
        test  = fset[(fset["month"] >= TEST_START) & (fset["month"] <= TEST_END)]
        return train, test

    train_wt, test_wt = split(features_with_temp)
    train_lo, test_lo = split(features_lag_only)

    print(f"\nTrain/test split (target: {TARGET}):")
    print(f"  features_with_temp — train: {train_wt.shape[0]} rows "
          f"({train_wt['month'].min()} → {train_wt['month'].max()})")
    print(f"  features_with_temp — test:  {test_wt.shape[0]} rows "
          f"({test_wt['month'].min()} → {test_wt['month'].max()})")
    print(f"  features_lag_only  — train: {train_lo.shape[0]} rows "
          f"({train_lo['month'].min()} → {train_lo['month'].max()})")
    print(f"  features_lag_only  — test:  {test_lo.shape[0]} rows "
          f"({test_lo['month'].min()} → {test_lo['month'].max()})")

    # ── Save ──────────────────────────────────────────────────────────────────
    features_with_temp.to_csv("features_with_temp.csv", index=False)
    features_lag_only.to_csv("features_lag_only.csv",   index=False)
    print("\n  Saved → features_with_temp.csv")
    print("  Saved → features_lag_only.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nFeature summary:")
    print(f"  Total lag/rolling features per target: 9 (lag 1,2,3,6,12,24 + roll3,roll12,roll3_std)")
    print(f"  Sectors covered: sales_total, sales_COM, sales_IND, sales_RES")
    print(f"  Calendar features: month_num, year, quarter")
    print(f"  Temperature features (with_temp only): temp_avg_f, HDD, CDD")
    print(f"  features_with_temp columns: {features_with_temp.shape[1]} "
          f"(incl. month + target)")
    print(f"  features_lag_only columns:  {features_lag_only.shape[1]} "
          f"(incl. month + target)")

    print(
        "\nPart 3 complete. feature_engineering.py built lag features (lags 1,2,3,6,12,24), "
        "rolling statistics (roll3, roll12, roll3_std) for sales_total and all three sectors, "
        "plus calendar features. First 24 rows dropped. "
        f"Train: 2001-01 to 2023-12 ({train_wt.shape[0]} rows). "
        f"Holdout: 2024-01 to 2026-03 ({test_wt.shape[0]} rows). "
        "Saved features_with_temp.csv and features_lag_only.csv."
    )


if __name__ == "__main__":
    main()
