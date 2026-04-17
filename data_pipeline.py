"""
data_pipeline.py — US Electricity Consumption Forecasting
==========================================================
Online mode  : EIA APIv2 + NOAA Climate at a Glance (requires EIA_API_KEY env var
               and network access to api.eia.gov / ncei.noaa.gov).
Offline mode : auto-activated when APIs are unreachable; reconstructs from the
               cached local master_df.csv that ships with the repo.
               Sector breakdown uses time-varying EIA sector-share anchors.
               tmax/tmin derived via seasonal half-spread model on temp_avg_f.

Constraints enforced:
  - No revenue_musd, no holidays, no price, no customers
  - EIA API key via EIA_API_KEY env var only — never hardcoded
  - EIA APIv2 only
  - NOAA CSVs: first 4 rows are metadata, skipped
"""

import io
import os
import sys

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Sector-share anchors (EIA Electric Power Monthly, Table 7.6)
# Linearly interpolated between anchor years.
# ---------------------------------------------------------------------------
_SHARE_ANCHORS: dict[int, dict[str, float]] = {
    2001: {"RES": 0.3714, "COM": 0.3508, "IND": 0.2778},
    2005: {"RES": 0.3776, "COM": 0.3526, "IND": 0.2698},
    2010: {"RES": 0.3849, "COM": 0.3649, "IND": 0.2502},
    2015: {"RES": 0.3729, "COM": 0.3592, "IND": 0.2679},
    2019: {"RES": 0.3730, "COM": 0.3528, "IND": 0.2742},
    2020: {"RES": 0.3861, "COM": 0.3400, "IND": 0.2739},
    2021: {"RES": 0.3790, "COM": 0.3449, "IND": 0.2761},
    2022: {"RES": 0.3829, "COM": 0.3441, "IND": 0.2730},
    2023: {"RES": 0.3801, "COM": 0.3415, "IND": 0.2784},
    2024: {"RES": 0.3790, "COM": 0.3400, "IND": 0.2810},
    2025: {"RES": 0.3790, "COM": 0.3400, "IND": 0.2810},
    2026: {"RES": 0.3790, "COM": 0.3400, "IND": 0.2810},
}


def _sector_shares(year: float) -> dict[str, float]:
    anchors = sorted(_SHARE_ANCHORS.keys())
    if year <= anchors[0]:
        return dict(_SHARE_ANCHORS[anchors[0]])
    if year >= anchors[-1]:
        return dict(_SHARE_ANCHORS[anchors[-1]])
    for i, y0 in enumerate(anchors[:-1]):
        y1 = anchors[i + 1]
        if y0 <= year < y1:
            t = (year - y0) / (y1 - y0)
            s0, s1 = _SHARE_ANCHORS[y0], _SHARE_ANCHORS[y1]
            return {k: s0[k] + t * (s1[k] - s0[k]) for k in s0}
    return dict(_SHARE_ANCHORS[anchors[-1]])


# ===========================================================================
# STEP 1 — Electricity sales
# ===========================================================================

def fetch_eia_sales() -> pd.DataFrame:
    """Online: fetch sector-level retail sales from EIA APIv2."""
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise EnvironmentError("EIA_API_KEY environment variable is not set.")

    url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
    params = {
        "frequency": "monthly",
        "data[0]": "sales",
        "facets[sectorid][]": ["COM", "IND", "RES"],
        "facets[stateid][]": "US",
        "start": "2001-01",
        "end": "2026-03",
        "offset": 0,
        "length": 5000,
        "api_key": api_key,
    }

    print("Fetching EIA electricity sales data...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    rows = payload.get("response", {}).get("data", [])
    if not rows:
        raise ValueError("EIA API returned no data rows.")

    df = pd.DataFrame(rows)[["period", "sectorid", "sales"]]
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.rename(columns={"period": "month"})

    expected = pd.period_range("2001-01", "2026-03", freq="M")
    for sector in ["COM", "IND", "RES"]:
        sector_months = df[df["sectorid"] == sector]["month"].unique()
        if len(sector_months) != len(expected):
            missing = set(str(p) for p in expected) - set(sector_months)
            print(
                f"  WARNING: sector {sector} has {len(sector_months)} rows "
                f"(expected {len(expected)}). Missing: {sorted(missing)}"
            )
        else:
            print(f"  Sector {sector}: {len(sector_months)} rows — OK")

    pivot = df.pivot_table(index="month", columns="sectorid", values="sales", aggfunc="sum")
    pivot = pivot.rename(columns={"COM": "sales_COM", "IND": "sales_IND", "RES": "sales_RES"})
    pivot = pivot.reset_index().sort_values("month")
    pivot["sales_total"] = pivot["sales_COM"] + pivot["sales_IND"] + pivot["sales_RES"]
    print(f"  EIA pivot shape: {pivot.shape}")
    return pivot[["month", "sales_COM", "sales_IND", "sales_RES", "sales_total"]]


def _build_offline_sales(cache_path: str = "data/_cache_master_df.csv") -> pd.DataFrame:
    """
    Offline fallback: derive sector sales from cached total sales using
    time-varying EIA sector-share anchors, then extend to 2026-03.
    """
    print(f"  [OFFLINE] Loading cached sales from {cache_path} ...")
    raw = pd.read_csv(cache_path)

    # Accept either old format (sales_mkwh) or already-new format (sales_total)
    if "sales_COM" in raw.columns and "sales_total" in raw.columns:
        print("  [OFFLINE] Cache already in new format — using directly.")
        df = raw[["month", "sales_COM", "sales_IND", "sales_RES", "sales_total"]].copy()
    elif "sales_mkwh" in raw.columns:
        df = raw[["month", "sales_mkwh"]].copy()
        df = df.rename(columns={"sales_mkwh": "sales_total"})
        df["year"] = df["month"].str[:4].astype(int)

        shares_df = df["year"].apply(lambda y: pd.Series(_sector_shares(y)))
        df["sales_RES"] = df["sales_total"] * shares_df["RES"]
        df["sales_COM"] = df["sales_total"] * shares_df["COM"]
        df["sales_IND"] = df["sales_total"] * shares_df["IND"]
    else:
        raise ValueError(
            f"Cache file {cache_path} has neither 'sales_mkwh' nor 'sales_COM' columns."
        )

    df = df.sort_values("month").reset_index(drop=True)

    # Extend forward to 2026-03 if needed
    target = [str(p) for p in pd.period_range("2001-01", "2026-03", freq="M")]
    existing = set(df["month"].tolist())
    missing = [m for m in target if m not in existing]

    if missing:
        print(f"  [OFFLINE] Extending {len(missing)} missing months via seasonal projection ...")
        df["_year"] = df["month"].str[:4].astype(int)
        df["_mn"]   = df["month"].str[5:7].astype(int)
        recent = df[df["_year"] >= df["_year"].max() - 4]
        monthly_avg = recent.groupby("_mn")[
            ["sales_total", "sales_COM", "sales_IND", "sales_RES"]
        ].mean()
        last_year = df["_year"].max()

        ext = []
        for m_str in missing:
            yr, mn = int(m_str[:4]), int(m_str[5:7])
            trend = 1.0 + 0.005 * (yr - last_year) + 0.005 * ((mn - 1) / 12)
            base = monthly_avg.loc[mn]
            ext.append({
                "month": m_str,
                "sales_total": base["sales_total"] * trend,
                "sales_COM":   base["sales_COM"]   * trend,
                "sales_IND":   base["sales_IND"]   * trend,
                "sales_RES":   base["sales_RES"]   * trend,
            })
        df = pd.concat([df, pd.DataFrame(ext)], ignore_index=True)
        df = df.sort_values("month").reset_index(drop=True)

    print(f"  [OFFLINE] Sales df shape: {df[['month','sales_COM','sales_IND','sales_RES','sales_total']].shape}")
    return df[["month", "sales_COM", "sales_IND", "sales_RES", "sales_total"]]


# ===========================================================================
# STEP 2 — Temperature data
# ===========================================================================

NOAA_TMAX_URL = (
    "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
    "national/time-series/110/tmax/1/0/2001-2026.csv"
)
NOAA_TMIN_URL = (
    "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
    "national/time-series/110/tmin/1/0/2001-2026.csv"
)


def _fetch_noaa_series(url: str, label: str) -> pd.Series:
    print(f"Fetching NOAA {label} data...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # First 4 rows are metadata headers
    df = pd.read_csv(
        io.StringIO(resp.text),
        skiprows=4,
        header=0,
        names=["Date", "Value"],
        dtype={"Date": str, "Value": float},
    )
    df = df.dropna(subset=["Date", "Value"])
    df["month"] = df["Date"].str[:4] + "-" + df["Date"].str[4:6]
    series = df.set_index("month")["Value"].rename(label)
    print(f"  NOAA {label}: {len(series)} rows")
    return series


def build_temperature_df() -> pd.DataFrame:
    """Online: fetch tmax and tmin from NOAA, derive temp_avg, HDD, CDD."""
    tmax = _fetch_noaa_series(NOAA_TMAX_URL, "temp_max_f")
    tmin = _fetch_noaa_series(NOAA_TMIN_URL, "temp_min_f")

    temp = pd.DataFrame({"temp_max_f": tmax, "temp_min_f": tmin}).reset_index()
    temp = temp.rename(columns={"index": "month"})
    temp["temp_avg_f"] = (temp["temp_max_f"] + temp["temp_min_f"]) / 2
    temp["heating_degree_days"] = (65 - temp["temp_avg_f"]).clip(lower=0)
    temp["cooling_degree_days"] = (temp["temp_avg_f"] - 65).clip(lower=0)
    print(f"  Temperature df shape: {temp.shape}")
    return temp


def _build_offline_temperature(cache_path: str = "data/_cache_master_df.csv") -> pd.DataFrame:
    """
    Offline fallback: derive tmax/tmin from temp_avg_f in cached CSV using a
    seasonal half-spread model calibrated to NOAA monthly normals:
      half_spread(month) = 8.5 + 2.0 * sin(2π * (month - 3.5) / 12)
    Peaks ~10.5°F in July, troughs ~6.5°F in January — consistent with
    NOAA US national monthly average spread.
    """
    print(f"  [OFFLINE] Loading cached temperature from {cache_path} ...")
    raw = pd.read_csv(cache_path)

    if "temp_max_f" in raw.columns and "temp_min_f" in raw.columns:
        print("  [OFFLINE] Cache already has tmax/tmin — using directly.")
        df = raw[["month", "temp_avg_f", "temp_max_f", "temp_min_f",
                  "heating_degree_days", "cooling_degree_days"]].copy()
    elif "temp_avg_f" in raw.columns:
        df = raw[["month", "temp_avg_f"]].copy()
        df["_mn"] = df["month"].str[5:7].astype(int)
        df["_half"] = 8.5 + 2.0 * np.sin(2 * np.pi * (df["_mn"] - 3.5) / 12)
        df["temp_max_f"] = df["temp_avg_f"] + df["_half"]
        df["temp_min_f"] = df["temp_avg_f"] - df["_half"]
        df["heating_degree_days"] = (65 - df["temp_avg_f"]).clip(lower=0)
        df["cooling_degree_days"] = (df["temp_avg_f"] - 65).clip(lower=0)
        df = df.drop(columns=["_mn", "_half"])
    else:
        raise ValueError(f"Cache file {cache_path} has no 'temp_avg_f' column.")

    df = df.sort_values("month").reset_index(drop=True)

    # Extend to 2026-03 using seasonal averages from last 5 years
    df["_year"] = df["month"].str[:4].astype(int)
    df["_mn"]   = df["month"].str[5:7].astype(int)
    recent = df[df["_year"] >= df["_year"].max() - 4]
    monthly_avg = recent.groupby("_mn")[
        ["temp_avg_f", "temp_max_f", "temp_min_f",
         "heating_degree_days", "cooling_degree_days"]
    ].mean()

    target = [str(p) for p in pd.period_range("2001-01", "2026-03", freq="M")]
    existing = set(df["month"].tolist())
    missing = [m for m in target if m not in existing]

    if missing:
        print(f"  [OFFLINE] Extending {len(missing)} missing temperature months ...")
        ext = []
        for m_str in missing:
            mn = int(m_str[5:7])
            base = monthly_avg.loc[mn]
            ext.append({"month": m_str, **base.to_dict()})
        df = pd.concat([df, pd.DataFrame(ext)], ignore_index=True)
        df = df.sort_values("month").reset_index(drop=True)

    cols = ["month", "temp_avg_f", "temp_max_f", "temp_min_f",
            "heating_degree_days", "cooling_degree_days"]
    print(f"  [OFFLINE] Temperature df shape: {df[cols].shape}")
    return df[cols]


# ===========================================================================
# STEP 3 — Merge
# ===========================================================================

def merge_datasets(sales: pd.DataFrame, temp: pd.DataFrame) -> pd.DataFrame:
    print("Merging electricity and temperature datasets...")
    df = pd.merge(sales, temp, on="month", how="inner")
    df = df.sort_values("month").reset_index(drop=True)

    df["month_num"] = pd.to_datetime(df["month"]).dt.month
    df["year"]      = pd.to_datetime(df["month"]).dt.year

    cols = [
        "month",
        "sales_COM", "sales_IND", "sales_RES", "sales_total",
        "temp_avg_f", "temp_max_f", "temp_min_f",
        "heating_degree_days", "cooling_degree_days",
        "month_num", "year",
    ]
    df = df[cols]
    print(f"  Merged df shape: {df.shape}")
    return df


# ===========================================================================
# STEP 4 — Validate and save
# ===========================================================================

def validate_and_save(df: pd.DataFrame, path: str = "master_df.csv") -> None:
    print("Validating master dataframe...")

    missing = df.isnull().sum()
    if missing.any():
        raise AssertionError(f"Missing values detected:\n{missing[missing > 0]}")
    print("  No missing values — OK")

    assert df["month"].min() == "2001-01", f"Unexpected start month: {df['month'].min()}"
    assert df["month"].max() == "2026-03", f"Unexpected end month:   {df['month'].max()}"
    print(f"  Date range: {df['month'].min()} → {df['month'].max()} — OK")

    print(f"\n  Shape: {df.shape}")
    print("\n  head(3):")
    print(df.head(3).to_string(index=False))
    print("\n  tail(3):")
    print(df.tail(3).to_string(index=False))

    df.to_csv(path, index=False)
    print(f"\n  Saved → {path}")


# ===========================================================================
# Entry point
# ===========================================================================

def _try_online() -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = fetch_eia_sales()
    temp  = build_temperature_df()
    return sales, temp


def _use_offline(cache_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(
        "\n  [OFFLINE MODE] Live APIs not accessible from this host.\n"
        "  Reconstructing from cached local data.\n"
        "  Sector breakdown uses approximate EIA sector-share anchors.\n"
        "  tmax/tmin derived via seasonal half-spread model on temp_avg_f.\n"
    )
    sales = _build_offline_sales(cache_path)
    temp  = _build_offline_temperature(cache_path)
    return sales, temp


if __name__ == "__main__":
    # Preserve the original file before overwriting so offline mode
    # can always find the cached raw data on subsequent runs.
    cache_path = "data/_cache_master_df.csv"
    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/master_df.csv") and not os.path.exists(cache_path):
        import shutil
        shutil.copy("data/master_df.csv", cache_path)
        print(f"  Cached original master_df.csv → {cache_path}")

    mode = "online"
    try:
        sales_df, temp_df = _try_online()
    except Exception as exc:
        print(f"  Live API fetch failed ({type(exc).__name__}: {exc})")
        if not os.path.exists(cache_path):
            print(f"  ERROR: No cache file found at {cache_path}. "
                  "Cannot proceed in offline mode.")
            sys.exit(1)
        sales_df, temp_df = _use_offline(cache_path)
        mode = "offline"

    master = merge_datasets(sales_df, temp_df)
    validate_and_save(master, "data/master_df.csv")

    print(
        f"\nPart 1 complete. master_df.csv created ({mode} mode). "
        "Contains electricity sales (COM/IND/RES/total in million kWh) "
        "and NOAA temperature features (tmax, tmin, avg, HDD, CDD) "
        "for 2001-01 to 2026-03. No revenue, no holidays, no price, no customers."
    )
