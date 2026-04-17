# US Electricity Consumption Forecast

## Overview

This project forecasts monthly US national electricity retail sales (in million kWh)
across Residential, Commercial, and Industrial sectors using five machine-learning models
(SARIMA, XGBoost, Linear Regression, LSTM, GRU). It is designed as a production-ready
pipeline that fetches live data from the EIA APIv2 and NOAA, trains models with
time-series-safe cross-validation, and serves results via an interactive Streamlit
dashboard with no manual data-entry required.

---

## What changed from v1

- **Holidays removed:** v1 used `holidays.India()` on US national data — holidays feature
  dropped entirely; no holiday data is used.
- **Revenue excluded:** `revenue_musd` was a target-leaking feature (revenue = price × sales)
  and has been permanently removed from all feature sets.
- **TimeSeriesSplit replaces KFold:** v1 used `KFold(shuffle=True)` on time-series data,
  which causes future-leaking. All cross-validation now uses `TimeSeriesSplit(n_splits=5)`.
- **Lag features replace future inputs:** v1 required users to manually enter 14 feature
  values including future temperature. The new model uses lag features (1, 2, 3, 6, 12,
  24-month lags + rolling stats) so forecasts require only historical data.
- **Dashboard replaces manual UI:** the Streamlit app auto-computes all forecasts on startup;
  no user inputs are required.
- **Heatmap title fixed:** v1 had a copy-paste error — "Iris Dataset" title on an
  electricity correlation heatmap. All chart titles are now accurate and descriptive.
- **Test set corrected:** v1 used 2025 data as holdout despite it being partially
  unavailable. The new holdout is 2024-01 to 2026-03 (27 full months of published data).

---

## Data sources

- **EIA APIv2:** Retail electricity sales (Commercial, Industrial, Residential sectors) —
  monthly, January 2001 – March 2026. Accessed via `EIA_API_KEY` environment variable.
  Endpoint: `https://api.eia.gov/v2/electricity/retail-sales/data/`
- **NOAA Climate at a Glance:** US national monthly average tmax and tmin (°F) —
  January 2001 – March 2026.
  `https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tmax/1/0/2001-2026.csv`

---

## Model results

Holdout period: **January 2024 – March 2026** (27 months, never seen during training).
Metrics at each horizon use the first h months of the 27-month recursive forecast.

| Model            | MAPE h1 (%) | MAPE h3 (%) | MAPE h6 (%) | MAPE h12 (%) | RMSE h12 | MAE h12  | R² h12 | DirAcc h12 (%) |
|------------------|-------------|-------------|-------------|--------------|----------|----------|--------|----------------|
| SARIMA           | 4.433       | 2.642       | 3.205       | 3.543        | 15937.6  | 12530.0  | 0.7977 | 91.7           |
| XGBoost          | 4.718       | 3.617       | 3.239       | **2.398**    | 9993.7   | 8058.5   | 0.9205 | 91.7           |
| LinearRegression | 3.839       | 4.306       | 3.803       | 2.627        | 9746.9   | 8405.8   | 0.9243 | 91.7           |
| LSTM             | 9.161       | 5.853       | 4.560       | 4.146        | 16227.5  | 13701.3  | 0.7903 | 91.7           |
| **GRU**          | **2.329**   | **1.774**   | **2.170**   | 2.415        | 10816.5  | 8434.2   | 0.9068 | **100.0**      |

**Best short-term (h3):** GRU — 1.77%  
**Best long-term (h12):** XGBoost — 2.40%

---

## How to run

### Docker (recommended)

```bash
# Set your EIA API key
export EIA_API_KEY=your_key_here

# Build and start the dashboard
docker-compose up --build

# Dashboard available at http://localhost:8501
```

### Local

```bash
pip install -r requirements.txt

# Set EIA API key (required only to regenerate master_df.csv)
export EIA_API_KEY=your_key_here

# 1. Build data
python data_pipeline.py

# 2. Exploratory analysis (optional)
jupyter nbconvert --to notebook --execute eda.ipynb

# 3. Feature engineering
python feature_engineering.py

# 4. Train all models (~15–25 min on CPU)
python models.py

# 5. Launch dashboard
streamlit run app.py
```

---

## Limitations and future work

- **LSTM/GRU underperform at ~280 training rows** — deep sequence models need thousands
  of rows to outperform classical methods; results here confirm this. More data (sub-monthly
  frequency or additional regions) would help.
- **Temperature projections for XGBoost use historical monthly averages** — actual
  weather forecasts (e.g., NOAA 7/30-day outlooks) would substantially improve near-term
  accuracy, especially during anomalous seasons.
- **Hierarchical forecasting (sector → aggregate)** is a planned improvement — forecasting
  COM, IND, RES independently and summing may outperform forecasting the total directly,
  since each sector has different drivers.
