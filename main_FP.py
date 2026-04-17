import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os

# --- 1. SETUP AUDIT LOGGING ---
logging.basicConfig(
    filename="drift_logs.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Energy Sales Forecasting API", version="2.0")

# --- 2. CALCULATE DRIFT THRESHOLDS (The "Reference Book") ---
# We load the historical data once on startup to know what "Normal" looks like.
reference_stats = {}

if os.path.exists("master_df.csv"):
    try:
        df_train = pd.read_csv("master_df.csv")
        
        # Calculate 99% Confidence Intervals (The Safe Zone)
        reference_stats["temp_avg_f"] = {
            "min": df_train["temp_avg_f"].quantile(0.01),
            "max": df_train["temp_avg_f"].quantile(0.99)
        }
        reference_stats["industrial_production_index"] = {
            "min": df_train["industrial_production_index"].quantile(0.01),
            "max": df_train["industrial_production_index"].quantile(0.99)
        }
        print(f"✅ Reference Statistics Loaded: {reference_stats}")
    except Exception as e:
        print(f"⚠️ Stats Calculation Failed: {e}")
else:
    print("⚠️ Warning: 'master_df.csv' not found. Using default drift thresholds.")
    reference_stats["temp_avg_f"] = {"min": 20, "max": 90}
    reference_stats["industrial_production_index"] = {"min": 80, "max": 120}

# --- 3. LOAD MODEL ---
model = None
try:
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Critical Error: Could not load 'xgboost_model.pkl'. {e}")

# --- 4. DEFINE INPUT SCHEMA (14 Features) ---
class EnergyInput(BaseModel):
    price_cents_per_kwh: float
    energy_consumption_million_mmbtu: float
    public_holidays: int
    saturdays: int
    sundays: int
    weekend_days: int
    holiday_on_weekend: int
    total_off_days: int
    industrial_production_index: float
    temp_avg_f: float
    heating_degree_days: float
    cooling_degree_days: float
    precip_inches: float
    month_num: int

@app.post("/predict")
def predict_sales(data: EnergyInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    drift_warnings = []
    
    # --- 5. DRIFT DETECTION LOGIC ---
    
    # Check 1: Temperature (Environmental Drift)
    t = data.temp_avg_f
    t_min = reference_stats.get("temp_avg_f", {}).get("min", 20)
    t_max = reference_stats.get("temp_avg_f", {}).get("max", 90)
    
    if t < t_min or t > t_max:
        msg = f"DRIFT ALERT: Temp {t}°F is outside historical 99% CI ({t_min:.1f}-{t_max:.1f})"
        drift_warnings.append(msg)
        logging.warning(msg) # Write to log file

    # Check 2: Economic Index (Population Shift)
    ipi = data.industrial_production_index
    i_min = reference_stats.get("industrial_production_index", {}).get("min", 80)
    i_max = reference_stats.get("industrial_production_index", {}).get("max", 120)
    
    if ipi < i_min or ipi > i_max:
        msg = f"DRIFT ALERT: IPI {ipi} is abnormal ({i_min:.1f}-{i_max:.1f})"
        drift_warnings.append(msg)
        logging.warning(msg) # Write to log file

    # --- 6. PREDICTION LOGIC ---
    try:
        input_data = {
            'price_cents_per_kwh': [data.price_cents_per_kwh],
            'energy_consumption_million_mmbtu': [data.energy_consumption_million_mmbtu],
            'public_holidays': [data.public_holidays],
            'saturdays': [data.saturdays],
            'sundays': [data.sundays],
            'weekend_days': [data.weekend_days],
            'holiday_on_weekend': [data.holiday_on_weekend],
            'total_off_days': [data.total_off_days],
            'industrial_production_index': [data.industrial_production_index],
            'temp_avg_f': [data.temp_avg_f],
            'heating_degree_days': [data.heating_degree_days],
            'cooling_degree_days': [data.cooling_degree_days],
            'precip_inches': [data.precip_inches],
            'month_num': [data.month_num]
        }
        df = pd.DataFrame(input_data)
        
        # Use .values to avoid column name mismatch issues
        prediction = model.predict(df.values)
        
        # Return both the Prediction AND the Drift Report
        return {
            "predicted_sales_mkwh": float(prediction[0]),
            "drift_alert": len(drift_warnings) > 0,
            "drift_messages": drift_warnings
        }

    except Exception as e:
        logging.error(f"Prediction Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))