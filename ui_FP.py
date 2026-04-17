import streamlit as st
import requests
import calendar
from datetime import date
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# 1. PAGE CONFIG & AGGRESSIVE CSS REMOVER
st.set_page_config(page_title="Energy Forecast", layout="centered")
st.markdown("""
    <style>
        /* 1. Remove the huge white gap at the top */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            margin-top: 0rem !important;
        }
        /* 2. Hide the top "Manage App" header bar */
        header {visibility: hidden;}
        /* 3. Remove footer */
        footer {visibility: hidden;}
        /* 4. Tighten header spacing */
        h1 { margin-bottom: 0px !important; padding-bottom: 0px !important; }
    </style>
""", unsafe_allow_html=True)

# 2. TITLE (Very Compact)
st.title("⚡ Utility Sales Forecast")
st.caption("Foundation Project | Group 10")

# 3. DATE SELECTION (Hidden in Expander to save space)
with st.expander("📅 Change Forecast Date", expanded=False):
    sel_date = st.date_input("Target Month", date.today())
    year = sel_date.year
    month = sel_date.month
    matrix = calendar.monthcalendar(year, month)
    sats = sum(1 for x in matrix if x[calendar.SATURDAY] != 0)
    suns = sum(1 for x in matrix if x[calendar.SUNDAY] != 0)
    weekends = sats + suns
    holidays = 1 if month in [1, 7, 11, 12] else 0 
    off_days = weekends + holidays
    st.info(f"Using Calendar: {calendar.month_name[month]} {year} ({sats} Sats, {suns} Suns)")

# 4. INPUTS (3 Rows, Very Compact)
with st.form("main_form"):
    
    st.write(" **1. Economics & Fuel**")
    c1, c2, c3 = st.columns(3)
    with c1: price = st.number_input("Price (¢)", 10.5)
    with c2: consump = st.number_input("Fuel (MMBtu)", 3000.0)
    with c3: ipi = st.number_input("IPI Index", 102.5)

    st.write(" **2. Weather**")
    w1, w2, w3, w4 = st.columns(4)
    with w1: temp = st.number_input("Temp(°F)", 75.0)
    with w2: precip = st.number_input("Rain(in)", 2.1)
    with w3: hdd = st.number_input("HDD", 10.0)
    with w4: cdd = st.number_input("CDD", 200.0)

    # Payload
    payload = {
        "price_cents_per_kwh": price,
        "energy_consumption_million_mmbtu": consump,
        "public_holidays": holidays,
        "saturdays": sats,
        "sundays": suns,
        "weekend_days": weekends,
        "holiday_on_weekend": 0,
        "total_off_days": off_days,
        "industrial_production_index": ipi,
        "temp_avg_f": temp,
        "heating_degree_days": hdd,
        "cooling_degree_days": cdd,
        "precip_inches": precip,
        "month_num": month
    }
    
    st.write("") 
    submit = st.form_submit_button("🚀 PREDICT SALES", use_container_width=True)

# 5. RESULTS
if submit:
    try:
        res = requests.post(API_URL, json=payload)
        if res.status_code == 200:
            data = res.json()
            st.divider()
            
            # Result
            c_res, c_alert = st.columns([1, 2])
            with c_res:
                st.metric("Forecast (mkWh)", f"{data['predicted_sales_mkwh']:,.0f}")
            
            # Drift Alert
            with c_alert:
                if data['drift_alert']:
                    st.error("⚠️ DATA DRIFT DETECTED")
                    for msg in data['drift_messages']:
                        st.caption(f"🔴 {msg}")
                else:
                    st.success("✅ System Status: Normal")
    except Exception as e:
        st.error(f"Error: {e}")