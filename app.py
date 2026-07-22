import streamlit as st
import pandas as pd
from predict_leak import predict_leak

st.set_page_config(page_title="WaterSense AI - Model Testing", page_icon="💧", layout="centered")

st.title("💧 WaterSense AI: Model Testing")
st.markdown("Enter telemetry data below to test the Random Forest leak detection model.")

col1, col2 = st.columns(2)

with col1:
    flow_rate_lpm = st.number_input("Flow Rate (L/min)", value=8.5)
    pressure_bar = st.number_input("Pressure (bar)", value=2.3)
    temperature_c = st.number_input("Temperature (°C)", value=33.0)
    humidity_pct = st.number_input("Humidity (%)", value=65.0)
    hour_of_day = st.slider("Hour of Day", 0, 23, 10)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

with col2:
    is_weekend = st.selectbox("Is Weekend?", [0, 1], index=0)
    wifi_rssi_dbm = st.number_input("WiFi RSSI (dBm)", value=-65.0)
    flow_rolling_mean = st.number_input("Flow Rolling Mean (L/min)", value=8.4)
    flow_rolling_std = st.number_input("Flow Rolling Std", value=0.1)
    pressure_deviation = st.number_input("Pressure Deviation (bar)", value=0.0)
    flow_zscore = st.number_input("Flow Z-Score", value=0.2)

if st.button("Run Model Prediction", type="primary"):
    sensor_data = {
        "flow_rate_lpm": flow_rate_lpm,
        "pressure_bar": pressure_bar,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "wifi_rssi_dbm": wifi_rssi_dbm,
        "flow_rolling_mean": flow_rolling_mean,
        "flow_rolling_std": flow_rolling_std,
        "pressure_deviation": pressure_deviation,
        "flow_zscore": flow_zscore
    }
    
    with st.spinner("Analyzing telemetry..."):
        result = predict_leak(sensor_data)
        
    st.subheader("Prediction Results")
    
    if result["is_leak"]:
        st.error(f"🚨 **LEAK DETECTED!** (Risk: {result['risk_level'].upper()})")
    else:
        st.success(f"✅ **NORMAL** (Risk: {result['risk_level'].upper()})")
        
    st.metric(label="Leak Probability / Confidence", value=f"{result['confidence'] * 100:.1f}%")

st.markdown("---")
st.markdown("*Try increasing the **Flow Z-Score** (e.g., to 4.5) and lowering **Pressure Deviation** (e.g., to -0.4) to simulate a pipe burst.*")
