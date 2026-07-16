"""
Smart Water Leak Detection — Inference Script
==============================================
Load the trained model and run predictions on new sensor data.

Usage:
    python predict_leak.py
    
    Or import in your code:
        from predict_leak import predict_leak
"""

import pickle
import json
import numpy as np
import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(model_path=None):
    """Load the trained Random Forest model."""
    if model_path is None:
        model_path = os.path.join(SCRIPT_DIR, "leak_detection_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_features(features_path=None):
    """Load the feature column list."""
    if features_path is None:
        features_path = os.path.join(SCRIPT_DIR, "feature_columns.json")
    with open(features_path, "r") as f:
        return json.load(f)

def predict_leak(sensor_data: dict) -> dict:
    """
    Predict whether a leak is occurring based on sensor readings.
    
    Args:
        sensor_data: dict with keys matching feature columns:
            - flow_rate_lpm: float (current flow rate in L/min)
            - pressure_bar: float (water pressure in bar)
            - temperature_c: float (ambient temperature C)
            - humidity_pct: float (humidity %)
            - hour_of_day: int (0-23)
            - day_of_week: int (0=Mon, 6=Sun)
            - is_weekend: int (0 or 1)
            - wifi_rssi_dbm: float (WiFi signal strength)
            - flow_rolling_mean: float (30-min rolling average flow)
            - flow_rolling_std: float (1-hour rolling std of flow)
            - pressure_deviation: float (deviation from building avg)
            - flow_zscore: float (z-score of flow vs building baseline)
    
    Returns:
        dict with:
            - is_leak: bool
            - confidence: float (0-1)
            - risk_level: str ("low", "medium", "high")
    """
    model = load_model()
    features = load_features()
    
    # Prepare input
    X = pd.DataFrame([sensor_data])[features]
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    # Risk level
    if probability >= 0.85:
        risk = "high"
    elif probability >= 0.5:
        risk = "medium"
    else:
        risk = "low"
    
    return {
        "is_leak": bool(prediction),
        "confidence": round(float(probability), 4),
        "risk_level": risk,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  Smart Water Leak Detection -- Inference Demo")
    print("=" * 55)
    
    # Example 1: Normal reading
    normal_reading = {
        "flow_rate_lpm": 8.5,
        "pressure_bar": 2.3,
        "temperature_c": 33.0,
        "humidity_pct": 65.0,
        "hour_of_day": 10,
        "day_of_week": 2,
        "is_weekend": 0,
        "wifi_rssi_dbm": -52.0,
        "flow_rolling_mean": 8.2,
        "flow_rolling_std": 1.5,
        "pressure_deviation": -0.05,
        "flow_zscore": 0.3,
    }
    
    result = predict_leak(normal_reading)
    print(f"\n  Normal Reading:")
    print(f"    Leak Detected: {result['is_leak']}")
    print(f"    Confidence:    {result['confidence']:.1%}")
    print(f"    Risk Level:    {result['risk_level']}")
    
    # Example 2: Leak reading
    leak_reading = {
        "flow_rate_lpm": 22.0,
        "pressure_bar": 1.6,
        "temperature_c": 34.0,
        "humidity_pct": 70.0,
        "hour_of_day": 3,
        "day_of_week": 1,
        "is_weekend": 0,
        "wifi_rssi_dbm": -58.0,
        "flow_rolling_mean": 18.5,
        "flow_rolling_std": 5.2,
        "pressure_deviation": -0.6,
        "flow_zscore": 3.8,
    }
    
    result = predict_leak(leak_reading)
    print(f"\n  Leak Reading:")
    print(f"    Leak Detected: {result['is_leak']}")
    print(f"    Confidence:    {result['confidence']:.1%}")
    print(f"    Risk Level:    {result['risk_level']}")
    
    print(f"\n  [OK] Inference complete!")
