"""
Smart Water Leak Detection — Dataset Generator & Model Trainer
==============================================================
Generates a realistic synthetic dataset simulating 63 ESP32 sensor nodes
deployed across a campus, then trains a Random Forest classifier.

Authors: Nitesh Prajapati & Ananya Gupta
Project: ML-Based Smart Water Leakage Detection & Conservation System
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import sys
from datetime import datetime, timedelta

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# ── Configuration ──────────────────────────────────────────────────────────────
np.random.seed(42)

NUM_NODES = 63
DAYS = 14  # 14-day pilot period
READINGS_PER_HOUR = 6  # One reading every 10 minutes
TOTAL_HOURS = DAYS * 24
TOTAL_READINGS = TOTAL_HOURS * READINGS_PER_HOUR

# Building assignments for 63 nodes
BUILDINGS = {
    "Hostel_A": list(range(1, 14)),       # 13 nodes
    "Hostel_B": list(range(14, 27)),      # 13 nodes
    "Hostel_C": list(range(27, 37)),      # 10 nodes
    "Academic_Block_1": list(range(37, 49)),  # 12 nodes
    "Academic_Block_2": list(range(49, 58)),  # 9 nodes
    "Canteen_Complex": list(range(58, 64)),   # 6 nodes
}

# Node types
TANK_NODES = [1, 14, 27, 37, 49, 58]  # One ultrasonic tank sensor per building cluster
FLOW_NODES = [n for n in range(1, 64) if n not in TANK_NODES]

START_DATE = datetime(2026, 6, 20, 0, 0, 0)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Helper Functions ───────────────────────────────────────────────────────────

def get_building(node_id):
    """Return building name for a given node ID."""
    for building, nodes in BUILDINGS.items():
        if node_id in nodes:
            return building
    return "Unknown"

def get_node_type(node_id):
    """Return sensor type for a given node ID."""
    return "ultrasonic_tank" if node_id in TANK_NODES else "flow_meter"

def base_flow_rate(hour, building):
    """
    Simulate realistic time-of-day water usage patterns.
    Hostels: peaks at morning (6-9) and evening (18-22)
    Academic: peaks during class hours (9-17)
    Canteen: peaks at meal times (7-9, 12-14, 19-21)
    """
    if "Hostel" in building:
        if 6 <= hour <= 9:
            return np.random.normal(12.0, 2.5)   # Morning rush
        elif 18 <= hour <= 22:
            return np.random.normal(10.0, 2.0)   # Evening usage
        elif 0 <= hour <= 5:
            return np.random.normal(1.5, 0.5)    # Night (minimal)
        else:
            return np.random.normal(4.0, 1.5)    # Daytime (some usage)
    elif "Academic" in building:
        if 9 <= hour <= 17:
            return np.random.normal(8.0, 2.0)    # Class hours
        elif 6 <= hour <= 8:
            return np.random.normal(3.0, 1.0)    # Early morning
        else:
            return np.random.normal(1.0, 0.5)    # Off hours
    elif "Canteen" in building:
        if 7 <= hour <= 9:
            return np.random.normal(15.0, 3.0)   # Breakfast
        elif 12 <= hour <= 14:
            return np.random.normal(18.0, 3.5)   # Lunch
        elif 19 <= hour <= 21:
            return np.random.normal(16.0, 3.0)   # Dinner
        else:
            return np.random.normal(2.0, 1.0)    # Between meals
    return np.random.normal(5.0, 2.0)

def base_tank_level(hour, building):
    """Simulate tank level patterns (in cm, max ~200cm)."""
    if "Hostel" in building:
        if 6 <= hour <= 9:
            return np.random.normal(120, 15)   # Dropping (usage)
        elif 10 <= hour <= 14:
            return np.random.normal(160, 10)   # Refilling
        elif 18 <= hour <= 22:
            return np.random.normal(110, 15)   # Evening drop
        else:
            return np.random.normal(175, 8)    # Night (full)
    else:
        if 9 <= hour <= 17:
            return np.random.normal(130, 20)
        else:
            return np.random.normal(170, 10)
    
def ambient_temperature(hour, day):
    """Simulate Lucknow-like summer temperatures (°C)."""
    # Base: ~35°C peak, ~27°C low
    daily_variation = 4 * np.sin((hour - 6) * np.pi / 12)  # Peak at noon
    day_variation = np.random.normal(0, 1.5)
    return 31 + daily_variation + day_variation

def water_pressure(hour, building, is_leak=False):
    """Simulate water pressure in bar."""
    base = 2.5  # Normal operating pressure
    if "Hostel" in building and 6 <= hour <= 9:
        base -= 0.4  # Pressure drops during peak usage
    elif "Canteen" in building and 12 <= hour <= 14:
        base -= 0.3
    
    noise = np.random.normal(0, 0.15)
    if is_leak:
        base -= np.random.uniform(0.3, 0.8)  # Significant pressure drop during leaks
    
    return max(0.5, base + noise)


# ── Generate Dataset ───────────────────────────────────────────────────────────

print("=" * 65)
print("  Smart Water Leak Detection — Dataset Generator")
print("=" * 65)
print(f"\n  Nodes: {NUM_NODES} | Days: {DAYS} | Interval: 10 min")
print(f"  Start: {START_DATE.strftime('%Y-%m-%d %H:%M')}")
print(f"  Buildings: {len(BUILDINGS)}\n")

records = []
leak_events = []

# Pre-define leak events (realistic: ~5-8% of readings are anomalous)
# We'll create structured leak events (not random noise)
LEAK_SCENARIOS = [
    # (node_id, start_hour_offset, duration_hours, leak_type, severity)
    (5, 48, 6, "pipe_rupture", "high"),
    (5, 49, 5, "pipe_rupture", "high"),
    (12, 120, 18, "slow_leak", "low"),
    (19, 72, 12, "pipe_rupture", "medium"),
    (23, 168, 8, "joint_failure", "medium"),
    (30, 200, 24, "slow_leak", "low"),
    (31, 201, 20, "slow_leak", "low"),
    (42, 96, 4, "valve_malfunction", "high"),
    (45, 280, 10, "pipe_rupture", "medium"),
    (52, 144, 16, "slow_leak", "low"),
    (60, 240, 6, "overflow", "high"),
    (61, 241, 5, "overflow", "high"),
    (8, 310, 3, "pipe_rupture", "high"),
    (35, 155, 8, "joint_failure", "medium"),
    (15, 100, 14, "slow_leak", "low"),
    (55, 220, 7, "valve_malfunction", "medium"),
    (3, 180, 10, "slow_leak", "low"),
    (47, 260, 5, "pipe_rupture", "high"),
    (25, 130, 12, "slow_leak", "low"),
    (38, 290, 9, "joint_failure", "medium"),
]

def is_leak_active(node_id, hour_offset):
    """Check if a leak is active for a given node at a given time."""
    for lid, (nid, start_h, dur_h, ltype, severity) in enumerate(LEAK_SCENARIOS):
        if nid == node_id and start_h <= hour_offset < start_h + dur_h:
            return True, ltype, severity, lid
    return False, None, None, None

print("  Generating sensor readings...")

for hour_idx in range(TOTAL_HOURS):
    current_time = START_DATE + timedelta(hours=hour_idx)
    hour = current_time.hour
    day = current_time.day
    day_of_week = current_time.weekday()
    
    for reading_idx in range(READINGS_PER_HOUR):
        timestamp = current_time + timedelta(minutes=reading_idx * 10)
        
        for node_id in range(1, NUM_NODES + 1):
            building = get_building(node_id)
            node_type = get_node_type(node_id)
            
            # Check leak status
            leak_active, leak_type, severity, leak_id = is_leak_active(node_id, hour_idx)
            
            # Generate sensor readings
            temp = ambient_temperature(hour, day)
            pressure = water_pressure(hour, building, leak_active)
            
            if node_type == "flow_meter":
                flow = base_flow_rate(hour, building)
                tank_level = np.nan  # Flow nodes don't have tank level
                
                if leak_active:
                    if severity == "high":
                        flow += np.random.uniform(8, 20)    # Major spike
                    elif severity == "medium":
                        flow += np.random.uniform(4, 10)    # Moderate spike  
                    else:  # low (slow leak)
                        flow += np.random.uniform(1.5, 4)   # Subtle increase
                
                flow = max(0, flow)
                
                # Compute derived features
                flow_deviation = flow - base_flow_rate(hour, building)
                
            else:  # ultrasonic_tank
                flow = np.nan
                tank_level = base_tank_level(hour, building)
                
                if leak_active:
                    if leak_type == "overflow":
                        tank_level = min(200, tank_level + np.random.uniform(15, 30))
                    else:
                        tank_level -= np.random.uniform(10, 30)  # Unexpected drop
                
                tank_level = np.clip(tank_level, 5, 200)
                flow_deviation = np.nan
            
            # Weekend factor (lower usage on weekends)
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Humidity (correlated with temperature)
            humidity = max(30, min(95, 80 - (temp - 30) * 2 + np.random.normal(0, 5)))
            
            # Battery voltage (simulate slight degradation)
            battery_voltage = 4.2 - (hour_idx / TOTAL_HOURS) * 0.3 + np.random.normal(0, 0.05)
            battery_voltage = np.clip(battery_voltage, 3.3, 4.2)
            
            # WiFi signal strength (dBm)
            wifi_rssi = np.random.normal(-55, 12)
            if building == "Hostel_C":  # Known dead zone
                wifi_rssi -= 15
            wifi_rssi = np.clip(wifi_rssi, -90, -20)
            
            record = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "node_id": node_id,
                "building": building,
                "node_type": node_type,
                "flow_rate_lpm": round(flow, 3) if not np.isnan(flow) else np.nan,
                "tank_level_cm": round(tank_level, 2) if not np.isnan(tank_level) else np.nan,
                "pressure_bar": round(pressure, 3),
                "temperature_c": round(temp, 1),
                "humidity_pct": round(humidity, 1),
                "hour_of_day": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "battery_voltage": round(battery_voltage, 2),
                "wifi_rssi_dbm": round(wifi_rssi, 1),
                "is_leak": 1 if leak_active else 0,
                "leak_type": leak_type if leak_active else "none",
                "severity": severity if leak_active else "none",
            }
            records.append(record)

df = pd.DataFrame(records)

# ── Feature Engineering ────────────────────────────────────────────────────────

print("  Engineering features...")

# Rolling averages per node (simulate what the model would see)
# We'll compute these as grouped statistics
df_sorted = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

# For flow nodes: compute rolling stats
flow_mask = df_sorted["node_type"] == "flow_meter"
tank_mask = df_sorted["node_type"] == "ultrasonic_tank"

# Add rolling mean flow (30-min window = 3 readings)
df_sorted["flow_rolling_mean"] = np.nan
df_sorted["flow_rolling_std"] = np.nan
df_sorted["tank_rolling_mean"] = np.nan
df_sorted["tank_rate_of_change"] = np.nan

for node_id in range(1, NUM_NODES + 1):
    node_mask = df_sorted["node_id"] == node_id
    if node_id not in TANK_NODES:
        flow_vals = df_sorted.loc[node_mask, "flow_rate_lpm"]
        df_sorted.loc[node_mask, "flow_rolling_mean"] = flow_vals.rolling(3, min_periods=1).mean().round(3)
        df_sorted.loc[node_mask, "flow_rolling_std"] = flow_vals.rolling(6, min_periods=1).std().round(3)
    else:
        tank_vals = df_sorted.loc[node_mask, "tank_level_cm"]
        df_sorted.loc[node_mask, "tank_rolling_mean"] = tank_vals.rolling(3, min_periods=1).mean().round(2)
        df_sorted.loc[node_mask, "tank_rate_of_change"] = tank_vals.diff().round(2)

# Pressure deviation from building mean
building_pressure_mean = df_sorted.groupby(["building", "hour_of_day"])["pressure_bar"].transform("mean")
df_sorted["pressure_deviation"] = (df_sorted["pressure_bar"] - building_pressure_mean).round(3)

# Flow anomaly score (z-score within building + hour)
flow_building_mean = df_sorted.groupby(["building", "hour_of_day"])["flow_rate_lpm"].transform("mean")
flow_building_std = df_sorted.groupby(["building", "hour_of_day"])["flow_rate_lpm"].transform("std")
df_sorted["flow_zscore"] = ((df_sorted["flow_rate_lpm"] - flow_building_mean) / flow_building_std.replace(0, 1)).round(3)

df = df_sorted

# ── Save Dataset ───────────────────────────────────────────────────────────────

dataset_path = os.path.join(OUTPUT_DIR, "water_leak_dataset.csv")
df.to_csv(dataset_path, index=False)

total_rows = len(df)
leak_rows = df["is_leak"].sum()
leak_pct = (leak_rows / total_rows) * 100

print(f"\n  ✅ Dataset saved: water_leak_dataset.csv")
print(f"     Total records: {total_rows:,}")
print(f"     Leak records:  {leak_rows:,} ({leak_pct:.1f}%)")
print(f"     Normal records: {total_rows - leak_rows:,} ({100 - leak_pct:.1f}%)")
print(f"     Features: {len(df.columns)}")

# Also save a smaller summary dataset
summary = df.groupby(["building", "node_type", "is_leak"]).agg(
    count=("node_id", "size"),
    avg_flow=("flow_rate_lpm", "mean"),
    avg_tank=("tank_level_cm", "mean"),
    avg_pressure=("pressure_bar", "mean"),
).round(3).reset_index()
summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"  ✅ Summary saved: dataset_summary.csv")

# ── Train Random Forest Model ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  Training Random Forest Classifier")
print("=" * 65)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Prepare features for ML
# Use only flow_meter nodes for the primary leak detection model
df_flow = df[df["node_type"] == "flow_meter"].copy()

FEATURE_COLUMNS = [
    "flow_rate_lpm",
    "pressure_bar",
    "temperature_c",
    "humidity_pct",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "wifi_rssi_dbm",
    "flow_rolling_mean",
    "flow_rolling_std",
    "pressure_deviation",
    "flow_zscore",
]

TARGET = "is_leak"

# Drop rows with NaN in features
df_ml = df_flow[FEATURE_COLUMNS + [TARGET]].dropna()

X = df_ml[FEATURE_COLUMNS]
y = df_ml[TARGET]

print(f"\n  Samples: {len(X):,}")
print(f"  Features: {len(FEATURE_COLUMNS)}")
print(f"  Class distribution: Normal={sum(y==0):,}, Leak={sum(y==1):,}")

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train set: {len(X_train):,} | Test set: {len(X_test):,}")

# Train Random Forest
print("\n  Training model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",  # Handle class imbalance
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n  ── Model Performance ──")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Leak"]))

print(f"  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")

fp_rate = cm[0][1] / (cm[0][0] + cm[0][1]) * 100
print(f"\n  False Positive Rate: {fp_rate:.1f}%")

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="f1")
print(f"  5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
importances = rf_model.feature_importances_
feat_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)

print(f"\n  ── Feature Importance ──")
for feat, imp in feat_imp:
    bar = "█" * int(imp * 50)
    print(f"    {feat:<22s} {imp:.4f}  {bar}")

# ── Save Model & Artifacts ────────────────────────────────────────────────────

# Save trained model
model_path = os.path.join(OUTPUT_DIR, "leak_detection_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(rf_model, f)
print(f"\n  ✅ Model saved: leak_detection_model.pkl")

# Save model metadata
model_meta = {
    "model_type": "RandomForestClassifier",
    "n_estimators": 200,
    "max_depth": 15,
    "features": FEATURE_COLUMNS,
    "target": TARGET,
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "metrics": {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "false_positive_rate_pct": round(fp_rate, 1),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
    },
    "confusion_matrix": {
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    },
    "feature_importance": {feat: round(imp, 4) for feat, imp in feat_imp},
    "class_distribution": {
        "normal": int(sum(y == 0)),
        "leak": int(sum(y == 1)),
    },
    "dataset_info": {
        "total_records": total_rows,
        "num_nodes": NUM_NODES,
        "num_buildings": len(BUILDINGS),
        "pilot_duration_days": DAYS,
        "reading_interval_minutes": 10,
    },
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "authors": "Nitesh Prajapati & Ananya Gupta",
    "project": "ML-Based Smart Water Leakage Detection & Conservation System",
}

meta_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
with open(meta_path, "w") as f:
    json.dump(model_meta, f, indent=2)
print(f"  ✅ Metadata saved: model_metadata.json")

# Save feature columns for inference
features_path = os.path.join(OUTPUT_DIR, "feature_columns.json")
with open(features_path, "w") as f:
    json.dump(FEATURE_COLUMNS, f, indent=2)
print(f"  ✅ Features saved: feature_columns.json")

# ── Also train a tank-level anomaly model ──────────────────────────────────────

print("\n" + "=" * 65)
print("  Training Tank Level Anomaly Model")
print("=" * 65)

df_tank = df[df["node_type"] == "ultrasonic_tank"].copy()

TANK_FEATURES = [
    "tank_level_cm",
    "pressure_bar",
    "temperature_c",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "tank_rolling_mean",
    "tank_rate_of_change",
    "pressure_deviation",
]

df_tank_ml = df_tank[TANK_FEATURES + [TARGET]].dropna()
X_tank = df_tank_ml[TANK_FEATURES]
y_tank = df_tank_ml[TARGET]

X_tank_train, X_tank_test, y_tank_train, y_tank_test = train_test_split(
    X_tank, y_tank, test_size=0.2, random_state=42, stratify=y_tank
)

rf_tank = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_tank.fit(X_tank_train, y_tank_train)

y_tank_pred = rf_tank.predict(X_tank_test)
tank_f1 = f1_score(y_tank_test, y_tank_pred)
tank_acc = accuracy_score(y_tank_test, y_tank_pred)

print(f"\n  Tank Model — Accuracy: {tank_acc:.4f}, F1: {tank_f1:.4f}")

tank_model_path = os.path.join(OUTPUT_DIR, "tank_anomaly_model.pkl")
with open(tank_model_path, "wb") as f:
    pickle.dump(rf_tank, f)
print(f"  ✅ Tank model saved: tank_anomaly_model.pkl")

# ── Create inference script ────────────────────────────────────────────────────

inference_script = '''"""
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
            - temperature_c: float (ambient temperature °C)
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
    print("  Smart Water Leak Detection — Inference Demo")
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
    print(f"\\n  Normal Reading:")
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
    print(f"\\n  Leak Reading:")
    print(f"    Leak Detected: {result['is_leak']}")
    print(f"    Confidence:    {result['confidence']:.1%}")
    print(f"    Risk Level:    {result['risk_level']}")
    
    print(f"\\n  ✅ Inference complete!")
'''

inference_path = os.path.join(OUTPUT_DIR, "predict_leak.py")
with open(inference_path, "w", encoding="utf-8") as f:
    f.write(inference_script)
print(f"\n  [OK] Inference script saved: predict_leak.py")

# ── Final Summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  ALL FILES GENERATED SUCCESSFULLY")
print("=" * 65)
print(f"""
  📁 Output Files:
  ├── water_leak_dataset.csv      ({os.path.getsize(dataset_path) / 1024 / 1024:.1f} MB)  — Full sensor dataset
  ├── dataset_summary.csv         — Aggregated summary statistics
  ├── leak_detection_model.pkl    — Trained RF model (flow meters)
  ├── tank_anomaly_model.pkl      — Trained RF model (tank sensors)
  ├── model_metadata.json         — Model performance & config
  ├── feature_columns.json        — Feature list for inference
  └── predict_leak.py             — Ready-to-use inference script

  📊 Dataset Stats:
  ├── Total records:    {total_rows:,}
  ├── Sensor nodes:     {NUM_NODES}
  ├── Buildings:        {len(BUILDINGS)}
  └── Pilot duration:   {DAYS} days

  🤖 Flow Leak Model:
  ├── F1-Score:         {f1:.4f}
  ├── Accuracy:         {accuracy:.4f}
  ├── ROC-AUC:          {roc_auc:.4f}
  └── FP Rate:          {fp_rate:.1f}%

  🛢️ Tank Anomaly Model:
  ├── F1-Score:         {tank_f1:.4f}
  └── Accuracy:         {tank_acc:.4f}
""")
