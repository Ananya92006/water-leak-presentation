import time
from predict_leak import predict_leak

def print_separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# 1. Explain the Input Fields
print_separator("INPUT FIELD DEFINITIONS")
print("""Here is what each key telemetry input field means:

• Flow Rate (L/min): 
  The raw amount of water flowing through the pipe right now.
  
• Pressure (bar): 
  The current water pressure in the line. Normal is usually 2.0 - 3.0 bar.
  
• Flow Rolling Mean (L/min): 
  The average flow rate over the last 30 minutes. Helps establish a baseline.
  
• Flow Rolling Std: 
  The standard deviation (volatility) of the flow over the last hour.
  
• Pressure Deviation (bar): 
  How much the current pressure differs from the historical average for this time.
  Negative values indicate a pressure drop (common during leaks).
  
• Flow Z-Score: 
  A statistical measure of how unusual the current flow is. 
  0 = totally normal, 2 = unusual, 4+ = highly anomalous (likely a leak).
""")

time.sleep(1)

# 2. Define the Situations (Scenarios)
scenarios = {
    "🟢 SCENARIO 1: NORMAL CONDITIONS (Low Risk)": {
        "description": "Standard water usage during the day. Flow and pressure are stable and match historical baselines.",
        "inputs": {
            "flow_rate_lpm": 8.5,
            "pressure_bar": 2.4,
            "temperature_c": 33.0,
            "humidity_pct": 65.0,
            "hour_of_day": 10,
            "day_of_week": 2,
            "is_weekend": 0,
            "wifi_rssi_dbm": -65.0,
            "flow_rolling_mean": 8.4,
            "flow_rolling_std": 0.2,
            "pressure_deviation": 0.0,
            "flow_zscore": 0.2
        }
    },
    "🟡 SCENARIO 2: SLOW DRIPPING LEAK (Medium Risk)": {
        "description": "A toilet is running in the middle of the night. The flow isn't huge, but it's highly unusual for 3:00 AM.",
        "inputs": {
            "flow_rate_lpm": 14.5,
            "pressure_bar": 2.1,
            "temperature_c": 28.0,
            "humidity_pct": 89.0,
            "hour_of_day": 3,
            "day_of_week": 2,
            "is_weekend": 0,
            "wifi_rssi_dbm": -65.0,
            "flow_rolling_mean": 4.0,
            "flow_rolling_std": 0.5,
            "pressure_deviation": -0.3,
            "flow_zscore": 2.5
        }
    },
    "🔴 SCENARIO 3: MAJOR PIPE BURST (High Risk)": {
        "description": "A main pipe has fractured. Water is rushing out extremely fast, causing a massive pressure drop in the system.",
        "inputs": {
            "flow_rate_lpm": 35.0,
            "pressure_bar": 1.2,
            "temperature_c": 35.0,
            "humidity_pct": 50.0,
            "hour_of_day": 14,
            "day_of_week": 4,
            "is_weekend": 0,
            "wifi_rssi_dbm": -70.0,
            "flow_rolling_mean": 12.5,
            "flow_rolling_std": 4.8,
            "pressure_deviation": -0.9,
            "flow_zscore": 5.2
        }
    }
}

# 3. Run the Scenarios through the Model
print_separator("RUNNING SCENARIOS THROUGH THE ML MODEL")

for scenario_name, data in scenarios.items():
    print(f"\n{scenario_name}")
    print(f"Description: {data['description']}")
    print("-" * 40)
    
    # Highlight the key inputs that drive the model
    print(f"Key Inputs Provided:")
    print(f"  - Flow Rate:          {data['inputs']['flow_rate_lpm']} L/min")
    print(f"  - Pressure:           {data['inputs']['pressure_bar']} bar")
    print(f"  - Pressure Deviation: {data['inputs']['pressure_deviation']} bar")
    print(f"  - Flow Z-Score:       {data['inputs']['flow_zscore']}")
    
    print("\nAnalyzing telemetry...")
    time.sleep(1) # Small pause for effect
    
    # Call the ML model
    result = predict_leak(data['inputs'])
    
    # Format output
    if result["is_leak"]:
        print(f"--> RESULT: 🚨 LEAK DETECTED (Risk: {result['risk_level'].upper()})")
    else:
        print(f"--> RESULT: ✅ NORMAL (Risk: {result['risk_level'].upper()})")
        
    print(f"--> CONFIDENCE: {result['confidence'] * 100:.1f}%\n")
    time.sleep(1)

print("=" * 60)
print("  Simulation Complete!")
print("=" * 60)
