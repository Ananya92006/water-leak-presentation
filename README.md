# 🌊 WaterSense AI — Smart Water Leak Detection & Conservation Software Platform

[![ML Model: Random Forest](https://img.shields.io/badge/ML%20Engine-Random%20Forest%20v2.0-00e5cc?style=for-the-badge&logo=scikit-learn)](predict_leak.py)
[![F1 Score: 89.8%](https://img.shields.io/badge/F1%20Score-89.8%25-2ecc71?style=for-the-badge)](model_metadata.json)
[![Status: Phase 1 Ready](https://img.shields.io/badge/Status-Phase%201%20SaaS%20Rollout-7c5cfc?style=for-the-badge)](index.html)
[![UN SDGs](https://img.shields.io/badge/UN%20SDGs-6%20%C2%B7%209%20%C2%B7%2011%20%C2%B7%2013-ff7043?style=for-the-badge)](index.html)

**WaterSense AI** is a hardware-agnostic, machine-learning-powered water leakage detection and conservation software platform developed originally for campus utility monitoring at **BBDNIIT (AKTU)** and scaling across partner universities (**MMMUT**). 

By transforming standard raw sensor feeds (pulse flow meters, ultrasonic tank levels, and pressure transducers) into actionable intelligence, WaterSense AI identifies underground pipe fractures, valve anomalies, and slow degradation before major structural damage occurs.

---

## 🚀 Key Platform Capabilities

- **🧠 High-Precision ML Engine (`leak_detection_model.pkl`)**
  - **Random Forest Classifier** trained on **127,008** multi-sensor time-series readings.
  - **Accuracy:** `99.79%` | **F1-Score:** `89.8%` | **False Positive Rate:** `< 0.10%`
  - Eliminates alarm fatigue by correlating 10-second rolling flow Z-scores (`flow_zscore`), pressure deviations (`pressure_deviation`), and ultrasonic tank depletion rates.

- **📊 Interactive Executive Presentation Suite (`index.html`)**
  - Dark-themed, responsive 12-slide web presentation built with vanilla HTML/CSS (`open index.html` directly in any web browser).
  - Includes full interactive charts, timeline Gantt roadmaps, system architecture layouts, and financial ROI calculators.
  - Accompanied by a print-optimized layout (`print_version.html` / `WaterSense_AI_ScalingStrategy.pdf`) for executive board review.

- **🌱 Three-Phase Software & Analytics Roadmap**
  - **Phase 1 (Months 1–3):** Campus Core Analytics (`63 → 150 Telemetry Feeds`) — software API connectors, edge aggregation scripts, and auto-calibration drift models without new custom hardware budgets.
  - **Phase 2 (Months 4–8):** Full Campus Software Suite (`150 → 500 Feeds`) — ERP/billing integration, pressure correlation algorithms, and mobile dispatch app.
  - **Phase 3 (Months 9–18):** Multi-Campus SaaS Platform (`500 → 2,000+ Feeds`) — federated ML across MMMUT and regional open API marketplace.

---

## 📁 Repository Structure

```text
water-leak-presentation/
├── index.html                           # Interactive 12-slide dark-themed presentation
├── print_version.html                   # Print-ready executive presentation layout
├── WaterSense_AI_ScalingStrategy.pdf    # Executive board PDF presentation
├── Nitesh_Ananya_ScalingStrategy.pdf    # Alternate PDF report export
├── generate_dataset.py                  # Telemetry simulation & Random Forest training pipeline
├── predict_leak.py                      # Standalone CLI inference script for live sensor feeds
├── water_leak_dataset.csv               # Full training dataset (127,008 sensor readings)
├── dataset_summary.csv                  # Summarized statistical metrics across features
├── leak_detection_model.pkl             # Serialized Scikit-Learn Random Forest model (v2.0)
├── tank_anomaly_model.pkl               # Auxiliary ultrasonic tank anomaly detector
├── feature_columns.json                 # Ordered JSON schema of ML model features
└── model_metadata.json                  # Performance metrics, thresholds, & model parameters
```

---

## 🛠️ Quickstart Guide

### 1. View the Interactive Presentation
Simply double-click `index.html` or open it in your favorite web browser (Chrome, Edge, Firefox, Safari):
```bash
# On Windows
start index.html
```

### 2. Run the Machine Learning Pipeline Locally
Ensure Python 3.8+ is installed with `scikit-learn`, `pandas`, and `numpy`:
```bash
pip install pandas numpy scikit-learn
```

#### Run Live Telemetry Inference:
```bash
python predict_leak.py
```
*Outputs real-time anomaly scores, leakage probabilities, and risk levels using the pre-trained `.pkl` model.*

#### Regenerate Dataset & Retrain Models from Scratch:
```bash
python generate_dataset.py
```

---

## 💰 Financial & Environmental Impact

| Component / Phase | Pilot (63 Feeds) | Phase 1 (150 Feeds) | Phase 2 (500 Feeds) | Phase 3 (2,000+ SaaS) |
| :--- | :---: | :---: | :---: | :---: |
| **Software Cost per Feed** | ₹1,800 | **₹1,300** | ₹900 | ₹600 |
| **Projected Annual Savings** | ₹4–6L | **₹12–18L** | ₹30–45L | ₹1.2–1.8 Cr |
| **Payback Period** | — | **14–18 Months** | 10–14 Months | < 9 Months |

---

## 👥 Authors & Project Leadership

- **Nitesh Prajapati** — BBDNIIT (AKTU)  
  *Role: IoT Telemetry & System Architecture*
- **Ananya Gupta** — MMMUT  
  *Role: Machine Learning & AI Analytics Suite*

**Program Affiliation:**  
*1M1B Green Skills & Applied AI for Climate Action*
