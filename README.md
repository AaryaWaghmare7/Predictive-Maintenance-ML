# ⚡ EV Predictive Maintenance Platform

### AI-Powered Fault Diagnosis & Predictive Maintenance for Electric Vehicles

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

---

## 🚘 Project Overview

An intelligent, research-oriented **EV Predictive Maintenance Platform** that uses machine learning to detect faults in electric vehicle components and support maintenance decisions.

The project focuses on three major capabilities:

- 🔍 **Fault Diagnosis** — Identify motor, inverter, and battery faults.
- ⏳ **Remaining Useful Life (RUL)** — Estimate the remaining operational life of components.
- ⚠️ **Risk Assessment** — Support maintenance prioritization using fault and degradation information.

The system combines machine learning, sensor-data analysis, FastAPI, and a web-based dashboard.

---

## 🏗️ System Architecture

```text
             EV Sensor Data
                    │
                    ▼
           Data Preprocessing
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    Fault Diagnosis       RUL Prediction
    Random Forest          Regression
          │                   │
          └─────────┬─────────┘
                    ▼
              Risk Engine
                    │
                    ▼
          Maintenance Dashboard
```

*Fault diagnosis is complete; RUL and risk integration are under development.*

---

## 🧪 Experiment 1 — EV Failure Prediction

**Dataset:** EV Predictive Maintenance Dataset

| Property | Details |
|---|---|
| Records | 175,393 |
| Variables | 30 |
| Target 1 | Failure Probability |
| Target 2 | Maintenance Type |
| Models | Logistic Regression, Random Forest |

### Results

| Metric | Logistic Regression |
|---|---:|
| Accuracy | 50.05% |
| Failure Precision | 9.76% |
| Failure Recall | 48.91% |
| F1 Score | 16.27% |
| ROC-AUC | 49.12% |

Further signal analysis demonstrated extremely weak relationships between operational features and failure labels. Random Forest also performed near chance level.

Maintenance Type analysis similarly revealed insufficient predictive signal, so further classification was intentionally discontinued.

**Conclusion:** Dataset suitability and meaningful feature-target relationships are essential for reliable predictive maintenance.

---

## ⚡ Experiment 2 — EV Fault Diagnosis

**Dataset:** New Energy Vehicle Fault Diagnosis Dataset

| Property | Details |
|---|---|
| Total Samples | 11,000 |
| Training Samples | 7,700 |
| Testing Samples | 3,300 |
| Input Features | 7 |
| Fault Classes | 4 |

### Input Features

`Voltage` · `Current` · `Motor Speed` · `Temperature` · `Vibration` · `Ambient Temperature` · `Humidity`

### Fault Categories

- 🟢 Normal
- ⚙️ Motor Fault
- ⚡ Inverter Fault
- 🔋 Battery Fault

### Model Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 98.09% | 97.53% |
| **Random Forest** | **99.42%** | **99.21%** |
| Histogram Gradient Boosting | 99.33% | 99.08% |

### 🏆 Selected Model — Random Forest

Random Forest achieved the strongest overall classification performance and was selected for the fault-diagnosis module.

**Final Test Accuracy: 99.42%**  
**Final Macro F1: 99.21%**

These results are based on the Experiment 2 test dataset and do not imply guaranteed accuracy on real-world EV telemetry.

---

## ⏳ Experiment 3 — Remaining Useful Life

**Status: Under Development**

The next stage investigates run-to-failure bearing data to estimate component Remaining Useful Life.

Planned workflow:

1. Dataset auditing and degradation analysis
2. Vibration feature extraction
3. RUL target construction
4. Regression model comparison
5. Evaluation using MAE, RMSE, and R²
6. Integration with the fault-diagnosis dashboard

RUL predictions will refer to the supported component, not the entire vehicle.

---

## 🌐 Web Dashboard

The platform is being developed with the following capabilities:

- Interactive EV sensor input
- AI-powered fault classification
- Model confidence visualization
- CSV telemetry analysis
- Fault-distribution summaries
- Component RUL estimation *(in development)*
- Maintenance risk assessment *(planned)*

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Model Serialization | Joblib |
| Version Control | Git, GitHub |

---

## 🔬 Key Research Findings

- Dataset size alone does not guarantee predictive performance.
- Signal analysis helps determine whether a dataset is suitable for modelling.
- Class imbalance makes accuracy alone an unreliable evaluation metric.
- Random Forest achieved strong multiclass fault diagnosis on the NEV dataset.
- Fault diagnosis and RUL estimation require different targets and modelling approaches.
- Reliable predictive maintenance requires careful validation and transparent limitations.

---

## 📌 Project Progress

| Module | Status |
|---|---|
| Experiment 1 — Dataset & Signal Analysis | ✅ Complete |
| Experiment 2 — Fault Diagnosis | ✅ Complete |
| Random Forest Classification | ✅ Complete |
| Model Evaluation & Comparison | ✅ Complete |
| FastAPI & Dashboard | 🔄 Development |
| RUL Prediction | 🔄 Development |
| Risk Assessment | ⏳ Planned |
| Cloud Deployment | ⏳ Planned |

---

## ⚠️ Limitations

The current fault model uses normalized sensor values from the NEV dataset. Its original physical-unit normalization parameters are unavailable.

The model has not yet been validated across real-world EV fleets or manufacturers. It is intended as a **research proof of concept**, not an automotive safety-certified diagnostic system.

---

## 🎯 Project Vision

To develop a modular, explainable machine-learning platform that combines **EV fault diagnosis, component Remaining Useful Life estimation, and maintenance risk assessment** into an accessible predictive-maintenance dashboard.

---

<p align="center">
  <strong>⚡ Intelligent Diagnostics. Predictive Insights. Smarter Maintenance.</strong>
</p>
