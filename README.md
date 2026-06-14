# silicon-guard
Demoing an E2E machine learning pipeline for predicting the **Remaining Useful Life (RUL)** of semiconductor manufacturing equipment

Executive Summary

SiliconGuard is an end-to-end predictive maintenance system designed to mitigate the catastrophic costs associated with unplanned downtime in semiconductor manufacturing. By leveraging sensor telemetry (temperature/vibration) and machine learning, the system predicts the Remaining Useful Life (RUL) of critical lithography equipment, allowing for a transition from expensive reactive "break-fix" cycles to high-margin proactive maintenance strategies.

Key Business Outcome: Demonstrates a projected 95% Return on Investment (ROI) by reducing unplanned failure costs and optimizing maintenance schedules.

The Problem

In semiconductor fabrication, unexpected equipment failure can cost hundreds of thousands of dollars per hour in lost production and damaged wafers. Traditional maintenance is either:

Reactive: High-cost, high-impact emergency repairs.
Preventative (Fixed Interval): Wasteful replacement of perfectly functional components.
The Solution

SiliconGuard implements a Predictive Maintenance (PdM) architecture. Using a Random Forest Regressor trained on engineered temporal features, the system identifies subtle degradation patterns in temperature streams before they reach critical failure thresholds.

Technical Architecture

Data Pipeline: Automated generation of high-fidelity sensor telemetry with simulated stochastic noise and thermal degradation trends.
Feature Engineering: Implementation of multi-scale rolling statistics (6h and 24h windows) to capture temporal volatility and trend shifts.
Predictive Engine: A regression-based model predicting Remaining Useful Life (RUL) in hours.
Economic Intelligence (ROI Engine): A dedicated module that translates ML metrics (RMSE/MAE) into financial KPIs, comparing the cost of downtime vs. the cost of scheduled intervention.





silicon_guard/
├── src/
│   ├── data_gen.py        # Synthetic sensor telemetry generator
│   ├── features.py        # Feature engineering (moving averages, volatility)
│   ├── model.py           # Random Forest RUL prediction engine
│   └── roi_template.py    # Business impact & ROI calculator

├── models/                # Trained model artifacts (.joblib)
├── data/                 # Raw and processed sensor datasets
├── notebooks/             # EDA, Model Validation, and Business Deep-dives
├── tests/                 # Unit tests for engineering pipelines
└── .gitignore             # Production-ready exclusion rules
