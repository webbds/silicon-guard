import os
import sys
import pandas as pd

# Add src to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_gen import generate_sensor_data
from src.model import RULPredictor
from src.roi_engine import ROIEngine

def main():
    # 1. Data Generation
    print("--- Step 1: Generating Raw Sensor Data ---")
    raw_data_path = 'silicon_guard/data/raw/sensor_data.csv'
    # The data_gen script itself handles the saving, we just call the function
    generate_sensor_data(n_days=120) 
    import subprocess
    subprocess.run(['python3', 'silicon_guard/src/data_gen.py'], check=True)
    print(f"Data generated at {raw_data_path}")

    # 2. Model Training
    print("\n--- Step 2: Training ML Model (Random Forest) ---")
    predictor = RULPredictor()
    model_path = predictor.train(raw_data_path)
    print(f"Model trained and saved at {model_path}")

    # 3. Inference/Prediction
    print("\n--- Step 3: Running Inference for ROI Calculation ---")
    predictions_df = predictor.predict(raw_data_path)
    print("Inference complete. Analyzing predictions...")

    # 4. ROI Analysis
    print("\n--- Step 4: Calculating Business Impact (ROI) ---")
    roi_engine = ROIEngine()
    results = roi_engine.calculate_impact(predictions_df)
    
    print("\n==========================================")
    print("      SILICONGUARD PROJECT RESULTS       ")
    print("==========================================")
    print(f"Reactive Total Cost:   ${results['reactive_total_cost']:,.2f}")
    print(f"Predictive Total Cost: ${results['predictive_total_cost']:,.2f}")
    print(f"Net Annual Savings:    ${results['net_savings']:,.2f}")
    print(f"ROI Percentage:        {results['roi_percentage']:.2f}%")
    print("==========================================\n")

if __name__ == "__main__":
    main()
