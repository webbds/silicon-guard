import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from src.features import engineer_features

class RULPredictor:
    """
    A simple but powerful predictor for Remaining Useful Life (RUL).
    It uses a Random Forest to learn how temperature trends signal failure.
    """
    def __init__(self, model_path='silicon_guard/models/model.joblib'):
        self.model_path = model_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, raw_data_path):
        print(f"Step 1: Loading data from {raw_data_path}...")
        df = pd.read_csv(raw_data_path, parse_dates=['timestamp'])
        
        print("Step 2: Engineering features...")
        processed_df = engineer_features(df)
        
        # Identify our input features and the target (what we want to predict)
        feature_cols = [col for col in processed_df.columns if 'temp_rolling' in col]
        X = processed_df[feature_cols]
        y = processed_df['RUL_hours']
        
        print(f"Step 3: Training model on {len(X)} samples...")
        self.model.fit(X, y)
        
        # Save the trained model and the feature list for later use
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'features': feature_cols}, self.model_path)
        print(f"Step 4: Model saved to {self.model_path}")
        return self.model_path

    def predict(self, raw_data_path):
        """Uses the trained model to estimate RUL for new sensor readings."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("No trained model found! Run .train() first.")
            
        print(f"Loading model from {self.model_path}...")
        payload = joblib.load(self.model_path)
        model = payload['model']
        feature_cols = payload['features']
        
        df = pd.read_csv(raw_data_path, parse_dates=['timestamp'])
        processed_df = engineer_features(df)
        X = processed_df[feature_cols]
        
        predictions = model.predict(X)
        processed_df['predicted_RUL'] = predictions
        return processed_df

if __name__ == "__main__":
    predictor = RULPredictor()
    path = 'silicon_guard/data/raw/sensor_data.int' # This needs to be valid or use the real path
    # Using a relative path that works if run from root
    actual_path = 'silicon_guard/data/raw/sensor_data.csv'
    if os.path.exists(actual_path):
        predictor.train(actual_path)
    else:
        print("Error: Data not found. Please run the data generator first.")
