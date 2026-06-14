import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sensor_data(n_days=120, freq='h'):
    """
    Creates synthetic sensor data for a semiconductor tool.
    Simulates a gradual temperature increase leading to failure.
    """
    start_date = datetime(2024, 1, 1)
    timestamps = pd.date_range(start=start_date, periods=n_days * 24, freq=freq)
    n_samples = len(timestamps)

    # Base temperature: stable around 150C
    base_temp = 150 + np.random.normal(0, 1, n_samples)

    # Simulate degradation: gradual upward trend starting halfway through
    degradation_start_idx = int(n_samples * 0.5)
    trend = np.zeros(n_samples)
    trend[degradation_start_idx:] = (np.arange(n_samples - degradation_start_idx) * 0.02)

    # Combine and add real-world sensor noise
    temp_series = base_temp + trend + np.random.normal(0, 1, n_samples)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature_C': temp_series
    })

    # Failure happens when temperature exceeds 175C
    failure_indices = np.where(df['temperature_C'] > 175)[0]

    if len(failure_indices) > 0:
        first_failure_idx = failure_indices[0]
        rul_values = first_failure_idx - np.arange(n_samples)
        df['RUL_hours'] = np.maximum(rul_values, 0)
    else:
        df['RUL_hours'] = 1000

    return df

if __name__ == "__main__":
    data = generate_sensor_data()
    output_path = 'silicon_guard/data/raw/sensor_data.csv'
    data.to_csv(output_path, index=False)
    print(f"Generated {len(data)} samples. Saved to {output_path}")
