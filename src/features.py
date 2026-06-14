import pandas as pd

def engineer_features(df):
    """
    Turns raw sensor readings into useful features like rolling averages and peaks.
    This helps the model see patterns over time instead of just single points.
    """
    # Make sure our data is in chronological order
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Create simple moving windows (6h and 24h) to capture trends
    for window in [6, 24]:
        df[f'temp_rolling_mean_{window}h'] = df['temperature_C'].rolling(window=window).mean()
        df[f'temp_rolling_std_{window}h'] = df['temperature_C'].rolling(window=window).std()
        df[f'temp_rolling_max_{window}h'] = df['temperature_C'].rolling(window=window).max()

    # Clean up the rows where we don't have enough history yet (the NaN rows)
    df = df.dropna().reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Quick test to ensure feature engineering works as expected
    import os
    raw_path = 'silicon_guard/data/raw/sensor_data.csv'
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, parse_dates=['timestamp'])
        processed_df = engineer_features(df)
        print(f"Done! Processed {len(processed_df)} samples.")
        print("New columns created:", [col for col in processed_df.columns if 'temp_rolling' in col])
    else:
        print("Error: Run the data generator first!")
