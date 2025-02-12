import pandas as pd
import numpy as np
import os

def generate_sample_data(n_samples=100, filename='sample_data.csv'):
    np.random.seed(42)
    
    # Generate realistic health data
    data = {
        'heart_rate': np.clip(np.random.normal(75, 10, n_samples), 50, 120),
        'systolic_bp': np.clip(np.random.normal(120, 15, n_samples), 90, 160),
        'diastolic_bp': np.clip(np.random.normal(80, 10, n_samples), 60, 100),
        'spo2': np.clip(np.random.normal(97, 2, n_samples), 90, 100),
        'healthy': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    
    # Save with specific format
    df.to_csv(filepath, index=False)
    print(f"Generated sample data at: {filepath}")
    print("Data preview:")
    print(df.head())
    return filepath

if __name__ == "__main__":
    generate_sample_data() 