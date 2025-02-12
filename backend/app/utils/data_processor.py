import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def process_data(self, df):
        """Process the input dataframe"""
        try:
            # Expected columns
            required_columns = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'healthy']
            
            # Verify columns exist
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Required: {required_columns}")
            
            # Extract features and target
            X = df[['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2']]
            y = df['healthy']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return (
                X_train_scaled, 
                X_test_scaled,
                y_train.values,
                y_test.values
            )
            
        except Exception as e:
            raise ValueError(f"Data processing error: {str(e)}") 