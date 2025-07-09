import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def normalize_continuous_features(df, columns):
    """
    Apply Min-Max normalization to specified continuous feature columns.
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def encode_ratio_feature(df, column):
    """
    Convert string ratio representations (e.g., '75:25') to float (e.g., 0.75).
    """
    def convert_ratio(ratio_str):
        try:
            parts = ratio_str.split(":")
            return float(parts[0]) / (float(parts[0]) + float(parts[1]))
        except:
            return np.nan

    df[column] = df[column].astype(str).apply(convert_ratio)
    return df

def label_encode_shape(df, column):
    """
    Encode shape types into integer labels.
    """
    shape_mapping = {
        "circular film": 1,
        "rectangular film": 2,
        "cube": 3
    }
    df[column] = df[column].map(shape_mapping).fillna(0).astype(int)
    return df

def clean_numeric_data(df):
    """
    Convert all values to float, replace non-numeric and missing values with 0.0.
    """
    return df.apply(lambda col: pd.to_numeric(col, errors='coerce').fillna(0.0))

def detect_outliers(model, X_scaled, y, data, threshold_factor=2):
    """
    Detect anomalies based on prediction errors exceeding a set threshold.
    """
    y_pred = model.predict(X_scaled)
    errors = np.abs(y - y_pred)
    threshold = np.mean(errors) + threshold_factor * np.std(errors)
    anomaly_indices = np.where(errors > threshold)[0]
    anomalies = data.iloc[anomaly_indices]
    return anomalies, y_pred
