import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Load dataset from CSV."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess data: remove unnecessary columns, encode target, standardize features."""
    df = df.drop(columns=["id"], errors="ignore")
    df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])
    scaler = StandardScaler()
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])
    return df