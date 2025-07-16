# tests/test_scaler_consistency.py

import pandas as pd
import numpy as np
import joblib
import os

def test_scaler_mean_std_consistency():
    # --- Step 1: Load raw dataset used before scaling ---
    column_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv("data/pima-indians-diabetes.data.csv", names=column_names)

    # Add synthetic 'sex' column (as done in 01_data_preperation.py)
    np.random.seed(42)
    df["sex"] = np.random.choice([0, 1], size=len(df))

    # Replace invalid 0s with median (same logic as main script)
    for col in ["Glucose", "BloodPressure", "BMI", "Insulin"]:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].median(), inplace=True)

    # --- Step 2: Drop target column and get features ---
    X = df.drop(["Outcome"], axis=1)

    # --- Step 3: Load fitted scaler ---
    scaler_path = "models/scaler.pkl"
    assert os.path.exists(scaler_path), f"Scaler not found at {scaler_path}"
    scaler = joblib.load(scaler_path)

    # --- Step 4: Transform using the scaler ---
    X_scaled = scaler.transform(X)

    # --- Step 5: Assertions on scaling ---
    mean = X_scaled.mean()
    std = X_scaled.std()

    assert np.isclose(mean, 0, atol=0.17), f"Mean not ~0, got {mean}"
    assert np.isclose(std, 1, atol=0.17), f"Std not ~1, got {std}"

    print("✅ Scaler consistency test passed.")
