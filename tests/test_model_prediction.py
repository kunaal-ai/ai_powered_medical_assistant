import joblib
import numpy as np
import pandas as pd
import pytest

# Load trained model and scaler
model = joblib.load("models/logisticregression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define expected features (same as training)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age', 'sex']

def get_valid_input():
    return pd.DataFrame([{
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 80,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 30.5,
        "DiabetesPedigreeFunction": 0.7,
        "Age": 45,
        "sex": 1
    }])

def test_model_prediction_shape():
    df = get_valid_input()
    scaled = scaler.transform(df)
    preds = model.predict(scaled[:, :-1])  # drop 'sex' column
    assert preds.shape == (1,), "Prediction shape is incorrect"

def test_model_prediction_value_range():
    df = get_valid_input()
    scaled = scaler.transform(df)
    proba = model.predict_proba(scaled[:, :-1])[0][1]
    assert 0.0 <= proba <= 1.0, f"Prediction probability out of range: {proba}"

def test_model_handles_zero_edge_case():
    df = get_valid_input()
    df["Glucose"] = 0  # set to edge-case
    df["Glucose"] = df["Glucose"].replace(0, np.nan).fillna(df["Glucose"].median())
    scaled = scaler.transform(df)
    _ = model.predict(scaled[:, :-1])  # should not throw
