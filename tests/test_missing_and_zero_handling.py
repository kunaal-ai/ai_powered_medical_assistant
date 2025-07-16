import pandas as pd
import numpy as np

def test_missing_and_zero_value_handling():
    # Simulate raw input with zeros
    sample = {
        "Pregnancies": [0],
        "Glucose": [0],
        "BloodPressure": [0],
        "SkinThickness": [0],
        "Insulin": [0],
        "BMI": [0],
        "DiabetesPedigreeFunction": [0.5],
        "Age": [33],
        "sex": [1]
    }
    df = pd.DataFrame(sample)

    # Load training data to compute read medians
    reference = pd.read_csv("data/X_train.csv")
    medians = reference.median()

    # Replace zeros as in preprocessing
    for col in ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(medians[col])

    # ✅ Test to ensure all missing values are filled
    assert df.isnull().sum().sum() == 0, f"There are still missing values:\n{df.isnull().sum()}"
    print("✅ Zero-value handling test passed.")
