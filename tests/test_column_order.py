# tests/test_column_order.py

import pandas as pd

def test_column_order_consistency():
    # Load original training data (after scaling)
    X_train = pd.read_csv("data/X_train.csv")

    # Get expected columns from training data
    expected_columns = list(X_train.columns)

    # Simulate a user input or a new sample with same columns
    sample = {
        "Pregnancies": [2],
        "Glucose": [120],
        "BloodPressure": [70],
        "SkinThickness": [20],
        "Insulin": [80],
        "BMI": [28.5],
        "DiabetesPedigreeFunction": [0.5],
        "Age": [33],
        "sex": [1]
    }
    df = pd.DataFrame(sample)

    # ✅ Check same columns and order
    assert list(df.columns) == expected_columns, (
        f"Column mismatch!\nExpected: {expected_columns}\nGot: {list(df.columns)}"
    )

    print("✅ test_column_order_consistency passed.")
