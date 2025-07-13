'''
This script is used to preprocess structured data into a format that can be used for training a machine learning model.

It takes a CSV file as input and converts it into a pandas DataFrame.

The script then performs data cleaning and preprocessing steps, including:

    Missing value detection and handling
    Zero value detection and handling
    Feature scaling
    Train-test split
    Data saving for later use
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
summary_report = {}
df = pd.read_csv(url, names=column_names)

print("Raw Data Sample:")
print(df.head())

# ------ Missing values ----------
print("Missing Values:")
missing_values = df.isnull().sum()
if (missing_values > 0).any():
    summary_report['missing_values'] = missing_values
    print("Missing Values")
    print(missing_values)
else:
    print("No missing values")

# ------ Zero values ----------
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    zero_values = (df[col] ==0).sum()
    if zero_values > 0:
        summary_report[f'zero_value_{col.lower()}'] = zero_values
    
# ------ Replace invalid zeros with NaN, then assign median  --------
for col in ["Glucose", "BloodPressure", "BMI", "Insulin"]:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# ------ Split features and target ----------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ------ Train-test split ---------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------- Normalize features ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------- Save datasets and scaler for later use ---------
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("data/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

import joblib
joblib.dump(scaler, "models/scaler.pkl")


print("Summary Report:")
print(summary_report)

