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

column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv("data/pima-indians-diabetes.data.csv", names=column_names)

# Add synthetic 'sex' column (0 = female, 1 = male)
# This is missing col so adding synthetically for biased and faireness testing
np.random.seed(42)
df['sex'] = np.random.choice([0, 1], size=len(df))

print('\n' +"=" * 10 + ' Raw Sample ' +'=' * 10)
print(df.head())

# ------ Missing values ----------
print('\n' +"=" * 10 + ' Missing Values ' +'=' * 10)
missing_values = df.isnull().sum()
if missing_values.any():
    print("\nColumns with missing values:")
    print(missing_values[missing_values > 0])
else:
    print("\nNo missing values found in any column.")

# ------ Zero values ----------
print('\n' + "=" * 10 + ' Zero Values ' +'=' * 10)
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    zero_values = (df[col] ==0).sum()
    print(f"{col} : {zero_values}")
    
# ------ Replace invalid zeros with NaN, then assign median  --------
for col in ["Glucose", "BloodPressure", "BMI", "Insulin","SkinThickness"]:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# ------ Split features and target ----------
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

# ------ Train-test split ---------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------- Normalize features ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✔️ Normalization completed")

# ------- Save datasets and scaler for later use ---------
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("data/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)
print("✔️ Datasets saved")


import joblib
joblib.dump(scaler, "models/scaler.pkl")


