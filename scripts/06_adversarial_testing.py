"""
This script tests how the model responds to adversarial or unusual inputs.
Used to assess model robustness under edge cases.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
# Load test features and labels
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Define column names
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "sex"]

# Define adversarial test inputs
adversarial_samples = [
    [0, 250, 80, 20, 0, 18, 0.1, 30, 1],   # Very high glucose
    [0, 0, 0, 0, 0, 0, 0.0, 0, 0],         # All zeros (invalid but test case)
    [2, 120, 70, 25, 600, 45, 0.2, 45, 1], # High insulin + BMI
    [1, 100, 70, 20, 80, 25, 0.5, 90, 0],  # Very high age
    [3, 60, 70, 20, 88, 28, 0.8, 65, 1],   # Low glucose, high age
]

sample_labels = ["High Glucose", "All Zeros", "High BMI/Insulin", "Old Age", "Low Glucose, High Age"]

# Convert to DataFrame
adv_df = pd.DataFrame(adversarial_samples, columns=columns)

# Scale (use only features used in training, drop 'sex')
scaler = StandardScaler()
X_train_nosex = X_train.drop('sex', axis=1)
X_test_nosex = X_test.drop('sex', axis=1)

X_train_scaled = scaler.fit_transform(X_train_nosex)
X_test_scaled = scaler.transform(X_test_nosex)

model.fit(X_train_scaled, y_train)

adv_scaled = scaler.transform(adv_df.drop('sex', axis=1))
y_adv_pred = model.predict(adv_scaled)

# Predict
preds = model.predict(adv_scaled)
probas = model.predict_proba(adv_scaled)[:, 1]

# Display results
print("\n📌 Adversarial Input Results:\n")
for i in range(len(preds)):
    print(f"{sample_labels[i]} ➤ Prediction: {preds[i]} (Confidence: {probas[i]:.4f})")

# Optional: Save to file
results_df = adv_df.copy()
results_df["Prediction"] = preds
results_df["Confidence"] = probas
results_df.to_csv("results/adversarial_test_results.csv", index=False)

print("\n✅ Adversarial testing complete. Results saved to 'results/adversarial_test_results.csv'")
