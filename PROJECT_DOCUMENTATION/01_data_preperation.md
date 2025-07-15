# Data Preparation Documentation

This document outlines the data preparation steps for the diabetes prediction model.

## Data Loading
- Loads the Pima Indians Diabetes dataset from `data/pima-indians-diabetes.data.csv`
- Column names are explicitly defined for clarity:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome

## Data Preprocessing

### 1. Adding Synthetic 'sex' Column
- A synthetic binary 'sex' column is added (0 = female, 1 = male) for bias and fairness testing
- Generated using a fixed random seed (42) for reproducibility

### 2. Handling Missing Values
- Checks for missing values in all columns
- Reports columns with missing values if any exist

### 3. Handling Zero Values
- Identifies and reports zero values in key medical measurements:
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI

### 4. Data Imputation
- Replaces zero values with NaN in the following columns:
  - Glucose
  - BloodPressure
  - BMI
  - Insulin
- Fills NaN values with the median of each respective column

## Data Splitting
- Splits the data into features (X) and target (y)
- Target variable: 'Outcome' column
- Performs an 80-20 train-test split
- Uses stratification to maintain class distribution
- Random state fixed at 42 for reproducibility

## Feature Scaling
- Applies StandardScaler to normalize features
- Fits the scaler only on training data
- Transforms both training and test sets using the same scaler

## Output Files
Saves the following files for model training:
- `data/X_train.csv`: Scaled training features
- `data/X_test.csv`: Scaled test features
- `data/y_train.csv`: Training target
- `data/y_test.csv`: Test target
- `models/scaler.pkl`: Fitted StandardScaler object

## Usage
Run the script to preprocess the data:
```bash
python scripts/01_data_preparation.py