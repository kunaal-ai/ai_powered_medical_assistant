# AI-Powered Medical Assistant: Diabetes Prediction

## Overview
An end-to-end machine learning system for diabetes risk prediction with a focus on fairness, explainability, and robustness.

## Project Structure
- `data/`: Input and processed data
- `models/`: Trained models and scalers
- `results/`: Evaluation outputs and visualizations
- `scripts/`: Implementation code

## Key Features
- Data preprocessing and feature engineering
- Model training with Logistic Regression and Random Forest
- Bias and fairness testing using AIF360
- Model explainability with SHAP values
- Adversarial testing for robustness
- MLflow for experiment tracking

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run scripts in order:
   - [01_data_preparation.py](cci:7://file:///Users/kt/Documents/github.com/ai_powered_medical_assistant/scripts/01_data_preparation.py:0:0-0:0)
   - [03_model_training.py](cci:7://file:///Users/kt/Documents/github.com/ai_powered_medical_assistant/scripts/03_model_training.py:0:0-0:0)
   - [04_bias_fairness_testing.py](cci:7://file:///Users/kt/Documents/github.com/ai_powered_medical_assistant/scripts/04_bias_fairness_testing.py:0:0-0:0)
   - [05_model_explainability.py](cci:7://file:///Users/kt/Documents/github.com/ai_powered_medical_assistant/scripts/05_model_explainability.py:0:0-0:0)
   - [06_adversarial_testing.py](cci:7://file:///Users/kt/Documents/github.com/ai_powered_medical_assistant/scripts/06_adversarial_testing.py:0:0-0:0)

## Dependencies
- Python 3.8+
- scikit-learn, pandas, numpy
- MLflow, SHAP, AIF360

## Results
- Model performance metrics
- Fairness evaluation reports
- Feature importance visualizations
- Adversarial test results