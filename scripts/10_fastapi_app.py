from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI()

class HealthRequest(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    sex: int
    question: str

# Load model and scaler at startup
model = joblib.load("models/logisticregression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def safe_query(question: str) -> str:
    # Mocked answer for load testing without OpenAI cost
    return "This is a mocked answer for testing."

@app.post("/predict")
def predict_health(data: HealthRequest):
    input_dict = data.dict()
    question = input_dict.pop("question")

    df = pd.DataFrame([input_dict])
    features = list(df.columns)
    scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(scaled, columns=features)

    # Model trained without 'sex', so drop it before predict
    pred = model.predict(df_scaled.drop(columns=['sex']))[0]
    confidence = model.predict_proba(df_scaled.drop(columns=['sex']))[0][1]

    answer = safe_query(question)

    return {
        "prediction": "Diabetic" if pred else "Not Diabetic",
        "confidence": confidence,
        "qa_answer": answer
    }
