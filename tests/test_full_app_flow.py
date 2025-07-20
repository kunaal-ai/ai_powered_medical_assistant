import pandas as pd
import numpy as np
import joblib
import pickle
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

def test_full_flow_prediction_and_qa():
    # --- Step 1: Prepare health input ---
    input_data = {
        "Pregnancies": [2],
        "Glucose": [140],
        "BloodPressure": [80],
        "SkinThickness": [25],
        "Insulin": [100],
        "BMI": [32.5],
        "DiabetesPedigreeFunction": [0.6],
        "Age": [45],
        "sex": [1]
    }
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age', 'sex']
    df = pd.DataFrame(input_data, columns=features)

    # --- Step 2: Load model and scaler ---
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load("models/logisticregression_model.pkl")
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    # --- Step 3: Make prediction ---
    prediction = model.predict(df_scaled[features[:-1]])[0]
    confidence = model.predict_proba(df_scaled[features[:-1]])[0][1]

    assert prediction in [0, 1], "Invalid prediction class"
    assert 0.0 <= confidence <= 1.0, "Confidence out of range"

    # --- Step 4: RAG query ---
    with open("models/document_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embedding)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    result = qa_chain.invoke("What are symptoms of diabetes?")
    assert isinstance(result['result'], str) and len(result['result']) > 10, "RAG returned empty or invalid response"
