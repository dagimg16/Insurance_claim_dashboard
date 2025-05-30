import joblib
import numpy as np

def load_model(path="model/fraud_model_XGB.pkl"):
    return joblib.load(path)

def predict_fraud(model, df_row, threshold=0.3):
    proba = model.predict_proba(df_row)[:, 1]
    return (proba >= threshold).astype(int), proba