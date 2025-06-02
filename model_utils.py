import joblib
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import shap

def load_model(path="model/fraud_model_XGB.pkl"):
    return joblib.load(path)

def predict_fraud(model, df_row, threshold=0.3):
    proba = model.predict_proba(df_row)[:, 1]
    return (proba >= threshold).astype(int), proba

EXPECTED_COLUMNS = joblib.load("model/expected_model_columns.pkl") 

def load_shap_explainer(model):
    explainer = shap.TreeExplainer(model)
    return explainer

def preprocess_input(df_row):
    df = df_row.copy()

    # Convert date column to datetime
    df['incident_date'] = pd.to_datetime(df['incident_date'])

    # Feature engineering: extract month and day of week
    df['incident_month'] = df['incident_date'].dt.month
    df['incident_dayofweek'] = df['incident_date'].dt.dayofweek

    # Drop columns you excluded during training
    df = df.drop(columns=['fraud_flag', 'claim_id', 'severity_score', 'claim_amount', 'fact_of_loss', 'policy_start', 'incident_date']) 

    # One-hot encoding for categorical features
    categorical_cols = ['state', 'vehicle_type', 'incident_type', 'reported_by']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0  
    df = df[EXPECTED_COLUMNS]

    return df

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)