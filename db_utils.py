import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import joblib
import os

print("Current working directory:", os.getcwd())

load_dotenv()  # load variables from .env
EXPECTED_COLUMNS = joblib.load("model/expected_model_columns.pkl") 

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
database = os.getenv("POSTGRES_DB")

def get_claim_by_id(claim_id):
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_str)

    query = text("SELECT * FROM claims WHERE claim_id = :claim_id")

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"claim_id": claim_id})

    return df

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

claim = get_claim_by_id('XAJI0Y6DP')

x_encod = preprocess_input(claim)

print(x_encod.T)
print(x_encod.shape)