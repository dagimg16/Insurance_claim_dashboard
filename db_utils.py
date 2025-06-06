import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()  # load variables from .env

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

def update_liability_in_db(claim_id, insured_liability, claimant_liability):
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_str)

    with engine.connect() as conn:
        update_query = text("""
            UPDATE claims
            SET insured_liability = :insured_liability,
                claimant_liability = :claimant_liability
            WHERE claim_id = :claim_id
        """)
        conn.execute(update_query, {
            'insured_liability': insured_liability,
            'claimant_liability': claimant_liability,
            'claim_id': claim_id
        })
        conn.commit()

def get_matching_claim_id(query, params):
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_str)


    with engine.connect() as conn:
        selected_claim_id = pd.read_sql(query, conn, params=params)

    return selected_claim_id       