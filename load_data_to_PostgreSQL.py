import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine,  text
import os

load_dotenv()  # load variables from .env

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
database = os.getenv("POSTGRES_DB")

# Create connection string
conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
engine = create_engine(conn_str)

claims_data = pd.read_csv('./data/claims_data.csv')

#Load data to PostgreSQL
claims_data.to_sql("claims", engine, if_exists="replace", index=False)

print("✅ Claims data uploaded successfully to PostgreSQL!")