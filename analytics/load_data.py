from pathlib import Path
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def load_daily_sales():
    connection = get_connection()

    try:
        query = """
            SELECT
                full_date,
                year_number,
                month_number,
                month_name,
                quarter_number,
                orders,
                units_sold,
                revenue,
                cost,
                profit,
                profit_margin_percent
            FROM warehouse.vw_sales_daily
            ORDER BY full_date;
        """

        return pd.read_sql(query, connection)

    finally:
        connection.close()


if __name__ == "__main__":
    df = load_daily_sales()

    print("Loaded daily sales data")
    print(f"Rows: {len(df):,}")
    print()
    print(df.head())
    print()
    print(df.info())