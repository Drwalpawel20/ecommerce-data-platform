from pathlib import Path
import os

import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv(PROJECT_ROOT / ".env")


TABLES = {
    "categories.csv": "staging.categories",
    "suppliers.csv": "staging.suppliers",
    "employees.csv": "staging.employees",
    "customers.csv": "staging.customers",
    "products.csv": "staging.products",
    "orders.csv": "staging.orders",
    "order_details.csv": "staging.order_details",
}


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def get_table_columns(cursor, schema, table):
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position;
        """,
        (schema, table),
    )

    return [row[0] for row in cursor.fetchall()]


def truncate_staging(cursor):
    cursor.execute(
        """
        TRUNCATE TABLE
            staging.order_details,
            staging.orders,
            staging.products,
            staging.customers,
            staging.employees,
            staging.suppliers,
            staging.categories
        RESTART IDENTITY CASCADE;
        """
    )


def load_table(cursor, filename, table_name):
    schema, table = table_name.split(".")
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    columns = get_table_columns(cursor, schema, table)

    column_list = ", ".join(
        f'"{column}"'
        for column in columns
    )

    copy_sql = f"""
        COPY {schema}.{table} ({column_list})
        FROM STDIN
        WITH (
            FORMAT CSV,
            HEADER TRUE,
            NULL ''
        )
    """

    print(f"Loading {filename}...", flush=True)

    with open(file_path, "r", encoding="utf-8", newline="") as file:
        cursor.copy_expert(copy_sql, file)

    cursor.execute(
        f"SELECT COUNT(*) FROM {schema}.{table}"
    )

    row_count = cursor.fetchone()[0]

    print(
        f"Loaded {filename}: {row_count:,} rows",
        flush=True
    )


def main():
    print("Starting staging ETL...")
    print(f"Data directory: {DATA_DIR}")

    connection = get_connection()

    try:
        cursor = connection.cursor()

        truncate_staging(cursor)

        for filename, table_name in TABLES.items():
            load_table(
                cursor,
                filename,
                table_name,
            )

        connection.commit()

        print()
        print("Staging ETL completed successfully.")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()