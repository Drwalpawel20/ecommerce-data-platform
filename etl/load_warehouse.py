from pathlib import Path
import os

import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


BATCH_SIZE = 25000


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def load_dimensions(cursor):
    print("Loading dim_category...")

    cursor.execute(
        """
        INSERT INTO warehouse.dim_category (
            category_id,
            category_name
        )
        SELECT
            category_id,
            category_name
        FROM staging.categories
        ON CONFLICT (category_id)
        DO UPDATE SET
            category_name = EXCLUDED.category_name;
        """
    )

    print(f"dim_category: {cursor.rowcount:,} rows")

    print("Loading dim_supplier...")

    cursor.execute(
        """
        INSERT INTO warehouse.dim_supplier (
            supplier_id,
            supplier_name,
            country,
            city
        )
        SELECT
            supplier_id,
            supplier_name,
            country,
            city
        FROM staging.suppliers
        ON CONFLICT (supplier_id)
        DO UPDATE SET
            supplier_name = EXCLUDED.supplier_name,
            country = EXCLUDED.country,
            city = EXCLUDED.city;
        """
    )

    print(f"dim_supplier: {cursor.rowcount:,} rows")

    print("Loading dim_employee...")

    cursor.execute(
        """
        INSERT INTO warehouse.dim_employee (
            employee_id,
            first_name,
            last_name,
            department,
            hire_date
        )
        SELECT
            employee_id,
            first_name,
            last_name,
            department,
            hire_date
        FROM staging.employees
        ON CONFLICT (employee_id)
        DO UPDATE SET
            first_name = EXCLUDED.first_name,
            last_name = EXCLUDED.last_name,
            department = EXCLUDED.department,
            hire_date = EXCLUDED.hire_date;
        """
    )

    print(f"dim_employee: {cursor.rowcount:,} rows")

    print("Loading dim_customer...")

    cursor.execute(
        """
        INSERT INTO warehouse.dim_customer (
            customer_id,
            first_name,
            last_name,
            email,
            gender,
            date_of_birth,
            country,
            city,
            registration_date
        )
        SELECT
            customer_id,
            first_name,
            last_name,
            email,
            gender,
            date_of_birth,
            country,
            city,
            registration_date
        FROM staging.customers
        ON CONFLICT (customer_id)
        DO UPDATE SET
            first_name = EXCLUDED.first_name,
            last_name = EXCLUDED.last_name,
            email = EXCLUDED.email,
            gender = EXCLUDED.gender,
            date_of_birth = EXCLUDED.date_of_birth,
            country = EXCLUDED.country,
            city = EXCLUDED.city,
            registration_date = EXCLUDED.registration_date;
        """
    )

    print(f"dim_customer: {cursor.rowcount:,} rows")

    print("Loading dim_product...")

    cursor.execute(
        """
        INSERT INTO warehouse.dim_product (
            product_id,
            product_name,
            category_key,
            supplier_key,
            cost,
            price
        )
        SELECT
            p.product_id,
            p.product_name,
            c.category_key,
            s.supplier_key,
            p.cost,
            p.price
        FROM staging.products p
        JOIN warehouse.dim_category c
            ON c.category_id = p.category_id
        JOIN warehouse.dim_supplier s
            ON s.supplier_id = p.supplier_id
        ON CONFLICT (product_id)
        DO UPDATE SET
            product_name = EXCLUDED.product_name,
            category_key = EXCLUDED.category_key,
            supplier_key = EXCLUDED.supplier_key,
            cost = EXCLUDED.cost,
            price = EXCLUDED.price;
        """
    )

    print(f"dim_product: {cursor.rowcount:,} rows")


def load_dates(cursor):
    print("Loading dim_date...")

    cursor.execute(
        """
        INSERT INTO warehouse.dim_date (
            date_key,
            full_date,
            day_number,
            month_number,
            month_name,
            quarter_number,
            year_number,
            day_of_week_number,
            day_of_week_name
        )
        SELECT DISTINCT
            TO_CHAR(order_date::date, 'YYYYMMDD')::integer,
            order_date::date,
            EXTRACT(DAY FROM order_date)::integer,
            EXTRACT(MONTH FROM order_date)::integer,
            TRIM(TO_CHAR(order_date, 'Month')),
            EXTRACT(QUARTER FROM order_date)::integer,
            EXTRACT(YEAR FROM order_date)::integer,
            EXTRACT(ISODOW FROM order_date)::integer,
            TRIM(TO_CHAR(order_date, 'Day'))
        FROM staging.orders
        ON CONFLICT (date_key)
        DO NOTHING;
        """
    )

    print(f"dim_date: {cursor.rowcount:,} rows")


def prepare_fact_sales(cursor):
    print("Preparing fact_sales...")

    cursor.execute(
        """
        TRUNCATE TABLE warehouse.fact_sales
        RESTART IDENTITY;
        """
    )


def load_fact_sales(cursor):
    print("Loading fact_sales...")

    cursor.execute(
        """
        SELECT MIN(order_id), MAX(order_id)
        FROM staging.orders;
        """
    )

    min_order_id, max_order_id = cursor.fetchone()

    if min_order_id is None:
        print("fact_sales: no orders found")
        return

    current_id = min_order_id
    total_rows = 0

    while current_id <= max_order_id:
        end_id = current_id + BATCH_SIZE - 1

        cursor.execute(
            """
            INSERT INTO warehouse.fact_sales (
                order_id,
                order_detail_id,
                date_key,
                customer_key,
                product_key,
                employee_key,
                quantity,
                unit_price,
                discount,
                revenue,
                cost,
                profit,
                order_status
            )
            SELECT
                o.order_id,
                od.order_detail_id,
                d.date_key,
                c.customer_key,
                p.product_key,
                e.employee_key,
                od.quantity,
                od.unit_price,
                od.discount,
                od.revenue,
                od.quantity * p.cost,
                od.revenue - od.quantity * p.cost,
                o.status
            FROM staging.order_details od
            INNER JOIN staging.orders o
                ON o.order_id = od.order_id
            INNER JOIN warehouse.dim_date d
                ON d.full_date = o.order_date::date
            INNER JOIN warehouse.dim_customer c
                ON c.customer_id = o.customer_id
            INNER JOIN warehouse.dim_product p
                ON p.product_id = od.product_id
            INNER JOIN warehouse.dim_employee e
                ON e.employee_id = o.employee_id
            WHERE o.order_id BETWEEN %s AND %s;
            """,
            (current_id, end_id)
        )

        rows = cursor.rowcount
        total_rows += rows

        print(
            f"Orders {current_id:,}-{end_id:,}: "
            f"{rows:,} rows | Total: {total_rows:,}",
            flush=True
        )

        current_id = end_id + 1

    print(f"fact_sales: {total_rows:,} rows")


def analyze_tables(cursor):
    print("Updating table statistics...")

    cursor.execute(
        """
        ANALYZE warehouse.dim_category;
        ANALYZE warehouse.dim_supplier;
        ANALYZE warehouse.dim_employee;
        ANALYZE warehouse.dim_customer;
        ANALYZE warehouse.dim_product;
        ANALYZE warehouse.dim_date;
        ANALYZE warehouse.fact_sales;
        """
    )


def main():
    print("Starting warehouse ETL...")

    connection = get_connection()

    try:
        cursor = connection.cursor()

        load_dimensions(cursor)
        load_dates(cursor)

        prepare_fact_sales(cursor)

        connection.commit()

        load_fact_sales(cursor)

        connection.commit()

        analyze_tables(cursor)

        connection.commit()

        print()
        print("Warehouse ETL completed successfully.")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()