from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


SEED = 42

N_CUSTOMERS = 50_000
N_PRODUCTS = 10_000
N_CATEGORIES = 50
N_SUPPLIERS = 500
N_EMPLOYEES = 100
N_ORDERS = 500_000

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

fake = Faker()
Faker.seed(SEED)
np.random.seed(SEED)


def generate_categories():
    categories = [
        "Electronics",
        "Computers",
        "Smartphones",
        "Home Appliances",
        "Furniture",
        "Clothing",
        "Shoes",
        "Sports",
        "Beauty",
        "Books",
        "Toys",
        "Garden",
        "Automotive",
        "Office",
        "Jewelry",
        "Pet Supplies",
        "Health",
        "Gaming",
        "Music",
        "Travel",
    ]

    categories = categories[:N_CATEGORIES]

    while len(categories) < N_CATEGORIES:
        categories.append(f"Category {len(categories) + 1}")

    return pd.DataFrame(
        {
            "category_id": range(1, N_CATEGORIES + 1),
            "category_name": categories,
        }
    )


def generate_suppliers():
    rows = []

    for supplier_id in range(1, N_SUPPLIERS + 1):
        rows.append(
            {
                "supplier_id": supplier_id,
                "supplier_name": fake.company(),
                "country": fake.country(),
                "city": fake.city(),
            }
        )

    return pd.DataFrame(rows)


def generate_employees():
    rows = []

    for employee_id in range(1, N_EMPLOYEES + 1):
        rows.append(
            {
                "employee_id": employee_id,
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "department": np.random.choice(
                    [
                        "Sales",
                        "Customer Service",
                        "Management",
                        "Warehouse",
                    ]
                ),
                "hire_date": fake.date_between(
                    start_date="-10y",
                    end_date="-30d",
                ),
            }
        )

    return pd.DataFrame(rows)


def generate_customers():
    rows = []

    for customer_id in range(1, N_CUSTOMERS + 1):
        registration_date = fake.date_between(
            start_date="-5y",
            end_date="-30d",
        )

        rows.append(
            {
                "customer_id": customer_id,
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "email": fake.unique.email(),
                "gender": np.random.choice(
                    ["Male", "Female"],
                    p=[0.49, 0.51],
                ),
                "date_of_birth": fake.date_of_birth(
                    minimum_age=18,
                    maximum_age=75,
                ),
                "country": fake.country(),
                "city": fake.city(),
                "registration_date": registration_date,
            }
        )

    return pd.DataFrame(rows)


def generate_products(categories):
    rows = []

    for product_id in range(1, N_PRODUCTS + 1):
        category_id = np.random.randint(1, N_CATEGORIES + 1)
        supplier_id = np.random.randint(1, N_SUPPLIERS + 1)

        cost = round(
            np.random.lognormal(
                mean=3.5,
                sigma=1.0,
            ),
            2,
        )

        price = round(
            cost * np.random.uniform(1.2, 2.5),
            2,
        )

        category_name = categories.loc[
            categories["category_id"] == category_id,
            "category_name",
        ].iloc[0]

        rows.append(
            {
                "product_id": product_id,
                "product_name": f"{category_name} Product {product_id}",
                "category_id": category_id,
                "supplier_id": supplier_id,
                "cost": cost,
                "price": price,
            }
        )

    return pd.DataFrame(rows)


def generate_orders():
    start_date = datetime.now() - timedelta(days=5 * 365)
    end_date = datetime.now()

    order_dates = pd.to_datetime(
        np.random.randint(
            start_date.timestamp(),
            end_date.timestamp(),
            N_ORDERS,
        ),
        unit="s",
    )

    customer_ids = np.random.randint(
        1,
        N_CUSTOMERS + 1,
        N_ORDERS,
    )

    employee_ids = np.random.randint(
        1,
        N_EMPLOYEES + 1,
        N_ORDERS,
    )

    statuses = np.random.choice(
        [
            "Completed",
            "Completed",
            "Completed",
            "Cancelled",
            "Returned",
        ],
        size=N_ORDERS,
    )

    return pd.DataFrame(
        {
            "order_id": np.arange(1, N_ORDERS + 1),
            "customer_id": customer_ids,
            "employee_id": employee_ids,
            "order_date": order_dates,
            "status": statuses,
        }
    )


def generate_order_details(orders, products):
    rng = np.random.default_rng(SEED)

    items_per_order = rng.choice(
        [1, 2, 3, 4, 5],
        size=len(orders),
        p=[0.30, 0.30, 0.20, 0.12, 0.08],
    )

    total_items = items_per_order.sum()

    order_ids = np.repeat(
        orders["order_id"].values,
        items_per_order,
    )

    product_ids = rng.integers(
        1,
        N_PRODUCTS + 1,
        size=total_items,
    )

    quantities = rng.choice(
        [1, 2, 3, 4, 5],
        size=total_items,
        p=[0.55, 0.25, 0.12, 0.06, 0.02],
    )

    discounts = rng.choice(
        [0.00, 0.05, 0.10, 0.15, 0.20],
        size=total_items,
        p=[0.55, 0.20, 0.15, 0.07, 0.03],
    )

    prices = products.set_index("product_id")["price"]

    unit_prices = prices.loc[product_ids].values

    rows = pd.DataFrame(
        {
            "order_detail_id": np.arange(
                1,
                total_items + 1,
            ),
            "order_id": order_ids,
            "product_id": product_ids,
            "quantity": quantities,
            "unit_price": unit_prices,
            "discount": discounts,
        }
    )

    rows["revenue"] = (
        rows["quantity"]
        * rows["unit_price"]
        * (1 - rows["discount"])
    ).round(2)

    return rows


def save_dataframe(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved {filename}: {len(df):,} rows")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating categories...")
    categories = generate_categories()

    print("Generating suppliers...")
    suppliers = generate_suppliers()

    print("Generating employees...")
    employees = generate_employees()

    print("Generating customers...")
    customers = generate_customers()

    print("Generating products...")
    products = generate_products(categories)

    print("Generating orders...")
    orders = generate_orders()

    print("Generating order details...")
    order_details = generate_order_details(
        orders,
        products,
    )

    save_dataframe(
        categories,
        "categories.csv",
    )

    save_dataframe(
        suppliers,
        "suppliers.csv",
    )

    save_dataframe(
        employees,
        "employees.csv",
    )

    save_dataframe(
        customers,
        "customers.csv",
    )

    save_dataframe(
        products,
        "products.csv",
    )

    save_dataframe(
        orders,
        "orders.csv",
    )

    save_dataframe(
        order_details,
        "order_details.csv",
    )

    print()
    print("Data generation completed.")


if __name__ == "__main__":
    main()