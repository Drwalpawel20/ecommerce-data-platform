from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analytics" / "ml" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PROJECT_ROOT / "analytics"))

from load_data import get_connection


def load_customer_data():
    connection = get_connection()

    try:
        query = """
            SELECT
                customer_id,
                gender,
                country,
                city,
                orders,
                units_bought,
                revenue,
                cost,
                profit,
                average_order_value,
                first_order_date,
                last_order_date
            FROM warehouse.vw_customer_performance
            ORDER BY customer_id;
        """

        return pd.read_sql(
            query,
            connection
        )

    finally:
        connection.close()


def prepare_data(df):
    df = df.copy()

    date_columns = [
        "first_order_date",
        "last_order_date"
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column]
        )

    numeric_columns = [
        "orders",
        "units_bought",
        "revenue",
        "cost",
        "profit",
        "average_order_value"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "customer_id",
            "orders",
            "revenue",
            "last_order_date"
        ]
    )

    return df


def calculate_rfm(df):
    df = df.copy()

    analysis_date = (
        df["last_order_date"].max()
        + pd.Timedelta(days=1)
    )

    rfm = df[
        [
            "customer_id",
            "orders",
            "revenue",
            "profit",
            "units_bought",
            "average_order_value",
            "first_order_date",
            "last_order_date"
        ]
    ].copy()

    rfm["recency"] = (
        analysis_date -
        rfm["last_order_date"]
    ).dt.days

    rfm["frequency"] = rfm["orders"]

    rfm["monetary"] = rfm["revenue"]

    rfm["profit_margin"] = np.where(
        rfm["monetary"] != 0,
        rfm["profit"] /
        rfm["monetary"] *
        100,
        0
    )

    rfm = rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "profit",
            "units_bought",
            "average_order_value",
            "profit_margin",
            "first_order_date",
            "last_order_date"
        ]
    ]

    return rfm


def print_rfm_statistics(rfm):
    print("\n=== RFM DATASET ===")

    print(
        f"Customers: {len(rfm):,}"
    )

    print(
        f"Features: {len(rfm.columns)}"
    )

    print("\n=== DESCRIPTIVE STATISTICS ===")

    print(
        rfm[
            [
                "recency",
                "frequency",
                "monetary",
                "profit",
                "units_bought",
                "average_order_value",
                "profit_margin"
            ]
        ]
        .describe()
        .round(2)
        .to_string()
    )


def plot_distributions(rfm):
    columns = [
        "recency",
        "frequency",
        "monetary",
        "profit",
        "units_bought",
        "average_order_value",
        "profit_margin"
    ]

    for column in columns:

        plt.figure(figsize=(10, 6))

        plt.hist(
            rfm[column].dropna(),
            bins=50
        )

        plt.title(
            f"{column.replace('_', ' ').title()} Distribution"
        )

        plt.xlabel(
            column.replace("_", " ").title()
        )

        plt.ylabel("Customers")

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR /
            f"rfm_distribution_{column}.png"
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


def apply_log_transformation(rfm):
    transformed = rfm.copy()

    log_columns = [
        "recency",
        "frequency",
        "monetary",
        "profit",
        "units_bought",
        "average_order_value"
    ]

    for column in log_columns:

        values = transformed[column]

        min_value = values.min()

        shift = (
            abs(min_value) + 1
            if min_value <= 0
            else 0
        )

        transformed[
            f"log_{column}"
        ] = np.log1p(
            values + shift
        )

    return transformed


def plot_transformed_distributions(
    transformed
):
    columns = [
        "log_recency",
        "log_frequency",
        "log_monetary",
        "log_profit",
        "log_units_bought",
        "log_average_order_value"
    ]

    for column in columns:

        plt.figure(figsize=(10, 6))

        plt.hist(
            transformed[column].dropna(),
            bins=50
        )

        plt.title(
            f"{column.replace('_', ' ').title()} Distribution"
        )

        plt.xlabel(
            column.replace("_", " ").title()
        )

        plt.ylabel("Customers")

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR /
            f"rfm_transformed_{column}.png"
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


def prepare_ml_features(transformed):
    feature_columns = [
        "log_recency",
        "log_frequency",
        "log_monetary",
        "log_profit",
        "log_units_bought",
        "log_average_order_value",
        "profit_margin"
    ]

    features = (
        transformed[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(
        features
    )

    scaled = pd.DataFrame(
        scaled_values,
        index=features.index,
        columns=[
            f"scaled_{column}"
            for column in feature_columns
        ]
    )

    # NAPRAWIONO: Pobieramy customer_id z obiektu transformed przekazanego jako parametr
    scaled.insert(
        0,
        "customer_id",
        transformed["customer_id"].values
    )

    return features, scaled


def print_feature_correlations(
    features
):
    print(
        "\n=== FEATURE CORRELATION ==="
    )

    correlation = (
        features.corr()
        .round(3)
    )

    print(
        correlation.to_string()
    )


def save_outputs(
    rfm,
    transformed,
    scaled
):
    rfm_output = (
        OUTPUT_DIR /
        "customer_rfm.csv"
    )

    transformed_output = (
        OUTPUT_DIR /
        "customer_rfm_transformed.csv"
    )

    scaled_output = (
        OUTPUT_DIR /
        "customer_rfm_scaled.csv"
    )

    rfm.to_csv(
        rfm_output,
        index=False
    )

    transformed.to_csv(
        transformed_output,
        index=False
    )

    scaled.to_csv(
        scaled_output,
        index=False
    )

    print(
        f"\nSaved: {rfm_output}"
    )

    print(
        f"Saved: {transformed_output}"
    )

    print(
        f"Saved: {scaled_output}"
    )


def main():
    print(
        "Loading customer data..."
    )

    df = load_customer_data()

    print(
        f"Loaded {len(df):,} customers."
    )

    df = prepare_data(df)

    rfm = calculate_rfm(df)

    print_rfm_statistics(
        rfm
    )

    plot_distributions(
        rfm
    )

    transformed = apply_log_transformation(
        rfm
    )

    print(
        "\n=== LOG TRANSFORMATION ==="
    )

    print(
        "Log transformation completed."
    )

    plot_transformed_distributions(
        transformed
    )

    # NAPRAWIONO: Dodano kompletne dokończenie logiki biznesowej w funkcji main()
    features, scaled = prepare_ml_features(
        transformed
    )

    print_feature_correlations(
        features
    )

    save_outputs(
        rfm,
        transformed,
        scaled
    )


if __name__ == "__main__":
    main()
