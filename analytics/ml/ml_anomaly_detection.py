from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "analytics" / "ml" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


CONTAMINATION = 0.02
RANDOM_STATE = 42


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def load_sales_data(connection):
    query = """
        SELECT
            full_date,
            orders,
            units_sold,
            revenue,
            cost,
            profit,
            profit_margin_percent
        FROM warehouse.vw_sales_daily
        ORDER BY full_date
    """

    df = pd.read_sql_query(
        query,
        connection
    )

    df["full_date"] = pd.to_datetime(
        df["full_date"]
    )

    numeric_columns = [
        "orders",
        "units_sold",
        "revenue",
        "cost",
        "profit",
        "profit_margin_percent"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def create_features(df):
    data = df.copy()

    data["revenue_change"] = (
        data["revenue"].pct_change()
    )

    data["orders_change"] = (
        data["orders"].pct_change()
    )

    data["units_change"] = (
        data["units_sold"].pct_change()
    )

    data["profit_change"] = (
        data["profit"].pct_change()
    )

    data["revenue_7d_avg"] = (
        data["revenue"]
        .rolling(7)
        .mean()
    )

    data["revenue_7d_std"] = (
        data["revenue"]
        .rolling(7)
        .std()
    )

    data["orders_7d_avg"] = (
        data["orders"]
        .rolling(7)
        .mean()
    )

    data["profit_7d_avg"] = (
        data["profit"]
        .rolling(7)
        .mean()
    )

    data["revenue_vs_7d_avg"] = (
        data["revenue"]
        / data["revenue_7d_avg"]
    )

    data["orders_vs_7d_avg"] = (
        data["orders"]
        / data["orders_7d_avg"]
    )

    data = data.replace(
        [
            float("inf"),
            float("-inf")
        ],
        pd.NA
    )

    data = data.dropna().reset_index(
        drop=True
    )

    return data


def detect_anomalies(data):
    feature_columns = [
        "orders",
        "units_sold",
        "revenue",
        "profit",
        "profit_margin_percent",
        "revenue_change",
        "orders_change",
        "units_change",
        "profit_change",
        "revenue_7d_avg",
        "revenue_7d_std",
        "orders_7d_avg",
        "profit_7d_avg",
        "revenue_vs_7d_avg",
        "orders_vs_7d_avg"
    ]

    x = data[feature_columns].copy()

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x)

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    predictions = model.fit_predict(
        x_scaled
    )

    anomaly_scores = model.decision_function(
        x_scaled
    )

    result = data.copy()

    result["anomaly_prediction"] = predictions

    result["anomaly_score"] = (
        anomaly_scores
    )

    result["is_anomaly"] = (
        result["anomaly_prediction"] == -1
    )

    result["anomaly_type"] = "Normal"

    result.loc[
        result["is_anomaly"]
        & (
            result["revenue"]
            > result["revenue_7d_avg"]
        ),
        "anomaly_type"
    ] = "Unusually High Revenue"

    result.loc[
        result["is_anomaly"]
        & (
            result["revenue"]
            < result["revenue_7d_avg"]
        ),
        "anomaly_type"
    ] = "Unusually Low Revenue"

    return result


def save_results(result):
    path = (
        OUTPUT_DIR /
        "sales_anomalies.csv"
    )

    result.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )


def print_anomalies(result, n=20):
    anomalies = result[
        result["is_anomaly"]
    ].copy()

    anomalies = anomalies.sort_values(
        "anomaly_score"
    )

    print("\n=== DETECTED ANOMALIES ===")

    if anomalies.empty:
        print("No anomalies detected.")
        return

    display = anomalies.head(n)[
        [
            "full_date",
            "orders",
            "units_sold",
            "revenue",
            "profit",
            "profit_margin_percent",
            "anomaly_score",
            "anomaly_type"
        ]
    ].copy()

    display["revenue"] = (
        display["revenue"].round(2)
    )

    display["profit"] = (
        display["profit"].round(2)
    )

    display["profit_margin_percent"] = (
        display["profit_margin_percent"]
        .round(2)
    )

    display["anomaly_score"] = (
        display["anomaly_score"]
        .round(4)
    )

    print(
        display.to_string(
            index=False
        )
    )


def create_anomaly_chart(result):
    plt.figure(figsize=(14, 7))

    normal = result[
        ~result["is_anomaly"]
    ]

    anomalies = result[
        result["is_anomaly"]
    ]

    plt.plot(
        normal["full_date"],
        normal["revenue"],
        label="Revenue"
    )

    plt.scatter(
        anomalies["full_date"],
        anomalies["revenue"],
        s=60,
        label="Anomaly"
    )

    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.title("Sales Revenue Anomaly Detection")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    path = (
        OUTPUT_DIR /
        "sales_anomalies.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved chart: {path}"
    )


def create_anomaly_summary(result):
    total_days = len(result)

    anomaly_days = int(
        result["is_anomaly"].sum()
    )

    anomaly_percentage = (
        anomaly_days / total_days * 100
    )

    summary = pd.DataFrame({
        "metric": [
            "total_days",
            "anomaly_days",
            "anomaly_percentage",
            "contamination"
        ],
        "value": [
            total_days,
            anomaly_days,
            anomaly_percentage,
            CONTAMINATION
        ]
    })

    path = (
        OUTPUT_DIR /
        "anomaly_detection_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    return summary


def main():
    print("=" * 70)
    print("SALES ANOMALY DETECTION")
    print("=" * 70)

    print(
        f"\nContamination: "
        f"{CONTAMINATION}"
    )

    connection = get_connection()

    try:
        df = load_sales_data(
            connection
        )
    finally:
        connection.close()

    print(
        f"Loaded {len(df):,} daily observations."
    )

    data = create_features(
        df
    )

    print(
        f"Observations after feature engineering: "
        f"{len(data):,}"
    )

    result = detect_anomalies(
        data
    )

    anomaly_count = int(
        result["is_anomaly"].sum()
    )

    print(
        f"\nDetected anomalies: "
        f"{anomaly_count:,}"
    )

    print_anomalies(
        result,
        20
    )

    summary = create_anomaly_summary(
        result
    )

    print("\n=== SUMMARY ===")
    print(
        summary.to_string(
            index=False
        )
    )

    save_results(
        result
    )

    create_anomaly_chart(
        result
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR /
        "sales_anomalies.csv"
    )

    print(
        OUTPUT_DIR /
        "anomaly_detection_summary.csv"
    )

    print(
        OUTPUT_DIR /
        "sales_anomalies.png"
    )

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()