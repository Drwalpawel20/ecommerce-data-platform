from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "analytics" / "ml" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


FORECAST_DAYS = 365
TEST_DAYS = 30


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

    df = df.dropna(
        subset=["full_date", "revenue"]
    )

    return df.reset_index(drop=True)


def create_features(df):
    data = df.copy()

    data["day_of_week"] = (
        data["full_date"].dt.dayofweek
    )

    data["day_of_month"] = (
        data["full_date"].dt.day
    )

    data["month"] = (
        data["full_date"].dt.month
    )

    data["quarter"] = (
        data["full_date"].dt.quarter
    )

    data["year"] = (
        data["full_date"].dt.year
    )

    data["day_of_year"] = (
        data["full_date"].dt.dayofyear
    )

    data["time_index"] = range(
        len(data)
    )

    data["sin_day_of_year"] = (
        __import__("numpy").sin(
            2
            * __import__("numpy").pi
            * data["day_of_year"]
            / 365.25
        )
    )

    data["cos_day_of_year"] = (
        __import__("numpy").cos(
            2
            * __import__("numpy").pi
            * data["day_of_year"]
            / 365.25
        )
    )

    data["lag_1"] = (
        data["revenue"].shift(1)
    )

    data["lag_7"] = (
        data["revenue"].shift(7)
    )

    data["lag_14"] = (
        data["revenue"].shift(14)
    )

    data["lag_30"] = (
        data["revenue"].shift(30)
    )

    data["lag_365"] = (
        data["revenue"].shift(365)
    )

    data["rolling_7"] = (
        data["revenue"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    data["rolling_14"] = (
        data["revenue"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    data["rolling_30"] = (
        data["revenue"]
        .shift(1)
        .rolling(30)
        .mean()
    )

    data["rolling_90"] = (
        data["revenue"]
        .shift(1)
        .rolling(90)
        .mean()
    )

    data = data.dropna().reset_index(
        drop=True
    )

    return data


def train_model(train_x, train_y):
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_leaf_nodes=20,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(
        train_x,
        train_y
    )

    return model


def evaluate_model(
    model,
    test_x,
    test_y
):
    predictions = model.predict(
        test_x
    )

    predictions = [
        max(0, float(value))
        for value in predictions
    ]

    mae = mean_absolute_error(
        test_y,
        predictions
    )

    rmse = mean_squared_error(
        test_y,
        predictions
    ) ** 0.5

    mae_percentage = (
        mae
        / test_y.mean()
        * 100
    )

    return (
        predictions,
        mae,
        rmse,
        mae_percentage
    )


def create_future_features(
    history,
    forecast_date
):
    import numpy as np

    data = history.copy()

    row = pd.DataFrame(
        [
            {
                "full_date": forecast_date,
                "revenue": None
            }
        ]
    )

    temp = pd.concat(
        [
            data[
                [
                    "full_date",
                    "revenue"
                ]
            ],
            row
        ],
        ignore_index=True
    )

    temp["day_of_week"] = (
        temp["full_date"].dt.dayofweek
    )

    temp["day_of_month"] = (
        temp["full_date"].dt.day
    )

    temp["month"] = (
        temp["full_date"].dt.month
    )

    temp["quarter"] = (
        temp["full_date"].dt.quarter
    )

    temp["year"] = (
        temp["full_date"].dt.year
    )

    temp["day_of_year"] = (
        temp["full_date"].dt.dayofyear
    )

    temp["time_index"] = range(
        len(temp)
    )

    temp["sin_day_of_year"] = (
        np.sin(
            2
            * np.pi
            * temp["day_of_year"]
            / 365.25
        )
    )

    temp["cos_day_of_year"] = (
        np.cos(
            2
            * np.pi
            * temp["day_of_year"]
            / 365.25
        )
    )

    temp["lag_1"] = (
        temp["revenue"].shift(1)
    )

    temp["lag_7"] = (
        temp["revenue"].shift(7)
    )

    temp["lag_14"] = (
        temp["revenue"].shift(14)
    )

    temp["lag_30"] = (
        temp["revenue"].shift(30)
    )

    temp["lag_365"] = (
        temp["revenue"].shift(365)
    )

    temp["rolling_7"] = (
        temp["revenue"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    temp["rolling_14"] = (
        temp["revenue"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    temp["rolling_30"] = (
        temp["revenue"]
        .shift(1)
        .rolling(30)
        .mean()
    )

    temp["rolling_90"] = (
        temp["revenue"]
        .shift(1)
        .rolling(90)
        .mean()
    )

    return temp.iloc[-1]


def forecast_future(
    model,
    history,
    feature_columns,
    forecast_days
):
    history = history[
        [
            "full_date",
            "revenue"
        ]
    ].copy()

    forecasts = []

    last_date = history[
        "full_date"
    ].max()

    for i in range(
        1,
        forecast_days + 1
    ):
        forecast_date = (
            last_date
            + pd.Timedelta(days=i)
        )

        row = create_future_features(
            history,
            forecast_date
        )

        features = row[
            feature_columns
        ].to_frame().T

        prediction = model.predict(
            features
        )[0]

        prediction = max(
            0,
            float(prediction)
        )

        forecasts.append(
            {
                "full_date": forecast_date,
                "forecast_revenue": prediction
            }
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "full_date": forecast_date,
                            "revenue": prediction
                        }
                    ]
                )
            ],
            ignore_index=True
        )

    return pd.DataFrame(
        forecasts
    )


def save_results(
    test_results,
    forecast_results,
    metrics
):
    test_results.to_csv(
        OUTPUT_DIR
        / "forecast_test_results.csv",
        index=False
    )

    forecast_results.to_csv(
        OUTPUT_DIR
        / "sales_forecast_365_days.csv",
        index=False
    )

    metrics.to_csv(
        OUTPUT_DIR
        / "forecast_metrics.csv",
        index=False
    )


def create_forecast_chart(
    actual,
    test_results,
    forecast_results
):
    plt.figure(figsize=(16, 7))

    plt.plot(
        actual["full_date"],
        actual["revenue"],
        label="Actual Revenue"
    )

    plt.plot(
        test_results["full_date"],
        test_results["predicted_revenue"],
        label="Test Prediction"
    )

    plt.plot(
        forecast_results["full_date"],
        forecast_results["forecast_revenue"],
        label="365-Day Forecast"
    )

    plt.axvline(
        actual["full_date"].max(),
        linestyle="--",
        label="Forecast Start"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Revenue"
    )

    plt.title(
        "365-Day Sales Revenue Forecast"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "sales_forecast_365_days.png"
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


def main():
    print("=" * 70)
    print("365-DAY SALES FORECASTING")
    print("=" * 70)

    print(
        f"\nForecast horizon: "
        f"{FORECAST_DAYS} days"
    )

    print(
        f"Test period: "
        f"{TEST_DAYS} days"
    )

    connection = get_connection()

    try:
        df = load_sales_data(
            connection
        )
    finally:
        connection.close()

    print(
        f"\nLoaded {len(df):,} daily observations."
    )

    data = create_features(
        df
    )

    feature_columns = [
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "year",
        "day_of_year",
        "time_index",
        "sin_day_of_year",
        "cos_day_of_year",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "lag_365",
        "rolling_7",
        "rolling_14",
        "rolling_30",
        "rolling_90"
    ]

    train = data.iloc[
        :-TEST_DAYS
    ].copy()

    test = data.iloc[
        -TEST_DAYS:
    ].copy()

    train_x = train[
        feature_columns
    ]

    train_y = train[
        "revenue"
    ]

    test_x = test[
        feature_columns
    ]

    test_y = test[
        "revenue"
    ]

    print(
        f"\nTraining observations: "
        f"{len(train):,}"
    )

    print(
        f"Testing observations: "
        f"{len(test):,}"
    )

    model = train_model(
        train_x,
        train_y
    )

    (
        predictions,
        mae,
        rmse,
        mae_percentage
    ) = evaluate_model(
        model,
        test_x,
        test_y
    )

    test_results = test[
        [
            "full_date",
            "revenue"
        ]
    ].copy()

    test_results[
        "predicted_revenue"
    ] = predictions

    metrics = pd.DataFrame(
        {
            "metric": [
                "MAE",
                "RMSE",
                "MAE_percentage",
                "Test_Days",
                "Forecast_Days"
            ],
            "value": [
                mae,
                rmse,
                mae_percentage,
                TEST_DAYS,
                FORECAST_DAYS
            ]
        }
    )

    print(
        "\n=== MODEL METRICS ==="
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    history = df[
        [
            "full_date",
            "revenue"
        ]
    ].copy()

    forecast_results = forecast_future(
        model,
        history,
        feature_columns,
        FORECAST_DAYS
    )

    print(
        "\n=== 365-DAY FORECAST ==="
    )

    display = forecast_results.copy()

    display[
        "forecast_revenue"
    ] = display[
        "forecast_revenue"
    ].round(2)

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\n=== FORECAST SUMMARY ==="
    )

    print(
        f"Forecasted revenue: "
        f"{forecast_results['forecast_revenue'].sum():,.2f}"
    )

    print(
        f"Average daily forecast: "
        f"{forecast_results['forecast_revenue'].mean():,.2f}"
    )

    print(
        f"Minimum daily forecast: "
        f"{forecast_results['forecast_revenue'].min():,.2f}"
    )

    print(
        f"Maximum daily forecast: "
        f"{forecast_results['forecast_revenue'].max():,.2f}"
    )

    save_results(
        test_results,
        forecast_results,
        metrics
    )

    create_forecast_chart(
        df,
        test_results,
        forecast_results
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "forecast_test_results.csv"
    )

    print(
        OUTPUT_DIR
        / "sales_forecast_365_days.csv"
    )

    print(
        OUTPUT_DIR
        / "forecast_metrics.csv"
    )

    print(
        OUTPUT_DIR
        / "sales_forecast_365_days.png"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "365-DAY FORECASTING COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()