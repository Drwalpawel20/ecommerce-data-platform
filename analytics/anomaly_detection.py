from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from load_data import load_daily_sales


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analytics" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data(df):
    df = df.copy()

    df["full_date"] = pd.to_datetime(df["full_date"])
    df = df.sort_values("full_date")
    df = df.set_index("full_date")

    numeric_columns = [
        "revenue",
        "profit",
        "orders",
        "units_sold"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["aov"] = (
        df["revenue"] /
        df["orders"]
    )

    return df


def calculate_global_zscore(series):
    mean = series.mean()
    std = series.std()

    if std == 0:
        return pd.Series(
            0,
            index=series.index
        )

    return (
        (series - mean) /
        std
    )


def calculate_iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper


def add_global_signals(df):
    df = df.copy()

    print("\n=== GLOBAL ANOMALY SIGNALS ===")

    metrics = [
        "revenue",
        "profit",
        "orders"
    ]

    for metric in metrics:

        zscore = calculate_global_zscore(
            df[metric]
        )

        df[f"{metric}_global_zscore"] = zscore

        df[f"{metric}_global_signal"] = (
            zscore.abs() >= 3
        )

        lower, upper = calculate_iqr_bounds(
            df[metric]
        )

        df[f"{metric}_iqr_signal"] = (
            (df[metric] < lower) |
            (df[metric] > upper)
        )

        print(
            f"{metric.capitalize()}: "
            f"Z-score = "
            f"{df[f'{metric}_global_signal'].sum()}, "
            f"IQR = "
            f"{df[f'{metric}_iqr_signal'].sum()}"
        )

    return df


def add_rolling_signals(df):
    df = df.copy()

    print("\n=== ROLLING ANOMALY SIGNALS ===")

    window = 30

    for metric in [
        "revenue",
        "profit",
        "orders"
    ]:

        rolling_mean = (
            df[metric]
            .rolling(
                window,
                min_periods=15
            )
            .mean()
        )

        rolling_std = (
            df[metric]
            .rolling(
                window,
                min_periods=15
            )
            .std()
        )

        rolling_median = (
            df[metric]
            .rolling(
                window,
                min_periods=15
            )
            .median()
        )

        df[f"{metric}_rolling_zscore"] = (
            (
                df[metric] -
                rolling_mean
            )
            /
            rolling_std.replace(
                0,
                np.nan
            )
        )

        df[f"{metric}_rolling_median"] = (
            rolling_median
        )

        df[f"{metric}_rolling_deviation"] = (
            (
                df[metric] -
                rolling_median
            )
            /
            rolling_median
            * 100
        )

        df[f"{metric}_rolling_signal"] = (
            df[f"{metric}_rolling_zscore"]
            .abs() >= 3
        )

    print(
        "Rolling window: 30 days"
    )

    return df


def add_yoy_signals(df):
    df = df.copy()

    print("\n=== YOY ANOMALY SIGNALS ===")

    for metric in [
        "revenue",
        "profit",
        "orders"
    ]:

        previous_year = (
            df[metric]
            .shift(365)
        )

        df[f"{metric}_yoy_expected"] = (
            previous_year
        )

        df[f"{metric}_yoy_deviation"] = (
            (
                df[metric] -
                previous_year
            )
            /
            previous_year
            * 100
        )

        df[f"{metric}_yoy_signal"] = (
            df[f"{metric}_yoy_deviation"]
            .abs() >= 30
        )

        count = df[
            f"{metric}_yoy_signal"
        ].sum()

        print(
            f"{metric.capitalize()}: "
            f"{count} YoY anomalies"
        )

    return df


def add_stl_signals(df):
    df = df.copy()

    print("\n=== STL ANOMALY SIGNAL ===")

    daily_revenue = (
        df["revenue"]
        .asfreq("D")
        .interpolate()
    )

    stl = STL(
        daily_revenue,
        period=365,
        robust=True
    )

    result = stl.fit()

    df["stl_trend"] = result.trend
    df["stl_seasonal"] = result.seasonal
    df["stl_residual"] = result.resid

    residual_mean = (
        df["stl_residual"]
        .rolling(
            30,
            min_periods=15
        )
        .mean()
    )

    residual_std = (
        df["stl_residual"]
        .rolling(
            30,
            min_periods=15
        )
        .std()
    )

    df["stl_zscore"] = (
        (
            df["stl_residual"] -
            residual_mean
        )
        /
        residual_std.replace(
            0,
            np.nan
        )
    )

    df["stl_signal"] = (
        df["stl_zscore"].abs() >= 3
    )

    print(
        f"STL anomalies: "
        f"{df['stl_signal'].sum()}"
    )

    return df


def add_expected_value(df):
    df = df.copy()

    print("\n=== EXPECTED VALUE MODEL ===")

    rolling_median = (
        df["revenue"]
        .rolling(
            30,
            min_periods=15
        )
        .median()
    )

    yoy_value = (
        df["revenue"]
        .shift(365)
    )

    expected_values = pd.concat(
        [
            rolling_median.rename(
                "rolling"
            ),
            yoy_value.rename(
                "yoy"
            ),
            df["stl_trend"].rename(
                "trend"
            )
        ],
        axis=1
    )

    df["expected_revenue"] = (
        expected_values
        .median(axis=1)
    )

    df["expected_deviation"] = (
        (
            df["revenue"] -
            df["expected_revenue"]
        )
        /
        df["expected_revenue"]
        * 100
    )

    df["expected_signal"] = (
        df["expected_deviation"]
        .abs() >= 30
    )

    return df


def create_ensemble_score(df):
    df = df.copy()

    signal_columns = [
        "revenue_global_signal",
        "revenue_iqr_signal",
        "revenue_rolling_signal",
        "revenue_yoy_signal",
        "stl_signal",
        "expected_signal"
    ]

    df["anomaly_score"] = (
        df[signal_columns]
        .fillna(False)
        .astype(int)
        .sum(axis=1)
    )

    df["anomaly_direction"] = np.where(
        df["expected_deviation"] > 0,
        "Positive",
        "Negative"
    )

    df["anomaly_level"] = "Normal"

    df.loc[
        df["anomaly_score"] >= 2,
        "anomaly_level"
    ] = "Warning"

    df.loc[
        df["anomaly_score"] >= 3,
        "anomaly_level"
    ] = "Anomaly"

    df.loc[
        df["anomaly_score"] >= 5,
        "anomaly_level"
    ] = "Extreme Anomaly"

    print("\n=== ENSEMBLE ANOMALY SCORE ===")

    print(
        df["anomaly_level"]
        .value_counts()
        .to_string()
    )

    return df


def print_top_anomalies(df):
    columns = [
        "revenue",
        "profit",
        "orders",
        "expected_revenue",
        "expected_deviation",
        "revenue_global_zscore",
        "revenue_rolling_zscore",
        "revenue_yoy_deviation",
        "stl_zscore",
        "anomaly_score",
        "anomaly_level",
        "anomaly_direction"
    ]

    anomalies = (
        df[
            df["anomaly_score"] >= 2
        ]
        .sort_values(
            [
                "anomaly_score",
                "expected_deviation"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    print("\n=== TOP 30 ANOMALIES ===")

    if anomalies.empty:
        print(
            "No significant anomalies detected."
        )
        return

    print(
        anomalies[columns]
        .head(30)
        .round(2)
        .to_string()
    )


def save_anomaly_report(df):
    report_columns = [
        "revenue",
        "profit",
        "orders",
        "expected_revenue",
        "expected_deviation",
        "revenue_global_zscore",
        "revenue_rolling_zscore",
        "revenue_yoy_deviation",
        "stl_zscore",
        "anomaly_score",
        "anomaly_level",
        "anomaly_direction"
    ]

    report = (
        df[
            df["anomaly_score"] >= 2
        ][report_columns]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .reset_index()
    )

    output_path = (
        OUTPUT_DIR /
        "anomaly_report.csv"
    )

    report.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved anomaly report: "
        f"{output_path}"
    )


def plot_revenue_anomalies(df):
    plt.figure(figsize=(16, 8))

    plt.plot(
        df.index,
        df["revenue"],
        alpha=0.35,
        label="Daily Revenue"
    )

    plt.plot(
        df.index,
        df["expected_revenue"],
        linewidth=2,
        label="Expected Revenue"
    )

    warning = df[
        df["anomaly_level"] == "Warning"
    ]

    anomaly = df[
        df["anomaly_level"] == "Anomaly"
    ]

    extreme = df[
        df["anomaly_level"] ==
        "Extreme Anomaly"
    ]

    plt.scatter(
        warning.index,
        warning["revenue"],
        s=25,
        label="Warning"
    )

    plt.scatter(
        anomaly.index,
        anomaly["revenue"],
        s=45,
        label="Anomaly"
    )

    plt.scatter(
        extreme.index,
        extreme["revenue"],
        s=80,
        label="Extreme Anomaly"
    )

    plt.title(
        "Ensemble Revenue Anomaly Detection"
    )

    plt.xlabel("Date")
    plt.ylabel("Revenue")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "advanced_revenue_anomalies.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_expected_deviation(df):
    plt.figure(figsize=(16, 7))

    plt.plot(
        df.index,
        df["expected_deviation"],
        linewidth=1
    )

    plt.axhline(
        30,
        linestyle="--",
        linewidth=1,
        label="+30%"
    )

    plt.axhline(
        -30,
        linestyle="--",
        linewidth=1,
        label="-30%"
    )

    plt.axhline(
        0,
        linestyle=":",
        linewidth=1
    )

    anomalies = df[
        df["expected_signal"]
    ]

    plt.scatter(
        anomalies.index,
        anomalies["expected_deviation"],
        s=35,
        label="Expected Value Anomaly"
    )

    plt.title(
        "Revenue Deviation from Expected Value"
    )

    plt.xlabel("Date")
    plt.ylabel("Deviation (%)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "expected_revenue_deviation.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_anomaly_score(df):
    plt.figure(figsize=(16, 7))

    plt.plot(
        df.index,
        df["anomaly_score"],
        linewidth=1
    )

    plt.axhline(
        2,
        linestyle="--",
        linewidth=1,
        label="Warning"
    )

    plt.axhline(
        3,
        linestyle="--",
        linewidth=1,
        label="Anomaly"
    )

    plt.axhline(
        5,
        linestyle="--",
        linewidth=1,
        label="Extreme Anomaly"
    )

    plt.title(
        "Ensemble Anomaly Score"
    )

    plt.xlabel("Date")
    plt.ylabel("Anomaly Score")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "advanced_anomaly_score.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_stl_residuals(df):
    plt.figure(figsize=(16, 7))

    plt.plot(
        df.index,
        df["stl_residual"],
        alpha=0.6,
        label="STL Residual"
    )

    anomalies = df[
        df["stl_signal"]
    ]

    plt.scatter(
        anomalies.index,
        anomalies["stl_residual"],
        s=40,
        label="STL Anomaly"
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.title(
        "STL Residual Anomaly Detection"
    )

    plt.xlabel("Date")
    plt.ylabel("Residual")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "advanced_stl_residuals.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def main():
    print(
        "Loading daily sales data..."
    )

    df = load_daily_sales()

    print(
        f"Loaded {len(df):,} rows."
    )

    df = prepare_data(df)

    df = add_global_signals(df)

    df = add_rolling_signals(df)

    df = add_yoy_signals(df)

    df = add_stl_signals(df)

    df = add_expected_value(df)

    df = create_ensemble_score(df)

    print_top_anomalies(df)

    save_anomaly_report(df)

    plot_revenue_anomalies(df)

    plot_expected_deviation(df)

    plot_anomaly_score(df)

    plot_stl_residuals(df)

    print(
        "\nAdvanced anomaly detection "
        "completed."
    )


if __name__ == "__main__":
    main()