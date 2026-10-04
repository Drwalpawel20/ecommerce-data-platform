from pathlib import Path

import matplotlib.pyplot as plt
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

    return df


def analyze_monthly_seasonality(df):
    monthly = (
        df.groupby(df.index.month)["revenue"]
        .agg(["mean", "sum", "median"])
    )

    monthly.index = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    print("\n=== MONTHLY SEASONALITY ===")
    print(monthly.round(2))

    best_month = monthly["mean"].idxmax()
    worst_month = monthly["mean"].idxmin()

    print(f"\nBest month: {best_month}")
    print(f"Worst month: {worst_month}")

    plt.figure(figsize=(14, 7))

    plt.bar(
        monthly.index,
        monthly["mean"]
    )

    plt.title("Average Daily Revenue by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Daily Revenue")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "monthly_revenue.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Chart saved to: {output_path}")


def analyze_weekday_seasonality(df):
    weekday = (
        df.groupby(df.index.dayofweek)["revenue"]
        .agg(["mean", "sum", "median"])
    )

    weekday.index = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    print("\n=== WEEKDAY SEASONALITY ===")
    print(weekday.round(2))

    best_day = weekday["mean"].idxmax()
    worst_day = weekday["mean"].idxmin()

    print(f"\nBest day: {best_day}")
    print(f"Worst day: {worst_day}")

    plt.figure(figsize=(12, 6))

    plt.bar(
        weekday.index,
        weekday["mean"]
    )

    plt.title("Average Daily Revenue by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Daily Revenue")
    plt.xticks(rotation=30)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "weekday_revenue.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Chart saved to: {output_path}")


def analyze_quarter_seasonality(df):
    quarterly = (
        df.groupby(df.index.quarter)["revenue"]
        .agg(["mean", "sum", "median"])
    )

    quarterly.index = [
        "Q1",
        "Q2",
        "Q3",
        "Q4"
    ]

    print("\n=== QUARTERLY SEASONALITY ===")
    print(quarterly.round(2))

    best_quarter = quarterly["mean"].idxmax()
    worst_quarter = quarterly["mean"].idxmin()

    print(f"\nBest quarter: {best_quarter}")
    print(f"Worst quarter: {worst_quarter}")

    plt.figure(figsize=(10, 6))

    plt.bar(
        quarterly.index,
        quarterly["mean"]
    )

    plt.title("Average Daily Revenue by Quarter")
    plt.xlabel("Quarter")
    plt.ylabel("Average Daily Revenue")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "quarterly_revenue.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Chart saved to: {output_path}")


def perform_stl_decomposition(df):
    daily_revenue = df["revenue"].asfreq("D")

    daily_revenue = daily_revenue.interpolate()

    stl = STL(
        daily_revenue,
        period=365,
        robust=True
    )

    result = stl.fit()

    print("\n=== STL DECOMPOSITION ===")

    print("\nTrend statistics:")
    print(result.trend.describe())

    print("\nSeasonal statistics:")
    print(result.seasonal.describe())

    print("\nResidual statistics:")
    print(result.resid.describe())

    fig = result.plot()

    fig.set_size_inches(15, 10)

    fig.suptitle(
        "STL Decomposition of Daily Revenue",
        fontsize=16
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "seasonal_decomposition.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nSTL decomposition saved to: {output_path}")


def analyze_seasonality_strength(df):
    daily_revenue = df["revenue"].asfreq("D")
    daily_revenue = daily_revenue.interpolate()

    stl = STL(
        daily_revenue,
        period=365,
        robust=True
    ).fit()

    seasonal_variance = stl.seasonal.var()
    residual_variance = stl.resid.var()

    strength = max(
        0,
        1 - residual_variance /
        (seasonal_variance + residual_variance)
    )

    print("\n=== SEASONALITY STRENGTH ===")
    print(f"Seasonal variance: {seasonal_variance:.2f}")
    print(f"Residual variance: {residual_variance:.2f}")
    print(f"Seasonality strength: {strength:.4f}")

    if strength >= 0.6:
        interpretation = "Strong seasonality"
    elif strength >= 0.3:
        interpretation = "Moderate seasonality"
    else:
        interpretation = "Weak seasonality"

    print(f"Interpretation: {interpretation}")


def main():
    print("Loading sales data...")

    df = load_daily_sales()

    print(f"Loaded {len(df):,} rows.")

    df = prepare_data(df)

    analyze_monthly_seasonality(df)

    analyze_weekday_seasonality(df)

    analyze_quarter_seasonality(df)

    perform_stl_decomposition(df)

    analyze_seasonality_strength(df)

    print("\nSeasonality analysis completed.")


if __name__ == "__main__":
    main()