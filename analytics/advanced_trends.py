from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import load_daily_sales


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analytics" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data(df):
    df = df.copy()

    df["full_date"] = pd.to_datetime(df["full_date"])
    df = df.sort_values("full_date")

    df["revenue"] = df["revenue"].astype(float)
    df["profit"] = df["profit"].astype(float)
    df["orders"] = df["orders"].astype(int)
    df["units_sold"] = df["units_sold"].astype(int)

    df["aov"] = df["revenue"] / df["orders"]

    df["profit_margin"] = (
        df["profit"] / df["revenue"] * 100
    )

    df["revenue_7d"] = (
        df["revenue"]
        .rolling(7)
        .mean()
    )

    df["revenue_30d"] = (
        df["revenue"]
        .rolling(30)
        .mean()
    )

    df["revenue_90d"] = (
        df["revenue"]
        .rolling(90)
        .mean()
    )

    df["profit_30d"] = (
        df["profit"]
        .rolling(30)
        .mean()
    )

    df["orders_30d"] = (
        df["orders"]
        .rolling(30)
        .mean()
    )

    df["revenue_volatility_30d"] = (
        df["revenue"]
        .rolling(30)
        .std()
    )

    df["revenue_mom"] = (
        df["revenue"]
        .pct_change(periods=30) * 100
    )

    df["revenue_yoy"] = (
        df["revenue"]
        .pct_change(periods=365) * 100
    )

    df["profit_yoy"] = (
        df["profit"]
        .pct_change(periods=365) * 100
    )

    df["orders_yoy"] = (
        df["orders"]
        .pct_change(periods=365) * 100
    )

    df["revenue_cumulative"] = (
        df["revenue"].cumsum()
    )

    df["profit_cumulative"] = (
        df["profit"].cumsum()
    )

    first_revenue = df["revenue"].iloc[0]

    df["revenue_index"] = (
        df["revenue"] / first_revenue * 100
    )

    return df


def calculate_trend(df):
    x = np.arange(len(df))
    y = df["revenue"].values

    slope, intercept = np.polyfit(x, y, 1)

    trend = slope * x + intercept

    ss_res = np.sum((y - trend) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (ss_res / ss_tot)

    print("\n=== LONG-TERM TREND ===")

    print(f"Slope: {slope:,.2f}")
    print(f"R²: {r_squared:.4f}")

    if slope > 0:
        direction = "Increasing"
    elif slope < 0:
        direction = "Decreasing"
    else:
        direction = "Stable"

    print(f"Trend direction: {direction}")

    return trend, slope, r_squared


def calculate_period_statistics(df):
    monthly = (
        df.assign(
            year_month=df["full_date"].dt.to_period("M")
        )
        .groupby("year_month")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("orders", "sum"),
            units_sold=("units_sold", "sum")
        )
    )

    monthly["aov"] = (
        monthly["revenue"] /
        monthly["orders"]
    )

    monthly["profit_margin"] = (
        monthly["profit"] /
        monthly["revenue"] * 100
    )

    monthly["revenue_growth"] = (
        monthly["revenue"]
        .pct_change() * 100
    )

    print("\n=== MONTHLY PERFORMANCE ===")

    print(
        monthly
        .sort_values("revenue", ascending=False)
        .head(10)
        .round(2)
        .to_string()
    )

    print("\nTop 10 months by revenue:")
    print(
        monthly
        .nlargest(10, "revenue")[
            ["revenue", "profit", "orders", "aov"]
        ]
        .round(2)
        .to_string()
    )

    print("\nBottom 10 months by revenue:")
    print(
        monthly
        .nsmallest(10, "revenue")[
            ["revenue", "profit", "orders", "aov"]
        ]
        .round(2)
        .to_string()
    )

    return monthly


def plot_advanced_revenue_trend(df, trend):
    plt.figure(figsize=(15, 7))

    plt.plot(
        df["full_date"],
        df["revenue"],
        alpha=0.25,
        label="Daily Revenue"
    )

    plt.plot(
        df["full_date"],
        df["revenue_7d"],
        label="7-Day Rolling Mean",
        linewidth=1.5
    )

    plt.plot(
        df["full_date"],
        df["revenue_30d"],
        label="30-Day Rolling Mean",
        linewidth=2
    )

    plt.plot(
        df["full_date"],
        df["revenue_90d"],
        label="90-Day Rolling Mean",
        linewidth=2
    )

    plt.plot(
        df["full_date"],
        trend,
        linestyle="--",
        linewidth=2,
        label="Linear Trend"
    )

    plt.title("Advanced Revenue Trend Analysis")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "advanced_revenue_trend.png"

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def plot_profit_trend(df):
    plt.figure(figsize=(15, 7))

    plt.plot(
        df["full_date"],
        df["profit"],
        alpha=0.25,
        label="Daily Profit"
    )

    plt.plot(
        df["full_date"],
        df["profit_30d"],
        linewidth=2,
        label="30-Day Rolling Mean"
    )

    plt.title("Advanced Profit Trend")
    plt.xlabel("Date")
    plt.ylabel("Profit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "advanced_profit_trend.png"

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def plot_growth_rates(df):
    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.plot(
        df["full_date"],
        df["revenue_mom"],
        label="30-Day Growth"
    )

    ax.plot(
        df["full_date"],
        df["revenue_yoy"],
        label="YoY Growth"
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Revenue Growth Rates"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Growth (%)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "growth_rates.png"

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def plot_volatility(df):
    plt.figure(figsize=(15, 7))

    plt.plot(
        df["full_date"],
        df["revenue_volatility_30d"],
        label="30-Day Revenue Volatility"
    )

    plt.title(
        "Rolling Revenue Volatility"
    )

    plt.xlabel("Date")
    plt.ylabel("Standard Deviation")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "rolling_volatility.png"

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def plot_cumulative_performance(df):
    plt.figure(figsize=(15, 7))

    plt.plot(
        df["full_date"],
        df["revenue_cumulative"],
        label="Cumulative Revenue",
        linewidth=2
    )

    plt.plot(
        df["full_date"],
        df["profit_cumulative"],
        label="Cumulative Profit",
        linewidth=2
    )

    plt.title(
        "Cumulative Business Performance"
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "cumulative_performance.png"

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def plot_profit_margin(df):
    plt.figure(figsize=(15, 7))

    rolling_margin = (
        df["profit_margin"]
        .rolling(30)
        .mean()
    )

    plt.plot(
        df["full_date"],
        df["profit_margin"],
        alpha=0.25,
        label="Daily Profit Margin"
    )

    plt.plot(
        df["full_date"],
        rolling_margin,
        linewidth=2,
        label="30-Day Rolling Margin"
    )

    plt.title(
        "Profit Margin Trend"
    )

    plt.xlabel("Date")
    plt.ylabel("Profit Margin (%)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "profit_margin_trend.png"

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def plot_aov(df):
    plt.figure(figsize=(15, 7))

    aov_30d = (
        df["aov"]
        .rolling(30)
        .mean()
    )

    plt.plot(
        df["full_date"],
        df["aov"],
        alpha=0.25,
        label="Daily AOV"
    )

    plt.plot(
        df["full_date"],
        aov_30d,
        linewidth=2,
        label="30-Day Rolling AOV"
    )

    plt.title(
        "Average Order Value Trend"
    )

    plt.xlabel("Date")
    plt.ylabel("AOV")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "aov_trend.png"

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def plot_revenue_index(df):
    plt.figure(figsize=(15, 7))

    plt.plot(
        df["full_date"],
        df["revenue_index"],
        linewidth=2
    )

    plt.axhline(
        100,
        linestyle="--",
        linewidth=1
    )

    plt.title(
        "Revenue Performance Index"
    )

    plt.xlabel("Date")
    plt.ylabel(
        "Index (First Day = 100)"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "revenue_index.png"

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    print("Loading daily sales data...")

    df = load_daily_sales()

    print(f"Loaded {len(df):,} rows.")

    df = prepare_data(df)

    trend, slope, r_squared = calculate_trend(df)

    calculate_period_statistics(df)

    print("\n=== KEY METRICS ===")

    print(
        f"Average daily revenue: "
        f"{df['revenue'].mean():,.2f}"
    )

    print(
        f"Average daily profit: "
        f"{df['profit'].mean():,.2f}"
    )

    print(
        f"Average AOV: "
        f"{df['aov'].mean():,.2f}"
    )

    print(
        f"Average profit margin: "
        f"{df['profit_margin'].mean():.2f}%"
    )

    print(
        f"Maximum revenue: "
        f"{df['revenue'].max():,.2f}"
    )

    print(
        f"Minimum revenue: "
        f"{df['revenue'].min():,.2f}"
    )

    print(
        f"Revenue coefficient of variation: "
        f"{df['revenue'].std() / df['revenue'].mean():.4f}"
    )

    plot_advanced_revenue_trend(
        df,
        trend
    )

    plot_profit_trend(df)

    plot_growth_rates(df)

    plot_volatility(df)

    plot_cumulative_performance(df)

    plot_profit_margin(df)

    plot_aov(df)

    plot_revenue_index(df)

    print("\nAdvanced trend analysis completed.")


if __name__ == "__main__":
    main()