from pathlib import Path

import matplotlib.pyplot as plt

from load_data import load_daily_sales


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analytics" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_trends(df):
    df = df.copy()

    df["full_date"] = df["full_date"].astype("datetime64[ns]")
    df = df.sort_values("full_date")

    df["revenue_30d_avg"] = (
        df["revenue"]
        .rolling(window=30, min_periods=1)
        .mean()
    )

    df["profit_30d_avg"] = (
        df["profit"]
        .rolling(window=30, min_periods=1)
        .mean()
    )

    print("\n=== SALES TREND ANALYSIS ===")

    print("\nRevenue statistics:")
    print(df["revenue"].describe())

    print("\nProfit statistics:")
    print(df["profit"].describe())

    print("\nHighest revenue days:")
    print(
        df.nlargest(10, "revenue")[
            ["full_date", "revenue", "profit", "orders"]
        ].to_string(index=False)
    )

    print("\nHighest profit days:")
    print(
        df.nlargest(10, "profit")[
            ["full_date", "revenue", "profit", "orders"]
        ].to_string(index=False)
    )

    return df


def plot_revenue_trend(df):
    plt.figure(figsize=(14, 7))

    plt.plot(
        df["full_date"],
        df["revenue"],
        label="Daily Revenue"
    )

    plt.plot(
        df["full_date"],
        df["revenue_30d_avg"],
        label="30-Day Moving Average",
        linewidth=2
    )

    plt.title("Revenue Trend")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "revenue_trend.png"

    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

    print(f"\nChart saved to: {output_path}")


def main():
    df = load_daily_sales()

    df = analyze_trends(df)

    plot_revenue_trend(df)


if __name__ == "__main__":
    main()