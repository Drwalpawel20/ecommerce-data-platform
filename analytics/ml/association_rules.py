from pathlib import Path
from itertools import combinations
import os

import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "analytics" / "ml" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


load_dotenv(PROJECT_ROOT / ".env")


MIN_PRODUCT_ORDERS = 100
MIN_PAIR_ORDERS = 2
MIN_LIFT = 1.0
TOP_N = 20


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def load_transactions(connection):
    query = """
        SELECT
            fs.order_id,
            dp.product_id,
            dp.product_name
        FROM warehouse.fact_sales fs
        INNER JOIN warehouse.dim_product dp
            ON fs.product_key = dp.product_key
        WHERE fs.order_id IS NOT NULL
          AND dp.product_id IS NOT NULL
          AND dp.product_name IS NOT NULL
    """

    df = pd.read_sql_query(
        query,
        connection
    )

    df = df.drop_duplicates(
        subset=["order_id", "product_id"]
    )

    print(
        f"Loaded {len(df):,} unique order-product pairs."
    )

    print(
        f"Orders: {df['order_id'].nunique():,}"
    )

    print(
        f"Products: {df['product_id'].nunique():,}"
    )

    return df


def filter_products(df, min_product_orders):
    product_order_counts = (
        df.groupby("product_id")["order_id"]
        .nunique()
    )

    valid_products = product_order_counts[
        product_order_counts >= min_product_orders
    ].index

    filtered = df[
        df["product_id"].isin(valid_products)
    ].copy()

    print(
        f"Products appearing in at least "
        f"{min_product_orders} orders: "
        f"{len(valid_products):,}"
    )

    print(
        f"Order-product pairs after filtering: "
        f"{len(filtered):,}"
    )

    print(
        f"Orders after product filtering: "
        f"{filtered['order_id'].nunique():,}"
    )

    return filtered


def create_baskets(df):
    baskets = (
        df.groupby("order_id")["product_id"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="products")
    )

    baskets = baskets[
        baskets["products"].apply(len) >= 2
    ].copy()

    print(
        f"Baskets with at least 2 products: "
        f"{len(baskets):,}"
    )

    return baskets


def calculate_product_counts(df):
    return (
        df.groupby("product_id")["order_id"]
        .nunique()
        .to_dict()
    )


def calculate_pairs(baskets, min_pair_orders):
    pair_counter = {}

    for products in baskets["products"]:
        for product_a, product_b in combinations(products, 2):
            pair = (product_a, product_b)

            pair_counter[pair] = (
                pair_counter.get(pair, 0) + 1
            )

    pairs = pd.DataFrame(
        [
            {
                "product_a": product_a,
                "product_b": product_b,
                "pair_orders": pair_orders
            }
            for (product_a, product_b), pair_orders
            in pair_counter.items()
            if pair_orders >= min_pair_orders
        ]
    )

    if pairs.empty:
        return pairs

    return pairs.sort_values(
        "pair_orders",
        ascending=False
    ).reset_index(drop=True)


def calculate_metrics(
    pairs,
    product_counts,
    total_orders,
    min_lift
):
    if pairs.empty:
        return pairs

    pairs = pairs.copy()

    pairs["product_a_orders"] = (
        pairs["product_a"].map(product_counts)
    )

    pairs["product_b_orders"] = (
        pairs["product_b"].map(product_counts)
    )

    pairs["support"] = (
        pairs["pair_orders"] / total_orders
    )

    pairs["support_a"] = (
        pairs["product_a_orders"] / total_orders
    )

    pairs["support_b"] = (
        pairs["product_b_orders"] / total_orders
    )

    pairs["confidence_a_to_b"] = (
        pairs["pair_orders"]
        / pairs["product_a_orders"]
    )

    pairs["confidence_b_to_a"] = (
        pairs["pair_orders"]
        / pairs["product_b_orders"]
    )

    pairs["lift_a_to_b"] = (
        pairs["confidence_a_to_b"]
        / pairs["support_b"]
    )

    pairs["lift_b_to_a"] = (
        pairs["confidence_b_to_a"]
        / pairs["support_a"]
    )

    pairs["max_lift"] = pairs[
        [
            "lift_a_to_b",
            "lift_b_to_a"
        ]
    ].max(axis=1)

    pairs["max_confidence"] = pairs[
        [
            "confidence_a_to_b",
            "confidence_b_to_a"
        ]
    ].max(axis=1)

    pairs = pairs[
        pairs["max_lift"] >= min_lift
    ].copy()

    return pairs.sort_values(
        [
            "max_lift",
            "pair_orders"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(drop=True)


def add_product_names(pairs, transactions):
    if pairs.empty:
        return pairs

    product_names = (
        transactions[
            [
                "product_id",
                "product_name"
            ]
        ]
        .drop_duplicates("product_id")
    )

    product_name_map = dict(
        zip(
            product_names["product_id"],
            product_names["product_name"]
        )
    )

    pairs = pairs.copy()

    pairs["product_a_name"] = (
        pairs["product_a"].map(product_name_map)
    )

    pairs["product_b_name"] = (
        pairs["product_b"].map(product_name_map)
    )

    return pairs


def create_directional_rules(pairs):
    if pairs.empty:
        return pd.DataFrame()

    rules_a_to_b = pd.DataFrame({
        "antecedent_product_id":
            pairs["product_a"],
        "antecedent_product_name":
            pairs["product_a_name"],
        "consequent_product_id":
            pairs["product_b"],
        "consequent_product_name":
            pairs["product_b_name"],
        "pair_orders":
            pairs["pair_orders"],
        "support":
            pairs["support"],
        "confidence":
            pairs["confidence_a_to_b"],
        "lift":
            pairs["lift_a_to_b"]
    })

    rules_b_to_a = pd.DataFrame({
        "antecedent_product_id":
            pairs["product_b"],
        "antecedent_product_name":
            pairs["product_b_name"],
        "consequent_product_id":
            pairs["product_a"],
        "consequent_product_name":
            pairs["product_a_name"],
        "pair_orders":
            pairs["pair_orders"],
        "support":
            pairs["support"],
        "confidence":
            pairs["confidence_b_to_a"],
        "lift":
            pairs["lift_b_to_a"]
    })

    rules = pd.concat(
        [
            rules_a_to_b,
            rules_b_to_a
        ],
        ignore_index=True
    )

    return rules.sort_values(
        [
            "lift",
            "confidence",
            "pair_orders"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).reset_index(drop=True)


def create_summary(pairs, rules):
    if pairs.empty or rules.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "metric": [
            "unique_product_pairs",
            "directional_rules",
            "average_support",
            "average_confidence",
            "average_lift",
            "maximum_lift",
            "maximum_confidence"
        ],
        "value": [
            len(pairs),
            len(rules),
            pairs["support"].mean(),
            rules["confidence"].mean(),
            rules["lift"].mean(),
            rules["lift"].max(),
            rules["confidence"].max()
        ]
    })


def save_outputs(pairs, rules, summary):
    pairs_path = (
        OUTPUT_DIR / "product_pair_analysis.csv"
    )

    rules_path = (
        OUTPUT_DIR / "association_rules.csv"
    )

    summary_path = (
        OUTPUT_DIR / "association_rules_summary.csv"
    )

    pairs.to_csv(
        pairs_path,
        index=False
    )

    rules.to_csv(
        rules_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\nSaved:")
    print(pairs_path)
    print(rules_path)
    print(summary_path)


def print_top_pairs(pairs, n=20):
    print("\n=== TOP PRODUCT PAIRS ===")

    if pairs.empty:
        print("No product pairs found.")
        return

    top = pairs.head(n).copy()

    display = top[
        [
            "product_a_name",
            "product_b_name",
            "pair_orders",
            "support",
            "max_confidence",
            "max_lift"
        ]
    ].copy()

    display["support"] = (
        display["support"] * 100
    ).round(4)

    display["max_confidence"] = (
        display["max_confidence"] * 100
    ).round(4)

    display["max_lift"] = (
        display["max_lift"]
    ).round(4)

    display = display.rename(
        columns={
            "product_a_name": "Product A",
            "product_b_name": "Product B",
            "pair_orders": "Pair Orders",
            "support": "Support %",
            "max_confidence": "Max Confidence %",
            "max_lift": "Max Lift"
        }
    )

    print(
        display.to_string(index=False)
    )


def print_top_rules(rules, n=20):
    print("\n=== TOP ASSOCIATION RULES ===")

    if rules.empty:
        print("No association rules found.")
        return

    top = rules.head(n).copy()

    display = top[
        [
            "antecedent_product_name",
            "consequent_product_name",
            "pair_orders",
            "support",
            "confidence",
            "lift"
        ]
    ].copy()

    display["support"] = (
        display["support"] * 100
    ).round(4)

    display["confidence"] = (
        display["confidence"] * 100
    ).round(4)

    display["lift"] = (
        display["lift"]
    ).round(4)

    display = display.rename(
        columns={
            "antecedent_product_name": "If Product",
            "consequent_product_name": "Then Product",
            "pair_orders": "Pair Orders",
            "support": "Support %",
            "confidence": "Confidence %",
            "lift": "Lift"
        }
    )

    print(
        display.to_string(index=False)
    )


def create_pair_chart(pairs, n=15):
    if pairs.empty:
        return

    top = pairs.head(n).copy()

    labels = (
        top["product_a_name"].str[:25]
        + " + "
        + top["product_b_name"].str[:25]
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        range(len(top)),
        top["pair_orders"]
    )

    plt.yticks(
        range(len(top)),
        labels
    )

    plt.xlabel(
        "Number of Orders"
    )

    plt.ylabel(
        "Product Pair"
    )

    plt.title(
        "Top Product Pairs by Co-Occurrence"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    path = (
        OUTPUT_DIR /
        "top_product_pairs.png"
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


def create_rule_chart(rules, n=20):
    if rules.empty:
        return

    top = rules.head(n).copy()

    labels = (
        top["antecedent_product_name"].str[:20]
        + " → "
        + top["consequent_product_name"].str[:20]
    )

    plt.figure(figsize=(12, 8))

    plt.scatter(
        top["confidence"],
        top["lift"],
        s=50
    )

    for i, (_, row) in enumerate(top.iterrows()):
        plt.annotate(
            labels.iloc[i],
            (
                row["confidence"],
                row["lift"]
            ),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel(
        "Confidence"
    )

    plt.ylabel(
        "Lift"
    )

    plt.title(
        "Association Rules: Confidence vs Lift"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR /
        "association_rules_support_confidence.png"
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
    print("ASSOCIATION RULES / MARKET BASKET ANALYSIS")
    print("=" * 70)

    print(
        f"\nMinimum product orders: "
        f"{MIN_PRODUCT_ORDERS}"
    )

    print(
        f"Minimum pair orders: "
        f"{MIN_PAIR_ORDERS}"
    )

    print(
        f"Minimum lift: "
        f"{MIN_LIFT}"
    )

    connection = get_connection()

    try:
        transactions = load_transactions(
            connection
        )

        filtered_transactions = filter_products(
            transactions,
            MIN_PRODUCT_ORDERS
        )

        if filtered_transactions.empty:
            print(
                "\nNo transactions after filtering."
            )
            return

        baskets = create_baskets(
            filtered_transactions
        )

        if baskets.empty:
            print(
                "\nNo baskets with at least "
                "2 products."
            )
            return

        product_counts = calculate_product_counts(
            filtered_transactions
        )

        total_orders = (
            filtered_transactions["order_id"]
            .nunique()
        )

        print(
            f"\nTotal orders used for analysis: "
            f"{total_orders:,}"
        )

        pairs = calculate_pairs(
            baskets,
            MIN_PAIR_ORDERS
        )

        print(
            f"Product pairs found: "
            f"{len(pairs):,}"
        )

        if pairs.empty:
            print(
                "\nNo product pairs passed "
                "the minimum pair order threshold."
            )
            return

        pairs = calculate_metrics(
            pairs,
            product_counts,
            total_orders,
            MIN_LIFT
        )

        print(
            f"Product pairs after lift filtering: "
            f"{len(pairs):,}"
        )

        pairs = add_product_names(
            pairs,
            transactions
        )

        rules = create_directional_rules(
            pairs
        )

        summary = create_summary(
            pairs,
            rules
        )

        print_top_pairs(
            pairs,
            TOP_N
        )

        print_top_rules(
            rules,
            TOP_N
        )

        print("\n=== SUMMARY ===")

        if not summary.empty:
            print(
                summary.to_string(
                    index=False
                )
            )

        save_outputs(
            pairs,
            rules,
            summary
        )

        create_pair_chart(
            pairs,
            15
        )

        create_rule_chart(
            rules,
            20
        )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED")
        print("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()