from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analytics" / "ml" / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42
N_CLUSTERS = 4


CLUSTER_NAMES = {
    0: "High-Margin Products",
    1: "Low-Performance Products",
    2: "High-Value Products",
    3: "High-Selling / Low-Margin Products"
}


def load_data():
    query = """
        SELECT
            product_id,
            product_name,
            category_id,
            category_name,
            supplier_id,
            supplier_name,
            units_sold,
            orders,
            revenue,
            cost,
            profit,
            profit_margin_percent
        FROM warehouse.vw_product_performance
        ORDER BY product_id;
    """

    from sqlalchemy import create_engine
    import os
    from dotenv import load_dotenv

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    connection_url = (
        f"postgresql+psycopg2://"
        f"{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:"
        f"{os.getenv('DB_PORT')}/"
        f"{os.getenv('DB_NAME')}"
    )

    engine = create_engine(
        connection_url
    )

    try:
        df = pd.read_sql(
            query,
            engine
        )
    finally:
        engine.dispose()

    if df.empty:
        raise ValueError(
            "No product data found."
        )

    print(
        f"Loaded {len(df):,} products."
    )

    return df


def prepare_features(df):
    feature_columns = [
        "units_sold",
        "orders",
        "revenue",
        "cost",
        "profit",
        "profit_margin_percent"
    ]

    features = df[
        feature_columns
    ].copy()

    features = (
        features
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    positive_columns = [
        "units_sold",
        "orders",
        "revenue",
        "cost",
        "profit"
    ]

    for column in positive_columns:
        features[column] = np.log1p(
            features[column].clip(
                lower=0
            )
        )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        features
    )

    scaled_columns = [
        f"scaled_{column}"
        for column in feature_columns
    ]

    scaled_df = pd.DataFrame(
        X,
        columns=scaled_columns
    )

    scaled_df.insert(
        0,
        "product_id",
        df["product_id"].values
    )

    scaled_path = (
        OUTPUT_DIR /
        "product_features_scaled.csv"
    )

    scaled_df.to_csv(
        scaled_path,
        index=False
    )

    print(
        f"Saved: {scaled_path}"
    )

    print(
        "\nML features:"
    )

    for column in feature_columns:
        print(
            f"  - {column}"
        )

    return X, feature_columns


def fit_kmeans(X):
    print(
        "\n=== K-MEANS ==="
    )

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = model.fit_predict(
        X
    )

    silhouette = silhouette_score(
        X,
        labels
    )

    calinski = calinski_harabasz_score(
        X,
        labels
    )

    davies = davies_bouldin_score(
        X,
        labels
    )

    print(
        f"Clusters: {N_CLUSTERS}"
    )

    print(
        f"Inertia: {model.inertia_:,.2f}"
    )

    print(
        f"Silhouette Score: {silhouette:.4f}"
    )

    print(
        f"Calinski-Harabasz Score: {calinski:,.2f}"
    )

    print(
        f"Davies-Bouldin Score: {davies:.4f}"
    )

    return model, labels


def save_segments(
    df,
    labels
):
    output = df.copy()

    output["cluster"] = labels

    output["segment"] = (
        output["cluster"]
        .map(CLUSTER_NAMES)
    )

    output_path = (
        OUTPUT_DIR /
        "product_segments.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )

    return output


def create_segment_profile(
    segments
):
    profile = (
        segments
        .groupby(
            ["cluster", "segment"],
            as_index=False
        )
        .agg(
            products=(
                "product_id",
                "count"
            ),
            avg_units_sold=(
                "units_sold",
                "mean"
            ),
            avg_orders=(
                "orders",
                "mean"
            ),
            avg_revenue=(
                "revenue",
                "mean"
            ),
            avg_cost=(
                "cost",
                "mean"
            ),
            avg_profit=(
                "profit",
                "mean"
            ),
            avg_profit_margin=(
                "profit_margin_percent",
                "mean"
            ),
            total_revenue=(
                "revenue",
                "sum"
            ),
            total_profit=(
                "profit",
                "sum"
            )
        )
    )

    profile["percentage"] = (
        profile["products"]
        / len(segments)
        * 100
    )

    profile = profile[
        [
            "cluster",
            "segment",
            "products",
            "percentage",
            "avg_units_sold",
            "avg_orders",
            "avg_revenue",
            "avg_cost",
            "avg_profit",
            "avg_profit_margin",
            "total_revenue",
            "total_profit"
        ]
    ]

    profile = profile.sort_values(
        "cluster"
    )

    output_path = (
        OUTPUT_DIR /
        "product_segment_profile.csv"
    )

    profile.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        "\n=== PRODUCT SEGMENTS ==="
    )

    print(
        profile.to_string(
            index=False
        )
    )

    return profile


def save_cluster_centers(
    model,
    feature_columns
):
    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=[
            f"scaled_{column}"
            for column in feature_columns
        ]
    )

    centers.insert(
        0,
        "cluster",
        range(N_CLUSTERS)
    )

    centers["segment"] = (
        centers["cluster"]
        .map(CLUSTER_NAMES)
    )

    output_path = (
        OUTPUT_DIR /
        "product_cluster_centers.csv"
    )

    centers.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


def create_pca(
    X,
    labels
):
    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE
    )

    components = pca.fit_transform(
        X
    )

    pca_df = pd.DataFrame(
        {
            "pca_1": components[:, 0],
            "pca_2": components[:, 1],
            "cluster": labels
        }
    )

    pca_df["segment"] = (
        pca_df["cluster"]
        .map(CLUSTER_NAMES)
    )

    return (
        pca_df,
        pca.explained_variance_ratio_
    )


def save_pca_output(
    pca_df
):
    output_path = (
        OUTPUT_DIR /
        "product_segments_pca.csv"
    )

    pca_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


def plot_clusters(
    pca_df,
    explained_variance
):
    plt.figure(
        figsize=(12, 8)
    )

    for cluster in sorted(
        pca_df["cluster"].unique()
    ):
        subset = pca_df[
            pca_df["cluster"] == cluster
        ]

        plt.scatter(
            subset["pca_1"],
            subset["pca_2"],
            s=20,
            alpha=0.6,
            label=CLUSTER_NAMES.get(
                cluster,
                f"Cluster {cluster}"
            )
        )

    plt.xlabel(
        f"PC1 ({explained_variance[0] * 100:.2f}%)"
    )

    plt.ylabel(
        f"PC2 ({explained_variance[1] * 100:.2f}%)"
    )

    plt.title(
        "Product Segmentation - K-Means"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "product_clusters_pca.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_cluster_sizes(
    segments
):
    counts = (
        segments
        .groupby(
            "segment"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    counts.plot(
        kind="bar"
    )

    plt.title(
        "Product Segment Sizes"
    )

    plt.xlabel(
        "Product Segment"
    )

    plt.ylabel(
        "Number of Products"
    )

    plt.xticks(
        rotation=15,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "product_segment_sizes.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_segment_value(
    profile
):
    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        profile["segment"],
        profile["total_revenue"]
    )

    plt.title(
        "Total Revenue by Product Segment"
    )

    plt.xlabel(
        "Product Segment"
    )

    plt.ylabel(
        "Total Revenue"
    )

    plt.xticks(
        rotation=15,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "product_segment_value.png"
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
        "Loading product ML data..."
    )

    df = load_data()

    X, feature_columns = prepare_features(
        df
    )

    model, labels = fit_kmeans(
        X
    )

    segments = save_segments(
        df,
        labels
    )

    profile = create_segment_profile(
        segments
    )

    save_cluster_centers(
        model,
        feature_columns
    )

    pca_df, explained_variance = create_pca(
        X,
        labels
    )

    save_pca_output(
        pca_df
    )

    plot_clusters(
        pca_df,
        explained_variance
    )

    plot_cluster_sizes(
        segments
    )

    plot_segment_value(
        profile
    )

    print(
        "\nProduct segmentation completed."
    )


if __name__ == "__main__":
    main()

