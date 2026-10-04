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


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analytics" / "ml" / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42
N_CLUSTERS = 2


def load_data():
    scaled_path = OUTPUT_DIR / "customer_rfm_scaled.csv"
    rfm_path = OUTPUT_DIR / "customer_rfm.csv"

    if not scaled_path.exists():
        raise FileNotFoundError(
            f"File not found: {scaled_path}"
        )

    if not rfm_path.exists():
        raise FileNotFoundError(
            f"File not found: {rfm_path}"
        )

    scaled_df = pd.read_csv(
        scaled_path
    )

    rfm_df = pd.read_csv(
        rfm_path
    )

    feature_columns = [
        column
        for column in scaled_df.columns
        if column.startswith("scaled_")
    ]

    if not feature_columns:
        raise ValueError(
            "No scaled features found."
        )

    X = (
        scaled_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    rfm_df = rfm_df[
        rfm_df["customer_id"].isin(
            scaled_df["customer_id"]
        )
    ].copy()

    rfm_df = (
        scaled_df[
            ["customer_id"]
        ]
        .merge(
            rfm_df,
            on="customer_id",
            how="left"
        )
    )

    print(
        f"Loaded {len(scaled_df):,} customers."
    )

    print(
        f"ML features: {len(feature_columns)}"
    )

    print(
        "Features:"
    )

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    return (
        rfm_df,
        X,
        feature_columns
    )


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

    cluster_names = {
        0: "Occasional / Low-Value Customers",
        1: "Loyal High-Value Customers"
    }

    output["segment"] = (
        output["cluster"]
        .map(cluster_names)
    )

    output_path = (
        OUTPUT_DIR /
        "customer_segments.csv"
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
            customers=(
                "customer_id",
                "count"
            ),
            avg_recency=(
                "recency",
                "mean"
            ),
            avg_frequency=(
                "frequency",
                "mean"
            ),
            avg_monetary=(
                "monetary",
                "mean"
            ),
            avg_profit=(
                "profit",
                "mean"
            ),
            avg_units_bought=(
                "units_bought",
                "mean"
            ),
            avg_order_value=(
                "average_order_value",
                "mean"
            ),
            avg_profit_margin=(
                "profit_margin",
                "mean"
            )
        )
    )

    profile["percentage"] = (
        profile["customers"]
        / len(segments)
        * 100
    )

    profile = profile[
        [
            "cluster",
            "segment",
            "customers",
            "percentage",
            "avg_recency",
            "avg_frequency",
            "avg_monetary",
            "avg_profit",
            "avg_units_bought",
            "avg_order_value",
            "avg_profit_margin"
        ]
    ]

    profile = profile.sort_values(
        "cluster"
    )

    output_path = (
        OUTPUT_DIR /
        "customer_segment_profile.csv"
    )

    profile.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        "\n=== CUSTOMER SEGMENTS ==="
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
        columns=feature_columns
    )

    centers.insert(
        0,
        "cluster",
        range(N_CLUSTERS)
    )

    output_path = (
        OUTPUT_DIR /
        "kmeans_cluster_centers.csv"
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

    return (
        pca_df,
        pca.explained_variance_ratio_
    )


def save_pca_output(
    pca_df
):
    output_path = (
        OUTPUT_DIR /
        "customer_segments_pca.csv"
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

    cluster_names = {
        0: "Occasional / Low-Value Customers",
        1: "Loyal High-Value Customers"
    }

    for cluster in sorted(
        pca_df["cluster"].unique()
    ):
        subset = pca_df[
            pca_df["cluster"] == cluster
        ]

        plt.scatter(
            subset["pca_1"],
            subset["pca_2"],
            s=10,
            alpha=0.5,
            label=cluster_names.get(
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
        "Customer Segmentation - K-Means"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "customer_clusters_pca.png"
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
        segments["segment"]
        .value_counts()
    )

    plt.figure(
        figsize=(10, 6)
    )

    counts.plot(
        kind="bar"
    )

    plt.title(
        "Customer Segment Sizes"
    )

    plt.xlabel(
        "Customer Segment"
    )

    plt.ylabel(
        "Number of Customers"
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
        "customer_segment_sizes.png"
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
        profile["avg_monetary"]
    )

    plt.title(
        "Average Customer Revenue by Segment"
    )

    plt.xlabel(
        "Customer Segment"
    )

    plt.ylabel(
        "Average Revenue"
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
        "customer_segment_value.png"
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
        "Loading customer ML data..."
    )

    (
        df,
        X,
        feature_columns
    ) = load_data()

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
        "\nCustomer segmentation completed."
    )


if __name__ == "__main__":
    main()