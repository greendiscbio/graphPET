"""Shared inputs, tracking, and evaluation for connectivity clustering."""

import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import variables
from .graph import get_subgraph
from .io import load_suvr_data


def load_inputs(args):
    """Load the numeric graph, dense connectivity tensor, SUVRs, and labels."""
    conn_matrix = pd.read_parquet(
        variables.GRAPHICAL_LASSO_FOLDER / args.conn_matrix
    ).values
    r_matrix = np.load(variables.INDV_CONN_MATRIX_FOLDER / args.r_matrix)
    _, suvr, groups = load_suvr_data(args.suvr_data, standarize=False)

    if args.groups is not None:
        selection_mask = np.array([group in args.groups for group in groups])
        suvr = np.ascontiguousarray(suvr[selection_mask])
        groups = np.ascontiguousarray(groups[selection_mask])
        r_matrix = np.ascontiguousarray(r_matrix[selection_mask])
        print(
            f"Selecting groups {args.groups} (N={suvr.shape[0]}): "
            f"{np.unique(groups, return_counts=True)}"
        )

    suvr = (suvr - np.mean(suvr, axis=0)) / np.std(suvr, axis=0, ddof=1)
    groups[groups == "controls-adni"] = "controls"

    header_file = variables.INDV_CONN_MATRIX_FOLDER / args.r_matrix.replace(
        ".npy", "_header.txt"
    )
    with open(header_file) as file:
        roi_names = file.read().split("\t")

    return conn_matrix, r_matrix, suvr, groups, roi_names


def configure_tracking(experiment_name: str, logger):
    """Select the MLflow tracking location and experiment."""
    tracking_uri = os.getenv("MLFLOW_URI")
    if tracking_uri:
        logger.info(f"Setting up MLflow with {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        logger.warning("MLFLOW_URI not set; using the default MLflow location")
    mlflow.set_experiment(experiment_name)


def cluster_class_mapping(y_true: np.ndarray, y_pred: np.ndarray):
    """Align cluster labels with clinical labels by optimal assignment."""
    true_encoder = LabelEncoder()
    pred_encoder = LabelEncoder()
    y_true_int = true_encoder.fit_transform(y_true)
    y_pred_int = pred_encoder.fit_transform(y_pred)
    confusion = metrics.confusion_matrix(y_true_int, y_pred_int)
    row_indices, column_indices = linear_sum_assignment(-confusion)
    mapping = {col: row for row, col in zip(row_indices, column_indices)}
    mapped_pred_int = np.array([mapping[label] for label in y_pred_int])
    return true_encoder.inverse_transform(mapped_pred_int)


def _plot_embedding(embedding, labels, title, path, cluster_labels=False):
    fig, ax = plt.subplots(figsize=(6, 5))
    for index, group in enumerate(np.unique(labels)):
        ax.scatter(
            embedding[labels == group, 0],
            embedding[labels == group, 1],
            label=f"Cluster {group}" if cluster_labels else group,
            c=variables.COLORS[index],
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.xlabel("t-SNE 1", size=12)
    plt.ylabel("t-SNE 2", size=12)
    plt.title(title, size=15, pad=15)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def _log_metrics(functions, first, second, path):
    rows = [
        {"metric": name, "value": float(function(first, second))}
        for name, function in functions
    ]
    pd.DataFrame(rows).round(decimals=3).to_csv(path)
    for row in rows:
        mlflow.log_metric(f'eval/{row["metric"]}', row["value"])


def save_evaluation(
    suvr, groups, r_matrix, conn_matrix, nodes, labels, artifacts_dir: Path, logger
):
    """Save projections, clustering metrics, and aligned classification metrics."""
    roi_embedding = TSNE(n_components=2, random_state=2025).fit_transform(
        suvr[:, nodes]
    )
    _plot_embedding(
        roi_embedding,
        groups,
        "t-SNE projection (ROI-level)",
        artifacts_dir / "tsne_rois_by_group.png",
    )

    features = get_subgraph(nodes, r_matrix, conn_matrix)
    features = StandardScaler().fit_transform(features)
    edge_embedding = TSNE(n_components=2, random_state=2025).fit_transform(features)
    _plot_embedding(
        edge_embedding,
        groups,
        "t-SNE projection ground truth (edge-level)",
        artifacts_dir / "tsne_edges_by_group.png",
    )
    _plot_embedding(
        edge_embedding,
        labels,
        "t-SNE projection cluster (edge-level)",
        artifacts_dir / "tsne_edges_by_cluster.png",
        cluster_labels=True,
    )

    try:
        _log_metrics(
            [
                ("ari", metrics.adjusted_rand_score),
                ("nmi", metrics.normalized_mutual_info_score),
                ("homogeneity", metrics.homogeneity_score),
                ("completeness", metrics.completeness_score),
                ("v_measure", metrics.v_measure_score),
                ("fmi", metrics.fowlkes_mallows_score),
            ],
            groups,
            labels,
            artifacts_dir / "external_clustering_metrics.csv",
        )
    except Exception as error:
        logger.error(
            f"Error saving external_clustering_metrics.csv: {type(error)}. {error}"
        )

    try:
        _log_metrics(
            [
                ("silhouette", metrics.silhouette_score),
                ("davies_bouldin", metrics.davies_bouldin_score),
            ],
            features,
            labels,
            artifacts_dir / "internal_clustering_metrics.csv",
        )
    except Exception as error:
        logger.error(
            f"Error saving internal_clustering_metrics.csv: {type(error)}. {error}"
        )

    try:
        mapped_preds = cluster_class_mapping(groups, labels)
        _log_metrics(
            [
                ("accuracy", metrics.accuracy_score),
                ("macro_f1", partial(metrics.f1_score, average="macro")),
                ("mcc", metrics.matthews_corrcoef),
                ("macro_recall", partial(metrics.recall_score, average="macro")),
                ("macro_precision", partial(metrics.precision_score, average="macro")),
            ],
            groups,
            mapped_preds,
            artifacts_dir / "classification_metrics.csv",
        )
    except Exception as error:
        logger.error(f"Error saving classification_metrics.csv: {type(error)}. {error}")

    try:
        cm_labels = np.unique(groups).tolist()
        confusion = metrics.confusion_matrix(groups, mapped_preds, labels=cm_labels)
        plt.figure(figsize=(4, 4))
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=cm_labels,
            yticklabels=cm_labels,
            cbar=False,
        )
        plt.xlabel("Predicted labels", size=12, labelpad=15)
        plt.ylabel("True labels", size=12, labelpad=15)
        plt.title("Confusion matrix", size=15, pad=25)
        plt.tight_layout()
        plt.savefig(
            artifacts_dir / "confusion_matrix.png", bbox_inches="tight", dpi=150
        )
        plt.close()
    except Exception as error:
        logger.error(f"Error saving confusion_matrix.png: {type(error)}. {error}")
