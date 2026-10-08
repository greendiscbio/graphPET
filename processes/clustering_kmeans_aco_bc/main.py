"""KMeans-ACO-BC clustering of individual metabolic connectivity."""

import argparse
from datetime import datetime

import mlflow
import numpy as np

import utils
from utils import clustering
from utils.logger import setup_logger

logger = setup_logger(__name__)


def cluster_class_mapping(y_true: np.ndarray, y_pred: np.ndarray):
    """Align cluster labels with clinical labels by optimal assignment."""
    return clustering.cluster_class_mapping(y_true, y_pred)


def main(args):
    """Optimize connectivity clusters and log their evaluation to MLflow."""
    curr_time = datetime.now().strftime("%Y%m%d%H%M%S")
    conn_matrix, r_matrix, suvr, groups, roi_names = clustering.load_inputs(args)
    clustering.configure_tracking(args.mlflow_experiment_name, logger)

    with mlflow.start_run(
        tags={"git_commit": utils.variables.GIT_TAG},
        run_name=args.mlflow_run_name,
        nested=True,
    ):
        mlflow.log_param("execution_timestamp", curr_time)

        optim_results = utils.optim_kmeans.kmeans(
            n_clusters=args.n_clusters,
            adj_matrix=conn_matrix,
            r_matrix=r_matrix,
            roi_names=roi_names,
            n_nodes=args.n_nodes,
        )

        temp_dir = utils.io.create_temp_dir()
        artifacts_dir = temp_dir / "stats"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        best_conn_solution_idx = None
        best_clust_assignment = None
        best_aco_score = -np.inf
        for num_nodes_sol in optim_results.values():
            if num_nodes_sol["aco_score"] > best_aco_score:
                best_aco_score = num_nodes_sol["aco_score"]
                best_conn_solution_idx = num_nodes_sol["aco_solution"]
                best_clust_assignment = np.array(
                    num_nodes_sol["kmeans_cluster_assignment"]
                )

        clustering.save_evaluation(
            suvr,
            groups,
            r_matrix,
            conn_matrix,
            best_conn_solution_idx,
            best_clust_assignment,
            artifacts_dir,
            logger,
        )
        mlflow.log_artifact(artifacts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KMeans-ACO-BC clustering.")

    parser.add_argument("--suvr_data", type=str, required=True)
    parser.add_argument("--conn_matrix", type=str, required=True)
    parser.add_argument("--r_matrix", type=str, required=True)
    parser.add_argument("--mlflow_experiment_name", type=str, required=True)
    parser.add_argument("--mlflow_run_name", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--n_nodes", type=int, nargs="+", required=True)
    parser.add_argument("--groups", type=str, nargs="+", required=False, default=None)

    args = parser.parse_args()
    main(args)
