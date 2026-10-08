"""GA-ACO-BC clustering of individual metabolic connectivity."""

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
    ga_config = utils.ga.GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        n_jobs=args.n_jobs,
    )
    clustering.configure_tracking(args.mlflow_experiment_name, logger)

    with mlflow.start_run(
        tags={"git_commit": utils.variables.GIT_TAG},
        run_name=args.mlflow_run_name,
        nested=True,
    ):
        mlflow.log_param("execution_timestamp", curr_time)

        best_individual, _ = utils.optim.basic_ga(
            config=ga_config,
            population_init=args.population_init,
            n_clusters=args.n_clusters,
            n_samples=r_matrix.shape[0],
            n_nodes=args.n_nodes,
            adj_matrix=conn_matrix,
            r_matrix=r_matrix,
            roi_names=roi_names,
            hof_size=3,
            min_samples_per_cluster=args.min_samples_per_cluster,
            seed=args.seed,
        )
        chromosome = best_individual.chromosome
        optim_results = best_individual.optim_results

        temp_dir = utils.io.create_temp_dir()
        artifacts_dir = temp_dir / "stats"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        best_conn_solution_idx = None
        best_aco_score = -np.inf
        for num_nodes_sol in optim_results.values():
            if num_nodes_sol["aco_score"] > best_aco_score:
                best_aco_score = num_nodes_sol["aco_score"]
                best_conn_solution_idx = num_nodes_sol["aco_solution"]

        clustering.save_evaluation(
            suvr,
            groups,
            r_matrix,
            conn_matrix,
            best_conn_solution_idx,
            np.array(chromosome),
            artifacts_dir,
            logger,
        )
        mlflow.log_artifact(artifacts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA-ACO-BC clustering.")

    parser.add_argument("--suvr_data", type=str, required=True)
    parser.add_argument("--conn_matrix", type=str, required=True)
    parser.add_argument("--r_matrix", type=str, required=True)
    parser.add_argument("--mlflow_experiment_name", type=str, required=True)
    parser.add_argument("--mlflow_run_name", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, required=True)

    # Genetic algorithm parameters
    parser.add_argument(
        "--population_init",
        choices=["random", "kmeans"],
        required=False,
        default="kmeans",
    )
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2000)

    # Subnetwork and cohort parameters
    parser.add_argument("--min_samples_per_cluster", type=int, default=5)
    parser.add_argument("--n_nodes", type=int, nargs="+", required=True)
    parser.add_argument("--groups", type=str, nargs="+", required=False, default=None)

    args = parser.parse_args()

    main(args)
