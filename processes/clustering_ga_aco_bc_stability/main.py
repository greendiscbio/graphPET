"""GA-ACO-BC clustering on random subject subsamples."""

import argparse
from datetime import datetime

import mlflow
import numpy as np

import utils
from utils import clustering
from utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_selection_indices(n_samples: int, sel_prop: float, seed: int) -> np.ndarray:
    """Select a fixed proportion of subjects without replacement."""
    if not 0 < sel_prop <= 1:
        raise ValueError("sel_prop must be greater than 0 and at most 1.")

    np.random.seed(seed)
    n_selected = int(np.floor(sel_prop * n_samples))
    selection_mask = np.zeros(n_samples, dtype=bool)
    selection_mask[:n_selected] = True
    np.random.shuffle(selection_mask)
    return selection_mask


def main(args):
    """Optimize a subject subsample and log its evaluation to MLflow."""
    curr_time = datetime.now().strftime("%Y%m%d%H%M%S")
    args.groups = None
    conn_matrix, r_matrix, suvr, groups, roi_names = clustering.load_inputs(args)
    n_samples_total = r_matrix.shape[0]
    selection_mask = generate_selection_indices(
        n_samples=n_samples_total,
        sel_prop=args.sel_prop,
        seed=args.seed,
    )
    n_selected = int(selection_mask.sum())
    if n_selected < args.n_clusters * args.min_samples_per_cluster:
        raise ValueError("The subsample cannot satisfy the minimum cluster sizes.")
    if n_selected <= 30:
        raise ValueError("The subsample must contain more than 30 subjects for t-SNE.")

    r_matrix = r_matrix[selection_mask]
    suvr = suvr[selection_mask]
    groups = groups[selection_mask]
    logger.info(f"Number of samples used for clustering: {n_selected}")

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
        mlflow.log_param("sel_prop", args.sel_prop)
        mlflow.log_param("n_samples_total", n_samples_total)

        temp_dir = utils.io.create_temp_dir()
        subsample_dir = temp_dir / "subsample"
        subsample_dir.mkdir(parents=True, exist_ok=True)
        np.save(subsample_dir / "selection_mask.npy", selection_mask)
        np.save(subsample_dir / "selected_positions.npy", np.flatnonzero(selection_mask))
        np.save(
            subsample_dir / "non_selected_positions.npy", np.flatnonzero(~selection_mask)
        )
        mlflow.log_artifact(subsample_dir)

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
    parser = argparse.ArgumentParser(description="Run GA-ACO-BC subject subsampling.")

    parser.add_argument("--suvr_data", type=str, required=True)
    parser.add_argument("--conn_matrix", type=str, required=True)
    parser.add_argument("--r_matrix", type=str, required=True)
    parser.add_argument("--mlflow_experiment_name", type=str, required=True)
    parser.add_argument("--mlflow_run_name", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--sel_prop", type=float, required=True)

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

    # Subnetwork parameters
    parser.add_argument("--min_samples_per_cluster", type=int, default=5)
    parser.add_argument("--n_nodes", type=int, nargs="+", required=True)

    args = parser.parse_args()

    main(args)
