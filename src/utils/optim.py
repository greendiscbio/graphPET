"""Nested genetic algorithm and ACS clustering optimization."""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import List, Literal, Tuple

import antco_css as antco
import mlflow
import numpy as np
import pandas as pd
from numba import jit

from . import aco_config
from .ga import (
    GAConfig,
    Individual,
    apply_cached_solutions,
    average_hamming_diversity,
    cache_solutions,
    crossover,
    evaluate_individual,
    get_elite,
    increase_population_age,
    kmeans_population_init,
    mutate,
    random_population_init,
    tournament_selection,
)
from .graph import get_subgraph
from .io import create_temp_dir
from .logger import setup_logger

logger = setup_logger(__name__)


@jit(nopython=True, fastmath=True, cache=True)
def _euclidean_distance_matrix_sq(X: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distances with an infinite diagonal."""
    N, D = X.shape
    dist_matrix = np.empty((N, N), dtype=np.float64)

    for i in range(N):
        dist_matrix[i, i] = np.inf

    for i in range(N):
        xi = X[i]
        for j in range(i + 1, N):
            xj = X[j]
            sq_sum = 0.0
            for k in range(D):
                diff = xi[k] - xj[k]
                sq_sum += diff * diff
            dist_matrix[i, j] = sq_sum
            dist_matrix[j, i] = sq_sum

    return dist_matrix


@jit(nopython=True, fastmath=True, cache=True)
def _count_same_cluster_neighbors(
    dist_matrix: np.ndarray,
    cluster_assignments: np.ndarray,
    V: int,
    min_dist: float,
    n_clusters: int,
):
    """Count same-cluster neighbors within the squared distance threshold."""
    N = dist_matrix.shape[0]
    counts_per_cluster = np.zeros(n_clusters, dtype=np.int32)
    samples_per_cluster = np.zeros(n_clusters, dtype=np.int32)

    for i in range(N):
        distances = dist_matrix[i]
        neighbor_indices = np.argpartition(distances, V)[:V]

        cluster_id = cluster_assignments[i]
        samples_per_cluster[cluster_id] += 1

        for j in range(V):
            neighbor_id = neighbor_indices[j]
            if (cluster_assignments[neighbor_id] == cluster_id) and (
                distances[neighbor_id] < min_dist
            ):
                counts_per_cluster[cluster_id] += 1

    return samples_per_cluster, counts_per_cluster


@jit(nopython=True, fastmath=True, cache=True)
def _compute_centroids(X: np.ndarray, cluster_assignments: np.ndarray, n_clusters: int):
    """Compute cluster centroids from feature vectors."""
    D = X.shape[1]
    centroids = np.zeros((n_clusters, D), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int32)

    for i in range(X.shape[0]):
        cluster_id = cluster_assignments[i]
        for d in range(D):
            centroids[cluster_id, d] += X[i, d]
        counts[cluster_id] += 1

    for k in range(n_clusters):
        if counts[k] > 0:
            for d in range(D):
                centroids[k, d] /= counts[k]
    return centroids


@jit(nopython=True, fastmath=True, cache=True)
def _intra_cluster_distances(
    X: np.ndarray, C: np.ndarray, cluster_assignments: np.ndarray, n_clusters: int
):
    """Average distances to centroids equally across nonempty clusters."""
    N = X.shape[0]
    D = X.shape[1]

    sum_dist_per_cluster = np.zeros(n_clusters, dtype=np.float64)
    count_per_cluster = np.zeros(n_clusters, dtype=np.int64)

    for i in range(N):
        cluster_id = cluster_assignments[i]
        diff_sq = 0.0
        for d in range(D):
            diff = X[i, d] - C[cluster_id, d]
            diff_sq += diff * diff

        sum_dist_per_cluster[cluster_id] += np.sqrt(diff_sq)
        count_per_cluster[cluster_id] += 1

    macro_sum = 0.0
    valid_clusters = 0
    for k in range(n_clusters):
        if count_per_cluster[k] > 0:
            macro_sum += sum_dist_per_cluster[k] / count_per_cluster[k]
            valid_clusters += 1

    if valid_clusters == 0:
        return 0.0
    return macro_sum / valid_clusters


@jit(nopython=True, fastmath=True, cache=True)
def _inter_cluster_distances(centroids: np.ndarray):
    """Average pairwise Euclidean distances between cluster centroids."""
    n_clusters = centroids.shape[0]
    n_pairs = n_clusters * (n_clusters - 1) / 2
    if n_pairs == 0:
        return 0.0

    S = 0.0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist_sq = 0.0
            for d in range(centroids.shape[1]):
                diff = centroids[i, d] - centroids[j, d]
                dist_sq += diff * diff
            S += np.sqrt(dist_sq)

    return S / n_pairs


def loss_function_local_cohesion_euc(
    sel_nodes: np.ndarray,
    graph: antco.graph.GraphData,
    cluster_assignments: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    V: int,
    lambda_: float,
    r_matrix: np.ndarray,
    min_dist: float = 99.0,
) -> float:
    """Score local cohesion and geometric separation on induced graph edges.

    Labels must be contiguous integers starting at zero. The dense input tensor
    has shape (subjects, regions, regions); only induced graph edges are used.
    V sets the neighborhood size, min_dist limits neighbor distances, and
    lambda_ weights cohesion relative to geometric separation.
    """
    n_clusters = len(np.unique(cluster_assignments))

    induced_edges = antco.c_graph.get_induced_subgraph_edges(graph, sel_nodes)

    r_matrix_flatten = np.ascontiguousarray(
        r_matrix[:, edge_u[induced_edges], edge_v[induced_edges]], dtype=np.float64
    )

    dist_matrix_sq = _euclidean_distance_matrix_sq(r_matrix_flatten)

    # Compare the squared distances with a squared threshold.
    samples_per_cluster, counts_per_cluster = _count_same_cluster_neighbors(
        dist_matrix_sq, cluster_assignments, V, min_dist * min_dist, n_clusters
    )

    local_coherence = (counts_per_cluster / samples_per_cluster).mean() / V

    centroids = _compute_centroids(r_matrix_flatten, cluster_assignments, n_clusters)

    D = _intra_cluster_distances(
        r_matrix_flatten, centroids, cluster_assignments, n_clusters
    )
    S = _inter_cluster_distances(centroids)

    if (S + D) == 0:
        I_geom = 0.0
    else:
        I_geom = S / (S + D)

    return float(lambda_ * local_coherence + (1 - lambda_) * I_geom)


def optimize_solution(
    graph: antco.graph.GraphData,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    r_matrix: np.ndarray,
    clust_assignment: np.ndarray,
    n_nodes: int,
    cost_function: callable = loss_function_local_cohesion_euc,
    cost_function_kwargs: dict = {"V": 5, "min_dist": 99.0, "lambda_": 0.7},
) -> Tuple[list, float]:
    """Return the ACS-selected nodes and their best fitness."""

    antco_cost_fn = antco.loss.LossWithCache(
        cost_function,
        {
            **{
                "graph": graph,
                "cluster_assignments": clust_assignment,
                "edge_u": edge_u,
                "edge_v": edge_v,
                "r_matrix": r_matrix,
            },
            **cost_function_kwargs,
        },
    )

    config = aco_config.build_config(n_nodes)

    best_nodes, history = antco.algorithm.acs(
        config=config,
        loss_fn=antco_cost_fn,
        graph=graph,
        edge_u=edge_u,
        edge_v=edge_v,
        heuristic_information=None,
    )

    return np.sort(best_nodes).tolist(), float(history["L_best"].values[-1])


class OptimizeSolutionWrapper(object):
    def __init__(
        self,
        graph: antco.graph.GraphData,
        edge_u: np.ndarray,
        edge_v: np.ndarray,
        r_matrix: np.ndarray,
        n_nodes: list,
    ):
        self.graph = graph
        self.edge_u = edge_u
        self.edge_v = edge_v
        self.r_matrix = r_matrix
        self.n_nodes = n_nodes

    def __call__(self, solution: list) -> dict:
        optim_results = {}
        for n in self.n_nodes:
            aco_solution, aco_score = optimize_solution(
                graph=self.graph,
                edge_u=self.edge_u,
                edge_v=self.edge_v,
                r_matrix=self.r_matrix,
                clust_assignment=np.array(solution),
                n_nodes=n,
            )
            optim_results[n] = {"aco_solution": aco_solution, "aco_score": aco_score}

        return optim_results


def _cost_function_wrapper(args):
    """Evaluate an individual using arguments prepared for a worker process."""
    (
        individual,
        graph,
        edge_u,
        edge_v,
        penalty_factor,
        n_clusters,
        min_count_per_cat,
        r_matrix,
        n_nodes,
    ) = args

    cost_function_obj = OptimizeSolutionWrapper(
        graph=graph, edge_u=edge_u, edge_v=edge_v, r_matrix=r_matrix, n_nodes=n_nodes
    )

    return evaluate_individual(
        individual, cost_function_obj, penalty_factor, n_clusters, min_count_per_cat
    )


def basic_ga(
    config: GAConfig,
    population_init: Literal["random", "kmeans"],
    n_clusters: int,
    n_samples: int,
    adj_matrix: np.ndarray,
    r_matrix: np.ndarray,
    roi_names: List[str],
    n_nodes: List[int],
    hof_size: int = 10,
    min_samples_per_cluster: int = 0,
    seed: int = None,
) -> Tuple[Individual, List[Individual]]:
    """Run the GA and return the best individual and hall of fame."""

    logger.info(
        f"Starting GA execution for {n_clusters} clusters with a minimum of "
        f"{min_samples_per_cluster} samples per cluster."
    )

    mlflow.log_param("population_init", population_init)
    mlflow.log_param("population_size", config.population_size)
    mlflow.log_param("generations", config.generations)
    mlflow.log_param("crossover_rate", config.crossover_rate)
    mlflow.log_param("mutation_rate", config.mutation_rate)
    mlflow.log_param("max_num_mutations", config.max_num_mutations)
    mlflow.log_param("elite_size", config.elite_size)
    mlflow.log_param("tournament_size", config.tournament_size)
    mlflow.log_param("penalty_factor", config.penalty_factor)
    mlflow.log_param("n_nodes", n_nodes)
    mlflow.log_param("n_jobs", config.n_jobs)
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("n_samples", n_samples)
    mlflow.log_param("hof_size", hof_size)
    mlflow.log_param("min_samples_per_cluster", min_samples_per_cluster)
    mlflow.log_param("seed", seed)

    temp_dir = create_temp_dir()
    input_data_dir = temp_dir / "input"
    input_data_dir.mkdir(parents=True, exist_ok=True)
    np.save(input_data_dir / "r_matrix.npy", r_matrix)
    np.save(input_data_dir / "adj_matrix.npy", adj_matrix)
    with open(input_data_dir / "roi_header.txt", "w") as f:
        f.write("\n".join(roi_names))
    mlflow.log_artifact(input_data_dir)

    min_count_per_cat = {idx: min_samples_per_cluster for idx in range(n_clusters)}

    r_matrix = (r_matrix - r_matrix.mean(axis=0)) / r_matrix.std(axis=0)

    fitness_cache = {}
    if population_init == "random":
        population = random_population_init(
            population_size=config.population_size,
            n_categories=n_clusters,
            min_per_category=min_count_per_cat,
            chromosome_size=r_matrix.shape[0],
        )
    elif population_init == "kmeans":
        population = kmeans_population_init(
            clust_data=get_subgraph(np.arange(r_matrix.shape[1]), r_matrix, adj_matrix),
            pca_n_components=25,
            population_size=config.population_size,
            n_categories=n_clusters,
            min_per_category=min_count_per_cat,
            random_deviation_perc=0.25,
            keep_initial_perc=0.05,
            random_initial_perc=0.5,
            seed=seed,
        )
    else:
        raise ValueError(
            f'Supported population initialization strategies are: "random" or "kmeans". Provided: "{population_init}"'
        )

    graph, edge_u, edge_v = antco.graph.adjacency_to_graphdata(adj_matrix)

    logger.info("Initial evaluation of the solutions...")
    init_eval_counter = time.time()
    eval_args = [
        (
            ind,
            graph,
            edge_u,
            edge_v,
            config.penalty_factor,
            n_clusters,
            min_count_per_cat,
            r_matrix,
            n_nodes,
        )
        for ind in population
    ]
    if config.n_jobs == 1:
        population = [_cost_function_wrapper(args) for args in eval_args]
    else:
        with ProcessPoolExecutor(max_workers=config.n_jobs) as executor:
            population = list(executor.map(_cost_function_wrapper, eval_args))
    end_eval_counter = time.time()

    cache_solutions(population, fitness_cache)

    curr_fitness = [ind.fitness for ind in population]
    curr_ages = [ind.age for ind in population]
    logger.info(f"Best initial individual fitness: {max(curr_fitness):.3f}")

    mlflow.log_metric("optim/num_evaluations", len(curr_fitness), step=0)
    mlflow.log_metric("optim/best_fitness", max(curr_fitness), step=0)
    mlflow.log_metric("optim/mean_fitness", float(np.mean(curr_fitness)), step=0)
    mlflow.log_metric("optim/std_fitness", float(np.std(curr_fitness)), step=0)
    mlflow.log_metric(
        "optim/hamming_diversity", average_hamming_diversity(population), step=0
    )
    mlflow.log_metric("optim/max_population_age", max(curr_ages), step=0)
    mlflow.log_metric("optim/mean_population_age", float(np.mean(curr_ages)), step=0)
    mlflow.log_metric("optim/std_population_age", float(np.std(curr_ages)), step=0)
    mlflow.log_metric("optim/eval_time", (end_eval_counter - init_eval_counter), step=0)

    sorted_indices = np.argsort(curr_fitness)[::-1][:hof_size]
    hof_fitness = [curr_fitness[idx] for idx in sorted_indices]
    hof_individuals = [deepcopy(population[idx]) for idx in sorted_indices]

    best_individual = deepcopy(population[np.argmax(curr_fitness)])
    for generation in range(config.generations):

        new_population = (
            get_elite(population, config.elite_size) if config.elite_size > 0 else []
        )

        while len(new_population) < config.population_size:
            parent1 = tournament_selection(population, config.tournament_size)
            parent2 = tournament_selection(population, config.tournament_size)

            count, max_count = 0, 10
            while (parent1.chromosome == parent2.chromosome) and (count < max_count):
                parent1 = tournament_selection(population, config.tournament_size)
                count += 1

            child1, child2 = crossover(
                parent1, parent2, config.crossover_rate, n_clusters, min_count_per_cat
            )
            child1 = mutate(
                child1,
                config.mutation_rate,
                n_clusters,
                min_count_per_cat,
                max_num_mutations=config.max_num_mutations,
            )
            child2 = mutate(
                child2,
                config.mutation_rate,
                n_clusters,
                min_count_per_cat,
                max_num_mutations=config.max_num_mutations,
            )

            new_population.extend([child1, child2])

        population = new_population[: config.population_size]

        evaluated_population = [ind for ind in population if ind.is_evaluated]
        non_evaluated_population = [ind for ind in population if not ind.is_evaluated]

        apply_cached_solutions(non_evaluated_population, fitness_cache)
        evaluated_population_cache = [
            ind for ind in non_evaluated_population if ind.is_evaluated
        ]
        evaluated_population = evaluated_population + evaluated_population_cache
        non_evaluated_population = [
            ind for ind in non_evaluated_population if not ind.is_evaluated
        ]

        init_eval_counter = time.time()
        if len(non_evaluated_population) > 0:
            eval_args = [
                (
                    ind,
                    graph,
                    edge_u,
                    edge_v,
                    config.penalty_factor,
                    n_clusters,
                    min_count_per_cat,
                    r_matrix,
                    n_nodes,
                )
                for ind in non_evaluated_population
            ]
            if config.n_jobs == 1:
                new_evaluated_population = [
                    _cost_function_wrapper(args) for args in eval_args
                ]
            else:
                with ProcessPoolExecutor(max_workers=config.n_jobs) as executor:
                    new_evaluated_population = list(
                        executor.map(_cost_function_wrapper, eval_args)
                    )

            cache_solutions(new_evaluated_population, fitness_cache)

            population = evaluated_population + new_evaluated_population

        else:
            population = evaluated_population
        end_eval_counter = time.time()

        curr_fitness = [ind.fitness for ind in population]
        curr_ages = [ind.age for ind in population]

        increase_population_age(population)

        logger.info(
            f"Best individual fitness (generation {generation+1}): {max(curr_fitness):.3f}"
        )

        mlflow.log_metric(
            "optim/num_evaluations", len(non_evaluated_population), step=generation + 1
        )
        mlflow.log_metric("optim/best_fitness", max(curr_fitness), step=generation + 1)
        mlflow.log_metric(
            "optim/mean_fitness", float(np.mean(curr_fitness)), step=generation + 1
        )
        mlflow.log_metric(
            "optim/std_fitness", float(np.std(curr_fitness)), step=generation + 1
        )
        mlflow.log_metric(
            "optim/hamming_diversity",
            average_hamming_diversity(population),
            step=generation + 1,
        )
        mlflow.log_metric(
            "optim/max_population_age", max(curr_ages), step=generation + 1
        )
        mlflow.log_metric(
            "optim/mean_population_age", float(np.mean(curr_ages)), step=generation + 1
        )
        mlflow.log_metric(
            "optim/std_population_age", float(np.std(curr_ages)), step=generation + 1
        )
        mlflow.log_metric(
            "optim/eval_time",
            (end_eval_counter - init_eval_counter),
            step=generation + 1,
        )

        curr_fitness_ = curr_fitness + hof_fitness
        population_ = population + hof_individuals
        sorted_indices = np.argsort(curr_fitness_)[::-1][:hof_size]
        hof_fitness = [curr_fitness_[idx] for idx in sorted_indices]
        hof_individuals = [deepcopy(population_[idx]) for idx in sorted_indices]

        new_best = deepcopy(population[np.argmax(curr_fitness)])
        if new_best.fitness > best_individual.fitness:
            best_individual = new_best

    ga_output_dir = temp_dir / "ga_results"
    ga_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        ga_output_dir / "best_individual.npy",
        np.array(best_individual.chromosome, dtype=int),
    )

    for i, ind in enumerate(hof_individuals):
        np.save(ga_output_dir / f"hof_{i}.npy", np.array(ind.chromosome, dtype=int))

    pd.DataFrame(
        {
            "solution": ["best_individual"]
            + [f"hof_{i}" for i in range(len(hof_individuals))],
            "fitness": [best_individual.fitness] + hof_fitness,
        }
    ).to_csv(ga_output_dir / "fitness.csv")

    conn_networks = {"best_individual": {}}
    for n_nodes_sol, n_nodes_sol_vals in best_individual.optim_results.items():
        conn_networks["best_individual"][n_nodes_sol] = {
            "aco_solution": n_nodes_sol_vals["aco_solution"],
            "aco_score": n_nodes_sol_vals["aco_score"],
            "aco_solution_rois": [
                roi_names[int(idx)] for idx in n_nodes_sol_vals["aco_solution"]
            ],
        }

    for i, ind in enumerate(hof_individuals):
        conn_networks[f"hof_{i}"] = {}
        for n_nodes_sol, n_nodes_sol_vals in ind.optim_results.items():
            conn_networks[f"hof_{i}"][n_nodes_sol] = {
                "aco_solution": n_nodes_sol_vals["aco_solution"],
                "aco_score": n_nodes_sol_vals["aco_score"],
                "aco_solution_rois": [
                    roi_names[int(idx)] for idx in n_nodes_sol_vals["aco_solution"]
                ],
            }
    with open(ga_output_dir / "aco_optimization.json", "w") as f:
        json.dump(conn_networks, f)

    mlflow.log_artifact(ga_output_dir)

    return best_individual, hof_individuals
