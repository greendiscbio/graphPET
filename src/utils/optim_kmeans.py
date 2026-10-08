"""K-Means clustering with ACS subnetwork selection."""

import json
from typing import List, Tuple

import antco_css as antco
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import aco_config
from .io import create_temp_dir
from .logger import setup_logger

logger = setup_logger(__name__)


def optimize_solution(
    graph: antco.graph.GraphData,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    r_matrix: np.ndarray,
    n_clusters: int,
    n_nodes: int,
) -> Tuple[list, float, pd.DataFrame]:
    """Return the ACS-selected nodes, best silhouette score, and history."""

    class Objective(antco.loss.Loss):
        """Maximize the K-Means silhouette score on standardized induced edges."""

        def __init__(
            self,
            graph: antco.graph.GraphData,
            edge_u: np.ndarray,
            edge_v: np.ndarray,
            r_matrix: np.ndarray,
            n_clusters: int,
        ):

            self.graph = graph
            self.edge_u = edge_u
            self.edge_v = edge_v
            self.r_matrix = r_matrix
            self.n_clusters = n_clusters
            self.kmeans = KMeans(
                n_clusters=n_clusters, init="k-means++", n_init=25, random_state=42
            )

        def eval(self, solution: np.ndarray) -> float:

            induced_edges = antco.c_graph.get_induced_subgraph_edges(
                self.graph, solution
            )

            r_matrix_flatten = np.ascontiguousarray(
                self.r_matrix[
                    :, self.edge_u[induced_edges], self.edge_v[induced_edges]
                ],
                dtype=np.float64,
            )
            r_matrix_flatten_std = StandardScaler().fit_transform(r_matrix_flatten)
            self.kmeans.fit(r_matrix_flatten_std)
            labels = self.kmeans.predict(r_matrix_flatten_std)
            score = silhouette_score(r_matrix_flatten_std, labels, random_state=42)

            return score

    antco_cost_fn = Objective(
        graph=graph,
        edge_u=edge_u,
        edge_v=edge_v,
        r_matrix=r_matrix,
        n_clusters=n_clusters,
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
    solution = np.sort(best_nodes).tolist()

    return solution, float(history["L_best"].values[-1]), history


class OptimizeSolutionWrapper(object):
    def __init__(
        self,
        graph: antco.graph.GraphData,
        edge_u: np.ndarray,
        edge_v: np.ndarray,
        r_matrix: np.ndarray,
        n_nodes: list,
        n_clusters: int,
    ):
        self.graph = graph
        self.edge_u = edge_u
        self.edge_v = edge_v
        self.r_matrix = r_matrix
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters

    def __call__(self, *args, **kwargs) -> dict:
        optim_results = {}
        for n in self.n_nodes:
            aco_solution, aco_score, report = optimize_solution(
                graph=self.graph,
                edge_u=self.edge_u,
                edge_v=self.edge_v,
                r_matrix=self.r_matrix,
                n_clusters=self.n_clusters,
                n_nodes=n,
            )
            optim_results[n] = {
                "aco_solution": aco_solution,
                "aco_score": aco_score,
                "report": report,
            }

        return optim_results


def kmeans(
    n_clusters: int,
    adj_matrix: np.ndarray,
    r_matrix: np.ndarray,
    roi_names: List[str],
    n_nodes: List[int],
    **_,
) -> dict:
    """Select connected subnetworks and return their K-Means partitions."""
    logger.info(f"Starting K-means execution for {n_clusters} clusters.")

    temp_dir = create_temp_dir()
    input_data_dir = temp_dir / "input"
    input_data_dir.mkdir(parents=True, exist_ok=True)
    np.save(input_data_dir / "r_matrix.npy", r_matrix)
    np.save(input_data_dir / "adj_matrix.npy", adj_matrix)
    with open(input_data_dir / "roi_header.txt", "w") as f:
        f.write("\n".join(roi_names))
    mlflow.log_artifact(input_data_dir)

    graph, edge_u, edge_v = antco.graph.adjacency_to_graphdata(adj_matrix)

    optim_obj = OptimizeSolutionWrapper(
        graph=graph,
        edge_u=edge_u,
        edge_v=edge_v,
        r_matrix=r_matrix,
        n_nodes=n_nodes,
        n_clusters=n_clusters,
    )
    optim_sol = optim_obj()

    keys = list(optim_sol.keys())
    for k in keys:
        n_nodes_ = k
        n_nodes_stats = optim_sol[k]
        induced_edges = antco.c_graph.get_induced_subgraph_edges(
            graph, np.array(n_nodes_stats["aco_solution"])
        )
        subgraph = np.ascontiguousarray(
            r_matrix[:, edge_u[induced_edges], edge_v[induced_edges]], dtype=np.float64
        )
        subgraph = StandardScaler().fit_transform(subgraph)

        kmeans_obj = KMeans(
            n_clusters=n_clusters, init="k-means++", n_init=25, random_state=42
        ).fit(subgraph)

        labels = kmeans_obj.predict(subgraph)
        score = silhouette_score(subgraph, labels, random_state=42)
        mlflow.log_metric(f"optim/{n_nodes_}/silhouette_score", float(score))

        optim_sol[k]["kmeans_cluster_assignment"] = [int(l) for l in labels]

    for n_nodes_, n_nodes_stats in optim_sol.items():
        mean_cost = n_nodes_stats["report"]["iteration_l_mean"].values
        max_cost = n_nodes_stats["report"]["L_best"].values
        max_cost_iter = n_nodes_stats["report"]["iteration_l_max"].values
        iteration = n_nodes_stats["report"]["iteration"].values

        for it, mean_, max_, max_iter_ in zip(
            iteration, mean_cost, max_cost, max_cost_iter
        ):
            mlflow.log_metric(f"optim/{n_nodes_}/mean_cost", float(mean_), step=it)
            mlflow.log_metric(f"optim/{n_nodes_}/max_cost", float(max_), step=it)
            mlflow.log_metric(
                f"optim/{n_nodes_}/max_cost_iter", float(max_iter_), step=it
            )

    aco_output_dir = temp_dir / "aco_results"
    aco_output_dir.mkdir(parents=True, exist_ok=True)
    with open(aco_output_dir / "aco_optimization.json", "w") as f:
        json.dump(
            {
                n_nodes_: {
                    "aco_solution": n_nodes_stats["aco_solution"],
                    "aco_score": n_nodes_stats["aco_score"],
                    "kmeans_cluster_assignment": n_nodes_stats[
                        "kmeans_cluster_assignment"
                    ],
                    "aco_solution_rois": [
                        roi_names[int(idx)] for idx in n_nodes_stats["aco_solution"]
                    ],
                }
                for n_nodes_, n_nodes_stats in optim_sol.items()
            },
            f,
        )

    mlflow.log_artifact(aco_output_dir)

    return optim_sol
