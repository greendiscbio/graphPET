"""Extract and reconstruct edge features on an undirected graph."""

import numpy as np


def get_subgraph(
    nodes: np.ndarray, r_matrix: np.ndarray, adj_matrix: np.ndarray
) -> np.ndarray:
    """Extract upper-triangle features for graph edges induced by selected nodes."""
    r_sub = r_matrix[:, nodes][:, :, nodes]
    a_sub = adj_matrix[np.ix_(nodes, nodes)]
    upper_mask = np.triu(a_sub, k=1).astype(bool)

    return r_sub[:, upper_mask]


def unflatten_graph(
    flat_array: np.ndarray, adj_matrix: np.ndarray, check_consistency: bool = True
) -> np.ndarray:
    """Reconstruct a symmetric graph from upper-triangle edge values."""
    restored_graph = np.zeros(adj_matrix.shape, dtype=float)
    mask = adj_matrix.astype(bool)
    idx = 0
    for i in range(mask.shape[0]):
        for j in range(i + 1, mask.shape[0]):
            if mask[i, j]:
                restored_graph[i, j] = flat_array[idx]
                restored_graph[j, i] = flat_array[idx]
                idx += 1

    if check_consistency:
        assert np.isclose(
            restored_graph.flatten().sum(),
            (restored_graph * adj_matrix).flatten().sum(),
        )

    return restored_graph
