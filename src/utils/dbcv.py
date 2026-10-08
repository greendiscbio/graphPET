"""Density-Based Clustering Validation.

Reference: https://doi.org/10.1137/1.9781611973440.96
Source: https://github.com/FelSiq/DBCV
"""

import functools
import itertools
import multiprocessing
import typing as t

import mpmath
import numpy as np
import numpy.typing as npt
import scipy.sparse.csgraph
import scipy.spatial.distance
import scipy.stats


def prim_mst(
    graph: npt.NDArray[np.float32], ind_root: int = 0
) -> npt.NDArray[np.float32]:
    n = len(graph)
    intree = np.full(n, fill_value=False)
    d = np.full(n, fill_value=np.inf)

    d[ind_root] = 0
    v = ind_root
    counter = 0

    G = {
        "MST_edges": {
            "node_inds": np.zeros((n - 1, 2), dtype=int),
            "weights": np.zeros(n - 1, dtype=float),
        },
        "MST_degrees": np.zeros(n, dtype=int),
        "MST_parent": np.arange(n),
    }

    while counter < n - 1:
        intree[v] = True
        dist = np.inf

        for w in np.arange(n):
            if w != v and not intree[w]:
                weight = graph[v, w]

                if d[w] > weight:
                    d[w] = weight
                    G["MST_parent"][w] = v

                if dist > d[w]:
                    dist = d[w]
                    next_v = w

        counter += 1
        G["MST_edges"]["node_inds"][counter - 1, :] = (G["MST_parent"][next_v], next_v)
        G["MST_edges"]["weights"][counter - 1] = graph[G["MST_parent"][next_v], next_v]
        G["MST_degrees"][G["MST_parent"][next_v]] += 1
        G["MST_degrees"][next_v] += 1
        v = next_v

    (inds_a, inds_b) = G["MST_edges"]["node_inds"].T
    weights = G["MST_edges"]["weights"]

    mst = np.zeros_like(graph)
    mst[inds_a, inds_b] = weights
    mst[inds_b, inds_a] = weights

    return mst


_MP = mpmath.mp.clone()


def compute_pair_to_pair_dists(
    X: npt.NDArray[np.float64], metric: str
) -> npt.NDArray[np.float64]:
    dists = scipy.spatial.distance.cdist(X, X, metric=metric)
    np.maximum(dists, 1e-12, out=dists)
    # Exclude each sample from its own neighborhood.
    np.fill_diagonal(dists, val=np.inf)
    return dists


def get_subarray(
    arr: npt.NDArray[np.float64],
    /,
    inds_a: t.Optional[npt.NDArray[np.int32]] = None,
    inds_b: t.Optional[npt.NDArray[np.int32]] = None,
) -> npt.NDArray[np.float64]:
    if inds_a is None:
        return arr
    if inds_b is None:
        inds_b = inds_a
    inds_a_mesh, inds_b_mesh = np.meshgrid(inds_a, inds_b)
    return arr[inds_a_mesh, inds_b_mesh].T


def get_internal_objects(
    mutual_reach_dists: npt.NDArray[np.float64], use_original_mst_implementation: bool
) -> npt.NDArray[np.float64]:
    if use_original_mst_implementation:
        mutual_reach_dists = np.copy(mutual_reach_dists)
        np.fill_diagonal(mutual_reach_dists, 0.0)
        mst = prim_mst(mutual_reach_dists)

    else:
        mst = scipy.sparse.csgraph.minimum_spanning_tree(mutual_reach_dists)
        mst = mst.toarray()
        mst += mst.T

    is_mst_edges = (mst > 0.0).astype(int, copy=False)

    internal_node_inds = is_mst_edges.sum(axis=0) > 1
    internal_node_inds = np.flatnonzero(internal_node_inds)

    internal_edge_weights = get_subarray(mst, inds_a=internal_node_inds)

    graph_has_internal_nodes = bool(internal_node_inds.size > 0)
    graph_has_at_least_two_internal_nodes = bool(internal_edge_weights.size > 1)

    return (
        (
            internal_node_inds
            if graph_has_internal_nodes
            else np.arange(mutual_reach_dists.shape[0])
        ),
        internal_edge_weights if graph_has_at_least_two_internal_nodes else mst,
    )


def compute_cluster_core_distance(
    dists: npt.NDArray[np.float64], d: int, enable_dynamic_precision: bool
) -> npt.NDArray[np.float64]:
    n, _ = dists.shape
    orig_dists_dtype = dists.dtype

    if enable_dynamic_precision:
        dists = np.asarray(_MP.matrix(dists), dtype=object).reshape(*dists.shape)

    core_dists = np.power(dists, -d).sum(axis=-1, keepdims=True) / (n - 1)

    if not enable_dynamic_precision:
        np.clip(core_dists, a_min=0.0, a_max=1e12, out=core_dists)

    np.power(core_dists, -1.0 / d, out=core_dists)

    if enable_dynamic_precision:
        core_dists = np.asarray(core_dists, dtype=orig_dists_dtype)

    return core_dists


def compute_mutual_reach_dists(
    dists: npt.NDArray[np.float64],
    d: float,
    enable_dynamic_precision: bool,
) -> npt.NDArray[np.float64]:
    core_dists = compute_cluster_core_distance(
        d=d, dists=dists, enable_dynamic_precision=enable_dynamic_precision
    )
    mutual_reach_dists = dists.copy()
    np.maximum(mutual_reach_dists, core_dists, out=mutual_reach_dists)
    np.maximum(mutual_reach_dists, core_dists.T, out=mutual_reach_dists)
    return (core_dists, mutual_reach_dists)


def fn_density_sparseness(
    cls_inds: npt.NDArray[np.int32],
    dists: npt.NDArray[np.float64],
    d: int,
    enable_dynamic_precision: bool,
    use_original_mst_implementation: bool,
) -> t.Tuple[float, npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    (core_dists, mutual_reach_dists) = compute_mutual_reach_dists(
        dists=dists, d=d, enable_dynamic_precision=enable_dynamic_precision
    )
    (internal_node_inds, internal_edge_weights) = get_internal_objects(
        mutual_reach_dists,
        use_original_mst_implementation=use_original_mst_implementation,
    )
    dsc = float(internal_edge_weights.max())
    internal_core_dists = core_dists[internal_node_inds]
    internal_node_inds = cls_inds[internal_node_inds]
    return (dsc, internal_core_dists, internal_node_inds)


def fn_density_separation(
    cls_i: int,
    cls_j: int,
    dists: npt.NDArray[np.float64],
    internal_core_dists_i: npt.NDArray[np.float64],
    internal_core_dists_j: npt.NDArray[np.float64],
) -> t.Tuple[int, int, float]:
    sep = dists.copy()
    np.maximum(sep, internal_core_dists_i, out=sep)
    np.maximum(sep, internal_core_dists_j.T, out=sep)
    dspc_ij = float(sep.min()) if sep.size else np.inf
    return (cls_i, cls_j, dspc_ij)


def _convert_singleton_clusters_to_noise(
    y: npt.NDArray[np.int32], noise_id: int
) -> npt.NDArray[np.int32]:
    """Cast clusters containing a single instance as noise."""
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)
    singleton_clusters = cluster_ids[cluster_sizes == 1]

    if singleton_clusters.size == 0:
        return y

    return np.where(np.isin(y, singleton_clusters), noise_id, y)


def dbcv(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int32],
    metric: str = "sqeuclidean",
    noise_id: int = -1,
    n_processes: t.Union[int, str] = "auto",
    enable_dynamic_precision: bool = True,
    bits_of_precision: int = 512,
    use_original_mst_implementation: bool = True,
) -> float:
    """Compute Density-Based Clustering Validation (DBCV).

    X contains sample features and y contains cluster labels. Noise and singleton
    clusters are excluded, but the score is weighted by the original sample size.
    The default metric is squared Euclidean distance. Automatic multiprocessing
    uses one process for up to 500 non-noise samples and four otherwise.

    Dynamic precision (512 bits) and the original Prim MST implementation are
    enabled by default. Disabling them uses floating-point arithmetic and SciPy's
    MST implementation, respectively, which may change the result.

    References:
        Moulavi et al., "Density-Based Clustering Validation" (2014).
        https://doi.org/10.1137/1.9781611973440.96
        https://github.com/pajaskowiak/dbcv/
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = np.asarray(y, dtype=int)

    n, d = X.shape  # Preserve the original sample size for score weighting.

    if n != y.size:
        raise ValueError(f"Mismatch in {X.shape[0]=} and {y.size=} dimensions.")

    y = _convert_singleton_clusters_to_noise(y, noise_id=noise_id)

    non_noise_inds = y != noise_id
    X = X[non_noise_inds, :]
    y = y[non_noise_inds]

    if y.size == 0:
        return 0.0

    y = scipy.stats.rankdata(y, method="dense") - 1
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)

    dists = compute_pair_to_pair_dists(X=X, metric=metric)

    dscs = np.zeros(cluster_ids.size, dtype=float)

    min_dspcs = np.full(cluster_ids.size, fill_value=np.inf)

    # Internal nodes have degree greater than one in the MST.
    internal_objects_per_cls: t.Dict[int, npt.NDArray[np.int32]] = {}

    internal_core_dists_per_cls: t.Dict[int, npt.NDArray[np.float32]] = {}

    cls_inds = [np.flatnonzero(y == cls_id) for cls_id in cluster_ids]

    if n_processes == "auto":
        n_processes = 4 if y.size > 500 else 1

    with _MP.workprec(bits_of_precision), multiprocessing.Pool(
        processes=min(n_processes, cluster_ids.size)
    ) as ppool:
        fn_density_sparseness_ = functools.partial(
            fn_density_sparseness,
            d=d,
            enable_dynamic_precision=enable_dynamic_precision,
            use_original_mst_implementation=use_original_mst_implementation,
        )

        args = [(cls_ind, get_subarray(dists, inds_a=cls_ind)) for cls_ind in cls_inds]

        for cls_id, (dsc, internal_core_dists, internal_node_inds) in enumerate(
            ppool.starmap(fn_density_sparseness_, args)
        ):
            internal_objects_per_cls[cls_id] = internal_node_inds
            internal_core_dists_per_cls[cls_id] = internal_core_dists
            dscs[cls_id] = dsc

    n_cls_pairs = (cluster_ids.size * (cluster_ids.size - 1)) // 2

    if n_cls_pairs > 0:
        with _MP.workprec(bits_of_precision), multiprocessing.Pool(
            processes=min(n_processes, n_cls_pairs)
        ) as ppool:
            args = [
                (
                    cls_i,
                    cls_j,
                    get_subarray(
                        dists,
                        inds_a=internal_objects_per_cls[cls_i],
                        inds_b=internal_objects_per_cls[cls_j],
                    ),
                    internal_core_dists_per_cls[cls_i],
                    internal_core_dists_per_cls[cls_j],
                )
                for cls_i, cls_j in itertools.combinations(cluster_ids, 2)
            ]

            for cls_i, cls_j, dspc_ij in ppool.starmap(fn_density_separation, args):
                min_dspcs[cls_i] = min(min_dspcs[cls_i], dspc_ij)
                min_dspcs[cls_j] = min(min_dspcs[cls_j], dspc_ij)

    np.nan_to_num(min_dspcs, copy=False, posinf=1e12)
    vcs = (min_dspcs - dscs) / (1e-12 + np.maximum(min_dspcs, dscs))
    np.nan_to_num(vcs, copy=False, nan=0.0)
    dbcv = float(np.sum(vcs * cluster_sizes)) / n

    return dbcv
