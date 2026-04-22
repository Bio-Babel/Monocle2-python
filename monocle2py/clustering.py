"""Density peak clustering (Rodriguez & Laio 2014).

Ports the ``densityPeak`` branch of R's ``monocle::clusterCells``, which
delegates to ``densityClust::densityClust`` / ``findClusters``. We
reimplement the algorithm in pure NumPy because no maintained Python
equivalent exists. ``louvain`` and ``DDRTree`` clustering modes are not
yet ported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform

from ._uns import ensure_state

__all__ = ["cluster_cells"]


def _estimate_dc(
    D: np.ndarray,
    neighbor_rate_low: float = 0.01,
    neighbor_rate_high: float = 0.02,
    random_state: int = 2017,
) -> float:
    """Binary search for the distance cutoff giving ~1-2% average neighbor rate."""
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    dist_vals = D[iu]
    if n > 448:
        rng = np.random.default_rng(random_state)
        sample_size = 100128  # = C(448, 2), matching densityClust's subsample
        if len(dist_vals) > sample_size:
            dist_vals = rng.choice(dist_vals, size=sample_size, replace=False)
        size = 448
    else:
        size = n

    low = float(dist_vals.min())
    high = float(dist_vals.max())
    while True:
        dc = (low + high) / 2.0
        # Matches densityClust:::estimateDc:
        #   ((sum(distance < dc) * 2 + size) / size - 1) / size
        nbr = (((np.sum(dist_vals < dc) * 2 + size) / size) - 1) / size
        if neighbor_rate_low <= nbr <= neighbor_rate_high:
            return dc
        if nbr < neighbor_rate_low:
            low = dc
        else:
            high = dc


def _local_density(D: np.ndarray, dc: float, gaussian: bool = True) -> np.ndarray:
    """Per-point local density rho.

    Gaussian kernel: ``rho_i = sum_{j != i} exp(-(d_ij / dc)^2)``.
    Hard-cutoff kernel: ``rho_i = #{j != i : d_ij < dc}``.
    """
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    if gaussian:
        h = np.exp(-((D[iu] / dc) ** 2))
    else:
        h = (D[iu] < dc).astype(float)
    rho = np.zeros(n)
    np.add.at(rho, iu[0], h)
    np.add.at(rho, iu[1], h)
    return rho


def _distance_to_peak(
    D: np.ndarray, rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point distance to the nearest strictly-higher-density point.

    For points whose rho is the global maximum (no strictly higher neighbor),
    delta is set to the maximum distance to any other point. Returns
    ``(delta, nearest_higher_density_neighbor)``.
    """
    n = D.shape[0]
    delta = np.zeros(n)
    nearest = np.arange(n, dtype=np.int64)
    for i in range(n):
        higher = np.where(rho > rho[i])[0]
        if len(higher) == 0:
            j = int(np.argmax(D[i]))
            delta[i] = D[i, j]
            nearest[i] = j
        else:
            d_i = D[i, higher]
            j_local = int(np.argmin(d_i))
            delta[i] = d_i[j_local]
            nearest[i] = higher[j_local]
    return delta, nearest


def _assign_clusters(
    D: np.ndarray, rho: np.ndarray, peaks_idx: np.ndarray
) -> np.ndarray:
    """Propagate cluster labels from peaks to all points along density flow.

    Iterates points in order of decreasing rho. Peaks receive labels
    ``1..len(peaks_idx)`` in the order they appear in ``peaks_idx``; each
    non-peak inherits the cluster of its nearest strictly-higher-density
    neighbor. Returns 1-based labels with ``-1`` marking unassigned points
    (e.g. tied at the global maximum but not selected as a peak).
    """
    n = D.shape[0]
    cluster = np.full(n, -1, dtype=np.int64)
    for k, p in enumerate(peaks_idx, start=1):
        cluster[p] = k
    run_order = np.argsort(-rho, kind="stable")
    for i in run_order:
        if cluster[i] != -1:
            continue
        higher = np.where(rho > rho[i])[0]
        if len(higher) == 0:
            continue
        j = higher[int(np.argmin(D[i, higher]))]
        cluster[i] = cluster[j]
    return cluster


def _compute_halo(
    D: np.ndarray,
    rho: np.ndarray,
    cluster: np.ndarray,
    n_peaks: int,
    dc: float,
) -> np.ndarray:
    """Mark points whose density is below the boundary density of their cluster."""
    border = np.zeros(n_peaks + 1)
    for i in range(1, n_peaks + 1):
        in_mask = cluster == i
        out_mask = (cluster != i) & (cluster != -1)
        if not in_mask.any() or not out_mask.any():
            continue
        in_idx = np.where(in_mask)[0]
        out_idx = np.where(out_mask)[0]
        d_block = D[np.ix_(in_idx, out_idx)]
        border_pairs = d_block <= dc
        if border_pairs.any():
            avg_rho = (rho[in_idx][:, None] + rho[out_idx][None, :]) / 2.0
            border[i] = avg_rho[border_pairs].max()
    halo = np.zeros(D.shape[0], dtype=bool)
    valid = cluster != -1
    halo[valid] = rho[valid] < border[cluster[valid]]
    return halo


def cluster_cells(
    adata: AnnData,
    num_clusters: int | None = None,
    skip_rho_sigma: bool = False,
    rho_threshold: float | None = None,
    delta_threshold: float | None = None,
    peaks: np.ndarray | None = None,
    gaussian: bool = True,
    method: str = "densityPeak",
    random_state: int = 2017,
    verbose: bool = False,
) -> AnnData:
    """Cluster cells via density peaks (Rodriguez & Laio 2014).

    Operates on ``adata.obsm['X_dr']`` (typically the tSNE or DDRTree
    embedding) and writes per-cell ``Cluster``, ``peaks``, ``halo``,
    ``delta``, ``rho``, and ``nearest_higher_density_neighbor`` to
    ``adata.obs``. Caches the chosen ``dc`` and rho/delta thresholds in
    ``adata.uns['monocle2']['density_peak']``.

    Parameters
    ----------
    adata : anndata.AnnData
    num_clusters : int, optional
        If supplied, ``rho_threshold`` is set to ``0`` and ``delta_threshold``
        becomes the ``num_clusters``-th largest delta minus a small epsilon,
        yielding exactly ``num_clusters`` peaks.
    skip_rho_sigma : bool, default False
        Reuse the cached ``dc`` and the existing per-cell rho/delta columns
        from a previous call instead of recomputing them.
    rho_threshold, delta_threshold : float, optional
        Manual rho/delta cutoffs for selecting peaks. If both are ``None``
        and ``num_clusters`` is also ``None``, defaults to the 95th
        percentile of each.
    peaks : numpy.ndarray, optional
        Pre-selected peak indices (overrides thresholds and ``num_clusters``).
    gaussian : bool, default True
        Use a Gaussian kernel for local density. ``False`` uses the
        hard-cutoff kernel.
    method : str, default ``"densityPeak"``
        Only ``"densityPeak"`` is ported.
    random_state : int, default 2017
        Seed for sub-sampling the distance vector when estimating ``dc`` on
        large datasets (matches R's ``set.seed(2017)``).
    verbose : bool, default False
        Print progress messages.

    Returns
    -------
    anndata.AnnData
        Same AnnData with per-cell density-peak columns and state added.
    """
    if method != "densityPeak":
        raise NotImplementedError(
            f"cluster_cells method={method!r} is not yet ported. "
            "Only 'densityPeak' is supported."
        )

    if "X_dr" not in adata.obsm:
        raise RuntimeError(
            "adata.obsm['X_dr'] is missing. Run reduce_dimension first."
        )

    state = ensure_state(adata)
    dp_state = state.setdefault("density_peak", {})

    X = np.asarray(adata.obsm["X_dr"], dtype=float)
    D = squareform(pdist(X))

    if (
        skip_rho_sigma
        and "dc" in dp_state
        and {"rho", "delta", "nearest_higher_density_neighbor"}.issubset(
            adata.obs.columns
        )
    ):
        if verbose:
            print("Skipping rho/sigma computation; reusing cached values.")
        rho = adata.obs["rho"].to_numpy(dtype=float)
        delta = adata.obs["delta"].to_numpy(dtype=float)
        nearest = adata.obs["nearest_higher_density_neighbor"].to_numpy(
            dtype=np.int64
        )
        dc = float(dp_state["dc"])
    else:
        if verbose:
            print("Estimating distance cutoff dc...")
        dc = _estimate_dc(D, random_state=random_state)
        if verbose:
            print(f"Distance cutoff calculated to {dc}")
        rho = _local_density(D, dc, gaussian=gaussian)
        delta, nearest = _distance_to_peak(D, rho)

    if rho_threshold is not None and delta_threshold is not None:
        if verbose:
            print("Using user-supplied rho/delta thresholds.")
    elif num_clusters is None:
        rho_threshold = float(np.quantile(rho, 0.95))
        delta_threshold = float(np.quantile(delta, 0.95))
    else:
        rho_threshold = 0.0
        sorted_delta = np.sort(delta)[::-1]
        delta_threshold = float(
            sorted_delta[num_clusters - 1] - np.finfo(float).eps
        )

    if peaks is None:
        peaks_idx = np.where((rho > rho_threshold) & (delta > delta_threshold))[0]
    else:
        peaks_idx = np.asarray(peaks, dtype=np.int64)

    if len(peaks_idx) == 0:
        raise RuntimeError(
            "No density peaks selected; try lowering rho_threshold/delta_threshold."
        )

    cluster = _assign_clusters(D, rho, peaks_idx)
    n_peaks = len(peaks_idx)
    halo = _compute_halo(D, rho, cluster, n_peaks, dc)

    # Reorder peaks by gamma = rho * delta descending and remap cluster labels.
    gamma = rho[peaks_idx] * delta[peaks_idx]
    pk_order = np.argsort(-gamma, kind="stable")
    new_peaks_idx = peaks_idx[pk_order]
    inv_perm = np.full(n_peaks + 1, -1, dtype=np.int64)
    for new0, old0 in enumerate(pk_order):
        inv_perm[old0 + 1] = new0 + 1
    new_cluster = cluster.copy()
    valid = cluster != -1
    new_cluster[valid] = inv_perm[cluster[valid]]

    peaks_bool = np.zeros(adata.n_obs, dtype=bool)
    peaks_bool[new_peaks_idx] = True

    cluster_labels = np.where(
        new_cluster == -1, "NA", new_cluster.astype(str)
    )
    categories = [str(c) for c in range(1, n_peaks + 1)]
    if (new_cluster == -1).any():
        categories.append("NA")
    adata.obs["Cluster"] = pd.Categorical(cluster_labels, categories=categories)
    adata.obs["peaks"] = peaks_bool
    adata.obs["halo"] = halo
    adata.obs["delta"] = delta
    adata.obs["rho"] = rho
    adata.obs["nearest_higher_density_neighbor"] = nearest

    dp_state["dc"] = dc
    dp_state["threshold"] = {"rho": rho_threshold, "delta": delta_threshold}

    return adata
