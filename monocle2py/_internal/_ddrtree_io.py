"""Adapters between ``ddrtree.DDRTreeResult`` and our ``adata.uns`` layout."""

from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .._uns import ensure_state

__all__ = [
    "store_ddrtree_result",
    "pairwise_distance_matrix",
    "mst_edges_from_coords",
]


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for rows of *points*."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def _sparse_to_edge_arrays(
    tree: csr_matrix, squared: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edge list from a (possibly symmetric) MST sparse matrix.

    When ``squared=True`` the input weights are squared Euclidean distances
    (as stored by ``ddrtree.DDRTree`` in its ``stree`` attribute); they are
    converted to true Euclidean distances via ``sqrt``. For ``scipy``-built
    MSTs on a Euclidean distance matrix the weights are already sqrt-ed and
    the default ``squared=False`` leaves them untouched.
    """
    coo = tree.tocoo()
    mask = coo.row < coo.col
    edges = np.column_stack([coo.row[mask], coo.col[mask]]).astype(np.int64)
    weights = np.asarray(coo.data[mask], dtype=np.float64)
    if squared:
        weights = np.sqrt(weights)
    return edges, weights


def mst_edges_from_coords(
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an MST on a point cloud and return ``(edges, weights)``.

    Parameters
    ----------
    coords : numpy.ndarray
        ``n_points x dim`` coordinate array.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``edges`` is ``(n_points - 1, 2)`` int64 pairs; ``weights`` is the
        Euclidean distance along each edge.
    """
    D = pairwise_distance_matrix(coords)
    tree = minimum_spanning_tree(D)
    return _sparse_to_edge_arrays(tree)


def _closest_vertex(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Index of the closest Y node for each column of Z (both ``dim x *``)."""
    # Expand to cells x dim and vertices x dim for cdist-style comparison.
    Zt = Z.T
    Yt = Y.T
    diff = Zt[:, None, :] - Yt[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return np.argmin(d2, axis=1)


def store_ddrtree_result(
    adata: AnnData, result: Any, gene_mask: np.ndarray | None = None
) -> None:
    """Write a ``ddrtree.DDRTreeResult`` into ``adata.obsm`` and ``uns``.

    Parameters
    ----------
    adata : anndata.AnnData
    result : ddrtree.DDRTreeResult
        The object returned by ``ddrtree.DDRTree``.
    gene_mask : numpy.ndarray, optional
        Boolean mask over ``adata.var_names`` recording which genes were
        kept after the row-SD and finite-value filters. Stored alongside
        ``W`` so downstream code can map back to the full gene space.
    """
    Z = np.asarray(result.Z, dtype=np.float64)
    Y = np.asarray(result.Y, dtype=np.float64)
    W = np.asarray(result.W, dtype=np.float64)

    adata.obsm["X_dr"] = Z.T  # cells x dim

    # ``ddrtree.DDRTreeResult.stree`` stores squared Euclidean distances
    # (see the essentials doc §2.3). Take sqrt so downstream consumers
    # like ``ordering._tree_diameter`` see real Euclidean lengths.
    mst_edges, mst_weights = _sparse_to_edge_arrays(result.stree, squared=True)

    ddrtree_state = {
        "K": Y,
        "W": W,
        "mst_edges": mst_edges,
        "mst_weights": mst_weights,
        "closest_vertex": _closest_vertex(Z, Y),
        "objective_vals": np.asarray(result.objective_vals, dtype=np.float64),
    }
    if gene_mask is not None:
        ddrtree_state["gene_mask"] = np.asarray(gene_mask, dtype=bool)

    state = ensure_state(adata)
    state["ddrtree"] = ddrtree_state
    state["dim_reduce_type"] = "DDRTree"
