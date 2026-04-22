"""Per-branch pseudotime plot: ``plot_multiple_branches_pseudotime``.

Walks each requested terminal state from the trajectory root, smooths the
expression of every gene along that path with a LOWESS curve, and overlays
the smoothed lines on a faceted, branch-coloured panel grid.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from ggplot2_py import (
    GGPlot,
    aes,
    expand_limits,
    facet_wrap,
    geom_line,
    ggplot,
    xlab,
    ylab,
)
from scipy.sparse import issparse
from statsmodels.nonparametric.smoothers_lowess import lowess

from .._uns import get_lower_detection_limit, get_state
from ._helpers import as_dense, feature_label_column
from ._theme import monocle_theme_opts

__all__ = ["plot_multiple_branches_pseudotime"]


def _branch_path_cells(
    adata: AnnData,
    branch_state: int,
) -> np.ndarray:
    """Return cell indices on the trajectory path from root to ``branch_state``.

    Uses the centroid MST stored by :func:`order_cells`. The root vertex is the
    centroid that has at least one cell with ``Pseudotime == 0``; the tip
    vertex is a leaf in the centroid MST that contains a cell of the requested
    state.
    """
    state = get_state(adata)
    if "ddrtree" not in state:
        raise RuntimeError(
            "DDRTree state missing — call reduce_dimension(method='DDRTree')."
        )
    ddr = state["ddrtree"]
    edges = np.asarray(ddr["mst_edges"], dtype=int)
    weights = np.asarray(ddr["mst_weights"], dtype=float)
    closest_vertex = np.asarray(ddr["closest_vertex"], dtype=int)
    n = int(np.asarray(ddr["K"]).shape[1])

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (u, v), w in zip(edges, weights):
        adj[int(u)].append((int(v), float(w)))
        adj[int(v)].append((int(u), float(w)))
    deg = np.array([len(a) for a in adj], dtype=int)

    pt = adata.obs["Pseudotime"].to_numpy(dtype=float)
    state_col = adata.obs["State"].astype(int).to_numpy()
    root_cells = np.flatnonzero(pt == 0.0)
    if root_cells.size == 0:
        raise RuntimeError(
            "No root cell (Pseudotime == 0) — call order_cells first."
        )
    root_state = int(state_col[root_cells[0]])
    root_centroids = np.unique(closest_vertex[state_col == root_state])
    root_leaves = root_centroids[deg[root_centroids] == 1]
    root_vertex = int(root_leaves[0]) if root_leaves.size else int(root_centroids[0])

    branch_centroids = np.unique(closest_vertex[state_col == int(branch_state)])
    if branch_centroids.size == 0:
        raise RuntimeError(f"No cells in State == {branch_state}.")
    branch_leaves = branch_centroids[deg[branch_centroids] == 1]
    tip_vertex = int(branch_leaves[0]) if branch_leaves.size else int(branch_centroids[0])

    parent = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    visited[root_vertex] = True
    queue = [root_vertex]
    while queue:
        u = queue.pop(0)
        if u == tip_vertex:
            break
        for v, _ in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)
    path_vertices: list[int] = []
    cur = tip_vertex
    while cur != -1:
        path_vertices.append(cur)
        cur = int(parent[cur])
    path_set = set(path_vertices)
    return np.flatnonzero(np.isin(closest_vertex, list(path_set)))


def plot_multiple_branches_pseudotime(
    adata: AnnData,
    branches: Sequence[int],
    branches_name: Sequence[str] | None = None,
    min_expr: float | None = None,
    cell_size: float = 0.75,
    norm_method: str = "log",
    nrow: int | None = None,
    ncol: int = 1,
    panel_order: Sequence[str] | None = None,
    color_by: str = "Branch",
    label_by_short_name: bool = True,
    TPM: bool = False,
    cores: int = 1,
) -> GGPlot:
    """Port of R's ``plot_multiple_branches_pseudotime``.

    For each requested terminal state in ``branches``, walks from the
    trajectory root to that branch's tip and smooths every gene's expression
    along the path with LOWESS. The branch-aligned smoothed curves share a
    common (rescaled) pseudotime axis so multiple branches can be compared on
    the same x-axis.
    """
    if "Pseudotime" not in adata.obs.columns or "State" not in adata.obs.columns:
        raise RuntimeError("Run order_cells before plot_multiple_branches_pseudotime.")
    states = adata.obs["State"].astype(int).to_numpy()
    states_set = set(states.tolist())
    for b in branches:
        if int(b) not in states_set:
            raise RuntimeError(
                f"Branch {b} is not a State value in the trajectory."
            )

    branch_label = list(branches) if branches_name is None else list(branches_name)
    if branches_name is not None and len(branches_name) != len(branches):
        raise ValueError("branches_name must match the length of branches.")

    if TPM:
        X = as_dense(adata.X).astype(float)
        col_sums = X.sum(axis=1)
        col_sums[col_sums == 0] = 1.0
        X = X / col_sums[:, None] * 1e6
        adata = adata.copy()
        adata.X = X

    cell_long_frames: list[pd.DataFrame] = []
    pt = adata.obs["Pseudotime"].to_numpy(dtype=float)

    for branch_in, label in zip(branches, branch_label):
        path_idx = _branch_path_cells(adata, int(branch_in))
        if path_idx.size == 0:
            continue
        sub = adata[path_idx].copy()
        sub_pt = sub.obs["Pseudotime"].to_numpy(dtype=float)
        order = np.argsort(sub_pt, kind="stable")
        ordered_cells = sub.obs_names.to_numpy()[order]
        X_ordered = as_dense(sub.X).astype(float)[order]
        smoothed = np.zeros_like(X_ordered)
        sorted_pt = sub_pt[order]
        for j in range(X_ordered.shape[1]):
            smoothed[:, j] = lowess(
                X_ordered[:, j], sorted_pt, frac=2.0 / 3.0, return_sorted=False,
            )

        gene_names = sub.var_names.astype(str).to_numpy()
        for j, gene in enumerate(gene_names):
            cell_long_frames.append(pd.DataFrame({
                "f_id": np.repeat(gene, ordered_cells.size),
                "Cell": ordered_cells,
                "expression": smoothed[:, j],
                "Branch": np.repeat(str(label), ordered_cells.size),
                "Pseudotime": sorted_pt,
            }))

    if not cell_long_frames:
        raise RuntimeError("No cells on any of the requested branches.")
    long_df = pd.concat(cell_long_frames, ignore_index=True)

    # R's plot_multiple_branches_pseudotime (plotting.R:2714-2715) has a dead
    # log2(tmp+1) line overwritten by the raw melt on the next line, so the
    # plotted data ends up as raw lowess-smoothed values; `norm_method='log'`
    # only affects the unused `m` matrix. Replicate that here: accept the
    # argument for API parity but do not apply it to the plot data.
    _ = norm_method  # intentionally unused; mirrors R's dead-code branch

    if min_expr is None:
        min_expr = get_lower_detection_limit(adata)

    labels = feature_label_column(adata.var, label_by_short_name)
    label_map = labels.to_dict()
    long_df["feature_label"] = long_df["f_id"].map(label_map).astype(str)
    if panel_order is not None:
        long_df["feature_label"] = pd.Categorical(
            long_df["feature_label"], categories=list(panel_order), ordered=True,
        )

    long_df = long_df.merge(
        adata.var.reset_index(drop=False).rename(columns={"index": "f_id"}),
        on="f_id", how="left", suffixes=("", "_var"),
    )

    rescaled = []
    for branch_val, sub in long_df.groupby("Branch", sort=False):
        sub = sub.copy()
        denom = sub["Pseudotime"].max() - sub["Pseudotime"].min()
        if denom > 0:
            sub["Pseudotime"] = (sub["Pseudotime"] - sub["Pseudotime"].min()) * 100.0 / denom
        rescaled.append(sub)
    long_df = pd.concat(rescaled, ignore_index=True)
    long_df["Branch"] = long_df["Branch"].astype("category")

    q = ggplot(long_df, aes(x="Pseudotime", y="expression"))
    if color_by is not None:
        q = q + geom_line(aes(color=color_by), linewidth=cell_size)
    else:
        q = q + geom_line(linewidth=cell_size)
    q = q + facet_wrap(
        "feature_label", nrow=nrow, ncol=ncol, scales="free_y",
    )
    q = q + ylab("Expression") + xlab("Pseudotime (stretched)")
    q = q + monocle_theme_opts() + expand_limits(y=min_expr)
    return q
