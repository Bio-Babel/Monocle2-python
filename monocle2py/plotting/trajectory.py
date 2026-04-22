"""Trajectory views: ``plot_cell_trajectory`` and ``plot_complex_cell_trajectory``.

Both functions draw cells on top of the principal-graph backbone learnt by
:func:`~monocle2py.reduce_dimension`. ``plot_cell_trajectory`` keeps the cells
in their DDRTree coordinates (`adata.obsm['X_dr']`), while
``plot_complex_cell_trajectory`` lays the principal graph out as a Reingold–
Tilford-style tree rooted at a user-chosen state.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from ggplot2_py import (
    GGPlot,
    aes,
    element_blank,
    element_rect,
    geom_jitter,
    geom_point,
    geom_segment,
    geom_text,
    ggplot,
    scale_color_viridis_c,
    theme,
    unit,
    xlab,
    ylab,
)

from .._uns import get_state
from ._helpers import as_dense, rotation_matrix
from ._theme import monocle_theme_opts

__all__ = ["plot_cell_trajectory", "plot_complex_cell_trajectory"]


def _centroid_adj(
    edges: np.ndarray, weights: np.ndarray, n: int,
) -> list[list[tuple[int, float]]]:
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (u, v), w in zip(edges, weights):
        adj[int(u)].append((int(v), float(w)))
        adj[int(v)].append((int(u), float(w)))
    return adj


def _centroid_state(adata: AnnData) -> dict:
    state = get_state(adata)
    if "ddrtree" not in state:
        raise RuntimeError(
            "DDRTree state missing — call reduce_dimension(method='DDRTree')."
        )
    return state["ddrtree"]


def _markers_long_df(
    adata: AnnData, markers: Sequence[str],
) -> pd.DataFrame:
    if "gene_short_name" in adata.var.columns:
        mask = adata.var["gene_short_name"].astype(str).isin(markers)
    else:
        mask = pd.Series(False, index=adata.var_names)
    mask = mask | adata.var_names.isin(markers)
    if not mask.any():
        return pd.DataFrame(columns=["cell_id", "feature_label", "value"])
    sub = adata[:, mask]
    X = as_dense(sub.X).astype(float)
    cell_names = sub.obs_names.to_numpy()
    rows = []
    for j, gene in enumerate(sub.var_names):
        if "gene_short_name" in sub.var.columns:
            short = sub.var["gene_short_name"].iloc[j]
            label = str(short) if pd.notna(short) else str(gene)
        else:
            label = str(gene)
        rows.append(pd.DataFrame({
            "cell_id": cell_names,
            "feature_label": np.repeat(label, len(cell_names)),
            "value": X[:, j],
        }))
    return pd.concat(rows, ignore_index=True)


def plot_cell_trajectory(
    adata: AnnData,
    x: int = 1,
    y: int = 2,
    color_by: str = "State",
    show_tree: bool = True,
    show_backbone: bool = True,
    backbone_color: str = "black",
    markers: Sequence[str] | None = None,
    use_color_gradient: bool = False,
    markers_linear: bool = False,
    show_cell_names: bool = False,
    show_state_number: bool = False,
    cell_size: float = 1.5,
    cell_link_size: float = 0.75,
    cell_name_size: float = 2,
    state_number_size: float = 2.9,
    show_branch_points: bool = True,
    theta: float = 0.0,
) -> GGPlot:
    """Port of R's ``plot_cell_trajectory``.

    Plots cells in DDRTree coordinates with the principal-graph backbone and
    branch-point markers overlaid. Mirrors the R signature 1:1; ``x`` and
    ``y`` are 1-based component indices for compatibility.
    """
    if "X_dr" not in adata.obsm:
        raise RuntimeError(
            "Reduced dimensions missing — call reduce_dimension first."
        )
    ddr = _centroid_state(adata)
    K = np.asarray(ddr["K"], dtype=float)
    edges = np.asarray(ddr["mst_edges"], dtype=int)
    weights = np.asarray(ddr["mst_weights"], dtype=float)
    n_centroids = K.shape[1]
    adj = _centroid_adj(edges, weights, n_centroids)

    Z = np.asarray(adata.obsm["X_dr"], dtype=float)  # n_cells x dim
    if Z.shape[1] < max(x, y):
        raise ValueError(
            f"Reduced space has {Z.shape[1]} components; need at least {max(x, y)}."
        )

    rot = rotation_matrix(theta)
    cell_xy = Z[:, [x - 1, y - 1]] @ rot.T
    Kt = K.T  # n_centroids x dim
    if Kt.shape[1] < max(x, y):
        raise ValueError("DDRTree centroids have fewer dims than requested x/y.")
    centroid_xy = Kt[:, [x - 1, y - 1]] @ rot.T

    obs = adata.obs.copy()
    cell_df = pd.DataFrame({
        "data_dim_1": cell_xy[:, 0],
        "data_dim_2": cell_xy[:, 1],
        "sample_name": adata.obs_names.astype(str),
        "sample_state": obs["State"].astype(str).to_numpy()
        if "State" in obs.columns else "1",
    })
    for col in obs.columns:
        if col not in cell_df.columns:
            cell_df[col] = obs[col].to_numpy()

    edge_rows = []
    for u, v in edges:
        edge_rows.append({
            "source_prin_graph_dim_1": centroid_xy[int(u), 0],
            "source_prin_graph_dim_2": centroid_xy[int(u), 1],
            "target_prin_graph_dim_1": centroid_xy[int(v), 0],
            "target_prin_graph_dim_2": centroid_xy[int(v), 1],
        })
    edge_df = pd.DataFrame(edge_rows)

    markers_df = None
    if markers is not None:
        markers_df = _markers_long_df(adata, list(markers))
        if markers_df.empty:
            markers_df = None

    if markers_df is not None:
        cell_df = cell_df.merge(
            markers_df, left_on="sample_name", right_on="cell_id",
        )
        if use_color_gradient and markers_linear:
            g = (
                ggplot(cell_df, aes(x="data_dim_1", y="data_dim_2"))
                + geom_point(aes(color="value"), size=cell_size)
                + scale_color_viridis_c(name="value")
            )
        elif use_color_gradient:
            cell_df["log_value"] = np.log10(cell_df["value"].to_numpy(float) + 0.1)
            g = (
                ggplot(cell_df, aes(x="data_dim_1", y="data_dim_2"))
                + geom_point(aes(color="log_value"), size=cell_size)
                + scale_color_viridis_c(name="log10(value + 0.1)")
            )
        else:
            g = ggplot(cell_df, aes(x="data_dim_1", y="data_dim_2"))
    else:
        g = ggplot(cell_df, aes(x="data_dim_1", y="data_dim_2"))

    if show_tree and not edge_df.empty:
        g = g + geom_segment(
            aes(
                x="source_prin_graph_dim_1",
                y="source_prin_graph_dim_2",
                xend="target_prin_graph_dim_1",
                yend="target_prin_graph_dim_2",
            ),
            data=edge_df,
            linewidth=cell_link_size,
            linetype="solid",
            color=backbone_color,
        )

    if not (markers_df is not None and use_color_gradient):
        g = g + geom_point(aes(color=color_by), size=cell_size)

    if show_branch_points:
        deg = np.array([len(a) for a in adj], dtype=int)
        branch_centroids = np.flatnonzero(deg > 2)
        if branch_centroids.size:
            bp_df = pd.DataFrame({
                "prin_graph_dim_1": centroid_xy[branch_centroids, 0],
                "prin_graph_dim_2": centroid_xy[branch_centroids, 1],
                "branch_point_idx": np.arange(1, branch_centroids.size + 1),
            })
            g = (
                g
                + geom_point(
                    aes(x="prin_graph_dim_1", y="prin_graph_dim_2"),
                    data=bp_df, size=5, color=backbone_color,
                )
                + geom_text(
                    aes(
                        x="prin_graph_dim_1", y="prin_graph_dim_2",
                        label="branch_point_idx",
                    ),
                    data=bp_df, size=4, color="white",
                )
            )

    if show_cell_names:
        g = g + geom_text(aes(label="sample_name"), size=cell_name_size)
    if show_state_number:
        g = g + geom_text(aes(label="sample_state"), size=state_number_size)

    g = (
        g
        + monocle_theme_opts()
        + xlab(f"Component {x}")
        + ylab(f"Component {y}")
        + theme(**{
            "legend.position": "top",
            "legend.key.height": unit(0.35, "in"),
            "legend.key": element_blank(),
            "panel.background": element_rect(fill="white"),
        })
    )
    return g


def _layout_as_tree(
    adj: Sequence[Sequence[tuple[int, float]]], root: int,
) -> np.ndarray:
    """Iterative Reingold–Tilford-style layout: x = leaf order, y = -depth."""
    n = len(adj)
    parent = np.full(n, -1, dtype=int)
    depth = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    visited[root] = True
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v, _ in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                queue.append(v)

    children: list[list[int]] = [[] for _ in range(n)]
    for v in range(n):
        if parent[v] >= 0:
            children[parent[v]].append(v)

    pos = np.zeros(n, dtype=float)
    counter = [0.0]
    stack: list[tuple[int, int]] = [(root, 0)]
    while stack:
        node, state = stack.pop()
        if state == 0:
            if not children[node]:
                pos[node] = counter[0]
                counter[0] += 1.0
            else:
                stack.append((node, 1))
                for child in reversed(children[node]):
                    stack.append((child, 0))
        else:
            child_positions = [pos[c] for c in children[node]]
            pos[node] = (child_positions[0] + child_positions[-1]) / 2.0

    coords = np.zeros((n, 2), dtype=float)
    coords[:, 0] = pos
    coords[:, 1] = -depth.astype(float)
    return coords


def _select_root_vertex_for_complex(
    adata: AnnData,
    adj: Sequence[Sequence[tuple[int, float]]],
    closest_vertex: np.ndarray,
    root_states: Iterable[int] | None,
) -> int:
    deg = np.array([len(a) for a in adj], dtype=int)
    if root_states is None:
        if "Pseudotime" in adata.obs.columns:
            pt = adata.obs["Pseudotime"].to_numpy(dtype=float)
            root_cells = np.flatnonzero(pt == 0.0)
        else:
            root_cells = np.array([0], dtype=int)
        candidate_centroids = np.unique(closest_vertex[root_cells]) \
            if root_cells.size else np.unique(closest_vertex)
    else:
        if "State" not in adata.obs.columns:
            raise RuntimeError("State column missing — call order_cells first.")
        s = adata.obs["State"].astype(int).to_numpy()
        root_states = list(map(int, root_states))
        cell_mask = np.isin(s, root_states)
        candidate_centroids = np.unique(closest_vertex[cell_mask])

    leaves = candidate_centroids[deg[candidate_centroids] == 1]
    if leaves.size:
        return int(leaves[0])
    return int(candidate_centroids[0])


def plot_complex_cell_trajectory(
    adata: AnnData,
    x: int = 1,
    y: int = 2,
    root_states: Iterable[int] | None = None,
    color_by: str = "State",
    show_tree: bool = True,
    show_backbone: bool = True,
    backbone_color: str = "black",
    markers: Sequence[str] | None = None,
    show_cell_names: bool = False,
    cell_size: float = 1.5,
    cell_link_size: float = 0.75,
    cell_name_size: float = 2,
    show_branch_points: bool = True,
) -> GGPlot:
    """Port of R's ``plot_complex_cell_trajectory``.

    Lays the principal-graph centroid MST out as a tree rooted at a chosen
    state, then places each cell at its closest centroid's layout coordinate
    with a small jitter to avoid over-plotting.
    """
    ddr = _centroid_state(adata)
    K = np.asarray(ddr["K"], dtype=float)
    edges = np.asarray(ddr["mst_edges"], dtype=int)
    weights = np.asarray(ddr["mst_weights"], dtype=float)
    closest_vertex = np.asarray(ddr["closest_vertex"], dtype=int)
    n_centroids = K.shape[1]
    adj = _centroid_adj(edges, weights, n_centroids)

    root_vertex = _select_root_vertex_for_complex(
        adata, adj, closest_vertex, root_states,
    )
    coords = _layout_as_tree(adj, root_vertex)  # n_centroids x 2

    cell_xy = coords[closest_vertex]
    obs = adata.obs.copy()
    cell_df = pd.DataFrame({
        "data_dim_1": cell_xy[:, 0],
        "data_dim_2": cell_xy[:, 1],
        "sample_name": adata.obs_names.astype(str),
    })
    for col in obs.columns:
        if col not in cell_df.columns:
            cell_df[col] = obs[col].to_numpy()

    edge_rows = []
    for u, v in edges:
        edge_rows.append({
            "source_prin_graph_dim_1": coords[int(u), 0],
            "source_prin_graph_dim_2": coords[int(u), 1],
            "target_prin_graph_dim_1": coords[int(v), 0],
            "target_prin_graph_dim_2": coords[int(v), 1],
        })
    edge_df = pd.DataFrame(edge_rows)

    markers_df = None
    if markers is not None:
        markers_df = _markers_long_df(adata, list(markers))
        if markers_df.empty:
            markers_df = None

    if markers_df is not None:
        cell_df = cell_df.merge(
            markers_df, left_on="sample_name", right_on="cell_id",
        )
        g = ggplot(cell_df, aes(x="data_dim_1", y="data_dim_2"))
    else:
        g = ggplot(cell_df, aes(x="data_dim_1", y="data_dim_2"))

    if show_tree and not edge_df.empty:
        g = g + geom_segment(
            aes(
                x="source_prin_graph_dim_1",
                y="source_prin_graph_dim_2",
                xend="target_prin_graph_dim_1",
                yend="target_prin_graph_dim_2",
            ),
            data=edge_df,
            linewidth=cell_link_size,
            linetype="solid",
            color=backbone_color,
        )

    use_numeric_color = (
        color_by in cell_df.columns
        and pd.api.types.is_numeric_dtype(cell_df[color_by])
    )
    if use_numeric_color:
        cell_df["__log_color"] = np.log10(
            cell_df[color_by].to_numpy(float) + 0.1
        )
        g = g + geom_jitter(
            aes(color="__log_color"), data=cell_df,
            size=cell_size, height=5, width=0,
        ) + scale_color_viridis_c(name=f"log10({color_by} + 0.1)")
    else:
        g = g + geom_jitter(
            aes(color=color_by), data=cell_df,
            size=cell_size, height=5, width=0,
        )

    if show_branch_points:
        deg = np.array([len(a) for a in adj], dtype=int)
        branch_centroids = np.flatnonzero(deg > 2)
        if branch_centroids.size:
            bp_df = pd.DataFrame({
                "x": coords[branch_centroids, 0],
                "y": coords[branch_centroids, 1],
                "branch_point_idx": np.arange(1, branch_centroids.size + 1),
            })
            g = (
                g
                + geom_point(
                    aes(x="x", y="y"),
                    data=bp_df, size=2 * cell_size, color=backbone_color,
                )
                + geom_text(
                    aes(x="x", y="y", label="branch_point_idx"),
                    data=bp_df, size=1.5 * cell_size, color="white",
                )
            )

    if show_cell_names:
        g = g + geom_text(aes(label="sample_name"), size=cell_name_size)

    g = (
        g
        + monocle_theme_opts()
        + xlab("")
        + ylab("")
        + theme(**{
            "legend.position": "top",
            "legend.key.height": unit(0.35, "in"),
            "legend.key": element_blank(),
            "panel.background": element_rect(fill="white"),
            "axis.text.x": element_blank(),
            "axis.text.y": element_blank(),
            "axis.ticks": element_blank(),
        })
    )
    return g
