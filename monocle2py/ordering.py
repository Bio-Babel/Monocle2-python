"""Ordering API (``set_ordering_filter`` + ``order_cells``).

Ports the DDRTree branch of R's ``orderCells``. Cells are projected onto the
principal graph learned by DDRTree, a cell-level MST is built on those
projections, and pseudotime is the geodesic distance along that MST from a
root cell. State labels come from the principal-graph traversal so cells in
the same trajectory segment share a state.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse.csgraph import minimum_spanning_tree

from ._uns import ensure_state, get_state

__all__ = ["set_ordering_filter", "order_cells"]


def set_ordering_filter(
    adata: AnnData, ordering_genes: Iterable[str]
) -> AnnData:
    """R's ``setOrderingFilter``: mark genes that drive trajectory ordering.

    Parameters
    ----------
    adata : anndata.AnnData
    ordering_genes : iterable[str]
        Gene names (matched against ``adata.var_names``) to flag.

    Returns
    -------
    anndata.AnnData
        The same AnnData (modified in place) with ``var['use_for_ordering']``
        set to a boolean mask.
    """
    wanted = set(map(str, ordering_genes))
    mask = np.fromiter(
        (str(g) in wanted for g in adata.var_names),
        dtype=bool, count=adata.n_vars,
    )
    adata.var["use_for_ordering"] = mask
    return adata


def _pairwise(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def _adj_from_edges(
    edges: np.ndarray, weights: np.ndarray, n_nodes: int,
) -> list[list[tuple[int, float]]]:
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for (u, v), w in zip(edges, weights):
        adj[int(u)].append((int(v), float(w)))
        adj[int(v)].append((int(u), float(w)))
    return adj


def _degree(adj: Sequence[Sequence[tuple[int, float]]]) -> np.ndarray:
    return np.array([len(nbrs) for nbrs in adj], dtype=np.int64)


def _farthest_node(
    adj: Sequence[Sequence[tuple[int, float]]], start: int,
) -> tuple[int, dict[int, int]]:
    """Return ``(farthest_node, parent_map)`` along the unique tree path."""
    n = len(adj)
    dist = np.full(n, -1.0)
    parent: dict[int, int] = {start: -1}
    dist[start] = 0.0
    stack = [start]
    while stack:
        u = stack.pop()
        for v, w in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + w
                parent[v] = u
                stack.append(v)
    far = int(np.argmax(dist))
    return far, parent


def _tree_diameter(
    adj: Sequence[Sequence[tuple[int, float]]],
) -> list[int]:
    """Endpoints of the longest weighted path in a tree, in path order."""
    if not adj:
        return []
    u, _ = _farthest_node(adj, 0)
    v, parent = _farthest_node(adj, u)
    path: list[int] = []
    cur = v
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _dfs_with_parents(
    adj: Sequence[Sequence[tuple[int, float]]], root: int,
) -> tuple[list[int], list[int]]:
    """Iterative DFS. Returns ``(visit_order, parent_in_order)``."""
    n = len(adj)
    parent = [-1] * n
    visited = np.zeros(n, dtype=bool)
    order: list[int] = []
    stack = [root]
    visited[root] = True
    while stack:
        u = stack.pop()
        order.append(u)
        for v, _ in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                stack.append(v)
    return order, parent


def _extract_ordering(
    dp: np.ndarray,
    adj: Sequence[Sequence[tuple[int, float]]],
    root: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertex-level pseudotime, state, parent from a tree.

    Mirrors R's ``extract_ddrtree_ordering``: DFS from *root*, accumulate edge
    weights into pseudotime, increment state whenever the current node's
    parent is a branch point (degree > 2).
    """
    n = dp.shape[0]
    deg = _degree(adj)
    order, parent_in_order = _dfs_with_parents(adj, root)

    pseudotime = np.zeros(n, dtype=np.float64)
    state = np.ones(n, dtype=np.int64)
    parents = np.full(n, -1, dtype=np.int64)

    curr_state = 1
    for node in order:
        p = parent_in_order[node]
        if p >= 0:
            pseudotime[node] = pseudotime[p] + dp[node, p]
            if deg[p] > 2:
                curr_state += 1
            state[node] = curr_state
            parents[node] = p
        else:
            pseudotime[node] = 0.0
            state[node] = curr_state
    return pseudotime, state, parents


def _project_point_to_segment(p: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Closest point on segment AB to p (clamps t to [0, 1])."""
    AB = B - A
    AB_sq = float(AB @ AB)
    if AB_sq == 0.0:
        return A.copy()
    t = float((p - A) @ AB) / AB_sq
    if t < 0.0:
        return A.copy()
    if t > 1.0:
        return B.copy()
    return A + t * AB


def _project_point_to_line(p: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal projection of p onto the infinite line through A, B."""
    AB = B - A
    AB_sq = float(AB @ AB)
    if AB_sq == 0.0:
        return A.copy()
    t = float((p - A) @ AB) / AB_sq
    return A + t * AB


def _project_cells(
    Z: np.ndarray,
    Y: np.ndarray,
    closest_vertex: np.ndarray,
    adj: Sequence[Sequence[tuple[int, float]]],
) -> np.ndarray:
    """Project each cell to nearest line segment of the principal graph.

    Tip leaves use orthogonal line projection (matching ``projPointOnLine``);
    internal vertices use clamped segment projection.
    """
    deg = _degree(adj)
    tip_leaves = deg == 1
    n_dim, n_cells = Z.shape
    P = np.zeros((n_dim, n_cells), dtype=np.float64)
    for i in range(n_cells):
        v = int(closest_vertex[i])
        z_i = Z[:, i]
        best_proj = None
        best_dist = np.inf
        for u, _ in adj[v]:
            A = Y[:, v]
            B = Y[:, u]
            if tip_leaves[v]:
                q = _project_point_to_line(z_i, A, B)
            else:
                q = _project_point_to_segment(z_i, A, B)
            d = float(np.linalg.norm(z_i - q))
            if d < best_dist:
                best_dist = d
                best_proj = q
        if best_proj is None:
            best_proj = Y[:, v].copy()
        P[:, i] = best_proj
    return P


def _cell_level_mst(P: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
    """Build the cell-level MST after projection (with R's min_dist offset)."""
    dp = _pairwise(P.T)
    nonzero = dp[dp > 0]
    min_dist = float(nonzero.min()) if nonzero.size else 0.0
    dp = dp + min_dist
    np.fill_diagonal(dp, 0.0)

    tree = minimum_spanning_tree(dp).tocoo()
    n = dp.shape[0]
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for r, c, w in zip(tree.row, tree.col, tree.data):
        adj[int(r)].append((int(c), float(w)))
        adj[int(c)].append((int(r), float(w)))
    return dp, adj


def _select_root_vertex(
    adj_vert: Sequence[Sequence[tuple[int, float]]],
    reverse: bool,
) -> int:
    """Pick a principal-graph root vertex that matches R's ``select_root_cell``.

    R (``order_cells.R:1014-1019``) uses ``get.diameter(mst)``, then
    ``diameter[1]`` for the default path and ``diameter[length(diameter)]``
    for ``reverse=TRUE``. Empirically, R's ``get.diameter`` returns the
    path from the **step-2-farthest** endpoint (the true diameter
    endpoint found by the second BFS) to the **step-1-farthest** endpoint
    (the first BFS seed). Our :func:`_tree_diameter` returns the path in
    the opposite order — from step-1 to step-2 — so the first endpoint
    Python sees is R's ``diameter[length(diameter)]``. Flip the index
    accordingly so that ``reverse=False`` picks R's ``diameter[1]``.

    Verified on the lung and Paul fixtures: Python+R pick the same
    root Y-vertex for both ``reverse=False`` and ``reverse=True``.
    """
    diam = _tree_diameter(adj_vert)
    if not diam:
        raise RuntimeError("Empty MST: cannot select a root vertex.")
    return diam[0] if reverse else diam[-1]


def _vertex_root_from_state(
    adata: AnnData,
    root_state: int,
    closest_vertex: np.ndarray,
    Z: np.ndarray,
    reverse: bool,
    prev_root_vertex: int | None = None,
) -> int:
    """Re-derive the principal-graph root vertex given a user state choice.

    Ports R's ``select_root_cell`` root_state branch
    (``order_cells.R:961-1010``):

    1. Candidate cells = cells where ``obs['State'] == root_state``.
    2. Build a **cell-level** Euclidean MST on
       ``reducedDimS[:, candidates]`` (R uses ``reducedDimS``; Python's
       equivalent is ``adata.obsm['X_dr']``).
    3. ``diameter = get.diameter(sub_mst)`` → a path of candidate cells.
    4. Pick the diameter endpoint cell by Pseudotime min/max:
       - if a previous ``root_cell`` exists and is in ``root_state``,
         pick the cell with **min** Pseudotime on the diameter;
       - otherwise pick the cell with **max** Pseudotime.
       - ``reverse=True`` forces the min-Pseudotime pick.
    5. Return that cell's ``closest_vertex`` as the Y-level root.
    """
    if "State" not in adata.obs.columns or "Pseudotime" not in adata.obs.columns:
        raise RuntimeError(
            "State / Pseudotime have not yet been set. Call order_cells "
            "without root_state first, then try this call again."
        )
    state_col = adata.obs["State"].astype(int).to_numpy()
    in_state = np.flatnonzero(state_col == int(root_state))
    if in_state.size == 0:
        raise RuntimeError(f"No cells for State = {root_state}")

    if in_state.size == 1:
        return int(closest_vertex[in_state[0]])

    # Cell-level MST on reducedDimS over the in-state subset
    sub_Z = Z[:, in_state]  # dim x k
    sub_dp = _pairwise(sub_Z.T)
    sub_tree = minimum_spanning_tree(sub_dp).tocoo()
    sub_adj: list[list[tuple[int, float]]] = [[] for _ in range(in_state.size)]
    for r, c, w in zip(sub_tree.row, sub_tree.col, sub_tree.data):
        sub_adj[int(r)].append((int(c), float(w)))
        sub_adj[int(c)].append((int(r), float(w)))

    sub_diam = _tree_diameter(sub_adj)
    if not sub_diam:
        raise RuntimeError(f"No valid root vertex for State = {root_state}")

    pseudotime = adata.obs["Pseudotime"].to_numpy(dtype=float)
    # Diameter endpoints are the FIRST and LAST elements (R returns the
    # full path). Pick by min/max of pseudotime among cells on the
    # diameter path.
    diam_cells = in_state[np.asarray(sub_diam, dtype=np.int64)]
    diam_pt = pseudotime[diam_cells]

    use_min = bool(reverse)
    if not use_min and prev_root_vertex is not None:
        prev_cells = np.flatnonzero(closest_vertex == int(prev_root_vertex))
        prev_state = state_col[prev_cells]
        if prev_cells.size > 0 and (prev_state == int(root_state)).any():
            use_min = True

    target_idx = int(np.argmin(diam_pt)) if use_min else int(np.argmax(diam_pt))
    chosen_cell = int(diam_cells[target_idx])
    return int(closest_vertex[chosen_cell])


def _pick_root_cell_from_cell_mst(
    adata: AnnData,
    adj_cell: Sequence[Sequence[tuple[int, float]]],
    Z: np.ndarray,
    closest_vertex: np.ndarray,
    root_state: int | None,
    reverse: bool,
    prev_root_vertex: int | None,
) -> int:
    """Fallback root-cell selection mirroring R's ``select_root_cell``.

    Ports ``order_cells.R:961-1023`` against the **cell-level**
    projection tree (which ``project2MST`` installs into
    ``minSpanningTree(cds)`` on line 1132). When the tip-leaf filter in
    ``_order_cells_ddrtree`` yields nothing, R re-runs
    ``select_root_cell(cds, root_state, reverse)``; this helper
    reproduces that behaviour:

    * ``root_state is None``: pick an endpoint of the weighted diameter
      of the cell MST (``order_cells.R:1015-1020``).
    * ``root_state`` given: build a sub-MST over cells in that state in
      ``reducedDimS`` space, pick the diameter endpoint by Pseudotime
      (min if ``reverse`` or the previous root lives in this state,
      else max). Matches ``order_cells.R:961-1009`` up to the
      DDRTree-specific closest-vertex detour (which is a no-op at the
      cell level since we already want a cell index).
    """
    if root_state is None:
        return _select_root_vertex(adj_cell, reverse=reverse)
    state_col = adata.obs["State"].astype(int).to_numpy()
    in_state = np.flatnonzero(state_col == int(root_state))
    if in_state.size == 0:
        raise RuntimeError(f"No cells for State = {root_state}")
    if in_state.size == 1:
        return int(in_state[0])
    sub_Z = Z[:, in_state]
    sub_dp = _pairwise(sub_Z.T)
    sub_tree = minimum_spanning_tree(sub_dp).tocoo()
    sub_adj: list[list[tuple[int, float]]] = [[] for _ in range(in_state.size)]
    for r_, c_, w_ in zip(sub_tree.row, sub_tree.col, sub_tree.data):
        sub_adj[int(r_)].append((int(c_), float(w_)))
        sub_adj[int(c_)].append((int(r_), float(w_)))
    sub_diam = _tree_diameter(sub_adj)
    if not sub_diam:
        raise RuntimeError(f"No valid root cell for State = {root_state}")
    pseudotime = adata.obs["Pseudotime"].to_numpy(dtype=float)
    diam_cells = in_state[np.asarray(sub_diam, dtype=np.int64)]
    diam_pt = pseudotime[diam_cells]
    use_min = bool(reverse)
    if not use_min and prev_root_vertex is not None:
        prev_cells = np.flatnonzero(closest_vertex == int(prev_root_vertex))
        prev_state = state_col[prev_cells]
        if prev_cells.size > 0 and (prev_state == int(root_state)).any():
            use_min = True
    target_idx = int(np.argmin(diam_pt)) if use_min else int(np.argmax(diam_pt))
    return int(diam_cells[target_idx])


def _order_cells_ddrtree(
    adata: AnnData,
    root_state: int | None,
    reverse: bool,
) -> None:
    state = get_state(adata)
    ddr = state["ddrtree"]
    K = np.asarray(ddr["K"], dtype=np.float64)            # dim x n_centers
    closest_vertex = np.asarray(ddr["closest_vertex"]).astype(np.int64)
    Z = np.asarray(adata.obsm["X_dr"], dtype=np.float64).T  # dim x n_cells

    n_centers = K.shape[1]
    edges = np.asarray(ddr["mst_edges"], dtype=np.int64)
    weights = np.asarray(ddr["mst_weights"], dtype=np.float64)
    adj_vert = _adj_from_edges(edges, weights, n_centers)

    dp_vert = _pairwise(K.T)

    if root_state is None:
        root_vertex = _select_root_vertex(adj_vert, reverse=reverse)
    else:
        prev_aux = state.get("aux_ordering", {}).get("DDRTree", {})
        prev_root_vertex = prev_aux.get("root_vertex")
        root_vertex = _vertex_root_from_state(
            adata, int(root_state), closest_vertex, Z,
            reverse=reverse,
            prev_root_vertex=(
                int(prev_root_vertex) if prev_root_vertex is not None else None
            ),
        )

    _, vertex_state, _ = _extract_ordering(dp_vert, adj_vert, root_vertex)

    P = _project_cells(Z, K, closest_vertex, adj_vert)
    dp_cell, adj_cell = _cell_level_mst(P)

    cells_at_root = np.flatnonzero(closest_vertex == root_vertex)
    if cells_at_root.size == 0:
        # R ``order_cells.R:1136-1138``: when no cells map to the root
        # Y-vertex, R reuses ``root_cell_idx`` (a Y-vertex index) as a
        # cell-array position. Replicate that quirk so the subsequent
        # tip-leaf filter sees the same candidate set as R.
        cells_at_root = np.array([int(root_vertex)], dtype=np.int64)

    deg_cell = _degree(adj_cell)
    tip_cells = cells_at_root[deg_cell[cells_at_root] == 1]
    if tip_cells.size > 0:
        root_cell = int(tip_cells[0])
    else:
        # R ``order_cells.R:1144-1146``: tip-leaf intersection is empty,
        # fall back to ``select_root_cell`` on the cell-level projection
        # tree (the MST ``project2MST`` installs on line 1132).
        prev_aux = state.get("aux_ordering", {}).get("DDRTree", {})
        prev_root_vertex = prev_aux.get("root_vertex")
        root_cell = _pick_root_cell_from_cell_mst(
            adata, adj_cell, Z, closest_vertex, root_state, reverse,
            prev_root_vertex=(
                int(prev_root_vertex) if prev_root_vertex is not None else None
            ),
        )

    pseudotime_cell, _, parents_cell = _extract_ordering(
        dp_cell, adj_cell, root_cell,
    )

    pseudotime = pseudotime_cell
    state_per_cell = vertex_state[closest_vertex]

    cell_names = adata.obs_names.to_numpy()
    parent_names = np.array(
        [cell_names[p] if p >= 0 else "" for p in parents_cell],
        dtype=object,
    )

    adata.obs["Pseudotime"] = pseudotime.astype(float)
    # R's ``orderCells`` (``order_cells.R:1153-1156``) only overwrites
    # ``State`` on the first call (``root_state is NULL``). When the user
    # passes ``root_state`` to pivot the root, R keeps the prior State
    # labels so downstream code (``branch_states``, plotting) can index by
    # the original numbering. Mirror that behaviour here.
    if root_state is None or "State" not in adata.obs.columns:
        adata.obs["State"] = state_per_cell.astype(np.int64)
    # R stores ``State`` as a ``factor`` (``factor(states)`` in
    # ``extract_ddrtree_ordering``/``pq_helper``), so plotting functions
    # that check ``class(data_df[, color_by]) == 'numeric'`` fall through
    # to a discrete colour scale. We mirror that with ``pd.Categorical``
    # on every order_cells call: bare ``int64`` (what h5ad stores)
    # would trip ``is_float_dtype == False`` fine, but downstream R-like
    # callers still want a true factor dtype. Categories are the sorted
    # unique ints so repeated calls are idempotent.
    state_col = adata.obs["State"]
    if not isinstance(state_col.dtype, pd.CategoricalDtype):
        # State here was either just produced by ``_extract_ordering`` as
        # int64 (line 473) or preserved from a previous ``order_cells``
        # call. Either way it must round-trip cleanly to int64; any
        # non-numeric content means an upstream contract violation and
        # should raise rather than be silently coerced to object dtype.
        values = state_col.to_numpy().astype(np.int64)
        categories = np.unique(values)
        adata.obs["State"] = pd.Categorical(values, categories=categories)
    adata.obs["Parent"] = parent_names

    # R's ``auxOrderingData$DDRTree$branch_points`` holds centroid (Y-node)
    # indices with degree>2 in the **principal-graph MST** (BEAM.R:107 reads
    # this list). Using cell-level degrees here would change BEAM branch
    # identification. See order_cells.R:1164 / 1177.
    centroid_branch_points = np.flatnonzero(_degree(adj_vert) > 2).astype(
        np.int64
    )
    aux = ensure_state(adata).setdefault("aux_ordering", {})
    aux["DDRTree"] = {
        "root_cell": cell_names[root_cell],
        "root_vertex": int(root_vertex),
        "pr_graph_cell_proj_dist": P,
        "branch_points": centroid_branch_points,
        # Keep the closest-vertex lookup here too so BEAM can re-use it
        # without digging into ``uns['monocle2']['ddrtree']``; mirrors R's
        # ``pr_graph_cell_proj_closest_vertex`` slot.
        "pr_graph_cell_proj_closest_vertex": closest_vertex.astype(np.int64),
    }


def order_cells(
    adata: AnnData,
    root_state: int | None = None,
    reverse: bool | None = None,
) -> AnnData:
    """Assign each cell a ``Pseudotime`` and ``State`` along the trajectory.

    Parameters
    ----------
    adata : anndata.AnnData
        Must already have been processed with
        :func:`~monocle2py.reduce_dimension` (DDRTree).
    root_state : int, optional
        State to use as the root of the trajectory. Requires a prior
        ``order_cells`` call so ``adata.obs['State']`` exists.
    reverse : bool, optional
        If True, swap the principal-graph MST diameter endpoint used as
        the root. This mirrors R's DDRTree path: ``select_root_cell``
        (``order_cells.R:1014-1019``) picks ``diameter[1]`` by default
        and ``diameter[length(diameter)]`` when ``reverse=TRUE``. Note
        this does **not** negate the resulting ``Pseudotime`` scalar —
        pseudotime is still measured as cumulative arc length from the
        (newly chosen) root, so the two ``reverse`` outputs are not
        related by ``t -> max(t) - t``. R's ``reverse_ordering`` helper
        (``order_cells.R:789``) that does negate pseudotime lives only
        on the ICA branch, which we do not port.

    Notes
    -----
    R's ``orderCells(cds, root_state, num_paths, reverse)`` carries a
    ``num_paths`` parameter that gates the ICA branch's k-cut behaviour
    and is silently ignored for DDRTree (R emits a warning at
    ``order_cells.R:1118-1119``). Since we do not port the ICA branch,
    we drop ``num_paths`` from this signature entirely — passing it
    raises the default ``TypeError: got an unexpected keyword argument
    'num_paths'``, which is louder and more accurate feedback than an
    accept-and-ignore stub.

    Returns
    -------
    anndata.AnnData
        The same AnnData (modified in place) with ``Pseudotime``, ``State``,
        ``Parent`` columns on ``obs``.
    """
    state = get_state(adata)
    dim_reduce_type = state.get("dim_reduce_type")
    if dim_reduce_type is None:
        raise RuntimeError(
            "dimensionality not yet reduced. Call reduce_dimension "
            "before order_cells."
        )
    if dim_reduce_type != "DDRTree":
        raise NotImplementedError(
            f"order_cells only ports the DDRTree branch (got {dim_reduce_type!r})."
        )
    _order_cells_ddrtree(adata, root_state=root_state, reverse=bool(reverse))
    return adata
