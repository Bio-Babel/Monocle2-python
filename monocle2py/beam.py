"""Slice 7: BEAM — branch-dependent expression analysis.

Ports ``BEAM``, ``branchTest``, ``buildBranchCellDataSet`` and ``calILRs``
from ``monocle2/R/BEAM.R``. Requires :func:`~monocle2py.order_cells` to
have run so ``Pseudotime``, ``State``, the principal-graph **centroid
MST** (Y-node level), and the ``closest_vertex`` mapping are available in
``adata.uns['monocle2']``.

R's ``buildBranchCellDataSet`` walks the principal graph at the Y-node
(centroid) level — *not* the cell level — and maps each Y-node back to
the cells whose ``closest_vertex`` sits on that node. This Python port
mirrors that choice; an earlier implementation that built a fresh
cell×cell MST produced incorrect branch identification and
non-deterministic A/B labelling on multifurcations.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from ._internal._formula import formula_terms, normalize_formula
from ._uns import SIZE_FACTOR_COL, ensure_state, get_state
from .differential import differential_gene_test, gen_smooth_curves
from .preprocess import vst_exprs

__all__ = [
    "build_branch_cell_dataset",
    "branch_test",
    "beam",
    "cal_ilrs",
]


def _pairwise(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def _centroid_mst_adj(
    adata: AnnData,
) -> tuple[list[list[int]], np.ndarray]:
    """Return the centroid-MST adjacency list and the ``closest_vertex`` map.

    Mirrors R's ``minSpanningTree(cds)`` (under DDRTree this holds the
    principal-graph MST over Y-nodes; ``order_cells.R:1132``) together
    with ``pr_graph_cell_proj_closest_vertex`` (``auxOrderingData``).
    """
    state = get_state(adata)
    ddr = state.get("ddrtree") or {}
    aux = state.get("aux_ordering", {}).get("DDRTree", {})
    if "mst_edges" not in ddr or "closest_vertex" not in ddr:
        raise RuntimeError(
            "order_cells has not been called under DDRTree (missing "
            "principal-graph MST or closest-vertex mapping)."
        )
    K = np.asarray(ddr["K"])
    n_centers = K.shape[1] if K.ndim == 2 else K.shape[0]
    edges = np.asarray(ddr["mst_edges"], dtype=np.int64)
    adj: list[list[int]] = [[] for _ in range(n_centers)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    closest_vertex = np.asarray(
        aux.get("pr_graph_cell_proj_closest_vertex", ddr["closest_vertex"]),
        dtype=np.int64,
    )
    return adj, closest_vertex


def _degree(adj: Sequence[Sequence[int]]) -> np.ndarray:
    return np.array([len(nbrs) for nbrs in adj], dtype=np.int64)


def _path_in_tree(
    adj: Sequence[Sequence[int]], source: int, target: int,
) -> list[int]:
    """Unique source→target path on a tree via BFS (order-independent)."""
    n = len(adj)
    parent = [-1] * n
    visited = np.zeros(n, dtype=bool)
    q = deque([source])
    visited[source] = True
    while q:
        u = q.popleft()
        if u == target:
            break
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                q.append(v)
    if not visited[target]:
        return []
    path: list[int] = []
    cur = target
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _bfs_component(
    adj: Sequence[Sequence[int]], source: int, removed: int,
) -> set[int]:
    """Nodes reachable from *source* with *removed* excluded from the graph."""
    visited = {source}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v == removed or v in visited:
                continue
            visited.add(v)
            q.append(v)
    return visited


def _sort_branch_points_by_vertex_pseudotime(
    branch_points: np.ndarray,
    pseudotime: np.ndarray,
    closest_vertex: np.ndarray,
) -> np.ndarray:
    """Sort centroid branch-point indices by the mean pseudotime of their cells.

    R orders branch points by their position along the trajectory; in the
    DDRTree path this is implicitly the order in which ``branch_points``
    were populated from the principal graph. We approximate with the mean
    pseudotime of cells whose ``closest_vertex`` is the branch point.
    """
    means = np.zeros(branch_points.size, dtype=float)
    for i, v in enumerate(branch_points):
        cells = closest_vertex == int(v)
        means[i] = float(pseudotime[cells].mean()) if cells.any() else np.inf
    return branch_points[np.argsort(means, kind="stable")]


def _compute_paths_to_root(
    adata: AnnData,
    adj_vert: Sequence[Sequence[int]],
    closest_vertex: np.ndarray,
    branch_states: Optional[Sequence[int]],
    branch_point: int,
) -> dict[str, list[int]]:
    """Identify the two branch paths, at the **centroid** level, and map
    each path back to the cells whose ``closest_vertex`` sits on it.

    Mirrors R ``BEAM.R:73-137``:
      1. Derive the root Y-vertex: the cells in the root state → union of
         their ``closest_vertex`` IDs → restrict the centroid MST to those
         IDs → pick the first degree-1 node as the root Y-vertex.
      2. For ``branch_states`` mode (R:73-104): for each leaf state, pick
         a tip Y-vertex among its cells' centroids, take the shortest
         path to the root Y-vertex, and materialise the path as the union
         of cells whose centroid is on that path.
      3. For ``branch_point`` mode (R:106-137): pick the ``k``-th centroid
         with degree>2, BFS its neighbours with the branch point removed,
         and attach each non-root component (plus the path to the root) to
         the branch keyed by the neighbour centroid ID.
    """
    state = get_state(adata)
    aux = state["aux_ordering"]["DDRTree"]
    root_cell_name = str(aux["root_cell"])
    cell_names = adata.obs_names.to_numpy()
    name_to_idx = {n: i for i, n in enumerate(cell_names)}
    root_cell_idx = name_to_idx[root_cell_name]
    root_state_val = int(adata.obs["State"].iloc[root_cell_idx])
    state_col = adata.obs["State"].astype(int).to_numpy()
    deg = _degree(adj_vert)

    # Root Y-vertex: first degree-1 centroid among those containing
    # root-state cells (R:67 derives ``root_cell`` via the restricted
    # centroid MST).
    root_state_cells = np.flatnonzero(state_col == root_state_val)
    root_vertices = np.unique(closest_vertex[root_state_cells])
    root_tip_candidates = root_vertices[deg[root_vertices] == 1]
    if root_tip_candidates.size > 0:
        root_y = int(root_tip_candidates[0])
    elif root_vertices.size > 0:
        root_y = int(root_vertices[0])
    else:
        root_y = int(aux.get("root_vertex", 0))

    paths_vertex: dict[str, list[int]] = {}
    if branch_states is not None:
        for leaf_state in branch_states:
            in_state = np.flatnonzero(state_col == int(leaf_state))
            if in_state.size == 0:
                raise RuntimeError(f"No cells in State == {leaf_state}")
            state_verts = np.unique(closest_vertex[in_state])
            tip_candidates = state_verts[deg[state_verts] == 1]
            tip = int(tip_candidates[0]) if tip_candidates.size else int(
                state_verts[0]
            )
            path = _path_in_tree(adj_vert, tip, root_y)
            if not path:
                raise RuntimeError(
                    f"Could not find a path from State {leaf_state} to root"
                )
            paths_vertex[str(int(leaf_state))] = path
    else:
        branch_points_vert = np.asarray(aux["branch_points"], dtype=np.int64)
        if branch_points_vert.size == 0:
            raise RuntimeError(
                "No branch points detected; ensure order_cells found a "
                "branching trajectory."
            )
        pt_all = adata.obs["Pseudotime"].to_numpy(dtype=float)
        branch_points_vert = _sort_branch_points_by_vertex_pseudotime(
            branch_points_vert, pt_all, closest_vertex,
        )
        if int(branch_point) < 1 or int(branch_point) > branch_points_vert.size:
            raise RuntimeError(
                f"branch_point={branch_point} is out of range "
                f"(only {branch_points_vert.size} branch points)."
            )
        branch_y = int(branch_points_vert[int(branch_point) - 1])
        path_to_anc_y = set(_path_in_tree(adj_vert, branch_y, root_y))
        # Iterate the immediate neighbours in sorted order to give
        # deterministic A/B labelling (R's iteration order is
        # igraph-dependent; sorting on the centroid index is a stable
        # canonicalisation that avoids the COO-order bug.)
        #
        # R keys the branch by the centroid's igraph vertex name
        # (``BEAM.R:132`` uses ``backbone_nei``, which under DDRTree is
        # ``V(minSpanningTree)$name`` == ``Y_1..Y_K``). Mirror that by
        # converting the 0-based Python centroid index to ``Y_{i+1}``.
        for nbr in sorted(adj_vert[branch_y]):
            descendants = _bfs_component(adj_vert, nbr, removed=branch_y)
            if root_y in descendants:
                continue
            combined = path_to_anc_y | {branch_y} | descendants
            paths_vertex[f"Y_{int(nbr) + 1}"] = sorted(combined)

    if len(paths_vertex) != 2:
        raise RuntimeError(
            f"Expected 2 branches, got {len(paths_vertex)} "
            "(buildBranchCellDataSet supports exactly two branches)."
        )

    paths: dict[str, list[int]] = {}
    for key, vert_path in paths_vertex.items():
        vert_set = set(vert_path)
        cells_on_path = np.flatnonzero(
            np.isin(closest_vertex, list(vert_set))
        )
        paths[key] = sorted(cells_on_path.tolist())
    return paths


def _rescale_pseudotime(
    pt: np.ndarray,
    paths_idx: dict[str, list[int]],
    common_ancestor: list[int],
) -> np.ndarray:
    """R's stretch branch: rescale each branch to a shared [0, 100] range."""
    pt_out = pt.astype(float).copy()
    max_pt = -1.0
    for path in paths_idx.values():
        if path:
            max_pt = max(max_pt, float(pt_out[path].max()))
    branch_pt = float(pt_out[common_ancestor].max()) if common_ancestor else 0.0
    for path in paths_idx.values():
        if not path:
            continue
        max_on_path = float(pt_out[path].max())
        denom = max_on_path - branch_pt
        if denom == 0.0:
            continue
        factor = (max_pt - branch_pt) / denom
        if not np.isfinite(factor):
            continue
        branch_only = [c for c in path if c not in common_ancestor]
        if branch_only:
            pt_out[branch_only] = (
                (pt_out[branch_only] - branch_pt) * factor + branch_pt
            )
    if max_pt > 0:
        pt_out = 100.0 * pt_out / max_pt
    return pt_out


def _duplicate_progenitors(
    adata: AnnData,
    paths_idx: dict[str, list[int]],
    common_ancestor: list[int],
    branch_labels: Optional[Sequence[str]],
    pseudotime: np.ndarray,
) -> AnnData:
    """Build a new AnnData with progenitor cells duplicated across branches."""
    cell_names = adata.obs_names.to_numpy()
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    keys = list(paths_idx.keys())
    if branch_labels is not None:
        if len(branch_labels) != 2:
            raise RuntimeError("branch_labels must have exactly two entries")
        keys = [str(b) for b in branch_labels]

    expr_blocks: list[np.ndarray] = []
    obs_blocks: list[pd.DataFrame] = []

    n_anc = len(common_ancestor)
    orig_obs = adata.obs.copy()
    orig_obs["Pseudotime"] = pseudotime
    orig_obs["original_cell_id"] = cell_names
    orig_obs["State"] = orig_obs["State"].astype(str)

    for i, key in enumerate(keys, start=1):
        path = paths_idx[list(paths_idx.keys())[i - 1]]
        branch_only = [c for c in path if c not in common_ancestor]

        dup_names = [f"duplicate_{i}_{j+1}" for j in range(n_anc)]
        anc_expr = X[common_ancestor, :]
        branch_expr = X[branch_only, :] if branch_only else np.empty((0, X.shape[1]))
        block_expr = np.vstack([anc_expr, branch_expr])

        anc_obs = orig_obs.iloc[common_ancestor].copy()
        anc_obs.index = pd.Index(dup_names)
        branch_obs = orig_obs.iloc[branch_only].copy() if branch_only else orig_obs.iloc[:0].copy()
        block_obs = pd.concat([anc_obs, branch_obs], axis=0)
        block_obs["Branch"] = key
        expr_blocks.append(block_expr)
        obs_blocks.append(block_obs)

    combined_expr = np.vstack(expr_blocks)
    combined_obs = pd.concat(obs_blocks, axis=0)
    combined_obs["Branch"] = pd.Categorical(
        combined_obs["Branch"].astype(str), categories=keys,
    )
    combined_obs["State"] = combined_obs["State"].astype("category")

    new = AnnData(
        X=combined_expr,
        obs=combined_obs,
        var=adata.var.copy(),
    )
    new.uns["monocle2"] = dict(get_state(adata))
    return new


def build_branch_cell_dataset(
    adata: AnnData,
    progenitor_method: str = "sequential_split",
    branch_states: Optional[Sequence[int]] = None,
    branch_point: int = 1,
    branch_labels: Optional[Sequence[str]] = None,
    stretch: bool = True,
) -> AnnData:
    """Construct an AnnData where cells are assigned to one of two branches.

    Ports R's ``buildBranchCellDataSet``. Progenitor cells (cells on the
    shared path to the root) are either duplicated across both branches
    (``progenitor_method="duplicate"``) or split alternately by pseudotime
    order (``progenitor_method="sequential_split"``). The returned AnnData
    gains a ``Branch`` categorical column in ``obs``; when ``stretch`` is
    true, ``Pseudotime`` is rescaled so both branches span ``[0, 100]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must have been processed with :func:`~monocle2py.order_cells`.
    progenitor_method : {"sequential_split", "duplicate"}, default "sequential_split"
        Matches R's ``buildBranchCellDataSet``: the formal default
        ``c('sequential_split', 'duplicate')`` resolves to ``sequential_split``
        because R's ``if(x == 'duplicate')`` uses only the vector's first
        element (``BEAM.R:26``, verified in ``monocle2`` R env).
    branch_states : sequence of int, optional
        Two state IDs specifying the two leaf branches. Overrides
        ``branch_point`` when supplied.
    branch_point : int, default 1
        Which branch point to split at (1-based, ordered by pseudotime).
    branch_labels : sequence of str, optional
        Two names to assign to the branches; defaults to the internal keys.
    stretch : bool, default True
        Rescale each branch's pseudotime to ``[0, 100]``.
    """
    if progenitor_method not in ("duplicate", "sequential_split"):
        raise ValueError(
            f"progenitor_method must be 'duplicate' or 'sequential_split', "
            f"got {progenitor_method!r}"
        )
    if "State" not in adata.obs or "Pseudotime" not in adata.obs:
        raise RuntimeError(
            "Please first order the cells in pseudotime using order_cells()"
        )
    if branch_states is None and branch_point is None:
        raise RuntimeError(
            "Please specify branch_point or branch_states"
        )

    adj_vert, closest_vertex = _centroid_mst_adj(adata)
    paths_idx = _compute_paths_to_root(
        adata, adj_vert, closest_vertex, branch_states, int(branch_point),
    )

    path_keys = list(paths_idx.keys())
    common_ancestor = sorted(
        set(paths_idx[path_keys[0]]) & set(paths_idx[path_keys[1]])
    )
    if not common_ancestor:
        raise RuntimeError(
            "common ancestors between selected State values on path to root State"
        )

    pt = adata.obs["Pseudotime"].to_numpy(dtype=float)
    if stretch:
        pt = _rescale_pseudotime(pt, paths_idx, common_ancestor)

    if progenitor_method == "duplicate":
        return _duplicate_progenitors(
            adata, paths_idx, common_ancestor, branch_labels, pt,
        )
    return _sequential_split(
        adata, paths_idx, common_ancestor, branch_labels, pt,
    )


def _sequential_split(
    adata: AnnData,
    paths_idx: dict[str, list[int]],
    common_ancestor: list[int],
    branch_labels: Optional[Sequence[str]],
    pseudotime: np.ndarray,
) -> AnnData:
    """R's ``sequential_split`` path: alternate progenitors across branches."""
    cell_names = adata.obs_names.to_numpy()
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    keys = list(paths_idx.keys())
    if branch_labels is not None:
        if len(branch_labels) != 2:
            raise RuntimeError("branch_labels must have exactly two entries")
        keys = [str(b) for b in branch_labels]

    all_cells = sorted(set(paths_idx[list(paths_idx)[0]]) | set(paths_idx[list(paths_idx)[1]]))
    obs = adata.obs.iloc[all_cells].copy()
    obs["Pseudotime"] = pseudotime[all_cells]
    obs["original_cell_id"] = cell_names[all_cells]

    # Assign Branch by alternating through progenitors sorted by pseudotime.
    branch_col = np.array([keys[0]] * len(all_cells), dtype=object)
    anc_in_subset = [all_cells.index(c) for c in common_ancestor]
    order = np.argsort(pseudotime[common_ancestor])
    for k, idx_in_anc in enumerate(order):
        target = keys[k % 2]
        branch_col[anc_in_subset[idx_in_anc]] = target
    for k, branch_key in enumerate(keys):
        original = list(paths_idx.keys())[k]
        branch_only = [c for c in paths_idx[original] if c not in common_ancestor]
        for c in branch_only:
            branch_col[all_cells.index(c)] = branch_key
    obs["Branch"] = pd.Categorical(branch_col, categories=keys)

    # Duplicate the root cell at pseudotime 0 onto branch 2.
    zero_root_local = anc_in_subset[order[0]]
    dup_row = obs.iloc[[zero_root_local]].copy()
    dup_row.index = pd.Index(["duplicate_root"])
    dup_row["Branch"] = pd.Categorical([keys[1]], categories=keys)
    obs = pd.concat([obs, dup_row], axis=0)

    expr = np.vstack([X[all_cells, :], X[all_cells[zero_root_local], :][None, :]])
    new = AnnData(X=expr, obs=obs, var=adata.var.copy())
    new.uns["monocle2"] = dict(get_state(adata))
    return new


def branch_test(
    adata: AnnData,
    full_model_formula_str: str = "~sm.ns(Pseudotime, df=3)*Branch",
    reduced_model_formula_str: str = "~sm.ns(Pseudotime, df=3)",
    branch_states: Optional[Sequence[int]] = None,
    branch_point: int = 1,
    branch_labels: Optional[Sequence[str]] = None,
    progenitor_method: str = "sequential_split",
    stretch: bool = True,
    relative_expr: bool = True,
    cores: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Test each gene for branch-dependent expression.

    Builds a branch-assigned AnnData via :func:`build_branch_cell_dataset`,
    then compares the two formulas via a per-gene LRT using the Slice 6
    machinery. If ``"Branch"`` is not referenced in the full formula, the
    test runs directly on the input AnnData without duplication.

    ``progenitor_method`` and ``stretch`` mirror R's ``...`` forwarding
    (``BEAM.R:331-333``): ``branchTest`` passes through to
    ``buildBranchCellDataSet``, so users can toggle between
    ``"sequential_split"`` (R default) and ``"duplicate"``.
    """
    if "Branch" in formula_terms(full_model_formula_str):
        subset = build_branch_cell_dataset(
            adata,
            progenitor_method=progenitor_method,
            branch_states=branch_states,
            branch_point=branch_point,
            branch_labels=branch_labels,
            stretch=stretch,
        )
    else:
        subset = adata
    return differential_gene_test(
        subset,
        full_model_formula_str=full_model_formula_str,
        reduced_model_formula_str=reduced_model_formula_str,
        relative_expr=relative_expr,
        cores=cores,
        verbose=verbose,
    )


def beam(
    adata: AnnData,
    full_model_formula_str: str = "~sm.ns(Pseudotime, df=3)*Branch",
    reduced_model_formula_str: str = "~sm.ns(Pseudotime, df=3)",
    branch_states: Optional[Sequence[int]] = None,
    branch_point: int = 1,
    branch_labels: Optional[Sequence[str]] = None,
    progenitor_method: str = "sequential_split",
    stretch: bool = True,
    relative_expr: bool = True,
    cores: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Identify genes with branch-dependent expression.

    Calls :func:`branch_test` and returns its DataFrame joined with the
    per-gene ``var`` columns from *adata*. ``progenitor_method`` and
    ``stretch`` are forwarded to :func:`build_branch_cell_dataset`,
    mirroring R's ``BEAM(..., ...)`` kwargs passthrough
    (``BEAM.R:822-840``).
    """
    res = branch_test(
        adata,
        full_model_formula_str=full_model_formula_str,
        reduced_model_formula_str=reduced_model_formula_str,
        branch_states=branch_states,
        branch_point=branch_point,
        branch_labels=branch_labels,
        progenitor_method=progenitor_method,
        stretch=stretch,
        relative_expr=relative_expr,
        cores=cores,
        verbose=verbose,
    )
    head = res[["status", "family", "pval", "qval"]]
    fd = adata.var.copy()
    fd.index = fd.index.astype(str)
    head = head.join(fd, how="left")
    return head


def cal_ilrs(
    adata: AnnData,
    trend_formula: str = "~sm.ns(Pseudotime, df=3)*Branch",
    branch_point: int = 1,
    trajectory_states: Optional[Sequence[int]] = None,
    relative_expr: bool = True,
    stretch: bool = True,
    cores: int = 1,
    ILRs_limit: float = 3.0,
    label_by_short_name: bool = True,
    useVST: bool = False,
    round_exprs: bool = False,
    output_type: str = "all",
    branch_labels: Optional[Sequence[str]] = None,
    n_points: int = 100,
    return_all: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Instantaneous log ratio between two branches.

    Builds a branch CDS, fits the trend formula per gene, evaluates on a
    pseudotime grid for each branch, and returns ``log2((A+1)/(B+1))``
    (or the VST-stabilized difference) clipped to ``[-ILRs_limit, ILRs_limit]``.
    """
    if trajectory_states is not None and len(trajectory_states) != 2:
        raise ValueError(
            "cal_ilrs only supports the calculation of ILRs between TWO branches"
        )

    subset = build_branch_cell_dataset(
        adata,
        progenitor_method="duplicate",
        branch_states=trajectory_states,
        branch_point=branch_point,
        branch_labels=branch_labels,
        stretch=stretch,
    )
    overlap_rng = (0.0, float(subset.obs["Pseudotime"].max()))
    trajectory_labels = list(subset.obs["Branch"].cat.categories)
    if len(trajectory_labels) != 2:
        raise RuntimeError(
            f"Expected two branches after subset, got {trajectory_labels}"
        )

    formula_vars = formula_terms(trend_formula)
    branch_var = next(
        (v for v in formula_vars if v in subset.obs.columns and v != "Pseudotime"),
        "Branch",
    )
    grid = np.linspace(overlap_rng[0], overlap_rng[1], n_points)
    new_A = pd.DataFrame({
        "Pseudotime": grid,
        branch_var: pd.Categorical(
            [trajectory_labels[0]] * n_points, categories=trajectory_labels,
        ),
    }, index=[f"A_{i}" for i in range(n_points)])
    new_B = pd.DataFrame({
        "Pseudotime": grid,
        branch_var: pd.Categorical(
            [trajectory_labels[1]] * n_points, categories=trajectory_labels,
        ),
    }, index=[f"B_{i}" for i in range(n_points)])
    new_data = pd.concat([new_A, new_B], axis=0)

    curves = gen_smooth_curves(
        subset, new_data, trend_formula=trend_formula,
        relative_expr=relative_expr, cores=cores,
    )
    mat_A = curves.iloc[:, :n_points]
    mat_B = curves.iloc[:, n_points:]

    if useVST:
        mat_A = pd.DataFrame(
            vst_exprs(adata, expr_matrix=mat_A.to_numpy(), round_vals=round_exprs),
            index=mat_A.index, columns=mat_A.columns,
        )
        mat_B = pd.DataFrame(
            vst_exprs(adata, expr_matrix=mat_B.to_numpy(), round_vals=round_exprs),
            index=mat_B.index, columns=mat_B.columns,
        )
        logfc = mat_A.to_numpy() - mat_B.to_numpy()
    else:
        logfc = np.log2((mat_A.to_numpy() + 1.0) / (mat_B.to_numpy() + 1.0))

    logfc = np.clip(logfc, -ILRs_limit, ILRs_limit)
    gene_index = (
        adata.var["gene_short_name"].astype(str).to_numpy()
        if label_by_short_name and "gene_short_name" in adata.var.columns
        else adata.var_names.astype(str).to_numpy()
    )
    gene_index = gene_index[: logfc.shape[0]]
    logfc_df = pd.DataFrame(logfc, index=gene_index, columns=mat_A.columns)

    if output_type == "after_bifurcation":
        pt_bif = float(adata.obs.loc[
            adata.obs["State"].astype(str).isin([str(s) for s in (trajectory_states or trajectory_labels)]),
            "Pseudotime",
        ].min())
        if np.isfinite(pt_bif) and overlap_rng[1] > 0:
            bif_index = int(100 * pt_bif / overlap_rng[1])
            bif_index = max(0, min(bif_index, n_points - 1))
            logfc_df = logfc_df.iloc[:, bif_index:]

    if return_all:
        raw_div = mat_A.to_numpy() - mat_B.to_numpy()
        row_max_abs = np.nanmax(np.abs(raw_div), axis=1, keepdims=True)
        row_max_abs = np.where(row_max_abs == 0, np.nan, row_max_abs)
        norm_div = raw_div / row_max_abs

        log_raw = np.log2(
            (mat_A.to_numpy() + 0.1) / (mat_B.to_numpy() + 0.1)
        )
        row_max_log = np.nanmax(np.abs(log_raw), axis=1, keepdims=True)
        row_max_log = np.where(row_max_log == 0, np.nan, row_max_log)
        norm_logfc = logfc / row_max_log

        return {
            "str_logfc_df": logfc_df,
            "norm_str_logfc_df": pd.DataFrame(
                norm_logfc, index=gene_index, columns=mat_A.columns,
            ),
            "str_norm_div_df": pd.DataFrame(
                norm_div, index=gene_index, columns=mat_A.columns,
            ),
            "str_raw_div_df": pd.DataFrame(
                raw_div, index=gene_index, columns=mat_A.columns,
            ),
            "str_branchA_expression_curve_matrix": pd.DataFrame(
                mat_A.to_numpy(), index=gene_index, columns=mat_A.columns,
            ),
            "str_branchB_expression_curve_matrix": pd.DataFrame(
                mat_B.to_numpy(), index=gene_index, columns=mat_B.columns,
            ),
        }
    return logfc_df
