"""Heatmap plot family: ``plot_pseudotime_heatmap``,
``plot_genes_branched_heatmap``, and ``plot_multiple_branches_heatmap``.

All three follow the same basic recipe: fit per-gene smooth expression
curves on a 100-point pseudotime grid via :func:`gen_smooth_curves`,
log10/VST-normalise, row-centre+scale, clip to ``[scale_min, scale_max]``,
correlation-distance cluster the rows, and feed the result into
``pheatmap.pheatmap``. The branched variants additionally split or stitch
multiple paths through the trajectory.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from pheatmap import pheatmap as _pheatmap
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from .._uns import get_disp_fit_info
from ..beam import build_branch_cell_dataset
from ..differential import gen_smooth_curves
from ..preprocess import vst_exprs
from ._helpers import feature_label_column
from .branches import _branch_path_cells

__all__ = [
    "plot_pseudotime_heatmap",
    "plot_genes_branched_heatmap",
    "plot_multiple_branches_heatmap",
]


def _table_ramp(n: int, mid: float, sill: float, base: float = 1.0,
                height: float = 1.0) -> np.ndarray:
    """Port of ``colorRamps::table.ramp``."""
    y = np.zeros(n, dtype=float)
    sill_min = max(1, int(round((n - 1) * (mid - sill / 2))) + 1)
    sill_max = min(n, int(round((n - 1) * (mid + sill / 2))) + 1)
    y[sill_min - 1:sill_max] = 1.0
    base_min = int(round((n - 1) * (mid - base / 2))) + 1
    base_max = int(round((n - 1) * (mid + base / 2))) + 1

    xi = np.arange(base_min, sill_min + 1)
    yi = np.linspace(0.0, 1.0, num=len(xi))
    mask = (xi > 0) & (xi <= n)
    y[xi[mask] - 1] = yi[mask]

    xi = np.arange(sill_max, base_max + 1)
    yi = np.linspace(1.0, 0.0, num=len(xi))
    mask = (xi > 0) & (xi <= n)
    y[xi[mask] - 1] = yi[mask]
    return height * y


def _blue2green2red(n: int) -> list[str]:
    """Port of ``colorRamps::matlab.like2(n)`` (= ``rgb.tables`` defaults)."""
    rr = _table_ramp(n, mid=0.8, sill=0.2, base=1.0)
    gr = _table_ramp(n, mid=0.5, sill=0.4, base=0.8)
    br = _table_ramp(n, mid=0.2, sill=0.2, base=1.0)
    rgb = np.clip(np.stack([rr, gr, br], axis=1), 0.0, 1.0)
    return [
        "#%02X%02X%02X" % (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        for r, g, b in rgb
    ]


def _correlation_distance(matrix: np.ndarray) -> np.ndarray:
    """Return condensed ``(1 - cor(t(matrix))) / 2`` row distances."""
    if matrix.shape[0] < 2:
        return np.zeros(0, dtype=float)
    corr = np.corrcoef(matrix)
    dist = (1.0 - corr) / 2.0
    dist[~np.isfinite(dist)] = 1.0
    np.fill_diagonal(dist, 0.0)
    return squareform(dist, checks=False)


def _row_center_scale(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, ddof=1, keepdims=True)
    centered = matrix - mean
    out = np.divide(
        centered, std, out=np.zeros_like(centered), where=std > 0,
    )
    return out


def _vst_or_log(
    adata: AnnData, m: pd.DataFrame, norm_method: str,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    if norm_method == "vstExprs":
        info = get_disp_fit_info(adata, "blind")
        if info is not None and info.get("disp_func") is not None:
            arr = vst_exprs(adata, expr_matrix=m.to_numpy().T).T
            return pd.DataFrame(arr, index=m.index, columns=m.columns)
    return np.log10(m + pseudocount)


def _clip_and_filter(
    m: pd.DataFrame, scale_min: float, scale_max: float,
) -> pd.DataFrame:
    arr = m.to_numpy(dtype=float)
    nonzero_sd = arr.std(axis=1, ddof=1) > 0
    arr = arr[nonzero_sd]
    index = m.index[nonzero_sd]
    arr = _row_center_scale(arr)
    arr[np.isnan(arr)] = 0.0
    arr = np.clip(arr, scale_min, scale_max)
    return pd.DataFrame(arr, index=index, columns=m.columns)


def _gene_label_index(adata: AnnData, ids: Sequence[str], use_short: bool) -> list[str]:
    if not use_short:
        return [str(i) for i in ids]
    labels = feature_label_column(adata.var, label_by_short_name=True)
    return [str(labels.get(i, i)) for i in ids]


def plot_pseudotime_heatmap(
    cds_subset: AnnData,
    cluster_rows: bool = True,
    hclust_method: str = "ward.D2",
    num_clusters: int = 6,
    hmcols: Sequence[str] | None = None,
    add_annotation_row: pd.DataFrame | None = None,
    add_annotation_col: pd.DataFrame | None = None,
    show_rownames: bool = False,
    use_gene_short_name: bool = True,
    norm_method: str = "log",
    scale_max: float = 3.0,
    scale_min: float = -3.0,
    trend_formula: str = "~sm.ns(Pseudotime, df=3)",
    return_heatmap: bool = False,
    cores: int = 1,
):
    """Port of R's ``plot_pseudotime_heatmap``.

    Smooths each gene's expression on a 100-point pseudotime grid, log10 or
    VST normalises, row-scales, and renders a row-clustered heatmap with
    ``num_clusters`` cuts.
    """
    if "Pseudotime" not in cds_subset.obs.columns:
        raise RuntimeError(
            "Pseudotime missing — call order_cells before plot_pseudotime_heatmap."
        )
    if norm_method not in ("log", "vstExprs"):
        raise ValueError("norm_method must be 'log' or 'vstExprs'.")

    num_clusters = int(min(num_clusters, cds_subset.n_vars))
    pt = cds_subset.obs["Pseudotime"].to_numpy(dtype=float)
    new_data = pd.DataFrame({
        "Pseudotime": np.linspace(pt.min(), pt.max(), num=100),
    })
    m = gen_smooth_curves(
        cds_subset, new_data=new_data, trend_formula=trend_formula,
        relative_expr=True, cores=cores,
    )
    nonzero = m.sum(axis=1) != 0
    m = m.loc[nonzero]
    m = _vst_or_log(cds_subset, m, norm_method)
    heatmap_matrix = _clip_and_filter(m, scale_min, scale_max)

    row_dist = _correlation_distance(heatmap_matrix.to_numpy())

    if hmcols is None:
        bks = np.arange(-3.1, 3.1 + 1e-9, 0.1)
        colors: list[str] | Sequence[str] = _blue2green2red(len(bks) - 1)
    else:
        bks = np.linspace(-3.1, 3.1, num=len(hmcols))
        colors = list(hmcols)

    feature_label = _gene_label_index(
        cds_subset, heatmap_matrix.index.tolist(), use_gene_short_name,
    )

    annotation_row = None
    if cluster_rows:
        ph_first = _pheatmap(
            heatmap_matrix.to_numpy(),
            cluster_cols=False, cluster_rows=True,
            show_rownames=False, show_colnames=False,
            clustering_distance_rows=row_dist,
            clustering_method=hclust_method,
            cutree_rows=num_clusters,
            silent=True, breaks=bks, color=colors, border_color=None,
        )
        clusters = fcluster(
            ph_first.tree_row.linkage, t=num_clusters, criterion="maxclust",
        )
        annotation_row = pd.DataFrame(
            {"Cluster": pd.Categorical(clusters.astype(int))},
            index=feature_label,
        )

    if add_annotation_row is not None:
        if annotation_row is None:
            annotation_row = add_annotation_row.loc[heatmap_matrix.index].copy()
            annotation_row.index = feature_label
        else:
            extra = add_annotation_row.loc[heatmap_matrix.index].copy()
            extra.index = annotation_row.index
            annotation_row = pd.concat([annotation_row, extra], axis=1)

    annotation_col = None
    if add_annotation_col is not None:
        if len(add_annotation_col) != 100:
            raise ValueError(
                "add_annotation_col should have only 100 rows "
                "(check gen_smooth_curves before supplying the annotation data)."
            )
        annotation_col = add_annotation_col.copy()
        annotation_col.index = [str(i + 1) for i in range(100)]

    display_matrix = heatmap_matrix.copy()
    display_matrix.index = feature_label
    display_matrix.columns = [str(i + 1) for i in range(display_matrix.shape[1])]

    ph_res = _pheatmap(
        display_matrix,
        cluster_cols=False, cluster_rows=cluster_rows,
        show_rownames=show_rownames, show_colnames=False,
        clustering_distance_rows=row_dist,
        clustering_method=hclust_method,
        cutree_rows=num_clusters,
        annotation_row=annotation_row,
        annotation_col=annotation_col,
        treeheight_row=20,
        breaks=bks, fontsize=6,
        color=colors, border_color=None,
        silent=True,
    )
    if return_heatmap:
        return ph_res
    return ph_res


def plot_genes_branched_heatmap(
    cds_subset: AnnData,
    branch_point: int = 1,
    branch_states: Sequence[int] | None = None,
    branch_labels: Sequence[str] = ("Cell fate 1", "Cell fate 2"),
    cluster_rows: bool = True,
    hclust_method: str = "ward.D2",
    num_clusters: int = 6,
    hmcols: Sequence[str] | None = None,
    branch_colors: Sequence[str] = ("#979797", "#F05662", "#7990C8"),
    add_annotation_row: pd.DataFrame | None = None,
    add_annotation_col: pd.DataFrame | None = None,
    show_rownames: bool = False,
    use_gene_short_name: bool = True,
    scale_max: float = 3.0,
    scale_min: float = -3.0,
    norm_method: str = "log",
    trend_formula: str = "~sm.ns(Pseudotime, df=3) * Branch",
    return_heatmap: bool = False,
    cores: int = 1,
):
    """Port of R's ``plot_genes_branched_heatmap``.

    Builds a duplicated-progenitor branch CDS, smooths each gene along both
    branches on a 0..100 pseudotime grid, stitches the reversed first branch
    to the second branch (with a gap at the branch point), row-scales, and
    renders the heatmap with branch + cluster annotations.
    """
    if norm_method not in ("log", "vstExprs"):
        raise ValueError("norm_method must be 'log' or 'vstExprs'.")
    if "State" not in cds_subset.obs.columns:
        raise RuntimeError(
            "State column missing — call order_cells before plot_genes_branched_heatmap."
        )

    new_cds = build_branch_cell_dataset(
        cds_subset, branch_states=branch_states, branch_point=branch_point,
        progenitor_method="duplicate",
    )
    blind = get_disp_fit_info(cds_subset, "blind")
    if blind is not None:
        new_cds.uns.setdefault("monocle2", {})
        new_cds.uns["monocle2"].setdefault("disp_fit", {})
        new_cds.uns["monocle2"]["disp_fit"]["blind"] = blind

    if branch_states is None:
        progenitor_state = cds_subset.obs.loc[
            cds_subset.obs["Pseudotime"] == 0, "State"
        ].astype(int).iloc[0]
        all_states = cds_subset.obs["State"].astype(int).unique().tolist()
        branch_states = [s for s in all_states if s != int(progenitor_state)]

    branch_keys = list(new_cds.obs["Branch"].cat.categories)
    if len(branch_keys) < 2:
        raise RuntimeError("Branch CDS must contain at least two branches.")

    col_gap_ind = 101
    grid = np.linspace(0.0, 100.0, num=100)
    new_data = pd.concat([
        pd.DataFrame({
            "Pseudotime": grid,
            "Branch": pd.Categorical([branch_keys[0]] * 100, categories=branch_keys),
        }),
        pd.DataFrame({
            "Pseudotime": grid,
            "Branch": pd.Categorical([branch_keys[1]] * 100, categories=branch_keys),
        }),
    ], ignore_index=True)

    branch_ab = gen_smooth_curves(
        new_cds, new_data=new_data, trend_formula=trend_formula,
        relative_expr=True, cores=cores,
    )
    branch_a = branch_ab.iloc[:, :100]
    branch_b = branch_ab.iloc[:, 100:200]

    common_state = int(set(new_cds.obs["State"].astype(int).tolist())
                       .difference(int(s) for s in branch_states).pop())
    common_pt = new_cds.obs.loc[
        new_cds.obs["State"].astype(int) == common_state, "Pseudotime"
    ].to_numpy(dtype=float)
    if common_pt.size == 0:
        common_pt = np.array([0.0])
    branch_a_num = int(np.floor(common_pt.max()))
    branch_p_num = 100 - branch_a_num
    branch_b_num = branch_a_num

    if norm_method == "vstExprs":
        blind_info = get_disp_fit_info(cds_subset, "blind")
        if blind_info is not None and blind_info.get("disp_func") is not None:
            branch_a = pd.DataFrame(
                vst_exprs(cds_subset, expr_matrix=branch_a.to_numpy().T).T,
                index=branch_a.index, columns=branch_a.columns,
            )
            branch_b = pd.DataFrame(
                vst_exprs(cds_subset, expr_matrix=branch_b.to_numpy().T).T,
                index=branch_b.index, columns=branch_b.columns,
            )
    else:
        branch_a = np.log10(branch_a + 1.0)
        branch_b = np.log10(branch_b + 1.0)

    reversed_a = branch_a.iloc[:, ::-1]
    heatmap_df = pd.concat([reversed_a, branch_b], axis=1)
    heatmap_df.columns = [str(i + 1) for i in range(heatmap_df.shape[1])]

    arr = heatmap_df.to_numpy(dtype=float)
    nonzero_sd = arr.std(axis=1, ddof=1) > 0
    arr = arr[nonzero_sd]
    index = heatmap_df.index[nonzero_sd]
    arr = _row_center_scale(arr)
    arr[np.isnan(arr)] = 0.0
    arr = np.clip(arr, scale_min, scale_max)
    heatmap_matrix_ori = pd.DataFrame(arr, index=index, columns=heatmap_df.columns)

    finite_mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, col_gap_ind - 1])
    heatmap_matrix = heatmap_matrix_ori.iloc[finite_mask]

    row_dist = _correlation_distance(heatmap_matrix.to_numpy())

    exp_rng = (heatmap_matrix.to_numpy().min(), heatmap_matrix.to_numpy().max())
    bks = np.arange(exp_rng[0] - 0.1, exp_rng[1] + 0.1 + 1e-9, 0.1)
    colors: list[str] | Sequence[str]
    if hmcols is None:
        colors = _blue2green2red(len(bks) - 1)
    else:
        colors = list(hmcols)

    ph_first = _pheatmap(
        heatmap_matrix.to_numpy(),
        cluster_cols=False, cluster_rows=True,
        show_rownames=False, show_colnames=False,
        clustering_distance_rows=row_dist,
        clustering_method=hclust_method,
        cutree_rows=num_clusters,
        silent=True, breaks=bks, color=colors, border_color=None,
    )
    clusters = fcluster(
        ph_first.tree_row.linkage, t=num_clusters, criterion="maxclust",
    )

    feature_label = _gene_label_index(
        cds_subset, heatmap_matrix.index.tolist(), use_gene_short_name,
    )
    annotation_row = pd.DataFrame(
        {"Cluster": pd.Categorical(clusters.astype(int))},
        index=feature_label,
    )
    if add_annotation_row is not None:
        extra = add_annotation_row.loc[heatmap_matrix.index].copy()
        extra.index = feature_label
        annotation_row = pd.concat([annotation_row, extra], axis=1)

    annotation_col = pd.DataFrame(
        {
            "Cell Type": pd.Categorical(
                [branch_labels[0]] * branch_a_num
                + ["Pre-branch"] * (2 * branch_p_num)
                + [branch_labels[1]] * branch_b_num,
                categories=["Pre-branch", branch_labels[0], branch_labels[1]],
            )
        },
        index=[str(i + 1) for i in range(2 * (branch_a_num + branch_p_num))],
    )

    annotation_colors = {
        "Cell Type": {
            "Pre-branch": branch_colors[0],
            branch_labels[0]: branch_colors[1],
            branch_labels[1]: branch_colors[2],
        }
    }

    display_matrix = heatmap_matrix.copy()
    display_matrix.index = feature_label

    ph_res = _pheatmap(
        display_matrix,
        cluster_cols=False, cluster_rows=True,
        show_rownames=show_rownames, show_colnames=False,
        clustering_distance_rows=row_dist,
        clustering_method=hclust_method,
        cutree_rows=num_clusters,
        annotation_row=annotation_row,
        annotation_col=annotation_col,
        annotation_colors=annotation_colors,
        gaps_col=[col_gap_ind],
        treeheight_row=20,
        breaks=bks, fontsize=6,
        color=colors, border_color=None,
        silent=True,
    )
    if return_heatmap:
        return {
            "BranchA_exprs": branch_a,
            "BranchB_exprs": branch_b,
            "heatmap_matrix": heatmap_matrix,
            "heatmap_matrix_ori": heatmap_matrix_ori,
            "ph": ph_first,
            "col_gap_ind": col_gap_ind,
            "row_dist": row_dist,
            "hmcols": list(colors),
            "annotation_colors": annotation_colors,
            "annotation_row": annotation_row,
            "annotation_col": annotation_col,
            "ph_res": ph_res,
        }
    return ph_res


def plot_multiple_branches_heatmap(
    cds: AnnData,
    branches: Sequence[int],
    branches_name: Sequence[str] | None = None,
    cluster_rows: bool = True,
    hclust_method: str = "ward.D2",
    num_clusters: int = 6,
    hmcols: Sequence[str] | None = None,
    add_annotation_row: pd.DataFrame | None = None,
    add_annotation_col: pd.DataFrame | None = None,
    show_rownames: bool = False,
    use_gene_short_name: bool = True,
    norm_method: str = "vstExprs",
    scale_max: float = 3.0,
    scale_min: float = -3.0,
    trend_formula: str = "~sm.ns(Pseudotime, df=3)",
    return_heatmap: bool = False,
    cores: int = 1,
):
    """Port of R's ``plot_multiple_branches_heatmap``.

    For each terminal state in ``branches``, walks the centroid-MST root
    → branch tip path, smooths every gene's expression along the path on a
    100-point pseudotime grid, stitches the per-branch matrices side-by-side
    with column gaps, and renders the heatmap with Branch + Cluster
    annotations.
    """
    if norm_method not in ("log", "vstExprs"):
        raise ValueError("norm_method must be 'log' or 'vstExprs'.")
    if "State" not in cds.obs.columns:
        raise RuntimeError(
            "State column missing — call order_cells before plot_multiple_branches_heatmap."
        )
    states_set = set(cds.obs["State"].astype(int).unique().tolist())
    for b in branches:
        if int(b) not in states_set:
            raise RuntimeError(
                f"Branch {b} is not a State value in the trajectory."
            )
    branch_label = list(branches) if branches_name is None else list(branches_name)
    if branches_name is not None and len(branches_name) != len(branches):
        raise ValueError("branches_name should have the same length as branches.")

    branch_curves: list[pd.DataFrame] = []
    for branch_in in branches:
        path_idx = _branch_path_cells(cds, int(branch_in))
        if path_idx.size == 0:
            continue
        sub = cds[path_idx].copy()
        max_pt = float(sub.obs["Pseudotime"].max())
        new_data = pd.DataFrame({
            "Pseudotime": np.linspace(0.0, max_pt, num=100),
        })
        tmp = gen_smooth_curves(
            sub, new_data=new_data, trend_formula=trend_formula,
            relative_expr=True, cores=cores,
        )
        branch_curves.append(tmp)
    if not branch_curves:
        raise RuntimeError("No cells on any of the requested branches.")

    m = pd.concat(branch_curves, axis=1)
    m.columns = [str(i + 1) for i in range(m.shape[1])]
    nonzero = m.sum(axis=1) != 0
    m = m.loc[nonzero]
    m = _vst_or_log(cds, m, norm_method)
    heatmap_matrix = _clip_and_filter(m, scale_min, scale_max)

    row_dist = _correlation_distance(heatmap_matrix.to_numpy())

    if hmcols is None:
        bks = np.arange(-3.1, 3.1 + 1e-9, 0.1)
        colors: list[str] | Sequence[str] = _blue2green2red(len(bks) - 1)
    else:
        bks = np.linspace(-3.1, 3.1, num=len(hmcols))
        colors = list(hmcols)

    ph_first = _pheatmap(
        heatmap_matrix.to_numpy(),
        cluster_cols=False, cluster_rows=True,
        show_rownames=False, show_colnames=False,
        clustering_distance_rows=row_dist,
        clustering_method=hclust_method,
        cutree_rows=num_clusters,
        silent=True, breaks=bks, color=colors, border_color=None,
    )
    clusters = fcluster(
        ph_first.tree_row.linkage, t=num_clusters, criterion="maxclust",
    )

    feature_label = _gene_label_index(
        cds, heatmap_matrix.index.tolist(), use_gene_short_name,
    )
    annotation_row = pd.DataFrame(
        {"Cluster": pd.Categorical(clusters.astype(int))},
        index=feature_label,
    )
    if add_annotation_row is not None:
        extra = add_annotation_row.loc[heatmap_matrix.index].copy()
        extra.index = feature_label
        annotation_row = pd.concat([annotation_row, extra], axis=1)

    annotation_col = pd.DataFrame(
        {"Branch": pd.Categorical(
            np.repeat([str(b) for b in branch_label], 100),
            categories=[str(b) for b in branch_label],
        )},
        index=[str(i + 1) for i in range(100 * len(branch_label))],
    )
    col_gaps_ind = [(i + 1) * 100 for i in range(len(branches) - 1)]

    if not cluster_rows:
        annotation_row = None

    display_matrix = heatmap_matrix.copy()
    display_matrix.index = feature_label

    ph_res = _pheatmap(
        display_matrix,
        cluster_cols=False, cluster_rows=cluster_rows,
        show_rownames=show_rownames, show_colnames=False,
        clustering_distance_rows=row_dist,
        clustering_method=hclust_method,
        cutree_rows=num_clusters,
        annotation_row=annotation_row,
        annotation_col=annotation_col,
        gaps_col=col_gaps_ind,
        treeheight_row=20,
        breaks=bks, fontsize=12,
        color=colors, border_color=None,
        silent=True,
    )
    if return_heatmap:
        return ph_res
    return ph_res
