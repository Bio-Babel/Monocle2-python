"""Internal helpers shared by the monocle2py plot functions.

Centralises the bits that recur across the R `plot_*` family: rotating a 2D
backbone for the trajectory views, melting the expression matrix to a long
data-frame indexed by ``f_id``/``Cell``, and resolving display labels via
``gene_short_name`` (falling back to the feature id).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from .._uns import (
    SIZE_FACTOR_COL,
    get_disp_fit_info,
    get_expression_family,
    get_lower_detection_limit,
)
from ..preprocess import vst_exprs

__all__ = [
    "rotation_matrix",
    "as_dense",
    "expression_long_df",
    "feature_label_column",
    "size_factor_normalised",
    "_vst_or_log",
]


def _vst_or_log(
    adata: AnnData, m: pd.DataFrame, norm_method: str,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Apply log10 + pseudocount or VST to a ``genes × cells`` DataFrame.

    Parameters
    ----------
    adata : AnnData
    m : pandas.DataFrame
        ``genes × cells`` matrix (matches the R heatmap convention; the
        helper transposes internally for ``vst_exprs`` which wants
        ``cells × genes``).
    norm_method : {"log", "vstExprs"}
    pseudocount : float, default 1.0
        Added before ``log10``. Matches R's hardcoded ``pseudocount <- 1``
        in the heatmap sibling functions (``plotting.R:1145, 2446``).

    Raises
    ------
    ValueError
        ``norm_method`` is neither ``"log"`` nor ``"vstExprs"``.
    RuntimeError
        ``norm_method == "vstExprs"`` but no ``"blind"`` dispersion fit
        is registered. R's heatmap code silently skips the vstExprs
        branch when ``disp_func`` is NULL (``plotting.R:1162``), leaving
        the matrix raw; this port raises loudly per the project's
        meta-principle so callers learn to call ``estimate_dispersions``
        rather than receive a silently-wrong plot.
    """
    if norm_method == "vstExprs":
        info = get_disp_fit_info(adata, "blind")
        if info is None or info.get("disp_func") is None:
            raise RuntimeError(
                "norm_method='vstExprs' requires a prior "
                "estimate_dispersions(adata) call. Either run "
                "estimate_dispersions first or use norm_method='log'."
            )
        arr = vst_exprs(adata, expr_matrix=m.to_numpy().T).T
        return pd.DataFrame(arr, index=m.index, columns=m.columns)
    if norm_method == "log":
        return np.log10(m + pseudocount)
    raise ValueError(
        f"norm_method must be 'log' or 'vstExprs'; got {norm_method!r}"
    )


def rotation_matrix(theta_deg: float) -> np.ndarray:
    """2x2 rotation matrix for ``theta_deg`` degrees (counter-clockwise)."""
    theta = float(theta_deg) / 180.0 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def as_dense(matrix) -> np.ndarray:
    """Materialise *matrix* (sparse or dense) as a contiguous numpy array."""
    if issparse(matrix):
        return np.asarray(matrix.todense())
    return np.asarray(matrix)


def size_factor_normalised(adata: AnnData, X: np.ndarray) -> np.ndarray:
    """Divide each cell's expression by its ``Size_Factor`` (cells = rows)."""
    if SIZE_FACTOR_COL not in adata.obs.columns:
        raise RuntimeError(
            "estimate_size_factors must be called before relative_expr=True."
        )
    sf = adata.obs[SIZE_FACTOR_COL].to_numpy(dtype=float)
    return X / sf[:, None]


def expression_long_df(
    adata: AnnData,
    relative_expr: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Melt ``adata.X`` to a long ``(f_id, Cell, expression)`` data-frame.

    Mirrors R's per-cell normalisation in ``plot_genes_in_pseudotime``: for
    integer (negative-binomial) families divide by ``Size_Factor`` then round;
    for continuous families take the raw matrix.

    Returns
    -------
    tuple of (DataFrame, bool)
        The long-format frame and a flag indicating whether the data are
        treated as integer-valued (negative-binomial style) by R.
    """
    family = get_expression_family(adata)
    integer_expression = family.vfamily in {"negbinomial", "negbinomial.size"}

    X = as_dense(adata.X).astype(float)  # cells x genes
    if integer_expression and relative_expr:
        X = size_factor_normalised(adata, X)
        X = np.round(X)
    rows = []
    cell_names = adata.obs_names.to_numpy()
    gene_names = adata.var_names.to_numpy()
    for j, gene in enumerate(gene_names):
        col = X[:, j]
        rows.append(pd.DataFrame({
            "f_id": np.repeat(str(gene), len(cell_names)),
            "Cell": cell_names,
            "expression": col,
        }))
    return pd.concat(rows, ignore_index=True), integer_expression


def feature_label_column(
    var_df: pd.DataFrame, label_by_short_name: bool,
) -> pd.Series:
    """Return a per-feature display label.

    When ``label_by_short_name`` is True and a ``gene_short_name`` column is
    present, use it (falling back to the feature id where missing); otherwise
    use the feature id directly.
    """
    if label_by_short_name and "gene_short_name" in var_df.columns:
        labels = var_df["gene_short_name"].astype(object)
        labels = labels.where(labels.notna(), pd.Series(var_df.index, index=var_df.index))
        return labels.astype(str)
    return pd.Series(var_df.index.astype(str), index=var_df.index)


def lower_detection_limit_default(
    adata: AnnData, min_expr: float | None,
) -> float:
    """Return ``min_expr`` if supplied, else the AnnData's lowerDetectionLimit."""
    if min_expr is not None:
        return float(min_expr)
    try:
        return float(get_lower_detection_limit(adata))
    except KeyError:
        return 0.1
