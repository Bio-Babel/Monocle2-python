"""Dimensionality reduction: ``reduce_dimension`` (DDRTree + tSNE).

Ports the DDRTree and tSNE branches of R's ``reduceDimension``. DDRTree is
delegated to the external ``ddrtree`` PyPI package; tSNE uses
``sklearn.manifold.TSNE`` after an optional PCA pre-projection. All monocle
state is written to ``adata.obsm["X_dr"]`` and ``adata.uns["monocle2"]``.
"""

from __future__ import annotations

from typing import Any

import ddrtree as _ddrtree
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ._internal._ddrtree_io import store_ddrtree_result
from ._uns import (
    SIZE_FACTOR_COL,
    ensure_state,
    get_expression_family,
    get_state,
)
from .preprocess import vst_exprs

__all__ = ["reduce_dimension", "normalize_expr_data", "cal_ncenter"]


def cal_ncenter(n_cells: int, n_cells_limit: int = 100) -> int:
    """R's ``cal_ncenter``: pick an MST vertex count for DDRTree.

    Parameters
    ----------
    n_cells : int
    n_cells_limit : int, default 100

    Returns
    -------
    int
        Recommended ``ncenter`` DDRTree argument.
    """
    ln = np.log(n_cells)
    ll = np.log(n_cells_limit)
    return int(round(2.0 * n_cells_limit * ln / (ln + ll)))


def _as_dense(X: Any) -> np.ndarray:
    return np.asarray(X.toarray()) if issparse(X) else np.asarray(X)


def _check_size_factors(adata: AnnData) -> None:
    family = get_expression_family(adata)
    if family.vfamily not in ("negbinomial", "negbinomial.size"):
        return
    if SIZE_FACTOR_COL not in adata.obs.columns:
        raise RuntimeError(
            "You must call estimate_size_factors before reduce_dimension."
        )
    sfs = adata.obs[SIZE_FACTOR_COL].to_numpy(dtype=float)
    if np.any(pd.isna(sfs)):
        raise RuntimeError("One or more cells has a size factor of NA.")


def normalize_expr_data(
    adata: AnnData,
    norm_method: str = "log",
    pseudo_expr: float = 1.0,
    relative_expr: bool = True,
) -> np.ndarray:
    """Port of ``normalize_expr_data``. Returns a genes x cells dense matrix.

    Implements the branches actually reached by DDRTree and tSNE:
    ``negbinomial{.size}`` with ``"log"`` / ``"vstExprs"`` / ``"none"`` and
    ``Tobit`` with ``"log"`` / ``"none"``. Uses ``var['use_for_ordering']``
    to restrict the matrix when set.
    """
    family = get_expression_family(adata)
    mask = None
    if "use_for_ordering" in adata.var.columns:
        ordering = adata.var["use_for_ordering"].astype(bool).to_numpy()
        if ordering.any():
            mask = ordering
    X = _as_dense(adata.X)  # cells x genes
    if mask is not None:
        X = X[:, mask]

    if family.vfamily in ("negbinomial", "negbinomial.size"):
        _check_size_factors(adata)
        if norm_method == "vstExprs":
            vst = vst_exprs(adata, expr_matrix=None, round_vals=False)
            if mask is not None:
                vst = vst[:, mask]
            FM = vst.T  # genes x cells
            return FM
        sfs = adata.obs[SIZE_FACTOR_COL].to_numpy(dtype=float)
        if norm_method == "log":
            if relative_expr:
                X = X / sfs[:, None]
            X = X + pseudo_expr
            X = np.log2(X)
        elif norm_method == "none":
            X = X / sfs[:, None]
            X = X + pseudo_expr
        else:
            raise ValueError(f"Unknown norm_method: {norm_method!r}")
    elif family.vfamily == "Tobit":
        X = X + pseudo_expr
        if norm_method == "log":
            X = np.log2(X)
        elif norm_method != "none":
            raise ValueError(
                "Tobit expression family only supports 'log' or 'none' "
                f"normalisation; got {norm_method!r}."
            )
    elif family.vfamily == "gaussianff":
        if norm_method != "none":
            raise ValueError(
                "gaussianff family only supports norm_method='none'."
            )
        X = X + pseudo_expr
    else:
        raise ValueError(f"Unsupported expression family: {family.vfamily!r}")

    return X.T  # return as genes x cells to match R orientation


def _drop_zero_sd(FM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows (genes) whose biased SD across cells is zero.

    Ports ``order_cells.R:1359-1361``:
    ``xsd <- sqrt(rowMeans((FM - rowMeans(FM))^2)); FM <- FM[xsd > 0, ]``.
    """
    row_mean = FM.mean(axis=1)
    row_sd = np.sqrt(((FM - row_mean[:, None]) ** 2).mean(axis=1))
    keep = row_sd > 0
    return FM[keep], keep


def _remove_batch_effects(
    FM: np.ndarray, adata: AnnData, residual_model_formula_str: str,
) -> np.ndarray:
    """Subtract non-intercept OLS effects from *FM* (genes x cells).

    Ports ``order_cells.R:1363-1375``::

        X.model_mat <- sparse.model.matrix(as.formula(residualModelFormulaStr),
                                           data = pData(cds))
        fit <- limma::lmFit(FM, X.model_mat)
        beta <- fit$coefficients[, -1, drop = FALSE]
        beta[is.na(beta)] <- 0
        FM <- as.matrix(FM) - beta %*% t(X.model_mat[, -1])

    ``limma::lmFit`` with default args is plain per-row OLS. The
    NumPy analogue uses :func:`numpy.linalg.lstsq` so rank-deficient
    design matrices degrade gracefully (R's ``beta[is.na(beta)] <- 0``
    handles the NA case; we use ``np.nan_to_num``).
    """
    from ._internal._formula import build_design_matrix

    X_model = build_design_matrix(
        residual_model_formula_str, adata.obs
    ).astype(np.float64)
    # Per-row OLS: β.T (p x G) = lstsq(X, FM.T)
    coef, _, _, _ = np.linalg.lstsq(X_model, FM.T, rcond=None)
    beta = np.nan_to_num(coef.T, nan=0.0)  # (G, p)
    if X_model.shape[1] <= 1:
        return FM
    return FM - beta[:, 1:] @ X_model[:, 1:].T


def _apply_scaling(FM: np.ndarray) -> np.ndarray:
    """Per-gene z-score (mean 0, ddof=1 SD) across cells.

    Mirrors R's ``scale(Matrix::t(FM))`` in ``order_cells.R:1378``.
    """
    row_mean = FM.mean(axis=1)
    row_sd = FM.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (FM - row_mean[:, None]) / row_sd[:, None]


def _drop_nonfinite(
    FM: np.ndarray, current_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows that contain non-finite values; return updated mask.

    Ports ``order_cells.R:1386``: ``FM[apply(FM, 1, function(x) all(is.finite(x))), ]``.
    """
    finite_rows = np.all(np.isfinite(FM), axis=1)
    kept_indices = np.flatnonzero(current_mask)
    final_mask = np.zeros_like(current_mask)
    final_mask[kept_indices[finite_rows]] = True
    return FM[finite_rows], final_mask


def _reduce_tsne(
    adata: AnnData,
    FM: np.ndarray,
    max_components: int,
    num_dim: int,
    random_state: int,
    **tsne_kwargs: Any,
) -> None:
    """tSNE branch: PCA → TSNE → obsm['X_dr'].

    The PCA step mirrors R's ``prcomp_irlba(t(FM), center=TRUE,
    scale.=TRUE)`` (``order_cells.R:1430-1432``) — per-gene z-score
    (centre + ``sd`` with ddof=1) followed by a standard SVD, returning
    PC scores unscaled (``irlba_res$x`` is ``U @ diag(S)``, not whitened).
    """
    n_cells = FM.shape[1]
    FM_t = FM.T.astype(np.float64)  # cells x genes
    mu = FM_t.mean(axis=0)
    sd = FM_t.std(axis=0, ddof=1)  # R's ``sd()``/``scale()`` use ddof=1
    keep = sd > 0
    FM_scaled = (FM_t[:, keep] - mu[keep]) / sd[keep]

    n_dim = min(num_dim, min(FM_scaled.shape) - 1)
    pca = PCA(n_components=n_dim, random_state=random_state)
    top_pca = pca.fit_transform(FM_scaled)  # cells x n_dim

    perplexity = tsne_kwargs.pop("perplexity", min(30, max(5, (n_cells - 1) / 3)))
    tsne = TSNE(
        n_components=max_components,
        init="random",
        random_state=random_state,
        perplexity=perplexity,
        **tsne_kwargs,
    )
    Y = tsne.fit_transform(top_pca)
    adata.obsm["X_dr"] = Y

    state = ensure_state(adata)
    state["dim_reduce_type"] = "tSNE"
    state.setdefault("aux_clustering", {})["tSNE"] = {
        "pca_components_used": n_dim,
        "reduced_dimension": top_pca,
    }


def _ddr_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    """Map R DDRTree argument names to the Python package's snake_case."""
    rename = {
        "maxIter": "max_iter",
        "param.gamma": "gamma",
        "lambda": "lambda_",
    }
    out: dict[str, Any] = {}
    allowed = {"initial_method", "max_iter", "sigma", "lambda_",
               "gamma", "tol"}
    for k, v in extra.items():
        key = rename.get(k, k)
        if key in allowed:
            out[key] = v
    return out


def _reduce_ddrtree(
    adata: AnnData,
    FM: np.ndarray,
    max_components: int,
    auto_param_selection: bool,
    extra_arguments: dict[str, Any],
    verbose: bool,
    gene_mask: np.ndarray | None,
) -> None:
    """DDRTree branch: call ddrtree.DDRTree and store outputs."""
    n_cells = FM.shape[1]

    if auto_param_selection and n_cells >= 100:
        ncenter = extra_arguments.pop("ncenter", cal_ncenter(n_cells))
    else:
        ncenter = extra_arguments.pop("ncenter", None)

    kwargs = _ddr_kwargs(extra_arguments)
    result = _ddrtree.DDRTree(
        FM,
        dimensions=max_components,
        ncenter=ncenter,
        verbose=verbose,
        **kwargs,
    )
    store_ddrtree_result(adata, result, gene_mask=gene_mask)


def reduce_dimension(
    adata: AnnData,
    max_components: int = 2,
    reduction_method: str = "DDRTree",
    norm_method: str = "log",
    residual_model_formula_str: str | None = None,
    pseudo_expr: float = 1.0,
    relative_expr: bool = True,
    auto_param_selection: bool = True,
    verbose: bool = False,
    scaling: bool = True,
    num_dim: int = 50,
    random_state: int = 2016,
    **kwargs: Any,
) -> AnnData:
    """Project cells into a low-dimensional space (DDRTree or tSNE).

    Parameters
    ----------
    adata : anndata.AnnData
    max_components : int, default 2
        Number of dimensions in the final embedding.
    reduction_method : str, default ``"DDRTree"``
        Either ``"DDRTree"`` or ``"tSNE"``.
    norm_method : str, default ``"log"``
        One of ``"log"``, ``"vstExprs"``, ``"none"`` (behaviour depends on
        the expression family; see :func:`normalize_expr_data`).
    residual_model_formula_str : str, optional
        R-style formula whose non-intercept effects are regressed out of
        the normalised expression matrix before dimensionality reduction.
        Mirrors R ``reduceDimension(residualModelFormulaStr=...)``
        (``order_cells.R:1363-1375``), which uses ``limma::lmFit`` for a
        per-gene OLS fit and subtracts ``beta[, -1] %*% t(X[, -1])`` from
        ``FM``.
    pseudo_expr : float, default 1.0
        Pseudocount added before log-transform.
    relative_expr : bool, default True
        Whether to divide by the ``Size_Factor`` column when normalising.
    auto_param_selection : bool, default True
        If True and ``n_cells >= 100``, DDRTree's ``ncenter`` is chosen via
        :func:`cal_ncenter`.
    verbose : bool, default False
        Forwarded to DDRTree.
    scaling : bool, default True
        Z-score genes across cells before projecting.
    num_dim : int, default 50
        PCA components used upstream of TSNE.
    random_state : int, default 2016
        Seed for the stochastic components (matches R's ``set.seed(2016)``).
    **kwargs
        Extra keyword arguments forwarded to the chosen reduction backend.
        For DDRTree: ``ncenter``, ``initial_method``, ``maxIter``, ``sigma``,
        ``lambda`` / ``lambda_``, ``param.gamma`` / ``gamma``, ``tol``.
        For tSNE: any other ``sklearn.manifold.TSNE`` kwarg.

    Returns
    -------
    anndata.AnnData
        Same AnnData with ``obsm['X_dr']`` populated and DDRTree / tSNE
        state registered under ``uns['monocle2']``.
    """
    np.random.seed(random_state)

    FM = normalize_expr_data(
        adata, norm_method=norm_method, pseudo_expr=pseudo_expr,
        relative_expr=relative_expr,
    )
    # Pipeline order matches R ``order_cells.R:1356-1386``:
    # normalize → drop zero-SD rows → batch removal → scale → drop non-finite.
    FM, zero_mask = _drop_zero_sd(FM)
    if residual_model_formula_str is not None:
        FM = _remove_batch_effects(FM, adata, residual_model_formula_str)
    if scaling:
        FM = _apply_scaling(FM)
    FM, gene_mask = _drop_nonfinite(FM, zero_mask)
    if FM.shape[0] == 0:
        raise RuntimeError(
            "All genes have standard deviation zero; cannot reduce dimensions."
        )

    if reduction_method == "tSNE":
        _reduce_tsne(
            adata, FM, max_components=max_components, num_dim=num_dim,
            random_state=random_state, **kwargs,
        )
    elif reduction_method == "DDRTree":
        _reduce_ddrtree(
            adata, FM, max_components=max_components,
            auto_param_selection=auto_param_selection,
            extra_arguments=dict(kwargs), verbose=verbose,
            gene_mask=gene_mask,
        )
    else:
        raise ValueError(
            f"Unsupported reduction_method: {reduction_method!r}. "
            "Only 'DDRTree' and 'tSNE' are ported."
        )

    return adata
