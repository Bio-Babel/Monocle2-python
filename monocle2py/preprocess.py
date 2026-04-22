"""Preprocessing: size factors, dispersion fit, gene detection, VST.

Direct Python ports of the R helpers in ``utils.R`` and ``expr_models.R``.
The default DESeq1-style size factor method ``"mean-geometric-mean-total"``
and the Gamma-identity parametric dispersion fit follow the original
implementations line-for-line.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from anndata import AnnData
from scipy.sparse import issparse

from ._uns import (
    SIZE_FACTOR_COL,
    ensure_state,
    get_disp_fit_info,
    get_expression_family,
    get_lower_detection_limit,
    set_disp_fit_info,
)

__all__ = [
    "estimate_size_factors",
    "estimate_dispersions",
    "detect_genes",
    "disp_table",
    "vst_exprs",
]


def _as_dense(X):
    """Return a dense 2-D ndarray view of a cell x gene matrix."""
    return X.toarray() if issparse(X) else np.asarray(X)


def _col_sums_cells(X) -> np.ndarray:
    """Per-cell total counts (== R's ``apply(CM, 2, sum)`` on gene x cell)."""
    sums = X.sum(axis=1)
    return np.asarray(sums).ravel()


def estimate_size_factors(
    adata: AnnData,
    locfunc: Callable[[np.ndarray], float] = np.median,
    round_exprs: bool = True,
    method: str = "mean-geometric-mean-total",
) -> AnnData:
    """Estimate per-cell size factors and store them in ``adata.obs``.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells x genes AnnData.
    locfunc : callable, default ``numpy.median``
        Location function used by the ``"weighted-median"`` and
        ``"median-geometric-mean"`` methods.
    round_exprs : bool, default True
        Round the expression matrix to integers before computing size factors.
    method : str, default ``"mean-geometric-mean-total"``
        One of ``"weighted-median"``, ``"median-geometric-mean"``,
        ``"median"``, ``"mode"``, ``"geometric-mean-total"``,
        ``"mean-geometric-mean-total"``.

    Returns
    -------
    anndata.AnnData
        The same AnnData with ``adata.obs['Size_Factor']`` populated.
    """
    X = adata.X
    if round_exprs:
        X = _as_dense(X)
        X = np.round(X)

    if method == "weighted-median":
        CM = _as_dense(X)
        log_medians = np.array([np.log(locfunc(row)) for row in CM.T])
        weights = np.array(
            [np.sum(row > 0) / len(row) for row in CM.T]
        )
        sfs = np.empty(CM.shape[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(CM.shape[0]):
                cnts = CM[i, :]
                norm = weights * (np.log(cnts) - log_medians)
                norm = norm[np.isfinite(norm)]
                sfs[i] = np.exp(np.mean(norm))
    elif method == "median-geometric-mean":
        CM = _as_dense(X)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_geo_means = np.mean(np.log(CM), axis=0)
        sfs = np.empty(CM.shape[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(CM.shape[0]):
                norm = np.log(CM[i, :]) - log_geo_means
                norm = norm[np.isfinite(norm)]
                sfs[i] = np.exp(locfunc(norm))
    elif method == "median":
        CM = _as_dense(X)
        gene_median = np.median(CM, axis=0)
        sfs = np.median(CM - gene_median, axis=1)
    elif method == "mode":
        raise NotImplementedError(
            "method='mode' requires estimate_t (see census.py in a later slice)"
        )
    elif method == "geometric-mean-total":
        cell_total = _col_sums_cells(X)
        with np.errstate(divide="ignore", invalid="ignore"):
            sfs = np.log(cell_total) / np.mean(np.log(cell_total))
    elif method == "mean-geometric-mean-total":
        cell_total = _col_sums_cells(X)
        with np.errstate(divide="ignore", invalid="ignore"):
            sfs = cell_total / np.exp(np.mean(np.log(cell_total)))
    else:
        raise ValueError(f"Unknown size factor method: {method!r}")

    sfs = np.asarray(sfs, dtype=float)
    sfs[np.isnan(sfs)] = 1.0
    adata.obs[SIZE_FACTOR_COL] = sfs
    return adata


def detect_genes(adata: AnnData, min_expr: float | None = None) -> AnnData:
    """Populate ``var['num_cells_expressed']`` and ``obs['num_genes_expressed']``.

    Parameters
    ----------
    adata : anndata.AnnData
    min_expr : float, optional
        Threshold above which a count is considered detected. Defaults to
        ``adata.uns['monocle2']['lower_detection_limit']``.

    Returns
    -------
    anndata.AnnData
    """
    if min_expr is None:
        min_expr = get_lower_detection_limit(adata)

    X = adata.X
    mask = (X > min_expr)
    if issparse(mask):
        num_cells_per_gene = np.asarray(mask.sum(axis=0)).ravel()
        num_genes_per_cell = np.asarray(mask.sum(axis=1)).ravel()
    else:
        num_cells_per_gene = np.asarray(mask).sum(axis=0)
        num_genes_per_cell = np.asarray(mask).sum(axis=1)

    adata.var["num_cells_expressed"] = num_cells_per_gene.astype(np.int64)
    adata.obs["num_genes_expressed"] = num_genes_per_cell.astype(np.int64)
    return adata


def _disp_calc_helper_nb(
    adata: AnnData, min_cells_detected: int
) -> pd.DataFrame:
    """Per-gene empirical mean and dispersion (moment estimator)."""
    lower = get_lower_detection_limit(adata)
    X = _as_dense(adata.X)
    rounded = np.round(X)

    detected = (rounded > lower).sum(axis=0)
    nz_mask = detected > min_cells_detected
    if not np.any(nz_mask):
        raise RuntimeError(
            "No genes pass min_cells_detected; cannot estimate dispersions."
        )

    rounded = rounded[:, nz_mask]
    sfs = adata.obs[SIZE_FACTOR_COL].to_numpy(dtype=float)
    if np.any(np.isnan(sfs)):
        raise ValueError(
            "NaNs found in size factors. Did you call estimate_size_factors?"
        )
    x = rounded / sfs[:, None]

    xim = float(np.mean(1.0 / sfs))
    f_mean = x.mean(axis=0)
    f_var = ((x - f_mean) ** 2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = (f_var - xim * f_mean) / (f_mean ** 2)
    disp = np.where(disp < 0, 0.0, disp)

    gene_ids = adata.var_names[nz_mask].to_numpy()
    mu = f_mean.copy()
    disp_arr = disp.copy()
    zero = mu == 0
    mu[zero] = np.nan
    disp_arr[zero] = np.nan

    return pd.DataFrame(
        {"gene_id": gene_ids, "mu": mu, "disp": disp_arr}
    )


def _parametric_dispersion_fit(
    disp_table: pd.DataFrame,
    initial_coefs: tuple[float, float] = (1e-6, 1.0),
    max_iter: int = 10,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, np.ndarray]:
    """Port of ``parametricDispersionFit`` using statsmodels Gamma(identity) GLM.

    Returns the fitted statsmodels result and the final coefficient vector
    ``[asymptDisp, extraPois]``.
    """
    coefs = np.array(initial_coefs, dtype=float)
    iter_count = 0
    tbl = disp_table.dropna(subset=["mu", "disp"]).copy()
    fit = None

    while True:
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals = tbl["disp"].to_numpy() / (
                coefs[0] + coefs[1] / tbl["mu"].to_numpy()
            )
        keep = (residuals > initial_coefs[0]) & (residuals < 10_000)
        good = tbl.loc[keep]

        X = np.column_stack([
            np.ones(len(good)), 1.0 / good["mu"].to_numpy()
        ])
        y = good["disp"].to_numpy()
        model = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Identity()))
        fit = model.fit(start_params=coefs)

        old = coefs
        coefs = np.array(fit.params, dtype=float)
        if coefs[0] < initial_coefs[0]:
            coefs[0] = initial_coefs[0]
        if coefs[1] < 0:
            raise RuntimeError(
                "Parametric dispersion fit failed: extraPois < 0. "
                "Try a different detection threshold."
            )
        if np.sum(np.log(coefs / old) ** 2) < initial_coefs[0]:
            break
        iter_count += 1
        if iter_count > max_iter:
            break

    if not np.all(coefs > 0):
        raise RuntimeError("Parametric dispersion fit failed: non-positive coefs.")
    return fit, coefs


def estimate_dispersions(
    adata: AnnData,
    min_cells_detected: int = 1,
    remove_outliers: bool = True,
    model_name: str = "blind",
) -> AnnData:
    """Fit a negative-binomial dispersion curve and store it on the AnnData.

    Parameters
    ----------
    adata : anndata.AnnData
    min_cells_detected : int, default 1
        Minimum number of cells with rounded expression above
        ``lower_detection_limit`` for a gene to enter the fit.
    remove_outliers : bool, default True
        Refit after dropping genes whose Cook's distance exceeds
        ``4 / nrow(disp_table)``.
    model_name : str, default ``"blind"``
        Name of the fit registered under ``adata.uns['monocle2']['disp_fit_info']``.

    Returns
    -------
    anndata.AnnData
    """
    family = get_expression_family(adata)
    if family.vfamily not in ("negbinomial", "negbinomial.size"):
        raise ValueError(
            "estimate_dispersions only supports negbinomial / negbinomial.size "
            f"families; got {family.vfamily!r}"
        )
    if np.any(pd.isna(adata.obs[SIZE_FACTOR_COL].to_numpy())):
        raise ValueError(
            "NaNs in Size_Factor column. Call estimate_size_factors first."
        )

    disp_df = _disp_calc_helper_nb(adata, min_cells_detected=min_cells_detected)
    disp_df = disp_df.dropna(subset=["mu"]).reset_index(drop=True)

    fit, coefs = _parametric_dispersion_fit(disp_df)

    if remove_outliers:
        cook = fit.get_influence().cooks_distance[0]
        cutoff = 4.0 / len(disp_df)
        keep = cook <= cutoff
        print(f"Removing {int((~keep).sum())} outliers")
        refit_df = disp_df.iloc[keep].reset_index(drop=True)
        fit, coefs = _parametric_dispersion_fit(refit_df)

    asymp, extra = float(coefs[0]), float(coefs[1])

    def disp_func(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return asymp + extra / q

    info = {
        "disp_table": disp_df,
        "disp_func": disp_func,
        "coefficients": {"asymptDisp": asymp, "extraPois": extra},
    }
    set_disp_fit_info(adata, info, name=model_name)
    ensure_state(adata)  # confirm container exists
    return adata


def disp_table(adata: AnnData, model_name: str = "blind") -> pd.DataFrame:
    """Return a data frame of fitted vs empirical dispersions.

    Parameters
    ----------
    adata : anndata.AnnData
    model_name : str, default ``"blind"``

    Returns
    -------
    pandas.DataFrame
        Columns ``gene_id``, ``mean_expression``, ``dispersion_fit``,
        ``dispersion_empirical``.
    """
    info = get_disp_fit_info(adata, model_name)
    if info is None:
        raise RuntimeError(
            f"No dispersion fit registered under {model_name!r}. "
            "Call estimate_dispersions first."
        )
    tbl = info["disp_table"]
    return pd.DataFrame({
        "gene_id": tbl["gene_id"].to_numpy(),
        "mean_expression": tbl["mu"].to_numpy(),
        "dispersion_fit": info["disp_func"](tbl["mu"].to_numpy()),
        "dispersion_empirical": tbl["disp"].to_numpy(),
    })


def vst_exprs(
    adata: AnnData,
    model_name: str = "blind",
    expr_matrix: np.ndarray | None = None,
    round_vals: bool = True,
) -> np.ndarray:
    """Variance-stabilising transform of a cell x gene count matrix.

    Uses the fitted ``asymptDisp`` and ``extraPois`` coefficients from
    :func:`estimate_dispersions`.

    Parameters
    ----------
    adata : anndata.AnnData
    model_name : str, default ``"blind"``
    expr_matrix : numpy.ndarray, optional
        Pre-normalised matrix to transform. If omitted, ``adata.X`` is
        divided by ``Size_Factor`` per cell (and optionally rounded).
    round_vals : bool, default True
        Round the size-factor-normalised matrix before transforming.

    Returns
    -------
    numpy.ndarray
        Dense ``cells x genes`` log2-VST matrix.
    """
    info = get_disp_fit_info(adata, model_name)
    if info is None:
        raise RuntimeError(
            f"No dispersion model named {model_name!r}. Call estimate_dispersions."
        )
    coefs = info["coefficients"]
    a = coefs["asymptDisp"]
    b = coefs["extraPois"]

    if expr_matrix is None:
        X = _as_dense(adata.X)
        sfs = adata.obs[SIZE_FACTOR_COL].to_numpy(dtype=float)
        ncounts = X / sfs[:, None]
        if round_vals:
            ncounts = np.round(ncounts)
    else:
        ncounts = np.asarray(expr_matrix, dtype=float)

    q = ncounts
    numerator = 1.0 + b + 2.0 * a * q + 2.0 * np.sqrt(a * q * (1.0 + b + a * q))
    denom = 4.0 * a
    return np.log(numerator / denom) / np.log(2.0)
