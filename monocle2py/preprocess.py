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


def _glm_fit_gamma_identity(
    X: np.ndarray,
    y: np.ndarray,
    start: np.ndarray,
    epsilon: float = 1e-8,
    maxit: int = 25,
) -> dict:
    """Port of R's ``glm.fit`` for ``Gamma(link="identity")``.

    Uses R's deviance-based convergence criterion and its step-halving
    safeguard so the coefficients match R's ``glm()`` to machine precision.

    For the Gamma family with identity link the working response collapses
    to ``y`` itself: ``eta = mu`` so ``d eta / d mu = 1`` and the IRLS
    weights are ``1 / mu^2``.  The inner linear system therefore reduces
    to a weighted-least-squares update ``(X' W X) beta = X' W y`` with
    ``W = diag(1 / mu^2)``.

    Returns a dict with the final ``params``, ``mu``, ``weights``,
    ``deviance``, ``dispersion`` and ``converged`` flag.
    """
    n, p = X.shape
    beta = np.asarray(start, dtype=float).copy()
    mu = X @ beta
    if np.any(mu <= 0):
        raise RuntimeError(
            "Parametric dispersion fit failed: start gives non-positive mu."
        )

    def _deviance(y: np.ndarray, mu: np.ndarray) -> float:
        return 2.0 * float(np.sum(-np.log(y / mu) + (y - mu) / mu))

    dev_old = _deviance(y, mu)
    converged = False
    for _ in range(maxit):
        W = 1.0 / (mu * mu)
        WX = X * W[:, None]
        A = X.T @ WX
        b = WX.T @ y  # z = y for the identity link
        beta_new = np.linalg.solve(A, b)
        mu_new = X @ beta_new
        if np.any(mu_new <= 0):
            step = beta_new - beta
            for _ in range(20):
                step = step * 0.5
                beta_try = beta + step
                mu_try = X @ beta_try
                if np.all(mu_try > 0):
                    beta_new, mu_new = beta_try, mu_try
                    break
            else:
                raise RuntimeError(
                    "Parametric dispersion fit failed: step-halving could not "
                    "recover a positive mu."
                )
        dev_new = _deviance(y, mu_new)
        beta, mu = beta_new, mu_new
        if abs(dev_new - dev_old) / (0.1 + abs(dev_new)) < epsilon:
            converged = True
            dev_old = dev_new
            break
        dev_old = dev_new

    # summary.glm's Gamma dispersion: sum(((y-mu)/mu)^2) / (n - p)
    pearson2 = np.sum(((y - mu) / mu) ** 2)
    dispersion = pearson2 / (n - p) if n > p else np.nan
    return {
        "params": beta,
        "mu": mu,
        "weights": 1.0 / (mu * mu),
        "deviance": dev_old,
        "dispersion": dispersion,
        "converged": converged,
        "X": X,
        "y": y,
        "rank": p,
    }


def _cooks_distance_glm(fit: dict) -> np.ndarray:
    """Cook's distance for a Gamma/identity GLM, matching R's ``cooks.distance``.

    Formula: ``CD_i = (pearson_i / (1 - h_i))^2 * h_i / (dispersion * p)``
    where ``pearson_i = (y_i - mu_i) / mu_i`` for Gamma and ``h`` are the
    hat-matrix diagonals of the IRLS weighted design.
    """
    X = fit["X"]
    y = fit["y"]
    mu = fit["mu"]
    w = fit["weights"]
    p = fit["rank"]
    dispersion = fit["dispersion"]
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    # Hat matrix diagonal via QR of the weighted design.
    Q, _ = np.linalg.qr(Xw, mode="reduced")
    h = np.sum(Q * Q, axis=1)
    pearson = (y - mu) / mu
    with np.errstate(divide="ignore", invalid="ignore"):
        cd = (pearson / (1.0 - h)) ** 2 * h / (dispersion * p)
    cd[~np.isfinite(cd)] = np.nan
    return cd


def _parametric_dispersion_fit(
    disp_table: pd.DataFrame,
    initial_coefs: tuple[float, float] = (1e-6, 1.0),
    max_iter: int = 10,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Port of ``parametricDispersionFit`` using a hand-rolled R-faithful GLM.

    statsmodels' IRLS uses a parameter-based convergence test and a final
    pinv refit that disagree with R's deviance-based convergence by
    ~1e-5 on ``(asymptDisp, extraPois)``.  Porting R's ``glm.fit`` directly
    matches R to machine precision, which cascades into VST and every
    downstream ``disp_func`` evaluation.

    Returns
    -------
    fit : dict
        Output of ``_glm_fit_gamma_identity`` on the final inner iteration,
        carrying everything ``_cooks_distance_glm`` needs.
    coefs : numpy.ndarray
        Final ``[asymptDisp, extraPois]``.
    keep_mask : numpy.ndarray of bool
        Boolean mask over ``disp_table`` rows actually used in the final fit
        (matches R's behaviour of retaining only rows that survive the
        residual cutoff on the last outer iteration).
    """
    coefs = np.array(initial_coefs, dtype=float)
    iter_count = 0
    tbl = disp_table.dropna(subset=["mu", "disp"]).copy()
    fit = None
    keep_idx = None

    while True:
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals = tbl["disp"].to_numpy() / (
                coefs[0] + coefs[1] / tbl["mu"].to_numpy()
            )
        keep = (residuals > initial_coefs[0]) & (residuals < 10_000)
        good = tbl.loc[keep]
        keep_idx = np.flatnonzero(keep.to_numpy()) if hasattr(keep, "to_numpy") else np.flatnonzero(keep)

        X = np.column_stack([
            np.ones(len(good)), 1.0 / good["mu"].to_numpy()
        ])
        y = good["disp"].to_numpy()
        fit = _glm_fit_gamma_identity(X, y, start=coefs)

        old = coefs
        coefs = fit["params"].astype(float).copy()
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
    return fit, coefs, keep_idx


def estimate_dispersions(
    adata: AnnData,
    min_cells_detected: int = 1,
    remove_outliers: bool = True,
    model_name: str = "blind",
    model_formula_str: str = "~1",
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
    model_formula_str : str, default ``"~1"``
        Covariate formula. When it references ``adata.obs`` columns
        (e.g. ``"~CellType"``), cells are partitioned by the unique
        covariate combinations, ``_disp_calc_helper_nb`` is run on each
        partition, and the per-group ``(gene_id, mu, disp)`` rows are
        concatenated before the single global parametric fit. Mirrors
        R's ``estimateDispersionsForCellDataSet``
        (``expr_models.R:515-522``).

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

    from ._internal._formula import formula_terms

    covariates = [t for t in formula_terms(model_formula_str) if t != "1"]
    if covariates:
        missing = [c for c in covariates if c not in adata.obs.columns]
        if missing:
            raise ValueError(
                f"Formula terms {missing!r} not present in adata.obs"
            )
        # R's ``group_by_`` with ``.dots = model_terms``: partition cells
        # by unique combinations of the covariate columns, run
        # ``disp_calc_helper_NB`` on each cell subset, and rbind the
        # per-group (gene_id, mu, disp) rows into a pooled table. The
        # later ``parametricDispersionFit`` then sees multiple rows per
        # gene — one per covariate level at which the gene was detected.
        group_keys = adata.obs[covariates]
        tables: list[pd.DataFrame] = []
        for _, idx in group_keys.groupby(
            list(covariates), observed=True, sort=True,
        ).groups.items():
            positions = adata.obs_names.get_indexer(idx)
            sub = adata[positions].copy()
            tables.append(
                _disp_calc_helper_nb(sub, min_cells_detected=min_cells_detected)
            )
        disp_df = pd.concat(tables, axis=0, ignore_index=True)
    else:
        disp_df = _disp_calc_helper_nb(
            adata, min_cells_detected=min_cells_detected
        )

    disp_df = disp_df.dropna(subset=["mu"]).reset_index(drop=True)

    fit, coefs, fit_rows = _parametric_dispersion_fit(disp_df)

    if remove_outliers:
        cook = _cooks_distance_glm(fit)
        cutoff = 4.0 / len(disp_df)
        # R's estimateDispersions treats any gene outside the fit (filtered
        # by the residual cutoff) as an outlier too, matching the
        # ``setdiff(row.names(disp_table), names(CD))`` branch.
        keep_mask = np.ones(len(disp_df), dtype=bool)
        keep_mask[fit_rows[cook > cutoff]] = False
        in_fit = np.zeros(len(disp_df), dtype=bool)
        in_fit[fit_rows] = True
        keep_mask &= in_fit
        print(f"Removing {int((~keep_mask).sum())} outliers")
        refit_df = disp_df.iloc[keep_mask].reset_index(drop=True)
        fit, coefs, _ = _parametric_dispersion_fit(refit_df)

    asymp, extra = float(coefs[0]), float(coefs[1])

    # Persist only the h5ad-safe payload — disp_table (DataFrame) and the
    # two fitted coefficients. The dispersion-vs-mean closure is rebuilt
    # by ``get_disp_fit_info`` from these coefficients on every read, so
    # callers always see a usable ``disp_func`` while ``adata.write_h5ad``
    # round-trips without warnings or silent callable drops.
    info = {
        "disp_table": disp_df,
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
