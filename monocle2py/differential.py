"""Per-gene differential expression tests (Slice 6).

Ports of ``differentialGeneTest``, ``responseMatrix``, and ``genSmoothCurves``
from ``monocle2/R/differential_expression.R`` and ``monocle2/R/expr_models.R``.
Each gene is fitted once under the full formula and once under the reduced
formula; the two fits are compared via an LRT. p-values are BH-adjusted
across the ``OK`` subset.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

from ._internal._formula import (
    DEFAULT_FULL_FORMULA,
    DEFAULT_REDUCED_FORMULA,
    build_design_matrix,
    formula_terms,
)
from ._internal._vgam import (
    FitOutcome,
    calculate_nb_dispersion_hint,
    fit_glm,
    lrt,
    make_family,
    make_response,
)
from ._uns import (
    SIZE_FACTOR_COL,
    get_disp_fit_info,
    get_expression_family,
)

__all__ = [
    "differential_gene_test",
    "fit_models",
    "response_matrix",
    "gen_smooth_curves",
]


def _dense_expression(adata: AnnData) -> np.ndarray:
    X = adata.X
    if issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _validate_terms(formula_strs: Sequence[str], obs: pd.DataFrame) -> None:
    for s in formula_strs:
        for name in formula_terms(s):
            if name not in obs.columns:
                raise ValueError(
                    f"Formula term {name!r} not found in adata.obs columns"
                )
            col = obs[name]
            if np.any(pd.isna(col)) or np.any(np.isinf(pd.to_numeric(col, errors="coerce"))):
                raise ValueError(
                    f"Inf/NaN values in adata.obs[{name!r}]; cannot fit"
                )


def _gene_dispersion_alpha(
    disp_func,
    x_orig: np.ndarray,
    family_name: str,
    default_alpha: float,
) -> float:
    if family_name not in ("negbinomial", "negbinomial.size"):
        return default_alpha
    hint = calculate_nb_dispersion_hint(disp_func, np.round(x_orig))
    if hint is None or hint <= 0:
        return default_alpha
    return float(hint)


def _fit_one_gene(
    x: np.ndarray,
    full_X: np.ndarray,
    reduced_X: np.ndarray,
    family_name: str,
    disp_func,
    relative_expr: bool,
    size_factor: np.ndarray,
    verbose: bool,
) -> FitOutcome:
    y = make_response(x, family_name, size_factor, relative_expr)
    alpha = _gene_dispersion_alpha(disp_func, x, family_name, default_alpha=1.0)
    family = make_family(family_name, alpha=alpha)
    try:
        full_fit = fit_glm(y, full_X, family)
        reduced_fit = fit_glm(y, reduced_X, family)
        stat, dof, pval = lrt(full_fit, reduced_fit)
        if np.isnan(pval):
            return FitOutcome(status="FAIL", family=family_name, pval=1.0,
                              statistic=stat, df=dof)
        return FitOutcome(status="OK", family=family_name, pval=float(pval),
                          statistic=stat, df=dof)
    except Exception as exc:
        if verbose:
            print(exc)
        return FitOutcome(status="FAIL", family=family_name, pval=1.0)


def differential_gene_test(
    adata: AnnData,
    full_model_formula_str: str = DEFAULT_FULL_FORMULA,
    reduced_model_formula_str: str = DEFAULT_REDUCED_FORMULA,
    relative_expr: bool = True,
    cores: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Test each gene for differential expression along a formula.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells x genes AnnData produced by :func:`new_cell_dataset`.
    full_model_formula_str : str, default ``"~sm.ns(Pseudotime, df=3)"``
        Full model. ``sm.ns``/``ns`` terms map to patsy ``bs``.
    reduced_model_formula_str : str, default ``"~1"``
        Reduced model for the LRT.
    relative_expr : bool, default True
        For negbinomial families, divide each cell's counts by
        ``obs['Size_Factor']`` before rounding.
    cores : int, default 1
        Accepted for signature parity; per-gene fits run serially.
    verbose : bool, default False
        Print the exception when a gene's fit fails.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``adata.var_names``; columns ``status, family, pval,
        qval`` plus every column of ``adata.var``.
    """
    family = get_expression_family(adata)
    _validate_terms([full_model_formula_str, reduced_model_formula_str], adata.obs)

    full_X = build_design_matrix(full_model_formula_str, adata.obs)
    reduced_X = build_design_matrix(reduced_model_formula_str, adata.obs)

    if relative_expr and family.vfamily in ("negbinomial", "negbinomial.size"):
        if SIZE_FACTOR_COL not in adata.obs.columns:
            raise ValueError(
                "Call estimate_size_factors before differential_gene_test"
            )
        if adata.obs[SIZE_FACTOR_COL].isna().any():
            raise ValueError("NaN values in Size_Factor column")

    sfs = (
        adata.obs[SIZE_FACTOR_COL].to_numpy(dtype=float)
        if SIZE_FACTOR_COL in adata.obs.columns
        else np.ones(adata.n_obs)
    )

    info = get_disp_fit_info(adata, "blind")
    disp_func = info.get("disp_func") if info is not None else None

    X = _dense_expression(adata)
    records: list[dict[str, Any]] = []
    gene_ids = adata.var_names.astype(str).to_numpy()
    for i in range(adata.n_vars):
        out = _fit_one_gene(
            X[:, i], full_X, reduced_X, family.vfamily, disp_func,
            relative_expr, sfs, verbose,
        )
        records.append({
            "gene_id": gene_ids[i],
            "status": out.status,
            "family": out.family,
            "pval": out.pval,
        })

    res = pd.DataFrame(records).set_index("gene_id")
    qval = np.ones(len(res), dtype=float)
    ok = (res["status"] == "OK").to_numpy()
    if ok.any():
        _, q_ok, _, _ = multipletests(res.loc[ok, "pval"].to_numpy(), method="fdr_bh")
        qval[ok] = q_ok
    res["qval"] = qval

    fdata = adata.var.copy()
    fdata.index = fdata.index.astype(str)
    merged = res.join(fdata, how="left")
    return merged.reindex(gene_ids)


def fit_models(
    adata: AnnData,
    model_formula_str: str = DEFAULT_FULL_FORMULA,
    relative_expr: bool = True,
    cores: int = 1,
) -> dict[str, Any]:
    """Fit one GLM per gene. Internal helper for ``gen_smooth_curves``.

    Returns a mapping ``gene_id -> statsmodels GLMResults`` (or ``None``
    when a gene's fit fails).
    """
    family = get_expression_family(adata)
    _validate_terms([model_formula_str], adata.obs)
    X_design = build_design_matrix(model_formula_str, adata.obs)

    if relative_expr and family.vfamily in ("negbinomial", "negbinomial.size"):
        if SIZE_FACTOR_COL not in adata.obs.columns:
            raise ValueError("Call estimate_size_factors before fit_models")

    sfs = (
        adata.obs[SIZE_FACTOR_COL].to_numpy(dtype=float)
        if SIZE_FACTOR_COL in adata.obs.columns
        else np.ones(adata.n_obs)
    )

    info = get_disp_fit_info(adata, "blind")
    disp_func = info.get("disp_func") if info is not None else None

    expr = _dense_expression(adata)
    out: dict[str, Any] = {}
    for i, gene in enumerate(adata.var_names.astype(str)):
        x = expr[:, i]
        y = make_response(x, family.vfamily, sfs, relative_expr)
        alpha = _gene_dispersion_alpha(
            disp_func, x, family.vfamily, default_alpha=1.0
        )
        gfamily = make_family(family.vfamily, alpha=alpha)
        try:
            out[gene] = fit_glm(y, X_design, gfamily)
        except Exception:
            out[gene] = None
    return out


def response_matrix(
    models: dict[str, Any] | Sequence[Any],
    newdata: pd.DataFrame | None = None,
    formula_str: str | None = None,
    response_type: str = "response",
    cores: int = 1,
) -> pd.DataFrame:
    """Predicted expression values for a list of GLM fits.

    Parameters
    ----------
    models : dict or sequence
        Output of :func:`fit_models`. When a value is ``None`` the row is
        filled with NaNs.
    newdata : pandas.DataFrame, optional
        Data frame used to evaluate the formula. Required when the models
        were built from a formula referring to columns of ``obs``.
    formula_str : str, optional
        Formula used to build the original design matrix. Required when
        ``newdata`` is supplied so that patsy can build a matching matrix.
    response_type : str, default ``"response"``
        Retained for signature parity; the fitted mean is returned for NB
        and Gaussian families.
    cores : int, default 1
        Accepted for signature parity.

    Returns
    -------
    pandas.DataFrame
        Genes x columns frame; columns are the row index of ``newdata`` when
        supplied, otherwise the fitted-value index.
    """
    if isinstance(models, dict):
        items = list(models.items())
    else:
        items = list(enumerate(models))

    if newdata is not None:
        if formula_str is None:
            raise ValueError(
                "formula_str is required when newdata is provided"
            )
        X_new = build_design_matrix(formula_str, newdata)
        col_names = [str(c) for c in newdata.index]
    else:
        X_new = None
        col_names = None

    preds: list[np.ndarray | None] = []
    names: list[str] = []
    for name, fit in items:
        names.append(str(name))
        if fit is None:
            preds.append(None)
            continue
        if X_new is not None:
            mu = np.asarray(fit.predict(X_new))
        else:
            mu = np.asarray(fit.fittedvalues)
        preds.append(mu)

    shapes = [p.shape[0] for p in preds if p is not None]
    if not shapes:
        raise RuntimeError("All model fits are None; cannot build matrix")
    n = shapes[0]
    out = np.full((len(preds), n), np.nan, dtype=float)
    for i, p in enumerate(preds):
        if p is not None:
            out[i, :] = p

    if col_names is None:
        col_names = [str(i) for i in range(n)]
    return pd.DataFrame(out, index=names, columns=col_names)


def gen_smooth_curves(
    adata: AnnData,
    new_data: pd.DataFrame,
    trend_formula: str = DEFAULT_FULL_FORMULA,
    relative_expr: bool = True,
    response_type: str = "response",
    cores: int = 1,
) -> pd.DataFrame:
    """Fit the trend formula per gene and return smoothed expression curves.

    Parameters
    ----------
    adata : anndata.AnnData
    new_data : pandas.DataFrame
        Data frame whose columns include the variables in ``trend_formula``
        (typically a grid of ``Pseudotime`` values).
    trend_formula : str, default ``"~sm.ns(Pseudotime, df=3)"``
    relative_expr : bool, default True
    response_type : str, default ``"response"``
    cores : int, default 1

    Returns
    -------
    pandas.DataFrame
        Genes x new_data rows of fitted mean expression.
    """
    models = fit_models(
        adata, model_formula_str=trend_formula,
        relative_expr=relative_expr, cores=cores,
    )
    return response_matrix(
        models, newdata=new_data, formula_str=trend_formula,
        response_type=response_type, cores=cores,
    )
