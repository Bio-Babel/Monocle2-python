"""Per-gene differential expression tests.

Ports of ``differentialGeneTest``, ``responseMatrix``, and ``genSmoothCurves``
from ``monocle2/R/differential_expression.R`` and ``monocle2/R/expr_models.R``.
Each gene is fitted once under the full formula and once under the reduced
formula; the two fits are compared via an LRT. p-values are BH-adjusted
across the ``OK`` subset.
"""

from __future__ import annotations

import math
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
    LOG10_RESPONSE_FAMILIES,
    _SuppressWarnings,
    fit_glm,
    fit_glm_with_fallback,
    fit_joint_nb,
    fit_tobit,
    lrt,
    make_family,
    make_response,
)
from .families import Negbinomial, NegbinomialSize, Tobit
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


def _map_per_gene(
    worker,
    n_items: int,
    cores: int,
    args: tuple,
):
    """Run ``worker(i, *args)`` for ``i in range(n_items)``.

    ``cores == 1`` stays on a pure serial list comprehension — no joblib
    import, no IPC. ``cores > 1`` dispatches via :class:`joblib.Parallel`
    with the default ``loky`` backend. All worker arguments are passed
    positionally so that joblib can auto-memmap shared NumPy arrays that
    exceed ``max_nbytes`` (default 1 MB) rather than re-pickling them per
    call.

    Mirrors the role of R's ``mclapply``/``cores`` argument across
    ``differentialGeneTest``, ``fit_models``, and ``responseMatrix``.
    """
    if cores <= 1:
        return [worker(i, *args) for i in range(n_items)]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=cores)(
        delayed(worker)(i, *args) for i in range(n_items)
    )


def _fit_one_gene_indexed(
    i: int,
    X: np.ndarray,
    full_X: np.ndarray,
    reduced_X: np.ndarray,
    family: Any,
    disp_func,
    relative_expr: bool,
    sfs: np.ndarray,
    verbose: bool,
) -> FitOutcome:
    """Top-level wrapper so :func:`_fit_one_gene` can be dispatched via joblib."""
    return _fit_one_gene(
        X[:, i], full_X, reduced_X, family, disp_func,
        relative_expr, sfs, verbose,
    )


def _gene_dispersion_alpha(
    disp_func,
    x_orig: np.ndarray,
    family: Any,
) -> float:
    """Resolve the NB dispersion ``alpha`` for a single gene.

    Mirrors R's ``fit_model_helper`` (``expr_models.R:25-44``):
      1. If ``disp_func`` is available and returns a valid positive
         value, override with ``alpha = disp_func(mean(round(orig_x)))``.
      2. Otherwise honour the family-supplied ``size`` (for
         ``NegbinomialSize``), yielding ``alpha = 1/size``. A ``size =
         inf`` produces ``alpha = 0``, which triggers a Poisson fit in
         :func:`make_family` (matches VGAM ``negbinomial.size(size=Inf)``).
      3. For ``Negbinomial`` (no ``.size``) return a seed alpha; joint
         MLE takes over via :func:`fit_joint_nb`.
    """
    family_name = family.vfamily
    if family_name not in ("negbinomial", "negbinomial.size"):
        return 1.0
    hint = calculate_nb_dispersion_hint(disp_func, np.round(x_orig))
    if hint is not None and hint > 0:
        return float(hint)
    if isinstance(family, NegbinomialSize):
        size = float(family.size)
        if not math.isfinite(size) or size <= 0:
            return 0.0  # Poisson degeneracy
        return 1.0 / size
    # Negbinomial (joint estimation): arbitrary seed
    return 1.0


def _fit_one_gene(
    x: np.ndarray,
    full_X: np.ndarray,
    reduced_X: np.ndarray,
    family: Any,
    disp_func,
    relative_expr: bool,
    size_factor: np.ndarray,
    verbose: bool,
) -> FitOutcome:
    # Mirror R ``fit_model_helper``'s ``suppressWarnings`` wrapper
    # (``expr_models.R:48-82``). We already suppress inside
    # ``fit_glm`` / ``fit_joint_nb`` / ``fit_tobit``, but downstream
    # consumers of the returned fit — ``lrt`` (reads ``.llf``) and any
    # ``.predict`` / ``.fittedvalues`` access — also emit RuntimeWarnings
    # from ``statsmodels.genmod.families.family.loglike_obs`` (NB /
    # Poisson log(0*mu)) on degenerate genes. R's ``suppressWarnings``
    # wraps the *whole* per-gene expression, so we mirror that by
    # extending suppression across the entire per-gene body.
    with _SuppressWarnings():
        return _fit_one_gene_inner(
            x, full_X, reduced_X, family, disp_func,
            relative_expr, size_factor, verbose,
        )


def _fit_one_gene_inner(
    x: np.ndarray,
    full_X: np.ndarray,
    reduced_X: np.ndarray,
    family: Any,
    disp_func,
    relative_expr: bool,
    size_factor: np.ndarray,
    verbose: bool,
) -> FitOutcome:
    family_name = family.vfamily
    y = make_response(x, family_name, size_factor, relative_expr)
    alpha = _gene_dispersion_alpha(disp_func, x, family)
    # For VGAM ``negbinomial()`` (no ``.size``): jointly estimate (mu, size)
    # using statsmodels.discrete.NegativeBinomial, matching R's
    # ``negbinomial(isize=1/alpha)`` (expr_models.R:40).
    if isinstance(family, Negbinomial):
        try:
            full_fit = fit_joint_nb(y, full_X, start_alpha=alpha)
            reduced_fit = fit_joint_nb(y, reduced_X, start_alpha=alpha)
        except Exception as exc:
            if verbose:
                print(f"joint NB fit failed: {exc}")
            return FitOutcome(status="FAIL", family=family_name, pval=1.0)
    elif isinstance(family, Tobit):
        # Mirrors R's ``VGAM::vglm(log10(x) ~ ..., family=tobit(Lower, Upper))``
        # (expr_models.R:45-55). The ``tryCatch`` around ``vglm`` in R's
        # ``fit_model_helper`` returns ``NULL`` for Tobit on error — match
        # that here by converting to FitOutcome(FAIL).
        try:
            full_fit = fit_tobit(y, full_X, lower=family.lower, upper=family.upper)
            reduced_fit = fit_tobit(
                y, reduced_X, lower=family.lower, upper=family.upper,
            )
        except Exception as exc:
            if verbose:
                print(f"tobit fit failed: {exc}")
            return FitOutcome(status="FAIL", family=family_name, pval=1.0)
    else:
        sm_family = make_family(family_name, alpha=alpha)
        full_fit = fit_glm_with_fallback(y, full_X, sm_family, family_name)
        reduced_fit = fit_glm_with_fallback(y, reduced_X, sm_family, family_name)
    if full_fit is None or reduced_fit is None:
        if verbose:
            print(f"gene fit failed for family={family_name}")
        return FitOutcome(status="FAIL", family=family_name, pval=1.0)
    try:
        stat, dof, pval = lrt(full_fit, reduced_fit)
    except Exception as exc:
        if verbose:
            print(exc)
        return FitOutcome(status="FAIL", family=family_name, pval=1.0)
    if np.isnan(pval):
        return FitOutcome(status="FAIL", family=family_name, pval=1.0,
                          statistic=stat, df=dof)
    return FitOutcome(status="OK", family=family_name, pval=float(pval),
                      statistic=stat, df=dof)


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
        Number of joblib workers for the per-gene LRT fits. ``cores=1``
        keeps a pure-serial fast path (no joblib import, no IPC); any
        value > 1 dispatches via ``joblib.Parallel`` with the default
        ``loky`` backend. R uses ``mclapply`` here (R
        ``differential_expression.R:164-179``).
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
    gene_ids = adata.var_names.astype(str).to_numpy()
    outs = _map_per_gene(
        _fit_one_gene_indexed,
        n_items=adata.n_vars,
        cores=cores,
        args=(X, full_X, reduced_X, family, disp_func,
              relative_expr, sfs, verbose),
    )
    records = [
        {
            "gene_id": gene_ids[i],
            "status": out.status,
            "family": out.family,
            "pval": out.pval,
        }
        for i, out in enumerate(outs)
    ]

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


def _predict_one_indexed(
    i: int,
    fits: Sequence[Any],
    X_new: np.ndarray | None,
):
    """Worker for :func:`response_matrix`. Returns the predicted mean or ``None``.

    Wraps the predict / fittedvalues access in :class:`_SuppressWarnings`
    so the per-gene call works correctly whether running serially or
    inside a joblib worker process (the outer context manager would not
    cross the IPC boundary).
    """
    fit = fits[i]
    if fit is None:
        return None
    with _SuppressWarnings():
        if X_new is not None:
            mu = np.asarray(fit.predict(X_new))
        else:
            mu = np.asarray(fit.fittedvalues)
        # R ``responseMatrix`` (``expr_models.R:158-160``) inverts the
        # log10 response transform for Gaussian/Tobit-style families.
        fit_family = getattr(fit, "_monocle2py_family_name", None)
        if fit_family in LOG10_RESPONSE_FAMILIES:
            mu = np.power(10.0, mu)
    return mu


def _fit_one_for_models(
    i: int,
    expr: np.ndarray,
    X_design: np.ndarray,
    family: Any,
    disp_func,
    relative_expr: bool,
    sfs: np.ndarray,
):
    """Worker for :func:`fit_models`. Returns the per-gene fit or ``None``.

    Lifted to module scope so :class:`joblib.Parallel` can ship it to
    worker processes by qualified name. Try/except blocks mirror R's
    ``tryCatch`` in ``fit_model_helper`` (``expr_models.R:48-95``).
    """
    family_name = family.vfamily
    x = expr[:, i]
    y = make_response(x, family_name, sfs, relative_expr)
    alpha = _gene_dispersion_alpha(disp_func, x, family)
    if isinstance(family, Negbinomial):
        try:
            return fit_joint_nb(y, X_design, start_alpha=alpha)
        except Exception:
            return None
    if isinstance(family, Tobit):
        try:
            return fit_tobit(y, X_design, lower=family.lower, upper=family.upper)
        except Exception:
            return None
    gfamily = make_family(family_name, alpha=alpha)
    return fit_glm_with_fallback(y, X_design, gfamily, family_name)


def fit_models(
    adata: AnnData,
    model_formula_str: str = DEFAULT_FULL_FORMULA,
    relative_expr: bool = True,
    cores: int = 1,
) -> dict[str, Any]:
    """Fit one GLM per gene. Internal helper for ``gen_smooth_curves``.

    ``cores`` is forwarded to :func:`_map_per_gene`: ``cores=1`` runs a
    pure serial loop, ``cores>1`` dispatches via :class:`joblib.Parallel`.

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
    gene_names = adata.var_names.astype(str)
    fits = _map_per_gene(
        _fit_one_for_models,
        n_items=adata.n_vars,
        cores=cores,
        args=(expr, X_design, family, disp_func, relative_expr, sfs),
    )
    return dict(zip(gene_names, fits))


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
        ``cores=1`` keeps a serial loop; ``cores>1`` dispatches per-gene
        predictions via :class:`joblib.Parallel`. R's ``responseMatrix``
        uses ``mclapply`` (``expr_models.R:158-160``).

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

    names = [str(name) for name, _ in items]
    fits = [fit for _, fit in items]
    # R's ``responseMatrix`` runs ``predict(x, ...)`` inside ``mclapply``;
    # when the underlying glm is degenerate, the prediction triggers
    # GLM-family RuntimeWarnings. Mirror R's per-item ``suppressWarnings``
    # by pushing the context manager into :func:`_predict_one_indexed`,
    # so the same code is correct serially and inside joblib workers
    # (the outer context manager couldn't follow workers across the IPC
    # boundary).
    preds = _map_per_gene(
        _predict_one_indexed,
        n_items=len(fits),
        cores=cores,
        args=(fits, X_new),
    )

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
        Forwarded to :func:`fit_models` and :func:`response_matrix` so
        per-gene fits and predictions both run in parallel when > 1.

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
