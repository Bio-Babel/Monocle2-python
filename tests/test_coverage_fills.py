"""Coverage-fill tests targeting branches the R-parity and primary
suites leave uncovered. Every test explicitly names the R equivalent
(or documents the Python-only behaviour) so the intent is clear.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from monocle2py import (
    cluster_cells,
    detect_genes,
    differential_gene_test,
    estimate_dispersions,
    estimate_size_factors,
    fit_models,
    gaussian_family,
    new_cell_dataset,
    negbinomial,
    negbinomial_size,
    order_cells,
    plot_cell_trajectory,
    reduce_dimension,
    relative2abs,
    response_matrix,
    set_ordering_filter,
    tobit,
)


def _nb_cds(n_cells: int = 80, n_genes: int = 30, seed: int = 11) -> AnnData:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(1, 20, size=n_genes)
    X = np.column_stack([
        rng.negative_binomial(n=4, p=4.0 / (4.0 + m), size=n_cells)
        for m in mu
    ]).astype(float)
    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({
        "Pseudotime": np.linspace(0, 10, n_cells),
        "Hours": np.tile([0, 24, 48, 72], n_cells // 4 + 1)[:n_cells],
    }, index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    return new_cell_dataset(
        X, pheno_data=obs, feature_data=var,
        lower_detection_limit=0.5,
        expression_family=negbinomial_size(),
    )


# ---------------------------------------------------------------------------
# preprocess.py: alternate size-factor methods + error paths
# ---------------------------------------------------------------------------


def test_size_factor_median_geometric_mean_matches_formula() -> None:
    """``"median-geometric-mean"`` size factor equals
    ``exp(median(log(x_i) - mean(log(x)))``. Verify Python matches the
    R formula on a hand-crafted matrix (``normalization.R``).
    """
    cds = _nb_cds(n_cells=30, n_genes=15)
    estimate_size_factors(cds, method="median-geometric-mean")
    sfs = cds.obs["Size_Factor"].to_numpy()

    # Independent reference implementation.
    X = np.round(cds.X)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_geo = np.mean(np.log(X), axis=0)
    expected = np.empty(cds.n_obs)
    for i in range(cds.n_obs):
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.log(X[i, :]) - log_geo
        norm = norm[np.isfinite(norm)]
        expected[i] = np.exp(np.median(norm))
    expected[np.isnan(expected)] = 1.0
    np.testing.assert_allclose(sfs, expected, rtol=0, atol=1e-10)


def test_size_factor_weighted_median_matches_formula() -> None:
    """``"weighted-median"`` size factor matches the closed-form
    weighted-detection-rate version of the geometric-median formula."""
    cds = _nb_cds(n_cells=30, n_genes=15)
    estimate_size_factors(cds, method="weighted-median")
    sfs = cds.obs["Size_Factor"].to_numpy()

    X = np.round(cds.X)
    log_medians = np.array([np.log(np.median(X[:, j])) for j in range(X.shape[1])])
    weights = np.array([np.sum(X[:, j] > 0) / X.shape[0] for j in range(X.shape[1])])
    expected = np.empty(cds.n_obs)
    for i in range(cds.n_obs):
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = weights * (np.log(X[i, :]) - log_medians)
        norm = norm[np.isfinite(norm)]
        expected[i] = np.exp(np.mean(norm))
    expected[np.isnan(expected)] = 1.0
    np.testing.assert_allclose(sfs, expected, rtol=0, atol=1e-10)


def test_size_factor_median_subtracts_per_gene_median() -> None:
    """``"median"`` computes ``median(CM - gene_median)`` row-wise."""
    cds = _nb_cds(n_cells=30, n_genes=15)
    estimate_size_factors(cds, method="median")
    sfs = cds.obs["Size_Factor"].to_numpy()

    X = np.round(cds.X)
    expected = np.median(X - np.median(X, axis=0), axis=1)
    np.testing.assert_allclose(sfs, expected, rtol=0, atol=1e-10)


def test_size_factor_mode_method_not_implemented() -> None:
    """R's ``"mode"`` relies on ``estimate_t``; in Python this stub
    raises ``NotImplementedError`` so the user goes through
    :func:`estimate_t` → :func:`relative2abs` explicitly."""
    cds = _nb_cds(n_cells=20, n_genes=10)
    with pytest.raises(NotImplementedError, match="mode"):
        estimate_size_factors(cds, method="mode")


def test_size_factor_geometric_mean_total_matches_formula() -> None:
    """``"geometric-mean-total"`` = ``log(total_i) / mean(log(totals))``."""
    cds = _nb_cds(n_cells=30, n_genes=15)
    estimate_size_factors(cds, method="geometric-mean-total")
    sfs = cds.obs["Size_Factor"].to_numpy()

    X = np.round(cds.X)
    totals = X.sum(axis=1)
    expected = np.log(totals) / np.mean(np.log(totals))
    np.testing.assert_allclose(sfs, expected, rtol=0, atol=1e-10)


def test_detect_genes_defaults_min_expr_to_lower_detection_limit() -> None:
    """``min_expr=None`` → pull from ``uns['monocle2']['lower_detection_limit']``.
    Running with ``min_expr=None`` and with the explicit limit must
    produce the same ``num_cells_expressed`` counts."""
    cds_a = _nb_cds()
    detect_genes(cds_a, min_expr=None)
    cds_b = _nb_cds()
    detect_genes(
        cds_b, min_expr=cds_b.uns["monocle2"]["lower_detection_limit"],
    )
    np.testing.assert_array_equal(
        cds_a.var["num_cells_expressed"].to_numpy(),
        cds_b.var["num_cells_expressed"].to_numpy(),
    )


def test_estimate_dispersions_rejects_nan_size_factor() -> None:
    """``preprocess.py:400-403``: explicit NaN check before fitting."""
    cds = _nb_cds()
    estimate_size_factors(cds)
    cds.obs["Size_Factor"] = np.nan
    with pytest.raises(ValueError, match="Size_Factor|NaN"):
        estimate_dispersions(cds)


def test_estimate_dispersions_missing_formula_covariate_raises() -> None:
    """When ``model_formula_str`` names a column absent from
    ``adata.obs``, :func:`estimate_dispersions` raises ``ValueError``
    rather than silently falling through."""
    cds = _nb_cds()
    estimate_size_factors(cds)
    with pytest.raises(ValueError, match="not present"):
        estimate_dispersions(cds, model_formula_str="~NoSuchColumn")


# ---------------------------------------------------------------------------
# census.py: edge cases
# ---------------------------------------------------------------------------


def test_relative2abs_rejects_unsupported_method() -> None:
    """R has a ``method`` enum; Python's port currently implements only
    ``num_genes`` (see ``port_reports/monocle2/05_design.md``)."""
    cds = _nb_cds()
    with pytest.raises(NotImplementedError, match="num_genes"):
        relative2abs(cds, method="tpm_fraction")


def test_relative2abs_accepts_user_supplied_t_estimate() -> None:
    """User-supplied ``t_estimate`` bypasses the inner :func:`estimate_t`."""
    cds = _nb_cds(n_cells=20, n_genes=10)
    # Manufacture a plausible t vector — positive, finite, length n_cells.
    t_hat = np.full(cds.n_obs, 1.5)
    out = relative2abs(cds, method="num_genes", t_estimate=t_hat, return_all=True)
    np.testing.assert_allclose(out["t_estimate"], t_hat)


# ---------------------------------------------------------------------------
# _internal/_vgam.py: make_family / make_response / fallback paths
# ---------------------------------------------------------------------------


def test_make_family_returns_poisson_when_alpha_non_positive() -> None:
    """Matches VGAM ``negbinomial.size(size=Inf)`` degenerating to Poisson
    (``_vgam.py:108-112``)."""
    from monocle2py._internal._vgam import make_family
    import statsmodels.api as sm

    fam = make_family("negbinomial.size", alpha=0.0)
    assert isinstance(fam, sm.families.Poisson)
    fam_inf = make_family("negbinomial.size", alpha=math.inf)
    # ``math.isfinite(inf) is False`` → Poisson too.
    assert isinstance(fam_inf, sm.families.Poisson)


def test_make_family_falls_back_to_gaussian_for_unknown_name() -> None:
    """Unknown family names → Gaussian, matching monocle2's permissive
    default (``_vgam.py:116``)."""
    from monocle2py._internal._vgam import make_family
    import statsmodels.api as sm

    fam = make_family("something_weird", alpha=0.5)
    assert isinstance(fam, sm.families.Gaussian)


def test_make_family_binomialff_returns_binomial() -> None:
    """``binomialff`` branch in ``_vgam.make_family``."""
    from monocle2py._internal._vgam import make_family
    import statsmodels.api as sm

    assert isinstance(make_family("binomialff"), sm.families.Binomial)


def test_make_response_uninormal_returns_raw() -> None:
    """``uninormal`` / ``binomialff`` → identity; all others → log10."""
    from monocle2py._internal._vgam import make_response

    y = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        make_response(y, "uninormal", np.ones_like(y), relative_expr=False), y,
    )
    np.testing.assert_allclose(
        make_response(y, "Tobit", np.ones_like(y), relative_expr=False),
        np.log10(y),
    )


def test_calculate_nb_dispersion_hint_none_when_disp_func_missing() -> None:
    from monocle2py._internal._vgam import calculate_nb_dispersion_hint

    assert calculate_nb_dispersion_hint(None, np.array([1.0, 2.0])) is None


def test_calculate_nb_dispersion_hint_none_on_nonpositive_expr() -> None:
    from monocle2py._internal._vgam import calculate_nb_dispersion_hint

    # mean of a 0-filled vector is 0 → returns None.
    assert (
        calculate_nb_dispersion_hint(lambda q: np.array([1.0]), np.zeros(5))
        is None
    )


def test_calculate_nb_dispersion_hint_nan_from_disp_func_returns_none() -> None:
    from monocle2py._internal._vgam import calculate_nb_dispersion_hint

    assert (
        calculate_nb_dispersion_hint(
            lambda q: np.array([np.nan]), np.array([5.0, 10.0]),
        )
        is None
    )


def test_fit_glm_with_fallback_returns_none_for_non_nb_family_failure(
    monkeypatch,
) -> None:
    """Non-NB families have no backup in R — ``fit_glm_with_fallback``
    returns ``None``."""
    from monocle2py._internal import _vgam

    def _always_raises(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(_vgam, "fit_glm", _always_raises)
    result = _vgam.fit_glm_with_fallback(
        np.ones(5), np.ones((5, 2)),
        _vgam.make_family("uninormal"),
        family_name="uninormal",
    )
    assert result is None


def test_fit_tobit_rejects_malformed_input() -> None:
    from monocle2py._internal._vgam import fit_tobit

    with pytest.raises(ValueError, match="1-D"):
        fit_tobit(np.ones((3, 2)), np.ones((3, 2)), lower=0.0)


def test_fit_tobit_all_below_lower_drops_into_sigma_fallback() -> None:
    """When every y is at/below the lower bound, no uncensored
    residuals exist; the ``sigma0`` seed falls back to
    ``std(resid, ddof=1)`` of the clipped residuals (``_vgam.py:351-354``).
    Verify the log-likelihood is finite and params are the right shape."""
    from monocle2py._internal._vgam import fit_tobit

    y = np.full(10, 0.5)  # everything == lower
    X = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
    res = fit_tobit(y, X, lower=0.5)
    assert res.beta.shape == (2,)
    assert np.isfinite(res.llf)
    assert res.sigma > 0


# ---------------------------------------------------------------------------
# differential.py: FAIL outcomes + Tobit fit_models + response_matrix edges
# ---------------------------------------------------------------------------


def test_differential_gene_test_fit_fail_status_recorded(monkeypatch) -> None:
    """Force ``fit_glm_with_fallback`` to return ``None`` on every gene
    and verify the per-gene status becomes ``FAIL``."""
    from monocle2py import differential as diff

    monkeypatch.setattr(
        diff, "fit_glm_with_fallback",
        lambda y, X, family, family_name: None,
    )
    cds = _nb_cds(n_cells=20, n_genes=5)
    estimate_size_factors(cds)
    res = differential_gene_test(
        cds, full_model_formula_str="~Hours", verbose=True,
    )
    assert (res["status"] == "FAIL").all()


def test_differential_gene_test_lrt_exception_records_fail(monkeypatch) -> None:
    """Raising inside ``lrt`` should promote to FAIL via the inner
    ``try/except`` (matches R's ``tryCatch`` in compareModels)."""
    from monocle2py import differential as diff

    def _boom(*a, **kw):
        raise RuntimeError("synthetic lrt failure")

    monkeypatch.setattr(diff, "lrt", _boom)
    cds = _nb_cds(n_cells=20, n_genes=3)
    estimate_size_factors(cds)
    res = differential_gene_test(
        cds, full_model_formula_str="~Hours", verbose=True,
    )
    assert (res["status"] == "FAIL").all()


def test_fit_models_and_response_matrix_tobit_path() -> None:
    """Exercise the Tobit dispatch inside ``fit_models`` plus the
    log10→response inversion in ``response_matrix`` for Tobit."""
    rng = np.random.default_rng(3)
    n_cells, n_genes = 30, 5
    fpkm = rng.lognormal(size=(n_cells, n_genes)) * 4.0
    obs = pd.DataFrame({
        "Pseudotime": np.linspace(0, 10, n_cells),
    }, index=[f"C{i}" for i in range(n_cells)])
    var = pd.DataFrame({"gene_short_name": [f"G{i}" for i in range(n_genes)]},
                       index=[f"G{i}" for i in range(n_genes)])
    cds = new_cell_dataset(
        fpkm, pheno_data=obs, feature_data=var,
        lower_detection_limit=0.1,
        expression_family=tobit(lower=0.1),
    )
    fits = fit_models(cds, model_formula_str="~Pseudotime")
    assert set(fits.keys()) == set(cds.var_names)

    grid = pd.DataFrame({"Pseudotime": np.linspace(0, 10, 12)},
                        index=[f"P{i}" for i in range(12)])
    preds = response_matrix(
        fits, newdata=grid, formula_str="~Pseudotime",
    )
    assert preds.shape == (n_genes, 12)
    # Predictions are on the response scale (post-10^ inversion).
    assert np.all(np.isfinite(preds.to_numpy()))


def test_fit_models_negbinomial_joint_path() -> None:
    """Exercise the ``isinstance(family, Negbinomial)`` branch in
    ``fit_models`` (``differential.py:284-289``)."""
    cds = _nb_cds(n_cells=30, n_genes=4)
    estimate_size_factors(cds)
    # Swap family in-place to joint-estimation variant.
    cds.uns["monocle2"]["expression_family"] = "negbinomial"
    cds.uns["monocle2"].pop("expression_family_params", None)
    fits = fit_models(cds, model_formula_str="~Pseudotime")
    assert set(fits.keys()) == set(cds.var_names)


def test_response_matrix_all_none_fits_raises() -> None:
    """All-``None`` input → RuntimeError."""
    with pytest.raises(RuntimeError, match="All model fits are None"):
        response_matrix({"g1": None, "g2": None})


def test_response_matrix_requires_formula_when_newdata_given() -> None:
    """When ``newdata`` is supplied, ``formula_str`` must also be supplied."""
    from monocle2py import fit_models, response_matrix

    cds = _nb_cds(n_cells=30, n_genes=3)
    estimate_size_factors(cds)
    fits = fit_models(cds, model_formula_str="~Pseudotime")
    with pytest.raises(ValueError, match="formula_str"):
        response_matrix(fits, newdata=cds.obs)


def test_response_matrix_list_input_uses_enumerated_names() -> None:
    cds = _nb_cds(n_cells=30, n_genes=3)
    estimate_size_factors(cds)
    fits_dict = fit_models(cds, model_formula_str="~Pseudotime")
    fits_list = list(fits_dict.values())
    preds = response_matrix(fits_list)
    assert list(preds.index) == [str(i) for i in range(len(fits_list))]


# ---------------------------------------------------------------------------
# clustering.py: halo/skip paths + error paths
# ---------------------------------------------------------------------------


def test_cluster_cells_respects_skip_rho_sigma(
) -> None:
    """``skip_rho_sigma=True`` reuses cached rho/delta — exercises the
    branch at ``clustering.py:222-237``."""
    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, 2))
    # Four blobs for peak detection.
    X[:10] += [10, 0]; X[10:20] += [0, 10]
    X[20:30] += [10, 10]
    adata = AnnData(np.zeros((n, 1)))
    adata.obsm["X_dr"] = X

    cluster_cells(adata, rho_threshold=0.0, delta_threshold=2.0)
    first_rho = adata.obs["rho"].to_numpy().copy()
    # Second call with skip_rho_sigma should reuse cached rho/delta.
    cluster_cells(
        adata, skip_rho_sigma=True, rho_threshold=0.0, delta_threshold=2.0,
        verbose=True,
    )
    np.testing.assert_array_equal(adata.obs["rho"].to_numpy(), first_rho)


def test_cluster_cells_no_peaks_raises() -> None:
    """If no cell exceeds both thresholds, raise (``clustering.py:265-268``)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 2))
    adata = AnnData(np.zeros((20, 1)))
    adata.obsm["X_dr"] = X
    with pytest.raises(RuntimeError, match="peaks"):
        cluster_cells(adata, rho_threshold=1e9, delta_threshold=1e9)


def test_cluster_cells_large_n_recovers_ground_truth_blobs() -> None:
    """At n>448 cells, ``_estimate_dc`` subsamples pairwise distances
    (``clustering.py:32-36``). Verify correctness by planting four
    well-separated blobs and recovering them as ≥4 distinct clusters."""
    rng = np.random.default_rng(4)
    per = 130  # 4 blobs × 130 = 520 cells → triggers n > 448 branch
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
    X = np.vstack([
        rng.normal(loc=c, scale=0.3, size=(per, 2)) for c in centers
    ])
    adata = AnnData(np.zeros((X.shape[0], 1)))
    adata.obsm["X_dr"] = X
    cluster_cells(
        adata, rho_threshold=1e-6, delta_threshold=3.0, random_state=0,
    )
    # 4 real blobs → at least 4 peaks
    assert int(adata.obs["peaks"].sum()) >= 4


# ---------------------------------------------------------------------------
# dim_reduction.py: norm_method branches + error paths
# ---------------------------------------------------------------------------


def test_reduce_dimension_rejects_unknown_method() -> None:
    cds = _nb_cds(n_cells=30, n_genes=15)
    estimate_size_factors(cds)
    with pytest.raises(ValueError, match="DDRTree|tSNE|DDRTree.*tSNE"):
        reduce_dimension(cds, reduction_method="not_a_method")


def test_normalize_expr_data_rejects_unknown_norm_method() -> None:
    from monocle2py import normalize_expr_data

    cds = _nb_cds(n_cells=20, n_genes=5)
    estimate_size_factors(cds)
    with pytest.raises(ValueError, match="norm_method"):
        normalize_expr_data(cds, norm_method="not_a_norm")


def test_normalize_expr_data_tobit_log_matches_formula() -> None:
    """Tobit + ``norm_method='log'`` applies ``log2(x + pseudo_expr)``
    per-gene (``dim_reduction.py:111-113``). Verify the arithmetic."""
    from monocle2py import normalize_expr_data

    rng = np.random.default_rng(5)
    n_cells, n_genes = 20, 10
    fpkm = rng.lognormal(size=(n_cells, n_genes)) * 4.0
    obs = pd.DataFrame(index=[f"C{i}" for i in range(n_cells)])
    var = pd.DataFrame({"gene_short_name": [f"G{i}" for i in range(n_genes)]},
                       index=[f"G{i}" for i in range(n_genes)])
    cds = new_cell_dataset(
        fpkm, pheno_data=obs, feature_data=var,
        lower_detection_limit=0.1,
        expression_family=tobit(lower=0.1),
    )
    FM = normalize_expr_data(cds, norm_method="log", pseudo_expr=1.0)
    expected = np.log2(fpkm + 1.0).T  # genes x cells
    np.testing.assert_allclose(FM, expected, rtol=0, atol=1e-12)


def test_normalize_expr_data_tobit_rejects_invalid_norm_method() -> None:
    from monocle2py import normalize_expr_data

    rng = np.random.default_rng(6)
    fpkm = rng.lognormal(size=(10, 5)) * 4.0
    obs = pd.DataFrame(index=[f"C{i}" for i in range(10)])
    var = pd.DataFrame({"gene_short_name": [f"G{i}" for i in range(5)]},
                       index=[f"G{i}" for i in range(5)])
    cds = new_cell_dataset(
        fpkm, pheno_data=obs, feature_data=var,
        lower_detection_limit=0.1,
        expression_family=tobit(lower=0.1),
    )
    with pytest.raises(ValueError, match="Tobit"):
        normalize_expr_data(cds, norm_method="vstExprs")


def test_normalize_expr_data_gaussian_family_path() -> None:
    """Cover the ``uninormal`` (Gaussian raw-response) branch."""
    from monocle2py import normalize_expr_data

    rng = np.random.default_rng(7)
    X = rng.normal(size=(10, 5)) + 3.0
    obs = pd.DataFrame(index=[f"C{i}" for i in range(10)])
    var = pd.DataFrame({"gene_short_name": [f"G{i}" for i in range(5)]},
                       index=[f"G{i}" for i in range(5)])
    cds = new_cell_dataset(
        X, pheno_data=obs, feature_data=var,
        expression_family=gaussian_family(),
    )
    FM = normalize_expr_data(cds, norm_method="none")
    assert FM.shape == (5, 10)


def test_cal_ncenter_matches_r_formula() -> None:
    """``cal_ncenter`` mirrors R's ``round(2*limit*log(n)/(log(n)+log(limit)))``."""
    from monocle2py import cal_ncenter

    for n, limit in [(1000, 100), (500, 50), (10_000, 100), (200, 100)]:
        expected = round(2 * limit * np.log(n) / (np.log(n) + np.log(limit)))
        assert cal_ncenter(n, n_cells_limit=limit) == expected


def test_reduce_dimension_scaling_true_and_false_produce_different_projections() -> None:
    """``scaling=True`` z-scores per-gene before DDRTree; ``scaling=False``
    skips that step. The resulting projection must differ materially —
    proves the branch fires and has real effect."""
    cds_on = _nb_cds(n_cells=40, n_genes=20)
    cds_off = _nb_cds(n_cells=40, n_genes=20)
    estimate_size_factors(cds_on); estimate_size_factors(cds_off)
    reduce_dimension(
        cds_on, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, scaling=True, max_iter=5,
    )
    reduce_dimension(
        cds_off, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, scaling=False, max_iter=5,
    )
    # Different preprocessing → different embeddings.
    assert not np.allclose(cds_on.obsm["X_dr"], cds_off.obsm["X_dr"])


# ---------------------------------------------------------------------------
# ordering.py: error paths + root_cell fallback
# ---------------------------------------------------------------------------


def test_order_cells_rejects_unknown_root_state() -> None:
    cds = _nb_cds(n_cells=50, n_genes=20)
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=5,
    )
    order_cells(cds)
    with pytest.raises(RuntimeError, match="State"):
        order_cells(cds, root_state=999)


def test_order_cells_requires_state_set_before_root_state() -> None:
    cds = _nb_cds(n_cells=30, n_genes=15)
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=5,
    )
    # Drop the State column to force the "State/Pseudotime not yet set" path.
    cds.obs = cds.obs.drop(columns=[c for c in ("State", "Pseudotime") if c in cds.obs.columns])
    with pytest.raises(RuntimeError, match="State"):
        order_cells(cds, root_state=1)


def test_order_cells_reverse_mirrors_pseudotime_distribution() -> None:
    """``reverse=True`` picks the other end of the MST diameter as the
    root. For a tree, the resulting pseudotime vectors should be
    (approximately) ``max - pt`` of each other — verify the corr is
    strongly negative."""
    cds = _nb_cds(n_cells=50, n_genes=20)
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=5,
    )
    order_cells(cds)
    pt1 = cds.obs["Pseudotime"].to_numpy().copy()
    order_cells(cds, reverse=True)
    pt2 = cds.obs["Pseudotime"].to_numpy()
    # Tree geodesic distances mean reverse ≈ max - forward. On a
    # perfect line the Pearson correlation is exactly -1; on a tree
    # with branches it's strongly negative.
    corr = np.corrcoef(pt1, pt2)[0, 1]
    assert corr < -0.3


# ---------------------------------------------------------------------------
# beam.py: error paths
# ---------------------------------------------------------------------------


def test_build_branch_cell_dataset_rejects_unknown_progenitor_method() -> None:
    """``build_branch_cell_dataset`` rejects an invalid method name."""
    from monocle2py import build_branch_cell_dataset

    cds = _nb_cds(n_cells=50, n_genes=20)
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=5,
    )
    order_cells(cds)
    with pytest.raises(ValueError, match="progenitor_method"):
        build_branch_cell_dataset(cds, progenitor_method="bogus")


def test_build_branch_cell_dataset_requires_orderCells() -> None:
    from monocle2py import build_branch_cell_dataset

    cds = _nb_cds(n_cells=30, n_genes=10)
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=5,
    )
    # Intentionally skip order_cells.
    with pytest.raises(RuntimeError, match="order_cells|Pseudotime|State"):
        build_branch_cell_dataset(cds, branch_point=1)


def test_build_branch_cell_dataset_rejects_unknown_state() -> None:
    from monocle2py import build_branch_cell_dataset

    cds = _nb_cds(n_cells=50, n_genes=20)
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=5,
    )
    order_cells(cds)
    with pytest.raises(RuntimeError, match="No cells|State"):
        build_branch_cell_dataset(cds, branch_states=[999, 998])


# ---------------------------------------------------------------------------
# _uns.py: family_from_name / rebuild / error paths
# ---------------------------------------------------------------------------


def test_family_from_name_registry_roundtrips_all_vfamilies() -> None:
    """Every registered VGAM family name must resolve to the expected
    dataclass and carry the same ``vfamily`` string back."""
    from monocle2py.families import (
        family_from_name, GaussianFamily, Negbinomial, NegbinomialSize, Tobit,
    )
    cases = [
        ("uninormal", GaussianFamily),
        ("negbinomial", Negbinomial),
        ("negbinomial.size", NegbinomialSize),
        ("Tobit", Tobit),
        ("tobit", Tobit),
    ]
    for name, cls in cases:
        fam = family_from_name(name)
        assert isinstance(fam, cls)
        # ``Tobit`` has vfamily='Tobit' regardless of input casing.
        expected_vfamily = "Tobit" if cls is Tobit else name
        assert fam.vfamily == expected_vfamily


def test_get_state_errors_when_uns_missing() -> None:
    from monocle2py._uns import get_state

    adata = AnnData(np.zeros((3, 2)))
    with pytest.raises(KeyError, match="monocle2"):
        get_state(adata)


def test_ensure_state_rejects_non_dict_payload() -> None:
    from monocle2py._uns import ensure_state

    adata = AnnData(np.zeros((3, 2)))
    adata.uns["monocle2"] = 12  # bogus non-dict
    with pytest.raises(TypeError, match="dict"):
        ensure_state(adata)


# ---------------------------------------------------------------------------
# _download.py: resolution errors (no network)
# ---------------------------------------------------------------------------


def test_resolve_data_path_unknown_file_raises(tmp_path, monkeypatch) -> None:
    """A filename not in ``REGISTRY`` raises ``FileNotFoundError``."""
    from monocle2py import _download as dl

    monkeypatch.setattr(dl, "REGISTRY", {})
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

    with pytest.raises(FileNotFoundError, match="not registered"):
        dl.resolve_data_path("never_exists.h5ad")


def test_resolve_data_path_uses_cache_copy(tmp_path, monkeypatch) -> None:
    """Existing cache copy short-circuits the download."""
    from monocle2py import _download as dl

    home = tmp_path / "home"
    cache_dir = home / ".cache" / dl.CACHE_DIR_NAME
    cache_dir.mkdir(parents=True)
    cached = cache_dir / "cached.h5ad"
    cached.write_bytes(b"cached")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)

    resolved = dl.resolve_data_path("cached.h5ad")
    assert resolved == cached


def test_verify_sha256_mismatch_raises(tmp_path) -> None:
    """SHA mismatch → file deleted + RuntimeError (``_download.py:106-111``)."""
    from monocle2py._download import _verify_sha256

    path = tmp_path / "payload"
    path.write_bytes(b"hello world")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _verify_sha256(path, "deadbeef" * 8)
    assert not path.exists()


def test_verify_sha256_skips_when_expected_none(tmp_path) -> None:
    """No expected hash → no check."""
    from monocle2py._download import _verify_sha256

    path = tmp_path / "payload"
    path.write_bytes(b"hello")
    _verify_sha256(path, None)  # no-op
    assert path.exists()


def test_verify_sha256_passes_on_match(tmp_path) -> None:
    import hashlib
    from monocle2py._download import _verify_sha256

    payload = b"abc123"
    path = tmp_path / "payload"
    path.write_bytes(payload)
    _verify_sha256(path, hashlib.sha256(payload).hexdigest())
    assert path.exists()


# Plotting: pure render-and-type-check smoke tests were removed per
# user guidance. Keep plotting tests only when they exercise
# computational logic (validation, branch selection, transformations).
