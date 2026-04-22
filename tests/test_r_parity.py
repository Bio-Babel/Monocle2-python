"""R-parity tests ported from ``monocle2/tests/testthat``.

Each test below mirrors an R ``testthat`` case — same expectation, same
error surface where applicable. Deferred features (``CellTypeHierarchy``,
``classifyCells``, ``markerDiffTable``, 10X loader, ICA, DPT, a few
plotting helpers) are NOT ported; the corresponding R tests are skipped
by omission. The cases below therefore represent the portable subset of
the R suite, targeting behaviour this Python package is in scope for.

Every test names the source R file + ``test_that`` description so a
reader can trace back to the authoritative R expectation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from monocle2py import (
    cluster_cells,
    detect_genes,
    differential_gene_test,
    disp_table,
    estimate_dispersions,
    estimate_size_factors,
    new_cell_dataset,
    negbinomial_size,
    order_cells,
    plot_cell_clusters,
    plot_cell_trajectory,
    reduce_dimension,
    relative2abs,
    set_ordering_filter,
    tobit,
)


def _hsmm_like(n_cells: int = 60, n_genes: int = 50, seed: int = 0) -> AnnData:
    """Small NB-count fixture emulating the HSMM RPC matrix shape.

    We can't redistribute the HSMM fixture used by R's tests; a deterministic
    NB draw serves the same role: non-trivial mean/variance structure and
    enough cells to exercise the full pipeline (size factors → dispersions
    → DDRTree → orderCells → differentialGeneTest).
    """
    rng = np.random.default_rng(seed)
    mu_per_gene = rng.uniform(0.5, 30.0, size=n_genes)
    X = np.column_stack([
        rng.negative_binomial(n=3, p=3.0 / (3.0 + m), size=n_cells)
        for m in mu_per_gene
    ]).astype(float)
    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(n_genes)]
    # Add a ``Hours`` covariate so ``~Hours`` formula tests can exercise
    # the group-by / differential paths, matching R's HSMM fixture.
    obs = pd.DataFrame({
        "Hours": np.repeat([0, 24, 48, 72], n_cells // 4 + 1)[:n_cells],
    }, index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    return new_cell_dataset(
        X, pheno_data=obs, feature_data=var,
        lower_detection_limit=0.5,
        expression_family=negbinomial_size(),
    )


def _fpkm_like(n_cells: int = 40, n_genes: int = 40, seed: int = 0) -> AnnData:
    """Lognormal-FPKM fixture matching the HSMM-expr-matrix shape R uses."""
    rng = np.random.default_rng(seed)
    X = rng.lognormal(size=(n_cells, n_genes)) * 5.0
    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    return new_cell_dataset(
        X, pheno_data=obs, feature_data=var,
        lower_detection_limit=0.1,
        expression_family=tobit(lower=0.1),
    )


# ---------------------------------------------------------------------------
# basic_tests.R
# ---------------------------------------------------------------------------


def test_basic_cds_construction_and_detect_genes_setOrderingFilter() -> None:
    """Ports ``basic_tests.R`` — `Complete workflow for small data set is OK`.

    Build a CDS, run ``detectGenes``, then ``setOrderingFilter`` and verify
    the two per-gene columns landed on ``fData``. Plus: use a second
    ``detect_genes`` call — R runs it after a subset step to make sure
    the update is idempotent.
    """
    cds = _fpkm_like()
    assert cds is not None
    detect_genes(cds, min_expr=0.1)
    assert "num_cells_expressed" in cds.var.columns
    set_ordering_filter(cds, cds.var_names.tolist())
    assert cds.var["use_for_ordering"].dtype == bool
    # Idempotent second call.
    detect_genes(cds, min_expr=0.1)


# ---------------------------------------------------------------------------
# test.newCellDataSet.R
# ---------------------------------------------------------------------------


def test_newCellDataSet_rejects_non_matrix_input() -> None:
    """``test.newCellData throws error if cellData is not a matrix``.

    R errors when ``cellData`` is something non-matrix-like. The Python
    port accepts ``pandas.DataFrame`` (a natural expression container),
    so we check a genuinely invalid input: a plain string. ``_coerce_matrix``
    (``cell_dataset.py:102``) raises ``TypeError`` for anything that isn't
    ndarray / sparse / DataFrame / AnnData.
    """
    with pytest.raises(TypeError, match="NumPy|sparse|DataFrame|AnnData"):
        new_cell_dataset("not a matrix")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test.detectGenes.R
# ---------------------------------------------------------------------------


def test_detectGenes_runs_after_size_factors_and_dispersions() -> None:
    """``test.detectGenes.R`` — ``detectGenes works properly``.

    R's test calls ``estimateSizeFactors`` then ``estimateDispersions``
    then ``detectGenes`` and only checks no error is thrown.
    On the real HSMM data R's outlier refit converges; on our small
    synthetic fixture the refit can land on ``extraPois < 0`` (which R
    also errors on — ``expr_models.R:376-378``). Set ``remove_outliers
    =False`` to stabilise the synthetic path while still exercising
    the full disp-fit + detect-genes flow.
    """
    cds = _hsmm_like(n_cells=120, n_genes=80)
    estimate_size_factors(cds)
    estimate_dispersions(cds, remove_outliers=False)
    detect_genes(cds, min_expr=0.1)
    assert "num_cells_expressed" in cds.var.columns


# ---------------------------------------------------------------------------
# test.relative2abs.R
# ---------------------------------------------------------------------------


def test_relative2abs_default_returns_counts_matrix() -> None:
    """``relative2abs works with valid input and return_all = FALSE``."""
    cds = _fpkm_like()
    rpc = relative2abs(cds, method="num_genes")
    assert rpc.shape == (cds.n_obs, cds.n_vars)
    assert np.all(np.isfinite(rpc))


def test_relative2abs_return_all_true_yields_expected_keys() -> None:
    """``relative2abs works with valid input and return_all = TRUE``.

    R returns a named list with ``norm_cds``, ``t_estimate``,
    ``expected_total_mRNAs``; the Python port returns a dict with the
    same three keys.
    """
    cds = _fpkm_like()
    out = relative2abs(cds, method="num_genes", return_all=True)
    assert set(out.keys()) == {"norm_cds", "t_estimate", "expected_total_mRNAs"}
    assert out["norm_cds"].shape == (cds.n_obs, cds.n_vars)
    assert out["t_estimate"].shape == (cds.n_obs,)


def test_relative2abs_rejects_infinite_t_estimate() -> None:
    """``throws error if infinite parameter is present``.

    R builds ``c(t_estimate, volume, dilution, detection_threshold)``
    and errors if any are non-finite. Python only consumes ``t_estimate``
    + ``expected_capture_rate`` (the ``num_genes`` path); the finite
    check fires on those.
    """
    cds = _fpkm_like()
    with pytest.raises(ValueError, match="finite"):
        relative2abs(
            cds, method="num_genes",
            t_estimate=np.full(cds.n_obs, np.inf),
        )


def test_relative2abs_t_estimate_shape_mismatch() -> None:
    """Python-side enforcement: ``t_estimate`` must be length ``n_cells``."""
    cds = _fpkm_like()
    with pytest.raises(ValueError, match="shape"):
        relative2abs(
            cds, method="num_genes",
            t_estimate=np.array([1.0, 2.0]),
        )


# ---------------------------------------------------------------------------
# test.dispersionTable.R
# ---------------------------------------------------------------------------


def test_dispersionTable_errors_without_fit() -> None:
    """R's ``dispersionTable validates input`` expects an error when no
    dispersion model has been fit yet. Python raises ``RuntimeError``.
    """
    cds = _hsmm_like()
    with pytest.raises(RuntimeError, match="dispersion"):
        disp_table(cds)


# ---------------------------------------------------------------------------
# test.estimateDispersions.R
# ---------------------------------------------------------------------------


def test_estimateDispersions_rejects_tobit_family() -> None:
    """R's ``estimateDispersion() properly validates its input``.

    R errors when the expression family is not ``negbinomial`` /
    ``negbinomial.size``. Python raises ``ValueError``.
    """
    cds = _fpkm_like()  # Tobit family
    estimate_size_factors(cds)
    with pytest.raises(ValueError, match="negbinomial"):
        estimate_dispersions(cds)


# ---------------------------------------------------------------------------
# test.clusterCells.R
# ---------------------------------------------------------------------------


def test_clusterCells_runs_on_tsne_reduction() -> None:
    """``clusterCells functions normally in vignette``.

    R runs DDRTree-adjacent densityPeak clustering at the end of a
    standard tSNE pipeline. Python exercises the same flow on the
    synthetic fixture.
    """
    cds = _hsmm_like(n_cells=80, n_genes=40)
    estimate_size_factors(cds)
    detect_genes(cds, min_expr=0.1)
    mask = cds.var["num_cells_expressed"] > 3
    set_ordering_filter(cds, cds.var_names[mask].tolist())
    reduce_dimension(
        cds, reduction_method="tSNE", num_dim=5,
        perplexity=10, auto_param_selection=False,
    )
    cluster_cells(cds, num_clusters=2)
    assert "Cluster" in cds.obs.columns
    assert cds.obs["Cluster"].nunique() >= 1


# ---------------------------------------------------------------------------
# test.reduceDimension.R
# ---------------------------------------------------------------------------


def test_reduceDimension_tsne_runs_in_vignette_path() -> None:
    """``reduceDimension works properly`` — vignette-style invocation."""
    cds = _hsmm_like(n_cells=60, n_genes=40)
    estimate_size_factors(cds)
    detect_genes(cds, min_expr=0.1)
    mask = cds.var["num_cells_expressed"] > 3
    set_ordering_filter(cds, cds.var_names[mask].tolist())
    reduce_dimension(
        cds, max_components=2, num_dim=6,
        reduction_method="tSNE", perplexity=10,
        auto_param_selection=False,
    )
    assert cds.obsm["X_dr"].shape == (cds.n_obs, 2)


# ---------------------------------------------------------------------------
# test.orderCells.R
# ---------------------------------------------------------------------------


def test_orderCells_rejects_bare_integer() -> None:
    """``orderCells throws error if cds is not type 'CellDataSet'``.

    R errors when the first argument is not a CellDataSet; the Python
    signature types ``adata: AnnData``, so passing ``8`` raises
    ``AttributeError`` / ``TypeError`` as soon as an AnnData accessor
    is hit.
    """
    with pytest.raises(Exception):
        order_cells(8)  # type: ignore[arg-type]


def test_orderCells_vignette_ddrtree() -> None:
    """``orderCells works properly in vignette setting``.

    Runs DDRTree → orderCells and asserts Pseudotime / State columns
    are populated — same assertion R's test makes (``expect_error(...,
    NA)`` just checks no error).
    """
    cds = _hsmm_like(n_cells=120, n_genes=80)
    estimate_size_factors(cds)
    estimate_dispersions(cds, remove_outliers=False)
    detect_genes(cds, min_expr=0.1)
    mask = cds.var["num_cells_expressed"] > 3
    set_ordering_filter(cds, cds.var_names[mask].tolist())
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=10, verbose=False,
    )
    order_cells(cds)
    assert "Pseudotime" in cds.obs.columns
    assert "State" in cds.obs.columns


# ---------------------------------------------------------------------------
# test.differentialGeneTest.R
# ---------------------------------------------------------------------------


def test_differentialGeneTest_vignette_hours_formula() -> None:
    """``differentialGeneTest functions under standard conditions``.

    R fits ``~Media`` on the myoblast subset; we use ``~Hours`` here
    because the synthetic fixture has a ``Hours`` covariate. Only
    requirement: no crash + expected columns.
    """
    cds = _hsmm_like(n_cells=60, n_genes=20)
    estimate_size_factors(cds)
    info = differential_gene_test(cds, full_model_formula_str="~Hours")
    for col in ("status", "pval", "qval"):
        assert col in info.columns


def test_differentialGeneTest_rejects_inf_in_covariate() -> None:
    """``Error is thrown if Inf is located in pData``.

    R raises ``Error: Inf, NaN, or NA values were located in pData of cds
    in columns mentioned in model terms``. Python's ``_validate_terms``
    raises ``ValueError`` for the same condition.
    """
    cds = _hsmm_like(n_cells=40, n_genes=10)
    estimate_size_factors(cds)
    cds.obs["Hours"] = cds.obs["Hours"].astype(float)
    cds.obs.iloc[0, cds.obs.columns.get_loc("Hours")] = np.inf
    with pytest.raises(ValueError, match="Inf|NaN"):
        differential_gene_test(cds, full_model_formula_str="~Hours")


def test_differentialGeneTest_rejects_nan_in_covariate() -> None:
    """``Error is thrown if NaN is located in pData``."""
    cds = _hsmm_like(n_cells=40, n_genes=10)
    estimate_size_factors(cds)
    cds.obs["Hours"] = cds.obs["Hours"].astype(float)
    cds.obs.iloc[0, cds.obs.columns.get_loc("Hours")] = np.nan
    with pytest.raises(ValueError, match="Inf|NaN"):
        differential_gene_test(cds, full_model_formula_str="~Hours")


def test_differentialGeneTest_rejects_missing_size_factor() -> None:
    """``Error is thrown if ... Size_Factor is null``.

    R checks that Size_Factor is populated when ``relative_expr=TRUE``
    and the expression family is ``negbinomial`` / ``negbinomial.size``.
    """
    cds = _hsmm_like(n_cells=40, n_genes=10)
    # Intentionally skip estimate_size_factors — column is absent.
    with pytest.raises((ValueError, KeyError), match="[Ss]ize.?[Ff]actor|Size_Factor"):
        differential_gene_test(cds, full_model_formula_str="~Hours")


def test_differentialGeneTest_rejects_nan_size_factor() -> None:
    """``Error is thrown if ... sum of size factors of cds is NA``."""
    cds = _hsmm_like(n_cells=40, n_genes=10)
    estimate_size_factors(cds)
    cds.obs["Size_Factor"] = np.nan
    with pytest.raises(ValueError, match="Size_Factor|NaN"):
        differential_gene_test(cds, full_model_formula_str="~Hours")


# ---------------------------------------------------------------------------
# test.plot_cell_clusters.R  /  test.plotCellTrajectory.R
# ---------------------------------------------------------------------------


def test_plot_cell_clusters_runs_after_tsne_and_clustering() -> None:
    """``plot_cell_clusters functions in vignette``."""
    from ggplot2_py import GGPlot
    cds = _hsmm_like(n_cells=60, n_genes=25)
    estimate_size_factors(cds)
    detect_genes(cds, min_expr=0.1)
    mask = cds.var["num_cells_expressed"] > 3
    set_ordering_filter(cds, cds.var_names[mask].tolist())
    reduce_dimension(
        cds, reduction_method="tSNE", num_dim=5, perplexity=10,
        auto_param_selection=False,
    )
    cluster_cells(cds, num_clusters=2)
    g = plot_cell_clusters(cds, 1, 2, color_by="Cluster")
    assert isinstance(g, GGPlot)


def test_plot_cell_trajectory_errors_before_reduce_dimension() -> None:
    """``plot_cell_trajectory throws an error if reduceDimensionality
    hasn't been called yet``."""
    cds = _hsmm_like(n_cells=40, n_genes=20)
    with pytest.raises(RuntimeError, match="reduce_dimension|X_dr|ddrtree"):
        plot_cell_trajectory(cds)
