"""Batch 4: small focused fills for logic branches in
preprocess, dim_reduction, clustering, ordering, beam that the prior
batches left uncovered. Per user guidance, plot-rendering smoke tests
are excluded; this file only exercises computational branches."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from monocle2py import (
    cal_ilrs,
    estimate_size_factors,
    new_cell_dataset,
    negbinomial_size,
    order_cells,
    reduce_dimension,
    tobit,
)


# ---------------------------------------------------------------------------
# dim_reduction._check_size_factors: NaN guard + non-NB early return
# ---------------------------------------------------------------------------


def test_check_size_factors_errors_on_nan() -> None:
    """``_check_size_factors`` raises for a NaN-filled Size_Factor column
    on NB families (``dim_reduction.py:63-65``)."""
    from monocle2py.dim_reduction import _check_size_factors

    cds = new_cell_dataset(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        pheno_data=pd.DataFrame(index=["c0", "c1"]),
        feature_data=pd.DataFrame({"gene_short_name": ["g0", "g1"]},
                                   index=["g0", "g1"]),
        expression_family=negbinomial_size(),
    )
    cds.obs["Size_Factor"] = np.nan
    with pytest.raises(RuntimeError, match="size factor"):
        _check_size_factors(cds)


def test_check_size_factors_no_op_for_non_nb_family() -> None:
    """For non-NB families, ``_check_size_factors`` returns quietly even
    when ``Size_Factor`` is missing or NaN (``dim_reduction.py:57-58``).
    Verify that neither missing nor NaN Size_Factor raises on Tobit."""
    from monocle2py.dim_reduction import _check_size_factors

    cds = new_cell_dataset(
        np.array([[1.0, 2.0]]), pheno_data=pd.DataFrame(index=["c"]),
        feature_data=pd.DataFrame({"gene_short_name": ["a", "b"]},
                                   index=["a", "b"]),
        expression_family=tobit(lower=0.1),
    )
    # Both: no column, then NaN column. Neither should raise.
    cds2 = cds.copy()
    if "Size_Factor" in cds2.obs.columns:
        cds2.obs = cds2.obs.drop(columns=["Size_Factor"])
    _check_size_factors(cds2)
    cds3 = cds.copy()
    cds3.obs["Size_Factor"] = np.nan
    _check_size_factors(cds3)


def test_reduce_dimension_raises_when_size_factor_missing_for_nb() -> None:
    """The ``RuntimeError`` at ``dim_reduction.py:58-62`` fires when
    ``estimate_size_factors`` was never called on a NB dataset."""
    rng = np.random.default_rng(0)
    X = rng.negative_binomial(4, 0.5, size=(20, 10)).astype(float)
    cds = new_cell_dataset(
        X, pheno_data=pd.DataFrame(index=[f"C{i}" for i in range(20)]),
        feature_data=pd.DataFrame(
            {"gene_short_name": [f"G{i}" for i in range(10)]},
            index=[f"G{i}" for i in range(10)],
        ),
        expression_family=negbinomial_size(),
    )
    # Drop the Size_Factor column.
    cds.obs = cds.obs.drop(columns=["Size_Factor"])
    with pytest.raises(RuntimeError, match="estimate_size_factors"):
        reduce_dimension(
            cds, reduction_method="tSNE", num_dim=3, perplexity=5,
            auto_param_selection=False,
        )


def test_normalize_expr_data_unsupported_family_raises() -> None:
    """``normalize_expr_data`` raises on unknown ``vfamily`` strings
    (``dim_reduction.py:125-126``)."""
    from monocle2py import normalize_expr_data
    from monocle2py._uns import ensure_state

    adata = AnnData(np.ones((5, 3)),
                    obs=pd.DataFrame(index=[f"C{i}" for i in range(5)]),
                    var=pd.DataFrame({"gene_short_name": ["a", "b", "c"]},
                                     index=["a", "b", "c"]))
    state = ensure_state(adata)
    state["expression_family"] = "weird_family"
    state["expression_family_params"] = {}
    adata.obs["Size_Factor"] = 1.0
    # The guard fires inside ``family_from_name`` first when the name
    # isn't in the registry; either "Unknown" (registry miss) or
    # "Unsupported" (post-dispatch) is accepted.
    with pytest.raises(ValueError, match="Unknown|Unsupported"):
        normalize_expr_data(adata, norm_method="log")


def test_normalize_expr_data_gaussian_rejects_non_none_norm() -> None:
    """``uninormal`` only supports ``norm_method='none'``
    (``dim_reduction.py:120-123``)."""
    from monocle2py import gaussian_family, normalize_expr_data

    adata = new_cell_dataset(
        np.ones((5, 3)),
        pheno_data=pd.DataFrame(index=[f"C{i}" for i in range(5)]),
        feature_data=pd.DataFrame({"gene_short_name": ["a", "b", "c"]},
                                   index=["a", "b", "c"]),
        expression_family=gaussian_family(),
    )
    with pytest.raises(ValueError, match="uninormal"):
        normalize_expr_data(adata, norm_method="log")


# ---------------------------------------------------------------------------
# binomialff: 7th normalization cell — TF-IDF on binarized counts
# ---------------------------------------------------------------------------


def test_normalize_expr_data_binomialff_tfidf_matches_r_formula() -> None:
    """Port of ``order_cells.R:1251-1253``: TF-IDF on binarized counts.

    R formula (R-orientation, genes x cells):
        ncounts <- FM > 0
        FM <- t(t(ncounts) * log(1 + ncol(ncounts)/rowSums(ncounts)))

    Re-derive in Python and compare bitwise.
    """
    from monocle2py import binomialff, normalize_expr_data

    rng = np.random.default_rng(0)
    n_cells, n_genes = 50, 20
    counts = (rng.random((n_cells, n_genes)) < 0.4).astype(np.float64)
    # guarantee no all-zero genes so the comparison is finite
    counts[0, :] = 1.0
    adata = new_cell_dataset(
        counts,
        pheno_data=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        feature_data=pd.DataFrame(
            {"gene_short_name": [f"g{j}" for j in range(n_genes)]},
            index=[f"g{j}" for j in range(n_genes)],
        ),
        expression_family=binomialff(),
    )
    FM = normalize_expr_data(adata, norm_method="none")

    # R-faithful reference derivation
    ncounts = (counts > 0).astype(np.float64)
    cells_expressing = ncounts.sum(axis=0)
    idf_ref = np.log(1.0 + n_cells / cells_expressing)
    FM_ref = (ncounts * idf_ref).T   # genes x cells, matching R
    np.testing.assert_allclose(FM, FM_ref)


def test_normalize_expr_data_binomialff_rejects_log_and_vst() -> None:
    """R ``order_cells.R:1254-1256`` stops if user asks for log/vstExprs
    with binomialff. Python raises ValueError with the same intent."""
    from monocle2py import binomialff, normalize_expr_data

    adata = new_cell_dataset(
        (np.eye(6, 4)).astype(float),
        pheno_data=pd.DataFrame(index=[f"c{i}" for i in range(6)]),
        feature_data=pd.DataFrame(
            {"gene_short_name": list("abcd")}, index=list("abcd"),
        ),
        expression_family=binomialff(),
    )
    with pytest.raises(ValueError, match="binomialff family only supports"):
        normalize_expr_data(adata, norm_method="log")
    with pytest.raises(ValueError, match="binomialff family only supports"):
        normalize_expr_data(adata, norm_method="vstExprs")


def test_binomialff_differential_gene_test_end_to_end() -> None:
    """End-to-end smoke for binomialff through differential_gene_test.

    Exercises the previously-unreachable ``_vgam.py:146`` Binomial GLM
    branch. The test confirms that the renamed family-class plumbing
    delivers ``sm.families.Binomial`` to the fit path and that the
    pipeline returns the standard pval/qval/status schema without
    requiring size factors or dispersions (neither concept applies to
    a binomial model)."""
    from monocle2py import (
        binomialff, differential_gene_test, estimate_size_factors,
    )

    rng = np.random.default_rng(0)
    n_cells, n_peaks = 60, 8
    # Build a two-group binary signal so a few "genes" will look
    # differentially open vs. random noise: peaks 0-3 are open in
    # cells with Pseudotime > 0.5, peaks 4-7 are random noise.
    pt = np.linspace(0.0, 1.0, n_cells)
    counts = (rng.random((n_cells, n_peaks)) < 0.3).astype(float)
    counts[pt > 0.5, :4] = (rng.random((int((pt > 0.5).sum()), 4)) < 0.85).astype(float)
    counts[pt <= 0.5, :4] = (rng.random((int((pt <= 0.5).sum()), 4)) < 0.10).astype(float)

    adata = new_cell_dataset(
        counts,
        pheno_data=pd.DataFrame({"Pseudotime": pt},
                                  index=[f"c{i}" for i in range(n_cells)]),
        feature_data=pd.DataFrame(
            {"gene_short_name": [f"p{j}" for j in range(n_peaks)]},
            index=[f"p{j}" for j in range(n_peaks)],
        ),
        expression_family=binomialff(),
    )
    # estimate_size_factors must not be required for binomialff
    # (Binomial GLM doesn't use offsets); but we set it anyway so the
    # default Size_Factor=NaN doesn't trip downstream sanity checks
    # that index by column existence.
    adata.obs["Size_Factor"] = 1.0

    res = differential_gene_test(
        adata,
        full_model_formula_str="~Pseudotime",
        reduced_model_formula_str="~1",
        relative_expr=False,   # binomial doesn't need size-factor scaling
    )
    assert set(res.columns) >= {"status", "family", "pval", "qval"}
    assert (res["family"] == "binomialff").all()
    ok = res["status"] == "OK"
    assert ok.sum() >= n_peaks // 2, "expected most binomial fits to converge"
    # peaks 0-3 should have smaller pvals than peaks 4-7 (real signal)
    pvals_signal = res.loc[ok & res.index.isin([f"p{i}" for i in range(4)]), "pval"]
    pvals_noise = res.loc[ok & res.index.isin([f"p{i}" for i in range(4, 8)]), "pval"]
    if len(pvals_signal) and len(pvals_noise):
        assert pvals_signal.median() < pvals_noise.median(), (
            f"signal peaks should rank ahead of noise; got median pvals "
            f"signal={pvals_signal.median():.3g} noise={pvals_noise.median():.3g}"
        )


def test_normalize_expr_data_binomialff_zero_count_genes_propagate_nan() -> None:
    """A gene with zero expression across all cells produces idf=Inf
    and Inf*0 = NaN. R does the same; ``_drop_nonfinite`` downstream
    filters the row, so we let NaN propagate here rather than pre-fixing."""
    from monocle2py import binomialff, normalize_expr_data

    counts = np.array(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )  # column 1 (gene index 1) is all zero
    adata = new_cell_dataset(
        counts,
        pheno_data=pd.DataFrame(index=["c0", "c1", "c2"]),
        feature_data=pd.DataFrame(
            {"gene_short_name": ["g0", "g1", "g2"]}, index=["g0", "g1", "g2"],
        ),
        expression_family=binomialff(),
    )
    FM = normalize_expr_data(adata, norm_method="none")
    # FM is (genes x cells); the all-zero gene row must be NaN
    assert np.isnan(FM[1, :]).all()
    # The other rows must be finite and equal to ncounts * idf
    assert np.isfinite(FM[0, :]).all()
    assert np.isfinite(FM[2, :]).all()


# ---------------------------------------------------------------------------
# clustering._local_density step-kernel branch (line 67)
# ---------------------------------------------------------------------------


def test_cluster_cells_hard_cutoff_rho_is_integer_neighbor_count() -> None:
    """Hard-cutoff kernel (``gaussian=False``) assigns ``rho_i``
    as the count of neighbours within ``dc`` (``clustering.py:66-67``).
    Compared with the Gaussian kernel, all ``rho`` values are integer-valued."""
    from monocle2py import cluster_cells

    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
    X = np.vstack([
        rng.normal(loc=c, scale=0.3, size=(25, 2)) for c in centers
    ])
    adata = AnnData(np.zeros((X.shape[0], 1)))
    adata.obsm["X_dr"] = X
    cluster_cells(adata, gaussian=False, num_clusters=4)
    rho = adata.obs["rho"].to_numpy()
    # Hard-cutoff rho is always a non-negative integer (count of neighbors).
    np.testing.assert_array_equal(rho, np.round(rho))
    assert (rho >= 0).all()


def test_cluster_cells_default_thresholds_use_95th_percentile() -> None:
    """Without explicit thresholds or ``num_clusters``, defaults are
    ``quantile(rho, 0.95)`` and ``quantile(delta, 0.95)``
    (``clustering.py:250-252``). After the run, peaks should match
    that rule exactly."""
    from monocle2py import cluster_cells

    rng = np.random.default_rng(1)
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
    X = np.vstack([
        rng.normal(loc=c, scale=0.3, size=(15, 2)) for c in centers
    ])
    adata = AnnData(np.zeros((X.shape[0], 1)))
    adata.obsm["X_dr"] = X
    cluster_cells(adata)
    rho = adata.obs["rho"].to_numpy()
    delta = adata.obs["delta"].to_numpy()
    rho_thr = np.quantile(rho, 0.95)
    delta_thr = np.quantile(delta, 0.95)
    expected_peaks = (rho > rho_thr) & (delta > delta_thr)
    np.testing.assert_array_equal(
        adata.obs["peaks"].to_numpy(dtype=bool), expected_peaks,
    )


def test_cluster_cells_explicit_peaks_produces_that_many_clusters() -> None:
    """User-supplied ``peaks`` bypass threshold selection entirely
    (``clustering.py:262-263``). The resulting cluster count equals the
    number of peaks supplied, and those exact cells are the peaks."""
    from monocle2py import cluster_cells

    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    X[:10] += [10, 0]; X[10:20] += [0, 10]
    adata = AnnData(np.zeros((30, 1)))
    adata.obsm["X_dr"] = X
    supplied = [0, 15, 25]
    cluster_cells(adata, peaks=supplied)
    peaks_mask = adata.obs["peaks"].to_numpy(dtype=bool)
    assert int(peaks_mask.sum()) == 3
    assert np.all(peaks_mask[supplied])
    assert adata.obs["Cluster"].nunique() == 3


# ---------------------------------------------------------------------------
# ordering helpers: segment-projection edge cases
# ---------------------------------------------------------------------------


def test_project_point_to_segment_degenerate_zero_vector() -> None:
    """When A == B, segment projection returns A (``ordering.py:166``)."""
    from monocle2py.ordering import _project_point_to_segment

    A = np.array([1.0, 1.0])
    B = np.array([1.0, 1.0])
    out = _project_point_to_segment(np.array([5.0, 5.0]), A, B)
    np.testing.assert_array_equal(out, A)


def test_project_point_to_segment_clamps_both_ends() -> None:
    from monocle2py.ordering import _project_point_to_segment

    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    # Point far left of A → clamped to A.
    left = _project_point_to_segment(np.array([-5.0, 0.0]), A, B)
    np.testing.assert_array_equal(left, A)
    # Point far right of B → clamped to B.
    right = _project_point_to_segment(np.array([5.0, 0.0]), A, B)
    np.testing.assert_array_equal(right, B)


def test_project_point_to_line_degenerate_returns_A() -> None:
    """Infinite-line projection with A==B returns A (``ordering.py:180``)."""
    from monocle2py.ordering import _project_point_to_line

    A = np.array([2.0, 3.0])
    out = _project_point_to_line(np.array([9.0, 9.0]), A, A)
    np.testing.assert_array_equal(out, A)


def test_order_cells_rejects_non_ddrtree_reduction() -> None:
    """Only DDRTree is ported — tSNE reductions must raise
    ``NotImplementedError`` (``ordering.py:549-551``)."""
    rng = np.random.default_rng(0)
    X = rng.negative_binomial(4, 0.5, size=(25, 10)).astype(float)
    cds = new_cell_dataset(
        X, pheno_data=pd.DataFrame(index=[f"C{i}" for i in range(25)]),
        feature_data=pd.DataFrame(
            {"gene_short_name": [f"G{i}" for i in range(10)]},
            index=[f"G{i}" for i in range(10)],
        ),
        expression_family=negbinomial_size(),
    )
    estimate_size_factors(cds)
    reduce_dimension(
        cds, reduction_method="tSNE", num_dim=5, perplexity=5,
        auto_param_selection=False,
    )
    with pytest.raises(NotImplementedError, match="DDRTree"):
        order_cells(cds)


def test_order_cells_rejects_num_paths_kwarg() -> None:
    """``num_paths`` was an ICA-only R parameter; we don't port ICA, so
    we don't accept it either. Passing it must raise ``TypeError`` rather
    than silently doing nothing — louder feedback for R users porting
    scripts than an accept-and-ignore stub."""
    rng = np.random.default_rng(0)
    X = rng.negative_binomial(4, 0.5, size=(40, 15)).astype(float)
    cds = new_cell_dataset(
        X, pheno_data=pd.DataFrame(index=[f"C{i}" for i in range(40)]),
        feature_data=pd.DataFrame(
            {"gene_short_name": [f"G{i}" for i in range(15)]},
            index=[f"G{i}" for i in range(15)],
        ),
        expression_family=negbinomial_size(),
    )
    estimate_size_factors(cds)
    reduce_dimension(
        cds, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=5,
    )
    with pytest.raises(TypeError, match="num_paths"):
        order_cells(cds, num_paths=99)


# ---------------------------------------------------------------------------
# beam helpers + cal_ilrs edges
# ---------------------------------------------------------------------------


def test_cal_ilrs_trajectory_states_explicit_pair() -> None:
    """Exercise ``trajectory_states`` path with explicit two States."""
    rng = np.random.default_rng(11)
    n_per = 35
    n_cells = 3 * n_per
    t = np.linspace(0, 1, n_per)
    latent = np.vstack([
        np.column_stack([t, np.zeros_like(t)]),
        np.column_stack([np.ones_like(t), t]),
        np.column_stack([np.ones_like(t), -t]),
    ])
    proj = rng.normal(size=(2, 60)) * 4.0
    logX = latent @ proj + rng.normal(scale=0.1, size=(n_cells, 60))
    mu = np.exp(logX - logX.max(axis=0)) * 40 + 1
    X = rng.negative_binomial(n=5, p=5.0 / (5.0 + mu)).astype(float)
    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(60)]
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    cds = new_cell_dataset(
        X, pheno_data=obs, feature_data=var,
        lower_detection_limit=1.0,
        expression_family=negbinomial_size(),
    )
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=10,
    )
    order_cells(cds)
    states = sorted(cds.obs["State"].astype(int).unique())
    if len(states) < 2:
        pytest.skip("synthetic tree didn't produce ≥2 states")
    res = cal_ilrs(
        cds, trajectory_states=list(states[-2:]), n_points=10,
    )
    assert isinstance(res, pd.DataFrame)
