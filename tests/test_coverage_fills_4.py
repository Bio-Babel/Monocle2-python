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
    """``gaussianff`` only supports ``norm_method='none'``
    (``dim_reduction.py:120-123``)."""
    from monocle2py import gaussian_family, normalize_expr_data

    adata = new_cell_dataset(
        np.ones((5, 3)),
        pheno_data=pd.DataFrame(index=[f"C{i}" for i in range(5)]),
        feature_data=pd.DataFrame({"gene_short_name": ["a", "b", "c"]},
                                   index=["a", "b", "c"]),
        expression_family=gaussian_family(),
    )
    with pytest.raises(ValueError, match="gaussianff"):
        normalize_expr_data(adata, norm_method="log")


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


def test_order_cells_num_paths_is_ignored_for_ddrtree() -> None:
    """R's ``orderCells`` ignores ``num_paths`` for DDRTree (only ICA
    uses it). Python must produce the same Pseudotime/State whether or
    not ``num_paths`` is passed (``ordering.py:552-554``)."""
    def _fresh():
        rng = np.random.default_rng(0)
        X = rng.negative_binomial(4, 0.5, size=(40, 15)).astype(float)
        c = new_cell_dataset(
            X, pheno_data=pd.DataFrame(index=[f"C{i}" for i in range(40)]),
            feature_data=pd.DataFrame(
                {"gene_short_name": [f"G{i}" for i in range(15)]},
                index=[f"G{i}" for i in range(15)],
            ),
            expression_family=negbinomial_size(),
        )
        estimate_size_factors(c)
        reduce_dimension(
            c, reduction_method="DDRTree",
            auto_param_selection=False, max_iter=5,
        )
        return c

    c0 = _fresh(); order_cells(c0)
    c1 = _fresh(); order_cells(c1, num_paths=99)
    np.testing.assert_allclose(
        c0.obs["Pseudotime"].to_numpy(), c1.obs["Pseudotime"].to_numpy(),
    )
    np.testing.assert_array_equal(
        c0.obs["State"].astype(int).to_numpy(),
        c1.obs["State"].astype(int).to_numpy(),
    )


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
