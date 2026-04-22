"""Batch 2 coverage fills: plotting helpers, BEAM paths, ordering fallback.

Every test exercises a specific branch the earlier suites skipped. Reasons
are inlined so future readers know what each test is guarding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from monocle2py import (
    beam,
    branch_test,
    build_branch_cell_dataset,
    cal_ilrs,
    detect_genes,
    estimate_size_factors,
    gen_smooth_curves,
    new_cell_dataset,
    negbinomial_size,
    order_cells,
    plot_cell_clusters,
    plot_cell_trajectory,
    plot_complex_cell_trajectory,
    plot_genes_branched_heatmap,
    plot_genes_branched_pseudotime,
    plot_genes_in_pseudotime,
    plot_multiple_branches_heatmap,
    plot_multiple_branches_pseudotime,
    plot_pseudotime_heatmap,
    reduce_dimension,
    set_ordering_filter,
)


def _branching_cds(
    n_per_branch: int = 30, n_genes: int = 50, seed: int = 1,
) -> AnnData:
    """Y-trajectory synthetic with two clean branches."""
    rng = np.random.default_rng(seed)
    n_cells = 3 * n_per_branch
    t = np.linspace(0, 1, n_per_branch)
    latent = np.vstack([
        np.column_stack([t, np.zeros_like(t)]),
        np.column_stack([np.ones_like(t), t]),
        np.column_stack([np.ones_like(t), -t]),
    ])
    proj = rng.normal(size=(2, n_genes)) * 4.0
    logX = latent @ proj + rng.normal(scale=0.1, size=(n_cells, n_genes))
    mu = np.exp(logX - logX.max(axis=0)) * 40 + 1
    X = rng.negative_binomial(n=5, p=5.0 / (5.0 + mu)).astype(float)
    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    return new_cell_dataset(
        X, pheno_data=obs, feature_data=var,
        lower_detection_limit=1.0,
        expression_family=negbinomial_size(),
    )


@pytest.fixture(scope="module")
def ordered_cds() -> AnnData:
    """A fully-ordered CDS usable across plotting/BEAM tests."""
    cds = _branching_cds(n_per_branch=35, n_genes=60)
    estimate_size_factors(cds)
    detect_genes(cds, min_expr=0.5)
    set_ordering_filter(cds, cds.var_names.tolist())
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=10, verbose=False,
    )
    order_cells(cds)
    return cds


# ---------------------------------------------------------------------------
# ordering fallback — _pick_root_cell_from_cell_mst directly
# ---------------------------------------------------------------------------


def test_pick_root_cell_from_cell_mst_no_state() -> None:
    """Directly exercise the root_state-is-None branch
    (``ordering.py:361-362``): picks the MST diameter endpoint."""
    from monocle2py.ordering import _pick_root_cell_from_cell_mst

    # Path graph 0 — 1 — 2 — 3 with unit edge weights.
    adj_cell = [[(1, 1.0)], [(0, 1.0), (2, 1.0)], [(1, 1.0), (3, 1.0)], [(2, 1.0)]]
    adata = AnnData(np.zeros((4, 1)))
    Z = np.zeros((2, 4))  # unused for root_state=None
    closest_vertex = np.zeros(4, dtype=np.int64)
    root = _pick_root_cell_from_cell_mst(
        adata, adj_cell, Z, closest_vertex,
        root_state=None, reverse=False, prev_root_vertex=None,
    )
    assert root in (0, 3)


def test_pick_root_cell_from_cell_mst_with_state_no_cells() -> None:
    """Raise when the requested ``root_state`` has no cells."""
    from monocle2py.ordering import _pick_root_cell_from_cell_mst

    adj_cell = [[(1, 1.0)], [(0, 1.0)]]
    adata = AnnData(np.zeros((2, 1)))
    adata.obs["State"] = np.array([1, 1], dtype=np.int64)
    adata.obs["Pseudotime"] = np.array([0.0, 1.0])
    with pytest.raises(RuntimeError, match="No cells"):
        _pick_root_cell_from_cell_mst(
            adata, adj_cell, np.zeros((2, 2)),
            np.zeros(2, dtype=np.int64),
            root_state=99, reverse=False, prev_root_vertex=None,
        )


def test_pick_root_cell_from_cell_mst_with_state_single_cell() -> None:
    """The ``in_state.size == 1`` shortcut (``ordering.py:367-368``)."""
    from monocle2py.ordering import _pick_root_cell_from_cell_mst

    adj_cell = [[(1, 1.0)], [(0, 1.0)]]
    adata = AnnData(np.zeros((2, 1)))
    adata.obs["State"] = np.array([1, 2], dtype=np.int64)
    adata.obs["Pseudotime"] = np.array([0.0, 1.0])
    root = _pick_root_cell_from_cell_mst(
        adata, adj_cell, np.zeros((2, 2)),
        np.zeros(2, dtype=np.int64),
        root_state=2, reverse=False, prev_root_vertex=None,
    )
    assert root == 1  # only cell in state==2


def test_pick_root_cell_from_cell_mst_with_state_diameter_pick() -> None:
    """Multi-cell state → build sub-MST, pick by pseudotime max/min."""
    from monocle2py.ordering import _pick_root_cell_from_cell_mst

    adj_cell = [[(1, 1.0)], [(0, 1.0), (2, 1.0)], [(1, 1.0)], [(4, 1.0)], [(3, 1.0)]]
    adata = AnnData(np.zeros((5, 1)))
    adata.obs["State"] = np.array([1, 1, 1, 2, 2], dtype=np.int64)
    adata.obs["Pseudotime"] = np.array([0.5, 1.5, 3.0, 0.0, 2.0])
    # Z layout: cells 0..2 at increasing x; 3..4 elsewhere.
    Z = np.array([[0.0, 1.0, 2.0, 10.0, 11.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0]])
    closest_vertex = np.zeros(5, dtype=np.int64)
    root = _pick_root_cell_from_cell_mst(
        adata, adj_cell, Z, closest_vertex,
        root_state=1, reverse=False, prev_root_vertex=None,
    )
    # State==1 diameter is cells 0 and 2 (endpoints). Max pseudotime is cell 2.
    assert root == 2


def test_pick_root_cell_from_cell_mst_reverse_flips_to_min() -> None:
    from monocle2py.ordering import _pick_root_cell_from_cell_mst

    adj_cell = [[(1, 1.0)], [(0, 1.0), (2, 1.0)], [(1, 1.0)]]
    adata = AnnData(np.zeros((3, 1)))
    adata.obs["State"] = np.array([1, 1, 1], dtype=np.int64)
    adata.obs["Pseudotime"] = np.array([0.5, 1.5, 3.0])
    Z = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    root = _pick_root_cell_from_cell_mst(
        adata, adj_cell, Z, np.zeros(3, dtype=np.int64),
        root_state=1, reverse=True, prev_root_vertex=None,
    )
    # Reverse → min pseudotime on diameter → cell 0.
    assert root == 0


# ---------------------------------------------------------------------------
# BEAM extras: sequential_split, branch_labels, stretch=False, cal_ilrs
#              return_all / useVST / after_bifurcation
# ---------------------------------------------------------------------------


def test_build_branch_cell_dataset_sequential_split(ordered_cds) -> None:
    """Sequential-split path adds one duplicate_root row and alternates
    progenitors across the two branches (``beam.py:_sequential_split``)."""
    sub = build_branch_cell_dataset(
        ordered_cds, progenitor_method="sequential_split", branch_point=1,
    )
    assert "duplicate_root" in sub.obs_names
    assert sub.obs["Branch"].cat.categories.size == 2


def test_build_branch_cell_dataset_with_branch_labels(ordered_cds) -> None:
    sub = build_branch_cell_dataset(
        ordered_cds, branch_point=1, branch_labels=["Alpha", "Beta"],
    )
    assert list(sub.obs["Branch"].cat.categories) == ["Alpha", "Beta"]


def test_build_branch_cell_dataset_duplicate_with_labels(ordered_cds) -> None:
    sub = build_branch_cell_dataset(
        ordered_cds, progenitor_method="duplicate", branch_point=1,
        branch_labels=["A", "B"],
    )
    assert list(sub.obs["Branch"].cat.categories) == ["A", "B"]
    # Duplicates explicitly named "duplicate_{i}_{j}" (beam.py:317).
    dup = [n for n in sub.obs_names if n.startswith("duplicate_")]
    assert len(dup) > 0


def test_build_branch_cell_dataset_rejects_wrong_branch_label_count(
    ordered_cds,
) -> None:
    with pytest.raises(RuntimeError, match="exactly two"):
        build_branch_cell_dataset(
            ordered_cds, branch_point=1,
            branch_labels=["only-one"],
        )


def test_build_branch_cell_dataset_sequential_split_rejects_wrong_label_count(
    ordered_cds,
) -> None:
    with pytest.raises(RuntimeError, match="exactly two"):
        build_branch_cell_dataset(
            ordered_cds, progenitor_method="sequential_split",
            branch_point=1, branch_labels=["only"],
        )


def test_build_branch_cell_dataset_branch_states_mode(ordered_cds) -> None:
    """Pick by explicit state IDs (``beam.py:191-206``)."""
    states = sorted(ordered_cds.obs["State"].astype(int).unique())
    if len(states) < 2:
        pytest.skip("need at least 2 states")
    sub = build_branch_cell_dataset(
        ordered_cds, branch_states=list(states[-2:]),
    )
    assert sub.obs["Branch"].cat.categories.size == 2


def test_cal_ilrs_return_all_output_after_bifurcation(ordered_cds) -> None:
    """Exercise the ``output_type='after_bifurcation'`` trim
    (``beam.py:635-643``)."""
    res = cal_ilrs(
        ordered_cds, branch_point=1,
        output_type="after_bifurcation", n_points=20,
    )
    assert isinstance(res, pd.DataFrame)
    assert res.shape[1] <= 20


def test_cal_ilrs_return_all_dict(ordered_cds) -> None:
    """``return_all=True`` yields the six-key dict."""
    res = cal_ilrs(
        ordered_cds, branch_point=1, return_all=True, n_points=12,
    )
    assert isinstance(res, dict)
    expected = {
        "str_logfc_df", "norm_str_logfc_df",
        "str_norm_div_df", "str_raw_div_df",
        "str_branchA_expression_curve_matrix",
        "str_branchB_expression_curve_matrix",
    }
    assert set(res.keys()) == expected


def test_cal_ilrs_rejects_trajectory_states_not_two(ordered_cds) -> None:
    with pytest.raises(ValueError, match="TWO branches"):
        cal_ilrs(ordered_cds, trajectory_states=[1])


def test_branch_test_without_branch_term_runs_directly(ordered_cds) -> None:
    """When ``Branch`` is absent from the formula, ``branch_test`` skips
    ``build_branch_cell_dataset`` and calls ``differential_gene_test``
    on ``adata`` directly (``beam.py:505-515``)."""
    res = branch_test(
        ordered_cds,
        full_model_formula_str="~sm.ns(Pseudotime, df=3)",
        reduced_model_formula_str="~1",
    )
    assert set(("status", "pval", "qval")).issubset(res.columns)


def test_beam_explicit_progenitor_method_duplicate(ordered_cds) -> None:
    """Forward ``progenitor_method='duplicate'`` through ``beam`` →
    ``branch_test`` → ``build_branch_cell_dataset``."""
    res = beam(
        ordered_cds, branch_point=1,
        progenitor_method="duplicate",
    )
    assert set(("status", "pval", "qval")).issubset(res.columns)


# ---------------------------------------------------------------------------
# Plotting modules: only validation logic is tested; pure "does the plot
# render" smoke tests were removed per user guidance (no computational
# logic → not worth the pytest footprint).
# ---------------------------------------------------------------------------


def test_plot_multiple_branches_heatmap_raises_on_missing_state(
    ordered_cds,
) -> None:
    """Validation: branch ID must be a State value in the trajectory."""
    genes = ordered_cds.var_names[:4].tolist()
    with pytest.raises(RuntimeError, match="not a State|State"):
        plot_multiple_branches_heatmap(
            ordered_cds[:, genes], branches=[9999],
        )


def test_plot_multiple_branches_pseudotime_raises_on_missing_state(
    ordered_cds,
) -> None:
    """Validation: branch ID must exist as a State value."""
    genes = ordered_cds.var_names[:4].tolist()
    with pytest.raises(RuntimeError, match="State"):
        plot_multiple_branches_pseudotime(
            ordered_cds[:, genes], branches=[9999],
        )
