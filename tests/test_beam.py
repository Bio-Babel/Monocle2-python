"""Tests for Slice 7 BEAM helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monocle2py import (
    beam,
    branch_test,
    build_branch_cell_dataset,
    cal_ilrs,
    estimate_size_factors,
    new_cell_dataset,
    order_cells,
    reduce_dimension,
)


def _install_disp_func(cds, asymp: float = 0.05, extra: float = 1.0) -> None:
    from monocle2py._uns import set_disp_fit_info

    def disp_func(q):
        return asymp + extra / np.asarray(q, dtype=float)

    set_disp_fit_info(cds, {
        "disp_func": disp_func,
        "coefficients": {"asymptDisp": asymp, "extraPois": extra},
        "disp_table": pd.DataFrame(),
    }, name="blind")


def _branching_cds(
    n_cells_per_branch: int = 30,
    n_genes: int = 40,
    seed: int = 1,
    n_signal: int = 8,
):
    """Y-shaped synthetic trajectory with branch-specific signal genes."""
    rng = np.random.default_rng(seed)
    n_cells = 3 * n_cells_per_branch
    t1 = np.linspace(0, 1, n_cells_per_branch)
    t2 = np.linspace(0, 1, n_cells_per_branch)
    t3 = np.linspace(0, 1, n_cells_per_branch)
    latent = np.concatenate([
        np.column_stack([t1, np.zeros_like(t1)]),
        np.column_stack([np.ones_like(t2), t2]),
        np.column_stack([np.ones_like(t3), -t3]),
    ])
    proj = rng.normal(size=(2, n_genes)) * 4
    logX = latent @ proj + rng.normal(scale=0.1, size=(n_cells, n_genes))
    mu = np.exp(logX - logX.max(axis=0)) * 40 + 1
    X = rng.negative_binomial(n=5, p=5.0 / (5.0 + mu)).astype(float)

    # Branch-specific injected signal on the first ``n_signal`` genes: high
    # on branch 2 cells, low on branch 3 cells.
    b2 = slice(n_cells_per_branch, 2 * n_cells_per_branch)
    b3 = slice(2 * n_cells_per_branch, 3 * n_cells_per_branch)
    X[b2, :n_signal] += 30
    X[b3, :n_signal] = np.maximum(X[b3, :n_signal] - 5, 0)

    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i:02d}" for i in range(n_genes)]
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    cds = new_cell_dataset(
        X, pheno_data=obs, feature_data=var, lower_detection_limit=1.0,
    )
    estimate_size_factors(cds)
    _install_disp_func(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=15, verbose=False,
    )
    order_cells(cds)
    return cds


def test_build_branch_cell_dataset_duplicates_progenitors() -> None:
    cds = _branching_cds()
    sub = build_branch_cell_dataset(cds, branch_point=1)
    assert "Branch" in sub.obs.columns
    assert sub.obs["Branch"].cat.categories.size == 2
    # Progenitors appear in both branches; total >= original cell count.
    assert sub.n_obs >= cds.n_obs
    # Duplicated progenitor rows use the ``duplicate_i_j`` naming.
    assert any(n.startswith("duplicate_") for n in sub.obs_names)


def test_build_branch_cell_dataset_stretch_is_bounded() -> None:
    cds = _branching_cds()
    sub = build_branch_cell_dataset(cds, branch_point=1, stretch=True)
    pt = sub.obs["Pseudotime"].to_numpy(dtype=float)
    assert pt.min() >= 0.0
    assert pt.max() <= 100.0 + 1e-9


def test_build_branch_cell_dataset_branch_labels_applied() -> None:
    cds = _branching_cds()
    sub = build_branch_cell_dataset(
        cds, branch_point=1, branch_labels=("AT1", "AT2"),
    )
    assert set(sub.obs["Branch"].cat.categories) == {"AT1", "AT2"}


def test_build_branch_cell_dataset_requires_order_cells() -> None:
    cds = _branching_cds()
    del cds.obs["Pseudotime"]
    with pytest.raises(RuntimeError, match="order_cells"):
        build_branch_cell_dataset(cds, branch_point=1)


def test_branch_test_returns_expected_columns() -> None:
    cds = _branching_cds()
    res = branch_test(cds, branch_point=1)
    for col in ("status", "family", "pval", "qval"):
        assert col in res.columns
    assert res["pval"].between(0, 1).all()


def test_beam_detects_branch_specific_signal() -> None:
    cds = _branching_cds()
    res = beam(cds, branch_point=1)
    signal = [f"G{i:02d}" for i in range(8)]
    bg = [g for g in res.index if g not in signal]
    # Signal genes should have smaller median q-value than background.
    assert res.loc[signal, "qval"].median() < res.loc[bg, "qval"].median()


def test_beam_returns_dataframe_with_gene_columns() -> None:
    cds = _branching_cds()
    res = beam(cds, branch_point=1)
    assert "gene_short_name" in res.columns
    assert list(res.index) == list(cds.var_names)


def test_cal_ilrs_returns_clipped_logfc() -> None:
    cds = _branching_cds()
    ilrs = cal_ilrs(cds, branch_point=1, n_points=20, ILRs_limit=3.0)
    assert ilrs.shape == (cds.n_vars, 20)
    arr = ilrs.to_numpy()
    assert np.nanmax(arr) <= 3.0 + 1e-9
    assert np.nanmin(arr) >= -3.0 - 1e-9


def test_cal_ilrs_return_all_yields_dict() -> None:
    cds = _branching_cds()
    out = cal_ilrs(cds, branch_point=1, n_points=15, return_all=True)
    expected = {
        "str_logfc_df", "norm_str_logfc_df", "str_norm_div_df",
        "str_raw_div_df", "str_branchA_expression_curve_matrix",
        "str_branchB_expression_curve_matrix",
    }
    assert set(out.keys()) == expected
