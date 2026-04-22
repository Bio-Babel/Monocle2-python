"""Smoke tests for the Slice 9 heatmap plot family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pheatmap import PHeatmap

from monocle2py import (
    estimate_size_factors,
    new_cell_dataset,
    order_cells,
    plot_genes_branched_heatmap,
    plot_multiple_branches_heatmap,
    plot_pseudotime_heatmap,
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


def _branching_cds(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_per = 30
    n_genes = 30
    t1 = np.linspace(0, 1, n_per)
    t2 = np.linspace(0, 1, n_per)
    t3 = np.linspace(0, 1, n_per)
    latent = np.concatenate([
        np.column_stack([t1, np.zeros_like(t1)]),
        np.column_stack([np.ones_like(t2), t2]),
        np.column_stack([np.ones_like(t3), -t3]),
    ])
    proj = rng.normal(size=(2, n_genes)) * 4
    logX = latent @ proj + rng.normal(scale=0.1, size=(3 * n_per, n_genes))
    mu = np.exp(logX - logX.max(axis=0)) * 40 + 1
    X = rng.negative_binomial(n=5, p=5.0 / (5.0 + mu)).astype(float)
    cells = [f"C{i}" for i in range(3 * n_per)]
    genes = [f"G{i:02d}" for i in range(n_genes)]
    cds = new_cell_dataset(
        X,
        pheno_data=pd.DataFrame(index=cells),
        feature_data=pd.DataFrame({"gene_short_name": genes}, index=genes),
        lower_detection_limit=1.0,
    )
    estimate_size_factors(cds)
    _install_disp_func(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=10, verbose=False,
    )
    order_cells(cds)
    return cds


def test_plot_pseudotime_heatmap_returns_pheatmap():
    cds = _branching_cds()
    sub = cds[:, [f"G{i:02d}" for i in range(8)]]
    ph = plot_pseudotime_heatmap(sub, num_clusters=3)
    assert isinstance(ph, PHeatmap)


def test_plot_pseudotime_heatmap_log_norm_no_disp_fit():
    cds = _branching_cds()
    cds.uns["monocle2"]["disp_fit"] = {}
    sub = cds[:, [f"G{i:02d}" for i in range(8)]]
    ph = plot_pseudotime_heatmap(sub, num_clusters=2, norm_method="log")
    assert isinstance(ph, PHeatmap)


def test_plot_pseudotime_heatmap_requires_pseudotime():
    cds = _branching_cds()
    del cds.obs["Pseudotime"]
    sub = cds[:, [f"G{i:02d}" for i in range(4)]]
    with pytest.raises(RuntimeError, match="Pseudotime missing"):
        plot_pseudotime_heatmap(sub)


def test_plot_genes_branched_heatmap_returns_pheatmap():
    cds = _branching_cds()
    sub = cds[:, [f"G{i:02d}" for i in range(8)]]
    ph = plot_genes_branched_heatmap(
        sub, branch_point=1, num_clusters=3,
    )
    assert isinstance(ph, PHeatmap)


def test_plot_genes_branched_heatmap_return_dict():
    cds = _branching_cds()
    sub = cds[:, [f"G{i:02d}" for i in range(8)]]
    out = plot_genes_branched_heatmap(
        sub, branch_point=1, num_clusters=3, return_heatmap=True,
    )
    assert isinstance(out, dict)
    assert "ph_res" in out and isinstance(out["ph_res"], PHeatmap)
    assert "BranchA_exprs" in out and "BranchB_exprs" in out
    assert "annotation_col" in out and "annotation_row" in out


def test_plot_multiple_branches_heatmap_returns_pheatmap():
    cds = _branching_cds()
    states = sorted(cds.obs["State"].astype(int).unique().tolist())
    branches = [s for s in states if s != states[0]][:2]
    sub = cds[:, [f"G{i:02d}" for i in range(8)]]
    ph = plot_multiple_branches_heatmap(
        sub, branches=branches, num_clusters=3, norm_method="log",
    )
    assert isinstance(ph, PHeatmap)


def test_plot_multiple_branches_heatmap_invalid_branch():
    cds = _branching_cds()
    sub = cds[:, [f"G{i:02d}" for i in range(4)]]
    with pytest.raises(RuntimeError, match="not a State value"):
        plot_multiple_branches_heatmap(sub, branches=[999])
