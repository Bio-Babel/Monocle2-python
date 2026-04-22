"""Smoke tests for the Slice 8 trajectory plot family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ggplot2_py import GGPlot

from monocle2py import (
    cluster_cells,
    estimate_size_factors,
    new_cell_dataset,
    order_cells,
    plot_cell_clusters,
    plot_cell_trajectory,
    plot_complex_cell_trajectory,
    plot_genes_branched_pseudotime,
    plot_genes_in_pseudotime,
    plot_multiple_branches_pseudotime,
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


def test_plot_cell_trajectory_returns_ggplot():
    cds = _branching_cds()
    g = plot_cell_trajectory(cds)
    assert isinstance(g, GGPlot)


def test_plot_cell_trajectory_color_by_pseudotime():
    cds = _branching_cds()
    g = plot_cell_trajectory(cds, color_by="Pseudotime", show_branch_points=True)
    assert isinstance(g, GGPlot)


def test_plot_cell_trajectory_with_markers_gradient():
    cds = _branching_cds()
    g = plot_cell_trajectory(
        cds, markers=["G00", "G01"], use_color_gradient=True, theta=15,
    )
    assert isinstance(g, GGPlot)


def test_plot_cell_trajectory_requires_reduce_dimension():
    cds = new_cell_dataset(
        np.zeros((4, 3)),
        pheno_data=pd.DataFrame(index=[f"C{i}" for i in range(4)]),
        feature_data=pd.DataFrame(index=[f"G{i}" for i in range(3)]),
        lower_detection_limit=1.0,
    )
    with pytest.raises(RuntimeError, match="Reduced dimensions missing"):
        plot_cell_trajectory(cds)


def test_plot_complex_cell_trajectory_returns_ggplot():
    cds = _branching_cds()
    g = plot_complex_cell_trajectory(cds)
    assert isinstance(g, GGPlot)


def test_plot_complex_cell_trajectory_with_root_states():
    cds = _branching_cds()
    states = sorted(cds.obs["State"].astype(int).unique().tolist())
    g = plot_complex_cell_trajectory(cds, root_states=[states[0]])
    assert isinstance(g, GGPlot)


def test_plot_cell_clusters_with_cluster_column():
    cds = _branching_cds()
    cds.obs["Cluster"] = pd.Categorical(
        [f"cl{i % 3}" for i in range(cds.n_obs)],
    )
    g = plot_cell_clusters(cds)
    assert isinstance(g, GGPlot)


def test_plot_cell_clusters_requires_cluster_column():
    cds = _branching_cds()
    if "Cluster" in cds.obs.columns:
        del cds.obs["Cluster"]
    with pytest.raises(RuntimeError, match="Cluster labels missing"):
        plot_cell_clusters(cds)


def test_plot_cell_clusters_with_markers():
    cds = _branching_cds()
    cds.obs["Cluster"] = pd.Categorical(
        [f"cl{i % 3}" for i in range(cds.n_obs)],
    )
    g = plot_cell_clusters(cds, markers=["G00", "G02"])
    assert isinstance(g, GGPlot)


def test_plot_genes_in_pseudotime_returns_ggplot():
    cds = _branching_cds()
    sub = cds[:, ["G00", "G01", "G02"]]
    g = plot_genes_in_pseudotime(sub)
    assert isinstance(g, GGPlot)


def test_plot_genes_in_pseudotime_requires_pseudotime():
    cds = _branching_cds()
    del cds.obs["Pseudotime"]
    sub = cds[:, ["G00"]]
    with pytest.raises(RuntimeError, match="Pseudotime missing"):
        plot_genes_in_pseudotime(sub)


def test_plot_genes_branched_pseudotime_returns_ggplot():
    cds = _branching_cds()
    sub = cds[:, ["G00", "G01", "G02"]]
    g = plot_genes_branched_pseudotime(sub, branch_point=1)
    assert isinstance(g, GGPlot)


def test_plot_genes_branched_pseudotime_with_reduced_model():
    cds = _branching_cds()
    sub = cds[:, ["G00", "G01"]]
    g = plot_genes_branched_pseudotime(
        sub, branch_point=1,
        reduced_model_formula_str="~sm.ns(Pseudotime, df=3)",
    )
    assert isinstance(g, GGPlot)


def test_plot_multiple_branches_pseudotime_returns_ggplot():
    cds = _branching_cds()
    states = sorted(cds.obs["State"].astype(int).unique().tolist())
    branches = [s for s in states if s != states[0]][:2]
    g = plot_multiple_branches_pseudotime(
        cds[:, ["G00", "G01", "G02"]], branches=branches,
    )
    assert isinstance(g, GGPlot)


def test_plot_multiple_branches_pseudotime_invalid_branch():
    cds = _branching_cds()
    with pytest.raises(RuntimeError, match="not a State value"):
        plot_multiple_branches_pseudotime(
            cds[:, ["G00"]], branches=[999],
        )
