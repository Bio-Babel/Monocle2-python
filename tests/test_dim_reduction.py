"""Tests for Slice 3 dimensionality reduction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monocle2py import (
    cal_ncenter,
    estimate_size_factors,
    new_cell_dataset,
    reduce_dimension,
    tobit,
)


def test_cal_ncenter_matches_r_formula() -> None:
    # Matches ``round(2 * n_cells_limit * log(ncells) /
    #                   (log(ncells) + log(n_cells_limit)))``.
    assert cal_ncenter(500) == 115
    assert cal_ncenter(2000) == 125


def _branching_cds(n_cells_per_branch: int = 30, n_genes: int = 80, seed: int = 1):
    """Synthetic 2-branch trajectory: two linear gradients diverging in latent space."""
    rng = np.random.default_rng(seed)
    n_cells = 3 * n_cells_per_branch
    t1 = np.linspace(0, 1, n_cells_per_branch)
    t2 = np.linspace(0, 1, n_cells_per_branch)
    t3 = np.linspace(0, 1, n_cells_per_branch)
    # Three branches in a Y shape
    latent = np.concatenate([
        np.column_stack([t1, np.zeros_like(t1)]),                      # trunk
        np.column_stack([np.ones_like(t2), t2]),                       # upper branch
        np.column_stack([np.ones_like(t3), -t3]),                      # lower branch
    ])

    # Build gene expression as random projection of latent + noise
    proj = rng.normal(size=(2, n_genes)) * 4
    logX = latent @ proj + rng.normal(scale=0.1, size=(n_cells, n_genes))
    # Exponentiate to get counts-like values then sample NB around them
    mu = np.exp(logX - logX.max(axis=0)) * 40 + 1
    X = rng.negative_binomial(n=5, p=5.0 / (5.0 + mu))
    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    return new_cell_dataset(X.astype(float), pheno_data=obs, feature_data=var,
                            lower_detection_limit=1.0)


def test_ddrtree_runs_and_stores_state() -> None:
    cds = _branching_cds()
    estimate_size_factors(cds)

    reduce_dimension(cds, max_components=2, reduction_method="DDRTree",
                     auto_param_selection=False, max_iter=5, verbose=False)

    ddr = cds.uns["monocle2"]["ddrtree"]
    assert "X_dr" in cds.obsm
    assert cds.obsm["X_dr"].shape == (cds.n_obs, 2)
    assert ddr["K"].shape[0] == 2  # dim x K
    # MST on K principal points has K-1 edges.
    K = ddr["K"].shape[1]
    assert ddr["mst_edges"].shape == (K - 1, 2)
    assert cds.uns["monocle2"]["dim_reduce_type"] == "DDRTree"


def test_tsne_runs_and_stores_state() -> None:
    cds = _branching_cds()
    estimate_size_factors(cds)

    reduce_dimension(cds, max_components=2, reduction_method="tSNE",
                     num_dim=10, perplexity=8)

    assert cds.obsm["X_dr"].shape == (cds.n_obs, 2)
    assert cds.uns["monocle2"]["dim_reduce_type"] == "tSNE"


def test_unsupported_method_errors() -> None:
    cds = _branching_cds(n_cells_per_branch=5, n_genes=20)
    estimate_size_factors(cds)
    with pytest.raises(ValueError, match="Only 'DDRTree' and 'tSNE'"):
        reduce_dimension(cds, reduction_method="ICA")


def test_tobit_log_normalisation() -> None:
    rng = np.random.default_rng(0)
    n_cells, n_genes = 20, 30
    fpkm = rng.lognormal(size=(n_cells, n_genes)) * 5
    obs = pd.DataFrame(index=[f"C{i}" for i in range(n_cells)])
    var = pd.DataFrame({"gene_short_name": [f"G{i}" for i in range(n_genes)]},
                       index=[f"G{i}" for i in range(n_genes)])
    cds = new_cell_dataset(fpkm, pheno_data=obs, feature_data=var,
                           expression_family=tobit(lower=0.1))
    reduce_dimension(cds, reduction_method="tSNE", num_dim=5,
                     perplexity=5, auto_param_selection=False)
    assert cds.obsm["X_dr"].shape == (n_cells, 2)
