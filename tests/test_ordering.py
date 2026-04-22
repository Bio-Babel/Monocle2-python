"""Tests for Slice 5 ordering helpers (``set_ordering_filter`` + ``order_cells``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from monocle2py import (
    estimate_size_factors,
    new_cell_dataset,
    order_cells,
    reduce_dimension,
    set_ordering_filter,
)


def _branching_cds(n_cells_per_branch: int = 30, n_genes: int = 80, seed: int = 1):
    """Y-shaped synthetic trajectory: trunk + two diverging branches."""
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
    X = rng.negative_binomial(n=5, p=5.0 / (5.0 + mu))
    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    return new_cell_dataset(
        X.astype(float), pheno_data=obs, feature_data=var,
        lower_detection_limit=1.0,
    )


def test_set_ordering_filter_marks_requested_genes() -> None:
    cds = _branching_cds(n_cells_per_branch=5, n_genes=10)
    wanted = {"G1", "G3", "G7"}
    set_ordering_filter(cds, wanted)
    assert "use_for_ordering" in cds.var.columns
    marked = set(cds.var_names[cds.var["use_for_ordering"].to_numpy()])
    assert marked == wanted


def test_order_cells_writes_state_pseudotime_parent() -> None:
    cds = _branching_cds()
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=10, verbose=False,
    )
    order_cells(cds)

    for col in ("Pseudotime", "State", "Parent"):
        assert col in cds.obs.columns

    pt = cds.obs["Pseudotime"].to_numpy()
    assert np.all(np.isfinite(pt))
    assert pt.min() == 0.0
    assert pt.max() > 0.0

    states = cds.obs["State"].to_numpy()
    assert states.min() == 1
    # Y-shape has one branch point, expect at least 2 states.
    assert cds.obs["State"].nunique() >= 2

    # Exactly one cell is the root (Pseudotime == 0).
    assert int((pt == 0.0).sum()) == 1
    root_cell = cds.obs_names[np.argmin(pt)]
    root_parent = cds.obs.loc[root_cell, "Parent"]
    assert root_parent == ""


def test_order_cells_requires_dim_reduction() -> None:
    cds = AnnData(np.zeros((5, 3)))
    cds.uns["monocle2"] = {}
    with pytest.raises(RuntimeError, match="dimensionality not yet reduced"):
        order_cells(cds)


def test_order_cells_rejects_non_ddrtree() -> None:
    cds = _branching_cds(n_cells_per_branch=10, n_genes=20)
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="tSNE",
        num_dim=5, perplexity=5,
    )
    with pytest.raises(NotImplementedError, match="DDRTree"):
        order_cells(cds)


def test_order_cells_root_state_reuses_existing_state() -> None:
    cds = _branching_cds()
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=10, verbose=False,
    )
    order_cells(cds)
    first_states = cds.obs["State"].unique().tolist()
    # Pick a state that exists and re-order from it.
    chosen = int(first_states[-1])
    order_cells(cds, root_state=chosen)
    # Same column family, still valid.
    assert cds.obs["State"].nunique() >= 1
    assert np.all(np.isfinite(cds.obs["Pseudotime"].to_numpy()))


def test_order_cells_reverse_flips_root() -> None:
    cds = _branching_cds()
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=10, verbose=False,
    )
    order_cells(cds)
    pt_forward = cds.obs["Pseudotime"].to_numpy().copy()
    order_cells(cds, reverse=True)
    pt_reverse = cds.obs["Pseudotime"].to_numpy()
    # Reverse should produce a different root (different pseudotime distribution).
    assert not np.allclose(pt_forward, pt_reverse)
