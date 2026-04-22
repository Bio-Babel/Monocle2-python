"""Smoke tests for the Slice 0 tutorial data loaders."""

from __future__ import annotations

import anndata as ad
import pandas as pd
import pytest

from monocle2py.data import (
    load_fig1b_ordering_genes,
    load_hsmm_fpkm,
    load_lung_cds,
    load_olsson_tpm,
    load_paul_cds,
    load_paul_gene_set,
)


def test_hsmm_fpkm_shape_and_meta() -> None:
    adata = load_hsmm_fpkm()
    assert isinstance(adata, ad.AnnData)
    assert adata.n_obs == 271
    assert adata.n_vars == 47192
    assert adata.uns["monocle2"]["expression_family"] == "tobit"
    assert adata.uns["monocle2"]["lower_detection_limit"] == pytest.approx(0.1)


def test_lung_cds_has_ddrtree_state() -> None:
    adata = load_lung_cds()
    assert adata.n_obs == 185
    assert adata.n_vars == 218
    assert adata.uns["monocle2"]["dim_reduce_type"] == "DDRTree"
    ddr = adata.uns["monocle2"]["ddrtree"]
    assert ddr["K"].shape[1] == ddr["W"].shape[1] or ddr["K"].shape[0] == ddr["W"].shape[1]
    assert "mst_edges" in ddr
    assert "closest_vertex" in ddr
    assert adata.obsm["X_dr"].shape == (adata.n_obs, 2)


def test_paul_cds_has_ddrtree_state() -> None:
    adata = load_paul_cds()
    assert adata.n_obs == 2699
    assert adata.n_vars == 3004
    assert adata.uns["monocle2"]["dim_reduce_type"] == "DDRTree"
    ddr = adata.uns["monocle2"]["ddrtree"]
    assert ddr["closest_vertex"].shape == (adata.n_obs,)
    n_vertices = ddr["K"].shape[1]
    # MST on n_vertices principal points has n_vertices - 1 edges.
    assert ddr["mst_edges"].shape == (n_vertices - 1, 2)


def test_olsson_tpm_type_column() -> None:
    adata = load_olsson_tpm()
    assert adata.n_obs == 640
    assert adata.n_vars == 23955
    assert "Type" in adata.obs.columns
    types = adata.obs["Type"].unique()
    # V2_3 pipeline: WT (1..452), single-KO (453..593), double-KO (594..640)
    assert any("knockout" in t for t in types)


def test_fig1b_ordering_genes() -> None:
    df = load_fig1b_ordering_genes()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0


def test_paul_gene_set_keys() -> None:
    gs = load_paul_gene_set()
    assert set(gs) == {"positive_score_genes", "negtive_score_genes",
                       "all_down_valid"}
    for v in gs.values():
        assert isinstance(v, list)
        assert all(isinstance(x, str) for x in v)
