"""Tests for ``new_cell_dataset`` (the R ``newCellDataSet`` port)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from monocle2py import negbinomial_size, new_cell_dataset, tobit
from monocle2py._uns import SIZE_FACTOR_COL


def _make_inputs(n_cells: int = 5, n_genes: int = 3):
    rng = np.random.default_rng(0)
    X = rng.poisson(lam=3, size=(n_cells, n_genes)).astype(float)
    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({"group": ["A", "B"] * (n_cells // 2) + ["A"] * (n_cells % 2)},
                       index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    return X, obs, var


def test_dense_inputs_populate_anndata() -> None:
    X, obs, var = _make_inputs()
    adata = new_cell_dataset(X, pheno_data=obs, feature_data=var,
                             lower_detection_limit=0.5)
    assert isinstance(adata, AnnData)
    assert adata.n_obs == X.shape[0]
    assert adata.n_vars == X.shape[1]
    assert adata.uns["monocle2"]["expression_family"] == "negbinomial.size"
    assert adata.uns["monocle2"]["lower_detection_limit"] == pytest.approx(0.5)
    assert SIZE_FACTOR_COL in adata.obs.columns
    assert adata.obs[SIZE_FACTOR_COL].isna().all()


def test_sparse_inputs_preserve_csr() -> None:
    X, obs, var = _make_inputs()
    adata = new_cell_dataset(csr_matrix(X), pheno_data=obs, feature_data=var)
    assert hasattr(adata.X, "tocsr")


def test_missing_gene_short_name_warns() -> None:
    X, obs, _ = _make_inputs()
    var = pd.DataFrame(index=[f"G{i}" for i in range(X.shape[1])])
    with pytest.warns(UserWarning, match="gene_short_name"):
        new_cell_dataset(X, pheno_data=obs, feature_data=var)


def test_custom_family_override() -> None:
    X, obs, var = _make_inputs()
    adata = new_cell_dataset(
        X, pheno_data=obs, feature_data=var,
        expression_family=tobit(lower=0.1),
    )
    assert adata.uns["monocle2"]["expression_family"] == "Tobit"


def test_shape_mismatch_errors() -> None:
    X, _, var = _make_inputs()
    bad_obs = pd.DataFrame(index=["x", "y"])
    with pytest.raises(ValueError):
        new_cell_dataset(X, pheno_data=bad_obs, feature_data=var)


def test_passthrough_anndata_populates_state() -> None:
    X, obs, var = _make_inputs()
    adata_in = AnnData(X=X, obs=obs, var=var)
    adata = new_cell_dataset(adata_in, expression_family=negbinomial_size(size=2.0))
    assert adata is adata_in
    # Family is persisted as (vfamily, params) so AnnData.write_h5ad does
    # not choke on a non-serialisable dataclass instance.
    state = adata.uns["monocle2"]
    assert state["expression_family"] == "negbinomial.size"
    assert float(state["expression_family_params"]["size"]) == pytest.approx(2.0)
