"""Tests for the Census port (``relative2abs`` / ``estimate_t``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monocle2py import (
    estimate_t,
    load_hsmm_fpkm,
    new_cell_dataset,
    relative2abs,
    tobit,
)


def _tiny_tpm(n_cells: int = 10, n_genes: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    raw = rng.lognormal(mean=0.0, sigma=2.0, size=(n_cells, n_genes))
    raw = raw / raw.sum(axis=1, keepdims=True) * 1e6  # TPM
    return raw


def test_estimate_t_returns_positive_vector() -> None:
    tpm = _tiny_tpm()
    t_hat = estimate_t(tpm)
    assert t_hat.shape == (tpm.shape[0],)
    assert np.all(t_hat > 0)


def test_relative2abs_num_genes_produces_counts() -> None:
    tpm = _tiny_tpm()
    obs = pd.DataFrame(index=[f"C{i}" for i in range(tpm.shape[0])])
    var = pd.DataFrame(
        {"gene_short_name": [f"G{i}" for i in range(tpm.shape[1])]},
        index=[f"G{i}" for i in range(tpm.shape[1])],
    )
    cds = new_cell_dataset(tpm, pheno_data=obs, feature_data=var,
                           expression_family=tobit(lower=0.1),
                           lower_detection_limit=0.1)
    counts = relative2abs(cds)
    assert counts.shape == tpm.shape
    assert np.all(counts >= 0)
    # Column sums per cell should be positive and finite.
    per_cell_totals = counts.sum(axis=1)
    assert np.all(per_cell_totals > 0)
    assert np.all(np.isfinite(per_cell_totals))


def test_relative2abs_return_all() -> None:
    tpm = _tiny_tpm()
    obs = pd.DataFrame(index=[f"C{i}" for i in range(tpm.shape[0])])
    var = pd.DataFrame(index=[f"G{i}" for i in range(tpm.shape[1])])
    var["gene_short_name"] = var.index
    cds = new_cell_dataset(tpm, pheno_data=obs, feature_data=var,
                           expression_family=tobit(lower=0.1))
    result = relative2abs(cds, return_all=True)
    assert set(result) == {"norm_cds", "t_estimate", "expected_total_mRNAs"}
    assert result["norm_cds"].shape == tpm.shape
    assert result["t_estimate"].shape == (tpm.shape[0],)
    assert result["expected_total_mRNAs"].shape == (tpm.shape[0],)


def test_unsupported_method_raises() -> None:
    tpm = _tiny_tpm()
    obs = pd.DataFrame(index=[f"C{i}" for i in range(tpm.shape[0])])
    var = pd.DataFrame(index=[f"G{i}" for i in range(tpm.shape[1])])
    var["gene_short_name"] = var.index
    cds = new_cell_dataset(tpm, pheno_data=obs, feature_data=var,
                           expression_family=tobit(lower=0.1))
    with pytest.raises(NotImplementedError):
        relative2abs(cds, method="tpm_fraction")


@pytest.mark.slow
def test_hsmm_fpkm_integration() -> None:
    """Real HSMM FPKM → census transcript counts."""
    adata = load_hsmm_fpkm()
    counts = relative2abs(adata)
    assert counts.shape == (adata.n_obs, adata.n_vars)
    assert np.all(np.isfinite(counts[counts > 0]))
    # Total transcript counts per cell should be a plausible mRNA count.
    per_cell = counts.sum(axis=1)
    assert np.median(per_cell) > 0
