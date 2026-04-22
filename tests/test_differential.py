"""Tests for Slice 6 differential testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from monocle2py import (
    differential_gene_test,
    estimate_dispersions,
    estimate_size_factors,
    fit_models,
    gen_smooth_curves,
    new_cell_dataset,
    response_matrix,
)
from monocle2py._internal._formula import (
    build_design_matrix,
    formula_terms,
    normalize_formula,
)


def _install_disp_func(cds, asymp: float = 0.05, extra: float = 1.0) -> None:
    """Hand-craft a dispersion fit info so tests don't depend on the
    Gamma-identity GLM, which is ill-conditioned on small synthetic data."""
    from monocle2py._uns import set_disp_fit_info

    def disp_func(q):
        return asymp + extra / np.asarray(q, dtype=float)

    info = {
        "disp_func": disp_func,
        "coefficients": {"asymptDisp": asymp, "extraPois": extra},
        "disp_table": pd.DataFrame(),
    }
    set_disp_fit_info(cds, info, name="blind")


def _synthetic_pseudotime_cds(
    n_cells: int = 60,
    n_genes: int = 25,
    n_signal: int = 5,
    seed: int = 0,
    install_disp: bool = True,
):
    """Mock CDS: a smooth ``Pseudotime`` signal on the first ``n_signal`` genes."""
    rng = np.random.default_rng(seed)
    pseudotime = np.linspace(0, 10, n_cells)

    n_signal = min(n_signal, n_genes)
    n_bg = n_genes - n_signal
    trend = np.sin(pseudotime / 2.0) * 30 + 50
    mu_signal = np.tile(trend[:, None], (1, n_signal))
    if n_bg > 0:
        mu_bg = np.full((n_cells, n_bg), 30.0)
        mu = np.concatenate([mu_signal, mu_bg], axis=1)
    else:
        mu = mu_signal

    k = 10.0
    X = rng.negative_binomial(n=k, p=k / (k + mu)).astype(float)

    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i:02d}" for i in range(n_genes)]
    obs = pd.DataFrame({"Pseudotime": pseudotime, "State": 1}, index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    cds = new_cell_dataset(
        X, pheno_data=obs, feature_data=var, lower_detection_limit=1.0,
    )
    estimate_size_factors(cds)
    if install_disp:
        _install_disp_func(cds)
    return cds


def test_normalize_formula_rewrites_sm_ns() -> None:
    assert normalize_formula("~sm.ns(Pseudotime, df=3)") == "~bs(Pseudotime, df=3)"
    assert normalize_formula("~1") == "~1"
    assert normalize_formula("sm.ns(x, df=4) + State") == "~bs(x, df=4) + State"


def test_formula_terms_extracts_referenced_columns() -> None:
    assert formula_terms("~sm.ns(Pseudotime, df=3)") == ["Pseudotime"]
    assert formula_terms("~State") == ["State"]
    assert formula_terms("~1") == []
    assert set(formula_terms("~sm.ns(Pseudotime, df=3) + State")) == {
        "Pseudotime", "State"
    }


def test_build_design_matrix_intercept_and_rank() -> None:
    data = pd.DataFrame({"Pseudotime": np.linspace(0, 1, 10)})
    X_full = build_design_matrix("~sm.ns(Pseudotime, df=3)", data)
    X_red = build_design_matrix("~1", data)
    assert X_full.shape == (10, 4)
    assert X_red.shape == (10, 1)
    assert np.allclose(X_full[:, 0], 1.0)
    assert np.allclose(X_red[:, 0], 1.0)


def test_differential_gene_test_returns_expected_columns() -> None:
    cds = _synthetic_pseudotime_cds(n_cells=40, n_genes=8, n_signal=2)
    res = differential_gene_test(cds, "~sm.ns(Pseudotime, df=3)", "~1")
    assert list(res.index) == list(cds.var_names)
    for col in ("status", "family", "pval", "qval"):
        assert col in res.columns
    assert "gene_short_name" in res.columns
    assert res["pval"].between(0, 1).all()
    assert res["qval"].between(0, 1).all()


def test_differential_gene_test_detects_signal() -> None:
    cds = _synthetic_pseudotime_cds()
    res = differential_gene_test(cds, "~sm.ns(Pseudotime, df=3)", "~1")
    signal_genes = [f"G{i:02d}" for i in range(5)]
    background_genes = [g for g in res.index if g not in signal_genes]
    signal_q = res.loc[signal_genes, "qval"].median()
    bg_q = res.loc[background_genes, "qval"].median()
    assert signal_q < bg_q


def test_differential_gene_test_rejects_missing_column() -> None:
    cds = _synthetic_pseudotime_cds(n_cells=20, n_genes=4)
    with pytest.raises(ValueError, match="not found in adata.obs"):
        differential_gene_test(cds, "~sm.ns(Foo, df=3)", "~1")


def test_differential_gene_test_rejects_nan_in_covariate() -> None:
    cds = _synthetic_pseudotime_cds(n_cells=20, n_genes=4)
    cds.obs.loc[cds.obs_names[0], "Pseudotime"] = np.nan
    with pytest.raises(ValueError, match="Inf/NaN"):
        differential_gene_test(cds, "~sm.ns(Pseudotime, df=3)", "~1")


def test_response_matrix_shape_matches_newdata() -> None:
    cds = _synthetic_pseudotime_cds(n_cells=40, n_genes=6)
    models = fit_models(cds, "~sm.ns(Pseudotime, df=3)")
    new_data = pd.DataFrame({"Pseudotime": np.linspace(0, 10, 20)})
    mat = response_matrix(
        models, newdata=new_data, formula_str="~sm.ns(Pseudotime, df=3)",
    )
    assert mat.shape == (cds.n_vars, 20)
    assert list(mat.index) == list(cds.var_names)


def test_gen_smooth_curves_returns_positive_means() -> None:
    cds = _synthetic_pseudotime_cds(n_cells=40, n_genes=5)
    new_data = pd.DataFrame({"Pseudotime": np.linspace(0, 10, 12)})
    curves = gen_smooth_curves(cds, new_data, "~sm.ns(Pseudotime, df=3)")
    assert curves.shape == (cds.n_vars, 12)
    fitted = curves.to_numpy()
    assert np.all(fitted[~np.isnan(fitted)] >= 0)
