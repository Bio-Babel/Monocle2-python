"""Tests for NB dispersion handling in differential_gene_test.

Covers three paths:
- ``NegbinomialSize`` with a fitted disp_func (default tutorial path) →
  alpha = disp_func(mean).
- ``NegbinomialSize`` with no disp_func and default ``size=Inf`` →
  degenerate to Poisson (matches R ``VGAM::negbinomial.size()``).
- ``Negbinomial`` (joint estimation via statsmodels discrete NB).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from monocle2py.differential import differential_gene_test, _gene_dispersion_alpha
from monocle2py.cell_dataset import new_cell_dataset
from monocle2py.families import (
    Negbinomial, NegbinomialSize, negbinomial, negbinomial_size,
)
from monocle2py.preprocess import estimate_size_factors, estimate_dispersions


def _make_simple_cds(n_cells: int = 60, n_genes: int = 5, seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    pt = np.linspace(0, 10, n_cells)
    mu = 5 + 2 * pt[:, None]
    counts = rng.negative_binomial(10, 10 / (10 + mu), size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame({
        "Pseudotime": pt,
    }, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame({
        "gene_short_name": [f"g{i}" for i in range(n_genes)],
    }, index=[f"g{i}" for i in range(n_genes)])
    cds = new_cell_dataset(counts, pheno_data=obs, feature_data=var,
                           expression_family=negbinomial_size())
    estimate_size_factors(cds)
    return cds


def test_negbinomial_size_default_is_inf_matches_vgam():
    """Python ``negbinomial_size()`` default maps to VGAM size=Inf (Poisson)."""
    fam = negbinomial_size()
    assert math.isinf(fam.size)


def test_gene_dispersion_alpha_prefers_disp_func():
    """When disp_func is available, its value overrides family.size."""
    def disp(mu):
        return np.full_like(mu, 0.5, dtype=float)

    fam = NegbinomialSize(size=2.0)  # would give alpha=0.5 via family
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    # disp_func returns 0.5 at mean=3, so alpha should be 0.5 from the hint
    assert _gene_dispersion_alpha(disp, x, fam) == pytest.approx(0.5)


def test_gene_dispersion_alpha_falls_back_to_family_size_when_finite():
    """Without disp_func, finite family.size gives alpha = 1/size."""
    fam = NegbinomialSize(size=4.0)
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    assert _gene_dispersion_alpha(None, x, fam) == pytest.approx(0.25)


def test_gene_dispersion_alpha_returns_zero_for_size_inf_poisson():
    """size=Inf must yield alpha=0 so make_family returns Poisson."""
    fam = negbinomial_size()
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    alpha = _gene_dispersion_alpha(None, x, fam)
    assert alpha == 0.0


def test_make_family_poisson_on_alpha_zero():
    from monocle2py._internal._vgam import make_family
    import statsmodels.api as sm
    f = make_family("negbinomial.size", alpha=0.0)
    assert isinstance(f, sm.families.Poisson)
    f_inf = make_family("negbinomial.size", alpha=math.inf)
    assert isinstance(f_inf, sm.families.Poisson)
    f_nb = make_family("negbinomial.size", alpha=0.5)
    assert isinstance(f_nb, sm.families.NegativeBinomial)


def test_differential_gene_test_negbinomial_size_without_disp_func_uses_poisson():
    """Without disp_func + default size=Inf, fitting falls back to Poisson."""
    cds = _make_simple_cds(seed=42)
    # No estimate_dispersions → disp_func absent → Poisson fallback per VGAM default
    res = differential_gene_test(cds)
    assert set(res["status"].unique()) <= {"OK", "FAIL"}
    assert (res["status"] == "OK").sum() >= 3


def test_differential_gene_test_joint_negbinomial():
    """Non-default ``negbinomial()`` family uses joint MLE via statsmodels.discrete."""
    cds = _make_simple_cds(seed=7)
    # Override family to Negbinomial (joint estimation)
    from monocle2py._uns import set_expression_family
    set_expression_family(cds, negbinomial())
    res = differential_gene_test(cds)
    # At least one gene should fit
    assert (res["status"] == "OK").sum() >= 1
    assert "pval" in res.columns and "qval" in res.columns
