"""Tests for Slice 1 preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monocle2py import (
    detect_genes,
    disp_table,
    estimate_dispersions,
    estimate_size_factors,
    new_cell_dataset,
    vst_exprs,
)
from monocle2py._uns import SIZE_FACTOR_COL


def _random_cds(n_cells: int = 50, n_genes: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Vary per-gene mean across a wide dynamic range so the ``a + b/mu``
    # dispersion curve has signal in both coefficients.
    mu_per_gene = rng.uniform(0.5, 30.0, size=n_genes)
    X = np.column_stack([
        rng.negative_binomial(n=3, p=3.0 / (3.0 + m), size=n_cells)
        for m in mu_per_gene
    ]).astype(float)
    cells = [f"C{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame({"gene_short_name": genes}, index=genes)
    return new_cell_dataset(X, pheno_data=obs, feature_data=var,
                            lower_detection_limit=1.0)


def test_mean_geomean_total_size_factors() -> None:
    cds = _random_cds()
    estimate_size_factors(cds)
    sfs = cds.obs[SIZE_FACTOR_COL].to_numpy()
    assert np.all(np.isfinite(sfs))
    assert np.all(sfs > 0)
    # DESeq1 normalisation: geometric mean of size factors ~ 1.
    assert np.mean(np.log(sfs)) == pytest.approx(0.0, abs=1e-9)


def test_geometric_mean_total_variant() -> None:
    cds = _random_cds()
    estimate_size_factors(cds, method="geometric-mean-total")
    sfs = cds.obs[SIZE_FACTOR_COL].to_numpy()
    assert np.all(sfs > 0)


def test_detect_genes_counts() -> None:
    cds = _random_cds()
    detect_genes(cds)
    assert (cds.var["num_cells_expressed"] >= 0).all()
    assert (cds.obs["num_genes_expressed"] >= 0).all()
    # With lower_detection_limit=1.0, a cell with all counts >=2 has all genes.
    assert cds.obs["num_genes_expressed"].max() <= cds.n_vars


def test_full_pipeline_dispersions_and_vst() -> None:
    cds = _random_cds(n_cells=80, n_genes=60, seed=1)
    estimate_size_factors(cds)
    estimate_dispersions(cds, min_cells_detected=1)
    detect_genes(cds)

    tbl = disp_table(cds)
    assert set(tbl.columns) == {
        "gene_id", "mean_expression", "dispersion_fit", "dispersion_empirical"
    }
    assert (tbl["dispersion_fit"] > 0).all()

    vst = vst_exprs(cds)
    assert vst.shape == (cds.n_obs, cds.n_vars)
    assert np.all(np.isfinite(vst))


def test_estimate_dispersions_requires_size_factors() -> None:
    cds = _random_cds()
    with pytest.raises(ValueError, match="Size_Factor"):
        estimate_dispersions(cds)


def test_unknown_method_rejected() -> None:
    cds = _random_cds()
    with pytest.raises(ValueError):
        estimate_size_factors(cds, method="bogus")


def test_estimate_dispersions_group_by_matches_r_gold() -> None:
    """Port of R ``estimateDispersions(cds, modelFormulaStr="~CellType")``
    (``expr_models.R:515-522``). Runs ``disp_calc_helper_NB`` per unique
    covariate level, concatenates the per-group ``(gene_id, mu, disp)``
    rows into a pooled table, then fits a single Gamma/identity curve on
    the pooled rows.

    Fixture recorded from R monocle v2.14 on 2026-04-22:
    asymptDisp=0.4108393, extraPois=0.1068619, disp_table rows=150."""
    from pathlib import Path
    from monocle2py import negbinomial_size
    from monocle2py._uns import get_disp_fit_info

    fix_dir = Path(__file__).parent / "_fixtures"
    if not (fix_dir / "disp_groupby_gold.txt").exists():
        pytest.skip("R gold fixture not generated yet")
    counts = pd.read_csv(fix_dir / "disp_groupby_counts.csv", index_col=0)
    pheno = pd.read_csv(fix_dir / "disp_groupby_pheno.csv", index_col=0)
    X = counts.to_numpy().astype(float)
    var = pd.DataFrame({"gene_short_name": counts.columns.tolist()},
                       index=counts.columns)
    cds = new_cell_dataset(
        X, pheno_data=pheno, feature_data=var, lower_detection_limit=0.5,
        expression_family=negbinomial_size(),
    )
    estimate_size_factors(cds)
    estimate_dispersions(cds, model_formula_str="~CellType")
    info = get_disp_fit_info(cds, "blind")

    gold = dict(
        line.strip().split("=", 1)
        for line in (fix_dir / "disp_groupby_gold.txt").read_text().splitlines()
        if "=" in line
    )
    np.testing.assert_allclose(
        info["coefficients"]["asymptDisp"], float(gold["asymptDisp"]),
        rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        info["coefficients"]["extraPois"], float(gold["extraPois"]),
        rtol=0, atol=1e-6,
    )
    assert len(info["disp_table"]) == int(gold["rows"])
