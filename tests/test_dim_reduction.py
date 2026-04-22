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


def test_ddrtree_backend_numpy_is_default() -> None:
    """Explicit ``backend='numpy'`` must match the default path bit-for-bit."""
    cds_default = _branching_cds()
    cds_numpy = _branching_cds()
    estimate_size_factors(cds_default)
    estimate_size_factors(cds_numpy)

    reduce_dimension(cds_default, max_components=2, reduction_method="DDRTree",
                     auto_param_selection=False, max_iter=5)
    reduce_dimension(cds_numpy, max_components=2, reduction_method="DDRTree",
                     auto_param_selection=False, max_iter=5, backend="numpy")

    np.testing.assert_array_equal(cds_default.obsm["X_dr"], cds_numpy.obsm["X_dr"])


def test_ddrtree_forwards_backend_kwargs(monkeypatch) -> None:
    """``backend`` is passed explicitly; ``mst_algorithm`` / ``device`` /
    ``dtype`` flow through ``**kwargs`` into ``ddrtree.DDRTree``."""
    from monocle2py import dim_reduction as _dr

    captured: dict[str, object] = {}

    def _fake_DDRTree(X, **kw):
        captured["kwargs"] = kw
        orig = _real_DDRTree(X, **{k: v for k, v in kw.items()
                                   if k not in {"device", "dtype", "mst_algorithm"}})
        return orig

    _real_DDRTree = _dr._ddrtree.DDRTree
    monkeypatch.setattr(_dr._ddrtree, "DDRTree", _fake_DDRTree)

    cds = _branching_cds()
    estimate_size_factors(cds)
    reduce_dimension(
        cds, max_components=2, reduction_method="DDRTree",
        auto_param_selection=False, max_iter=5,
        backend="numpy", mst_algorithm="prim", device="cpu", dtype="float64",
    )
    kw = captured["kwargs"]
    assert kw["backend"] == "numpy"
    assert kw["mst_algorithm"] == "prim"
    assert kw["device"] == "cpu"
    assert kw["dtype"] == "float64"


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


def test_remove_batch_effects_matches_limma_lmFit_gold() -> None:
    """Port of R ``reduceDimension(residualModelFormulaStr=...)`` batch
    removal (``order_cells.R:1363-1375``). ``_remove_batch_effects``
    must match ``limma::lmFit(FM, X.model_mat)`` per-gene OLS followed
    by ``FM - beta[, -1] %*% t(X[, -1])`` to machine precision."""
    from pathlib import Path
    from anndata import AnnData
    from monocle2py.dim_reduction import _remove_batch_effects

    fix_dir = Path(__file__).parent / "_fixtures"
    if not (fix_dir / "residual_fm.csv").exists():
        pytest.skip("R gold fixture not generated yet")
    FM = pd.read_csv(fix_dir / "residual_fm.csv", index_col=0).to_numpy()
    pheno = pd.read_csv(fix_dir / "residual_pdata.csv", index_col=0)
    FM_adj_R = pd.read_csv(fix_dir / "residual_fm_adj.csv", index_col=0).to_numpy()
    adata = AnnData(X=np.zeros((FM.shape[1], FM.shape[0])), obs=pheno)
    FM_adj_py = _remove_batch_effects(FM, adata, "~batch")
    # Machine-precision agreement per-entry (R reports 1.15e-14 on this fixture).
    np.testing.assert_allclose(FM_adj_py, FM_adj_R, rtol=0, atol=1e-10)


def test_tsne_pca_preproc_matches_prcomp_irlba_gold() -> None:
    """Port of R ``prcomp_irlba(t(FM), center=TRUE, scale.=TRUE)``
    (``order_cells.R:1430-1432``). Singular values must match exactly;
    PC scores agree up to a per-column sign flip because SVD's left
    singular vectors are defined only up to sign."""
    from pathlib import Path
    from sklearn.decomposition import PCA

    fix_dir = Path(__file__).parent / "_fixtures"
    if not (fix_dir / "prcomp_fm.csv").exists():
        pytest.skip("R gold fixture not generated yet")
    FM = pd.read_csv(fix_dir / "prcomp_fm.csv", index_col=0).to_numpy()
    scores_R = pd.read_csv(fix_dir / "prcomp_scores.csv").to_numpy()
    sdev_R = pd.read_csv(fix_dir / "prcomp_sdev.csv")["sdev"].to_numpy()

    FM_t = FM.T.astype(np.float64)
    mu = FM_t.mean(axis=0)
    sd = FM_t.std(axis=0, ddof=1)
    keep = sd > 0
    FM_scaled = (FM_t[:, keep] - mu[keep]) / sd[keep]
    pca = PCA(n_components=10, random_state=2016)
    scores_py = pca.fit_transform(FM_scaled)
    sdev_py = pca.singular_values_ / np.sqrt(FM_scaled.shape[0] - 1)

    np.testing.assert_allclose(sdev_py, sdev_R, rtol=0, atol=1e-8)
    for col in range(scores_R.shape[1]):
        corr = np.corrcoef(scores_py[:, col], scores_R[:, col])[0, 1]
        assert abs(corr) > 1 - 1e-8, f"PC{col} correlation |{corr}| < 1"


def test_reduce_dimension_accepts_residual_model_formula_str() -> None:
    """End-to-end smoke test for ``residual_model_formula_str``: with a
    batch covariate, the tSNE embedding must differ from the no-batch
    run, confirming the batch-removal step fires."""
    from monocle2py import estimate_size_factors

    rng = np.random.default_rng(0)
    n_cells, n_genes = 40, 60
    X = rng.negative_binomial(n=4, p=0.5, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame({
        "batch": np.repeat(["A", "B"], n_cells // 2),
    }, index=[f"C{i}" for i in range(n_cells)])
    var = pd.DataFrame({"gene_short_name": [f"G{i}" for i in range(n_genes)]},
                       index=[f"G{i}" for i in range(n_genes)])
    cds_no = new_cell_dataset(X, pheno_data=obs.copy(), feature_data=var.copy())
    cds_yes = new_cell_dataset(X, pheno_data=obs.copy(), feature_data=var.copy())
    estimate_size_factors(cds_no)
    estimate_size_factors(cds_yes)
    reduce_dimension(
        cds_no, reduction_method="tSNE", num_dim=5,
        perplexity=5, auto_param_selection=False,
    )
    reduce_dimension(
        cds_yes, reduction_method="tSNE", num_dim=5,
        perplexity=5, auto_param_selection=False,
        residual_model_formula_str="~batch",
    )
    assert cds_yes.obsm["X_dr"].shape == (n_cells, 2)
    assert not np.allclose(cds_no.obsm["X_dr"], cds_yes.obsm["X_dr"])
