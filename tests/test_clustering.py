"""Tests for Slice 4 density-peak clustering and the rho/delta decision plot."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from monocle2py import cluster_cells, plot_rho_delta


def _four_blobs(n_per_blob: int = 60, seed: int = 0) -> AnnData:
    """Synthetic 2-D dataset with four well-separated isotropic Gaussian blobs."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    blocks = [
        rng.normal(loc=c, scale=0.4, size=(n_per_blob, 2)) for c in centers
    ]
    X = np.vstack(blocks)
    n = X.shape[0]
    adata = AnnData(np.zeros((n, 1)))
    adata.obsm["X_dr"] = X
    return adata


def test_cluster_cells_recovers_four_blobs() -> None:
    # Explicit thresholds matter: matching R's densityClust, num_clusters can
    # under-select when delta values are large because the eps subtraction
    # collapses to no-op at floating-point precision.
    adata = _four_blobs()
    cluster_cells(adata, rho_threshold=0.0, delta_threshold=5.0, gaussian=True)

    assert "Cluster" in adata.obs.columns
    assert adata.obs["Cluster"].nunique() == 4
    assert int(adata.obs["peaks"].sum()) == 4
    sizes = adata.obs["Cluster"].value_counts().to_numpy()
    assert sizes.min() > 0.8 * 60
    assert sizes.max() < 1.2 * 60


def test_cluster_cells_writes_state_and_obs_columns() -> None:
    adata = _four_blobs(n_per_blob=40)
    cluster_cells(adata, rho_threshold=0.0, delta_threshold=5.0)

    for col in ("Cluster", "peaks", "halo", "delta", "rho",
                "nearest_higher_density_neighbor"):
        assert col in adata.obs.columns

    dp = adata.uns["monocle2"]["density_peak"]
    assert "dc" in dp and dp["dc"] > 0
    assert dp["threshold"]["rho"] == 0.0
    assert dp["threshold"]["delta"] == 5.0
    # Cluster '1' should correspond to the highest-gamma peak.
    rho = adata.obs["rho"].to_numpy()
    delta = adata.obs["delta"].to_numpy()
    peak_idx = np.flatnonzero(adata.obs["peaks"].to_numpy())
    gamma_peaks = rho[peak_idx] * delta[peak_idx]
    cluster_str = adata.obs["Cluster"].to_numpy()
    cluster_one_peak = peak_idx[cluster_str[peak_idx] == "1"][0]
    assert (rho[cluster_one_peak] * delta[cluster_one_peak]) == gamma_peaks.max()


def test_cluster_cells_unsupported_method() -> None:
    adata = _four_blobs(n_per_blob=10)
    with pytest.raises(NotImplementedError, match="densityPeak"):
        cluster_cells(adata, method="louvain")


def test_cluster_cells_requires_dim_reduction() -> None:
    adata = AnnData(np.zeros((10, 1)))
    with pytest.raises(RuntimeError, match="X_dr"):
        cluster_cells(adata)


def test_plot_rho_delta_returns_ggplot() -> None:
    adata = _four_blobs(n_per_blob=30)
    cluster_cells(adata, num_clusters=4)
    g = plot_rho_delta(adata)
    # ggplot2_py returns a GGPlot object; just check the data was attached.
    from ggplot2_py import GGPlot
    assert isinstance(g, GGPlot)


def test_plot_rho_delta_requires_clustering() -> None:
    adata = _four_blobs(n_per_blob=10)
    with pytest.raises(RuntimeError, match="cluster_cells"):
        plot_rho_delta(adata)


def test_plot_rho_delta_threshold_matches_cluster_cells() -> None:
    """``plot_rho_delta`` must use the same strict ``>`` peak rule as
    ``cluster_cells`` (and R's ``densityClust::findClusters`` at
    ``which(x$rho > rho & x$delta > delta)``). The peak mask produced
    from threshold overrides must agree with the one ``cluster_cells``
    stored."""
    adata = _four_blobs(n_per_blob=30)
    rho_thr = 0.0
    delta_thr = 5.0
    cluster_cells(adata, rho_threshold=rho_thr, delta_threshold=delta_thr)
    stored = adata.obs["peaks"].to_numpy(dtype=bool)
    # Recompute the plot-side mask directly from rho/delta via the plot
    # function's threshold override. The returned plot embeds the mask;
    # we instead grab it from the DataFrame inside.
    g = plot_rho_delta(adata, rho_threshold=rho_thr, delta_threshold=delta_thr)
    plot_mask = g.data["peaks"].to_numpy(dtype=bool)
    assert np.array_equal(stored, plot_mask)
