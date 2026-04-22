"""Plots that visualise density-peak clustering results."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from ggplot2_py import (
    GGPlot,
    aes,
    element_blank,
    element_rect,
    element_text,
    geom_point,
    ggplot,
    scale_color_manual,
    scale_color_viridis_c,
    theme,
    unit,
    xlab,
    ylab,
)

from ._helpers import as_dense
from ._theme import monocle_theme_opts

__all__ = ["plot_rho_delta", "plot_cell_clusters"]


def plot_rho_delta(
    adata: AnnData,
    rho_threshold: float | None = None,
    delta_threshold: float | None = None,
) -> GGPlot:
    """Decision plot for density-peak clustering: rho on x, delta on y.

    Points marked as peaks are coloured black and the rest grey. If both
    thresholds are supplied, the peak mask is recomputed on the fly so users
    can explore alternative cutoffs without rerunning :func:`cluster_cells`.

    Parameters
    ----------
    adata : anndata.AnnData
        Must already have ``rho``, ``delta``, and ``peaks`` populated by
        :func:`monocle2py.cluster_cells`.
    rho_threshold, delta_threshold : float, optional
        Override the cached peak selection. Either both must be supplied
        or neither.

    Returns
    -------
    ggplot2_py.GGPlot
        Scatter plot of rho vs. delta.
    """
    required = {"rho", "delta", "peaks"}
    if not required.issubset(adata.obs.columns):
        raise RuntimeError(
            "Run cluster_cells before plot_rho_delta — adata.obs is missing "
            f"{sorted(required - set(adata.obs.columns))}."
        )
    rho = adata.obs["rho"].to_numpy(dtype=float)
    delta = adata.obs["delta"].to_numpy(dtype=float)
    if rho_threshold is not None and delta_threshold is not None:
        peaks = (rho > rho_threshold) & (delta > delta_threshold)
    else:
        peaks = adata.obs["peaks"].to_numpy(dtype=bool)
    df = pd.DataFrame({
        "rho": rho,
        "delta": delta,
        "peaks": np.asarray(peaks, dtype=bool),
    })
    return (
        ggplot(df, aes(x="rho", y="delta", color="peaks"))
        + geom_point(alpha=0.5)
        + scale_color_manual(values=["grey", "black"])
        + monocle_theme_opts()
        + theme(**{
            "legend.position": "top",
            "legend.key.height": unit(0.35, "in"),
            "legend.key": element_blank(),
            "panel.background": element_rect(fill="white"),
        })
    )


def _markers_long(
    adata: AnnData, markers: Sequence[str],
) -> pd.DataFrame:
    """Long-form (cell_id, feature_label, value) for the requested markers."""
    if "gene_short_name" in adata.var.columns:
        mask = adata.var["gene_short_name"].astype(str).isin(markers)
    else:
        mask = pd.Series(False, index=adata.var_names)
    mask = mask | adata.var_names.isin(markers)
    if not mask.any():
        return pd.DataFrame(columns=["cell_id", "feature_label", "value"])
    sub = adata[:, mask]
    X = as_dense(sub.X).astype(float)
    if "Size_Factor" in adata.obs.columns:
        sf = adata.obs["Size_Factor"].to_numpy(dtype=float)
        X = X / sf[:, None]
    rows = []
    cell_names = sub.obs_names.to_numpy()
    for j, gene in enumerate(sub.var_names):
        if "gene_short_name" in sub.var.columns:
            short = sub.var["gene_short_name"].iloc[j]
            label = str(short) if pd.notna(short) else str(gene)
        else:
            label = str(gene)
        rows.append(pd.DataFrame({
            "cell_id": cell_names,
            "feature_label": np.repeat(label, len(cell_names)),
            "value": np.round(X[:, j]),
        }))
    return pd.concat(rows, ignore_index=True)


def plot_cell_clusters(
    adata: AnnData,
    x: int = 1,
    y: int = 2,
    color_by: str = "Cluster",
    markers: Sequence[str] | None = None,
    show_cell_names: bool = False,
    cell_size: float = 1.5,
    cell_name_size: float = 2,
) -> GGPlot:
    """Port of R's ``plot_cell_clusters``.

    Plots cells in their reduced-dimension embedding (typically tSNE) coloured
    by ``Cluster`` (or any other ``adata.obs`` column). When ``markers`` is
    supplied, faceted gene-expression overlays replace the cluster colouring.
    """
    if "X_dr" not in adata.obsm:
        raise RuntimeError(
            "Reduced dimensions missing — call reduce_dimension first."
        )
    if color_by == "Cluster" and "Cluster" not in adata.obs.columns:
        raise RuntimeError(
            "Cluster labels missing — call cluster_cells first."
        )
    Z = np.asarray(adata.obsm["X_dr"], dtype=float)
    if Z.shape[1] < max(x, y):
        raise ValueError(
            f"Reduced space has {Z.shape[1]} components; need at least {max(x, y)}."
        )

    obs = adata.obs.reset_index(drop=True)
    cell_df = pd.DataFrame({
        "data_dim_1": Z[:, x - 1],
        "data_dim_2": Z[:, y - 1],
        "sample_name": adata.obs_names.astype(str),
    })
    for col in obs.columns:
        if col not in cell_df.columns:
            cell_df[col] = obs[col]

    markers_df = None
    if markers is not None:
        markers_df = _markers_long(adata, list(markers))
        if markers_df.empty:
            markers_df = None

    if markers_df is not None:
        cell_df = cell_df.merge(
            markers_df, left_on="sample_name", right_on="cell_id",
        )
        cell_df["log_value"] = np.log10(
            cell_df["value"].to_numpy(float) + 0.1
        )
        g = (
            ggplot(cell_df, aes(x="data_dim_1", y="data_dim_2"))
            + geom_point(aes(color="log_value"), size=cell_size)
            + scale_color_viridis_c(name="log10(value + 0.1)")
        )
    else:
        g = (
            ggplot(cell_df, aes(x="data_dim_1", y="data_dim_2"))
            + geom_point(aes(color=color_by), size=cell_size)
        )

    g = (
        g
        + monocle_theme_opts()
        + xlab(f"Component {x}")
        + ylab(f"Component {y}")
        + theme(**{
            "legend.position": "top",
            "legend.key.height": unit(0.35, "in"),
            "legend.key": element_blank(),
            "panel.background": element_rect(fill="white"),
            "text": element_text(size=15),
        })
    )
    return g
