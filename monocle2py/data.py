"""Public data loaders for monocle2 tutorial fixtures.

All loaders resolve data via the three-tier strategy defined in
``_download.py`` (cwd-local staging → user cache → registry download).
"""

from __future__ import annotations

import json
from typing import Any

import anndata as ad
import pandas as pd

from ._download import resolve_data_path

__all__ = [
    "load_hsmm_fpkm",
    "load_lung_cds",
    "load_paul_cds",
    "load_olsson_tpm",
    "load_fig1b_ordering_genes",
    "load_paul_gene_set",
]


def load_hsmm_fpkm() -> ad.AnnData:
    """Load the HSMM (Trapnell 2014) FPKM expression matrix.

    Returns
    -------
    anndata.AnnData
        ``n_obs`` myoblast cells x ``n_vars`` genes. ``X`` holds FPKM values
        from the tutorial RDS bundle. ``adata.uns["monocle2"]`` reports
        ``expression_family="tobit"`` and ``lower_detection_limit=0.1``.
    """
    path = resolve_data_path("HSMM_fpkm.h5ad")
    return ad.read_h5ad(path)


def load_lung_cds() -> ad.AnnData:
    """Load the pre-fit lung epithelium CellDataSet from the V2_2 tutorial.

    Returns
    -------
    anndata.AnnData
        Pre-trained DDRTree state is stored under
        ``adata.uns["monocle2"]["ddrtree"]`` with keys ``K``, ``W``,
        ``mst_edges``, ``mst_weights``, ``proj_edges``, ``proj_weights``,
        ``closest_vertex``. ``obsm["X_dr"]`` holds the 2-D embedding.
    """
    path = resolve_data_path("lung_cds.h5ad")
    return ad.read_h5ad(path)


def load_paul_cds() -> ad.AnnData:
    """Load the pre-fit Paul 2015 hematopoiesis CellDataSet from V2_4.

    Returns
    -------
    anndata.AnnData
        Cells x genes with the same pre-trained DDRTree layout as
        :func:`load_lung_cds`.
    """
    path = resolve_data_path("paul_cds.h5ad")
    return ad.read_h5ad(path)


def load_olsson_tpm() -> ad.AnnData:
    """Load the Olsson 2016 TPM-normalised expression matrix used in V2_3.

    Returns
    -------
    anndata.AnnData
        Cells x genes. ``obs`` carries the ``Type`` column constructed from
        the column-name split (WT / single-KO / double-KO) that the R tutorial
        uses as the grouping variable.
    """
    path = resolve_data_path("olsson_tpm.h5ad")
    return ad.read_h5ad(path)


def load_fig1b_ordering_genes() -> pd.DataFrame:
    """Load the fig1b ordering-gene table used to seed the Olsson trajectory.

    Returns
    -------
    pandas.DataFrame
        Columns as in ``fig1b.txt`` from the original tutorial (gene symbol
        and its fig1b cluster assignment).
    """
    path = resolve_data_path("fig1b.tsv")
    return pd.read_csv(path, sep="\t")


def load_paul_gene_set() -> dict[str, list[str]]:
    """Load the Paul branch-analysis gene panels from the Zenodo record.

    Returns
    -------
    dict[str, list[str]]
        Keys ``positive_score_genes``, ``negtive_score_genes``,
        ``all_down_valid`` exactly as in the original R ``gene_set`` save.
        The misspelling ``negtive_score_genes`` is preserved to match the
        tutorial code.
    """
    path = resolve_data_path("paul_gene_set.json")
    with open(path, "r") as fh:
        data: dict[str, Any] = json.load(fh)
    return {k: list(v) for k, v in data.items()}
