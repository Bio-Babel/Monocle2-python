"""AnnData constructor matching R's ``newCellDataSet``.

Monocle2's CellDataSet stores counts (genes x cells), phenoData (cells),
featureData (genes), ``lowerDetectionLimit``, and ``expressionFamily``.
Per the porting essentials the Python container is ``anndata.AnnData``:
cells x genes for ``X``, ``obs`` for cell metadata, ``var`` for gene
metadata, and ``uns['monocle2']`` for family and detection limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, spmatrix

from ._uns import (
    SIZE_FACTOR_COL,
    ensure_state,
    set_expression_family,
    set_lower_detection_limit,
)
from .families import ExpressionFamily, negbinomial_size

__all__ = ["new_cell_dataset"]


def new_cell_dataset(
    cell_data: np.ndarray | spmatrix | pd.DataFrame | AnnData,
    pheno_data: pd.DataFrame | None = None,
    feature_data: pd.DataFrame | None = None,
    lower_detection_limit: float = 0.1,
    expression_family: ExpressionFamily | None = None,
) -> AnnData:
    """Construct an AnnData that mirrors ``newCellDataSet``'s output.

    Parameters
    ----------
    cell_data : array-like or AnnData
        Expression matrix. A NumPy/SciPy/pandas object is interpreted as
        **cells x genes** (Python convention). Passing an ``AnnData`` reuses
        it directly and only annotates the Monocle2 state slots.
    pheno_data : pandas.DataFrame, optional
        Cell metadata indexed by cell name. When omitted, an empty frame
        aligned with ``cell_data`` rows is created.
    feature_data : pandas.DataFrame, optional
        Gene metadata indexed by gene name. Must contain
        ``gene_short_name`` for downstream plotting helpers — a warning is
        emitted if missing, matching the R behaviour.
    lower_detection_limit : float, default 0.1
        Lower expression threshold used by ``detect_genes`` and downstream
        dispersion/ordering helpers.
    expression_family : ExpressionFamily, optional
        VGAM-compatible family marker. Defaults to
        :func:`monocle2py.families.negbinomial_size`.

    Returns
    -------
    anndata.AnnData
        AnnData with ``adata.uns['monocle2']`` populated and
        ``adata.obs['Size_Factor']`` initialised to ``NaN``.
    """
    import warnings

    if isinstance(cell_data, AnnData):
        adata = cell_data
    else:
        X = _coerce_matrix(cell_data)
        obs = _build_frame(pheno_data, n=X.shape[0], axis="obs")
        var = _build_frame(feature_data, n=X.shape[1], axis="var")
        adata = AnnData(X=X, obs=obs, var=var)

    if "gene_short_name" not in adata.var.columns:
        warnings.warn(
            "featureData must contain a column verbatim named "
            "'gene_short_name' for certain functions",
            stacklevel=2,
        )

    if expression_family is None:
        expression_family = negbinomial_size()

    ensure_state(adata)
    set_expression_family(adata, expression_family)
    set_lower_detection_limit(adata, lower_detection_limit)

    if SIZE_FACTOR_COL not in adata.obs.columns:
        adata.obs[SIZE_FACTOR_COL] = np.full(adata.n_obs, np.nan, dtype=float)

    return adata


def _coerce_matrix(x: Any) -> np.ndarray | spmatrix:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    if issparse(x):
        return csr_matrix(x)
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(
        "cell_data must be a NumPy ndarray, scipy sparse matrix, pandas "
        "DataFrame, or AnnData."
    )


def _build_frame(
    df: pd.DataFrame | None, *, n: int, axis: str
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(index=pd.RangeIndex(n).astype(str))
    if len(df) != n:
        raise ValueError(
            f"{axis} data length {len(df)} does not match cell_data "
            f"axis length {n}"
        )
    return df.copy()
