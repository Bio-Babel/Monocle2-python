"""Census: relative → absolute transcript counts (``relative2abs``).

Ports the non-ERCC ``method="num_genes"`` branch of ``normalization.R``. The
ERCC spike-in branch and the ``tpm_fraction`` branch are intentionally not
carried over (see ``port_reports/monocle2/05_design.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import gaussian_kde

__all__ = ["estimate_t", "relative2abs"]


def _as_dense(X: Any) -> np.ndarray:
    return np.asarray(X.toarray()) if issparse(X) else np.asarray(X)


def _dmode(x: np.ndarray) -> float:
    """Gaussian-kernel mode of *x*. Mirrors ``dmode`` in normalization.R."""
    if x.size < 2:
        return 0.0
    kde = gaussian_kde(x, bw_method="scott")
    grid = np.linspace(x.min(), x.max(), 512)
    den = kde(grid)
    return float(grid[int(np.argmax(den))])


def estimate_t(
    relative_expr_matrix: np.ndarray,
    relative_expr_thresh: float = 0.1,
) -> np.ndarray:
    """Per-cell ``t`` estimate — the abundance that marks one transcript copy.

    Parameters
    ----------
    relative_expr_matrix : numpy.ndarray
        Cells x genes relative-expression matrix (FPKM/TPM).
    relative_expr_thresh : float, default 0.1
        Lower threshold applied before the log10 mode is computed.

    Returns
    -------
    numpy.ndarray
        Length-``n_cells`` vector of ``t`` estimates in the original units.
    """
    X = _as_dense(relative_expr_matrix)
    n_cells = X.shape[0]
    t_hat = np.empty(n_cells, dtype=float)
    for i in range(n_cells):
        values = X[i, :]
        selected = values[values > relative_expr_thresh]
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log10(selected)
        logs = logs[np.isfinite(logs)]
        mode = _dmode(logs)
        t_hat[i] = 10.0 ** float(mode)
    return t_hat


def _calibrate_per_cell_total_proposal(
    relative_expr_matrix: np.ndarray,
    t_estimate: np.ndarray,
    expected_capture_rate: float,
) -> np.ndarray:
    """``method='num_genes'`` branch of ``calibrate_per_cell_total_proposal``.

    For each cell, computes the per-cell proposed mRNA total as
    ``num_single_copy_genes / P(t_estimate) / expected_capture_rate``
    where ``P`` is the empirical CDF of the cell's relative expression
    (restricted to values > 0.1) and ``num_single_copy_genes`` is the number
    of values at or below ``t_estimate[i]``.
    """
    X = _as_dense(relative_expr_matrix)
    n_cells = X.shape[0]
    proposals = np.empty(n_cells, dtype=float)
    for i in range(n_cells):
        x = X[i, :]
        x = x[x > 0.1]
        ti = float(t_estimate[i])
        frac_x = float(np.mean(x <= ti)) if x.size else 0.0
        num_single = int(np.sum(x <= ti))
        if frac_x == 0 or expected_capture_rate == 0:
            proposals[i] = 0.0
        else:
            proposals[i] = num_single / frac_x / expected_capture_rate
    return proposals


def relative2abs(
    relative_cds: AnnData,
    t_estimate: np.ndarray | None = None,
    model_formula_str: str = "~1",
    expected_capture_rate: float = 0.25,
    method: str = "num_genes",
    return_all: bool = False,
) -> np.ndarray | dict[str, Any]:
    """Convert a relative-expression matrix (FPKM/TPM) to absolute counts.

    Parameters
    ----------
    relative_cds : anndata.AnnData
        Cells x genes AnnData holding the relative expression matrix.
    t_estimate : numpy.ndarray, optional
        Pre-computed per-cell ``t`` vector. If omitted it is derived via
        :func:`estimate_t`.
    model_formula_str : str, default ``"~1"``
        Retained for API parity with the R signature. Ignored in the
        ``num_genes`` path.
    expected_capture_rate : float, default 0.25
        Assumed fraction of transcripts captured per cell (Census prior).
    method : str, default ``"num_genes"``
        Only ``"num_genes"`` is supported in this port. ``"tpm_fraction"``
        and the ERCC spike-in route are intentionally not ported.
    return_all : bool, default False
        If True, return a dict with the census matrix plus the intermediate
        ``t_estimate`` and ``expected_total_mRNAs`` vectors.

    Returns
    -------
    numpy.ndarray or dict
        Cells x genes matrix of absolute transcript counts, or a dict when
        ``return_all`` is True.
    """
    if method != "num_genes":
        raise NotImplementedError(
            "Only method='num_genes' is supported; 'tpm_fraction' and the "
            "ERCC branch are not ported."
        )

    X = _as_dense(relative_cds.X)

    if t_estimate is None:
        t_hat = estimate_t(X)
    else:
        t_hat = np.asarray(t_estimate, dtype=float)
        if t_hat.shape != (X.shape[0],):
            raise ValueError(
                f"t_estimate shape {t_hat.shape} does not match n_cells={X.shape[0]}"
            )

    params = np.concatenate([t_hat, [expected_capture_rate]])
    if not np.all(np.isfinite(params)):
        raise ValueError(
            "Input parameters must be finite (t_estimate, expected_capture_rate)."
        )

    expected_total_mRNAs = _calibrate_per_cell_total_proposal(
        X, t_hat, expected_capture_rate
    )

    cell_totals = X.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        expr_probs = np.where(
            cell_totals[:, None] > 0, X / cell_totals[:, None], 0.0
        )
    census = expr_probs * expected_total_mRNAs[:, None]

    if return_all:
        return {
            "norm_cds": census,
            "t_estimate": t_hat,
            "expected_total_mRNAs": expected_total_mRNAs,
        }
    return census
