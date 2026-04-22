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

__all__ = ["estimate_t", "relative2abs"]


def _as_dense(X: Any) -> np.ndarray:
    return np.asarray(X.toarray()) if issparse(X) else np.asarray(X)


def _bw_nrd0(x: np.ndarray) -> float:
    """Port of R's ``bw.nrd0`` (Silverman's rule-of-thumb bandwidth).

    Returns ``0.9 * min(sd(x), IQR(x) / 1.34) * n^(-0.2)``, with the same
    degenerate-input fallbacks used by ``stats::bw.nrd0``.
    """
    n = int(x.size)
    if n < 2:
        raise ValueError("need at least 2 data points")
    hi = float(np.std(x, ddof=1))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    lo = min(hi, float(q75 - q25) / 1.34)
    if not (lo > 0.0):
        if hi > 0.0:
            lo = hi
        elif abs(float(x[0])) > 0.0:
            lo = abs(float(x[0]))
        else:
            lo = 1.0
    return 0.9 * lo * n ** (-0.2)


def _bin_dist(
    x: np.ndarray, weights: np.ndarray, lo: float, up: float, n: int
) -> np.ndarray:
    """Port of R's ``C_BinDist`` linear binning.

    Returns an array of length ``2 * n`` whose first ``n`` entries hold the
    linearly-binned mass of *x* on the equally-spaced grid ``linspace(lo, up, n)``
    and whose trailing ``n`` entries are zero (R zero-pads for the linear
    FFT convolution used by :func:`_r_density`).
    """
    result = np.zeros(2 * n, dtype=float)
    xdelta = (up - lo) / (n - 1)
    pos = (x - lo) / xdelta
    ix = np.floor(pos).astype(np.int64)
    fx = pos - ix
    in_range = (ix >= 0) & (ix < n - 1)
    np.add.at(result, ix[in_range], (1.0 - fx[in_range]) * weights[in_range])
    np.add.at(result, ix[in_range] + 1, fx[in_range] * weights[in_range])
    edge = (ix == n - 1) & (fx == 0.0)
    np.add.at(result, ix[edge], weights[edge])
    return result


def _r_density(
    x: np.ndarray,
    bw: float,
    n: int = 512,
    cut: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Byte-faithful port of ``stats::density`` for the Gaussian kernel.

    Uses linear binning on the extended grid ``[min-cut*bw-4*bw, max+cut*bw+4*bw]``
    and convolves with a Gaussian kernel of standard deviation ``bw`` via an
    FFT that matches R's ``fft(y) * Conj(fft(kords))`` step by zero-padding
    the binned counts to length ``2n``. The output is resampled back to the
    user grid ``linspace(min-cut*bw, max+cut*bw, n)``.
    """
    x = np.ascontiguousarray(x, dtype=float)
    x = x[np.isfinite(x)]
    nx = int(x.size)
    if nx < 2:
        raise ValueError("need at least 2 finite data points")

    n_user = int(n)
    n_fft = max(n_user, 512)
    if n_fft > 512:
        n_fft = 1 << int(np.ceil(np.log2(n_fft)))

    from_ = float(np.min(x)) - cut * bw
    to = float(np.max(x)) + cut * bw
    lo = from_ - 4.0 * bw
    up = to + 4.0 * bw

    weights = np.full(nx, 1.0 / nx, dtype=float)
    y = _bin_dist(x, weights, lo, up, n_fft)  # length 2*n_fft, zero-padded

    kords = np.linspace(0.0, 2.0 * (up - lo), 2 * n_fft)
    kords[n_fft + 1:2 * n_fft] = -kords[1:n_fft][::-1]
    inv = 1.0 / (np.sqrt(2.0 * np.pi) * bw)
    kords = inv * np.exp(-0.5 * (kords / bw) ** 2)

    # ``numpy.fft.ifft`` divides by ``N`` while R's ``fft(..., inverse=TRUE)``
    # does not, so multiply back by ``2 * n_fft``. R then divides the real part
    # by ``length(y) == 2 * n_fft``.
    conv = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kords))) * (2 * n_fft)
    kords_dens = np.maximum(0.0, conv.real[:n_fft]) / (2 * n_fft)

    xords = np.linspace(lo, up, n_fft)
    x_out = np.linspace(from_, to, n_user)
    y_out = np.interp(x_out, xords, kords_dens)
    return x_out, y_out


def _dmode(x: np.ndarray) -> float:
    """Gaussian-kernel mode of *x*. Mirrors ``dmode`` in normalization.R.

    Uses :func:`_r_density` with :func:`_bw_nrd0` bandwidth so the result is
    byte-equivalent to ``density(x, bw = bw.nrd0(x), kernel = "gaussian",
    n = 512)$x[which.max(...$y)]`` from R's ``stats`` package. If multiple
    grid points tie for the max (rare), returns their mean, matching R's
    ``den$x[den$y == max(den$y)]`` followed by ``mean`` in ``estimate_t``.
    """
    if x.size < 2:
        return 0.0
    bw = _bw_nrd0(x)
    gx, gy = _r_density(x, bw=bw, n=512)
    max_y = gy.max()
    tied = gx[gy == max_y]
    return float(tied.mean())


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
