"""Port of R ``splines::ns`` (natural cubic spline with boundary linearity).

R's ``sm.ns`` (VGAM alias) and ``splines::ns`` are byte-identical on dense
numeric input; this module reproduces that basis exactly. The resulting
patsy transform ``ns`` memorises Boundary knots and interior knots on the
training pass so predictions on a new grid (used by ``response_matrix`` /
``gen_smooth_curves``) re-use the training basis — matching R's behaviour
when the ``ns`` object attributes are carried into ``predict.lm``.

Parity against R ``splines::ns`` verified to 2e-15 (machine precision)
across four basis shapes (df=3 / df=3 intercept=TRUE / df=4 / explicit
knots). See tests/test_nspline.py.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from patsy.state import stateful_transform
from scipy.interpolate import BSpline

__all__ = ["ns", "ns_basis"]


def _spline_design(
    knots: np.ndarray,
    x: np.ndarray,
    ord: int = 4,
    derivs: Iterable[int] | int | None = None,
) -> np.ndarray:
    """Port of R ``splines::splineDesign``.

    Evaluates the B-spline basis of order ``ord`` (degree ``ord - 1``) on
    the knot sequence ``knots`` at points ``x``, optionally returning the
    ``derivs[i]``-th derivative at each point.
    """
    knots = np.asarray(knots, dtype=float)
    x = np.asarray(x, dtype=float)
    k = ord - 1
    n_basis = len(knots) - ord
    n = len(x)
    if derivs is None:
        derivs = np.zeros(n, dtype=int)
    else:
        derivs = np.asarray(derivs, dtype=int)
        if derivs.size == 1:
            derivs = np.full(n, int(derivs), dtype=int)
    out = np.zeros((n, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl = BSpline(knots, c, k, extrapolate=False)
        for d_val in np.unique(derivs):
            mask = derivs == d_val
            if not mask.any():
                continue
            if d_val == 0:
                y = spl(x[mask])
            else:
                y = spl.derivative(int(d_val))(x[mask])
            out[mask, j] = np.where(np.isnan(y), 0.0, y)
    return out


def ns_basis(
    x,
    *,
    knots: np.ndarray,
    Boundary_knots: np.ndarray,
    intercept: bool = False,
) -> np.ndarray:
    """Evaluate the natural-spline basis with fixed knots + boundary.

    Pure numerical kernel; state (knot selection) is handled by the patsy
    :func:`ns` transform.
    """
    x = np.asarray(x, dtype=float)
    nax = np.isnan(x)
    x_valid = x[~nax] if nax.any() else x
    Boundary_knots = np.sort(np.asarray(Boundary_knots, dtype=float))
    knots = np.asarray(knots, dtype=float)
    nIknots = len(knots)

    ol = x_valid < Boundary_knots[0]
    or_ = x_valid > Boundary_knots[1]
    outside = ol | or_

    Aknots = np.sort(np.concatenate([np.repeat(Boundary_knots, 4), knots]))
    n = len(x_valid)

    if outside.any():
        basis = np.zeros((n, nIknots + 4))
        if ol.any():
            k_pivot = Boundary_knots[0]
            xl = np.column_stack([np.ones(ol.sum()), x_valid[ol] - k_pivot])
            tt = _spline_design(
                Aknots, np.array([k_pivot, k_pivot]), ord=4, derivs=[0, 1]
            )
            basis[ol, :] = xl @ tt
        if or_.any():
            k_pivot = Boundary_knots[1]
            xr = np.column_stack([np.ones(or_.sum()), x_valid[or_] - k_pivot])
            tt = _spline_design(
                Aknots, np.array([k_pivot, k_pivot]), ord=4, derivs=[0, 1]
            )
            basis[or_, :] = xr @ tt
        inside = ~outside
        if inside.any():
            basis[inside, :] = _spline_design(Aknots, x_valid[inside], ord=4)
    else:
        basis = _spline_design(Aknots, x_valid, ord=4)

    const = _spline_design(Aknots, Boundary_knots, ord=4, derivs=[2, 2])

    if not intercept:
        const = const[:, 1:]
        basis = basis[:, 1:]

    Q, _ = np.linalg.qr(const.T, mode="complete")
    basis = (Q.T @ basis.T)[2:].T

    if nax.any():
        out = np.full((len(x), basis.shape[1]), np.nan)
        out[~nax, :] = basis
        return out
    return basis


def _compute_knots_from_data(
    x_all: np.ndarray,
    df: int | None,
    knots: np.ndarray | None,
    intercept: bool,
    Boundary_knots: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve Boundary knots + interior knots from training data — matches R ``ns``."""
    nax = np.isnan(x_all)
    x_valid = x_all[~nax] if nax.any() else x_all
    if Boundary_knots is None:
        bknots = np.array([x_valid.min(), x_valid.max()], dtype=float)
    else:
        bknots = np.sort(np.asarray(Boundary_knots, dtype=float))
    if knots is not None:
        iknots = np.asarray(knots, dtype=float)
    elif df is not None:
        nI = df - 1 - int(intercept)
        if nI <= 0:
            iknots = np.array([], dtype=float)
        else:
            outside = (x_valid < bknots[0]) | (x_valid > bknots[1])
            probs = np.linspace(0.0, 1.0, nI + 2)[1:-1]
            iknots = np.quantile(x_valid[~outside], probs)
    else:
        iknots = np.array([], dtype=float)
    return iknots, bknots


class _NS:
    """Stateful patsy transform implementing ``ns(x, df=..., ...)``."""

    def __init__(self):
        self._x_chunks: list[np.ndarray] | None = []
        self._args: dict | None = None
        self.knots_: np.ndarray | None = None
        self.Boundary_knots_: np.ndarray | None = None
        self.intercept_: bool | None = None

    def memorize_chunk(
        self,
        x,
        df=None,
        knots=None,
        intercept=False,
        Boundary_knots=None,
    ):
        if self._args is None:
            self._args = {
                "df": df,
                "knots": knots,
                "intercept": intercept,
                "Boundary_knots": Boundary_knots,
            }
        self._x_chunks.append(np.asarray(x, dtype=float))

    def memorize_finish(self):
        x_all = np.concatenate(self._x_chunks)
        args = self._args
        iknots, bknots = _compute_knots_from_data(
            x_all,
            df=args["df"],
            knots=args["knots"],
            intercept=bool(args["intercept"]),
            Boundary_knots=args["Boundary_knots"],
        )
        self.knots_ = iknots
        self.Boundary_knots_ = bknots
        self.intercept_ = bool(args["intercept"])
        self._x_chunks = None

    def transform(
        self,
        x,
        df=None,
        knots=None,
        intercept=False,
        Boundary_knots=None,
    ):
        return ns_basis(
            x,
            knots=self.knots_,
            Boundary_knots=self.Boundary_knots_,
            intercept=self.intercept_,
        )


ns = stateful_transform(_NS)
