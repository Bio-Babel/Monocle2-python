"""Formula utilities mirroring R's VGAM formula syntax.

R's ``differentialGeneTest`` and ``fitModel`` accept formula strings such as
``"~sm.ns(Pseudotime, df=3)"``. ``sm.ns`` is VGAM's re-export of
``splines::ns`` (natural cubic spline with linearity beyond Boundary knots).
We rewrite ``sm.ns(...)``/``splines::ns(...)``/``ns(...)`` to our own
``ns(...)`` patsy stateful transform (:mod:`_nspline`), which reproduces R's
basis to machine precision (1e-15) and memorises Boundary + interior knots
on the training pass so predictions on a new grid re-use the training basis.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from patsy import dmatrix

from ._nspline import ns  # noqa: F401 — exposed to patsy's eval_env

__all__ = [
    "DEFAULT_FULL_FORMULA",
    "DEFAULT_REDUCED_FORMULA",
    "formula_terms",
    "normalize_formula",
    "build_design_matrix",
]

DEFAULT_FULL_FORMULA = "~sm.ns(Pseudotime, df=3)"
DEFAULT_REDUCED_FORMULA = "~1"

_NS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bsm\.ns\(", "ns("),
    (r"\bsplines::ns\(", "ns("),
    (r"\bVGAM::sm\.ns\(", "ns("),
)
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FUNC_NAMES = {
    "ns", "bs", "cr", "cc", "te", "I", "C", "log", "log10", "log2", "exp",
    "sqrt", "sin", "cos", "scale", "center", "df", "degree", "intercept",
    "include_intercept", "knots", "lower_bound", "upper_bound",
    "Boundary_knots",
}


def normalize_formula(formula_str: str) -> str:
    """Translate R-side smoothers (``sm.ns``, ``ns``) to patsy ``bs``.

    Parameters
    ----------
    formula_str : str
        Formula string in R notation. Must include the leading ``~``.

    Returns
    -------
    str
        Formula string with smoother names rewritten for patsy.
    """
    s = formula_str.strip()
    if not s.startswith("~"):
        s = "~" + s
    for pat, repl in _NS_PATTERNS:
        s = re.sub(pat, repl, s)
    return s


def formula_terms(formula_str: str) -> list[str]:
    """Return the bare variable names referenced by *formula_str*.

    Mirrors R's ``all.vars(formula(...))`` for the purpose of locating
    columns in ``adata.obs`` that the formula will need.
    """
    s = normalize_formula(formula_str)
    rhs = s.split("~", 1)[-1]
    out: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(\s*=)?(\s*\()?", rhs):
        name = match.group(1)
        is_kwarg = match.group(2) is not None
        is_call = match.group(3) is not None
        if is_call or is_kwarg:
            continue
        if name in _FUNC_NAMES or name in {"True", "False", "None"}:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def build_design_matrix(
    formula_str: str,
    data: pd.DataFrame,
    *,
    extra_cols: Iterable[tuple[str, np.ndarray]] = (),
) -> np.ndarray:
    """Build a patsy design matrix from an R-style formula.

    Parameters
    ----------
    formula_str : str
        R-style formula. Smoothers ``sm.ns``/``ns`` are rewritten to ``bs``.
    data : pandas.DataFrame
        Source columns (typically ``adata.obs``).
    extra_cols : iterable of ``(name, array)`` pairs, optional
        Additional columns to splice into the data frame before evaluation
        (e.g. ``("f_expression", y)`` for response-side terms).

    Returns
    -------
    numpy.ndarray
        Dense design matrix. Always includes an intercept column unless the
        formula explicitly drops it via ``- 1`` / ``+ 0``.
    """
    formula = normalize_formula(formula_str)
    df = data.copy()
    for name, arr in extra_cols:
        df[name] = np.asarray(arr)
    return np.asarray(dmatrix(formula, df, return_type="matrix"))
