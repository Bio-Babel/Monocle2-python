"""statsmodels GLM wrappers mirroring VGAM's ``vglm`` calls in monocle2.

R's ``diff_test_helper`` and ``fit_model_helper`` build a per-gene GLM via
``VGAM::vglm`` with ``epsilon=1e-1`` and a family that depends on the cds.
For the negbinomial / negbinomial.size path we substitute ``statsmodels.GLM``
with a fixed dispersion ``alpha`` derived from the same dispersion-hint
function used by R (``calculate_NB_dispersion_hint``).

The two model fits (full and reduced) are compared with a likelihood ratio
test. ``compareModels`` in R returns ``status``, ``family``, ``pval``;
this module returns the same triple plus the test statistic and df for
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

__all__ = [
    "FitOutcome",
    "fit_glm",
    "lrt",
    "calculate_nb_dispersion_hint",
    "make_response",
    "make_family",
]


@dataclass
class FitOutcome:
    """Result of a single per-gene LRT fit."""

    status: str
    family: str
    pval: float
    statistic: float = float("nan")
    df: int = 0


def calculate_nb_dispersion_hint(
    disp_func: Callable[[np.ndarray], np.ndarray] | None,
    f_expression: np.ndarray,
    expr_selection_func: Callable[[np.ndarray], float] = np.mean,
) -> float | None:
    """Port of R's ``calculate_NB_dispersion_hint``.

    Returns the dispersion-fit value at the mean of the gene's expression,
    or ``None`` if the gene has no signal.
    """
    if disp_func is None:
        return None
    expr_hint = float(expr_selection_func(f_expression))
    if expr_hint <= 0 or np.isnan(expr_hint):
        return None
    val = float(np.asarray(disp_func(np.asarray([expr_hint])))[0])
    if np.isnan(val):
        return None
    return val


def make_response(
    x: np.ndarray,
    family_name: str,
    size_factor: np.ndarray,
    relative_expr: bool = True,
) -> np.ndarray:
    """Build the per-gene response vector matching ``diff_test_helper``."""
    x = np.asarray(x, dtype=float)
    if family_name in ("negbinomial", "negbinomial.size"):
        if relative_expr:
            x = x / np.asarray(size_factor, dtype=float)
        return np.round(x).astype(float)
    if family_name in ("uninormal", "binomialff", "gaussianff"):
        return x
    return np.log10(x)


def make_family(family_name: str, alpha: float = 1.0) -> Any:
    """Map a VGAM ``vfamily`` name to a statsmodels GLM family."""
    if family_name in ("negbinomial", "negbinomial.size"):
        return sm.families.NegativeBinomial(alpha=float(alpha))
    if family_name in ("uninormal", "gaussianff", "Tobit", "tobit"):
        return sm.families.Gaussian()
    if family_name == "binomialff":
        return sm.families.Binomial()
    return sm.families.Gaussian()


def fit_glm(y: np.ndarray, X: np.ndarray, family: Any) -> Any:
    """Fit a single statsmodels GLM with VGAM-comparable tolerance."""
    return sm.GLM(y, X, family=family).fit(maxiter=100, atol=1e-1)


def lrt(full_fit: Any, reduced_fit: Any) -> tuple[float, int, float]:
    """Likelihood-ratio test on two nested GLM fits."""
    stat = 2.0 * float(full_fit.llf - reduced_fit.llf)
    if stat < 0:
        stat = 0.0
    dof = int(full_fit.df_model - reduced_fit.df_model)
    if dof <= 0:
        return stat, dof, float("nan")
    return stat, dof, float(chi2.sf(stat, dof))
