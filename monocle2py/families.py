"""Expression-family markers mirroring R's VGAM family objects.

Monocle2 carries a VGAM family on the CellDataSet and dispatches algorithm
choices (size-factor normalisation, dispersion fitting, VGAM regression) based
on ``family@vfamily``. The Python port preserves this contract: family objects
are thin value classes with a stable ``vfamily`` string and parameter slots.
Downstream algorithms branch on the ``vfamily`` identifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "ExpressionFamily",
    "NegbinomialSize",
    "Negbinomial",
    "Tobit",
    "GaussianFamily",
    "BinomialFamily",
    "negbinomial_size",
    "negbinomial",
    "tobit",
    "gaussian_family",
    "binomialff",
    "family_from_name",
]


@dataclass(frozen=True)
class ExpressionFamily:
    """Base class for expression families.

    Attributes
    ----------
    vfamily : str
        The VGAM-compatible family identifier used for dispatch.
    """

    vfamily: str


@dataclass(frozen=True)
class NegbinomialSize(ExpressionFamily):
    """Negative-binomial with fixed ``size`` parameter (VGAM ``negbinomial.size``).

    ``size`` is VGAM's parameter ``k`` in ``Var(Y) = mu + mu^2 / k``.
    Default ``float('inf')`` matches R ``VGAM::negbinomial.size()`` which
    degenerates to a Poisson model. When Monocle's ``fit_model_helper``
    has a valid ``disp_func`` it rebuilds the family with
    ``size = 1 / disp_func(mean(x))``.
    """

    size: float = math.inf
    vfamily: str = "negbinomial.size"


@dataclass(frozen=True)
class Negbinomial(ExpressionFamily):
    """Standard negative binomial family (VGAM ``negbinomial``).

    Unlike ``NegbinomialSize``, the size (dispersion) parameter is
    **jointly estimated** alongside the mean parameters.
    """

    vfamily: str = "negbinomial"


@dataclass(frozen=True)
class Tobit(ExpressionFamily):
    """Tobit censored-regression family (VGAM ``tobit``).

    Mirrors ``VGAM::tobit(Lower, Upper, lmu="identitylink", lsd="loglink")``.
    In monocle2 the response fed to the GLM is ``log10(x)``
    (``expr_models.R:45-47``), so ``lower`` / ``upper`` are thresholds on the
    log10 scale. Defaults match R's ``tobit()`` (Lower=0, Upper=+Inf).
    """

    lower: float = 0.0
    upper: float = math.inf
    vfamily: str = "Tobit"


@dataclass(frozen=True)
class GaussianFamily(ExpressionFamily):
    """Gaussian family for raw-response continuous data (VGAM ``uninormal``).

    Monocle2 R uses VGAM's ``uninormal`` family (univariate normal with
    raw, non-log-transformed response) — see ``order_cells.R:1266-1271``
    and ``expr_models.R:42-47``. Earlier versions of this port stored
    ``vfamily = "gaussianff"`` (a different VGAM alias that Monocle2 R
    never references), which caused two divergences from R:

    * ``normalize_expr_data`` matched on the wrong string;
    * ``make_response`` in ``_vgam.py`` log10-transformed the response
      instead of feeding it raw to the GLM (R's ``uninormal`` rule).

    The canonical ``vfamily`` is now ``"uninormal"``. The previous
    ``"gaussianff"`` string is intentionally rejected by
    :func:`family_from_name` with a one-line migration hint rather than
    silently aliased — see :doc:`feedback-r-port-algorithm-not-signature`
    for the meta-principle.
    """

    vfamily: str = "uninormal"


@dataclass(frozen=True)
class BinomialFamily(ExpressionFamily):
    """Binomial family for binarized data (VGAM ``binomialff``).

    Used by Monocle2 for binarized single-cell ATAC peak-presence data
    or any other 0/1 response. The trajectory pipeline applies a
    TF-IDF transform in ``normalize_expr_data`` (mirrors
    ``order_cells.R:1248-1256``); per-gene differential tests fit a
    logistic GLM via :func:`statsmodels.api.families.Binomial`
    (``_vgam.py:make_family`` handles the dispatch).
    """

    vfamily: str = "binomialff"


def negbinomial_size(size: float = math.inf) -> NegbinomialSize:
    """Build a :class:`NegbinomialSize` family (mirrors ``VGAM::negbinomial.size``).

    Default ``size=Inf`` matches R's ``VGAM::negbinomial.size()`` (Poisson
    degeneracy). Pass an explicit finite ``size`` to get a fixed-dispersion
    NB; Monocle's differential test overrides this with
    ``size = 1/disp_func(mean)`` when a dispersion function is available.
    """
    return NegbinomialSize(size=float(size))


def negbinomial() -> Negbinomial:
    """Build a :class:`Negbinomial` family (mirrors ``VGAM::negbinomial``)."""
    return Negbinomial()


def tobit(lower: float = 0.0, upper: float = math.inf) -> Tobit:
    """Build a :class:`Tobit` family (mirrors ``VGAM::tobit(Lower, Upper)``).

    Both thresholds are on the log10 response scale that monocle2 feeds to
    the GLM. R defaults: ``Lower=0``, ``Upper=+Inf``.
    """
    return Tobit(lower=float(lower), upper=float(upper))


def gaussian_family() -> GaussianFamily:
    """Build a :class:`GaussianFamily` family (mirrors ``VGAM::uninormal``)."""
    return GaussianFamily()


def binomialff() -> BinomialFamily:
    """Build a :class:`BinomialFamily` (mirrors ``VGAM::binomialff``)."""
    return BinomialFamily()


_BY_NAME: dict[str, ExpressionFamily] = {
    "negbinomial.size": NegbinomialSize(),
    "negbinomial": Negbinomial(),
    "Tobit": Tobit(),
    "tobit": Tobit(),
    "uninormal": GaussianFamily(),
    "binomialff": BinomialFamily(),
}


def family_from_name(name: str) -> ExpressionFamily:
    """Resolve a VGAM family name to an :class:`ExpressionFamily` instance.

    Parameters
    ----------
    name : str
        The VGAM ``vfamily`` string (case sensitive except ``Tobit``/``tobit``).

    Returns
    -------
    ExpressionFamily
    """
    if name in _BY_NAME:
        return _BY_NAME[name]
    if name == "gaussianff":
        raise ValueError(
            "Expression family 'gaussianff' was a pre-rename Python-side "
            "mislabel of the VGAM 'uninormal' family — Monocle2 R always uses "
            "'uninormal' (see order_cells.R:1266 and expr_models.R:42). If "
            "loading a legacy h5ad written by an earlier monocle2-python, run:\n"
            "    adata.uns['monocle2']['expression_family'] = 'uninormal'\n"
            "and re-save."
        )
    raise ValueError(f"Unknown expression family: {name!r}")
