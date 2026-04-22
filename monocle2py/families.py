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
    "negbinomial_size",
    "negbinomial",
    "tobit",
    "gaussian_family",
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
    """Gaussian family for already-transformed expression (VGAM ``gaussianff``)."""

    vfamily: str = "gaussianff"


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
    """Build a :class:`GaussianFamily` family (mirrors ``VGAM::gaussianff``)."""
    return GaussianFamily()


_BY_NAME: dict[str, ExpressionFamily] = {
    "negbinomial.size": NegbinomialSize(),
    "negbinomial": Negbinomial(),
    "Tobit": Tobit(),
    "tobit": Tobit(),
    "gaussianff": GaussianFamily(),
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
    raise ValueError(f"Unknown expression family: {name!r}")
