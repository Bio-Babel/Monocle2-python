"""Expression-family markers mirroring R's VGAM family objects.

Monocle2 carries a VGAM family on the CellDataSet and dispatches algorithm
choices (size-factor normalisation, dispersion fitting, VGAM regression) based
on ``family@vfamily``. The Python port preserves this contract: family objects
are thin value classes with a stable ``vfamily`` string and parameter slots.
Downstream algorithms branch on the ``vfamily`` identifier.
"""

from __future__ import annotations

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
    """Negative-binomial with fixed ``mu / k`` relationship (VGAM ``negbinomial.size``)."""

    size: float = 1.0
    vfamily: str = "negbinomial.size"


@dataclass(frozen=True)
class Negbinomial(ExpressionFamily):
    """Standard negative binomial family (VGAM ``negbinomial``)."""

    vfamily: str = "negbinomial"


@dataclass(frozen=True)
class Tobit(ExpressionFamily):
    """Tobit censored-regression family (VGAM ``tobit``).

    ``Lower`` matches VGAM's lower-censoring threshold. In Monocle2 this is
    used for log-scaled FPKM/TPM matrices before ``relative2abs`` converts
    them to transcript counts.
    """

    lower: float = 0.1
    vfamily: str = "Tobit"


@dataclass(frozen=True)
class GaussianFamily(ExpressionFamily):
    """Gaussian family for already-transformed expression (VGAM ``gaussianff``)."""

    vfamily: str = "gaussianff"


def negbinomial_size(size: float = 1.0) -> NegbinomialSize:
    """Build a :class:`NegbinomialSize` family (mirrors ``VGAM::negbinomial.size``)."""
    return NegbinomialSize(size=float(size))


def negbinomial() -> Negbinomial:
    """Build a :class:`Negbinomial` family (mirrors ``VGAM::negbinomial``)."""
    return Negbinomial()


def tobit(lower: float = 0.1) -> Tobit:
    """Build a :class:`Tobit` family (mirrors ``VGAM::tobit``)."""
    return Tobit(lower=float(lower))


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
