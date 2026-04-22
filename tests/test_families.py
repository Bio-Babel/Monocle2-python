"""Tests for the expression-family marker classes."""

from __future__ import annotations

import pytest

from monocle2py.families import (
    GaussianFamily,
    Negbinomial,
    NegbinomialSize,
    Tobit,
    family_from_name,
    gaussian_family,
    negbinomial,
    negbinomial_size,
    tobit,
)


def test_vfamily_strings_match_vgam() -> None:
    assert negbinomial_size().vfamily == "negbinomial.size"
    assert negbinomial().vfamily == "negbinomial"
    assert tobit().vfamily == "Tobit"
    assert gaussian_family().vfamily == "gaussianff"


def test_family_factory_parameters() -> None:
    nb = negbinomial_size(size=3.5)
    assert isinstance(nb, NegbinomialSize)
    assert nb.size == pytest.approx(3.5)

    tb = tobit(lower=0.01)
    assert isinstance(tb, Tobit)
    assert tb.lower == pytest.approx(0.01)


def test_family_from_name_round_trip() -> None:
    assert isinstance(family_from_name("negbinomial.size"), NegbinomialSize)
    assert isinstance(family_from_name("negbinomial"), Negbinomial)
    assert isinstance(family_from_name("Tobit"), Tobit)
    assert isinstance(family_from_name("tobit"), Tobit)
    assert isinstance(family_from_name("gaussianff"), GaussianFamily)
    with pytest.raises(ValueError):
        family_from_name("not_a_family")
