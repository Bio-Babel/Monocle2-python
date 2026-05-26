"""Tests for the expression-family marker classes."""

from __future__ import annotations

import pytest

from monocle2py.families import (
    BinomialFamily,
    GaussianFamily,
    Negbinomial,
    NegbinomialSize,
    Tobit,
    binomialff,
    family_from_name,
    gaussian_family,
    negbinomial,
    negbinomial_size,
    tobit,
)


def test_vfamily_strings_match_vgam() -> None:
    """R Monocle2 contract: vfamily strings as used in
    ``order_cells.R:1203, 1248, 1257, 1266`` and ``expr_models.R:42``."""
    assert negbinomial_size().vfamily == "negbinomial.size"
    assert negbinomial().vfamily == "negbinomial"
    assert tobit().vfamily == "Tobit"
    # Pre-rename Python stored "gaussianff" here; canonical is now
    # "uninormal" to match R Monocle2 (see family class docstring).
    assert gaussian_family().vfamily == "uninormal"
    assert binomialff().vfamily == "binomialff"


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
    assert isinstance(family_from_name("uninormal"), GaussianFamily)
    assert isinstance(family_from_name("binomialff"), BinomialFamily)
    with pytest.raises(ValueError):
        family_from_name("not_a_family")


def test_family_from_name_gaussianff_migration_hint() -> None:
    """Legacy h5ad files stored 'gaussianff'; must point users at the
    one-line uns rewrite rather than silently aliasing."""
    with pytest.raises(ValueError, match="adata.uns\\['monocle2'\\]\\['expression_family'\\]"):
        family_from_name("gaussianff")
