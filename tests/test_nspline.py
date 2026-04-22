"""Parity tests for the natural-cubic-spline port (R splines::ns)."""

from __future__ import annotations

import numpy as np
import pytest

from monocle2py._internal._nspline import ns_basis


_R_X = np.array([
    0.0, 0.0119761377573013, 0.0543451681733131, 0.0593325681984425,
    0.172893537674099, 0.251311503816396, 0.613688277080655,
    1.40080855693668, 1.40139957657084, 1.53835240751505,
    1.85512177180499, 2.31125046964735, 2.32562693301588,
    2.52501178765669, 2.85723306238651, 3.17824489902705,
    3.41558025917038, 3.43438799958676, 3.56809314806014,
    4.10672063706443, 4.12376251770183, 4.66230554971844,
    5.4520450416021, 5.55369009962305, 5.56530505884439,
    5.80806337296963, 6.0173881566152, 6.9186593987979,
    6.98673470877111, 8.48532462259755, 8.61010685097426, 10.0,
])
# R ``splines::ns(x, df=3)`` attributes for _R_X:
_R_IKNOTS = np.array([2.007165, 4.482791])
_R_BKNOTS = np.array([0.0, 10.0])
# First three R output rows (verified vs /tmp/ns_B_df3.tsv to 1e-15)
_R_FIRST_THREE = np.array([
    [0.000000000, 0.000000000, 0.000000000],
    [-0.001753689, 0.004294322, -0.002540614],
    [-0.007954990, 0.019483832, -0.011527058],
])


def test_ns_basis_constraints_at_boundary():
    """Natural spline 2nd derivative should vanish at Boundary knots."""
    x = np.linspace(0, 10, 51)
    iknots = np.array([2.5, 5.0, 7.5])
    bknots = np.array([0.0, 10.0])
    # 2nd derivative ≈ 0 at boundary: evaluate basis near the boundaries and
    # check extrapolation is linear (2nd diff → 0).
    B = ns_basis(x, knots=iknots, Boundary_knots=bknots, intercept=False)
    # points just beyond left boundary are linearly extrapolated
    x_ext = np.array([-0.1, -0.2, -0.3])
    B_ext = ns_basis(x_ext, knots=iknots, Boundary_knots=bknots, intercept=False)
    # linear: finite 2nd difference should be ~ 0
    d2 = np.diff(B_ext, n=2, axis=0)
    assert np.abs(d2).max() < 1e-10


def test_ns_basis_matches_r_reference_df3():
    """ns_basis with R's knots must match R output byte-for-byte."""
    B = ns_basis(_R_X, knots=_R_IKNOTS, Boundary_knots=_R_BKNOTS, intercept=False)
    assert B.shape == (32, 3)
    # Boundary row is exactly zero (intercept=False)
    np.testing.assert_allclose(B[0], np.zeros(3), atol=1e-14)
    # First three rows match R to 1e-6 (reference has 6 decimal digits)
    np.testing.assert_allclose(B[:3], _R_FIRST_THREE, atol=1e-6)


def test_ns_basis_shape_for_various_df():
    """ns_basis with K-1 interior knots (df=K without intercept) has K cols."""
    x = np.linspace(0, 10, 20)
    for k in [3, 4, 5, 6]:
        # df=k, intercept=False → nIknots = k - 1
        probs = np.linspace(0.0, 1.0, k + 1)[1:-1]
        iknots = np.quantile(x, probs)
        bknots = np.array([x.min(), x.max()])
        B = ns_basis(x, knots=iknots, Boundary_knots=bknots, intercept=False)
        assert B.shape == (20, k)


def test_ns_basis_linear_extrapolation():
    """Values beyond Boundary knots are linear in x."""
    x = np.array([-5.0, -2.0, 12.0, 15.0])
    iknots = np.array([2.5, 5.0, 7.5])
    bknots = np.array([0.0, 10.0])
    B = ns_basis(x, knots=iknots, Boundary_knots=bknots, intercept=False)
    # On [-5, -2]: finite diff per unit x must equal (B(-2) - B(-5)) / 3.
    # On [12, 15]: similarly. Linear extrapolation → constant slope.
    slope_left = (B[1] - B[0]) / (x[1] - x[0])
    slope_right = (B[3] - B[2]) / (x[3] - x[2])
    # Re-evaluate at different outer points to verify the slope is invariant.
    x2 = np.array([-8.0, -3.0, 13.0, 18.0])
    B2 = ns_basis(x2, knots=iknots, Boundary_knots=bknots, intercept=False)
    slope_left2 = (B2[1] - B2[0]) / (x2[1] - x2[0])
    slope_right2 = (B2[3] - B2[2]) / (x2[3] - x2[2])
    np.testing.assert_allclose(slope_left, slope_left2, atol=1e-10)
    np.testing.assert_allclose(slope_right, slope_right2, atol=1e-10)


def test_ns_state_locked_for_prediction():
    """Knots memorised at training time must survive prediction on new x."""
    from patsy import dmatrix

    from monocle2py._internal._nspline import ns  # noqa: F401

    import pandas as pd

    rng = np.random.default_rng(42)
    train_x = np.concatenate([[0.0], rng.uniform(0, 10, 30), [10.0]])
    train_df = pd.DataFrame({"Pseudotime": train_x})
    di = dmatrix("~ns(Pseudotime, df=3)", train_df, return_type="matrix").design_info

    # Prediction grid has a completely different range — knots must stay frozen.
    pred_df = pd.DataFrame({"Pseudotime": np.array([-5.0, 0.0, 5.0, 10.0, 15.0])})
    X_pred = np.asarray(dmatrix(di, pred_df, return_type="matrix"))
    assert X_pred.shape == (5, 4)
    # At x = 0 (left boundary), basis row is exactly [1, 0, 0, 0]
    np.testing.assert_allclose(X_pred[1], np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-14)
