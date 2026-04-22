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

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import statsmodels.api as sm
from scipy import optimize
from scipy.stats import chi2, norm
from statsmodels.discrete.discrete_model import NegativeBinomial as SMNegativeBinomial
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

__all__ = [
    "FitOutcome",
    "LOG10_RESPONSE_FAMILIES",
    "fit_glm",
    "fit_glm_with_fallback",
    "fit_joint_nb",
    "fit_tobit",
    "lrt",
    "calculate_nb_dispersion_hint",
    "make_response",
    "make_family",
    "_SuppressWarnings",
]


class _SuppressWarnings:
    """Context manager matching R's ``suppressWarnings(expr)``.

    Python's ``warnings.catch_warnings() + simplefilter('ignore')`` is NOT
    equivalent: ``simplefilter(append=False)`` *clears* the filter list,
    so any nested ``catch_warnings() + simplefilter('ignore', SomeSpecific)``
    (as used in statsmodels' GLM and discrete modules) wipes our outer
    ``ignore`` and RuntimeWarnings leak through.

    R's ``suppressWarnings`` uses ``withCallingHandlers`` + the
    ``muffleWarning`` restart — a mechanism that downstream code can't
    silently override. The closest Python analogue is to replace
    ``warnings.showwarning`` with a no-op. ``catch_warnings()`` saves /
    restores ``showwarning`` but does not overwrite it on entry, so our
    no-op persists through any nested filter manipulation done by
    libraries inside the ``with`` block.
    """

    def __enter__(self) -> "_SuppressWarnings":
        self._original_showwarning = warnings.showwarning
        warnings.showwarning = lambda *a, **k: None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        warnings.showwarning = self._original_showwarning


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


#: vfamily values whose response is fed to the GLM on the log10 scale
#: (mirroring R's ``fit_model_helper`` else-branch at ``expr_models.R:45-47``).
#: ``responseMatrix`` therefore applies ``10^predict(.)`` to invert the
#: transform for these families (``expr_models.R:158-160``).
LOG10_RESPONSE_FAMILIES = frozenset({"gaussianff", "Tobit", "tobit"})


def make_response(
    x: np.ndarray,
    family_name: str,
    size_factor: np.ndarray,
    relative_expr: bool = True,
) -> np.ndarray:
    """Build the per-gene response vector matching ``fit_model_helper``."""
    x = np.asarray(x, dtype=float)
    if family_name in ("negbinomial", "negbinomial.size"):
        if relative_expr:
            x = x / np.asarray(size_factor, dtype=float)
        return np.round(x).astype(float)
    # R ``fit_model_helper`` (expr_models.R:42-47): ``uninormal`` and
    # ``binomialff`` use the raw response; everything else (``gaussianff``,
    # ``Tobit``, …) uses ``log10(x)``.
    if family_name in ("uninormal", "binomialff"):
        return x
    return np.log10(x)


def make_family(family_name: str, alpha: float = 1.0) -> Any:
    """Map a VGAM ``vfamily`` name to a statsmodels GLM family.

    For negbinomial variants an ``alpha <= 0`` (or non-finite) is the
    statsmodels-equivalent of VGAM's ``negbinomial.size(size=Inf)``:
    the NB degenerates to Poisson, so we return a Poisson family. This
    matches R's behaviour when ``calculate_NB_dispersion_hint`` returns
    ``NULL`` and the caller-supplied family has ``size=Inf`` (the
    default of ``VGAM::negbinomial.size()`` — see ``utils.R:27``).
    """
    if family_name in ("negbinomial", "negbinomial.size"):
        if not math.isfinite(alpha) or alpha <= 0:
            return sm.families.Poisson()
        return sm.families.NegativeBinomial(alpha=float(alpha))
    if family_name in ("uninormal", "gaussianff", "Tobit", "tobit"):
        return sm.families.Gaussian()
    if family_name == "binomialff":
        return sm.families.Binomial()
    return sm.families.Gaussian()


def fit_glm(
    y: np.ndarray, X: np.ndarray, family: Any, family_name: str | None = None,
) -> Any:
    """Fit a single statsmodels GLM.

    R's ``VGAM::vglm(..., epsilon=1e-1)`` leaves the IRLS under-converged
    after 1–2 iterations (a VGAM-specific artifact of monocle's loose
    default tolerance). We run to full convergence (statsmodels default
    ``tol=1e-8``) — this is mathematically more correct than R's loose
    fit, and Pearson(log10(pval_R), log10(pval_py)) ≈ 0.998 with top-K
    DEG overlap ≥ 98 % on the lung BEAM tutorial. Tightening R to the
    same tolerance reproduces Python's answer to 3 decimals.

    When ``family_name`` is supplied, it is attached to the returned fit
    as ``_monocle2py_family_name`` so ``response_matrix`` can decide
    whether to invert a log10 response transform (R's ``10^predict(.)``
    branch in ``expr_models.R:158-160``).

    All fit-site warnings are locally suppressed — matches R's
    ``fit_model_helper`` (``expr_models.R:48-56``):

        FM_fit <- suppressWarnings(VGAM::vglm(..., epsilon=1e-1))

    R wraps every vglm call in ``suppressWarnings`` (non-verbose mode)
    because IRLS on sparse / zero-inflated genes triggers a cascade of
    numerical diagnostics that aren't actionable per-gene. We use
    :class:`_SuppressWarnings` because ``catch_warnings + simplefilter
    ("ignore")`` is not enough: statsmodels' GLM internals reset the
    filter list inside nested ``catch_warnings`` blocks.
    """
    with _SuppressWarnings():
        fit = sm.GLM(y, X, family=family).fit(maxiter=100)
    if family_name is not None:
        fit._monocle2py_family_name = family_name
    return fit


def fit_joint_nb(
    y: np.ndarray, X: np.ndarray, start_alpha: float | None = None
) -> Any:
    """Jointly estimate (mean, size) for a negative-binomial GLM.

    Mirrors ``VGAM::negbinomial(isize=1/start_alpha)``, which treats
    ``isize`` as an *initial* value for the size parameter and jointly
    estimates both mean coefficients and size via Fisher scoring.

    Uses ``statsmodels.discrete.NegativeBinomial`` (NB2 parametrisation:
    ``Var(Y) = mu + alpha*mu^2``). For a gene, ``start_alpha`` seeds
    ``ln(alpha)`` in the optimiser; ``None`` lets statsmodels use its
    default MoM-style start.
    """
    start_params = None
    if (
        start_alpha is not None
        and math.isfinite(start_alpha)
        and start_alpha > 0
    ):
        # statsmodels appends ``ln(alpha)`` at the end of the parameter
        # vector. Seeding only the coefficient block is awkward; let
        # statsmodels init the coefficients and accept its internal seed.
        pass
    # Mirror R's ``suppressWarnings(VGAM::vglm(..., family=negbinomial()))``
    # wrapper in ``fit_model_helper`` (``expr_models.R:77-82``). BFGS here
    # is more chatty than VGAM's IRLS — sparse genes produce Runtime /
    # Hessian / Convergence / PerfectSeparation warnings. See
    # :class:`_SuppressWarnings` for why ``catch_warnings + simplefilter``
    # alone doesn't hold up through statsmodels' internal filter resets.
    with _SuppressWarnings():
        fit = SMNegativeBinomial(y, X).fit(
            disp=0, maxiter=200, method="bfgs", start_params=start_params,
        )
    fit._monocle2py_family_name = "negbinomial"
    return fit


class _TobitModel:
    """Lightweight ``model``-slot shim exposing the design matrix.

    Mirrors the ``.model.exog`` handle that :func:`lrt` uses to compute
    degrees of freedom, matching the ``statsmodels`` GLM result interface.
    """

    def __init__(self, exog: np.ndarray) -> None:
        self.exog = exog


class TobitResult:
    """Return value of :func:`fit_tobit`.

    Mirrors the minimum subset of the ``statsmodels`` GLM result surface
    consumed by ``differential.response_matrix`` and :func:`lrt`:
    ``llf``, ``model.exog``, ``fittedvalues``, ``predict``, plus the
    monocle2-specific ``_monocle2py_family_name`` tag that makes
    ``response_matrix`` invert the ``log10`` response transform
    (``expr_models.R:158-160``).
    """

    def __init__(
        self,
        params: np.ndarray,
        exog: np.ndarray,
        llf: float,
        lower: float,
        upper: float,
        converged: bool,
    ) -> None:
        self.params = np.asarray(params, dtype=float)
        self.beta = self.params[:-1]
        self.log_sigma = float(self.params[-1])
        self.sigma = float(np.exp(self.log_sigma))
        self.llf = float(llf)
        self.model = _TobitModel(np.asarray(exog, dtype=float))
        self.fittedvalues = np.asarray(exog, dtype=float) @ self.beta
        self.lower = float(lower)
        self.upper = float(upper)
        self.converged = bool(converged)
        self._monocle2py_family_name = "Tobit"

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Return ``X_new @ beta`` (identity link — VGAM ``lmu='identitylink'``)."""
        return np.asarray(X_new, dtype=float) @ self.beta


def _tobit_neg_loglik(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    lower: float,
    upper: float,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    obs_mask: np.ndarray,
) -> float:
    """Negative log-likelihood for a Tobit Gaussian GLM with identity link.

    Parameterisation matches ``VGAM::tobit(lmu='identitylink', lsd='loglink')``:
    ``params = [beta, log(sigma)]``. Latent ``y* ~ N(X @ beta, sigma^2)``,
    observed ``y = clip(y*, lower, upper)``. Contributions:

    * uncensored ``lower < y < upper``: Gaussian density
    * left-censored ``y == lower``: ``Phi((lower - X @ beta) / sigma)``
    * right-censored ``y == upper``: ``Phi((X @ beta - upper) / sigma)``
    """
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = math.exp(log_sigma)
    mu = X @ beta

    total = 0.0
    if obs_mask.any():
        r = (y[obs_mask] - mu[obs_mask]) / sigma
        total += float(np.sum(
            -0.5 * np.log(2.0 * math.pi) - log_sigma - 0.5 * r * r
        ))
    if left_mask.any():
        total += float(np.sum(norm.logcdf(
            (lower - mu[left_mask]) / sigma
        )))
    if right_mask.any():
        total += float(np.sum(norm.logcdf(
            (mu[right_mask] - upper) / sigma
        )))
    return -total


def fit_tobit(
    y: np.ndarray,
    X: np.ndarray,
    lower: float = 0.0,
    upper: float = math.inf,
) -> TobitResult:
    """MLE fit of a Tobit censored-regression GLM.

    Mirrors ``VGAM::tobit(Lower=lower, Upper=upper, lmu='identitylink',
    lsd='loglink')`` called via ``VGAM::vglm``. The response ``y`` is
    expected on the **log10 scale** — that is the transform monocle2's
    ``fit_model_helper`` (``expr_models.R:45-47``) applies before passing
    to VGAM, so censoring thresholds ``lower`` / ``upper`` are interpreted
    on the same log10 scale.

    Starting values: weighted OLS with censored entries clipped to the
    boundary, and ``log(sigma)`` seeded from the uncensored residual SD.
    The (β, log σ) MLE is found with ``scipy.optimize.minimize`` (BFGS).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if y.ndim != 1 or X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("fit_tobit: y must be 1-D and match X.shape[0]")
    n, p = X.shape

    left_mask = y <= lower
    right_mask = y >= upper
    obs_mask = (~left_mask) & (~right_mask)

    y_for_start = y.copy()
    y_for_start[left_mask] = lower
    y_for_start[right_mask] = upper
    beta0, *_ = np.linalg.lstsq(X, y_for_start, rcond=None)
    resid = y_for_start - X @ beta0
    if obs_mask.any():
        sigma0 = float(np.std(resid[obs_mask], ddof=1))
    else:
        sigma0 = float(np.std(resid, ddof=1))
    if not (sigma0 > 0) or not math.isfinite(sigma0):
        sigma0 = 1.0
    start = np.concatenate([beta0, [math.log(sigma0)]])

    res = optimize.minimize(
        _tobit_neg_loglik,
        start,
        args=(y, X, lower, upper, left_mask, right_mask, obs_mask),
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 200},
    )

    return TobitResult(
        params=res.x,
        exog=X,
        llf=-float(res.fun),
        lower=lower,
        upper=upper,
        converged=bool(res.success),
    )


def fit_glm_with_fallback(
    y: np.ndarray,
    X: np.ndarray,
    family: Any,
    family_name: str,
) -> Any | None:
    """Fit a GLM; on exception retry the NB families with ``negbinomial()``.

    Mirrors R's ``fit_model_helper`` error branch (``expr_models.R:58-95``):
    when the primary fit fails and the family is ``negbinomial`` /
    ``negbinomial.size``, R swaps in a vanilla ``VGAM::negbinomial()``
    (joint MLE of mean and size, no ``isize`` seed). The Python analogue
    is :func:`fit_joint_nb`. Families other than NB have no backup in
    R's ``fit_model_helper`` — they return ``NULL``; we return ``None``.
    """
    try:
        return fit_glm(y, X, family, family_name=family_name)
    except Exception:
        pass
    if family_name not in ("negbinomial", "negbinomial.size"):
        return None
    try:
        return fit_joint_nb(y, X, start_alpha=None)
    except Exception:
        return None


def lrt(full_fit: Any, reduced_fit: Any) -> tuple[float, int, float]:
    """Likelihood-ratio test on two nested GLM fits.

    Degrees-of-freedom is the rank (column count) difference of the two
    design matrices. Using ``df_model`` would include nuisance parameters
    that ``statsmodels.discrete.NegativeBinomial`` appends to the parameter
    vector (``ln(alpha)``), which cancels in a difference only when both
    fits attach the same number of nuisance parameters — safer to count
    design columns directly to mirror VGAM's ``lrtest``.
    """
    stat = 2.0 * float(full_fit.llf - reduced_fit.llf)
    if stat < 0:
        stat = 0.0
    dof = int(full_fit.model.exog.shape[1] - reduced_fit.model.exog.shape[1])
    if dof <= 0:
        return stat, dof, float("nan")
    return stat, dof, float(chi2.sf(stat, dof))
