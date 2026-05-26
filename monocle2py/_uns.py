"""Conventions for Monocle2 state stored in ``adata.uns['monocle2']``.

All algorithmic modules read and write state through the helpers defined here
so the layout stays consistent and documented in one place. Mirrors the slot
mapping in ``port_reports/monocle2/05_design.md``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from anndata import AnnData

__all__ = [
    "UNS_KEY",
    "ensure_state",
    "get_state",
    "get_expression_family",
    "set_expression_family",
    "get_lower_detection_limit",
    "set_lower_detection_limit",
    "get_disp_fit_info",
    "set_disp_fit_info",
    "SIZE_FACTOR_COL",
]

UNS_KEY = "monocle2"
SIZE_FACTOR_COL = "Size_Factor"


def ensure_state(adata: AnnData) -> dict[str, Any]:
    """Return ``adata.uns['monocle2']``, creating it if missing.

    Parameters
    ----------
    adata : anndata.AnnData

    Returns
    -------
    dict[str, Any]
        The mutable state dict.
    """
    if UNS_KEY not in adata.uns:
        adata.uns[UNS_KEY] = {}
    state = adata.uns[UNS_KEY]
    if not isinstance(state, dict):
        raise TypeError(
            f"adata.uns['{UNS_KEY}'] must be a dict, got {type(state).__name__}"
        )
    return state


def get_state(adata: AnnData) -> dict[str, Any]:
    """Read the Monocle2 state dict (raises if absent)."""
    try:
        state = adata.uns[UNS_KEY]
    except KeyError as exc:
        raise KeyError(
            f"AnnData has no '{UNS_KEY}' state. Call new_cell_dataset first."
        ) from exc
    return state


def _rebuild_family(name: str, params: dict[str, Any] | None) -> Any:
    """Rebuild a family dataclass from a ``vfamily`` string + numeric params.

    Used both for in-memory reads (so we never hold a stale dataclass
    reference) and for h5ad round-trips where anndata cannot serialise
    arbitrary Python objects inside ``uns``.
    """
    from .families import family_from_name, negbinomial_size, tobit
    if not params:
        return family_from_name(name)
    kwargs = {k: float(v) for k, v in dict(params).items()}
    if name == "negbinomial.size":
        return negbinomial_size(**kwargs)
    if name in ("Tobit", "tobit"):
        return tobit(**kwargs)
    return family_from_name(name)


def get_expression_family(adata: AnnData) -> Any:
    """Return the family object stored on the AnnData.

    Reads ``expression_family`` (the ``vfamily`` string) and
    ``expression_family_params`` (a dict of numeric parameters) and
    calls the matching factory. Storing the dataclass itself under
    ``uns`` breaks ``write_h5ad`` (anndata has no serialiser for custom
    Python objects), so the string+params pair is the canonical
    representation that survives round-trips.
    """
    state = get_state(adata)
    name = state["expression_family"]
    params = state.get("expression_family_params")
    return _rebuild_family(name, dict(params) if params is not None else None)


def _serialisable_family_params(family: Any) -> dict[str, Any]:
    """Pull numeric parameters off a family dataclass for h5ad storage."""
    params: dict[str, Any] = {}
    for attr in ("size", "lower", "upper"):
        val = getattr(family, attr, None)
        if val is not None:
            params[attr] = float(val)
    return params


def set_expression_family(adata: AnnData, family: Any) -> None:
    """Record the expression family on the AnnData state.

    Stores the ``vfamily`` string and any numeric parameters so
    :func:`get_expression_family` can rebuild the family faithfully
    after an ``AnnData.write_h5ad`` / ``read_h5ad`` cycle.
    """
    state = ensure_state(adata)
    state["expression_family"] = family.vfamily
    params = _serialisable_family_params(family)
    if params:
        state["expression_family_params"] = params
    else:
        state.pop("expression_family_params", None)


def get_lower_detection_limit(adata: AnnData) -> float:
    return float(get_state(adata)["lower_detection_limit"])


def set_lower_detection_limit(adata: AnnData, value: float) -> None:
    ensure_state(adata)["lower_detection_limit"] = float(value)


def _make_disp_func(
    coefficients: dict[str, float],
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the dispersion-vs-mean closure from fitted coefficients.

    Kept as a factory rather than stored alongside ``coefficients`` because
    Python callables are not h5ad-serialisable; rebuilding on retrieval is
    cheap (closure captures two floats) and keeps ``adata.uns['monocle2']``
    round-trippable through ``adata.write_h5ad`` / ``read_h5ad``.
    """
    a = float(coefficients["asymptDisp"])
    b = float(coefficients["extraPois"])

    def disp_func(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return a + b / q

    return disp_func


def get_disp_fit_info(
    adata: AnnData, name: str = "blind"
) -> dict[str, Any] | None:
    """Return the dispersion fit result registered under *name*, or ``None``.

    The persisted dict only carries h5ad-safe payload (``disp_table`` and
    ``coefficients``). ``disp_func`` is reconstructed on the fly from
    ``coefficients`` so downstream callers (``differential_gene_test``,
    ``dispersion_table``, etc.) see a consistent dict whether the AnnData
    was just fitted or freshly loaded from disk.
    """
    state = get_state(adata)
    fits = state.get("disp_fit_info")
    if fits is None:
        return None
    persisted = fits.get(name)
    if persisted is None:
        return None
    info = dict(persisted)
    if "disp_func" not in info and "coefficients" in info:
        info["disp_func"] = _make_disp_func(info["coefficients"])
    return info


def set_disp_fit_info(
    adata: AnnData, info: dict[str, Any], name: str = "blind"
) -> None:
    """Persist a dispersion fit. Strips non-serialisable ``disp_func``
    entries — the canonical persisted shape is ``disp_table`` plus
    ``coefficients``; ``get_disp_fit_info`` rebuilds the closure on read.
    """
    state = ensure_state(adata)
    fits = state.setdefault("disp_fit_info", {})
    persisted = {k: v for k, v in info.items() if k != "disp_func"}
    fits[name] = persisted
