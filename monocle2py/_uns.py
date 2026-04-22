"""Conventions for Monocle2 state stored in ``adata.uns['monocle2']``.

All algorithmic modules read and write state through the helpers defined here
so the layout stays consistent and documented in one place. Mirrors the slot
mapping in ``port_reports/monocle2/05_design.md``.
"""

from __future__ import annotations

from typing import Any

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


def get_expression_family(adata: AnnData) -> Any:
    """Return the family object stored on the AnnData."""
    state = get_state(adata)
    if "expression_family_obj" in state:
        return state["expression_family_obj"]
    from .families import family_from_name
    return family_from_name(state["expression_family"])


def set_expression_family(adata: AnnData, family: Any) -> None:
    """Record the expression family on the AnnData state."""
    state = ensure_state(adata)
    state["expression_family"] = family.vfamily
    state["expression_family_obj"] = family


def get_lower_detection_limit(adata: AnnData) -> float:
    return float(get_state(adata)["lower_detection_limit"])


def set_lower_detection_limit(adata: AnnData, value: float) -> None:
    ensure_state(adata)["lower_detection_limit"] = float(value)


def get_disp_fit_info(
    adata: AnnData, name: str = "blind"
) -> dict[str, Any] | None:
    """Return the dispersion fit result registered under *name*, or ``None``."""
    state = get_state(adata)
    fits = state.get("disp_fit_info")
    if fits is None:
        return None
    return fits.get(name)


def set_disp_fit_info(
    adata: AnnData, info: dict[str, Any], name: str = "blind"
) -> None:
    state = ensure_state(adata)
    fits = state.setdefault("disp_fit_info", {})
    fits[name] = info
