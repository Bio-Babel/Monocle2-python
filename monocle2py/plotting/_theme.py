"""Shared ggplot2_py theme used by monocle2py plot helpers."""

from __future__ import annotations

from ggplot2_py import (
    element_blank,
    element_line,
    element_rect,
    theme,
)

__all__ = ["monocle_theme_opts"]


def monocle_theme_opts() -> theme:
    """Port of R ``monocle:::monocle_theme_opts``.

    Strips panel borders and grid lines, draws thin black axis lines, and
    sets a white panel background — matching the look used across all
    monocle plot functions.
    """
    return (
        theme(**{"strip.background": element_rect(colour="white", fill="white")})
        + theme(**{"panel.border": element_blank()})
        + theme(**{"axis.line.x": element_line(linewidth=0.25, color="black")})
        + theme(**{"axis.line.y": element_line(linewidth=0.25, color="black")})
        + theme(**{
            "panel.grid.minor.x": element_blank(),
            "panel.grid.minor.y": element_blank(),
        })
        + theme(**{
            "panel.grid.major.x": element_blank(),
            "panel.grid.major.y": element_blank(),
        })
        + theme(**{"panel.background": element_rect(fill="white")})
        + theme(**{"legend.key": element_blank()})
    )
