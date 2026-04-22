"""Per-gene pseudotime plots: ``plot_genes_in_pseudotime`` and ``plot_genes_branched_pseudotime``.

Both functions overlay raw cell expression with the smoothed trend fit by
:func:`~monocle2py.gen_smooth_curves`. ``plot_genes_branched_pseudotime`` is
the branch-aware sibling: it duplicates progenitors via
:func:`~monocle2py.build_branch_cell_dataset` so each branch carries its own
trend curve.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from ggplot2_py import (
    GGPlot,
    aes,
    expand_limits,
    facet_wrap,
    geom_line,
    geom_point,
    ggplot,
    position_jitter,
    scale_y_log10,
    xlab,
    ylab,
)

from .._internal._formula import formula_terms
from .._uns import get_expression_family, get_lower_detection_limit
from ..beam import build_branch_cell_dataset
from ..differential import gen_smooth_curves
from ._helpers import expression_long_df, feature_label_column
from ._theme import monocle_theme_opts

__all__ = ["plot_genes_in_pseudotime", "plot_genes_branched_pseudotime"]


def _attach_feature_labels(
    long_df: pd.DataFrame, var_df: pd.DataFrame, label_by_short_name: bool,
) -> pd.DataFrame:
    labels = feature_label_column(var_df, label_by_short_name)
    label_map = labels.to_dict()
    long_df = long_df.copy()
    long_df["feature_label"] = long_df["f_id"].map(label_map).astype(str)
    return long_df


def _apply_panel_order(
    long_df: pd.DataFrame, panel_order: Sequence[str] | None,
) -> pd.DataFrame:
    if panel_order is None:
        return long_df
    long_df = long_df.copy()
    long_df["feature_label"] = pd.Categorical(
        long_df["feature_label"], categories=list(panel_order), ordered=True,
    )
    return long_df


def plot_genes_in_pseudotime(
    cds_subset: AnnData,
    min_expr: float | None = None,
    cell_size: float = 0.75,
    nrow: int | None = None,
    ncol: int = 1,
    panel_order: Sequence[str] | None = None,
    color_by: str | None = "State",
    trend_formula: str = "~sm.ns(Pseudotime, df=3)",
    label_by_short_name: bool = True,
    relative_expr: bool = True,
    vertical_jitter: float | None = None,
    horizontal_jitter: float | None = None,
) -> GGPlot:
    """Port of R's ``plot_genes_in_pseudotime``.

    Faceted scatter of expression against pseudotime, one panel per gene, with
    the smoothed trend curve overlaid. Operates on a gene-subset AnnData
    (``cds_subset``) just like the R function.
    """
    if "Pseudotime" not in cds_subset.obs.columns:
        raise RuntimeError(
            "Pseudotime missing — call order_cells before plotting."
        )

    long_df, integer_expression = expression_long_df(
        cds_subset, relative_expr=relative_expr,
    )
    if min_expr is None:
        min_expr = get_lower_detection_limit(cds_subset)

    long_df = _attach_feature_labels(
        long_df, cds_subset.var, label_by_short_name,
    )

    obs = cds_subset.obs[[
        c for c in cds_subset.obs.columns if c not in long_df.columns
    ]].copy()
    obs["Cell"] = cds_subset.obs_names.astype(str)
    long_df = long_df.merge(obs, on="Cell", how="left")

    new_data = pd.DataFrame({
        "Pseudotime": cds_subset.obs["Pseudotime"].to_numpy(dtype=float),
    })
    if "Branch" in formula_terms(trend_formula):
        new_data["Branch"] = cds_subset.obs.get(
            "Branch", cds_subset.obs.get("State"),
        ).astype("category").to_numpy()
    expectation = gen_smooth_curves(
        cds_subset, new_data=new_data, trend_formula=trend_formula,
        relative_expr=True,
    )
    expectation.columns = cds_subset.obs_names.astype(str)
    exp_long = expectation.reset_index().melt(
        id_vars="index", var_name="Cell", value_name="expectation",
    ).rename(columns={"index": "f_id"})
    exp_long["f_id"] = exp_long["f_id"].astype(str)
    long_df = long_df.merge(exp_long, on=["f_id", "Cell"], how="left")

    long_df.loc[long_df["expression"] < min_expr, "expression"] = min_expr
    long_df.loc[long_df["expectation"] < min_expr, "expectation"] = min_expr

    long_df = _apply_panel_order(long_df, panel_order)

    q = ggplot(long_df, aes(x="Pseudotime", y="expression"))
    if color_by is not None:
        q = q + geom_point(
            aes(color=color_by), size=cell_size,
            position=position_jitter(width=horizontal_jitter, height=vertical_jitter),
        )
    else:
        q = q + geom_point(
            size=cell_size,
            position=position_jitter(width=horizontal_jitter, height=vertical_jitter),
        )
    q = q + geom_line(
        aes(x="Pseudotime", y="expectation"), data=long_df,
    )
    q = q + scale_y_log10() + facet_wrap(
        "feature_label", nrow=nrow, ncol=ncol, scales="free_y",
    )
    if min_expr < 1:
        q = q + expand_limits(y=[min_expr, 1])
    if relative_expr:
        q = q + ylab("Relative Expression")
    else:
        q = q + ylab("Absolute Expression")
    q = q + xlab("Pseudo-time") + monocle_theme_opts()
    return q


def plot_genes_branched_pseudotime(
    cds: AnnData,
    branch_states: Iterable[int] | None = None,
    branch_point: int = 1,
    branch_labels: Sequence[str] | None = None,
    method: str = "fitting",
    min_expr: float | None = None,
    cell_size: float = 0.75,
    nrow: int | None = None,
    ncol: int = 1,
    panel_order: Sequence[str] | None = None,
    color_by: str | None = "State",
    expression_curve_linetype_by: str = "Branch",
    trend_formula: str = "~sm.ns(Pseudotime, df=3) * Branch",
    reduced_model_formula_str: str | None = None,
    label_by_short_name: bool = True,
    relative_expr: bool = True,
) -> GGPlot:
    """Port of R's ``plot_genes_branched_pseudotime``.

    Builds a branch-aware AnnData via :func:`build_branch_cell_dataset` (when
    ``Branch`` appears in ``trend_formula``), fits the per-branch trend, and
    overlays expression points with branch-typed line curves.
    """
    if "Branch" in formula_terms(trend_formula):
        sub = build_branch_cell_dataset(
            cds, branch_states=branch_states, branch_point=branch_point,
            branch_labels=branch_labels, progenitor_method="duplicate",
        )
    else:
        sub = cds.copy()
        if "Branch" not in sub.obs.columns:
            sub.obs["Branch"] = sub.obs["State"].astype(str).astype("category")

    long_df, _ = expression_long_df(sub, relative_expr=relative_expr)
    if min_expr is None:
        min_expr = get_lower_detection_limit(sub)

    long_df = _attach_feature_labels(long_df, sub.var, label_by_short_name)
    obs = sub.obs[[c for c in sub.obs.columns if c not in long_df.columns]].copy()
    obs["Cell"] = sub.obs_names.astype(str)
    long_df = long_df.merge(obs, on="Cell", how="left")
    long_df["Branch"] = long_df["Branch"].astype(str).astype("category")

    new_data = pd.DataFrame({
        "Pseudotime": sub.obs["Pseudotime"].to_numpy(dtype=float),
        "Branch": sub.obs["Branch"].astype("category"),
    })
    full_expectation = gen_smooth_curves(
        sub, new_data=new_data, trend_formula=trend_formula,
        relative_expr=True,
    )
    full_expectation.columns = sub.obs_names.astype(str)
    full_long = full_expectation.reset_index().melt(
        id_vars="index", var_name="Cell", value_name="full_model_expectation",
    ).rename(columns={"index": "f_id"})
    full_long["f_id"] = full_long["f_id"].astype(str)
    long_df = long_df.merge(full_long, on=["f_id", "Cell"], how="left")

    if reduced_model_formula_str is not None:
        reduced_expectation = gen_smooth_curves(
            sub, new_data=new_data, trend_formula=reduced_model_formula_str,
            relative_expr=True,
        )
        reduced_expectation.columns = sub.obs_names.astype(str)
        red_long = reduced_expectation.reset_index().melt(
            id_vars="index", var_name="Cell", value_name="reduced_model_expectation",
        ).rename(columns={"index": "f_id"})
        red_long["f_id"] = red_long["f_id"].astype(str)
        long_df = long_df.merge(red_long, on=["f_id", "Cell"], how="left")

    if method == "loess":
        long_df["expression"] = long_df["expression"] + min_expr

    expr = long_df["expression"].to_numpy(float)
    expr[np.isnan(expr)] = min_expr
    expr[expr < min_expr] = min_expr
    long_df["expression"] = expr
    full = long_df["full_model_expectation"].to_numpy(float)
    full[np.isnan(full)] = min_expr
    full[full < min_expr] = min_expr
    long_df["full_model_expectation"] = full
    if reduced_model_formula_str is not None:
        red = long_df["reduced_model_expectation"].to_numpy(float)
        red[np.isnan(red)] = min_expr
        red[red < min_expr] = min_expr
        long_df["reduced_model_expectation"] = red

    long_df = _apply_panel_order(long_df, panel_order)

    q = ggplot(long_df, aes(x="Pseudotime", y="expression"))
    if color_by is not None:
        q = q + geom_point(aes(color=color_by), size=cell_size)
    if method == "fitting":
        q = q + geom_line(
            aes(x="Pseudotime", y="full_model_expectation",
                linetype=expression_curve_linetype_by),
            data=long_df,
        )
    if reduced_model_formula_str is not None:
        q = q + geom_line(
            aes(x="Pseudotime", y="reduced_model_expectation"),
            data=long_df, color="black", linetype=2,
        )
    q = q + scale_y_log10() + facet_wrap(
        "feature_label", nrow=nrow, ncol=ncol, scales="free_y",
    )
    q = q + ylab("Expression") + xlab("Pseudotime (stretched)")
    q = q + monocle_theme_opts() + expand_limits(y=min_expr)
    return q
