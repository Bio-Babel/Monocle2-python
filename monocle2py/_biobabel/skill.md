---
name: use-monocle2py
description: How to run DDRTree pseudotime trajectories and BEAM branch-dependent expression tests on AnnData with monocle2py — a 1:1 Python port of R Monocle2.
---

# monocle2py

Python port of [cole-trapnell-lab/monocle-release](https://github.com/cole-trapnell-lab/monocle-release) — tracks R Monocle 2.9.0 @ commit `7df1050`. The R `CellDataSet` S4 is replaced by `anndata.AnnData` (cells x genes); every other API choice mirrors R as faithfully as possible. All persistent Monocle2 state lives under `adata.uns["monocle2"]` and a small fixed set of obs / var / obsm columns.

## When to reach for monocle2py (vs monocle3)

- **monocle2py** — DDRTree-based principal-graph trajectories with **explicit numeric `State` segments**, branch-point indices, and the **BEAM** family of branch-dependent expression tests. The right tool when you need the classical Monocle2 workflow: `~sm.ns(Pseudotime, df=3) * Branch` LRTs, `plot_genes_branched_heatmap`, FPKM-via-Census transcript reconstruction.
- **monocle3** — partition-aware principal graphs, Leiden/Louvain clustering, Moran's-I `graph_test`. Reach for monocle3 when the dataset has multiple disconnected lineages or you want UMAP+principal-graph rather than DDRTree.

Do **not** reach for monocle2py if you need RNA velocity (use `scvelo`) or if your trajectory has multiple disconnected components — DDRTree forces a single connected MST.

## Mental model — the canonical 8-step pipeline

Monocle2 is a **state machine** over an `AnnData`. Each step writes specific slots that downstream steps read:

```
1. new_cell_dataset           → uns.monocle2 (family + lower_detection_limit), obs.Size_Factor (NaN init)
2. detect_genes               → var.num_cells_expressed, obs.num_genes_expressed
3. estimate_size_factors      → obs.Size_Factor (filled)
4. estimate_dispersions       → uns.monocle2.disp_fit_info["blind"]   (NB families only)
5. set_ordering_filter        → var.use_for_ordering (boolean mask)
6. reduce_dimension(DDRTree)  → obsm.X_dr, uns.monocle2.ddrtree{K, W, mst_edges, mst_weights, closest_vertex}, uns.monocle2.dim_reduce_type
7. order_cells                → obs.Pseudotime, obs.State (Categorical), obs.Parent, uns.monocle2.aux_ordering.DDRTree{root_cell, root_vertex, branch_points, pr_graph_cell_proj_*}
8. differential_gene_test / beam → DataFrame (NOT mutating)
```

Steps 1-7 mutate the AnnData; the DE/BEAM tests **return** a DataFrame and never write back. Plotting helpers consume the populated slots and return `ggplot2_py.GGPlot` (or a `pheatmap` object for heatmaps).

## Invariants — do not get these wrong

- **`adata.uns["monocle2"]` is non-optional.** `new_cell_dataset` always creates it. If you subset an AnnData and want to keep DDRTree state, propagate `uns["monocle2"]` manually (`sub.uns["monocle2"] = dict(adata.uns["monocle2"])`).
- **`obs["State"]` is a `pd.Categorical` of stringified 1-based integers** (`"1"`, `"2"`, ...). Selecting by `State == 1` (int) silently miscompares against the categorical; cast first: `adata.obs["State"].astype(int) == 1`. BEAM accepts ints (`branch_states=[2, 3]`) and casts internally.
- **`obs["Pseudotime"] == 0` marks the trajectory root cell** in DDRTree mode. Multiple cells can share `Pseudotime==0`; the first one is treated as the canonical root by branch-path discovery.
- **`set_ordering_filter` is required before `reduce_dimension(method='DDRTree')` for variable-gene selection** to take effect. If `var["use_for_ordering"]` is absent (or all-False), `reduce_dimension` runs on **every gene**, which is almost never what you want.
- **NB families need size factors.** Calling `reduce_dimension` / `differential_gene_test` on a `negbinomial`/`negbinomial.size` family without `estimate_size_factors` raises. The `Tobit` and Gaussian (`gaussian_family()`, vfamily `uninormal`) families do **not** need size factors.
- **Re-running `order_cells(root_state=k)` does NOT renumber `State`.** Only the first call (`root_state=None`) populates `State`; subsequent root-state pivots only update `Pseudotime`. This mirrors R behavior and is load-bearing for BEAM (which reads the original numbering).
- **BEAM requires `Branch` in the full formula.** The default `full_model_formula_str="~sm.ns(Pseudotime, df=3)*Branch"` triggers the branch CDS build internally; remove `Branch` from the formula and BEAM degenerates to a vanilla pseudotime DE test.

## Common pitfalls

- **Mixing FPKM and counts.** Monocle2 NB families need integer-like counts. If your input is FPKM/TPM, route through `relative2abs(method='num_genes')` first and build a fresh `new_cell_dataset` on the census output with the default `negbinomial_size` family.
- **State labels are 1-based strings.** Indexing branches at `branch_states=[0, 1]` against a State range of `{"1","2","3"}` raises `No cells in State == 0`.
- **`differential_gene_test` returns a DataFrame indexed by `gene_id`.** It does not mutate the AnnData. If you want to keep results, store them yourself: `adata.uns["beam_results"] = beam(...)`.
- **`branch_point` is 1-based and sorted by pseudotime.** `branch_point=1` is always the **earliest** branch in pseudotime, not the lowest centroid index.

## Quick reference

```python
import monocle2py as m2

# Build the dataset — picks negbinomial.size + lower_detection_limit=0.1 by default
adata = m2.new_cell_dataset(counts, pheno_data=obs, feature_data=var)

# Preprocessing
m2.detect_genes(adata, min_expr=0.1)
m2.estimate_size_factors(adata)
m2.estimate_dispersions(adata)

# Ordering-gene selection (required before DDRTree to restrict the variable-gene space)
ordering_genes = adata.var_names[adata.var["num_cells_expressed"] >= 10].tolist()
m2.set_ordering_filter(adata, ordering_genes)

# Trajectory
m2.reduce_dimension(adata, max_components=2, reduction_method="DDRTree")
m2.order_cells(adata)                          # picks root from MST diameter
m2.order_cells(adata, root_state=2)            # re-anchor root to State "2"

# DE along pseudotime
de = m2.differential_gene_test(
    adata, full_model_formula_str="~sm.ns(Pseudotime, df=3)", reduced_model_formula_str="~1",
)

# BEAM — branch-dependent expression at branch point 1
beam_res = m2.beam(adata, branch_point=1)

# Plots
m2.plot_cell_trajectory(adata, color_by="State")
m2.plot_genes_branched_heatmap(adata[:, top_genes], branch_point=1, num_clusters=4)
```

For each step's full signature: `biobabel.describe_symbol(symbol_id="monocle2py.<name>")`.

For the full canonical workflow as a `WorkflowContract`: `biobabel.describe_workflow(workflow_id="monocle2py.basic_trajectory")`.
