# monocle2py

Python port of the R **monocle 2** package
([Trapnell lab](https://github.com/cole-trapnell-lab/monocle-release))
for single-cell RNA-seq trajectory inference and branch analysis.

## Highlights

- **AnnData-native** — `AnnData` is the data container; no bespoke
  `CellDataSet` wrapper.
- **Faithful Census + DDRTree pipeline** — `relative2abs`,
  `reduce_dimension(method="DDRTree")`, `order_cells`, `differential_gene_test`,
  `BEAM`.
- **External DDRTree backend** — uses the standalone
  [`ddrtree`](https://pypi.org/project/ddrtree/) Python package for the
  numerical core.
- **Bio-Babel plotting** — all visual outputs are built on
  [`ggplot2_py`](https://github.com/Bio-Babel/ggplot2_py),
  [`pheatmap_py`](https://github.com/Bio-Babel/pheatmap_py),
  `grid_py`, `gtable_py`, `scales`, and `ggrepel_py` — no Matplotlib axes,
  no `seaborn`.

## Installation

```bash
pip install -e .
```

Bundled tutorial data (HSMM, lung, Olsson, Paul) is fetched on first use from
the package's Zenodo sandbox record (record id `491746`) and cached locally.

## Quick start

```python
from monocle2py import (
    load_hsmm_fpkm, new_cell_dataset, estimate_size_factors,
    detect_genes, estimate_dispersions, reduce_dimension, order_cells,
    plot_cell_trajectory,
)

adata = load_hsmm_fpkm()
cds = new_cell_dataset(adata.X, pheno_data=adata.obs, feature_data=adata.var)
estimate_size_factors(cds)
detect_genes(cds, min_expr=0.1)
estimate_dispersions(cds)
reduce_dimension(cds, max_components=2, reduction_method="DDRTree")
order_cells(cds)
plot_cell_trajectory(cds, color_by="Pseudotime")
```

## Tutorials

Four end-to-end notebooks reproducing the original R monocle2 vignettes:

- [V2_1 · HSMM myoblast differentiation](tutorials/V2_1_HSMM.ipynb)
- [V2_2 · Lung BEAM branched analysis](tutorials/V2_2_lung_BEAM.ipynb)
- [V2_3 · Olsson myeloid lineage with fig 1B genes](tutorials/V2_3_Olsson.ipynb)
- [V2_4 · Paul haematopoiesis multi-branch downstream](tutorials/V2_4_Paul.ipynb)

## API

See the [API reference](api.md) for the full public surface.
