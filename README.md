# monocle2-python

[![PyPI](https://img.shields.io/pypi/v/monocle2-python)](https://pypi.org/project/monocle2-python/)

Python port of the R [**monocle2**](https://github.com/cole-trapnell-lab/monocle-release) single-cell trajectory toolkit.

## Installation

```bash
pip install monocle2-python                # from PyPI
```

For local development:

```bash
git clone https://github.com/Bio-Babel/Monocle2-python.git
cd Monocle2-python
pip install -e ".[dev]"
```


## Quickstart

```python
import monocle2py as m2

cds = m2.load_hsmm_fpkm()
m2.estimate_size_factors(cds)
m2.estimate_dispersions(cds)
m2.reduce_dimension(cds, reduction_method="DDRTree")
m2.order_cells(cds)
m2.plot_cell_trajectory(cds, color_by="State")
```

## Tutorials

Four runnable notebooks that reproduce the R monocle2 tutorials live under [`tutorials/`](tutorials/):

| Notebook | Dataset |
|---|---|
| `V2_1_HSMM.ipynb` | HSMM myoblast differentiation |
| `V2_2_lung_BEAM.ipynb` | Lung epithelium BEAM walkthrough |
| `V2_3_Olsson_corrected.ipynb` | Olsson myeloid progenitors (WT + KOs) |
| `V2_4_Paul.ipynb` | Paul hematopoiesis |

## License

Artistic-2.0, matching the upstream R monocle2.
