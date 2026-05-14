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

### DDRTree backend (CPU / GPU)

If you want to use GPU, please install the torch build that fits your machine first.

```python
# CPU, reference path (default — matches R numerically)
m2.reduce_dimension(cds, reduction_method="DDRTree")

# GPU via PyTorch (Borůvka MST stays on-device)
m2.reduce_dimension(cds, reduction_method="DDRTree",
                    backend="torch", device="cuda")
```

GPU is never selected automatically — passing `backend="torch"` and
`device="cuda"` is required. K-means initialisation still runs on CPU
(small, constant overhead).

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
