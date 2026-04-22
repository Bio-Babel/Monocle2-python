"""Remote data asset registry.

All tutorial inputs are hosted on Zenodo record 19699224
(https://zenodo.org/records/19699224) and fetched lazily via
``_download.resolve_data_path``.
"""

CACHE_DIR_NAME = "monocle2_py"

_ZENODO_BASE = "https://zenodo.org/records/19699224/files"


REGISTRY: dict[str, dict[str, str]] = {
    "HSMM_fpkm.h5ad": {
        "url": f"{_ZENODO_BASE}/HSMM_fpkm.h5ad?download=1",
        "sha256": "12f1c3852c9952d0c188a8e9bf31ee7e88bb871a6cd143b5f78c514612160e6e",
    },
    "fig1b.tsv": {
        "url": f"{_ZENODO_BASE}/fig1b.tsv?download=1",
        "sha256": "35ac7fee4d91e0af1c403ce105e3d4581f757419003b63f62b14c5d50c7b6567",
    },
    "lung_cds.h5ad": {
        "url": f"{_ZENODO_BASE}/lung_cds.h5ad?download=1",
        "sha256": "bcfd6d21711bdfb5a0ca1fc92052f3f8c15f09e813cd432d17576dad4932c55f",
    },
    "olsson_tpm.h5ad": {
        "url": f"{_ZENODO_BASE}/olsson_tpm.h5ad?download=1",
        "sha256": "b422d1d85013f82032578991f0909d6e4634ed315cf7f46c7960f719ee48e9d1",
    },
    "paul_cds.h5ad": {
        "url": f"{_ZENODO_BASE}/paul_cds.h5ad?download=1",
        "sha256": "ba51425c412e84d35e59544b15e08716b79caf47223396d68802eedacb3e3bb5",
    },
    "paul_gene_set.json": {
        "url": f"{_ZENODO_BASE}/paul_gene_set.json?download=1",
        "sha256": "bd1642da89f51c6544ceef7a06052c6dca9973c74a46157e4b0e3562c3cb9d4a",
    },
}
