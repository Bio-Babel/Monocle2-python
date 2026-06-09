"""Recipe: canonical Monocle2 DDRTree pseudotime trajectory.

Mirrors workflows/basic_trajectory.yaml step-by-step. Loads the HSMM
tutorial bundle (FPKM → Census → NB) and walks the full pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import monocle2py as m2


def main(out_path: Path | None = None) -> "ad.AnnData":  # type: ignore[name-defined]
    """Run the full DDRTree trajectory pipeline on HSMM and return the AnnData.

    Returns
    -------
    AnnData
        With obs.Pseudotime, obs.State, obs.Parent, obs.Size_Factor,
        obs.num_genes_expressed; var.num_cells_expressed,
        var.use_for_ordering; obsm.X_dr; uns.monocle2 (with ddrtree,
        aux_ordering, disp_fit_info populated).
    """
    fpkm = m2.load_hsmm_fpkm()

    # FPKM → Census transcript counts → NB CellDataSet.
    fpkm_cds = m2.new_cell_dataset(
        fpkm.X,
        pheno_data=fpkm.obs.copy(),
        feature_data=fpkm.var.copy(),
        lower_detection_limit=0.1,
        expression_family=m2.tobit(lower=0.1),
    )
    counts = m2.relative2abs(fpkm_cds, method="num_genes")
    adata = m2.new_cell_dataset(
        counts,
        pheno_data=fpkm.obs.copy(),
        feature_data=fpkm.var.copy(),
        lower_detection_limit=0.5,
    )

    # Per-cell + per-gene QC counters.
    m2.estimate_size_factors(adata)
    m2.detect_genes(adata, min_expr=0.1)
    m2.estimate_dispersions(adata)

    # Pick ordering genes: any gene detected in >= 10 cells (toy heuristic;
    # real workflows use a cluster-DEG test — see V2_1_HSMM tutorial).
    expressed = adata.var_names[adata.var["num_cells_expressed"] >= 10].tolist()
    m2.set_ordering_filter(adata, expressed)

    # DDRTree projection + ordering.
    m2.reduce_dimension(
        adata,
        max_components=2,
        reduction_method="DDRTree",
        auto_param_selection=True,
        verbose=False,
    )
    m2.order_cells(adata)

    # Pseudotime DE.
    pt_de = m2.differential_gene_test(
        adata,
        full_model_formula_str="~sm.ns(Pseudotime, df=3)",
        reduced_model_formula_str="~1",
    )
    adata.uns["pseudotime_de"] = pt_de  # type: ignore[assignment]

    if out_path is not None:
        # gen_smooth_curves stores callables in uns; drop before write.
        adata.write_h5ad(out_path)
    return adata


if __name__ == "__main__":
    main()
