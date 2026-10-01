"""
Exploratory ligand-receptor inference between annotated cell types with
liana-py's CellChat method. P-values permute cells, not patients, so results are
hypotheses rather than replicate-level findings.
"""

import os
import scanpy as sc
import liana as li

from labels import real_cell_type_mask

RESULTS_DIR = "results"

# Plot-only short labels; computation and the CSV use full cell_type names.
PLOT_DISPLAY_LABELS = {
    "Stromal cell": "Stromal",
    "Myeloid/Monocyte": "Myeloid",
    "Epithelial cell": "Epithelial",
    "Plasma cell": "Plasma",
}


def run_cellchat_analysis(adata, groupby="cell_type"):
    """
    Run liana's CellChat-method implementation, inferring ligand-receptor
    interactions between every pair of annotated cell types.
    """
    real_cells = real_cell_type_mask(adata.obs[groupby])
    n_excluded = int((~real_cells).sum())
    if n_excluded > 0:
        print(f"Excluding {n_excluded} cells without a real cell type (Ambiguous/Unknown/doublet) from communication analysis.")
    adata = adata[real_cells].copy()
    if hasattr(adata.obs[groupby], "cat"):
        adata.obs[groupby] = adata.obs[groupby].cat.remove_unused_categories()

    print(f"Running CellChat-method ligand-receptor inference across {adata.obs[groupby].nunique()} cell types...")
    li.mt.cellchat(
        adata,
        groupby=groupby,
        expr_prop=0.1,       # ligand/receptor must be expressed in >=10% of cells in a group
        resource_name="consensus",
        verbose=True,
        use_raw=True,
    )

    return adata


def main():
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    input_file = "data/annotated_data.h5ad"
    print(f"Loading {input_file}...")
    adata = sc.read(input_file)

    if "cell_type" not in adata.obs:
        raise KeyError("'cell_type' not found. Run scripts/annotate.py first.")

    adata = run_cellchat_analysis(adata)

    result_key = "liana_res"
    if result_key not in adata.uns:
        raise KeyError(
            f"Expected '{result_key}' in adata.uns after li.mt.cellchat() -- "
            "the result key may differ by liana-py version; check "
            "adata.uns.keys() and adjust this script."
        )

    liana_results = adata.uns[result_key]
    liana_results.to_csv(os.path.join(RESULTS_DIR, "cell_communication_results.csv"), index=False)
    print(f"Saved {len(liana_results)} ligand-receptor interaction results to results/cell_communication_results.csv "
          f"(full, precise cell type names).")

    # Dot plot of the top interactions. The results table is temporarily
    # swapped for a copy with short display labels, then restored.
    try:
        top_n = 20
        display_results = liana_results.copy()
        display_results["source"] = display_results["source"].map(lambda x: PLOT_DISPLAY_LABELS.get(x, x))
        display_results["target"] = display_results["target"].map(lambda x: PLOT_DISPLAY_LABELS.get(x, x))
        adata.uns[result_key] = display_results

        dotplot = li.pl.dotplot(
            adata=adata,
            colour="lr_probs",
            size="cellchat_pvals",
            top_n=top_n,
            orderby="lr_probs",
            orderby_ascending=False,
            source_labels=None,
            target_labels=None,
        )
        dotplot.save(os.path.join(RESULTS_DIR, "figures", "cell_communication_dotplot.png"))
        print(
            f"Saved dot plot of top {top_n} interactions to results/figures/cell_communication_dotplot.png "
            f"(shortened display labels; full names are in the CSV  "
            f"in annotate.py; full names are in the CSV above)."
        )

        adata.uns[result_key] = liana_results  # restore full-precision version
    except Exception as e:
        print(f"Dot plot generation failed ({e}) -- the CSV results are still saved and usable directly.")

    print("Done.")


if __name__ == "__main__":
    main()
