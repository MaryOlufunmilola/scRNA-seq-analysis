"""
Export a cell-type signature matrix (mean linear-scale CP10K expression per cell
type, over the top DE marker genes per type) for bulk RNA-seq deconvolution.
Values are averaged after expm1 because deconvolution assumes linear mixing.
"""

import os
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

from labels import real_cell_type_mask

RESULTS_DIR = "results"
N_GENES_PER_TYPE = 50
MIN_LOG2FC = 1.0
MAX_PADJ = 0.05


def _real_cells_raw_adata(adata):
    """adata.raw as an AnnData with cell_type attached, restricted to real cell types."""
    if adata.raw is None:
        raise ValueError(
            "adata.raw is not set -- it should have been preserved by preprocess.py. "
            "Re-run the pipeline from preprocess.py."
        )
    if "cell_type" not in adata.obs:
        raise KeyError("'cell_type' not found. Run scripts/annotate.py first.")

    raw = adata.raw.to_adata()
    raw.obs["cell_type"] = adata.obs["cell_type"].astype(str).values

    keep = real_cell_type_mask(raw.obs["cell_type"])
    n_excluded = int((~keep).sum())
    if n_excluded > 0:
        excluded = sorted(set(raw.obs["cell_type"][~keep]))
        print(f"Excluding {n_excluded} cells not assigned a real cell type {excluded} from the reference.")
    raw = raw[keep].copy()
    raw.obs["cell_type"] = raw.obs["cell_type"].astype("category")
    return raw


def build_signature_matrix(adata):
    """
    Mean linear-scale (CP10K) expression per real cell type, over all genes
    in adata.raw. Returns a genes x cell types DataFrame.
    """
    raw = _real_cells_raw_adata(adata)
    cell_types = sorted(raw.obs["cell_type"].cat.categories)
    print(f"Computing mean expression profile for {len(cell_types)} cell types: {cell_types}")

    profiles = {}
    for ct in cell_types:
        mask = (raw.obs["cell_type"] == ct).values
        linear_expr = np.expm1(raw[mask].X)  # exact inverse of log1p
        profiles[ct] = np.asarray(linear_expr.mean(axis=0)).flatten()

    return pd.DataFrame(profiles, index=raw.var_names)


def select_signature_genes(adata, n_per_type=N_GENES_PER_TYPE, min_log2fc=MIN_LOG2FC, max_padj=MAX_PADJ):
    """
    Union of the top n_per_type up-regulated genes per real cell type
    (Wilcoxon on log-normalized adata.raw, each type vs. the rest, requiring
    log2FC >= min_log2fc and adjusted p <= max_padj), in rank order.
    """
    raw = _real_cells_raw_adata(adata)
    if raw.obs["cell_type"].nunique() < 2:
        raise ValueError("Need at least two real cell types to select marker genes.")

    sc.tl.rank_genes_groups(raw, groupby="cell_type", method="wilcoxon")

    selected = []
    per_type_counts = {}
    for ct in raw.obs["cell_type"].cat.categories:
        de = sc.get.rank_genes_groups_df(raw, group=ct)
        de = de[(de["logfoldchanges"] >= min_log2fc) & (de["pvals_adj"] <= max_padj)]
        top = de.sort_values("scores", ascending=False)["names"].head(n_per_type).tolist()
        per_type_counts[ct] = len(top)
        selected.extend(top)

    genes = list(dict.fromkeys(selected))
    print(f"Selected {len(genes)} signature genes (up to {n_per_type} per type): {per_type_counts}")
    thin = [ct for ct, n in per_type_counts.items() if n < min(10, n_per_type)]
    if thin:
        print(f"  Warning: few qualifying marker genes for {thin}; their columns will be poorly determined.")
    return genes


def main():
    parser = argparse.ArgumentParser(description="Export a cell-type signature matrix for bulk deconvolution.")
    parser.add_argument("--n-genes-per-type", type=int, default=N_GENES_PER_TYPE,
                        help=f"Top DE genes per cell type to include (default {N_GENES_PER_TYPE}).")
    args = parser.parse_args()

    input_file = "data/annotated_data.h5ad"
    print(f"Loading {input_file}...")
    adata = sc.read(input_file)

    signature_matrix = build_signature_matrix(adata)
    genes = select_signature_genes(adata, n_per_type=args.n_genes_per_type)
    signature_matrix = signature_matrix.loc[genes]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "cell_type_signature_matrix.csv")
    signature_matrix.to_csv(output_path)
    print(
        f"\nSaved signature matrix ({signature_matrix.shape[0]} genes x "
        f"{signature_matrix.shape[1]} cell types, linear CP10K scale) to {output_path}"
    )


if __name__ == "__main__":
    main()
