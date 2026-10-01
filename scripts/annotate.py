"""
Cell-type annotation for endometrial tumor tissue.

Primary method: a marker-gene panel. Each marker's per-cluster mean (all genes,
adata.raw) is z-scored across clusters; a type's score is the mean z-score of its
markers. A cluster is "Ambiguous" when the best score is not positive or is within
min_margin of the runner-up.

CellTypist (immune-only models) is a consistency check on immune clusters only.
"""

import os
import re
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scanpy as sc
import celltypist
from celltypist import models

from scipy.stats import mannwhitneyu

from labels import AMBIGUOUS_LABEL, UNKNOWN_LABEL, DOUBLET_LABEL
from provenance import package_versions, verify_checksum

RESULTS_DIR = "results"


def _results_path(filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)


def _figures_dir(results_dir=None):
    path = os.path.join(results_dir or RESULTS_DIR, "figures")
    os.makedirs(path, exist_ok=True)
    sc.settings.figdir = path
    return path

CELLTYPIST_MODEL_NAME = "Immune_All_Low.pkl"
# SHA-256 of the model file the README's results were produced with.
CELLTYPIST_MODEL_SHA256 = "290874d35dac039d4c9218c343fde4aac1077709b72a331ce7266f6828c36502"

MIN_MARGIN_DEFAULT = 0.15

MARKER_GENES = {
    # Epithelial (EPCAM pan-epithelial; KRT8/KRT18 cytokeratins; PAX8 Mullerian lineage)
    "EPCAM": "Epithelial cell",
    "KRT8": "Epithelial cell",
    "KRT18": "Epithelial cell",
    "PAX8": "Epithelial cell",
    # Fibroblast (COL1A1, DCN) and pericyte (RGS5, ACTA2, PDGFRB) markers, merged:
    # on this data one stromal cluster scored the two within 0.013 of each other.
    "COL1A1": "Stromal cell",
    "DCN": "Stromal cell",
    "RGS5": "Stromal cell",
    "ACTA2": "Stromal cell",
    "PDGFRB": "Stromal cell",
    # Endothelial
    "PECAM1": "Endothelial cell",
    "VWF": "Endothelial cell",
    "CDH5": "Endothelial cell",
    # T (CD3D, CD2) and NK (GNLY, NKG7) markers, merged
    "CD3D": "T/NK cell",
    "CD2": "T/NK cell",
    "GNLY": "T/NK cell",
    "NKG7": "T/NK cell",
    # B cell
    "MS4A1": "B cell",
    "CD79A": "B cell",
    "CD19": "B cell",
    # Plasma cell
    "MZB1": "Plasma cell",
    "JCHAIN": "Plasma cell",
    "XBP1": "Plasma cell",
    # Myeloid / monocyte / macrophage
    "CD14": "Myeloid/Monocyte",
    "CD68": "Myeloid/Monocyte",
    "LYZ": "Myeloid/Monocyte",
    # Dendritic cell
    "ITGAX": "Dendritic cell",
    "CD1C": "Dendritic cell",
    "CLEC9A": "Dendritic cell",
    # Neutrophil
    "FCGR3B": "Neutrophil",
    "CSF3R": "Neutrophil",
    "S100A8": "Neutrophil"
}

T_NK_LABEL = "T/NK cell"
STROMAL_LABEL = "Stromal cell"

T_NK_SUBTYPE_MARKERS = {
    "T-leaning": ["CD3D", "CD2"],
    "NK-leaning": ["GNLY", "NKG7"],
}
STROMAL_SUBTYPE_MARKERS = {
    "Fibroblast-leaning": ["COL1A1", "DCN"],
    "Pericyte-leaning": ["RGS5", "ACTA2", "PDGFRB"],
}

# Marker-panel labels checked against CellTypist in cross_check_immune_cells().
IMMUNE_LABELS = {T_NK_LABEL, "B cell", "Plasma cell", "Myeloid/Monocyte", "Dendritic cell", "Neutrophil"}


def _model_is_cached(model_name):
    try:
        model_path = os.path.join(models.models_path, model_name)
        return os.path.exists(model_path)
    except Exception:
        return False


def _celltypist_model_info(model, model_name=CELLTYPIST_MODEL_NAME):
    """Model name, file SHA-256, and the version/date fields CellTypist stores in the model."""
    info = {"name": model_name, "sha256": "unavailable"}
    try:
        model_path = os.path.join(models.models_path, model_name)
    except Exception:
        model_path = None
    if model_path and os.path.exists(model_path):
        info["sha256"] = verify_checksum(model_path, CELLTYPIST_MODEL_SHA256, f"CellTypist model {model_name}",
                                         strict=False)
    description = getattr(model, "description", None)
    if isinstance(description, dict):
        for key in ("version", "date", "source", "details"):
            if key in description:
                info[key] = str(description[key])
    return info


def run_celltypist(adata, over_clustering_key="leiden_clusters"):
    """
    Run CellTypist's Immune_All_Low model and merge its prediction columns
    back onto adata.
    """
    if adata.raw is None:
        raise ValueError("adata.raw is not set -- CellTypist needs all genes; re-run scripts/preprocess.py.")

    if _model_is_cached(CELLTYPIST_MODEL_NAME):
        print(f"Using cached CellTypist model: {CELLTYPIST_MODEL_NAME}.")
    else:
        print(f"Downloading CellTypist model: {CELLTYPIST_MODEL_NAME}...")
        models.download_models(model=[CELLTYPIST_MODEL_NAME])

    model = models.Model.load(model=CELLTYPIST_MODEL_NAME)
    adata.uns["celltypist_model_info"] = _celltypist_model_info(model)

    celltypist_input = adata.raw.to_adata()
    celltypist_input.obs = adata.obs.copy()
    kwargs = {"model": model, "majority_voting": True}
    if over_clustering_key in adata.obs.columns:
        kwargs["over_clustering"] = adata.obs[over_clustering_key].astype(str).values
    print(f"Running CellTypist on {celltypist_input.n_vars} genes (adata.raw).")
    predictions = celltypist.annotate(celltypist_input, **kwargs)
    result_adata = predictions.to_adata()

    for col in ["predicted_labels", "majority_voting", "conf_score"]:
        if col in result_adata.obs.columns:
            adata.obs[col] = result_adata.obs[col].values

    return adata


# Maps CellTypist's fine-grained vocabulary to the marker-panel categories it is compatible with, so
# the cross-check separates genuine lineage conflicts from finer subtyping or wording differences
# ("Memory B cells" vs "B cell").
CELLTYPIST_COMPATIBILITY_RULES = [
    (r"plasmablasts?", {"Plasma cell", "B cell"}),
    (r"plasma", {"Plasma cell"}),
    (r"b cells?|memory b|naive b|b-cells?", {"B cell"}),
    (r"(p|mig)?dcs?\d?|dendritic", {"Dendritic cell", "Myeloid/Monocyte"}),
    (r"neutrophils?", {"Neutrophil"}),
    (r"macrophages?|monocytes?|myeloid|mono-mac", {"Myeloid/Monocyte"}),
    (r"t cells?|t-cells?|tregs?|regulatory t|tem|tcm|trm|tfh|th1|th2|th17|mait|cytotoxic|"
     r"nk cells?|nk|natural killer|ilcs?|ilc\d", {T_NK_LABEL}),
]


def _normalize_label(label):
    return re.sub(r"[^a-z0-9\-]+", " ", label.strip().lower()).strip()


def _celltypist_label_compatible_with_marker_label(celltypist_label, marker_label):
    """
    True if the CellTypist label maps (via CELLTYPIST_COMPATIBILITY_RULES) to
    a set of categories containing marker_label, or if the two labels are the
    same apart from case, whitespace, and a trailing 's'. False otherwise,
    i.e. a genuine lineage mismatch.
    """
    a, b = celltypist_label.strip().lower(), marker_label.strip().lower()
    if a == b or a.rstrip("s") == b.rstrip("s"):
        return True

    normalized = _normalize_label(celltypist_label)
    for pattern, compatible in CELLTYPIST_COMPATIBILITY_RULES:
        if re.search(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", normalized):
            return marker_label in compatible
    return False


def cross_check_immune_cells(adata, immune_labels=None):
    """
    Compare CellTypist's majority_voting call against the marker-gene
    label, restricted to clusters the marker panel identified as immune.
    """
    if immune_labels is None:
        immune_labels = IMMUNE_LABELS

    if "majority_voting" not in adata.obs.columns:
        print("No CellTypist majority_voting column found -- skipping cross-check.")
        return None

    is_immune_cluster = adata.obs["marker_gene_label"].isin(immune_labels)
    if is_immune_cluster.sum() == 0:
        print("No clusters labeled as immune by the marker panel -- skipping cross-check.")
        return None

    comparison = (
        adata.obs.loc[is_immune_cluster]
        .groupby("leiden_clusters", observed=True)[["majority_voting", "marker_gene_label"]]
        .agg(lambda s: s.mode().iat[0])
    )
    comparison["compatible"] = comparison.apply(
        lambda row: _celltypist_label_compatible_with_marker_label(row["majority_voting"], row["marker_gene_label"]),
        axis=1,
    )
    print("\nCellTypist majority_voting vs. marker-gene label (consistency check only, immune clusters):")
    print(comparison)
    comparison.to_csv(_results_path("annotation_cross_check_immune_only.csv"))

    disagreements = comparison[~comparison["compatible"]]
    if len(disagreements) > 0:
        print(
            f"\n{len(disagreements)} cluster(s) show a GENUINE disagreement (not just subtype "
            f"granularity or wording). This is a PROMPT for marker-level review, not evidence "
            f"either method is simply wrong. For each, check:"
        )
        for cluster in disagreements.index:
            print(
                f"  Cluster {cluster}: CellTypist='{disagreements.loc[cluster, 'majority_voting']}', "
                f"marker panel='{disagreements.loc[cluster, 'marker_gene_label']}' "
                f"results/figures/dotplot_cluster_diagnostic.png (cluster {cluster} column) and "
                f"results/marker_annotation_confidence.csv (cluster {cluster} row) before trusting either call."
            )

    return comparison


def _get_marker_raw_subset(adata, genes):
    available = [g for g in genes if g in adata.raw.var_names]
    missing = [g for g in genes if g not in adata.raw.var_names]
    subset = adata.raw[:, available].to_adata()
    return subset, available, missing


def _label_from_scores(scores, min_margin):
    """
    Shared labeling logic: given a dict of {cell_type: score}, apply the
    same competing-lineage "Ambiguous" rule at both the per-cluster and
    per-cell level, so the two levels agree on what counts as ambiguous.

    Returns (label, top_type, top_score, second_type, second_score, margin).
    """
    if not scores:
        return UNKNOWN_LABEL, None, np.nan, None, np.nan, np.nan

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_type, top_score = ranked[0]
    second_type, second_score = ranked[1] if len(ranked) > 1 else (None, -np.inf)
    margin = top_score - second_score

    label = AMBIGUOUS_LABEL if (top_score <= 0 or margin < min_margin) else top_type
    return label, top_type, top_score, second_type, second_score, margin


def record_marker_panel_coverage(available_genes):
    """
    Record how many of each cell type's markers are present in the data; a
    type scored from 2 of 3 markers is less well supported than one scored
    from 4 of 4.
    """
    coverage_records = []
    all_types = sorted(set(MARKER_GENES.values()))
    for ct in all_types:
        assigned = [g for g, t in MARKER_GENES.items() if t == ct]
        found = [g for g in assigned if g in available_genes]
        coverage_records.append({
            "cell_type": ct,
            "markers_assigned": len(assigned),
            "markers_available": len(found),
            "coverage_fraction": round(len(found) / len(assigned), 3) if assigned else None,
            "missing_markers": [g for g in assigned if g not in available_genes],
        })

    coverage_df = pd.DataFrame(coverage_records)
    coverage_df.to_csv(_results_path("marker_panel_coverage.csv"), index=False)
    print("\nMarker panel coverage per cell type:")
    print(coverage_df[["cell_type", "markers_assigned", "markers_available", "coverage_fraction"]].to_string(index=False))
    return coverage_df


def marker_gene_annotation(adata, min_margin=MIN_MARGIN_DEFAULT):
    """
    Assign a cell type per Leiden cluster from the marker panel.

    Returns (adata, cluster_level_gene_scale_stats). The scale statistics
    are computed from cluster means and must not be reused to z-score
    individual cells (see compute_cluster_concordance()).
    """
    if adata.raw is None:
        raise ValueError("adata.raw is not set -- re-run scripts/preprocess.py.")

    marker_gene_list = list(MARKER_GENES.keys())
    raw_subset, available_genes, missing_genes = _get_marker_raw_subset(adata, marker_gene_list)
    if missing_genes:
        print(f"Note: {len(missing_genes)} marker genes not found in the data and skipped: {missing_genes}")

    record_marker_panel_coverage(available_genes)

    clusters = list(adata.obs["leiden_clusters"].cat.categories)

    gene_cluster_means = {}
    gene_cluster_pct_positive = {}
    for gene in available_genes:
        means, pct_pos = [], []
        for cluster in clusters:
            mask = (adata.obs["leiden_clusters"] == cluster).values
            vals = raw_subset[mask, gene].X
            vals = np.asarray(vals.todense()) if hasattr(vals, "todense") else np.asarray(vals)
            means.append(float(vals.mean()))
            pct_pos.append(float((vals > 0).mean()))
        gene_cluster_means[gene] = np.array(means)
        gene_cluster_pct_positive[gene] = np.array(pct_pos)

    gene_zscores = {}
    cluster_level_gene_scale_stats = {}  # from cluster means; not valid for individual cells
    for gene, means in gene_cluster_means.items():
        mean, std = means.mean(), means.std()
        cluster_level_gene_scale_stats[gene] = (mean, std)
        gene_zscores[gene] = np.zeros_like(means) if std == 0 else (means - mean) / std

    marker_label_per_cluster = {}
    confidence_records = []
    for i, cluster in enumerate(clusters):
        type_zscore_lists = {}
        for gene, cell_type in MARKER_GENES.items():
            if gene in gene_zscores:
                type_zscore_lists.setdefault(cell_type, []).append(gene_zscores[gene][i])

        scores = {ct: float(np.mean(vals)) for ct, vals in type_zscore_lists.items()}
        label, top_type, top_score, second_type, second_score, margin = _label_from_scores(scores, min_margin)
        marker_label_per_cluster[cluster] = label

        if top_type is not None:
            # Fraction of cells expressing the winning type's markers, which separates
            # consistent expression from a mean driven by a few very high cells.
            top_type_genes = [g for g, ct in MARKER_GENES.items() if ct == top_type and g in gene_cluster_pct_positive]
            pct_positive = float(np.mean([gene_cluster_pct_positive[g][i] for g in top_type_genes])) if top_type_genes else np.nan

            confidence_records.append({
                "leiden_cluster": cluster, "assigned_label": label,
                "top_type": top_type, "top_score": top_score,
                "second_type": second_type, "second_score": second_score, "margin": margin,
                "top_type_pct_cells_expressing_markers": round(pct_positive, 3) if not np.isnan(pct_positive) else None,
            })

    adata.obs["marker_gene_label"] = adata.obs["leiden_clusters"].map(marker_label_per_cluster)

    def _compute_lean(merged_label, subtype_markers, lean_a, lean_b):
        result = {}
        for i, cluster in enumerate(clusters):
            if marker_label_per_cluster[cluster] != merged_label:
                continue
            lean_scores = {}
            for lean, genes in subtype_markers.items():
                vals = [gene_zscores[g][i] for g in genes if g in gene_zscores]
                if vals:
                    lean_scores[lean] = float(np.mean(vals))
            if len(lean_scores) == 2:
                diff = lean_scores[lean_a] - lean_scores[lean_b]
                result[cluster] = lean_a if diff > 0.3 else lean_b if diff < -0.3 else "Balanced"
        return result

    t_nk_lean = _compute_lean(T_NK_LABEL, T_NK_SUBTYPE_MARKERS, "T-leaning", "NK-leaning")
    adata.obs["t_nk_subtype_lean"] = adata.obs["leiden_clusters"].map(t_nk_lean).fillna("N/A")

    stromal_lean = _compute_lean(STROMAL_LABEL, STROMAL_SUBTYPE_MARKERS, "Fibroblast-leaning", "Pericyte-leaning")
    adata.obs["stromal_subtype_lean"] = adata.obs["leiden_clusters"].map(stromal_lean).fillna("N/A")

    confidence_df = pd.DataFrame(confidence_records)
    confidence_df.to_csv(_results_path("marker_annotation_confidence.csv"), index=False)

    return adata, cluster_level_gene_scale_stats


def run_min_margin_sensitivity(adata, margins_to_test=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):
    """Count Ambiguous clusters across a range of min_margin values, to show how much the default (0.15) matters."""
    if adata.raw is None:
        raise ValueError("adata.raw is not set.")

    marker_gene_list = list(MARKER_GENES.keys())
    raw_subset, available_genes, _ = _get_marker_raw_subset(adata, marker_gene_list)
    clusters = list(adata.obs["leiden_clusters"].cat.categories)

    gene_cluster_means = {}
    for gene in available_genes:
        means = []
        for cluster in clusters:
            mask = (adata.obs["leiden_clusters"] == cluster).values
            vals = raw_subset[mask, gene].X
            vals = np.asarray(vals.todense()) if hasattr(vals, "todense") else np.asarray(vals)
            means.append(float(vals.mean()))
        gene_cluster_means[gene] = np.array(means)

    gene_zscores = {}
    for gene, means in gene_cluster_means.items():
        mean, std = means.mean(), means.std()
        gene_zscores[gene] = np.zeros_like(means) if std == 0 else (means - mean) / std

    results = []
    for margin in margins_to_test:
        n_ambiguous = 0
        for i, cluster in enumerate(clusters):
            type_zscore_lists = {}
            for gene, cell_type in MARKER_GENES.items():
                if gene in gene_zscores:
                    type_zscore_lists.setdefault(cell_type, []).append(gene_zscores[gene][i])
            scores = {ct: float(np.mean(vals)) for ct, vals in type_zscore_lists.items()}
            label, *_ = _label_from_scores(scores, margin)
            if label == AMBIGUOUS_LABEL:
                n_ambiguous += 1
        results.append({
            "min_margin": margin, "n_clusters": len(clusters), "n_ambiguous_clusters": n_ambiguous,
            "pct_ambiguous": round(100 * n_ambiguous / len(clusters), 1) if clusters else None,
        })

    sensitivity_df = pd.DataFrame(results)
    sensitivity_df.to_csv(_results_path("min_margin_sensitivity.csv"), index=False)
    print(f"\nmin_margin sensitivity analysis (current default: {MIN_MARGIN_DEFAULT}):")
    print(sensitivity_df.to_string(index=False))
    return sensitivity_df


def compute_cluster_concordance(adata, marker_gene_list, min_margin=MIN_MARGIN_DEFAULT):
    """
    Score each cell individually with the same rule used for clusters, and
    report per cluster the fraction of cells that are:
      concordant        : the cell's own label matches the cluster label
      ambiguous_cell    : the cell itself scores Ambiguous
      discordant_other  : the cell scores a different specific type
    A low concordant fraction marks a cluster that may mix populations.
    """
    raw_subset, available_genes, _ = _get_marker_raw_subset(adata, marker_gene_list)

    X = raw_subset[:, available_genes].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    cell_level_gene_scale_stats = {}
    for j, gene in enumerate(available_genes):
        col = X[:, j]
        cell_level_gene_scale_stats[gene] = (float(col.mean()), float(col.std()))

    per_cell_zscores = np.zeros_like(X, dtype=float)
    for j, gene in enumerate(available_genes):
        mean, std = cell_level_gene_scale_stats[gene]
        per_cell_zscores[:, j] = 0.0 if std == 0 else (X[:, j] - mean) / std

    cell_types_by_gene = [MARKER_GENES[g] for g in available_genes]
    unique_types = sorted(set(cell_types_by_gene))
    type_to_gene_idx = {ct: [j for j, t in enumerate(cell_types_by_gene) if t == ct] for ct in unique_types}

    per_cell_type_scores = np.zeros((X.shape[0], len(unique_types)))
    for k, ct in enumerate(unique_types):
        idx = type_to_gene_idx[ct]
        per_cell_type_scores[:, k] = per_cell_zscores[:, idx].mean(axis=1)

    per_cell_best_label = np.empty(X.shape[0], dtype=object)
    for row in range(X.shape[0]):
        row_scores = {ct: per_cell_type_scores[row, k] for k, ct in enumerate(unique_types)}
        label, _, _, _, _, _ = _label_from_scores(row_scores, min_margin)
        per_cell_best_label[row] = label

    concordance_df = pd.DataFrame({
        "leiden_cluster": adata.obs["leiden_clusters"].values,
        "cluster_label": adata.obs["marker_gene_label"].values,
        "per_cell_best_type": per_cell_best_label,
    })
    concordance_df["outcome"] = np.select(
        [
            concordance_df["per_cell_best_type"] == concordance_df["cluster_label"],
            concordance_df["per_cell_best_type"] == AMBIGUOUS_LABEL,
        ],
        ["concordant", "ambiguous_cell"],
        default="discordant_other",
    )

    cluster_concordance = (
        concordance_df.groupby("leiden_cluster", observed=True)["outcome"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for col in ["concordant", "ambiguous_cell", "discordant_other"]:
        if col not in cluster_concordance.columns:
            cluster_concordance[col] = 0.0
    cluster_concordance = cluster_concordance[["leiden_cluster", "concordant", "ambiguous_cell", "discordant_other"]]
    cluster_concordance.to_csv(_results_path("cluster_annotation_concordance.csv"), index=False)

    low_concordance = cluster_concordance[cluster_concordance["concordant"] < 0.5]
    if len(low_concordance) > 0:
        print(
            f"\nWarning: {len(low_concordance)} cluster(s) have concordant fraction < 0.5 -- "
            f"likely heterogeneous, worth manual review: {low_concordance['leiden_cluster'].tolist()}"
        )
    print(f"Saved per-cluster annotation concordance (concordant/ambiguous_cell/discordant_other fractions) "
          f"to {RESULTS_DIR}/cluster_annotation_concordance.csv")

    return cluster_concordance


def plot_marker_validation(adata, marker_genes, results_dir=RESULTS_DIR, suffix="_marker_validation"):
    if adata.raw is None:
        print("adata.raw is not set -- cannot generate marker validation plots.")
        return

    _figures_dir(results_dir)

    validation_data, available_genes, missing_genes = _get_marker_raw_subset(adata, marker_genes)
    validation_data.obs["cell_type"] = adata.obs["cell_type"].values

    if missing_genes:
        print(f"Warning: these marker genes are not in the dataset and will be skipped: {missing_genes}")
    if not available_genes:
        print("None of the requested marker genes are present in the data -- skipping validation plots.")
        return

    # scanpy prepends "dotplot_" / "stacked_violin_" to the save name itself
    save_name = f"{suffix.lstrip('_')}.png"
    dotplot_name = f"dotplot_{save_name}"
    violin_name = f"stacked_violin_{save_name}"
    sc.pl.dotplot(validation_data, available_genes, groupby="cell_type", save=save_name, show=False)
    sc.pl.stacked_violin(validation_data, available_genes, groupby="cell_type", save=save_name, show=False)
    print(f"Saved marker validation plots to {sc.settings.figdir}/{dotplot_name} and {sc.settings.figdir}/{violin_name}.")


def flag_doublet_enriched_clusters(adata, min_fold=2.0, alpha=0.01, cluster_key="leiden_clusters"):
    """
    Cluster-level doublet check on the per-cell Scrublet scores from
    preprocess.py.

    Returns one row per cluster; writes results/cluster_doublet_enrichment.csv.
    """
    if "doublet_score" not in adata.obs.columns:
        print("No 'doublet_score' column -- skipping cluster-level doublet check.")
        return pd.DataFrame(columns=["leiden_cluster", "n_cells", "mean_doublet_score", "fold_vs_rest",
                                     "pval", "padj", "flagged"])

    scores = adata.obs["doublet_score"].astype(float).values
    clusters = adata.obs[cluster_key].astype(str).values
    records = []
    for cluster in sorted(set(clusters), key=lambda c: (len(c), c)):
        in_cluster = clusters == cluster
        inside, outside = scores[in_cluster], scores[~in_cluster]
        mean_in, mean_out = float(inside.mean()), float(outside.mean())
        pval = float(mannwhitneyu(inside, outside, alternative="greater").pvalue) if len(outside) else 1.0
        records.append({
            "leiden_cluster": cluster,
            "n_cells": int(in_cluster.sum()),
            "mean_doublet_score": round(mean_in, 4),
            "fold_vs_rest": round(mean_in / mean_out, 3) if mean_out > 0 else np.inf,
            "pval": pval,
        })

    table = pd.DataFrame(records)
    # Benjamini-Hochberg adjustment across clusters
    order = np.argsort(table["pval"].values)
    ranked = table["pval"].values[order] * len(table) / (np.arange(len(table)) + 1)
    padj_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    padj = np.empty(len(table))
    padj[order] = padj_sorted
    table["padj"] = padj
    table["flagged"] = (table["padj"] < alpha) & (table["fold_vs_rest"] >= min_fold)
    table.to_csv(_results_path("cluster_doublet_enrichment.csv"), index=False)

    flagged = table[table["flagged"]]
    if len(flagged):
        for _, row in flagged.iterrows():
            print(f"  Cluster {row['leiden_cluster']} ({row['n_cells']} cells): mean doublet score "
                  f"{row['mean_doublet_score']:.3f}, {row['fold_vs_rest']:.1f}x the other cells "
                  f"(BH-adjusted p = {row['padj']:.1e}) -- labeled '{DOUBLET_LABEL}'.")
    else:
        print(f"  No cluster has a significantly elevated doublet score >= {min_fold}x the other cells.")
    print(f"Saved per-cluster doublet test to {RESULTS_DIR}/cluster_doublet_enrichment.csv")
    return table


def apply_doublet_label(labels, clusters, doublet_table):
    """Return labels (categorical) with every cell in a flagged cluster set to DOUBLET_LABEL."""
    flagged = set(doublet_table.loc[doublet_table["flagged"], "leiden_cluster"].astype(str)) if len(doublet_table) else set()
    out = np.where(np.isin(np.asarray(clusters).astype(str), list(flagged)), DOUBLET_LABEL,
                   np.asarray(labels).astype(str))
    return pd.Categorical(out)


def _record_run_metadata(adata, min_margin=MIN_MARGIN_DEFAULT):
    """Record software, model, marker panel, and parameters for reproducibility."""
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": package_versions(["scanpy", "anndata", "celltypist", "pandas", "numpy", "scipy"]),
        "celltypist_model": dict(adata.uns.get("celltypist_model_info", {"name": CELLTYPIST_MODEL_NAME})),
        "celltypist_role": "consistency check on immune clusters only, not independent validation",
        "celltypist_input": "adata.raw (all genes), majority voting over leiden_clusters",
        "doublet_cluster_rule": "one-sided Mann-Whitney vs. other cells, BH padj < 0.01 and mean >= 2x other cells",
        "min_margin": min_margin,
        "marker_gene_panel": MARKER_GENES,
        "n_cells_final": int(adata.n_obs),
        "n_clusters": int(adata.obs["leiden_clusters"].nunique()),
    }
    with open(_results_path("run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata to {RESULTS_DIR}/run_metadata.json")


def plot_cluster_diagnostic(adata, results_dir=RESULTS_DIR):
    curated_markers = {
        "Epithelial": ["EPCAM", "KRT8", "PAX8"],
        "Fibroblast": ["COL1A1", "DCN", "COL1A2"],
        "Endothelial": ["PECAM1", "VWF", "CDH5"],
        "Pericyte": ["RGS5", "ACTA2", "PDGFRB"],
        "T/NK": ["CD3D", "GNLY", "CD2"],
        "B cell": ["MS4A1", "CD79A", "CD19"],
        "Plasma cell": ["MZB1", "JCHAIN", "XBP1"],
        "Myeloid": ["CD14", "CD68", "LYZ"],
        "Dendritic": ["ITGAX", "CD1C", "CLEC9A"],
        "Neutrophil": ["S100A8", "FCGR3B", "CSF3R"],
    }
    all_genes = [g for genes in curated_markers.values() for g in genes]

    if adata.raw is None:
        print("adata.raw is not set -- cannot generate cluster diagnostic plot.")
        return

    diag_data, available_genes, missing_genes = _get_marker_raw_subset(adata, all_genes)
    if missing_genes:
        print(f"Note: diagnostic markers not in dataset, skipped: {missing_genes}")
    diag_data.obs["leiden_clusters"] = adata.obs["leiden_clusters"].values

    _figures_dir(results_dir)
    sc.pl.dotplot(diag_data, available_genes, groupby="leiden_clusters", save="cluster_diagnostic.png", show=False)
    print(f"Saved cluster diagnostic dot plot to {results_dir}/figures/dotplot_cluster_diagnostic.png.")


def main():
    input_file = "data/processed_data.h5ad"
    print(f"Loading {input_file}...")
    adata = sc.read(input_file)

    if "leiden_clusters" not in adata.obs:
        raise KeyError("'leiden_clusters' not found. Run scripts/clustering.py first.")

    print("Assigning primary cell type labels from marker gene panel...")
    adata, cluster_level_gene_scale_stats = marker_gene_annotation(adata)
    print(adata.obs["marker_gene_label"].value_counts())

    print("\nTesting sensitivity of Ambiguous-cluster count to min_margin...")
    run_min_margin_sensitivity(adata)

    print("\nChecking individual-cell agreement within each cluster (heterogeneity check)...")
    compute_cluster_concordance(adata, marker_gene_list=list(MARKER_GENES.keys()))

    print("\nRunning CellTypist on all genes (only prediction columns are merged back)...")
    adata = run_celltypist(adata)

    cross_check_immune_cells(adata)

    print("\nTesting each cluster for elevated Scrublet doublet scores...")
    doublet_table = flag_doublet_enriched_clusters(adata)

    # cell_type is the marker-panel label, except that doublet-enriched
    # clusters get DOUBLET_LABEL. Nothing is removed here; downstream scripts
    # exclude Ambiguous, Unknown, and doublet labels.
    _figures_dir()
    adata.obs["cell_type"] = apply_doublet_label(adata.obs["marker_gene_label"], adata.obs["leiden_clusters"],
                                                 doublet_table)
    print(adata.obs["cell_type"].value_counts())

    sc.pl.umap(adata, color=["cell_type"], save="_annotation.png", show=False)

    canonical_markers = ["EPCAM", "DCN", "PECAM1", "CD14", "CD3D", "CD79A"]
    plot_marker_validation(adata, canonical_markers, suffix="_marker_validation_canonical")

    plot_cluster_diagnostic(adata)

    full_marker_list = list(dict.fromkeys(MARKER_GENES.keys()))
    plot_marker_validation(adata, full_marker_list, suffix="_marker_validation_full")

    _record_run_metadata(adata)

    output_file = "data/annotated_data.h5ad"
    adata.write(output_file)
    print(f"Saved annotated data (with cell_type) to {output_file}")
    print("(Note: data/processed_data.h5ad, clustering.py's output, is left untouched.)")


if __name__ == "__main__":
    main()
