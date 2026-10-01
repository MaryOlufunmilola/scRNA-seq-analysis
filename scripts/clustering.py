"""
PCA, Harmony integration across patients, Leiden clustering, and UMAP.

Harmony uses sample_id (patient) as the batch key and removes both technical and
biological between-patient variation. Cluster x library tables are written because
within-patient library effects are not corrected. PCA runs without per-gene scaling.
"""

import os
import platform
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy
from provenance import package_versions

RESULTS_DIR = "results"
N_PCS_COMPUTED = 50  # components computed by pca_reduction(); n_pcs downstream must not exceed this

CANONICAL_SANITY_MARKERS = ["EPCAM", "COL1A1", "PECAM1", "CD3D", "CD14", "MS4A1"]


def _figures_dir(results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, "figures")
    os.makedirs(path, exist_ok=True)
    sc.settings.figdir = path
    return path


def pca_reduction(adata, n_components=N_PCS_COMPUTED):
    """PCA on the processed (log-normalized, HVG) matrix."""
    sc.tl.pca(adata, n_comps=n_components)
    return adata


def plot_pca_variance(adata, results_dir=RESULTS_DIR):
    """Plot variance explained per PC to guide the choice of n_pcs."""
    fig_dir = _figures_dir(results_dir)
    variance_ratio = adata.uns["pca"]["variance_ratio"]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(variance_ratio) + 1), variance_ratio, marker="o", markersize=3)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio explained")
    ax.set_title("PCA variance explained per component")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pca_variance_ratio.png"), dpi=150)
    plt.close(fig)
    print(f"Saved PCA variance ratio plot to {fig_dir}/pca_variance_ratio.png")


def harmony_integration(adata, batch_key="sample_id", n_pcs=None):
    """
    Run Harmony on the first n_pcs principal components (all computed PCs if
    None) and store the corrected embedding in adata.obsm['X_pca_harmony'].
    """
    if batch_key not in adata.obs.columns or adata.obs[batch_key].nunique() < 2:
        print(f"Skipping Harmony: '{batch_key}' not found or has <2 unique values.")
        return adata, "X_pca"

    n_available = adata.obsm["X_pca"].shape[1]
    n_use = n_available if n_pcs is None else n_pcs
    if not (1 <= n_use <= n_available):
        raise ValueError(f"n_pcs={n_pcs} is out of range for X_pca, which has {n_available} components.")

    pcs = np.asarray(adata.obsm["X_pca"][:, :n_use])
    ho = harmonypy.run_harmony(pcs, adata.obs, [batch_key])

    raw = np.asarray(ho.Z_corr)
    expected_shape = (adata.n_obs, n_use)
    if raw.shape == expected_shape:
        corrected = raw
    elif raw.T.shape == expected_shape:
        corrected = raw.T
    else:
        raise ValueError(
            f"Harmony output shape {raw.shape} (or its transpose {raw.T.shape}) "
            f"doesn't match expected {expected_shape} -- inspect ho.Z_corr directly."
        )

    adata.obsm["X_pca_harmony"] = corrected
    print(f"Harmony integration on the first {n_use} PCs, batch key '{batch_key}'.")
    return adata, "X_pca_harmony"


def leiden_clustering(adata, use_rep="X_pca", resolution=1.0, n_neighbors=20, n_pcs=20, random_state=0):
    """Build a kNN graph on adata.obsm[use_rep] and cluster it with Leiden."""
    if use_rep not in adata.obsm:
        raise ValueError(f"'{use_rep}' not found in adata.obsm -- available: {list(adata.obsm.keys())}")

    n_pcs_available = adata.obsm[use_rep].shape[1]
    if not (1 <= n_pcs <= n_pcs_available):
        raise ValueError(
            f"n_pcs={n_pcs} is out of range for '{use_rep}', which has only "
            f"{n_pcs_available} components actually computed. Pass a value between "
            f"1 and {n_pcs_available}, or recompute with more components via pca_reduction()."
        )

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep, random_state=random_state)
    sc.tl.leiden(
        adata, resolution=resolution, random_state=random_state, key_added="leiden_clusters",
        flavor="igraph", n_iterations=2, directed=False,
    )
    return adata


def assess_clustering_stability(adata, use_rep, resolution, n_neighbors, n_pcs, n_repeats=5,
                                subsample_fraction=0.9):
    """
    Re-cluster n_repeats times on random subsamples of cells, each with a
    different seed, and compare each result to the original clustering on
    the same cells using the Adjusted Rand Index (1.0 = identical, ~0 =
    chance agreement).
    """
    from sklearn.metrics import adjusted_rand_score

    if not (0 < subsample_fraction <= 1.0):
        raise ValueError("subsample_fraction must be in (0, 1].")

    rng = np.random.default_rng(0)
    original_labels = adata.obs["leiden_clusters"].values
    n_sub = max(int(round(adata.n_obs * subsample_fraction)), n_neighbors + 1)
    ari_scores = []

    for i in range(n_repeats):
        seed = 1000 + i
        idx = np.sort(rng.choice(adata.n_obs, size=n_sub, replace=False)) if n_sub < adata.n_obs else np.arange(adata.n_obs)
        temp = adata[idx].copy()
        sc.pp.neighbors(temp, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep, random_state=seed)
        sc.tl.leiden(
            temp, resolution=resolution, random_state=seed, key_added="_stability_check",
            flavor="igraph", n_iterations=2, directed=False,
        )
        ari_scores.append(adjusted_rand_score(original_labels[idx], temp.obs["_stability_check"].values))

    mean_ari = float(np.mean(ari_scores))
    min_ari = float(np.min(ari_scores))
    print(
        f"\nClustering stability ({n_repeats} re-runs on {subsample_fraction:.0%} subsamples, new seeds): "
        f"mean ARI = {mean_ari:.3f}, min ARI = {min_ari:.3f}."
    )
    if min_ari < 0.7:
        print(
            "  Warning: at least one re-run disagreed substantially with the original clustering "
            "(ARI < 0.7); some cluster boundaries may be sensitive to which cells are present or "
            "to initialization."
        )

    return {
        "mean_ari": mean_ari, "min_ari": min_ari, "individual_ari_scores": ari_scores,
        "subsample_fraction": subsample_fraction,
    }


def plot_quick_marker_sanity_check(adata, results_dir=RESULTS_DIR):
    """
    Small canonical-marker dotplot by cluster: an early check that clusters
    separate broad lineages at all, before the full annotation in annotate.py.
    """
    available = [g for g in CANONICAL_SANITY_MARKERS if g in adata.raw.var_names] if adata.raw is not None else []
    if not available:
        print("No canonical sanity-check markers found in adata.raw -- skipping quick marker check.")
        return

    fig_dir = _figures_dir(results_dir)
    check_data = adata.raw[:, available].to_adata()
    check_data.obs["leiden_clusters"] = adata.obs["leiden_clusters"].values

    sc.pl.dotplot(
        check_data, available, groupby="leiden_clusters",
        save="quick_marker_sanity_check.png", show=False  # scanpy prepends "dotplot_"
    )
    print(f"Saved quick canonical-marker dotplot to {fig_dir}/dotplot_quick_marker_sanity_check.png")


def compute_cluster_composition_diagnostics(adata, key, name, results_dir=RESULTS_DIR, dominance_threshold=0.8):
    """
    Cluster x <key> count and proportion tables, flagging clusters where one
    level contributes more than dominance_threshold of the cells.

    Returns (counts, proportions, dominated) where dominated maps cluster ->
    largest share.
    """
    if key not in adata.obs.columns:
        print(f"No '{key}' column found -- skipping cluster x {name} diagnostics.")
        return None, None, {}

    os.makedirs(results_dir, exist_ok=True)
    counts = pd.crosstab(adata.obs["leiden_clusters"], adata.obs[key])
    proportions = counts.div(counts.sum(axis=1), axis=0)

    counts.to_csv(os.path.join(results_dir, f"cluster_by_{name}_counts.csv"))
    proportions.to_csv(os.path.join(results_dir, f"cluster_by_{name}_proportions.csv"))

    max_share = proportions.max(axis=1)
    dominated = max_share[max_share > dominance_threshold].round(3).to_dict()

    print(f"\nSaved cluster x {name} diagnostics to {results_dir}/cluster_by_{name}_{{counts,proportions}}.csv")
    if dominated:
        print(f"  {len(dominated)} cluster(s) have >{dominance_threshold:.0%} of cells from a single {name}: {dominated}")
    else:
        print(f"  No cluster is dominated (>{dominance_threshold:.0%}) by a single {name}.")
    return counts, proportions, dominated


def compute_cluster_patient_diagnostics(adata, results_dir=RESULTS_DIR):
    """Cluster x patient composition (see compute_cluster_composition_diagnostics)."""
    return compute_cluster_composition_diagnostics(adata, "sample_id", "patient", results_dir)


def compute_cluster_library_diagnostics(adata, results_dir=RESULTS_DIR):
    """
    Cluster x library composition. Harmony corrects between patients only,
    so a cluster dominated by one library of a multi-library patient points
    to an uncorrected technical effect.
    """
    if not {"sample_id", "library_id"}.issubset(adata.obs.columns):
        print("No sample_id/library_id columns -- skipping cluster x library diagnostics.")
        return None, None, {}
    adata.obs["library_key"] = (adata.obs["sample_id"].astype(str) + "/" + adata.obs["library_id"].astype(str)).astype("category")
    return compute_cluster_composition_diagnostics(adata, "library_key", "library", results_dir)


def umap_visualization(adata, use_rep="X_pca"):
    """
    Compute UMAP from the existing neighbor graph and plot it by cluster and
    by patient. use_rep is checked against the representation the neighbor
    graph was actually built from, so a mismatch fails loudly.
    """
    recorded_use_rep = adata.uns.get("neighbors", {}).get("params", {}).get("use_rep")
    if recorded_use_rep is not None and recorded_use_rep != use_rep:
        raise ValueError(
            f"umap_visualization() was called with use_rep='{use_rep}', but the precomputed "
            f"neighbor graph was actually built from use_rep='{recorded_use_rep}'. These must "
            f"match -- pass the same use_rep that was used in leiden_clustering()."
        )
    _figures_dir()
    print(f"Computing UMAP from the neighbor graph built on '{use_rep}'.")

    sc.tl.umap(adata, neighbors_key=None)
    sc.pl.umap(adata, color=["leiden_clusters"], save="_clustering.png", show=False)
    if "sample_id" in adata.obs.columns:
        sc.pl.umap(adata, color=["sample_id"], save="_by_sample.png", show=False)
    if "library_key" in adata.obs.columns:
        sc.pl.umap(adata, color=["library_key"], save="_by_library.png", show=False)
    return adata


def _record_clustering_metadata(adata, args, stability_results, use_rep):
    """Store parameters and software versions in adata.uns so provenance travels with the .h5ad."""
    adata.uns["clustering_params"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        #"scanpy_version": sc.__version__,  
        "scanpy_version": package_versions(["scanpy"])["scanpy"],
        "harmonypy_version": getattr(harmonypy, "__version__", "unknown"),
        "pca_input": "log-normalized HVG matrix, not scaled",
        "embedding": use_rep,
        "harmony_batch_key": "sample_id",
        "resolution": args.resolution,
        "n_neighbors": args.n_neighbors,
        "n_pcs": args.n_pcs,
        "n_pcs_computed": N_PCS_COMPUTED,
        "random_state": 0,
        "leiden_flavor": "igraph",
        "stability_mean_ari": stability_results["mean_ari"] if stability_results else None,
        "stability_min_ari": stability_results["min_ari"] if stability_results else None,
        "stability_subsample_fraction": stability_results["subsample_fraction"] if stability_results else None,
    }
    print("Recorded clustering parameters and software versions to adata.uns['clustering_params'].")


def main():
    parser = argparse.ArgumentParser(description="PCA, Harmony integration, and Leiden clustering.")
    parser.add_argument(
        "--resolution", type=float, default=1.0,
        help="Leiden resolution (default 1.0). See the README for how the defaults were chosen.",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=20,
        help=("Neighbors in the kNN graph (default 20; scanpy's default is 15). Smaller values are "
              "more locally sensitive; larger values smooth the graph and can merge populations."),
    )
    parser.add_argument(
        "--n-pcs", type=int, default=20,
        help=(f"PCs used for Harmony and the neighbor graph (1-{N_PCS_COMPUTED}). "
              f"Check results/figures/pca_variance_ratio.png."),
    )
    parser.add_argument(
        "--skip-stability-check", action="store_true",
        help="Skip the subsampling stability check (several extra Leiden runs).",
    )
    args = parser.parse_args()

    input_file = "data/processed_data.h5ad"
    print(f"Loading {input_file}...")
    adata = sc.read(input_file)

    adata = pca_reduction(adata)
    plot_pca_variance(adata)
    adata, use_rep = harmony_integration(adata, n_pcs=args.n_pcs)
    adata = leiden_clustering(
        adata, use_rep=use_rep, resolution=args.resolution,
        n_neighbors=args.n_neighbors, n_pcs=args.n_pcs,
    )

    stability_results = None
    if not args.skip_stability_check:
        stability_results = assess_clustering_stability(
            adata, use_rep=use_rep, resolution=args.resolution,
            n_neighbors=args.n_neighbors, n_pcs=args.n_pcs,
        )

    compute_cluster_patient_diagnostics(adata)
    compute_cluster_library_diagnostics(adata)
    adata = umap_visualization(adata, use_rep=use_rep)
    plot_quick_marker_sanity_check(adata)
    _record_clustering_metadata(adata, args, stability_results, use_rep)

    n_clusters = adata.obs["leiden_clusters"].nunique()
    print(
        f"\nLeiden found {n_clusters} clusters at resolution={args.resolution}, "
        f"n_neighbors={args.n_neighbors}, n_pcs={args.n_pcs} (using {use_rep} embedding)."
    )

    output_file = "data/processed_data.h5ad"
    adata.write(output_file)
    print(f"Saved clustered data (with leiden_clusters) to {output_file}")


if __name__ == "__main__":
    main()
