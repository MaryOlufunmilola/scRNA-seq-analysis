"""
Load, QC-filter, and preprocess GSE203612 endometrial carcinoma (UCEC) scRNA-seq
(Barkley et al., Nature Genetics 2022): three patients, six inDrop libraries.

Data tiers in the saved object:
  adata.layers["counts"]  raw counts, HVG genes
  adata.raw.X             log-normalized (CP10K + log1p), all post-QC genes
  adata.X                 log-normalized, HVG subset (analysis matrix)
Full-gene raw counts: expm1(adata.raw.X) * adata.obs["total_counts"] / 1e4
"""

import os
import re
import json
import glob
import gzip
import tarfile
import argparse
import urllib.request
import numpy as np
import pandas as pd
import scipy.io
import scanpy as sc
import anndata as ad
import scrublet as scr

from provenance import package_versions, verify_checksum

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "GSE203612_raw")

GEO_TAR_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE203nnn/GSE203612/suppl/GSE203612_RAW.tar"
GEO_TAR_PATH = os.path.join(DATA_DIR, "GSE203612_RAW.tar")

# SHA-256 of the GEO archive these results were produced with. 
GEO_TAR_SHA256 = "5178ae0bea3f2d95f7155aaba149a7e332499cc166b0d6c0f9e913f3a4db3a4d"

UCEC_GSM_IDS = {
    "GSM6177620": "NYU_UCEC1",
    "GSM6177621": "NYU_UCEC2",
    "GSM6177622": "NYU_UCEC3",
}


def select_ucec_members(members, gsm_ids=tuple(UCEC_GSM_IDS)):
    """Regular-file archive members belonging to the UCEC samples (by GEO sample ID prefix)."""
    return [m for m in members if m.isfile() and os.path.basename(m.name).startswith(tuple(gsm_ids))]


def safe_extract(tar_path, dest, member_selector=select_ucec_members):
    """
    Extract selected regular files from a tar archive, refusing any member
    whose path would land outside dest (absolute paths, "..") and skipping
    links and device files. Python's tarfile.extractall trusts member paths
    by default, so a malicious archive could otherwise write anywhere; the
    "data" extraction filter is also applied when this Python supports it.

    Returns the list of extracted member names.
    """
    dest_real = os.path.realpath(dest)
    os.makedirs(dest_real, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        selected = member_selector(tar.getmembers())
        for member in selected:
            target = os.path.realpath(os.path.join(dest_real, member.name))
            if os.path.isabs(member.name) or os.path.commonpath([dest_real, target]) != dest_real:
                raise ValueError(f"Refusing to extract '{member.name}': path escapes {dest}.")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest_real, members=selected, filter="data")
        else:
            tar.extractall(dest_real, members=selected)
    return [m.name for m in selected]


def _ucec_files_present(raw_dir=RAW_DIR):
    return all(glob.glob(os.path.join(raw_dir, f"{gsm}_*_gene_expression.mtx.gz")) for gsm in UCEC_GSM_IDS)


def download_geo_data():
    """
    Download the GEO archive if needed, verify its checksum, and extract only
    the UCEC libraries. Returns the archive's SHA-256.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(GEO_TAR_PATH):
        print(f"Downloading {GEO_TAR_URL} ...")
        urllib.request.urlretrieve(GEO_TAR_URL, GEO_TAR_PATH)
        print("Download complete.")
    else:
        print("GSE203612_RAW.tar already downloaded, skipping.")

    digest = verify_checksum(GEO_TAR_PATH, GEO_TAR_SHA256, "GSE203612_RAW.tar")

    if not _ucec_files_present():
        print("Extracting the UCEC libraries from the archive...")
        extracted = safe_extract(GEO_TAR_PATH, RAW_DIR)
        print(f"Extracted {len(extracted)} files to {RAW_DIR}")
    else:
        print(f"UCEC files already present in {RAW_DIR}, skipping extraction.")
    return digest


def _library_id_from_filename(mtx_path):
    matches = re.findall(r"lib\d+", os.path.basename(mtx_path))
    return matches[-1] if matches else "lib1"


def _read_genes_file(genes_path, expected_n):
    genes_df = pd.read_csv(genes_path, header=None, sep="\t")

    if len(genes_df) == expected_n:
        return genes_df

    genes_df_no_header = pd.read_csv(genes_path, header=0, sep="\t")
    if len(genes_df_no_header) == expected_n:
        print(
            f"  Note: {genes_path} appears to have a header row "
            f"(row count matched only after skipping it)."
        )
        return genes_df_no_header

    return genes_df


def _select_gene_symbol_and_id_columns(genes_df):
    """
    Pick the gene symbol and Ensembl ID columns from a genes table.

    Returns (gene_symbols, ensembl_ids_or_None).
    """
    symbol_col_names = ["name", "gene_name", "symbol", "gene_symbol"]
    id_col_names = ["ensembl_id", "gene_id", "ensembl", "id"]

    symbols, ensembl_ids = None, None
    if all(isinstance(c, str) for c in genes_df.columns):
        for cand in symbol_col_names:
            matches = [c for c in genes_df.columns if c.lower() == cand]
            if matches:
                symbols = genes_df[matches[0]].astype(str).values
                break
        for cand in id_col_names:
            matches = [c for c in genes_df.columns if c.lower() == cand]
            if matches:
                ensembl_ids = genes_df[matches[0]].astype(str).values
                break

    if symbols is None:
        if genes_df.shape[1] >= 2:
            symbols = genes_df.iloc[:, 1].astype(str).values
            ensembl_ids = genes_df.iloc[:, 0].astype(str).values
        else:
            symbols = genes_df.iloc[:, 0].astype(str).values

    return symbols, ensembl_ids


def _load_library(mtx_path, sample_id, library_id):
    genes_path = mtx_path.replace("_gene_expression.mtx.gz", "_genes.tsv.gz")
    if not os.path.exists(genes_path):
        raise FileNotFoundError(f"Expected matching genes file not found: {genes_path}")

    with gzip.open(mtx_path, "rb") as f:
        mat = scipy.io.mmread(f).tocsr()

    genes_df = _read_genes_file(genes_path, expected_n=max(mat.shape))
    genes, ensembl_ids = _select_gene_symbol_and_id_columns(genes_df)
    n_genes = len(genes)

    if mat.shape[0] == n_genes:
        X = mat.T.tocsr()
        n_cells = mat.shape[1]
    elif mat.shape[1] == n_genes:
        X = mat.tocsr()
        n_cells = mat.shape[0]
    else:
        raise ValueError(
            f"Matrix shape {mat.shape} in {mtx_path} doesn't match gene count "
            f"{n_genes} on either axis -- inspect this file directly."
        )

    adata = ad.AnnData(X=X)
    adata.var_names = genes
    if ensembl_ids is not None:
        adata.var["ensembl_id"] = ensembl_ids
    adata.obs_names = [f"{sample_id}_{library_id}_cell{i}" for i in range(n_cells)]
    adata.obs["sample_id"] = sample_id
    adata.obs["library_id"] = library_id
    adata.var_names_make_unique()
    return adata


def load_data():
    archive_sha256 = download_geo_data()

    adatas = []
    for gsm_id, sample_label in UCEC_GSM_IDS.items():
        pattern = os.path.join(RAW_DIR, f"{gsm_id}_*_gene_expression.mtx.gz")
        mtx_files = sorted(glob.glob(pattern))
        if not mtx_files:
            raise FileNotFoundError(
                f"No '*_gene_expression.mtx.gz' files found for {gsm_id} ({sample_label}) "
                f"in {RAW_DIR}. Run `ls {RAW_DIR} | grep {gsm_id}` to see what's actually there."
            )

        for mtx_path in mtx_files:
            library_id = _library_id_from_filename(mtx_path)
            print(f"Loading {sample_label} / {library_id} from {os.path.basename(mtx_path)} ...")
            adatas.append(_load_library(mtx_path, sample_label, library_id))

    adata = ad.concat(adatas, join="outer", fill_value=0)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.uns["geo_archive_sha256"] = archive_sha256
    return adata


def detect_doublets(adata, expected_doublet_rate=0.06, per_sample_rate_overrides=None,
                     remove_predicted_doublets=True):
    """Run Scrublet separately per library and concatenate the results."""
    per_sample_rate_overrides = per_sample_rate_overrides or {}
    filtered = []

    grouping_keys = adata.obs[["sample_id", "library_id"]].drop_duplicates().values.tolist()

    for sample_id, library_id in grouping_keys:
        group_mask = ((adata.obs["sample_id"] == sample_id) & (adata.obs["library_id"] == library_id)).values
        sub = adata[group_mask].copy()

        sample_rate = per_sample_rate_overrides.get(sample_id, expected_doublet_rate)
        if sample_id in per_sample_rate_overrides:
            print(f"  {sample_id}/{library_id}: using override expected_doublet_rate={sample_rate} (default was {expected_doublet_rate})")

        scrub = scr.Scrublet(sub.X, expected_doublet_rate=sample_rate)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(
            min_counts=2, min_cells=3, min_gene_variability_pctl=85, n_prin_comps=min(30, sub.n_obs - 1)
        )
        sub.obs["doublet_score"] = doublet_scores
        sub.obs["predicted_doublet"] = predicted_doublets
        sub.obs["scrublet_threshold"] = scrub.threshold_
        n_doublets = int(predicted_doublets.sum())
        print(
            f"  {sample_id}/{library_id}: {n_doublets} likely doublets (probabilistic Scrublet call) "
            f"out of {sub.n_obs} cells (Scrublet threshold: {scrub.threshold_:.4f})."
        )

        if remove_predicted_doublets:
            filtered.append(sub[~sub.obs["predicted_doublet"], :].copy())
        else:
            filtered.append(sub)

    return ad.concat(filtered, join="outer", fill_value=0)


def preprocess_data(adata, qc_dir="results", min_genes=100, max_genes=6000, max_pct_mt=15,
                     expected_doublet_rate=0.06, per_sample_doublet_rate_overrides=None,
                     remove_predicted_doublets=True):
    os.makedirs(os.path.join(qc_dir, "figures"), exist_ok=True)
    stage_counts = [{"stage": "raw (loaded)", "n_cells": adata.n_obs}]

    raw_snapshot = adata.copy()
    raw_snapshot.var["mt"] = raw_snapshot.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(raw_snapshot, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    qc_before = raw_snapshot.obs[["sample_id", "library_id", "n_genes_by_counts", "pct_counts_mt"]].copy()
    qc_before["stage"] = "before QC"

    print("\nPre-filter distribution percentiles, GLOBAL (use these to judge threshold choices):")
    print("  n_genes_by_counts:", {
        f"{p}th": round(float(np.percentile(qc_before["n_genes_by_counts"], p)), 1)
        for p in [1, 5, 10, 25, 50, 75, 90]
    })
    print("  pct_counts_mt:", {
        f"{p}th": round(float(np.percentile(qc_before["pct_counts_mt"], p)), 2)
        for p in [50, 75, 90, 95, 99]
    })

    # Per-library view: a global threshold can hide libraries with different
    # loading density or sequencing depth.
    print("\nPre-filter distribution percentiles BY LIBRARY:")
    for (sample_id, library_id), group in qc_before.groupby(["sample_id", "library_id"], observed=True):
        print(
            f"  {sample_id}/{library_id} (n={len(group)}): "
            f"genes median={np.percentile(group['n_genes_by_counts'], 50):.0f}, "
            f"mito% median={np.percentile(group['pct_counts_mt'], 50):.2f}"
        )
    del raw_snapshot

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=3)
    stage_counts.append({"stage": f"after min_genes={min_genes}/min_cells filter", "n_cells": adata.n_obs})

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < max_genes, :].copy()
    adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :].copy()
    stage_counts.append({"stage": f"after max_genes={max_genes}/max_pct_mt={max_pct_mt} filter", "n_cells": adata.n_obs})

    adata = detect_doublets(
        adata, expected_doublet_rate=expected_doublet_rate,
        per_sample_rate_overrides=per_sample_doublet_rate_overrides,
        remove_predicted_doublets=remove_predicted_doublets,
    )
    stage_counts.append({"stage": "after doublet removal", "n_cells": adata.n_obs})

    qc_after = adata.obs[["sample_id", "n_genes_by_counts", "pct_counts_mt", "doublet_score"]].copy()
    qc_after["stage"] = "after QC"

    _save_qc_report(stage_counts, qc_before, qc_after, qc_dir)

    # Keep raw counts before normalization overwrites X (needed by count-based
    # methods and for export to other tools).
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key="sample_id"
    )

    # Snapshot all genes (log-normalized) before HVG subsetting; marker lookups
    # and CellTypist downstream need the full gene set.
    adata.raw = adata.copy()
    adata = adata[:, adata.var.highly_variable].copy()

    print(
        "\nData tiers in the returned object: "
        "adata.layers['counts'] = raw integer counts (HVG genes, matching adata.X's gene set); "
        "adata.raw.X = log-normalized, ALL post-QC genes; "
        "adata.X = log-normalized, HVG-subsetted analysis matrix. "
        "See this module's docstring for how to reconstruct full-gene raw counts if ever needed."
    )

    return adata


def _save_qc_report(stage_counts, qc_before, qc_after, qc_dir):
    stage_df = pd.DataFrame(stage_counts)
    stage_df["cells_removed"] = stage_df["n_cells"].shift(1) - stage_df["n_cells"]
    stage_df["pct_removed_this_stage"] = (
        100 * stage_df["cells_removed"] / stage_df["n_cells"].shift(1)
    ).round(2)
    stage_df.to_csv(os.path.join(qc_dir, "qc_cell_counts_by_stage.csv"), index=False)
    print("\nCell counts by QC stage:")
    print(stage_df.to_string(index=False))

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, title in zip(
        axes, ["n_genes_by_counts", "pct_counts_mt"],
        ["Genes detected per cell", "Mitochondrial %"]
    ):
        combined = pd.concat([
            qc_before[["stage", metric]],
            qc_after[["stage", metric]],
        ])
        sns.violinplot(data=combined, x="stage", y=metric, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "figures", "qc_before_after.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=qc_after, x="sample_id", y="doublet_score", ax=ax)
    ax.set_title("Post-QC doublet score by patient sample")
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "figures", "qc_doublet_score_by_sample.png"), dpi=150)
    plt.close(fig)

    print(f"Saved QC report to {qc_dir}/qc_cell_counts_by_stage.csv and {qc_dir}/figures/qc_*.png")


def record_preprocess_params(adata, args, overrides, n_loaded, n_libraries, archive_sha256, results_dir="results"):
    """
    Store the input archive checksum, QC settings, and software versions in
    adata.uns["preprocess_params"] and results/preprocess_params.json.
    """
    params = {
        "geo_archive_url": GEO_TAR_URL,
        "geo_archive_sha256": archive_sha256,
        "gsm_ids": dict(UCEC_GSM_IDS),
        "n_libraries": int(n_libraries),
        "n_cells_loaded": int(n_loaded),
        "n_cells_after_qc": int(adata.n_obs),
        "min_genes": args.min_genes,
        "max_genes": args.max_genes,
        "max_pct_mt": args.max_pct_mt,
        "expected_doublet_rate": args.doublet_rate,
        "doublet_rate_overrides": ",".join(f"{k}={v}" for k, v in overrides.items()) or "none",
        "remove_predicted_doublets": not args.keep_predicted_doublets,
        "software": package_versions(["scanpy", "anndata", "scrublet", "numpy", "scipy", "pandas"]),
    }
    adata.uns["preprocess_params"] = params
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "preprocess_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"Recorded input checksum, QC settings, and versions to {results_dir}/preprocess_params.json")
    return params


def count_libraries(obs):
    """Number of distinct sequencing libraries. library_id repeats across patients, so count (sample_id, library_id) pairs."""
    return int(obs[["sample_id", "library_id"]].drop_duplicates().shape[0])


def save_data(adata, output_file):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    adata.write(output_file)


def main():
    parser = argparse.ArgumentParser(description="Load, QC filter, and preprocess GSE203612 data.")
    parser.add_argument("--min-genes", type=int, default=100,
                         help="Minimum genes detected per cell (default 100: above the empty-droplet peaks "
                              "in the per-library log-scale histograms).")
    parser.add_argument("--max-genes", type=int, default=6000,
                         help="Maximum genes detected per cell, an unusually high count is a doublet signal (default 6000).")
    parser.add_argument("--max-pct-mt", type=float, default=15,
                         help="Maximum mitochondrial %% per cell (default 15; check results/figures/qc_before_after.png "
                              "to judge whether a tighter bound like 10 is appropriate for this data).")
    parser.add_argument("--doublet-rate", type=float, default=0.06,
                         help="Default expected_doublet_rate passed to Scrublet for all samples (default 0.06).")
    parser.add_argument("--doublet-rate-overrides", type=str, default="",
                         help="Per-PATIENT overrides as 'sample1=rate1,sample2=rate2', e.g. "
                              "'NYU_UCEC2=0.15', applied to every library belonging to that patient. "
                              "Worth using when a patient's own Scrublet-ESTIMATED rate (printed each "
                              "run) differs substantially from the shared default.")
    parser.add_argument("--keep-predicted-doublets", action="store_true",
                         help="Keep cells Scrublet flags as likely doublets (still labeled via "
                              "predicted_doublet/doublet_score columns) instead of removing them. "
                              "Scrublet's call is a probabilistic estimate, not a definitive label; "
                              "use this to treat it as soft evidence downstream instead.")
    args = parser.parse_args()

    overrides = {}
    if args.doublet_rate_overrides:
        for pair in args.doublet_rate_overrides.split(","):
            sample, rate = pair.split("=")
            overrides[sample.strip()] = float(rate)

    print("Loading GSE203612 endometrial (UCEC) tumor samples...")
    adata = load_data()
    n_libraries = count_libraries(adata.obs)
    n_loaded = int(adata.n_obs)
    archive_sha256 = adata.uns.get("geo_archive_sha256")  # captured now: concat in doublet detection drops .uns
    n_patients = adata.obs["sample_id"].nunique()
    print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes across {n_patients} patients, {n_libraries} libraries.")

    adata = preprocess_data(
        adata,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        max_pct_mt=args.max_pct_mt,
        expected_doublet_rate=args.doublet_rate,
        per_sample_doublet_rate_overrides=overrides,
        remove_predicted_doublets=not args.keep_predicted_doublets,
    )
    print(f"After QC, doublet removal, and preprocessing: {adata.n_obs} cells x {adata.n_vars} genes.")

    record_preprocess_params(adata, args, overrides, n_loaded=n_loaded, n_libraries=n_libraries,
                             archive_sha256=archive_sha256)

    output_file = os.path.join(DATA_DIR, "processed_data.h5ad")
    save_data(adata, output_file)
    print(f"Saved processed data to {output_file}")


if __name__ == "__main__":
    main()
