"""
Write a small synthetic dataset in the same 10x format as GSE203612, so
the whole pipeline can be run in a couple of minutes without downloading
anything.

Layout: 3 patients x 2 libraries, about 250 cells each, five cell types with
planted marker programs, plus a few empty droplets and high mitochondrial
cells for QC to remove. True labels go to truth.csv, keyed by the cell
names preprocess.py assigns ({patient}_{library}_cell{i}).

Usage (from the repo root):
    python example/make_toy_data.py                # writes example/toy_data/
    python example/make_toy_data.py --out some/dir --seed 1
"""

import argparse
import gzip
import os

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

# Cell types use the pipeline's own label names (scripts/annotate.py), so the
# check can compare them directly.
CELL_TYPE_MARKERS = {
    "Epithelial cell": ["EPCAM", "KRT8", "KRT18", "PAX8", "WFDC2"],
    "Stromal cell": ["COL1A1", "DCN", "RGS5", "ACTA2", "PDGFRB"],
    "T/NK cell": ["CD3D", "CD2", "GNLY", "NKG7"],
    "Myeloid/Monocyte": ["CD14", "CD68", "LYZ"],
    "B cell": ["MS4A1", "CD79A", "CD19"],
}
CELL_TYPE_FRACTIONS = [0.30, 0.15, 0.30, 0.15, 0.10]

# Other panel genes, present at background level as in real data.
OTHER_PANEL_GENES = ["PECAM1", "VWF", "CDH5", "MZB1", "JCHAIN", "XBP1", "ITGAX", "CD1C", "CLEC9A",
                     "FCGR3B", "CSF3R", "S100A8"]
MT_GENES = ["MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6", "MT-CO3", "MT-ND3",
            "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6", "MT-CYB"]

N_FILLER_GENES = 1500
PROGRAM_GENES_PER_TYPE = 25   # extra unnamed genes raised with each type, so HVG selection has signal
PATIENTS = ["TOYP1", "TOYP2", "TOYP3"]
LIBRARIES = ["lib1", "lib2"]
CELLS_PER_LIBRARY = 250
EMPTY_DROPLETS_PER_LIBRARY = 15
HIGH_MITO_PER_LIBRARY = 10


def build_gene_table(rng):
    marker_genes = [g for genes in CELL_TYPE_MARKERS.values() for g in genes]
    program_genes = {ct: [f"PROG{i}_{j}" for j in range(PROGRAM_GENES_PER_TYPE)]
                     for i, ct in enumerate(CELL_TYPE_MARKERS)}
    filler = [f"GENE{i:04d}" for i in range(N_FILLER_GENES)]
    symbols = marker_genes + OTHER_PANEL_GENES + MT_GENES + [g for gs in program_genes.values() for g in gs] + filler
    base_rate = rng.gamma(shape=0.6, scale=0.5, size=len(symbols))   # background mean for each gene
    return symbols, base_rate, program_genes


def simulate_library(rng, symbols, base_rate, program_genes, patient_shift):
    gene_index = {g: i for i, g in enumerate(symbols)}
    mt_idx = [gene_index[g] for g in MT_GENES]
    n_real = CELLS_PER_LIBRARY
    types = rng.choice(list(CELL_TYPE_MARKERS), size=n_real, p=CELL_TYPE_FRACTIONS)

    rates = np.tile(base_rate, (n_real, 1)) * patient_shift
    rates[:, mt_idx] = 2.0                                           # ~5% mito in healthy cells
    for ct in CELL_TYPE_MARKERS:
        rows = np.where(types == ct)[0]
        for g in CELL_TYPE_MARKERS[ct]:
            rates[np.ix_(rows, [gene_index[g]])] += rng.uniform(6, 12)
        for g in program_genes[ct]:
            rates[np.ix_(rows, [gene_index[g]])] += rng.uniform(2, 5)

    size_factor = rng.lognormal(mean=0.0, sigma=0.3, size=(n_real, 1))
    counts = rng.poisson(rates * size_factor)

    # Empty droplets: a handful of reads spread over few genes (removed by --min-genes).
    empty = rng.poisson(np.tile(base_rate * 0.02, (EMPTY_DROPLETS_PER_LIBRARY, 1)))
    # Damaged cells: normal profile plus very high mitochondrial reads (removed by --max-pct-mt).
    damaged_rates = np.tile(base_rate, (HIGH_MITO_PER_LIBRARY, 1))
    damaged_rates[:, mt_idx] = 60.0
    damaged = rng.poisson(damaged_rates)

    all_counts = np.vstack([counts, empty, damaged])
    labels = list(types) + ["empty droplet"] * EMPTY_DROPLETS_PER_LIBRARY + ["damaged"] * HIGH_MITO_PER_LIBRARY
    order = rng.permutation(len(labels))
    return all_counts[order], [labels[i] for i in order]


def write_library(out_dir, patient, library, counts, symbols):
    stem = os.path.join(out_dir, f"{patient}_{library}")
    # 10x convention: genes x cells.
    with gzip.open(f"{stem}_gene_expression.mtx.gz", "wb") as f:
        scipy.io.mmwrite(f, sp.coo_matrix(counts.T.astype(np.int32)), field="integer")
    genes = pd.DataFrame({"id": [f"ENSGTOY{i:06d}" for i in range(len(symbols))], "symbol": symbols})
    genes.to_csv(f"{stem}_genes.tsv.gz", sep="\t", header=False, index=False, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_data"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    symbols, base_rate, program_genes = build_gene_table(rng)

    truth = []
    for p, patient in enumerate(PATIENTS):
        patient_shift = rng.lognormal(mean=0.0, sigma=0.15, size=len(symbols))  # mild batch effect for each patient
        for library in LIBRARIES:
            counts, labels = simulate_library(rng, symbols, base_rate, program_genes, patient_shift)
            write_library(args.out, patient, library, counts, symbols)
            truth += [{"cell": f"{patient}_{library}_cell{i}", "patient": patient, "library": library,
                       "true_type": lab} for i, lab in enumerate(labels)]

    truth_df = pd.DataFrame(truth)
    truth_df.to_csv(os.path.join(args.out, "truth.csv"), index=False)
    print(f"Wrote {len(PATIENTS) * len(LIBRARIES)} libraries, {len(truth_df)} barcodes x {len(symbols)} genes "
          f"to {args.out}")
    print(truth_df["true_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
