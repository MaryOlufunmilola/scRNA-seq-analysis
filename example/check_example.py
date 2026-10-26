"""
Check the toy run against the planted ground truth. Exits with an error if any
check fails, so CI catches a pipeline change that breaks the full run.

Checks (thresholds are deliberately loose: the point is "the pipeline works",
not exact numbers, which shift slightly across package versions):
  1. QC removed the empty droplets and damaged barcodes with high mitochondrial reads.
  2. Each planted cell type is recovered: >= 90% of its cells get the matching label.
  3. Classifier accuracy >= 0.85 for every patient when that patient is held out.

Usage: python example/check_example.py [--work example/work] [--truth example/toy_data/truth.csv]
"""

import argparse
import os
import sys

import anndata as ad
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_TYPE_RECOVERY = 0.90
MIN_LOPO_ACCURACY = 0.85
MAX_QC_LEAK = 0.02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=os.path.join(HERE, "work"))
    parser.add_argument("--truth", default=os.path.join(HERE, "toy_data", "truth.csv"))
    args = parser.parse_args()

    truth = pd.read_csv(args.truth).set_index("cell")
    adata = ad.read_h5ad(os.path.join(args.work, "data", "annotated_data.h5ad"))
    obs = adata.obs.join(truth, how="left")
    if obs["true_type"].isna().any():
        print("Some cells in the output are missing from truth.csv; were cell names changed?")
        return 1

    failures = []

    # 1. QC
    bad = truth["true_type"].isin(["empty droplet", "damaged"])
    leaked = obs["true_type"].isin(["empty droplet", "damaged"]).sum()
    leak_frac = leaked / bad.sum()
    print(f"QC: {bad.sum() - leaked}/{bad.sum()} empty or damaged barcodes removed")
    if leak_frac > MAX_QC_LEAK:
        failures.append(f"QC let through {leaked} empty/damaged barcodes ({leak_frac:.1%})")

    # 2. Annotation
    print("\nCell type recovery (fraction of each planted type given the matching label):")
    rows = []
    for true_type, group in obs.groupby("true_type"):
        recovered = (group["cell_type"].astype(str) == true_type).mean()
        rows.append({"planted type": true_type, "cells kept": len(group), "recovered": round(recovered, 3)})
        if recovered < MIN_TYPE_RECOVERY:
            top = group["cell_type"].astype(str).value_counts().head(3).to_dict()
            failures.append(f"{true_type}: only {recovered:.1%} recovered (labels given: {top})")
    print(pd.DataFrame(rows).to_string(index=False))

    # 3. Leave one patient out
    lopo_path = os.path.join(args.work, "results", "leave_one_patient_out.csv")
    lopo = pd.read_csv(lopo_path)
    print("\nLeave one patient out:")
    print(lopo[["held_out_patient", "accuracy", "macro_f1"]].round(3).to_string(index=False))
    for _, row in lopo.iterrows():
        if row["accuracy"] < MIN_LOPO_ACCURACY:
            failures.append(f"LOPO accuracy {row['accuracy']:.3f} for {row['held_out_patient']}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll toy example checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
