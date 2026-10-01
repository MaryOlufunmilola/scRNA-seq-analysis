"""
Train a neural network to predict marker-panel cell types from PCA-reduced
expression, evaluate it, and attribute its predictions to genes.

Evaluation: a random cell-level split is optimistic (cells of one cluster land on
both sides); leave-one-patient-out is the informative test of label transfer. PCA is
fit on training cells only. Gene attributions use Integrated Gradients through the
PCA projection, with the average training cell as baseline.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from labels import real_cell_type_mask

MODEL_DIR = "models"
RESULTS_DIR = "results"
N_PCS = 20  # matches clustering.py's default n_pcs
SEED = 42
MIN_CELLS_PER_CLASS = 5

# Dissociation-induced stress/immediate-early genes (van den Brink et al.,
# Nature Methods 2017).
DISSOCIATION_STRESS_GENES = {
    "FOS", "FOSB", "JUN", "JUNB", "JUND", "ATF3", "EGR1", "EGR2",
    "IER2", "IER3", "IER5", "DUSP1", "DUSP2",
    "HSPA1A", "HSPA1B", "HSPA6", "HSP90AA1", "HSPB1",
    "NFKBIA", "NFKBIZ", "ZFP36", "ZFP36L1", "ZFP36L2",
    "KLF2", "KLF4", "KLF6", "CXCL1", "CXCL2", "CXCL3",
    "SOCS3", "BTG2", "PPP1R15A", "RGS1", "RGS2",
    "NR4A1", "NR4A2", "NR4A3", "CD69", "GADD45B",
}

# Core cell-cycle genes, S and G2M phase (Tirosh et al., Science 2016), the same reference list
# underlying Seurat's CellCycleScoring and scanpy's sc.tl.score_genes_cell_cycle.
CELL_CYCLE_GENES = {
    # S phase
    "MCM5", "PCNA", "TYMS", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS",
    "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP",
    "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2",
    "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2",
    "USP1", "CLSPN", "POLA1", "CHAF1B", "E2F8",
    # G2M phase
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
}


class CellTypePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CellTypePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Raw logits: CrossEntropyLoss applies softmax itself.
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PCAProjectedModel(nn.Module):
    """
    Wraps a classifier trained on PCA coordinates so it takes gene
    expression as input: logits = model((x - pca_mean) @ loadings).
    Used to attribute predictions to genes rather than to PCs.
    """

    def __init__(self, model, loadings, pca_mean):
        super().__init__()
        self.model = model
        self.register_buffer("loadings", torch.as_tensor(np.asarray(loadings), dtype=torch.float32))
        self.register_buffer("pca_mean", torch.as_tensor(np.asarray(pca_mean), dtype=torch.float32))

    def forward(self, x):
        return self.model((x - self.pca_mean) @ self.loadings)


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, train_loader, criterion, optimizer, epochs=20, verbose=True):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | loss: {epoch_loss / len(train_loader):.4f}")


def train_classifier(X_train, y_train, n_classes, hidden_dim=128, epochs=20, lr=1e-3, batch_size=32,
                     seed=SEED, verbose=True):
    """Seeded training of a CellTypePredictor on (X_train, y_train); returns the model in eval mode."""
    set_seed(seed)
    X_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y_train), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, generator=generator
    )
    model = CellTypePredictor(X_t.shape[1], hidden_dim, n_classes)
    train_model(model, loader, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=lr), epochs=epochs,
                verbose=verbose)
    model.eval()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(np.asarray(X), dtype=torch.float32)).argmax(dim=1).numpy()


def evaluate_model(model, test_loader, class_names, figure_name="confusion_matrix.png"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    label_ids = np.arange(len(class_names))

    report = classification_report(
        all_labels, all_preds, labels=label_ids, target_names=list(class_names), zero_division=0
    )
    print(report)

    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)
    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True (marker-gene-derived label)")
    plt.title("Cell Type Prediction vs. Marker-Gene-Derived Label")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, figure_name), dpi=150)
    plt.close(fig)

    return report


def fit_train_only_pca(X_expr_train, X_expr_all, n_components=N_PCS):
    """
    Fit PCA on the training cells only and project all cells through that
    basis, so test cells never influence the feature space.

    Returns (X_pca_train, X_pca_all, loadings); loadings is genes x
    n_components. The PCA mean is the column mean of X_expr_train.
    """
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca_train = pca.fit_transform(X_expr_train)
    X_pca_all = pca.transform(X_expr_all)
    loadings = pca.components_.T
    return X_pca_train, X_pca_all, loadings


def leave_one_patient_out(X_expr, y, groups, class_names, n_components=N_PCS, epochs=20, seed=SEED):
    """
    For each patient: fit PCA and train on the other patients' cells, then
    predict the held-out patient's cells.

    Returns a DataFrame with one row per held-out patient: n_train, n_test,
    accuracy, macro-F1 over the classes present in that patient, the
    classes present in the patient but absent from training (these cannot
    be predicted correctly), and per-class F1 with the number of held-out
    and training cells per class. Macro-F1 weights every class equally, so
    one small, poorly predicted class can pull it well below accuracy; the
    per-class columns show which class is responsible.
    """
    X_expr = np.asarray(X_expr)
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_classes = len(class_names)
    records = []

    for held_out in sorted(pd.unique(groups)):
        test = groups == held_out
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue
        n_comp = min(n_components, int(train.sum()) - 1, X_expr.shape[1])
        X_tr, X_all, _ = fit_train_only_pca(X_expr[train], X_expr, n_components=n_comp)
        model = train_classifier(X_tr, y[train], n_classes, epochs=epochs, seed=seed, verbose=False)
        preds = predict(model, X_all[test])

        present = np.unique(y[test])
        missing = sorted(set(present) - set(np.unique(y[train])))
        per_class_f1 = f1_score(y[test], preds, labels=present, average=None, zero_division=0)
        records.append({
            "held_out_patient": held_out,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "accuracy": float(accuracy_score(y[test], preds)),
            "macro_f1": float(f1_score(y[test], preds, labels=present, average="macro", zero_division=0)),
            "classes_missing_from_training": [class_names[i] for i in missing],
            "per_class_f1": {class_names[c]: round(float(f), 3) for c, f in zip(present, per_class_f1)},
            "n_test_per_class": {class_names[c]: int((y[test] == c).sum()) for c in present},
            "n_train_per_class": {class_names[c]: int((y[train] == c).sum()) for c in present},
        })

    return pd.DataFrame(records)


def integrated_gradients_attributions(model, class_idx, X, baseline=None, n_steps=50):
    """Per-cell Integrated Gradients attributions for one class logit."""
    model.eval()
    X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if baseline is None:
        base = torch.zeros_like(X_tensor)
    else:
        base = torch.as_tensor(np.asarray(baseline), dtype=torch.float32).expand_as(X_tensor)

    total_grads = torch.zeros_like(X_tensor)
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interpolated = (base + alpha * (X_tensor - base)).detach().requires_grad_(True)
        class_logit = model(interpolated)[:, class_idx].sum()
        grads = torch.autograd.grad(class_logit, interpolated)[0]
        total_grads += grads

    return ((X_tensor - base) * total_grads / n_steps).detach().numpy()


def integrated_gradients(model, class_idx, X_class, n_steps=50):
    """Mean |IG attribution| per input dimension across the given cells (zero baseline); None if no cells."""
    if len(X_class) == 0:
        return None
    attributions = integrated_gradients_attributions(model, class_idx, X_class, baseline=None, n_steps=n_steps)
    return np.abs(attributions).mean(axis=0)


def gene_level_attributions(model, loadings, pca_mean, X_expr_class, class_idx, n_steps=50,
                            max_cells=500, seed=SEED):
    """
    Mean signed IG attribution per gene for one class, computed through the
    PCA projection with the average training cell (pca_mean) as baseline.
    Subsamples to max_cells cells for speed. Returns an (n_genes,) array, or
    None if there are no cells.
    """
    X_expr_class = np.asarray(X_expr_class)
    if len(X_expr_class) == 0:
        return None
    if len(X_expr_class) > max_cells:
        idx = np.random.default_rng(seed).choice(len(X_expr_class), size=max_cells, replace=False)
        X_expr_class = X_expr_class[idx]
    wrapped = PCAProjectedModel(model, loadings, pca_mean)
    attributions = integrated_gradients_attributions(wrapped, class_idx, X_expr_class, baseline=pca_mean,
                                                     n_steps=n_steps)
    return attributions.mean(axis=0)


def _is_mt_gene(gene_name):
    return gene_name.upper().startswith("MT-")


def discover_data_driven_markers(model, adata, le, loadings, pca_mean, X_expr, y_all, top_n_genes=15,
                                 exclude_confounders=True):
    """
    Rank genes per cell type by mean signed gene-level IG attribution,
    excluding confounder genes, and annotate each
    candidate with classical DE statistics (Wilcoxon on all genes in
    adata.raw), an ambient-RNA heuristic, doublet-score correlation, and
    per-patient detection.
    """
    gene_names = adata.var_names.values
    X_full = np.asarray(X_expr)

    if exclude_confounders:
        excluded_mask = np.array([
            g in DISSOCIATION_STRESS_GENES or g in CELL_CYCLE_GENES or _is_mt_gene(g)
            for g in gene_names
        ])
        print(f"Excluding {int(excluded_mask.sum())} known confounder genes "
              f"(dissociation-stress, cell-cycle, mitochondrial) from the candidate pool.")
    else:
        excluded_mask = np.zeros(len(gene_names), dtype=bool)

    try:
        from annotate import MARKER_GENES
    except ImportError:
        print("Could not import MARKER_GENES from annotate.py -- proceeding without curated-panel cross-reference.")
        MARKER_GENES = {}

    print("Computing differential-expression statistics (Wilcoxon, all genes) for cross-validation...")
    de_adata = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    de_adata.obs["cell_type"] = adata.obs["cell_type"].values
    sc.tl.rank_genes_groups(de_adata, groupby="cell_type", method="wilcoxon", pts=True)

    has_sample_id = "sample_id" in adata.obs.columns
    patient_ids = adata.obs["sample_id"].unique().tolist() if has_sample_id else []
    if not has_sample_id:
        print("No 'sample_id' column found -- skipping cross-donor consistency check.")
    has_doublet_score = "doublet_score" in adata.obs.columns

    per_class_means = [X_full[y_all == ci].mean(axis=0) for ci in range(len(le.classes_))]

    discovery_records = []
    for class_idx, class_name in enumerate(le.classes_):
        class_mask = (y_all == class_idx)
        gene_scores = gene_level_attributions(model, loadings, pca_mean, X_full[class_mask], class_idx)
        if gene_scores is None:
            continue
        gene_scores = np.where(excluded_mask, -np.inf, gene_scores)
        top_idx = np.argsort(gene_scores)[::-1][:top_n_genes]
        curated_genes_for_class = {g for g, ct in MARKER_GENES.items() if ct == class_name}
        de_df = sc.get.rank_genes_groups_df(de_adata, group=class_name).set_index("names")

        for rank, gi in enumerate(top_idx, start=1):
            gene = gene_names[gi]

            log2fc = pval_adj = pct_in = pct_out = np.nan
            if gene in de_df.index:
                row = de_df.loc[gene]
                log2fc = float(row["logfoldchanges"])
                pval_adj = float(row["pvals_adj"])
                pct_in = float(row["pct_nz_group"]) if "pct_nz_group" in row else np.nan
                pct_out = float(row["pct_nz_reference"]) if "pct_nz_reference" in row else np.nan

            # Ambient-RNA heuristic: a gene specific to this class should be near
            # silent in every other class.
            target_mean = per_class_means[class_idx][gi]
            others = [m[gi] for i, m in enumerate(per_class_means) if i != class_idx]
            min_other_mean = min(others) if others else 0.0
            broad_expression_ratio = float(min_other_mean / target_mean) if target_mean > 0 else np.nan
            possible_ambient = bool(broad_expression_ratio > 0.5) if not np.isnan(broad_expression_ratio) else False

            doublet_corr = np.nan
            if has_doublet_score:
                try:
                    doublet_corr, _ = pearsonr(X_full[:, gi], adata.obs["doublet_score"].values)
                except Exception:
                    pass

            patient_consistent, patient_detail = None, {}
            if has_sample_id:
                for pid in patient_ids:
                    p_mask = class_mask & (adata.obs["sample_id"].values == pid)
                    if p_mask.sum() > 0:
                        patient_detail[pid] = round(float((X_full[p_mask, gi] > 0).mean()), 3)
                detected_in = sum(1 for v in patient_detail.values() if v > 0.1)
                patient_consistent = bool(detected_in == len(patient_detail)) if patient_detail else None

            discovery_records.append({
                "cell_type": class_name,
                "rank": rank,
                "gene": gene,
                "mean_ig_attribution": float(gene_scores[gi]),
                "in_curated_panel": gene in curated_genes_for_class,
                "de_log2fc": log2fc,
                "de_pval_adj": pval_adj,
                "pct_expressing_in_group": pct_in,
                "pct_expressing_out_group": pct_out,
                "possible_ambient_flag": possible_ambient,
                "broad_expression_ratio": round(broad_expression_ratio, 3) if not np.isnan(broad_expression_ratio) else None,
                "doublet_score_correlation": round(float(doublet_corr), 3) if not np.isnan(doublet_corr) else None,
                "consistent_across_patients": patient_consistent,
                "per_patient_pct_expressing": patient_detail,
            })

    discovery_df = pd.DataFrame(discovery_records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    discovery_df.to_csv(os.path.join(RESULTS_DIR, "data_driven_marker_discovery.csv"), index=False)

    print(f"\nData-driven marker discovery (top {top_n_genes} genes per cell type by signed gene-level IG):")
    for class_name in le.classes_:
        class_rows = discovery_df[discovery_df["cell_type"] == class_name] if len(discovery_df) else discovery_df
        if len(class_rows) == 0:
            continue
        known = class_rows[class_rows["in_curated_panel"]]["gene"].tolist()
        strong_novel = class_rows[
            (~class_rows["in_curated_panel"])
            & (class_rows["de_pval_adj"] < 0.05)
            & (class_rows["de_log2fc"] > 0)
            & (~class_rows["possible_ambient_flag"])
        ]["gene"].tolist()[:5]
        print(f"  {class_name}:")
        print(f"    curated-panel genes in top {top_n_genes}: {known if known else 'none'}")
        print(f"    novel candidates (up-regulated by DE, not ambient-flagged): {strong_novel if strong_novel else 'none'}")

    print(f"\nFull results saved to {RESULTS_DIR}/data_driven_marker_discovery.csv")
    return discovery_df


def select_trainable_cells(adata, min_cells_per_class=MIN_CELLS_PER_CLASS):
    """
    Restrict to cells with a real cell type (no Ambiguous/Unknown/doublet)
    in classes with at least min_cells_per_class cells.
    """
    keep = real_cell_type_mask(adata.obs["cell_type"])
    labels = adata.obs["cell_type"].astype(str)
    counts = labels[keep].value_counts()
    too_small = set(counts[counts < min_cells_per_class].index)
    keep &= ~labels.isin(too_small).values

    excluded = labels[~keep].value_counts().to_dict()
    if excluded:
        print(f"Excluding {int((~keep).sum())} cells from classifier training: {excluded}")
    subset = adata[keep].copy()
    subset.obs["cell_type"] = subset.obs["cell_type"].astype(str).astype("category")
    return subset


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    input_file = "data/annotated_data.h5ad"
    print(f"Loading {input_file}...")
    adata = sc.read(input_file)
    if "cell_type" not in adata.obs:
        raise KeyError(
            "'cell_type' not found in adata.obs. "
            "Run scripts/clustering.py then scripts/annotate.py before this script."
        )

    adata = select_trainable_cells(adata)
    X_expr = adata.X
    X_expr = np.asarray(X_expr.todense()) if hasattr(X_expr, "todense") else np.asarray(X_expr)

    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"])
    class_names = list(le.classes_)

    # 1. Random cell-level split (optimistic; see module docstring).
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    print(f"Fitting PCA ({N_PCS} components) on the {len(train_idx)}-cell training split only...")
    X_pca_train, X_pca_all, loadings = fit_train_only_pca(X_expr[train_idx], X_expr, n_components=N_PCS)
    pca_mean = X_expr[train_idx].mean(axis=0)

    model = train_classifier(X_pca_train, y[train_idx], len(class_names), seed=SEED)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_pca_all[test_idx], dtype=torch.float32),
            torch.tensor(y[test_idx], dtype=torch.long),
        ),
        batch_size=32, shuffle=False,
    )
    print("\nRandom cell-level split (optimistic -- same clusters on both sides):")
    report = evaluate_model(model, test_loader, class_names=class_names)

    # 2. Leave-one-patient-out.
    lopo_text = ""
    if "sample_id" in adata.obs.columns and adata.obs["sample_id"].nunique() >= 2:
        print("\nLeave-one-patient-out evaluation (train on the other patients, test on the held-out one)...")
        lopo = leave_one_patient_out(X_expr, y, adata.obs["sample_id"].astype(str).values, class_names)
        lopo.to_csv(os.path.join(RESULTS_DIR, "leave_one_patient_out.csv"), index=False)
        summary_cols = ["held_out_patient", "n_train", "n_test", "accuracy", "macro_f1", "classes_missing_from_training"]
        lopo_text = lopo[summary_cols].to_string(index=False)

        per_class = pd.DataFrame([
            {"held_out_patient": row["held_out_patient"], "cell_type": ct, "f1": f1,
             "n_test": row["n_test_per_class"][ct], "n_train": row["n_train_per_class"][ct]}
            for _, row in lopo.iterrows() for ct, f1 in row["per_class_f1"].items()
        ])
        per_class.to_csv(os.path.join(RESULTS_DIR, "leave_one_patient_out_per_class.csv"), index=False)
        per_class_text = per_class.pivot(index="cell_type", columns="held_out_patient", values="f1").round(3).to_string()
        lopo_text += "\n\nPer-class F1 by held-out patient (see leave_one_patient_out_per_class.csv for cell counts):\n"
        lopo_text += per_class_text
        print(lopo_text)
        print(f"Mean accuracy across held-out patients: {lopo['accuracy'].mean():.3f}; "
              f"mean macro-F1: {lopo['macro_f1'].mean():.3f}")
    else:
        print("Fewer than two patients -- skipping leave-one-patient-out evaluation.")

    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write("Random cell-level split (optimistic: cells from the same clusters in train and test)\n\n")
        f.write(report)
        if lopo_text:
            f.write("\n\nLeave-one-patient-out evaluation\n\n")
            f.write(lopo_text + "\n")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "cell_type_predictor.pth"))
    print(f"Saved trained model to {MODEL_DIR}/cell_type_predictor.pth and report to {MODEL_DIR}/classification_report.txt")

    print("\nRunning data-driven marker discovery on the trained model...")
    discover_data_driven_markers(model, adata, le, loadings, pca_mean, X_expr, y)


if __name__ == "__main__":
    main()
