"""Shared fixtures: small synthetic AnnData objects, so tests run fast and offline."""

import numpy as np
import pandas as pd
import anndata as ad
import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Scripts write results/ relative to the working directory; keep it out of the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def synthetic_adata():
    """200 cells x 500 genes of Poisson counts, with one boosted marker gene per block of 50 cells."""
    rng = np.random.default_rng(seed=0)
    n_cells, n_genes = 200, 500

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    marker_genes = ["CD3D", "MS4A1", "CD14", "FCGR3A", "GNLY", "NKG7", "PPBP", "MT-CO1"]
    for i, gene in enumerate(marker_genes):
        gene_names[i] = gene

    counts = rng.poisson(lam=1.5, size=(n_cells, n_genes)).astype(np.float32)

    block_size = n_cells // 4
    counts[0:block_size, gene_names.index("CD3D")] += 20        # T cells
    counts[block_size:2*block_size, gene_names.index("MS4A1")] += 20   # B cells
    counts[2*block_size:3*block_size, gene_names.index("CD14")] += 20  # Monocytes
    counts[3*block_size:, gene_names.index("GNLY")] += 20       # NK cells

    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=counts, obs=obs, var=var)

    # Two patients, three (patient, library) pairs; "lib1" is reused across patients as in the real data.
    half = n_cells // 2
    adata.obs["sample_id"] = ["PATIENT_A"] * half + ["PATIENT_B"] * (n_cells - half)
    quarter = n_cells // 4
    adata.obs["library_id"] = (
        ["lib1"] * quarter + ["lib2"] * (half - quarter) + ["lib1"] * (n_cells - half)
    )

    return adata


@pytest.fixture
def synthetic_adata_clustered(synthetic_adata):
    """Synthetic data with raw, PCA, neighbors, and leiden_clusters, for downstream tests."""
    import scanpy as sc

    adata = synthetic_adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    assert adata.var["highly_variable"].sum() >= 100
    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.tl.pca(adata, n_comps=20)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden_clusters", random_state=0,
                 flavor="igraph", n_iterations=2, directed=False)

    return adata


BLOCK_CELL_TYPES = ["T/NK cell", "B cell", "Myeloid/Monocyte", "T/NK cell"]
BLOCK_MARKERS = ["CD3D", "MS4A1", "CD14", "GNLY"]


@pytest.fixture
def synthetic_adata_annotated(synthetic_adata_clustered):
    """Clustered synthetic data with cell_type set from the known blocks, plus 5 Ambiguous cells."""
    adata = synthetic_adata_clustered.copy()
    block_size = adata.n_obs // 4
    labels = []
    for i in range(adata.n_obs):
        labels.append(BLOCK_CELL_TYPES[min(i // block_size, 3)])
    labels[-5:] = ["Ambiguous"] * 5
    adata.obs["cell_type"] = pd.Categorical(labels)
    return adata
