"""
Unit tests for scripts/clustering.py.
"""

import os

import numpy as np
import pytest

import clustering


@pytest.fixture
def normalized_adata(synthetic_adata):
    """Synthetic data normalized and log-transformed, ready for PCA."""
    import scanpy as sc

    adata = synthetic_adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


class _FakeHarmony:
    def __init__(self, Z_corr):
        self.Z_corr = Z_corr


def test_pca_reduction_adds_pca_embedding(normalized_adata):
    result = clustering.pca_reduction(normalized_adata, n_components=20)

    assert "X_pca" in result.obsm
    assert result.obsm["X_pca"].shape == (result.n_obs, 20)


def test_leiden_clustering_assigns_clusters(normalized_adata):
    adata = clustering.pca_reduction(normalized_adata, n_components=20)
    result = clustering.leiden_clustering(adata, resolution=1.0)

    assert "leiden_clusters" in result.obs.columns
    n_clusters = result.obs["leiden_clusters"].nunique()
    # Four boosted marker blocks: expect real structure, not one blob or a fragmented graph.
    assert 2 <= n_clusters <= result.n_obs // 5


def test_leiden_clustering_is_reproducible_with_fixed_seed(normalized_adata):
    adata1 = clustering.pca_reduction(normalized_adata.copy(), n_components=20)
    adata1 = clustering.leiden_clustering(adata1, resolution=1.0, random_state=0)

    adata2 = clustering.pca_reduction(normalized_adata.copy(), n_components=20)
    adata2 = clustering.leiden_clustering(adata2, resolution=1.0, random_state=0)

    assert list(adata1.obs["leiden_clusters"]) == list(adata2.obs["leiden_clusters"])


def test_leiden_clustering_rejects_out_of_range_n_pcs(normalized_adata):
    adata = clustering.pca_reduction(normalized_adata, n_components=20)

    with pytest.raises(ValueError, match="out of range"):
        clustering.leiden_clustering(adata, resolution=1.0, n_pcs=999)


@pytest.mark.parametrize("transposed", [True, False])
def test_harmony_uses_only_requested_pcs_and_handles_orientation(normalized_adata, monkeypatch, transposed):
    """Harmony should see exactly n_pcs components, and either output orientation should yield (n_cells, n_pcs)."""
    adata = clustering.pca_reduction(normalized_adata, n_components=20)
    seen = {}

    def fake_run_harmony(pcs, obs, keys):
        seen["shape"] = pcs.shape
        seen["keys"] = keys
        corrected = pcs + 1.0
        return _FakeHarmony(corrected.T if transposed else corrected)

    monkeypatch.setattr(clustering.harmonypy, "run_harmony", fake_run_harmony)
    result, use_rep = clustering.harmony_integration(adata, n_pcs=10)

    assert use_rep == "X_pca_harmony"
    assert seen["shape"] == (adata.n_obs, 10)
    assert seen["keys"] == ["sample_id"]
    assert result.obsm["X_pca_harmony"].shape == (adata.n_obs, 10)
    np.testing.assert_allclose(result.obsm["X_pca_harmony"], adata.obsm["X_pca"][:, :10] + 1.0)


def test_harmony_rejects_bad_output_shape(normalized_adata, monkeypatch):
    adata = clustering.pca_reduction(normalized_adata, n_components=20)
    monkeypatch.setattr(clustering.harmonypy, "run_harmony", lambda pcs, obs, keys: _FakeHarmony(np.zeros((3, 3))))

    with pytest.raises(ValueError, match="doesn't match expected"):
        clustering.harmony_integration(adata, n_pcs=10)


def test_harmony_skipped_with_single_batch(normalized_adata):
    adata = clustering.pca_reduction(normalized_adata, n_components=20)
    adata.obs["sample_id"] = "ONLY_PATIENT"

    _, use_rep = clustering.harmony_integration(adata, n_pcs=10)

    assert use_rep == "X_pca"


def test_assess_clustering_stability_returns_valid_ari(normalized_adata):
    adata = clustering.pca_reduction(normalized_adata, n_components=20)
    adata = clustering.leiden_clustering(adata, resolution=1.0, n_pcs=20)

    result = clustering.assess_clustering_stability(
        adata, use_rep="X_pca", resolution=1.0, n_neighbors=20, n_pcs=20, n_repeats=2, subsample_fraction=0.8
    )

    assert {"mean_ari", "min_ari", "individual_ari_scores", "subsample_fraction"} <= set(result)
    assert len(result["individual_ari_scores"]) == 2
    for score in result["individual_ari_scores"]:
        assert -1.0 <= score <= 1.0


def test_assess_clustering_stability_rejects_bad_fraction(normalized_adata):
    adata = clustering.pca_reduction(normalized_adata, n_components=20)
    adata = clustering.leiden_clustering(adata, resolution=1.0, n_pcs=20)

    with pytest.raises(ValueError, match="subsample_fraction"):
        clustering.assess_clustering_stability(
            adata, use_rep="X_pca", resolution=1.0, n_neighbors=20, n_pcs=20, n_repeats=1, subsample_fraction=0
        )


def test_cluster_library_diagnostics_use_sample_library_pairs(normalized_adata):
    """Libraries are identified by (patient, library): the fixture has 3 libraries, not 2."""
    adata = clustering.pca_reduction(normalized_adata, n_components=20)
    adata = clustering.leiden_clustering(adata, resolution=1.0, n_pcs=20)

    counts, proportions, _ = clustering.compute_cluster_library_diagnostics(adata)

    assert counts.shape[1] == 3
    assert counts.values.sum() == adata.n_obs
    np.testing.assert_allclose(proportions.sum(axis=1).values, 1.0)
    assert os.path.exists(os.path.join("results", "cluster_by_library_counts.csv"))


def test_cluster_composition_flags_dominated_clusters(normalized_adata):
    adata = clustering.pca_reduction(normalized_adata, n_components=20)
    adata = clustering.leiden_clustering(adata, resolution=1.0, n_pcs=20)
    adata.obs["sample_id"] = "PATIENT_A"  # every cluster is 100% one patient

    _, _, dominated = clustering.compute_cluster_patient_diagnostics(adata)

    assert set(dominated) == set(adata.obs["leiden_clusters"].unique())
    assert all(share == 1.0 for share in dominated.values())
