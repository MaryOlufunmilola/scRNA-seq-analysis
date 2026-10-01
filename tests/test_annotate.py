"""
Unit tests for scripts/annotate.py.

CellTypist's model download and prediction are replaced by stand-ins; the
tests check what this pipeline passes to CellTypist and how it handles the
output, not CellTypist itself.
"""

import numpy as np
import pandas as pd
import pytest

import annotate


def test_marker_gene_annotation_requires_raw(synthetic_adata_clustered):
    adata = synthetic_adata_clustered.copy()
    adata.raw = None

    with pytest.raises(ValueError, match="adata.raw is not set"):
        annotate.marker_gene_annotation(adata)


def test_marker_gene_annotation_assigns_label_per_cluster(synthetic_adata_clustered):
    result, gene_scale_stats = annotate.marker_gene_annotation(synthetic_adata_clustered)

    assert "marker_gene_label" in result.obs.columns
    assert result.obs["marker_gene_label"].isna().sum() == 0
    assert isinstance(gene_scale_stats, dict)
    # One label per cluster: every cell in a cluster shares it.
    assert (result.obs.groupby("leiden_clusters", observed=True)["marker_gene_label"].nunique() == 1).all()


def test_marker_gene_annotation_finds_known_markers(synthetic_adata_clustered):
    """The fixture boosts CD3D/GNLY, MS4A1, and CD14 in separate blocks; those lineages should be found."""
    result, _ = annotate.marker_gene_annotation(synthetic_adata_clustered)
    labels_found = set(result.obs["marker_gene_label"].unique())

    assert labels_found & {"T/NK cell", "B cell", "Myeloid/Monocyte"}


def test_compute_cluster_concordance_returns_three_fractions_per_cluster(synthetic_adata_clustered):
    adata, _ = annotate.marker_gene_annotation(synthetic_adata_clustered)

    concordance = annotate.compute_cluster_concordance(adata, marker_gene_list=list(annotate.MARKER_GENES.keys()))

    for col in ["leiden_cluster", "concordant", "ambiguous_cell", "discordant_other"]:
        assert col in concordance.columns
    fractions = concordance[["concordant", "ambiguous_cell", "discordant_other"]]
    assert ((fractions >= 0) & (fractions <= 1)).all().all()
    np.testing.assert_allclose(fractions.sum(axis=1).values, 1.0, atol=1e-6)


def test_run_min_margin_sensitivity_returns_one_row_per_margin(synthetic_adata_clustered):
    margins = (0.05, 0.15, 0.30)
    result = annotate.run_min_margin_sensitivity(synthetic_adata_clustered, margins_to_test=margins)

    assert len(result) == len(margins)
    assert set(result["min_margin"]) == set(margins)
    # A stricter margin can only make more clusters Ambiguous, never fewer.
    ordered = result.sort_values("min_margin")["n_ambiguous_clusters"].tolist()
    assert ordered == sorted(ordered)


def test_record_marker_panel_coverage_covers_every_cell_type():
    coverage = annotate.record_marker_panel_coverage(["CD3D", "MS4A1"])

    assert set(coverage["cell_type"]) == set(annotate.MARKER_GENES.values())
    assert (coverage["markers_available"] <= coverage["markers_assigned"]).all()


def test_label_from_scores_marks_close_calls_ambiguous():
    label, *_ = annotate._label_from_scores({"B cell": 1.0, "Plasma cell": 0.95}, min_margin=0.15)
    assert label == "Ambiguous"
    label, *_ = annotate._label_from_scores({"B cell": 1.0, "Plasma cell": 0.5}, min_margin=0.15)
    assert label == "B cell"
    label, *_ = annotate._label_from_scores({}, min_margin=0.15)
    assert label == "Unknown"


def test_run_celltypist_uses_all_genes_and_leiden_clusters(synthetic_adata_clustered, monkeypatch):
    """
    CellTypist must receive adata.raw (all genes), not the HVG-subset .X, and
    majority voting must use this pipeline's Leiden clusters.
    """
    adata = synthetic_adata_clustered.copy()
    captured = {}

    class FakePredictions:
        def __init__(self, data):
            self.data = data

        def to_adata(self):
            out = self.data.copy()
            out.obs["predicted_labels"] = "Memory B cells"
            out.obs["majority_voting"] = "Memory B cells"
            out.obs["conf_score"] = 0.9
            return out

    def fake_annotate(data, model, majority_voting, over_clustering=None):
        captured["var_names"] = list(data.var_names)
        captured["over_clustering"] = over_clustering
        captured["majority_voting"] = majority_voting
        return FakePredictions(data)

    monkeypatch.setattr(annotate, "_model_is_cached", lambda name: True)
    monkeypatch.setattr(annotate.models.Model, "load", staticmethod(lambda model: "fake-model"))
    monkeypatch.setattr(annotate.celltypist, "annotate", fake_annotate)

    result = annotate.run_celltypist(adata)

    assert captured["var_names"] == list(adata.raw.var_names)
    assert captured["majority_voting"] is True
    assert list(captured["over_clustering"]) == list(adata.obs["leiden_clusters"].astype(str))
    assert (result.obs["majority_voting"] == "Memory B cells").all()
    assert result.n_vars == synthetic_adata_clustered.n_vars  # original object keeps its HVG matrix


def test_run_celltypist_requires_raw(synthetic_adata_clustered):
    adata = synthetic_adata_clustered.copy()
    adata.raw = None
    with pytest.raises(ValueError, match="adata.raw is not set"):
        annotate.run_celltypist(adata)


def test_cross_check_flags_only_genuine_disagreements(synthetic_adata_clustered):
    adata = synthetic_adata_clustered.copy()
    clusters = adata.obs["leiden_clusters"].cat.categories
    first = clusters[0]
    adata.obs["marker_gene_label"] = "B cell"
    adata.obs["majority_voting"] = np.where(adata.obs["leiden_clusters"] == first, "Macrophages", "Memory B cells")

    comparison = annotate.cross_check_immune_cells(adata)

    assert not comparison.loc[first, "compatible"]
    assert comparison.drop(index=first)["compatible"].all()


@pytest.mark.parametrize("celltypist_label,marker_label,expected", [
    # Cases observed on real data, where exact string matching cried wolf:
    ("Plasma cells", "Plasma cell", True),
    ("Memory B cells", "B cell", True),
    ("Tem/Trm cytotoxic T cells", "T/NK cell", True),
    ("Regulatory T cells", "T/NK cell", True),
    ("Macrophages", "Myeloid/Monocyte", True),
    ("Tcm/Naive helper T cells", "T/NK cell", True),
    ("MAIT cells", "T/NK cell", True),
    ("CD16+ NK cells", "T/NK cell", True),
    ("Plasmablasts", "B cell", True),
    ("DC2", "Dendritic cell", True),
    ("pDC", "Dendritic cell", True),
    ("Migratory DCs", "Myeloid/Monocyte", True),
    ("Neutrophils", "Neutrophil", True),
    # Genuine mismatches:
    ("Plasma cells", "B cell", False),
    ("Macrophages", "T/NK cell", False),
    ("Mast cells", "Myeloid/Monocyte", False),
    # Whole-word matching: "stem" must not match the T-cell keyword "tem".
    ("Hematopoietic stem cells", "T/NK cell", False),
])
def test_celltypist_label_compatibility(celltypist_label, marker_label, expected):
    assert annotate._celltypist_label_compatible_with_marker_label(celltypist_label, marker_label) == expected


def _with_doublet_scores(adata, high_cluster, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.10, 0.02, adata.n_obs).clip(0, 1)
    in_high = (adata.obs["leiden_clusters"] == high_cluster).values
    scores[in_high] = rng.normal(0.35, 0.03, in_high.sum()).clip(0, 1)
    adata.obs["doublet_score"] = scores
    return adata


def test_flag_doublet_enriched_clusters_flags_only_the_elevated_cluster(synthetic_adata_clustered):
    adata = synthetic_adata_clustered.copy()
    high = adata.obs["leiden_clusters"].cat.categories[0]
    adata = _with_doublet_scores(adata, high)

    table = annotate.flag_doublet_enriched_clusters(adata)

    assert set(table.loc[table["flagged"], "leiden_cluster"]) == {str(high)}
    assert (table["padj"] >= table["pval"] - 1e-12).all()  # BH never makes p-values smaller


def test_flag_doublet_enriched_clusters_requires_fold_change(synthetic_adata_clustered):
    """A significant but small increase (well under 2x) must not be flagged."""
    adata = synthetic_adata_clustered.copy()
    high = adata.obs["leiden_clusters"].cat.categories[0]
    rng = np.random.default_rng(1)
    scores = rng.normal(0.10, 0.005, adata.n_obs)
    scores[(adata.obs["leiden_clusters"] == high).values] += 0.03  # ~1.3x, clearly significant
    adata.obs["doublet_score"] = scores

    table = annotate.flag_doublet_enriched_clusters(adata)

    assert not table["flagged"].any()


def test_flag_doublet_enriched_clusters_without_scores(synthetic_adata_clustered):
    table = annotate.flag_doublet_enriched_clusters(synthetic_adata_clustered.copy())
    assert len(table) == 0


def test_apply_doublet_label_relabels_whole_cluster():
    labels = ["B cell", "B cell", "Epithelial cell", "Epithelial cell"]
    clusters = ["0", "0", "1", "1"]
    table = pd.DataFrame({"leiden_cluster": ["0", "1"], "flagged": [False, True]})

    result = annotate.apply_doublet_label(labels, clusters, table)

    assert list(result) == ["B cell", "B cell", annotate.DOUBLET_LABEL, annotate.DOUBLET_LABEL]


def test_celltypist_model_info_records_checksum_and_version(tmp_path, monkeypatch):
    import hashlib

    model_file = tmp_path / "Immune_All_Low.pkl"
    model_file.write_bytes(b"fake model bytes")
    monkeypatch.setattr(annotate.models, "models_path", str(tmp_path))

    class FakeModel:
        description = {"version": "v2", "date": "2022-10-01", "number_celltypes": 98}

    info = annotate._celltypist_model_info(FakeModel(), model_name="Immune_All_Low.pkl")

    assert info["sha256"] == hashlib.sha256(b"fake model bytes").hexdigest()
    assert info["version"] == "v2"
    assert info["date"] == "2022-10-01"
    assert "number_celltypes" not in info
