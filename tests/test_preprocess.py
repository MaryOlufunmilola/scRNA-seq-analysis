"""
Unit tests for scripts/preprocess.py.

Scrublet is replaced by a deterministic stand-in (it is slow and unstable on tiny data);
these tests check that the pipeline calls it per library and handles its output.
"""

import argparse
import gzip
import io
import json
import tarfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import preprocess


class DummyScrublet:
    """Stand-in for scrublet.Scrublet: flags exactly the top 5% of each library as doublets."""

    def __init__(self, counts_matrix, expected_doublet_rate=0.06):
        self.n_cells = counts_matrix.shape[0]

    def scrub_doublets(self, **kwargs):
        scores = np.linspace(0, 1, self.n_cells)
        threshold_idx = int(self.n_cells * 0.95)
        predicted = np.zeros(self.n_cells, dtype=bool)
        predicted[threshold_idx:] = True
        self.threshold_ = scores[threshold_idx]  # real Scrublet sets this after scrub_doublets()
        return scores, predicted


def _expected_flagged(obs):
    per_library = obs.groupby(["sample_id", "library_id"], observed=True).size()
    return int(sum(n - int(n * 0.95) for n in per_library))


def test_detect_doublets_removes_exactly_the_flagged_cells(synthetic_adata, monkeypatch):
    monkeypatch.setattr(preprocess.scr, "Scrublet", DummyScrublet)
    expected_removed = _expected_flagged(synthetic_adata.obs)

    result = preprocess.detect_doublets(synthetic_adata)

    assert expected_removed > 0
    assert result.n_obs == synthetic_adata.n_obs - expected_removed
    assert "doublet_score" in result.obs.columns
    assert not result.obs["predicted_doublet"].any()


def test_detect_doublets_can_keep_flagged_cells(synthetic_adata, monkeypatch):
    monkeypatch.setattr(preprocess.scr, "Scrublet", DummyScrublet)

    result = preprocess.detect_doublets(synthetic_adata, remove_predicted_doublets=False)

    assert result.n_obs == synthetic_adata.n_obs
    assert int(result.obs["predicted_doublet"].sum()) == _expected_flagged(synthetic_adata.obs)


def test_preprocess_data_sets_raw_before_hvg_subset(synthetic_adata, monkeypatch):
    """adata.raw must hold the full gene set, set before HVG subsetting, so marker lookups work downstream."""
    monkeypatch.setattr(preprocess.scr, "Scrublet", DummyScrublet)

    result = preprocess.preprocess_data(synthetic_adata)

    assert result.raw is not None
    assert result.raw.n_vars >= result.n_vars
    assert "counts" in result.layers


def _with_low_quality_cells(adata, n_each=5):
    """Append cells with too few detected genes and cells with a high mitochondrial fraction."""
    n_genes = adata.n_vars
    few_genes = np.zeros((n_each, n_genes), dtype=np.float32)
    few_genes[:, 10:30] = 5  # 20 detected genes, below min_genes=100
    high_mt = np.ones((n_each, n_genes), dtype=np.float32)
    high_mt[:, list(adata.var_names).index("MT-CO1")] = 2000  # ~80% mitochondrial counts

    extra = ad.AnnData(
        X=np.vstack([few_genes, high_mt]),
        obs=pd.DataFrame(
            {"sample_id": "PATIENT_B", "library_id": "lib1"},
            index=[f"lowgenes{i}" for i in range(n_each)] + [f"highmt{i}" for i in range(n_each)],
        ),
        var=adata.var.copy(),
    )
    return ad.concat([adata, extra], join="outer", merge="same")


def test_preprocess_data_filters_low_quality_cells(synthetic_adata, monkeypatch):
    monkeypatch.setattr(preprocess.scr, "Scrublet", DummyScrublet)
    adata = _with_low_quality_cells(synthetic_adata)
    bad_cells = {c for c in adata.obs_names if c.startswith(("lowgenes", "highmt"))}

    result = preprocess.preprocess_data(adata, remove_predicted_doublets=False)

    assert not bad_cells & set(result.obs_names)
    assert (result.obs["n_genes_by_counts"] >= 100).all()
    assert (result.obs["pct_counts_mt"] < 15).all()
    assert result.n_obs == synthetic_adata.n_obs  # every good cell survives


def test_preprocess_data_selects_highly_variable_genes(synthetic_adata, monkeypatch):
    monkeypatch.setattr(preprocess.scr, "Scrublet", DummyScrublet)

    result = preprocess.preprocess_data(synthetic_adata)

    assert 0 < result.n_vars <= synthetic_adata.n_vars


def test_count_libraries_counts_sample_library_pairs(synthetic_adata):
    """library_id 'lib1' is reused across patients; counting unique library_id alone undercounts."""
    assert synthetic_adata.obs["library_id"].nunique() == 2
    assert preprocess.count_libraries(synthetic_adata.obs) == 3


def test_library_id_from_filename():
    assert preprocess._library_id_from_filename(
        "GSM6177620_NYU_UCEC1_lib2_lib2_gene_expression.mtx.gz") == "lib2"
    assert preprocess._library_id_from_filename("GSM6177621_NYU_UCEC2_gene_expression.mtx.gz") == "lib1"


def test_select_gene_columns_by_header_name():
    genes_df = pd.DataFrame({"ensembl_id": ["ENSG1", "ENSG2"], "name": ["CD3D", "EPCAM"], "chromosome": ["11", "2"]})
    symbols, ids = preprocess._select_gene_symbol_and_id_columns(genes_df)
    assert list(symbols) == ["CD3D", "EPCAM"]
    assert list(ids) == ["ENSG1", "ENSG2"]


def test_select_gene_columns_without_header_uses_positions():
    genes_df = pd.DataFrame([["ENSG1", "CD3D"], ["ENSG2", "EPCAM"]])  # integer column labels
    symbols, ids = preprocess._select_gene_symbol_and_id_columns(genes_df)
    assert list(symbols) == ["CD3D", "EPCAM"]
    assert list(ids) == ["ENSG1", "ENSG2"]


def test_select_gene_columns_single_column():
    symbols, ids = preprocess._select_gene_symbol_and_id_columns(pd.DataFrame([["CD3D"], ["EPCAM"]]))
    assert list(symbols) == ["CD3D", "EPCAM"]
    assert ids is None


def test_read_genes_file_detects_header_row(tmp_path):
    path = tmp_path / "genes.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write("ensembl_id\tname\n")
        f.write("ENSG1\tCD3D\nENSG2\tEPCAM\nENSG3\tPTPRC\n")

    genes_df = preprocess._read_genes_file(str(path), expected_n=3)

    assert len(genes_df) == 3
    assert list(genes_df.columns) == ["ensembl_id", "name"]


def test_read_genes_file_without_header(tmp_path):
    path = tmp_path / "genes.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write("ENSG1\tCD3D\nENSG2\tEPCAM\n")

    genes_df = preprocess._read_genes_file(str(path), expected_n=2)

    assert len(genes_df) == 2
    assert genes_df.iloc[0, 1] == "CD3D"


def _make_tar(path, members):
    """Write a tar with the given {name: bytes} regular files, plus optional (name, 'symlink', target) entries."""
    with tarfile.open(path, "w") as tar:
        for entry in members:
            if len(entry) == 3:
                name, _, target = entry
                info = tarfile.TarInfo(name)
                info.type = tarfile.SYMTYPE
                info.linkname = target
                tar.addfile(info)
            else:
                name, data = entry
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def test_safe_extract_keeps_only_ucec_regular_files(tmp_path):
    archive = tmp_path / "raw.tar"
    _make_tar(archive, [
        ("GSM6177620_NYU_UCEC1_lib1_lib1_gene_expression.mtx.gz", b"a"),
        ("GSM6177622_NYU_UCEC3_lib2_lib2_genes.tsv.gz", b"b"),
        ("GSM6177623_NYU_UCEC3_Vis_filtered.h5", b"visium"),       # Visium sample: not selected
        ("GSM9999999_OTHER_TUMOR_gene_expression.mtx.gz", b"c"),   # other tumor type: not selected
        ("GSM6177621_link", "symlink", "/etc/passwd"),              # link: never extracted
    ])
    dest = tmp_path / "out"

    extracted = preprocess.safe_extract(str(archive), str(dest))

    assert sorted(extracted) == [
        "GSM6177620_NYU_UCEC1_lib1_lib1_gene_expression.mtx.gz",
        "GSM6177622_NYU_UCEC3_lib2_lib2_genes.tsv.gz",
    ]
    assert sorted(p.name for p in dest.iterdir()) == sorted(extracted)


def test_safe_extract_refuses_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar"
    _make_tar(archive, [("../GSM6177620_escape_gene_expression.mtx.gz", b"x")])
    dest = tmp_path / "out"

    with pytest.raises(ValueError, match="escapes"):
        preprocess.safe_extract(str(archive), str(dest))
    assert not (tmp_path / "GSM6177620_escape_gene_expression.mtx.gz").exists()


def test_record_preprocess_params_writes_settings_and_checksum(synthetic_adata):
    args = argparse.Namespace(min_genes=100, max_genes=6000, max_pct_mt=15, doublet_rate=0.06,
                              keep_predicted_doublets=False)
    params = preprocess.record_preprocess_params(
        synthetic_adata, args, {"PATIENT_B": 0.15}, n_loaded=250, n_libraries=3, archive_sha256="ab" * 32
    )

    with open("results/preprocess_params.json") as f:
        saved = json.load(f)
    assert saved == params
    assert saved["geo_archive_sha256"] == "ab" * 32
    assert saved["min_genes"] == 100
    assert saved["doublet_rate_overrides"] == "PATIENT_B=0.15"
    assert saved["n_cells_after_qc"] == synthetic_adata.n_obs
    assert "scanpy" in saved["software"]
    assert synthetic_adata.uns["preprocess_params"]["n_libraries"] == 3
