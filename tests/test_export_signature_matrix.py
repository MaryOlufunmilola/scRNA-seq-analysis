"""
Unit tests for scripts/export_signature_matrix.py.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import export_signature_matrix as esm


def _tiny_annotated(counts, cell_types, genes=("G1", "G2")):
    """AnnData whose .raw holds log1p(counts), as preprocess.py stores it."""
    logged = np.log1p(np.asarray(counts, dtype=np.float64))
    adata = ad.AnnData(X=logged.copy(), var=pd.DataFrame(index=list(genes)))
    adata.raw = adata.copy()
    adata.obs["cell_type"] = pd.Categorical(cell_types)
    return adata


def test_signature_matrix_averages_on_linear_scale():
    """Mean of CP10K values (1 and 3 -> 2), not expm1 of the mean log value (~1.83)."""
    adata = _tiny_annotated([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 7.0]], ["A", "A", "B", "B"])

    sig = esm.build_signature_matrix(adata)

    assert sig.loc["G1", "A"] == pytest.approx(2.0)
    assert sig.loc["G2", "B"] == pytest.approx(6.0)


def test_signature_matrix_excludes_non_cell_type_labels():
    adata = _tiny_annotated(
        [[1.0, 0.0], [3.0, 0.0], [100.0, 100.0], [0.0, 5.0]], ["A", "A", "Ambiguous", "B"]
    )

    sig = esm.build_signature_matrix(adata)

    assert list(sig.columns) == ["A", "B"]
    assert sig.loc["G1", "A"] == pytest.approx(2.0)  # the Ambiguous cell did not leak into A


def test_signature_matrix_requires_raw():
    adata = _tiny_annotated([[1.0, 0.0], [0.0, 1.0]], ["A", "B"])
    adata.raw = None
    with pytest.raises(ValueError, match="adata.raw is not set"):
        esm.build_signature_matrix(adata)


def test_select_signature_genes_finds_block_markers(synthetic_adata_annotated):
    genes = esm.select_signature_genes(synthetic_adata_annotated, n_per_type=5)

    assert {"MS4A1", "CD14"} <= set(genes)
    assert {"CD3D", "GNLY"} & set(genes)  # both mark the merged T/NK blocks
    assert len(genes) == len(set(genes))
