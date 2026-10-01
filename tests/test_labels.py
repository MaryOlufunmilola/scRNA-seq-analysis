"""
Unit tests for scripts/labels.py.
"""

import numpy as np
import pandas as pd

import labels


def test_real_cell_type_mask_excludes_non_cell_type_labels():
    values = pd.Series(["B cell", "Ambiguous", "Unknown", "Likely doublet (technical artifact)", None, "Plasma cell"])
    np.testing.assert_array_equal(
        labels.real_cell_type_mask(values), [True, False, False, False, False, True]
    )


def test_real_cell_type_mask_accepts_categoricals():
    values = pd.Categorical(["Ambiguous", "Epithelial cell"])
    np.testing.assert_array_equal(labels.real_cell_type_mask(values), [False, True])
