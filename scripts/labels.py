"""
Non-cell-type labels shared across the pipeline (Ambiguous, Unknown, doublet),
excluded from classifier training, signature export, and cell communication.
"""

AMBIGUOUS_LABEL = "Ambiguous"
UNKNOWN_LABEL = "Unknown"
DOUBLET_LABEL = "Likely doublet (technical artifact)"

NON_CELL_TYPE_LABELS = frozenset({AMBIGUOUS_LABEL, UNKNOWN_LABEL, DOUBLET_LABEL})


def real_cell_type_mask(labels):
    """Boolean numpy array: True for cells whose label is an actual cell type."""
    import numpy as np

    values = [None if v is None else str(v) for v in labels]
    return np.array(
        [v is not None and v != "nan" and v not in NON_CELL_TYPE_LABELS for v in values],
        dtype=bool,
    )
