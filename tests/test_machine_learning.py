"""
Unit tests for scripts/machine_learning.py.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

import machine_learning as ml


@pytest.fixture(autouse=True)
def _deterministic_torch():
    """Same seed and a single thread, so results match across machines and CI runners."""
    torch.manual_seed(0)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n_threads)


def _separable_data(n_per_group=30, n_features=10, groups=("A", "B", "C"), shift=4.0, seed=0):
    """Two linearly separable classes, present in every group (patient)."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for group in groups:
        labels = np.array([0, 1] * (n_per_group // 2))
        feats = rng.normal(size=(len(labels), n_features))
        feats[labels == 1, :3] += shift
        X.append(feats)
        y.append(labels)
        g.extend([group] * len(labels))
    return np.vstack(X).astype(np.float32), np.concatenate(y), np.array(g)


def test_model_forward_pass_output_shape():
    model = ml.CellTypePredictor(input_dim=20, hidden_dim=32, output_dim=4)
    assert model(torch.randn(10, 20)).shape == (10, 4)


def test_model_does_not_apply_softmax():
    """Regression test: forward() must return logits, not softmax probabilities (CrossEntropyLoss applies softmax)."""
    model = ml.CellTypePredictor(input_dim=20, hidden_dim=32, output_dim=4)
    row_sums = model(torch.randn(10, 20)).sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones(10), atol=1e-4)


def test_train_model_reduces_loss():
    torch.manual_seed(0)
    n_samples, n_features, n_classes = 40, 10, 2
    X = torch.randn(n_samples, n_features)
    y = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)]).long()
    X[y == 1] += 5.0

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=8, shuffle=True)
    model = ml.CellTypePredictor(input_dim=n_features, hidden_dim=16, output_dim=n_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.eval()
    with torch.no_grad():
        initial_loss = criterion(model(X), y).item()
    ml.train_model(model, loader, criterion, optimizer, epochs=15, verbose=False)
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(X), y).item()

    assert final_loss < initial_loss


def test_train_classifier_is_deterministic_with_seed():
    X, y, _ = _separable_data()
    preds_1 = ml.predict(ml.train_classifier(X, y, 2, epochs=5, seed=7, verbose=False), X)
    preds_2 = ml.predict(ml.train_classifier(X, y, 2, epochs=5, seed=7, verbose=False), X)
    np.testing.assert_array_equal(preds_1, preds_2)


def test_evaluate_model_reports_every_class_even_if_absent_from_test():
    torch.manual_seed(0)
    X = torch.randn(20, 10)
    y = torch.zeros(20, dtype=torch.long)  # only class 0 appears in the test data
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=8)
    model = ml.CellTypePredictor(input_dim=10, hidden_dim=16, output_dim=3)

    report = ml.evaluate_model(model, loader, class_names=["T cell", "B cell", "Plasma cell"])

    for name in ["T cell", "B cell", "Plasma cell"]:
        assert name in report


def test_fit_train_only_pca_shapes_and_mean():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 15)) + 3.0
    X_all = rng.normal(size=(50, 15))

    X_pca_train, X_pca_all, loadings = ml.fit_train_only_pca(X_train, X_all, n_components=5)

    assert X_pca_train.shape == (30, 5)
    assert X_pca_all.shape == (50, 5)
    assert loadings.shape == (15, 5)
    # Projection uses the training mean: (x - train_mean) @ loadings.
    np.testing.assert_allclose(X_pca_all, (X_all - X_train.mean(axis=0)) @ loadings, atol=1e-8)


@pytest.mark.slow
def test_leave_one_patient_out_returns_one_row_per_patient():
    X, y, groups = _separable_data()
    result = ml.leave_one_patient_out(X, y, groups, class_names=["neg", "pos"], n_components=5, epochs=100)

    assert list(result["held_out_patient"]) == ["A", "B", "C"]
    assert (result["n_test"] == 30).all()
    assert (result["accuracy"] > 0.9).all(), result
    assert all(missing == [] for missing in result["classes_missing_from_training"])
    for per_class, n_test in zip(result["per_class_f1"], result["n_test_per_class"]):
        assert set(per_class) == {"neg", "pos"}
        assert sum(n_test.values()) == 30


def test_leave_one_patient_out_reports_classes_unseen_in_training():
    X, y, groups = _separable_data()
    y = y.copy()
    y[(groups == "C") & (y == 1)] = 2  # class 2 exists only in patient C
    result = ml.leave_one_patient_out(X, y, groups, class_names=["neg", "pos", "only_C"], n_components=5, epochs=5)

    row_c = result.set_index("held_out_patient").loc["C"]
    assert row_c["classes_missing_from_training"] == ["only_C"]


def test_select_trainable_cells_drops_non_cell_types_and_tiny_classes():
    labels = ["B cell"] * 10 + ["Ambiguous"] * 4 + ["Unknown"] * 2 + ["Plasma cell"] * 2
    adata = ad.AnnData(X=np.ones((len(labels), 3), dtype=np.float32))
    adata.obs["cell_type"] = pd.Categorical(labels)

    result = ml.select_trainable_cells(adata, min_cells_per_class=5)

    assert set(result.obs["cell_type"]) == {"B cell"}
    assert list(result.obs["cell_type"].cat.categories) == ["B cell"]


def test_integrated_gradients_baseline_has_near_zero_attribution():
    """Attributing the baseline itself should give ~0: there is no difference from the baseline to explain."""
    torch.manual_seed(0)
    model = ml.CellTypePredictor(input_dim=10, hidden_dim=16, output_dim=3)
    model.eval()

    result = ml.integrated_gradients(model, class_idx=0, X_class=np.zeros((5, 10), dtype=np.float32), n_steps=20)

    assert np.allclose(result, 0.0, atol=1e-5)


def test_integrated_gradients_completeness():
    """IG attributions for each cell should sum to logit(x) - logit(baseline)."""
    torch.manual_seed(0)
    model = ml.CellTypePredictor(input_dim=10, hidden_dim=16, output_dim=3)
    model.eval()
    X = np.random.default_rng(1).normal(size=(8, 10)).astype(np.float32)

    attributions = ml.integrated_gradients_attributions(model, class_idx=1, X=X, n_steps=300)

    with torch.no_grad():
        logits = model(torch.tensor(X))[:, 1].numpy()
        baseline_logit = model(torch.zeros(1, 10))[0, 1].item()
    np.testing.assert_allclose(attributions.sum(axis=1), logits - baseline_logit, atol=0.02)


def test_pca_projected_model_matches_model_on_pca_coordinates():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(40, 12))
    _, X_pca_all, loadings = ml.fit_train_only_pca(X_train, X_train, n_components=4)
    torch.manual_seed(0)
    model = ml.CellTypePredictor(input_dim=4, hidden_dim=8, output_dim=2)
    model.eval()

    wrapped = ml.PCAProjectedModel(model, loadings, X_train.mean(axis=0))
    with torch.no_grad():
        direct = model(torch.tensor(X_pca_all, dtype=torch.float32))
        via_genes = wrapped(torch.tensor(X_train, dtype=torch.float32))

    torch.testing.assert_close(via_genes, direct, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_gene_level_attributions_point_to_the_discriminating_genes():
    """With classes separated on genes 0-2, class-1 attributions should be largest and positive on those genes."""
    X, y, _ = _separable_data(n_features=12)
    X_pca, _, loadings = ml.fit_train_only_pca(X, X, n_components=6)
    model = ml.train_classifier(X_pca, y, 2, epochs=100, seed=0, verbose=False)

    scores = ml.gene_level_attributions(model, loadings, X.mean(axis=0), X[y == 1], class_idx=1, n_steps=50)

    assert scores.shape == (12,)
    assert set(np.argsort(scores)[::-1][:3]) == {0, 1, 2}
    assert (scores[:3] > 0).all()
