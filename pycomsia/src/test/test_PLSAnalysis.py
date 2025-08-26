
import os
import pytest

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# Import seaborn and set backend for headless testing
import seaborn as sns
plt.switch_backend('Agg')  # Prevent GUI issues in test environment

from pycomsia.src.PLSAnalysis import PLSAnalysis


@pytest.fixture
def pls_analysis():
    return PLSAnalysis()


class MockPLSModel:
    def __init__(self, coef_array):
        self.coef_ = coef_array


def test_calculate_contribution_fractions_basic(pls_analysis):
    pls_analysis.field_names = ["field1", "field2"]
    pls_analysis.kept_indices = {
        "field1": [0, 1],
        "field2": [2, 3, 4]
    }

    # Coefficients: 2 for field1, 3 for field2
    coefs = np.array([[1.0, -2.0, 0.5, 1.5, -1.0]])  # shape (1, 5)
    pls_analysis.pls_model = MockPLSModel(coefs)

    pls_analysis.calculate_contribution_fractions()

    expected_abs_sums = {
        "field1": abs(1.0) + abs(-2.0),  # = 3.0
        "field2": abs(0.5) + abs(1.5) + abs(-1.0)  # = 3.0
    }
    total = sum(expected_abs_sums.values())  # = 6.0

    assert np.isclose(pls_analysis.contribution_fractions["field1"], 0.5)
    assert np.isclose(pls_analysis.contribution_fractions["field2"], 0.5)


def test_calculate_contribution_fractions_uneven(pls_analysis):
    pls_analysis.field_names = ["A", "B"]
    pls_analysis.kept_indices = {
        "A": [0],
        "B": [1, 2]
    }

    coefs = np.array([[2.0, 1.0, 1.0]])  # abs sums: A=2, B=2
    pls_analysis.pls_model = MockPLSModel(coefs)

    pls_analysis.calculate_contribution_fractions()

    assert np.isclose(pls_analysis.contribution_fractions["A"], 0.5)
    assert np.isclose(pls_analysis.contribution_fractions["B"], 0.5)


def test_calculate_contribution_fractions_zero_contrib(pls_analysis):
    pls_analysis.field_names = ["A", "B"]
    pls_analysis.kept_indices = {
        "A": [0, 1],
        "B": [2]
    }

    coefs = np.array([[0.0, 0.0, 1.0]])
    pls_analysis.pls_model = MockPLSModel(coefs)

    pls_analysis.calculate_contribution_fractions()

    assert np.isclose(pls_analysis.contribution_fractions["A"], 0.0)
    assert np.isclose(pls_analysis.contribution_fractions["B"], 1.0)


def test_calculate_contribution_fractions_no_model(pls_analysis):
    pls_analysis.field_names = ["A"]
    pls_analysis.kept_indices = {"A": [0]}
    pls_analysis.pls_model = None

    with pytest.raises(ValueError, match="PLS model has not been fitted yet"):
        pls_analysis.calculate_contribution_fractions()


def test_export_predictions_and_residuals(tmp_path, pls_analysis):
    # Mock data
    pls_analysis.y_train_final = np.array([1.0, 2.0])
    pls_analysis.y_train_predicted = np.array([1.5, 1.8])
    pls_analysis.y_train_final_original = np.array([1.0, 2.0])
    pls_analysis.y_train_predicted_original = np.array([1.5, 1.8])
    pls_analysis.y_test_final_original = np.array([3.0, 4.0])
    pls_analysis.y_test_predicted_original = np.array([3.1, 3.8])
    pls_analysis.train_indices = [0, 1]
    pls_analysis.test_indices = [2, 3]

    outputdir = tmp_path
    pls_analysis.export_predictions_and_residuals(outputdir)

    base_path = outputdir / "PLS_Analysis"

    combined_file = base_path / "Predictions_and_Residuals.csv"
    train_file = base_path / "Training_Predictions_and_Residuals.csv"
    test_file = base_path / "Test_Predictions_and_Residuals.csv"

    # Check files exist
    assert combined_file.exists()
    assert train_file.exists()
    assert test_file.exists()

    # Check contents
    df_combined = pd.read_csv(combined_file)
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Check structure
    assert list(df_combined.columns) == ["Set", "Index", "Actual Values", "Predicted Values", "Residuals"]
    assert len(df_combined) == 4
    assert df_train["Set"].iloc[0] == "Training"
    assert df_test["Set"].iloc[0] == "Test"

    # Check residuals
    expected_train_residuals = pls_analysis.y_train_predicted_original - pls_analysis.y_train_final_original
    np.testing.assert_array_almost_equal(df_train["Residuals"], expected_train_residuals)

    expected_test_residuals = pls_analysis.y_test_predicted_original - pls_analysis.y_test_final_original
    np.testing.assert_array_almost_equal(df_test["Residuals"], expected_test_residuals)


def test_export_predictions_and_residuals_not_fitted(tmp_path, pls_analysis):
    # Missing y_train_final and y_train_predicted
    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        pls_analysis.export_predictions_and_residuals(tmp_path)


def test_export_metrics_to_csv(tmp_path, pls_analysis):
    # Populate with dummy metrics
    pls_analysis.r2_train = 0.91
    pls_analysis.r2_test = 0.87
    pls_analysis.q2_scores = [0.80, 0.85, 0.83]
    pls_analysis.spress = 0.42
    pls_analysis.s_train = 0.35
    pls_analysis.s_test = 0.48
    pls_analysis.optimal_n_components = 5
    pls_analysis.contribution_fractions = {
        "field1": 0.6,
        "field2": 0.4
    }

    outputdir = tmp_path
    pls_analysis.export_metrics_to_csv(outputdir)

    file_path = outputdir / "PLS_Analysis" / "PLS_Metrics.csv"
    assert file_path.exists()

    df = pd.read_csv(file_path)

    # Check for all expected columns
    expected_columns = [
        "r2_train", "r2_test", "q2", "SPRESS", "S_train", "S_test", "Number of Components",
        "Contribution Fraction (field1)", "Contribution Fraction (field2)"
    ]
    assert list(df.columns) == expected_columns

    # Check content
    assert np.isclose(df.loc[0, "r2_train"], 0.91)
    assert np.isclose(df.loc[0, "r2_test"], 0.87)
    assert np.isclose(df.loc[0, "q2"], 0.85)  # max of q2_scores
    assert np.isclose(df.loc[0, "SPRESS"], 0.42)
    assert np.isclose(df.loc[0, "S_train"], 0.35)
    assert np.isclose(df.loc[0, "S_test"], 0.48)
    assert int(df.loc[0, "Number of Components"]) == 5
    assert np.isclose(df.loc[0, "Contribution Fraction (field1)"], 0.6)
    assert np.isclose(df.loc[0, "Contribution Fraction (field2)"], 0.4)


def test_convert_fields_to_X_basic(tmp_path, pls_analysis):
    # 3 molecules, 2 fields, each with 4 points
    train_fields = {
        "steric_field": [
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 3.5, 4.5],
            [2.0, 3.0, 4.0, 5.0]
        ],
        "electro_field": [
            [0.1, 0.2, 0.3, 0.4],
            [0.15, 0.25, 0.35, 0.45],
            [0.2, 0.3, 0.4, 0.5]
        ]
    }

    pred_fields = {
        "steric_field": [
            [1.1, 2.1, 3.1, 4.1],
            [1.6, 2.6, 3.6, 4.6]
        ],
        "electro_field": [
            [0.11, 0.21, 0.31, 0.41],
            [0.16, 0.26, 0.36, 0.46]
        ]
    }

    X_train, X_pred = pls_analysis.convert_fields_to_X(train_fields, pred_fields, filter=0.05)

    # Check shapes
    assert X_train.shape[0] == 3  # 3 training molecules
    assert X_pred.shape[0] == 2   # 2 prediction molecules

    # Check columns count matches: (number of kept points) * fields
    total_kept = sum(len(idx) for idx in pls_analysis.kept_indices.values())
    assert X_train.shape[1] == total_kept
    assert X_pred.shape[1] == total_kept

    # Check mean centering
    np.testing.assert_array_almost_equal(np.mean(X_train, axis=0), np.zeros_like(np.mean(X_train, axis=0)), decimal=6)

    # Check internal state
    assert "steric_field" in pls_analysis.kept_indices
    assert "electro_field" in pls_analysis.kept_indices
    assert pls_analysis.X_train.shape == X_train.shape
    assert pls_analysis.X_pred.shape == X_pred.shape
    assert pls_analysis.field_names == ["steric_field", "electro_field"]
    assert pls_analysis.field_shape == (4,)

def test_convert_fields_to_X_no_pred(pls_analysis):
    train_fields = {
        "steric_field": [
            [1, 2, 3],
            [1.5, 2.5, 3.5],
        ],
        "electro_field": [
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35],
        ]
    }

    X_train, X_pred = pls_analysis.convert_fields_to_X(train_fields, pred_fields=None, filter=0.01)

    assert X_train.shape[0] == 2
    assert X_pred is None
    assert pls_analysis.X_pred is None
    assert isinstance(pls_analysis.X_train, np.ndarray)
    assert "steric_field" in pls_analysis.field_names

def test_convert_fields_to_X_all_filtered_out(pls_analysis):
    train_fields = {
        "steric_field": [
            [1.0, 1.0, 1.0],  # No variance
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    }

    pred_fields = {
        "steric_field": [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    }

    X_train, X_pred = pls_analysis.convert_fields_to_X(train_fields, pred_fields, filter=0.01)

    assert X_train.shape[1] == 0
    assert X_pred.shape[1] == 0
    assert len(pls_analysis.kept_indices["steric_field"]) == 0
    assert np.allclose(X_train, np.zeros_like(X_train))


def test_get_coefficient_fields_basic(pls_analysis):
    # Simulated model with 5 coefficients
    pls_analysis.pls_model = MockPLSModel(np.array([[0.1, -0.2, 0.3, 0.4, -0.5]]))

    pls_analysis.field_names = ["field1", "field2"]
    pls_analysis.kept_indices = {
        "field1": np.array([0, 2]),
        "field2": np.array([1, 3, 4])
    }
    pls_analysis.field_stdevs = {
        "field1": 2.0,   # scalar std for all kept indices in field1
        "field2": 1.5    # scalar std for all kept indices in field2
    }
    pls_analysis.field_shape = (5,)

    result = pls_analysis.get_coefficient_fields()

    # Expected full arrays
    # field1: [0.1, 0.0, -0.2, 0.0, 0.0] * [2.0, 0, 2.0, 0, 0] => [0.2, 0.0, -0.4, 0.0, 0.0]
    # field2: [0.0, 0.3, 0.0, 0.4, -0.5] * [0, 1.5, 0, 1.5, 1.5] => [0.0, 0.45, 0.0, 0.6, -0.75]
    np.testing.assert_array_almost_equal(result["field1"], np.array([0.2, 0.0, -0.4, 0.0, 0.0]))
    np.testing.assert_array_almost_equal(result["field2"], np.array([0.0, 0.45, 0.0, 0.6, -0.75]))


def test_get_coefficient_fields_model_not_fitted(pls_analysis):
    pls_analysis.field_names = ["field1"]
    pls_analysis.kept_indices = {"field1": np.array([0])}
    pls_analysis.field_stdevs = {"field1": 1.0}
    pls_analysis.field_shape = (1,)

    # No model assigned
    with pytest.raises(ValueError, match="PLS model has not been fitted yet"):
        pls_analysis.get_coefficient_fields()


def test_perform_loo_analysis_basic(pls_analysis):
    # Create simple training data (5 samples, 3 features)
    X_train = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
    ])
    y_train = np.array([1.1, 2.1, 3.0, 3.9, 5.0])

    pls_analysis.X_train = X_train
    pls_analysis.perform_loo_analysis(y_train, max_components=3)

    # Check q2_scores list length
    assert len(pls_analysis.q2_scores) == 3

    # Check press_values shape: (n_components, n_samples)
    assert pls_analysis.press_values.shape == (3, 5)

    # Check that optimal_n_components is within valid range
    assert 1 <= pls_analysis.optimal_n_components <= 3

    # Check that SPRESS is non-negative
    assert pls_analysis.spress >= 0

    # Check that Q² values are floats between -∞ and 1
    for q2 in pls_analysis.q2_scores:
        assert isinstance(q2, float)


def test_perform_loo_analysis_fewer_samples_than_max_components(pls_analysis):
    # Only 4 samples
    pls_analysis.X_train = np.random.rand(4, 2)
    y_train = np.array([1.0, 2.0, 3.0, 4.0])

    pls_analysis.perform_loo_analysis(y_train, max_components=10)

    # Should compute only for 1 to 4 components (not 10)
    assert len(pls_analysis.q2_scores) == 4
    assert pls_analysis.press_values.shape == (4, 4)


def test_perform_loo_analysis_invalid_input_raises_error(pls_analysis):
    pls_analysis.X_train = None
    with pytest.raises(AttributeError):
        pls_analysis.perform_loo_analysis([1, 2, 3])


def test_fit_final_model_basic(tmp_path, pls_analysis):
    np.random.seed(0)

    # Create mock training data
    pls_analysis.X_train = np.random.rand(10, 5)
    train_activities = np.random.rand(10)

    # Create mock prediction set
    pls_analysis.X_pred = np.random.rand(3, 5)
    smiles_list = ["CCO", "CCC", "CCN"]

    pls_analysis.fit_final_model(train_activities, test_size=0.3, predict_smiles_list=smiles_list)

    # Assertions
    assert pls_analysis.pls_model is not None
    assert pls_analysis.y_train_predicted is not None
    assert pls_analysis.y_test_predicted is not None
    assert pls_analysis.r2_train <= 1.0
    assert pls_analysis.r2_test <= 1.0
    assert pls_analysis.s_train >= 0
    assert pls_analysis.s_test >= 0
    assert pls_analysis.spress == 0.1  # as mocked
    assert pls_analysis.calculate_contribution_fractions_called is True
    assert hasattr(pls_analysis, "y_pred_unseen_original")
    assert len(pls_analysis.y_pred_unseen_original) == 3

    # Check output file
    df = pd.read_csv("predictions.csv")
    assert list(df.columns) == ["SMILES", "Predicted Activity"]
    assert df.shape[0] == 3
    assert df["SMILES"].tolist() == smiles_list


def test_plot_results(tmp_path, pls_analysis):
    outputdir = tmp_path  # Temporary directory for test

    # Call plot_results
    pls_analysis.plot_results(outputdir)

    # Check that plot file was created
    plot_path = tmp_path / "PLS_Analysis" / "PLSplots.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0  # File is not empty

