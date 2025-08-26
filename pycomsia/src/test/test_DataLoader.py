import os
import pytest
import pandas as pd

from rdkit import Chem
from unittest.mock import patch, MagicMock

from pycomsia.src.DataLoader import DataLoader


@pytest.fixture
def data_loader():
    return DataLoader()


def test_load_data_training_success(tmp_path, data_loader):
    # Create a temporary CSV file for training
    csv_file = tmp_path / "training_data.csv"
    df = pd.DataFrame({
        'SMILES': ['CCO', 'CCN', 'CCC'],
        'Activity': [1.2, 3.4, 5.6]
    })
    df.to_csv(csv_file, index=False)

    smiles, activities = data_loader.load_data(csv_file, is_training=True)

    assert smiles == ['CCO', 'CCN', 'CCC']
    assert all(activities == [1.2, 3.4, 5.6])


def test_load_data_prediction_success(tmp_path, data_loader):
    # Create a temporary CSV file for prediction
    csv_file = tmp_path / "prediction_data.csv"
    df = pd.DataFrame({
        'SMILES': ['CNC', 'COC']
    })
    df.to_csv(csv_file, index=False)

    smiles, activities = data_loader.load_data(csv_file, is_training=False)

    assert smiles == ['CNC', 'COC']
    assert activities is None


def test_load_data_training_missing_activity(tmp_path, data_loader):
    # Missing Activity column
    csv_file = tmp_path / "bad_training_data.csv"
    df = pd.DataFrame({
        'SMILES': ['CCO', 'CCN']
    })
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Activity column not found in training dataset"):
        data_loader.load_data(csv_file, is_training=True)


@patch("rdkit.Chem.SDMolSupplier")
@patch("rdkit.Chem.MolToSmiles")
@patch("rdkit.Chem.AddHs")
def test_load_sdf_data_training_success(mock_add_hs, mock_to_smiles, mock_supplier, data_loader):
    # Mock molecule
    mol_mock = MagicMock()
    mol_mock.GetPropNames.return_value = ['Activity']
    mol_mock.GetProp.return_value = "5.5"
    mock_to_smiles.return_value = "CCO"
    mock_add_hs.side_effect = lambda m: m
    mock_supplier.return_value = [mol_mock]

    smiles, mols_with_flag, activities = data_loader.load_sdf_data("fake.sdf", activity_property="Activity", is_training=True)

    assert smiles == ["CCO"]
    assert mols_with_flag == [(mol_mock, True)]
    assert activities == [5.5]

@patch("rdkit.Chem.SDMolSupplier")
@patch("rdkit.Chem.MolToSmiles")
@patch("rdkit.Chem.AddHs")
def test_load_sdf_data_prediction_success(mock_add_hs, mock_to_smiles, mock_supplier, data_loader):
    mol_mock = MagicMock()
    mock_to_smiles.return_value = "CCN"
    mock_add_hs.side_effect = lambda m: m
    mock_supplier.return_value = [mol_mock]

    smiles, mols_with_flag, activities = data_loader.load_sdf_data("fake.sdf", is_training=False)

    assert smiles == ["CCN"]
    assert mols_with_flag == [(mol_mock, False)]
    assert activities is None

@patch("rdkit.Chem.SDMolSupplier")
def test_load_sdf_data_missing_activity_property(mock_supplier, data_loader):
    mol_mock = MagicMock()
    mol_mock.GetPropNames.return_value = []
    mock_supplier.return_value = [mol_mock]

    with pytest.raises(ValueError, match="Activity property 'Activity' not found in SDF file"):
        data_loader.load_sdf_data("fake.sdf", activity_property="Activity", is_training=True)

@patch("rdkit.Chem.SDMolSupplier")
def test_load_sdf_data_no_activity_key_specified(mock_supplier, data_loader):
    mol_mock = MagicMock()
    mock_supplier.return_value = [mol_mock]

    with pytest.raises(ValueError, match="Activity property must be specified for training data"):
        data_loader.load_sdf_data("fake.sdf", activity_property=None, is_training=True)