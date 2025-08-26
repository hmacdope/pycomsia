import numpy as np
import pytest

from unittest.mock import patch, MagicMock

from rdkit import Chem
from rdkit.Chem import AllChem

from pycomisa.src.MolecularFieldCalculator import MolecularFieldCalculator



@pytest.fixture
def molecular_field_calculator():
    return MolecularFieldCalculator()


@patch.object(MolecularFieldCalculator, '_calc_single_molecule_field')
def test_calc_field_sorts_correctly(mock_calc_field, molecular_field_calculator):
    # Create mock field values
    dummy_field = np.ones((2, 2, 2))  # shape doesn't matter, just needs to be flattenable
    field_output = {
        'steric_field': dummy_field,
        'electrostatic_field': dummy_field,
        'hydrophobic_field': dummy_field,
        'hbond_donor_field': dummy_field,
        'hbond_acceptor_field': dummy_field
    }
    mock_calc_field.return_value = field_output

    mol1 = MagicMock()
    mol2 = MagicMock()
    mol3 = MagicMock()

    aligned_results = [
        (mol1, True),   # training
        (mol2, False),  # prediction
        (mol3, True)    # training
    ]

    result = molecular_field_calculator.calc_field(
        aligned_results,
        grid_spacing=(1.0, 1.0, 1.0),
        grid_dimensions=(2, 2, 2),
        grid_origin=(0.0, 0.0, 0.0)
    )

    assert 'train_fields' in result
    assert 'pred_fields' in result

    # Ensure training has 2 molecules, prediction has 1
    for field in field_output.keys():
        assert len(result['train_fields'][field]) == 2
        assert len(result['pred_fields'][field]) == 1

        for entry in result['train_fields'][field] + result['pred_fields'][field]:
            assert isinstance(entry, np.ndarray)
            assert entry.shape == (8,)  # 2*2*2 flattened


@patch("rdkit.Chem.AllChem.ComputeGasteigerCharges")
@patch("rdkit.Chem.rdMolDescriptors._CalcCrippenContribs")
@patch("rdkit.Chem.GetPeriodicTable")
def test_calc_single_molecule_field_basic(mock_get_table, mock_crippen, mock_gasteiger, molecular_field_calculator):
    # Mock molecule and conformer
    mol = MagicMock()
    mol.GetNumAtoms.return_value = 1
    atom = MagicMock()
    atom.GetProp.return_value = "0.5"
    atom.GetAtomicNum.return_value = 6  # Carbon
    mol.GetAtoms.return_value = [atom]
    mol.GetSubstructMatches.return_value = [(0,)]

    conformer = MagicMock()
    conformer.GetAtomPosition.return_value = np.array([0.0, 0.0, 0.0])
    mol.GetConformer.return_value = conformer

    # Mock Crippen and VDW
    mock_crippen.return_value = [(0.3, 0.0)]  # Hydrophobicity
    ptable = MagicMock()
    ptable.GetRvdw.return_value = 1.5
    mock_get_table.return_value = ptable

    grid_spacing = (1.0, 1.0, 1.0)
    grid_dimensions = (2, 2, 2)
    grid_origin = (0.0, 0.0, 0.0)

    fields = molecular_field_calculator._calc_single_molecule_field(mol, grid_spacing, grid_dimensions, grid_origin)

    assert set(fields.keys()) == {
        'steric_field',
        'electrostatic_field',
        'hydrophobic_field',
        'hbond_donor_field',
        'hbond_acceptor_field'
    }

    for field in fields.values():
        assert isinstance(field, np.ndarray)
        assert field.shape == (2, 2, 2)


@pytest.fixture
def simple_molecule():
    mol = Chem.MolFromSmiles('CCO')  # ethanol
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    return mol

def test_generate_pseudoatoms(simple_molecule, molecular_field_calculator):
    # Mock internal methods
    molecular_field_calculator._get_hbond_positions = MagicMock(return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    molecular_field_calculator._filter_positions = MagicMock(return_value=[[1.0, 0.0, 0.0]])

    matches = [(2,)]  # Assuming atom index 2 is O in ethanol
    is_donor = True

    result = molecular_field_calculator.generate_pseudoatoms(simple_molecule, matches, is_donor)

    assert result == [[1.0, 0.0, 0.0]]
    molecular_field_calculator._get_hbond_positions.assert_called_once()
    molecular_field_calculator._filter_positions.assert_called_once_with(simple_molecule, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], True, donor_acceptor_idx=2)


# === FIXTURE: A simple water molecule for donor test ===
@pytest.fixture
def water_molecule():
    mol = Chem.MolFromSmiles("O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    return mol


# === FIXTURE: A molecule with acceptor for acceptor test ===
@pytest.fixture
def ethanol_molecule():
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    return mol


def test_get_hbond_positions_donor(water_molecule, molecular_field_calculator):
    oxygen = water_molecule.GetAtomWithIdx(0)
    atom_pos = np.array(water_molecule.GetConformer().GetAtomPosition(0))

    positions = molecular_field_calculator._get_hbond_positions(water_molecule, oxygen, atom_pos, is_donor=True)

    assert isinstance(positions, list)
    assert all(isinstance(p, np.ndarray) for p in positions)
    assert len(positions) > 0  # should find 2 H-bond directions for water


def test_get_hbond_positions_acceptor(ethanol_molecule, monkeypatch, molecular_field_calculator):
    oxygen = ethanol_molecule.GetAtomWithIdx(8)  # Index of O in CCO
    atom_pos = np.array(ethanol_molecule.GetConformer().GetAtomPosition(8))

    # Mock _get_hybridization_vectors
    mock_vectors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    monkeypatch.setattr(molecular_field_calculator, "_get_hybridization_vectors", lambda atom: mock_vectors)

    positions = molecular_field_calculator._get_hbond_positions(ethanol_molecule, oxygen, atom_pos, is_donor=False)

    assert len(positions) == len(mock_vectors)
    for p in positions:
        assert isinstance(p, np.ndarray)
        assert p.shape == (3,)


# === Fixtures ===

@pytest.fixture
def sp2_atom():
    mol = Chem.MolFromSmiles("C=C")  # Ethene
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    return mol.GetAtomWithIdx(0)  # First carbon, sp2


@pytest.fixture
def sp3_atom():
    mol = Chem.MolFromSmiles("CC")  # Ethane
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    return mol.GetAtomWithIdx(0)  # First carbon, sp3


@pytest.fixture
def other_hybrid_atom():
    mol = Chem.MolFromSmiles("C#N")  # Hydrogen cyanide, sp
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    return mol.GetAtomWithIdx(0)  # Carbon, sp


# === Tests ===

def test_hybridization_vectors_sp2(sp2_atom, molecular_field_calculator):
    vectors = molecular_field_calculator._get_hybridization_vectors(sp2_atom)

    assert len(vectors) == 3
    for vec in vectors:
        assert np.isclose(np.linalg.norm(vec), 1.0)


def test_hybridization_vectors_sp3(sp3_atom, molecular_field_calculator):
    vectors = molecular_field_calculator._get_hybridization_vectors(sp3_atom)

    assert len(vectors) == 4
    for vec in vectors:
        assert np.isclose(np.linalg.norm(vec), 1.0)


def test_hybridization_vectors_other(other_hybrid_atom, molecular_field_calculator):
    vectors = molecular_field_calculator._get_hybridization_vectors(other_hybrid_atom)

    assert len(vectors) == 3
    for vec in vectors:
        assert np.isclose(np.linalg.norm(vec), 1.0)


# === Fixture ===



# === Tests ===

def test_filter_positions_donor(ethanol_molecule, molecular_field_calculator):
    conf = ethanol_molecule.GetConformer()

    # Get position of atom 0 (carbon), and create test positions around it
    reference_idx = 0
    reference_pos = np.array(conf.GetAtomPosition(reference_idx))

    # Too close (should be filtered out)
    close_pos = reference_pos + np.array([0.5, 0.0, 0.0])

    # Far enough (should be retained)
    far_pos = reference_pos + np.array([3.0, 0.0, 0.0])

    test_positions = [close_pos, far_pos]

    filtered = molecular_field_calculator._filter_positions(
        mol=ethanol_molecule,
        positions=test_positions,
        is_donor=True,
        donor_acceptor_idx=1  # exclude atom 1 from proximity check
    )

    # Only far_pos should remain
    assert len(filtered) == 1
    assert np.allclose(filtered[0], far_pos)

def test_filter_positions_acceptor(ethanol_molecule, molecular_field_calculator):
    conf = ethanol_molecule.GetConformer()

    # Get position of oxygen atom (assumed index 2)
    oxygen_idx = 2
    oxygen_pos = np.array(conf.GetAtomPosition(oxygen_idx))

    # Too close to other atoms (<1.8 Å)
    close_pos = oxygen_pos + np.array([0.5, 0.5, 0.0])

    # Far enough
    far_pos = oxygen_pos + np.array([2.0, 2.0, 0.0])

    test_positions = [close_pos, far_pos]

    filtered = molecular_field_calculator._filter_positions(
        mol=ethanol_molecule,
        positions=test_positions,
        is_donor=False,
        donor_acceptor_idx=oxygen_idx
    )

    assert len(filtered) == 1
    assert np.allclose(filtered[0], far_pos)