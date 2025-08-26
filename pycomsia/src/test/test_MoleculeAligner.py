
import pytest

from rdkit import Chem

from pycomsia.src.MoleculeAligner import MoleculeAligner


@pytest.fixture
def molecule_aligner():
    return MoleculeAligner()


def test_align_molecules_valid(molecule_aligner):
    train = ["CCO", "CC(C)O"]
    result = molecule_aligner.align_molecules(train)
    assert isinstance(result, list)
    assert len(result) == 2
    for mol, is_train in result:
        assert mol is not None
        assert isinstance(is_train, bool)
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumConformers() > 0


def test_align_molecules_with_prediction(molecule_aligner):
    train = ["CCO", "CC(C)O"]
    pred = ["CCCC", "CCN"]
    result = molecule_aligner.align_molecules(train, pred)
    assert len(result) == 4
    assert all(isinstance(item[1], bool) for item in result)
    train_count = sum(1 for _, is_train in result if is_train)
    pred_count = sum(1 for _, is_train in result if not is_train)
    assert train_count == 2
    assert pred_count == 2


def test_align_molecules_empty_input(molecule_aligner):
    result = molecule_aligner.align_molecules([])
    assert result == []


def test_align_molecules_invalid_smiles(molecule_aligner):
    train = ["C1CC1", "INVALID"]
    result = molecule_aligner.align_molecules(train)
    assert result == []  # Invalid input should return empty list


def test_align_molecules_partial_invalid_prediction(molecule_aligner):
    train = ["C1CC1"]
    pred = ["INVALID"]
    result = molecule_aligner.align_molecules(train, pred)
    assert result == []  # Because one of the molecules is invalid


def test_align_molecules_mixed_sizes(molecule_aligner):
    train = ["C", "CCCCCCC", "CC"]
    result = molecule_aligner.align_molecules(train)
    largest_mol = max(result, key=lambda x: x[0].GetNumAtoms() if x[0] else 0)
    assert largest_mol[0].GetNumAtoms() >= 7


def test_get_template(molecule_aligner):
    molecule_aligner.template = "test_template"
    receive_template = molecule_aligner.get_template()
    assert receive_template == "test_template"


def test_get_core(molecule_aligner):
    molecule_aligner.core = "test_core"
    receive_core = molecule_aligner.get_core()
    assert receive_core == "test_core"
