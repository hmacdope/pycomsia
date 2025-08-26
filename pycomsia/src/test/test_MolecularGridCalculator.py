import numpy as np
import pytest

from rdkit import Chem
from rdkit.Chem import AllChem

from pycomisa.src.MolecularGridCalculator import MolecularGridCalculator

@pytest.fixture
def molecular_grid_calculator():
    return MolecularGridCalculator()


def test_snap_coords_to_grid(molecular_grid_calculator):

    coords = np.array([
        [0.1, 0.2, 0.3],
        [1.49, 1.51, 1.99],
        [-0.49, -1.51, -2.49]
    ])
    grid_spacing = 1.0

    expected = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 2.0],
        [-0.0, -2.0, -2.0]
    ])

    snapped = molecular_grid_calculator._snap_coords_to_grid(coords, grid_spacing)

    # Use np.allclose for float comparisons
    assert np.allclose(snapped, expected), f"Expected {expected}, but got {snapped}"


# Helper to generate a molecule with known 3D coords
def create_test_molecule():
    mol = Chem.MolFromSmiles("CC")  # Ethane
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol

def test_generate_grid(molecular_grid_calculator):
    mol = create_test_molecule()
    aligned_results = [(mol, True)]

    resolution = 1.0
    padding = 2.0

    grid_spacing, grid_dimensions, grid_origin = molecular_grid_calculator.generate_grid(
        aligned_results, resolution=resolution, padding=padding
    )

    # Validate grid_spacing
    assert grid_spacing == (1.0, 1.0, 1.0)

    # Validate grid_dimensions are positive
    assert all(d > 0 for d in grid_dimensions)

    # Validate origin has padding subtracted (approximate check)
    coords = []
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    coords = np.array(coords)
    snapped_coords = molecular_grid_calculator._snap_coords_to_grid(coords, resolution)
    min_coords = np.min(snapped_coords, axis=0)
    expected_origin = tuple(min_coords - padding)

    assert np.allclose(grid_origin, expected_origin)

def test_generate_grid_empty_input(molecular_grid_calculator):
    with pytest.raises(ValueError, match="No valid molecules found"):
        molecular_grid_calculator.generate_grid([], resolution=1.0, padding=1.0)