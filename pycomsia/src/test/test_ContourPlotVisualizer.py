import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from pycomisa.src.ContourPlotVisualizer import ContourPlotVisualizer


@pytest.fixture
def contour_visualizer():
    return ContourPlotVisualizer()


def test_calculate_significant_ranges_basic(contour_visualizer):
    data = {
        'field1': np.array([[1, 2, 3], [4, 5, 6]]),
        'field2': np.array([[10, 20, 30], [40, 50, 60]])
    }

    result = contour_visualizer.calculate_significant_ranges(data, top_percent=10, bottom_percent=10)

    # Only 10% of 6 elements => 0.6 => 0 (int cast)
    assert result['field1']['low'] == (None, None)
    assert result['field1']['high'] == (None, None)
    assert result['field2']['low'] == (None, None)
    assert result['field2']['high'] == (None, None)


def test_calculate_significant_ranges_50_percent(contour_visualizer):
    data = {
        'field1': np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    }

    result = contour_visualizer.calculate_significant_ranges(data, top_percent=50, bottom_percent=50)

    flat = data['field1'].flatten()
    sorted_flat = np.sort(flat)

    bottom_index = int(len(sorted_flat) * 50 / 100)
    top_index = int(len(sorted_flat) * (1 - 50 / 100))

    expected_low = sorted_flat[:bottom_index]
    expected_high = sorted_flat[top_index:]

    assert result['field1']['low'] == (np.min(expected_low), np.max(expected_low))
    assert result['field1']['high'] == (np.min(expected_high), np.max(expected_high))


def test_calculate_significant_ranges_edge_case_empty_range(contour_visualizer):
    data = {
        'field1': np.array([[1]])
    }

    result = contour_visualizer.calculate_significant_ranges(data, top_percent=90, bottom_percent=90)

    # With only one element, both top and bottom ranges will be empty
    assert result['field1']['low'] == (None, None)
    assert result['field1']['high'] == (None, None)


@patch("pyvista.ImageData")
@patch("pyvista.Plotter")
def test_visualize_contour_plots(mock_plotter_cls, mock_imagedata_cls, contour_visualizer, tmp_path):
    # Setup mocks
    mock_grid = MagicMock()
    mock_contour = MagicMock()
    mock_contour.interpolate.return_value = "interpolated_contour"

    mock_grid.contour.return_value = mock_contour
    mock_imagedata_cls.return_value = mock_grid

    mock_plotter = MagicMock()
    mock_plotter_cls.return_value = mock_plotter

    # Input values
    field_data = np.random.rand(2 * 2 * 2)  # Flat field data
    reconstructed_coeffs = {'steric_field': field_data}
    grid_dimensions = (2, 2, 2)
    grid_origin = (0.0, 0.0, 0.0)
    grid_spacing = (1.0, 1.0, 1.0)
    output = tmp_path

    mol = MagicMock()

    contour_visualizer.visualize_contour_plots(
        mol, reconstructed_coeffs, grid_dimensions, grid_origin, grid_spacing, str(output)
    )

    # Assertions
    assert mock_imagedata_cls.called
    assert mock_plotter_cls.called
    assert mock_grid.contour.call_count == 2  # Low and high
    assert mock_plotter.add_mesh.call_count == 2
    assert mock_plotter.screenshot.called
    assert mock_plotter.close.called


@patch("pyvista.Sphere")
@patch("pyvista.PolyData")
@patch("pyvista.Tube")
@patch("rdkit.Chem.RemoveHs")
def test_add_molecule_to_plot(mock_remove_hs, mock_tube, mock_polydata, contour_visualizer, mock_sphere):
    # Create mock molecule and plotter
    mock_plotter = MagicMock()
    mock_mol = MagicMock()
    mock_remove_hs.return_value = mock_mol

    # Mock atoms
    atom1 = MagicMock()
    atom1.GetIdx.return_value = 0
    atom1.GetAtomicNum.return_value = 6
    atom2 = MagicMock()
    atom2.GetIdx.return_value = 1
    atom2.GetAtomicNum.return_value = 1
    mock_mol.GetAtoms.return_value = [atom1, atom2]

    # Mock conformer and positions
    mock_conformer = MagicMock()
    mock_conformer.GetAtomPosition.side_effect = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    mock_mol.GetConformer.return_value = mock_conformer

    # Mock bonds
    bond = MagicMock()
    bond.GetBeginAtomIdx.return_value = 0
    bond.GetEndAtomIdx.return_value = 1
    mock_mol.GetBonds.return_value = [bond]

    # Mock PyVista behavior
    mock_sphere.return_value = "sphere"
    mock_glyph = MagicMock()
    mock_polydata.return_value.glyph.return_value = mock_glyph
    mock_line = MagicMock()
    mock_tube.return_value = mock_line

    # Run method
    contour_visualizer._add_molecule_to_plot(mock_plotter, mock_mol)

    # Assert atoms added
    assert mock_plotter.add_mesh.call_count == 3  # 2 atoms + 1 bond
    mock_plotter.add_mesh.assert_any_call(mock_glyph, color='white')
    mock_plotter.add_mesh.assert_any_call(mock_line, color='gray', smooth_shading=True)

    # Assert RDKit RemoveHs used
    mock_remove_hs.assert_called_once_with(mock_mol)


@pytest.mark.parametrize("atomic_num, expected_color", [
    (1, 'white'),       # Hydrogen
    (6, 'silver'),      # Carbon
    (7, 'lightblue'),   # Nitrogen
    (8, 'red'),         # Oxygen
    (9, 'lightgray'),   # Fluorine (not in map)
    (0, 'lightgray'),   # Dummy atom
    (16, 'lightgray'),  # Sulfur (not in map)
    (-1, 'lightgray'),  # Invalid atomic number
])
def test_get_atom_color(atomic_num, contour_visualizer, expected_color):
    assert contour_visualizer._get_atom_color(atomic_num) == expected_color


@pytest.mark.parametrize("bond_type, expected_color", [
    (1.0, 'silver'),   # Single bond
    (1.5, 'silver'),   # Aromatic bond
    (2.0, 'silver'),   # Double bond
    (3.0, 'silver'),   # Triple bond
    (4.0, 'silver'),   # Unknown bond type
    (None, 'silver')   # Invalid or missing bond type
])
def test_get_bond_color(bond_type, contour_visualizer, expected_color):
    # Create a mock bond
    mock_bond = MagicMock()
    mock_bond.GetBondTypeAsDouble.return_value = bond_type

    assert contour_visualizer._get_bond_color(mock_bond) == expected_color
