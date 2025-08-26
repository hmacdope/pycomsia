from unittest.mock import Mock, patch

import pytest
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from unittest import mock

from pycomsia.src.MolecularVisualizer import MolecularVisualizer


@pytest.fixture
def molecular_visualizer():
    return MolecularVisualizer()


# Helper: create a simple molecule with 3D coords
def create_dummy_molecule():
    mol = Chem.MolFromSmiles("CC")  # Ethane
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


@mock.patch("pyvista.Plotter")
def test_visualize_aligned_molecules(mock_plotter_class, tmp_path, molecular_visualizer):
    # Arrange
    mol = create_dummy_molecule()
    aligned_molecules = [mol]

    # Set up mock plotter instance and its methods
    mock_plotter = mock.Mock()
    mock_plotter_class.return_value = mock_plotter

    # Create output directory structure
    outputdir = tmp_path
    alignments_dir = outputdir / "Alignments"
    alignments_dir.mkdir()

    # Act
    molecular_visualizer.visualize_aligned_molecules(aligned_molecules, str(outputdir))

    # Assert: Check correct calls to pyvista.Plotter
    mock_plotter_class.assert_called_once_with(off_screen=True, border_width=2)
    mock_plotter.screenshot.assert_called_once_with(
        f"{outputdir}/Alignments/aligned_molecules.png", scale=5
    )
    mock_plotter.close.assert_called_once()
    assert molecular_visualizer.plotter.camera_position == 'iso'


@mock.patch("pyvista.Plotter")
def test_visualize_field(mock_plotter_class, tmp_path, molecular_visualizer):
    mol = create_dummy_molecule()

    # Dummy data
    grid_shape = (10, 10, 10)
    grid_spacing = (1.0, 1.0, 1.0)
    grid_origin = (0.0, 0.0, 0.0)
    field_values = {
        "electrostatic": np.random.rand(1000)
    }

    # Create output dir
    output_dir = tmp_path
    (output_dir / "Field_Plots").mkdir()

    # Mock plotter methods
    mock_plotter = mock.Mock()
    mock_plotter_class.return_value = mock_plotter

    # Patch internal methods
    with mock.patch.object(molecular_visualizer, "_get_visualization_params", return_value=([0.5], 'coolwarm')) as mock_vis_params, \
         mock.patch.object(molecular_visualizer, "_add_molecule_to_plot") as mock_add_mol, \
         mock.patch.object(molecular_visualizer, "_add_finishing_touches") as mock_add_finish:

        molecular_visualizer.visualize_field(
            mol,
            grid_spacing,
            grid_shape,
            grid_origin,
            field_values,
            str(output_dir)
        )

        # Checks
        mock_plotter_class.assert_called_once_with(off_screen=True, border_width=0)
        mock_plotter.add_volume.assert_called_once()
        mock_plotter.screenshot.assert_called_once_with(f"{output_dir}/Field_Plots/electrostatic.png", scale=5)
        mock_plotter.close.assert_called_once()
        mock_vis_params.assert_called_once_with("electrostatic", mock.ANY)
        mock_add_mol.assert_called_once_with(mol)
        mock_add_finish.assert_called_once_with("electrostatic")


def test_add_finishing_touches_without_title(molecular_visualizer):
    molecular_visualizer.plotter = Mock()

    # Act
    molecular_visualizer._add_finishing_touches()

    # Assert
    assert molecular_visualizer.plotter.camera_position == 'iso'
    molecular_visualizer.plotter.add_bounding_box.assert_not_called()
    molecular_visualizer.plotter.add_text.assert_not_called()


def test_add_finishing_touches_with_known_title(molecular_visualizer):
    molecular_visualizer.plotter = Mock()

    # Act
    molecular_visualizer._add_finishing_touches('electrostatic')

    # Assert
    assert molecular_visualizer.plotter.camera_position == 'iso'
    molecular_visualizer.plotter.add_bounding_box.assert_called_once_with(
        line_width=1, color='lightgray', opacity=0.1, outline=False, culling='front'
    )
    molecular_visualizer.plotter.add_text.assert_called_once_with(
        'Electrostatic Field', position='upper_edge', font_size=100, color='black', font='arial'
    )


def test_add_finishing_touches_with_custom_title(molecular_visualizer):
    molecular_visualizer.plotter = Mock()

    # Act
    molecular_visualizer._add_finishing_touches('custom_title')

    # Assert
    assert molecular_visualizer.plotter.camera_position == 'iso'
    molecular_visualizer.plotter.add_bounding_box.assert_called_once()
    molecular_visualizer.plotter.add_text.assert_called_once_with(
        'custom_title', position='upper_edge', font_size=100, color='black', font='arial'
    )


@patch("pyvista.Sphere")
@patch("pyvista.PolyData")
@patch("pyvista.Tube")
def test_add_molecule_to_plot(mock_tube, mock_polydata, mock_sphere, molecular_visualizer):
    # Arrange
    molecular_visualizer.plotter = Mock()

    mol = create_dummy_molecule()

    # Mock glyph result
    mock_glyph = Mock()
    mock_polydata.return_value.glyph.return_value = mock_glyph
    mock_line = Mock()
    mock_tube.return_value = mock_line

    # Act
    molecular_visualizer._add_molecule_to_plot(mol)

    # RemoveHs removes Hs, so we only get the two carbon atoms
    expected_atom_calls = 2  # Only C atoms
    expected_bond_calls = 1  # One bond between two C atoms

    # Assert
    assert molecular_visualizer.plotter.add_mesh.call_count == expected_atom_calls + expected_bond_calls

    # Assert atoms were added with color and without smooth shading
    atom_calls = molecular_visualizer.plotter.add_mesh.call_args_list[:expected_atom_calls]
    for call in atom_calls:
        args, kwargs = call
        assert "color" in kwargs
        assert "smooth_shading" not in kwargs  # Not used for atoms

    # Assert bond was added with correct kwargs
    bond_call = molecular_visualizer.plotter.add_mesh.call_args_list[-1]
    args, kwargs = bond_call
    assert "color" in kwargs
    assert kwargs.get("smooth_shading") is True


def test_electrostatic_with_data(molecular_visualizer):
    data = np.random.uniform(-5, 5, size=1000)
    opacity, cmap = molecular_visualizer._get_visualization_params("electrostatic", data)

    assert isinstance(opacity, list)
    assert len(opacity) == 6
    assert all(0.0 <= o <= 0.5 for o in opacity)
    assert cmap == 'coolwarm'


def test_hydrophobic_with_data(molecular_visualizer):
    data = np.random.normal(0, 1, size=1000)
    opacity, cmap = molecular_visualizer._get_visualization_params("hydrophobic", data)

    assert isinstance(opacity, list)
    assert len(opacity) == 6
    assert all(0.0 <= o <= 0.5 for o in opacity)
    assert cmap == 'viridis'


@pytest.mark.parametrize("field_name, expected_opacity, expected_cmap", [
    ('electrostatic', [0.5, 0.3, 0.0, 0.0, 0.3, 0.5], 'coolwarm'),
    ('steric', [0.6, 0.5, 0.4, 0.3, 0.2, 0], 'hot'),
    ('hydrophobic', [0.6, 0.3, 0.0, 0.3, 0.6], 'viridis'),
    ('hbond_acceptor', [0, 0.1, 0.3, 0.4, 0.5, 0.6], 'inferno'),
    ('hbond_donor', [0, 0.1, 0.3, 0.4, 0.5, 0.6], 'cividis')
])
def test_predefined_fields_without_data(field_name, expected_opacity, expected_cmap, molecular_visualizer):
    opacity, cmap = molecular_visualizer._get_visualization_params(field_name, None)

    assert opacity == expected_opacity
    assert cmap == expected_cmap


def test_unknown_field_without_data(molecular_visualizer):
    opacity, cmap = molecular_visualizer._get_visualization_params("unknown_field", None)

    assert opacity == [0, 0.1, 0.3, 0.4, 0.5, 0.6]
    assert cmap == "viridis"



# Parametrized test for known elements
@pytest.mark.parametrize("atomic_num, expected_color", [
    (1, 'white'),      # H
    (6, 'silver'),     # C
    (7, 'lightblue'),  # N
    (8, 'red'),        # O
])
def test_get_atom_color_known(atomic_num, expected_color, molecular_visualizer):
    assert molecular_visualizer._get_atom_color(atomic_num) == expected_color


# Test for unknown element
@pytest.mark.parametrize("atomic_num", [9, 12, 16, 20, 26])
def test_get_atom_color_unknown(atomic_num, molecular_visualizer):
    assert molecular_visualizer._get_atom_color(atomic_num) == 'lightgray'


# Test known bond types
@pytest.mark.parametrize("bond_type, expected_color", [
    (1.0, 'silver'),      # Single
    (1.5, 'lightblue'),   # Aromatic
    (2.0, 'lightgray'),   # Double
    (3.0, 'red'),         # Triple
])
def test_get_bond_color_known(bond_type, expected_color, molecular_visualizer):
    assert molecular_visualizer._get_bond_color(bond_type) == expected_color


# Test unknown bond types
@pytest.mark.parametrize("bond_type", [0.0, 4.0, 5.5, None])
def test_get_bond_color_unknown(bond_type, molecular_visualizer):
    assert molecular_visualizer._get_bond_color(bond_type) == 'silver'


# Test default normalization to [0, 1]
def test_normalize_default_range(molecular_visualizer):
    data = np.array([2, 4, 6, 8])
    result = molecular_visualizer._custom_normalize_field(data)

    assert np.allclose(result, [0.0, 0.3333, 0.6666, 1.0], atol=1e-3)


# Test normalization to [-1, 1]
def test_normalize_custom_range_minus1_to_1(molecular_visualizer):
    data = np.array([10, 20, 30])
    result = molecular_visualizer._custom_normalize_field(data, new_min=-1, new_max=1)

    assert np.allclose(result, [-1.0, 0.0, 1.0], atol=1e-3)


# Test normalization to [10, 20]
def test_normalize_custom_range_10_to_20(molecular_visualizer):
    data = np.array([0, 5, 10])
    result = molecular_visualizer._custom_normalize_field(data, new_min=10, new_max=20)

    assert np.allclose(result, [10.0, 15.0, 20.0], atol=1e-3)


# Edge case: all values are the same
def test_normalize_single_unique_value(molecular_visualizer):
    data = np.array([7, 7, 7])

    with pytest.raises(ZeroDivisionError):
        molecular_visualizer._custom_normalize_field(data)
