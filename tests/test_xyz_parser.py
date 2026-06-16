from pathlib import Path

import pytest

from gpu4pyscf_h200_runner.io_utils import atoms_to_pyscf_atom_text, read_xyz


def test_standard_xyz(tmp_path: Path):
    path = tmp_path / "water.xyz"
    path.write_text("3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n", encoding="utf-8")
    atoms = read_xyz(path)
    assert len(atoms) == 3
    assert atoms[0].symbol == "O"
    assert "O" in atoms_to_pyscf_atom_text(atoms)


def test_simple_coordinate_file(tmp_path: Path):
    path = tmp_path / "coords.xyz"
    path.write_text("C 0.0 0.0 0.0\nH 0.0 0.0 1.0\n", encoding="utf-8")
    atoms = read_xyz(path)
    assert len(atoms) == 2
    assert atoms[1].z == 1.0


def test_invalid_coordinate_file(tmp_path: Path):
    path = tmp_path / "bad.xyz"
    path.write_text("C 0.0 nope 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="x/y/z"):
        read_xyz(path)


def test_standard_xyz_atom_count_mismatch(tmp_path: Path):
    path = tmp_path / "bad_count.xyz"
    path.write_text("3\nbad\nO 0 0 0\nH 0 0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="atom count mismatch"):
        read_xyz(path)

