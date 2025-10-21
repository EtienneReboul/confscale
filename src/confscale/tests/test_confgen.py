"""Conformer Generation Tests.

This module contains tests for the conformer generation functionality from the
confscale package. It verifies the conversion of SMILES to molecules with hydrogens,
the generation of 3D conformers, and ensures the reproducibility of conformer
generation with fixed seeds.

The tests cover both valid and invalid inputs, the impact of different parameters
on conformer generation, and the consistency of results with the same parameters.
"""

import pytest

from confscale.confgen import embeding3D_wrapper
from confscale.confgen import smi2molhbin


def test_smi2molh_valid_smiles():
    """Test conversion of valid SMILES string to RDKit Mol with hydrogens."""
    # Test with simple molecules
    mol = smi2molhbin("CCO")
    assert mol is not None
    assert mol.GetNumAtoms() > 0

    # Test a more complex molecule
    mol = smi2molhbin("c1ccccc1")
    assert mol is not None
    assert mol.GetNumAtoms() > 0


def test_smi2molh_invalid_smiles():
    """Test error handling for invalid SMILES string."""
    with pytest.raises(ValueError, match="Invalid SMILES string"):
        smi2molhbin("invalid_smiles")


def test_embeding3D_wrapper():
    """Test generation of 3D conformers."""
    # Create a molecule for testing
    mol = smi2molhbin("CCO")

    # Test with different conformer numbers
    for nb_confs in [1, 5]:
        mol_with_confs = embeding3D_wrapper(mol, nb_conformers=nb_confs, seed=42)
        assert mol_with_confs.GetNumConformers() == nb_confs


def test_embeding3D_reproducibility():
    """Test reproducibility of conformer generation with same seed."""
    mol = smi2molhbin("CCO")

    # Generate conformers with the same seed
    mol1 = embeding3D_wrapper(mol, nb_conformers=1, seed=42)
    mol2 = embeding3D_wrapper(mol, nb_conformers=1, seed=42)

    # Get coordinates of first conformer
    conf1 = mol1.GetConformer(0)
    conf2 = mol2.GetConformer(0)

    # Compare positions of the first atom
    pos1 = conf1.GetAtomPosition(0)
    pos2 = conf2.GetAtomPosition(0)

    assert pos1.x == pytest.approx(pos2.x)
    assert pos1.y == pytest.approx(pos2.y)
    assert pos1.z == pytest.approx(pos2.z)


def test_embeding3D_threads_parameter():
    """Test that different thread values don't cause errors."""
    mol = smi2molhbin("CCO")

    # Test with different thread values
    for threads in [1, 4]:
        mol_with_confs = embeding3D_wrapper(mol, nb_conformers=1, seed=42, nb_thread=threads)
        assert mol_with_confs.GetNumConformers() == 1


def test_embeding3D_forcetol_parameter():
    """Test that different forcetol values don't cause errors."""
    mol = smi2molhbin("CCO")

    # Test with different force tolerance values
    for forcetol in [0.01, 0.02]:
        mol_with_confs = embeding3D_wrapper(mol, nb_conformers=1, seed=42, forcetol=forcetol)
        assert mol_with_confs.GetNumConformers() == 1
