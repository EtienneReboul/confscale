"""Geometry utility functions for analyzing molecular conformations.

This module provides functions to compute and analyze dihedral angles in molecular conformations
using RDKit. It includes functionality for analyzing dihedral angles across multiple conformers
of a molecule, computing angles for specific conformers, and performing angular calculations
and transformations.

Functions
---------
compute_dihedrals_conformers : Calculate dihedral angles for a specific dihedral across all conformers
compute_dihedral_angle : Calculate dihedral angles for multiple dihedrals in a specific conformer
degree2cosin : Convert angles from degrees to cosine and sine values
smallest_distance_degree : Calculate the smallest angular distance between two angles

Notes
-----
The module assumes that molecules are provided as RDKit Mol objects with valid 3D coordinates.
Dihedral angles are normalized to the range [0, 360) degrees.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def compute_dihedrals_conformers(mol: Chem.Mol, dihedral_idces: tuple[int]) -> np.ndarray:
    """
    Compute dihedral angles for a specific dihedral across all conformers in a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object containing one or more conformers
    dihedral_idces : tuple[int]
        A tuple of 4 atom indices defining the dihedral angle to compute
    Returns
    -------
    np.ndarray
        Array of dihedral angles (in degrees) for the specified dihedral across all
        conformers in the molecule. Angles are normalized to be in the range [0, 360].
    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.MolFromSmiles('CCCC')  # n-butane
    >>> mol = Chem.AddHs(mol)
    >>> _ = AllChem.EmbedMultipleConfs(mol, numConfs=5)
    >>> dihedral_idces = (0, 1, 2, 3)  # C-C-C-C dihedral
    >>> angles = compute_dihedrals_conformers(mol, dihedral_idces)
    >>> # Check that the output is between 0 and 360 degrees
    >>> all(0 <= angle < 360 for angle in angles)
    True
    >>> # Check the shape of the output
    >>> angles.shape
    (5,)
    """
    # declare list
    dihedrals = []

    # loop over the conformers
    for conf in mol.GetConformers():
        # compute the dihedral angle
        angle = rdMolTransforms.GetDihedralDeg(conf, *dihedral_idces)
        angle = angle if angle >= 0 else angle + 360  # ensure angle is positive
        dihedrals.append(angle)

    # cast to numpy array
    return np.array(dihedrals)


def compute_dihedral_angle(mol: Chem.Mol, dihedrals_idces: list[tuple[int]], confid: int = -1) -> np.ndarray:
    """
    Compute dihedral angles for a set of atom quartets in a specific conformer of a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object containing 3D coordinates
    dihedrals_idces : list[tuple[int]]
        List of tuples, where each tuple contains four atom indices defining a dihedral angle
    confid : int, optional
        Conformer ID to use, by default -1 (the last conformer in the molecule)

    Returns
    -------
    np.ndarray
        Array of dihedral angles in degrees, with values normalized to [0, 360)

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.MolFromSmiles('CCCC')  # n-butane
    >>> mol = Chem.AddHs(mol)
    >>> _ = AllChem.EmbedMultipleConfs(mol, numConfs=1)
    >>> dihedral_idces = [(0, 1, 2, 3)]  # C-C-C-C dihedral
    >>> angles = compute_dihedral_angle(mol, dihedral_idces)
    >>> # Check that the output is between 0 and 360 degrees
    >>> all(0 <= angle < 360 for angle in angles)
    True

    >>> # Multiple dihedrals
    >>> mol = Chem.MolFromSmiles('CCCCO')  # n-butanol
    >>> mol = Chem.AddHs(mol)
    >>> _ = AllChem.EmbedMultipleConfs(mol, numConfs=1)
    >>> dihedral_idces = [(0, 1, 2, 3), (1, 2, 3, 4)]  # C-C-C-C and C-C-C-O dihedrals
    >>> angles = compute_dihedral_angle(mol, dihedral_idces)
    >>> print(f"Number of angles computed: {len(angles)}")
    Number of angles computed: 2
    """

    # declare list
    dihedrals = []

    # get conformer
    conf = mol.GetConformer(confid)

    # loop over the torsions
    for dihedrals_idx in dihedrals_idces:
        # compute the dihedral angle
        angle = rdMolTransforms.GetDihedralDeg(conf, *dihedrals_idx)
        angle = angle if angle >= 0 else angle + 360  # ensure angle is positive
        dihedrals.append(angle)

    # cast to numpy array
    return np.array(dihedrals)


def degree2cosin(angles: np.ndarray) -> np.ndarray:
    """Convert angles in degrees to cosines and sines.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in degrees.

    Returns
    -------
    np.ndarray
        2D array where each row contains the cosine and sine of the corresponding angle.
        The first column contains cosines and the second column contains sines.

    Examples
    --------
    >>> import numpy as np
    >>> angles = np.array([0, 90, 180])
    >>> result = degree2cosin(angles)
    >>> expected = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    >>> np.allclose(result, expected)
    True

    >>> result = degree2cosin(np.array([45]))
    >>> expected = np.array([[0.70710678, 0.70710678]])
    >>> np.allclose(result, expected)
    True
    """
    # convert to radians
    radians = np.deg2rad(angles)

    # compute cosines
    return np.column_stack((np.cos(radians), np.sin(radians)))


def smallest_distance_degree(exp_angle: float, ref_angle: float) -> float:
    """Calculate the smallest angle distance between two angles in degrees.

    Parameters
    ----------
    exp_angle : float
        Experimental angle in degrees
    ref_angle : float
        Reference angle in degrees

    Returns
    -------
    float
        The smallest distance in degrees between the two angles, considering the circular nature of angles.
    Examples
    --------
    >>> smallest_distance_degree(10, 350)
    20
    >>> smallest_distance_degree(350, 10)
    20
    >>> smallest_distance_degree(90, 180)
    90
    >>> smallest_distance_degree(180, 90)
    90
    >>> smallest_distance_degree(-10, 10)
    20
    """

    return min(360 - abs(exp_angle - ref_angle), abs(exp_angle - ref_angle))
