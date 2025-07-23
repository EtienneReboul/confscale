"""Conformer Generation Module.

This module provides utilities for generating 3D conformers from SMILES strings
using RDKit's distance geometry algorithms. It includes functions to convert SMILES
to molecule objects with hydrogens and to embed molecules in 3D space with multiple
conformers.

Functions:
    smi2molh: Convert a SMILES string to an RDKit Mol object with hydrogens.
    embeding3D_wrapper: Generate 3D conformers for a given RDKit Mol object.

Examples:
    >>> from confscale.confgen import smi2molh, embeding3D_wrapper
    >>> mol = smi2molh('CCO')
    >>> mol_with_conformers = embeding3D_wrapper(mol, nb_conformers=10, seed=42)
"""

import time

from rdkit import Chem
from rdkit.Chem import rdDistGeom


def smi2molh(smi: str) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit Mol object with hydrogens added.

    Parameters
    ----------
    smi : str
        The SMILES string to convert.

    Returns
    -------
    Chem.Mol
        The RDKit Mol object with hydrogens added.

    Raises
    ------
    ValueError
        If the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smi}")
    mol = Chem.AddHs(mol)
    return mol


def embeding3D_wrapper(
    molh: Chem.Mol, nb_conformers: int, seed: int | None = None, nb_thread: int = 2, forcetol: float = 0.0135
) -> Chem.Mol:
    """
    Generate 3D conformers for a given RDKit Mol object
    Parameters
    ----------
    molh : Chem.Mol
        The RDKit Mol object with hydrogens added.
    nb_conformers : int
        The number of conformers to generate.
    seed : int or None, optional
        Random seed for reproducibility. If None, a time-based seed will be generated.
    nb_thread : int, default=2
        Number of threads to use for conformer generation.
    forcetol : float, default=0.0135
        Force tolerance for the embedding algorithm.

    Returns
    -------
    Chem.Mol
        The RDKit Mol object with generated 3D conformers.

    Notes
    -----
        - When seed is not provided, a time-based seed is generated using the current time
        multiplied by 1,000,000 and modulo 2^31 to ensure it fits within the range of a 32-bit
        signed integer, in order to avoid potential issues with large integers.
        - The `forcetol` parameter is set to 0.0135, in order
    """
    etkdg = rdDistGeom.ETKDGv3()
    etkdg.randomSeed = seed if seed is not None else int(time.time() * 1000000) % (2**31)
    etkdg.verbose = False
    etkdg.numThreads = nb_thread
    etkdg.useRandomCoords = True
    etkdg.optimizerForceTol = forcetol

    rdDistGeom.EmbedMultipleConfs(molh, numConfs=nb_conformers, params=etkdg)

    return molh
