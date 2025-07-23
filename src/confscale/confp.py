"""
The `confscale.confp` module provides tools for analyzing molecular conformers
using dihedral angles and creating conformational fingerprints.

This module implements functionality for:
- Finding and analyzing rotatable bonds in molecules
- Computing dihedral angles across multiple conformers
- Clustering similar conformational states based on dihedral angle patterns
- Creating binary fingerprints that capture molecular flexibility

The module helps identify, analyze and compare 3D conformations of molecules by
examining their torsional preferences and rotational patterns.

Main features:
- Identifying rotatable bonds and enumerating their corresponding dihedral angles
- Computing dihedral angles across multiple molecular conformations
- Clustering dihedral angles using DBSCAN in cosine-sine space to handle angular periodicity
- Generating binary conformational fingerprints based on torsional preferences
- Handling molecular symmetry through canonical atom indexing
- Statistical analysis of angle distributions including circular mean and standard deviation
- Congruence checking for equivalent dihedrals in symmetric structures
- Integration with ETKDG (Experimental-Torsion Knowledge Distance Geometry) for reference angles

The module relies on RDKit for molecular representations and conformer handling,
and integrates with NumPy and SciPy for numerical processing, scikit-learn for
clustering algorithms, and pandas for data organization.
"""

from collections import Counter
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Lipinski import RotatableBondSmarts
from scipy.spatial.distance import pdist
from scipy.stats import circmean
from scipy.stats import circstd
from sklearn.cluster import DBSCAN


def check_conformers(mol: Chem.Mol) -> None:
    """
    Check if a molecule has at least 2 conformers.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule object to check.

    Raises
    ------
    ValueError
        If the molecule has less than 2 conformers.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> mol = Chem.AddHs(mol)
    >>> _=AllChem.EmbedMultipleConfs(mol, numConfs=2)
    >>> check_conformers(mol)  # Should not raise an error
    >>> mol_single_conf = Chem.MolFromSmiles('C')
    >>> mol_single_conf = Chem.AddHs(mol_single_conf)
    >>> try:
    ...     check_conformers(mol_single_conf)  # Should raise ValueError
    ...     print("No error raised")  # This shouldn't execute
    ... except ValueError:
    ...     print("ValueError correctly raised")  # This should execute
    ValueError correctly raised
    """
    if mol.GetNumConformers() < 2:
        raise ValueError("The molecule must have at least 2 conformers to compute the fingerprint.")


def findrotatable_bonds(mol: Chem.Mol, smarts: str | Chem.Mol = RotatableBondSmarts) -> tuple[tuple[int]]:
    """
    Find rotatable bonds in a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to find rotatable bonds.

    Returns
    -------
    tuple[tuple[int]]
        A tuple of tuples, where each inner tuple contains the indices of the atoms forming a rot
    Examples
    --------
    >>> from rdkit import Chem
    >>> ethane = Chem.MolFromSmiles('CCCC')
    >>> rotatable_bonds = findrotatable_bonds(ethane)
    >>> print(rotatable_bonds)
    ((1, 2),)
    """
    # find rotatable bonds
    rotatable_bonds = mol.GetSubstructMatches(smarts)

    # cast to list of tuples
    return rotatable_bonds


def index2hybridization(mol: Chem.Mol, atom_idx: int) -> str:
    """
    Get the hybridization of atoms in a molecule based on their indices.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to analyze.
    atom_idces : tuple[int]
        A tuple of atom indices for which to retrieve hybridization.

    Returns
    -------
    set[str]
        A set of hybridization types for the specified atoms.

    Examples
    --------
    >>> from rdkit import Chem
    >>> ethane = Chem.MolFromSmiles('CCCC')
    >>> hybridization = index2hybridization(ethane, 0)
    >>> print(hybridization)
    SP3
    """
    # get the atom
    atom = mol.GetAtomWithIdx(atom_idx)
    return str(atom.GetHybridization())


def enumerate_dihedrals(mol: Chem.Mol, rotatable_idces: tuple[tuple[int]]) -> list[tuple[int]]:
    """
    Find atomic indices for one dihedral angle for each rotatable bond in the molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to enumerate torsions.
    rotatable_idces : tuple[tuple[int]]
        A tuple of tuples of atom indices defining the rotatable bonds.

    Returns
    -------
    list[tuple[int]]
        A list of quadruplet atoms index that define the dihedral angles.

    Examples
    --------
    >>> from rdkit import Chem
    >>> ethane = Chem.MolFromSmiles('CCCC')
    >>> rotatable_bonds = findrotatable_bonds(ethane)
    >>> dihedral_indices = enumerate_dihedrals(ethane, rotatable_bonds)
    >>> print(dihedral_indices)
    [(0, 1, 2, 3)]
    """
    # declare the list
    dihedral_idces = []
    allowedhybridization = {"SP2", "SP3"}

    # loop over the matches
    for duet in rotatable_idces:
        # loop variables
        found_dihedral = False

        # unpack atoms index
        idx2, idx3 = duet

        # retrieve the bond
        bond = mol.GetBondBetweenAtoms(idx2, idx3)

        # get the atoms
        jAtom = mol.GetAtomWithIdx(idx2)
        kAtom = mol.GetAtomWithIdx(idx3)

        # get hybridization of the atoms
        hybridization_set = {str(jAtom.GetHybridization()), str(kAtom.GetHybridization())}

        # check if the atoms are sp2 or sp3
        if not hybridization_set.issubset(allowedhybridization):
            continue

        # iterate over the bond shared by the first atom of rotatable bond
        for b1 in jAtom.GetBonds():
            if found_dihedral:
                break

            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)

            # check if the hybridization of the first atom is allowed
            if index2hybridization(mol, idx1) not in allowedhybridization:
                continue

            # iterate over the bond shared by the second atom of rotatable bond
            for b2 in kAtom.GetBonds():
                # skip rotable bond and if second examine bond is the same as the first bond
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue

                idx4 = b2.GetOtherAtomIdx(idx3)

                # skip 3-membered rings
                if idx4 == idx1:
                    continue

                # check if the hybridization of the fourth atom is allowed
                if index2hybridization(mol, idx4) not in allowedhybridization:
                    continue

                dihedral_idces.append((idx1, idx2, idx3, idx4))
                found_dihedral = True
                break

    return dihedral_idces


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


def get_etkdg_info(mol: Chem.Mol) -> list[dict]:
    """Get ETKDG parameters from a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get ETKDG parameters from.

    Returns
    -------
    list[dict]
        A list of dictionaries containing ETKDG parameters, including:
        - "smarts": The SMARTS pattern for the dihedral
        - "atomIndices": The atom indices defining the dihedral angle
        - Additional parameters related to the experimental torsion distribution

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> # Create a cyclohexane molecule
    >>> cyclohexane = Chem.MolFromSmiles('C1CCCCC1')
    >>> cyclohexane = Chem.AddHs(cyclohexane)
    >>> _ = AllChem.EmbedMultipleConfs(cyclohexane, numConfs=1)
    >>> # Get ETKDG information
    >>> infos = get_etkdg_info(cyclohexane)
    >>> # Print the first entry to show SMARTS pattern and atom indices
    >>> print(f"SMARTS: {infos[0]['smarts']}")
    SMARTS: [!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]
    >>> print(f"Atom indices: {list(infos[0]['atomIndices'])}")
    Atom indices: [0, 1, 2, 3]
    >>> # Length of returned infos shows total number of dihedrals found
    >>> print(f"Number of dihedrals: {len(infos)}")
    Number of dihedrals: 6
    """

    ps = rdDistGeom.ETKDGv3()
    ps.verbose = False
    ps.useSmallRingTorsions = True
    ps.useMacrocycleTorsions = True
    infos = rdDistGeom.GetExperimentalTorsions(mol, ps)

    return infos


def get_idces2canidces(mol: Chem.Mol) -> dict[int, int]:
    """Get canonical mapping of a molecule without breaking ties.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the canonical mapping from.

    Returns
    -------
    dict
        A dictionary mapping original atom indices to canonical atom indices.
        This is used to handle molecular symmetry.

    Notes
    -----
    The keys are the original atom indices, and the values are the canonical atom indices.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CC')  # ethane
    >>> mapping = get_idces2canidces(mol)
    >>> print(mapping)  # Both carbons have the same canonical rank
    {0: 0, 1: 0}

    >>> mol = Chem.MolFromSmiles('CCO')  # ethanol
    >>> mapping = get_idces2canidces(mol)
    >>> print(mapping)  # Oxygen has different rank from carbons
    {0: 0, 1: 2, 2: 1}

    >>> mol = Chem.MolFromSmiles('c1ccccc1')  # benzene (all atoms equivalent)
    >>> mapping = get_idces2canidces(mol)
    >>> print(mapping)  # All carbons have same canonical rank
    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    """
    # Get the canonical mapping
    idces2canidces = Chem.CanonicalRankAtoms(mol, breakTies=False)

    return dict(enumerate(idces2canidces))


def dihedralidces2sortcandices(dihedral_idces: tuple[int] | list[int], idces2canidces: dict[int, int]) -> tuple[int]:
    """
    Convert the atomic indices of dihedral angles to their sorted canonical indices.

    Parameters
    ----------
    dihedral_idces : tuple[int]
        A tuple containing four atom indices defining a dihedral angle.
    idces2canidces : dict[int, int]
        A mapping from atom indices to their canonical indices.

    Returns
    -------
    tuple[int]
        A tuple of canonical indices of the atoms involved in the dihedral angle,
        sorted in ascending order. The first index is always less than or equal to the last index.

    Examples
    --------
    >>> from rdkit import Chem
    >>> # Example with a simple molecule
    >>> mol = Chem.MolFromSmiles('CCCC')  # butane
    >>> idces2canidces = get_idces2canidces(mol)
    >>> print(idces2canidces)  # The two end carbons have same canonical index, middle ones too
    {0: 0, 1: 2, 2: 2, 3: 0}
    >>> dihedral_idces = (0, 1, 2, 3)
    >>> result = dihedralidces2sortcandices(dihedral_idces, idces2canidces)
    >>> print(result)  # Should preserve order since first index <= last index
    (0, 2, 2, 0)

    >>> # Example with order reversal needed
    >>> mol = Chem.MolFromSmiles('CCC(C)O')  # 2-butanol
    >>> idces2canidces = get_idces2canidces(mol)
    >>> print(idces2canidces)  # Different canonical ranks for different atoms
    {0: 0, 1: 3, 2: 4, 3: 1, 4: 2}
    >>> dihedral_idces = (4, 2, 1, 0)  # O-C-C-C dihedral (reversed order)
    >>> result = dihedralidces2sortcandices(dihedral_idces, idces2canidces)
    >>> print(result)  # Should reverse order since canonical index of first atom (4->2) > last atom (0->0)
    (0, 3, 4, 2)

    """
    # convert the dihedral indices to canonical indices
    can_dices = [idces2canidces.get(idx) for idx in dihedral_idces]

    if can_dices[0] > can_dices[-1]:
        can_dices.reverse()
    # sort the dihedral indices

    return tuple(can_dices)


def get_angle_identifiers(info: dict, idces2canidces: dict[int, int]) -> dict:
    """
    This function processes angle information to extract and organize identifiers
    related to angles in a molecular structure.

    Parameters
    ----------
    info : dict
        A dictionary containing angle information, specifically the 'atomIndices' field.
    idces2canidces : dict[int, int]
        A mapping from atom indices to canonical atom indices.

    Returns
    -------
    dict
        A dictionary containing angle identifiers with the following keys:
        - "canidces": A tuple of canonical indices of the atoms involved in the angle,
                        ordered such that the first index is less than or equal to the last index.
        - "current_idces": A tuple containing the original atom indices.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCCC')  # butane
    >>> idces2canidces = get_idces2canidces(mol)
    >>> info = {'atomIndices': [0, 1, 2, 3]}  # C-C-C-C dihedral
    >>> result = get_angle_identifiers(info, idces2canidces)
    >>> print(result['current_idces'])
    (0, 1, 2, 3)
    >>> print(result['midbond_atomidces'])
    frozenset({1, 2})
    >>> print(result['canidces'])  # Canonical indices
    (0, 2, 2, 0)

    >>> # Example with a different molecule
    >>> mol = Chem.MolFromSmiles('CCO')  # ethanol
    >>> idces2canidces = get_idces2canidces(mol)
    >>> info = {'atomIndices': [0, 1, 2]}  # C-C-O angle
    >>> result = get_angle_identifiers(info, idces2canidces)
    >>> print(result['current_idces'])
    (0, 1, 2)
    >>> print(result['midbond_atomidces'])
    frozenset({1, 2})
    >>> print(result['canidces'])
    (0, 2, 1)

    >>> # Example with reversed order
    >>> info = {'atomIndices': [2, 1, 0]}  # O-C-C angle (reversed)
    >>> result = get_angle_identifiers(info, idces2canidces)
    >>> print(result['current_idces'])
    (2, 1, 0)
    >>> print(result['midbond_atomidces'])
    frozenset({0, 1})
    >>> print(result['canidces'])  # Canonical indices with potential reordering
    (0, 2, 1)
    """
    # declare dict
    angle_identifiers = {
        "canidces": tuple,
        "midbond_atomidces": frozenset,
        "current_idces": tuple,
    }

    angle_idces = list(info["atomIndices"])
    angle_identifiers["current_idces"] = tuple(angle_idces)
    angle_identifiers["midbond_atomidces"] = frozenset(angle_idces[1:3])
    angle_identifiers["canidces"] = dihedralidces2sortcandices(angle_idces, idces2canidces)
    return angle_identifiers


def package_metadata(log: dict, smarts_df: pd.DataFrame, angle_identifier: dict) -> dict:
    """
    Extract metadata for a conformational search based on bond rotation.

    This function retrieves information about a rotatable bond including its atom indices,
    possible rotation angles, and other relevant properties from a SMARTS pattern database.

    Parameters
    ----------
    log : dict
        Dictionary containing at least a 'smarts' key with the SMARTS pattern for the bond.
    smarts_df : pd.DataFrame
        DataFrame containing SMARTS patterns and their associated conformational properties,
        including angle information, multiplicity, and bond type.
    angle_identifier : dict
        Dictionary containing at least 'midbond_atomidces' (indices of the middle atoms defining
        the rotatable bond) and 'current_idces' (current atom indices involved in the rotation).

    Returns
    -------
    dict
        A metadata dictionary containing:
        - 'midbond_atomidces': Set of atom indices defining the rotatable bond
        - 'current_idces': List of current atom indices involved
        - 'smarts': SMARTS pattern describing the bond
        - 'type': Type of the rotatable bond (can be 'macrocycle', 'general', 'fallback', or 'smallring')
        - 'expected_angles': List of possible dihedral angles (in degrees) for the rotation
        - 'multiplicity': Number of preferred rotational positions

    Notes
    -----
    If the provided SMARTS pattern is not found in the dataframe, a fallback generic
    rotatable bond pattern will be used with a starting point of 30° followed by 60° angle increments.
    """
    # declare the metadata dictionary
    metadata = {
        "midbond_atomidces": {angle_identifier["midbond_atomidces"]},
        "current_idces": [angle_identifier["current_idces"]],
    }
    # find if there is a match in the smarts dataframe
    if log["smarts"] not in smarts_df["smarts"].values:
        metadata.update(
            {
                "smarts": "[*:1][*:2]!@;-[*:3][*:4]",
                "type": "fallback",
                "expected_angles": np.array([30, 90, 150, 210, 270, 330]),
                "multiplicity": 6,
            }
        )
    else:
        # retrieve the row from the smarts dataframe
        row = smarts_df[smarts_df["smarts"] == log["smarts"]].iloc[0]

        # update the metadata with the row values
        metadata.update(
            {
                "smarts": row["smarts"],
                "type": row["type"],
                "expected_angles": row["expected_angles"],
                "multiplicity": row["multiplicity"],
            }
        )

    return metadata


def make_confp_metadata(mol: Chem.Mol, smarts_df: pd.DataFrame) -> list[dict]:
    """
    Generate conformational fingerprint metadata for a molecule.

    This function creates metadata for conformational fingerprints by extracting information from ETKDG
    (Experimental-Torsion Knowledge Distance Geometry) and organizing it based on canonical indices.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule object for which metadata is generated.
    smarts_df : pd.DataFrame
        DataFrame containing SMARTS patterns used for pattern matching within the molecule.
        Each row contains information like:
        - 'smarts': SMARTS pattern (e.g., '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]')
        - 'expected_angles': Array of preferred dihedral angles (e.g., [0, 120, 240])
        - 'type': Type of the pattern (e.g., 'smallring', 'general', 'macrocycle')
        - 'multiplicity': Number of preferred rotational positions (e.g., 3)

    Returns
    -------
    list[dict]
        A list of dictionaries containing metadata for conformational fingerprints.
        Each dictionary contains information about:
        - 'midbond_atomidces': Set of atom indices defining the rotatable bond
        - 'current_idces': List of atom indices defining dihedral angles
        - 'smarts': SMARTS pattern describing the bond
        - 'type': Type of the rotatable bond (e.g., 'smallring', 'general', 'fallback')
        - 'expected_angles': Array of preferred dihedral angles
        - 'multiplicity': Number of preferred rotational positions

    Examples
    --------
    >>> from rdkit import Chem
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create a cyclohexane molecule
    >>> mol = Chem.MolFromSmiles('C1CCCCC1')
    >>> # Example dataframe with a single SMARTS pattern for small rings
    >>> smarts_df = pd.DataFrame({
    ...     'smarts': ['[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]'],
    ...     'expected_angles': [np.array([0, 120, 240], dtype=np.uint16)],
    ...     'type': ['smallring'],
    ...     'multiplicity': [np.uint8(3)]
    ... })
    >>> metadata = make_confp_metadata(mol, smarts_df)
    >>> print(len(metadata))
    1
    >>> print(len(metadata[0]['midbond_atomidces']))  # Six bonds in cyclohexane
    6
    >>> print(len(metadata[0]['current_idces']))  # Six dihedral angles
    6
    >>> print(metadata[0]['type'])
    smallring
    """

    # declarre the metadata dictionary
    metadata = defaultdict(dict)
    middle_atom_idces = set()

    # get infos from ETKDG
    infos = get_etkdg_info(mol)

    # get the canonical mapping
    idces2canidces = get_idces2canidces(mol)

    # iterate over the infos
    for info in infos:
        angle_identifiers = get_angle_identifiers(info, idces2canidces)
        # record middle atom indices
        middle_atom_idces.add(angle_identifiers["midbond_atomidces"])

        if angle_identifiers["canidces"] in metadata:
            # retrieve the existing metadata
            existing_metadata = metadata[angle_identifiers["canidces"]]

            # check if the midbond atom indices are already present
            if angle_identifiers["midbond_atomidces"] in existing_metadata:
                continue  # skip if already present

            existing_metadata["midbond_atomidces"].add(angle_identifiers["midbond_atomidces"])
            existing_metadata["current_idces"].append(angle_identifiers["current_idces"])
        else:
            # create new metadata entry
            metadata[angle_identifiers["canidces"]].update(
                package_metadata(log=info, smarts_df=smarts_df, angle_identifier=angle_identifiers)
            )

    # find all rotatable bonds in the molecule
    rotatable_bonds = findrotatable_bonds(mol)
    if rotatable_bonds:
        rotatable_set = {frozenset(rotatable_bond_pair) for rotatable_bond_pair in rotatable_bonds}
        rotatable_diff = rotatable_set.difference(middle_atom_idces)
    else:
        rotatable_diff = None

    # add missing rotatable bonds if any
    if rotatable_diff:
        # get all the dihedral atomic indices for the rotatable bonds
        rotatable_bond_idces = enumerate_dihedrals(mol, rotatable_diff)
        keys_record = set(metadata.keys())  # use set to create an independent record of keys

        for idces in rotatable_bond_idces:
            canidces = dihedralidces2sortcandices(idces, idces2canidces)

            if canidces in keys_record:
                # skip if already present
                continue

            # create new metadata entry for the rotatable bond
            if canidces not in metadata:
                metadata[canidces] = {
                    "midbond_atomidces": {frozenset(idces[1:3])},
                    "current_idces": [idces],
                    "smarts": "[*:1][*:2]!@;-[*:3][*:4]",
                    "type": "fallback",
                    "expected_angles": np.array([30, 90, 150, 210, 270, 330]),
                    "multiplicity": 6,
                }
            else:
                # update existing metadata
                existing_metadata = metadata[canidces]
                existing_metadata["midbond_atomidces"].add(frozenset(idces[1:3]))
                existing_metadata["current_idces"].append(idces)
                metadata[canidces] = existing_metadata

    return [metadata[canidces] for canidces in sorted(metadata.keys())]


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


def dbscan_angle_clustering(angles: np.ndarray, eps: float = 0.2, min_samples: int = 10) -> np.ndarray:
    """
    Cluster angles using DBSCAN algorithm after transforming to cosine-sine space.

    This function converts angles (in degrees) to their cosine-sine representation
    to properly handle the circular nature of angles, then applies DBSCAN clustering.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in degrees to be clustered.
    eps : float, optional
        The maximum distance between samples for DBSCAN. Default is 0.2.
    min_samples : int, optional
        Minimum number of samples in a cluster. Default is 10.

    Returns
    -------
    list[int]
        Cluster labels for each input angle (-1 represents noise points)

    Notes
    -----
        This approach handles the periodicity of angles (e.g., 359° and 1° are close)
        by operating in the 2D circular space defined by (cos, sin) coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> # Example with distinct angle groups
    >>> angles = np.array([10, 12, 15, 175, 178, 180, 300, 302, 305])
    >>> labels = dbscan_angle_clustering(angles, eps=0.1, min_samples=2)
    >>> print(labels)  # Should identify 3 clusters
    [0 0 0 1 1 1 2 2 2]

    >>> # Example with angles wrapping around 0/360 degrees
    >>> angles = np.array([355, 358, 2, 5, 180, 183])
    >>> labels = dbscan_angle_clustering(angles, eps=0.1, min_samples=2)
    >>> print(labels)  # Should identify the cluster around 0 degrees
    [0 0 0 0 1 1]

    >>> # Example with noise points
    >>> angles = np.array([10, 12, 90, 175, 178, 250])
    >>> labels = dbscan_angle_clustering(angles, eps=0.1, min_samples=2)
    >>> print(labels)  # Should identify 2 clusters and 2 noise points
    [ 0  0 -1  1  1 -1]
    """

    # Convert angles to cosine-sine representation
    angles_transformed = degree2cosin(angles)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(angles_transformed)

    # Extract cluster labels
    data_labels = clustering.labels_

    return data_labels


def find_cluster_range(angles: np.ndarray, mask: list[bool]) -> tuple[float, float]:
    """
    Find the range of angles for a specific cluster.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in degrees.
    cluster_labels : list[int]
        Cluster labels for each angle.
    cluster_id : int
        The ID of the cluster to find the range for.

    Returns
    -------
    tuple[float, float]
        The minimum and maximum angles in the specified cluster.
    Examples
    --------
    >>> import numpy as np
    >>> angles = np.array([10, 15, 20, 180, 190, 200])
    >>> mask = [True, True, True, False, False, False]
    >>> min_angle, max_angle = find_cluster_range(angles, mask)
    >>> print(min_angle, max_angle)
    10.0 20.0

    >>> angles = np.array([350, 5, 15, 180, 190])
    >>> mask = [True, True, True, False, False]
    >>> min_angle, max_angle = find_cluster_range(angles, mask)
    >>> print(min_angle, max_angle)  # Should find the two angles with maximum separation
    350.0 15.0
    """
    # Get angles belonging to the specified cluster
    cluster_angles = angles[mask]

    # transform angles in cosine-sine space
    cluster_angles = degree2cosin(cluster_angles)

    # compute pairwise cosine distances
    distances = pdist(cluster_angles, metric="cosine")

    # find pair with maximum distance
    max_id = np.argmax(distances)

    # Convert condensed matrix index to pair of indices
    n = len(cluster_angles)
    # Formula to convert from condensed matrix index to square matrix indices
    i = 0
    while max_id >= n - i - 1:
        max_id -= n - i - 1
        i += 1
    j = i + 1 + max_id

    # Get the original angles corresponding to the maximum distance
    angle1 = float(angles[np.where(mask)[0][i]])
    angle2 = float(angles[np.where(mask)[0][j]])

    return (angle1, angle2)  # return the angles


def annotate_degree_cluster(
    angles: np.ndarray, cluster_labels: np.ndarray, expected_angles: np.ndarray, current_idces: tuple[int]
) -> tuple[dict[int, dict], list[int], np.ndarray]:
    """
    Annotate clusters of angles with their ranges, means, and standard deviations.
    Check if the angles match any of the possible angles.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in degrees.
    cluster_labels : np.ndarray
        Cluster labels for each angle, including labels for expected angles.
        The last len(expected_angles) elements correspond to expected angles.
    expected_angles : np.ndarray
        Array of possible angles to check against the cluster ranges.
    current_idces : tuple[int]
        Tuple of atom indices defining the dihedral angle.

    Returns
    -------
    tuple[dict[int, dict], list[int], np.ndarray]
        A dictionary where keys are cluster labels and values are dictionaries containing:
        - "range": The range of angles in the cluster.
        - "circular_mean": Circular Mean angle of the cluster. [1]
        - "circular_std":  Circular Standard deviation of angles in the cluster.[1,2]
        - "match_found": Boolean indicating if a match was found with possible angles.
        - "matched_angle": The matched angle if found, otherwise None.
        - "dihedral_idces": The atom indices defining the dihedral angle.

        A reordered list of cluster labels based on ascending mean angles.

        The original angles array without the expected angles.

    Examples
    --------
    >>> import numpy as np
    >>> # Combine data angles with expected angles
    >>> angles = np.array([10, 15, 20, 180, 185, 190, 12, 183])  # Last two are expected angles
    >>> cluster_labels = np.array([0, 0, 0, 1, 1, 1, 0, 1])  # Labels for all angles including expected
    >>> expected_angles = np.array([12, 183])
    >>> current_idces = (1, 2, 3, 4)  # Example atom indices
    >>> annotations, new_labels, data_angles = annotate_degree_cluster(angles, cluster_labels, expected_angles, current_idces)
    >>> print(annotations[0]['match_found'])  # First cluster matches 12 degrees
    True
    >>> print(annotations[0]['matched_angle'])
    [12]
    >>> print(annotations[0]['dihedral_idces'])
    (1, 2, 3, 4)
    >>> print(new_labels)  # Labels for data points only, without expected angles
    [0, 0, 0, 1, 1, 1]
    >>> print(len(data_angles))  # Should be 6 (original angles minus the expected angles)
    6

    >>> # Example with angle wrapping around 0/360
    >>> angles = np.array([350, 355, 5, 10, 180, 185, 0, 182])  # Last two are expected angles
    >>> cluster_labels = np.array([0, 0, 0, 0, 1, 1, 0, 1])  # Labels including expected angles
    >>> expected_angles = np.array([0, 182])
    >>> current_idces = (2, 3, 4, 5)
    >>> annotations, new_labels, data_angles = annotate_degree_cluster(angles, cluster_labels, expected_angles, current_idces)
    >>> print(annotations[0]['match_found'])  # Cluster around 0 degrees
    True
    >>> print(annotations[0]['matched_angle'])
    [0]
    >>> print(annotations[0]['dihedral_idces'])
    (2, 3, 4, 5)
    >>> print(new_labels)  # Labels for data points only
    [0, 0, 0, 0, 1, 1]
    >>> print(data_angles)  # Should contain only the first 6 angles
    [350 355   5  10 180 185]

    >>> # Example with no matches for one cluster
    >>> angles = np.array([200, 210, 220, 40, 50, 60, 100, 200])  # Last two are expected angles
    >>> cluster_labels = np.array([0, 0, 0, 1, 1, 1, -1, 0])  # Only one cluster matches an expected angle
    >>> expected_angles = np.array([100, 200])
    >>> current_idces = (3, 4, 5, 6)
    >>> annotations, new_labels, data_angles = annotate_degree_cluster(angles, cluster_labels, expected_angles, current_idces)
    >>> print(annotations[0]['match_found'])  # First cluster has a match with 200
    False
    >>> print(annotations[0]['matched_angle'])
    None
    >>> print(annotations[1]['match_found'])  # Second cluster has no match
    True
    >>> print(annotations[1]['matched_angle'])
    [200]
    >>> print(new_labels)  # Labels are reordered based on mean angles
    [1, 1, 1, 0, 0, 0]
    >>> print(len(data_angles))  # Should be 6 (original angles minus the expected angles)
    6

    >>> # Example with noise points
    >>> angles = np.array([10, 15, 20, 80, 185, 190, 12, 183])  # Last two are expected angles
    >>> cluster_labels = np.array([0, 0, 0, -1, 1, 1, 0, 1])  # -1 indicates noise
    >>> expected_angles = np.array([12, 183])
    >>> current_idces = (5, 6, 7, 8)
    >>> annotations, new_labels, data_angles = annotate_degree_cluster(angles, cluster_labels, expected_angles, current_idces)
    >>> print(annotations[0]['match_found'])  # First cluster matches 12 degrees
    True
    >>> print(annotations[0]['matched_angle'])
    [12]
    >>> print(annotations[1]['dihedral_idces'])  # Check dihedral indices in second cluster
    (5, 6, 7, 8)
    >>> print(new_labels)  # Noise point (-1) is preserved
    [0, 0, 0, -1, 1, 1]
    >>> print(data_angles[3])  # Check the noise point value
    80

    References
    ----------
    - [1] Mardia, K. V. and Jupp, P. E. Directional Statistics. John Wiley & Sons, 1999
    - [2] Mardia, K. V. (1972). 2. In Statistics of Directional Data (pp. 18-24). Academic Press. DOI:10.1016/C2013-0-07425-7
    """
    # Initialize the annotation dictionary
    annotation = defaultdict(dict)
    cluster2expected = defaultdict(list)  # in most cases this will be a single value, but can be multiple

    # convert labels and expected angles to list
    labels_list = cluster_labels.tolist()
    expected_angles_list = expected_angles.tolist()

    # retrieve expected labels stored at the end of the angles
    expected_labels = labels_list[-len(expected_angles_list) :]
    data_labels = labels_list[: -len(expected_angles_list)]

    # remove expected angles from the angles array
    angles = angles[: -len(expected_angles_list)]

    # populate the cluster2expected mapping
    for cl, ea in zip(expected_labels, expected_angles_list):
        cluster2expected[cl].append(ea)

    # Loop through unique cluster labels
    for label in set(labels_list):
        # Skip noise points (label -1)
        if label == -1:
            continue

        # loop variables
        temp_annotation = {
            "range": None,
            "circular_mean": None,
            "circular_std": None,
            "match_found": True if label in cluster2expected else False,
            "matched_angle": cluster2expected[label] if label in cluster2expected else None,
            "dihedral_idces": current_idces,
        }

        # Get mask for current cluster
        mask = [cl == label for cl in data_labels]

        # Find the range of angles for this cluster
        temp_annotation["range"] = find_cluster_range(angles, mask)
        temp_annotation["circular_mean"] = float(np.degrees(circmean(np.radians(angles[mask])))) % 360
        temp_annotation["circular_std"] = float(np.degrees(circstd(np.radians(angles[mask]))))

        # Store the range and angles in the annotation dictionary
        annotation[label].update(temp_annotation)

    # Reorder labels by ascending circular_mean

    # Sort remaining labels by circular_mean
    sorted_labels = sorted(annotation.keys(), key=lambda x: annotation[x]["circular_mean"])
    sorted_labels = {label: i for i, label in enumerate(sorted_labels)}
    sorted_labels[-1] = -1  # Ensure noise label (-1) is included for new labels

    # reorder the annotation dictionary
    annotation = {sorted_labels[label]: annotation[label] for label in sorted_labels if label != -1}
    # reorder the cluster labels
    new_labels = [sorted_labels[label] for label in data_labels]

    return annotation, new_labels, angles


def congruence_check(annotation_list: list[dict]) -> bool:
    """
    Check if all clusters in the annotation list have a match with the possible angles.

    Parameters
    ----------
    annotation_list : list[dict]
        List of dictionaries containing cluster annotations.

    Returns
    -------
    bool
        True if all clusters have a match with the possible angles, False otherwise.

    Examples
    --------
    >>> # Example where all clusters match
    >>> annotations = [
    ...     {0: {"match_found": True, "matched_angle": 60, "circular_mean": 60.5}},
    ...     {0: {"match_found": True, "matched_angle": 60, "circular_mean": 59.8}}
    ... ]
    >>> congruence_check(annotations)
    True

    >>> # Example where one cluster doesn't match but circular_mean are close
    >>> annotations = [
    ...     {0: {"match_found": True, "matched_angle": 60, "circular_mean": 61.2}},
    ...     {0: {"match_found": False, "matched_angle": None, "circular_mean": 59.5}}
    ... ]
    >>> congruence_check(annotations)
    True

    >>> # Example where clusters have different circular_mean but too far apart
    >>> annotations = [
    ...     {0: {"match_found": False, "matched_angle": None, "circular_mean": 60.0}},
    ...     {0: {"match_found": False, "matched_angle": None, "circular_mean": 75.0}}
    ... ]
    >>> congruence_check(annotations)
    False

    >>> # Example with multiple clusters all matching
    >>> annotations = [
    ...     {0: {"match_found": True, "matched_angle": 60, "circular_mean": 59.8},
    ...      1: {"match_found": True, "matched_angle": 180, "circular_mean": 178.9}},
    ...     {0: {"match_found": True, "matched_angle": 60, "circular_mean": 60.2},
    ...      1: {"match_found": True, "matched_angle": 180, "circular_mean": 181.1}}
    ... ]
    >>> congruence_check(annotations)
    True

    >>> # Example with mixed matching and non-matching but close circular_mean
    >>> annotations = [
    ...     {0: {"match_found": True, "matched_angle": 60, "circular_mean": 60.1},
    ...      1: {"match_found": False, "matched_angle": None, "circular_mean": 179.3}},
    ...     {0: {"match_found": False, "matched_angle": None, "circular_mean": 59.7},
    ...      1: {"match_found": True, "matched_angle": 180, "circular_mean": 180.8}}
    ... ]
    >>> congruence_check(annotations)
    True

    """
    congruence_status = False
    # check if all aggregated angles have the same number of clusters
    size_counter = Counter(map(len, annotation_list))

    if len(size_counter) > 1:
        return congruence_status

    # check if all aggregated angles have the same number of possible angles
    for i in range(len(annotation_list[0])):
        if all(annotation[i]["match_found"] for annotation in annotation_list):
            # check that they all match the same angle
            if len({annotation[i]["matched_angle"] for annotation in annotation_list}) == 1:
                continue
            else:
                # if any of the clusters does not match, return False
                return congruence_status
        elif (
            smallest_distance_degree(
                *find_cluster_range(
                    angles=np.array([annotation[i]["circular_mean"] for annotation in annotation_list]),
                    mask=[True] * len(annotation_list),  # All True since we are checking all clusters
                )
            )
            <= 10
        ):
            # if the angles are close enough, we consider them congruent
            continue
        else:
            # if any of the clusters does not match, return False
            return congruence_status

    # if we reach here, all clusters match the same angle
    # and are congruent
    congruence_status = True

    return congruence_status


def make_confp_fragment(labels: list[int] | list[list[int]]) -> np.ndarray:
    """
    Create a binary fingerprint from cluster labels.

    This function takes a list of cluster labels and creates a binary fingerprint
    where each unique label corresponds to a bit in the fingerprint. The resulting
    fingerprint is a 1D numpy array with bits set to 1 for each unique label present.

    Parameters
    ----------
    labels : list[int] | list[list[int]]
        List of cluster labels or lists of cluster labels.

    Returns
    -------
    np.ndarray
        Binary fingerprint array where each bit corresponds to a unique label.

    Notes
    -----
    - If the input is a list of lists, it will be casted into a 2D array
      where each row corresponds to a set of labels.
    - The function handles both aggregated labels (lists of lists) and single labels.

    Examples
    --------
    >>> # Example with a simple list of labels
    >>> labels = [0, 1, 0, 2]
    >>> fp = make_confp_fragment(labels)
    >>> print(fp)
    [[ True False False]
     [False  True False]
     [ True False False]
     [False False  True]]

    >>> # Example with aggregated labels (list of lists)
    >>> labels = [[0, 1], [0,1], [0,1]]
    >>> fp = make_confp_fragment(labels)
    >>> print(fp)
    [[ True  True  True False False False]
     [False False False  True  True  True]]
    >>> # Example with a list containing -1 (noise points)
    >>> labels = [0, -1, 1, 0]
    >>> fp = make_confp_fragment(labels)
    >>> print(fp)
    [[ True False]
     [False False]
     [False  True]
     [ True False]]

    >>> # Example with multiple aggregated lists containing noise
    >>> labels = [[0, 1, -1], [2, -1, 0], [1, 0, 2]]
    >>> fp = make_confp_fragment(labels)
    >>> print(fp)
    [[ True False False  True False False  True False False]
     [ True False False  True False False False False False]
     [ True False False False False False  True False False]]
    """
    #
    aggregate = isinstance(labels[0], list)
    # flatten the list if it contains lists of labels
    if aggregate:
        label_array = np.column_stack(labels)
        values, counts = np.unique(label_array, return_counts=True, axis=0)
    else:
        label_array = np.array(labels)

    unique_labels = np.unique(labels)
    nb_pos_labels = sum(unique_labels >= 0)  # does not count noise (-1)

    # initialize the fingerprint with zeros
    if aggregate:
        fp_fragment = np.zeros((label_array.shape[0], nb_pos_labels * label_array.shape[1]), dtype=bool)
    else:
        fp_fragment = np.zeros((label_array.shape[0], nb_pos_labels), dtype=bool)

    # make the fingerprint
    if aggregate:
        counter_array = np.apply_along_axis(Counter, 1, label_array)

        for i, counter in enumerate(counter_array):
            for label in sorted(counter.keys()):
                if label >= 0:
                    lower_bound = label * label_array.shape[1]
                    upper_bound = label * label_array.shape[1] + counter[label]
                    fp_fragment[i, lower_bound:upper_bound] = True
    else:
        for i, label in enumerate(label_array):
            if label >= 0:
                fp_fragment[i, label] = True

    return fp_fragment


def repackage_cluster_annotations(annotations: dict[int, dict] | list[dict[int, dict]], start_pos: int = 0) -> list[dict[str, object]]:
    """
    Repackage cluster annotations into a list of dictionaries with positional information.

    This function converts cluster annotations into a standardized list format, adding
    positional information to each cluster entry. It handles both single annotation dictionaries
    and aggregated lists of annotation dictionaries.

    Parameters
    ----------
    annotations : dict[int, dict] | list[dict[int, dict]]
        Either a single dictionary of cluster annotations (where keys are cluster labels)
        or a list of such dictionaries for aggregated annotations.
    start_pos : int, optional
        Starting position for indexing the clusters in the confp fingerprint, by default 0.

    Returns
    -------
    list[dict[str, object]]
        List of dictionaries, each containing:
        - Original cluster properties
        - "label": The original cluster label
        - "aggregated": Boolean indicating if this came from an aggregated annotation
        - "start_pos": Starting position in the fingerprint
        - "end_pos": Ending position in the fingerprint

    Notes
    -----
    - For non-aggregated annotations, each cluster gets sequential positions
    - For aggregated annotations, positions are calculated based on cluster and annotation count

    Examples
    --------
    >>> # Example with single annotation dictionary
    >>> annotations = {0: {"range": (60, 70), "circular_mean": 65.0, "match_found": True, "matched_angle": 60},
    ...                1: {"range": (180, 190), "circular_mean": 185.0, "match_found": True, "matched_angle": 180}}
    >>> result = repackage_cluster_annotations(annotations)
    >>> print(len(result))
    2
    >>> print(result[0]["label"], result[0]["start_pos"], result[0]["end_pos"])
    0 0 1
    >>> print(result[1]["label"], result[1]["start_pos"], result[1]["end_pos"])
    1 1 2
    >>> print(result[0]["aggregated"])
    False

    >>> # Example with aggregated annotations (list of annotation dictionaries)
    >>> aggregated_annotations = [
    ...     {0: {"range": (60, 70), "circular_mean": 65.0, "match_found": True, "matched_angle": 60}},
    ...     {0: {"range": (58, 68), "circular_mean": 63.0, "match_found": True, "matched_angle": 60}}
    ... ]
    >>> result = repackage_cluster_annotations(aggregated_annotations, start_pos=5)
    >>> print(len(result))
    2
    >>> print(result[0]["label"], result[0]["start_pos"], result[0]["end_pos"])
    0 5 6
    >>> print(result[1]["label"], result[1]["start_pos"], result[1]["end_pos"])
    0 5 6
    >>> print(result[0]["aggregated"])
    True

    >>> # Example with custom start position
    >>> annotations = {0: {"range": (10, 20), "circular_mean": 15.0, "match_found": False, "matched_angle": None}}
    >>> result = repackage_cluster_annotations(annotations, start_pos=10)
    >>> print(result[0]["start_pos"], result[0]["end_pos"])
    10 11
    """
    # declare local variables
    annotation_list = []
    isaggregated = isinstance(annotations, list)

    if not isaggregated:
        # if the annotations are not aggregated, we can just return them as a list
        for i, (label, annotation) in enumerate(annotations.items()):
            annotation_list.append(
                {
                    "label": label,
                    "aggregated": False,
                    "start_pos": start_pos + i,
                    "end_pos": start_pos + i + 1,
                    **annotation,
                }
            )
    else:
        # if the annotations are aggregated, we need to repackage them
        for annotation in annotations:
            for i, (label, properties) in enumerate(annotation.items()):
                annotation_list.append(
                    {
                        "label": label,
                        "aggregated": True,
                        "start_pos": start_pos + i * len(annotation),
                        "end_pos": start_pos + (i + 1) * len(annotation),
                        **properties,
                    }
                )
    return annotation_list


def get_conformer_confp(mol: Chem.Mol, metadata: list[dict], epsilon: float = 0.2, min_samples: int = 10) -> tuple[np.ndarray, list[dict]]:
    """
    Generate a conformational fingerprint from dihedral angles across all conformers.

    This function analyzes the dihedral angle patterns across all conformers of a molecule
    and creates a binary fingerprint that encodes these patterns. It uses DBSCAN clustering
    to identify preferred dihedral angle states and handles molecular symmetry by analyzing
    multiple equivalent dihedrals.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object containing multiple conformers (at least 2).
    metadata : list[dict]
        List of dictionaries containing angle metadata with keys:
        - "expected_angles": Array of reference angles in degrees
        - "current_idces": List of atom index tuples (i,j,k,l) defining dihedral angles
        - "multiplicity": Number of preferred rotational positions
        - "type": Type of the rotatable bond (e.g., 'smallring', 'general', 'fallback')
    epsilon : float, default=0.2
        The maximum distance between samples for DBSCAN clustering in cosine-sine space.
    min_samples : int, default=10
        Minimum number of samples in a cluster for DBSCAN.

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        A tuple containing:
        - np.ndarray: Binary fingerprint array (bool dtype) encoding conformational patterns
        - list[dict]: Annotation details for each cluster including:
          * range: Tuple of min/max angles in the cluster
          * circular_mean: Circular mean of angles in the cluster
          * circular_std: Circular standard deviation of angles
          * match_found: Whether cluster matches an expected reference angle
          * matched_angle: The matched reference angle (if any)
          * dihedral_idces: Atom indices defining the dihedral
          * label: Cluster label
          * aggregated: Whether this cluster comes from aggregated analysis
          * start_pos: Starting bit position in fingerprint
          * end_pos: Ending bit position in fingerprint

    Notes
    -----
    - For symmetric parts of molecules, the function performs congruence checks
      to determine if multiple equivalent dihedrals can be represented together
    - Each rotatable bond's dihedral angle distribution is analyzed separately
      and encoded in the resulting fingerprint
    - Noise points from DBSCAN clustering are represented with -1 labels
    - The final fingerprint is created by column-stacking individual fragment fingerprints
    """
    # check if the molecule has at least 2 conformers
    check_conformers(mol)

    # initialize the fingerprint list
    fingerprint = []
    start_pos = 0
    metadata_list = []
    for angle_metadata in metadata:
        # check if there is more then one angle with same signature
        if len(angle_metadata["current_idces"]) == 1:
            dihedrals = compute_dihedrals_conformers(mol, angle_metadata["current_idces"][0])
            # add the expected angles to the dihedrals
            dihedrals = np.concatenate((dihedrals, angle_metadata["expected_angles"]), axis=0)
            # cluster the dihedrals
            labels = dbscan_angle_clustering(dihedrals, eps=epsilon, min_samples=min_samples)
            # annotate the clusters
            annotation, new_labels, _ = annotate_degree_cluster(
                dihedrals, labels, angle_metadata["expected_angles"], angle_metadata["current_idces"][0]
            )
            # make the fingerprint
            fingerprint.append(make_confp_fragment(new_labels))
            # aggregate the annotations
            annotation = repackage_cluster_annotations(annotation, start_pos=start_pos)
            metadata_list.extend(annotation)
            # update the start position
            start_pos += fingerprint[-1].shape[1] + 1
        else:
            # if there are multiple angles with the same signature, we need to compute them all
            annotation_list = []
            labels_list = []
            dihedrals = []
            for current_idces in angle_metadata["current_idces"]:
                dihedrals = compute_dihedrals_conformers(mol, current_idces)
                # add the expected angles to the dihedrals
                dihedrals = np.concatenate((dihedrals, angle_metadata["expected_angles"]), axis=0)
                # cluster the dihedrals
                labels = dbscan_angle_clustering(dihedrals, eps=epsilon, min_samples=min_samples)
                # annotate the clusters
                annotation, new_labels, _ = annotate_degree_cluster(dihedrals, labels, angle_metadata["expected_angles"], current_idces)
                # aggregate the annotations and labels
                annotation_list.append(annotation)
                labels_list.append(new_labels)
            # check if the clusters are congruent
            congruence = congruence_check(annotation_list)

            if congruence:
                # if the clusters are congruent, we can aggregate the labels
                fingerprint.append(make_confp_fragment(labels_list))
                # aggregate the annotations
                annotation = repackage_cluster_annotations(annotation_list, start_pos=start_pos)
                metadata_list.extend(annotation)
                start_pos += fingerprint[-1].shape[1] + 1
            else:
                # if the clusters are not congruent, we need to create a fingerprint for each angle
                for j, labels in enumerate(labels_list):
                    fingerprint.append(make_confp_fragment(labels))
                    # aggregate the annotations
                    annotation = repackage_cluster_annotations(annotation_list[j], start_pos=start_pos)
                    metadata_list.extend(annotation)
                    start_pos += fingerprint[-1].shape[1] + 1

    # concatenate the fingerprints
    fingerprint = np.column_stack(fingerprint)

    return fingerprint, metadata_list
