"""Module for angles metadata extraction using RDKit.

Key functionalities include:
- Finding rotatable bonds and dihedral angles in molecules
- Getting ETKDG (Experimental Torsion Knowledge Distance Geometry) parameters
- Converting atom indices to canonical form for symmetry handling
- Packaging  angles metadata for conformational searches
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.Lipinski import RotatableBondSmarts


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
        hybridization_set = {index2hybridization(mol, idx2), index2hybridization(mol, idx3)}

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
        - 'angles_bins': List of possible dihedral angles (in degrees) for the rotation
        - 'multiplicity': Number of preferred rotational positions

    Notes
    -----
    If the provided SMARTS pattern is not found in the dataframe, a fallback generic
    rotatable bond pattern will be used with a starting point of 30° followed by 60° angle increments.

    Examples
    --------
    >>> from rdkit import Chem
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create a simple DataFrame with SMARTS patterns
    >>> smarts_df = pd.DataFrame({
    ...     'smarts': ['[*:1]~[*:2]@;-[*:3]~[*:4]', '[*:1][*:2]!@;-[*:3][*:4]'],
    ...     'angles_bins': [np.array([0, 120, 240]), np.array([0, 60, 120, 180, 240, 300])],
    ...     'type': ['smallring', 'general'],
    ...     'multiplicity': [3, 6]
    ... })
    >>> # Sample log and angle identifier
    >>> log = {'smarts': '[*:1]~[*:2]@;-[*:3]~[*:4]'}
    >>> angle_identifier = {
    ...     'midbond_atomidces': frozenset({1, 2}),
    ...     'current_idces': (0, 1, 2, 3)
    ... }
    >>> metadata = package_metadata(log, smarts_df, angle_identifier)
    >>> print(metadata['type'])
    smallring
    >>> print(metadata['multiplicity'])
    3
    >>> # Example with fallback pattern
    >>> log_fallback = {'smarts': 'non_matching_pattern'}
    >>> metadata_fallback = package_metadata(log_fallback, smarts_df, angle_identifier)
    >>> print(metadata_fallback['type'])
    fallback
    >>> print(metadata_fallback['angles_bins'])
    [ 30  90 150 210 270 330]
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
                "angles_bins": np.array([30, 90, 150, 210, 270, 330]),
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
                "angles_bins": row["angles_bins"],
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
        - 'angles_bins': Array of preferred dihedral angles (e.g., [0, 120, 240])
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
        - 'angles_bins': Array of preferred dihedral angles
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
    ...     'angles_bins': [np.array([0, 120, 240], dtype=np.uint16)],
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
                    "angles_bins": np.array([30, 90, 150, 210, 270, 330]),
                    "multiplicity": 6,
                }
            else:
                # update existing metadata
                existing_metadata = metadata[canidces]
                existing_metadata["midbond_atomidces"].add(frozenset(idces[1:3]))
                existing_metadata["current_idces"].append(idces)
                metadata[canidces] = existing_metadata

    return [metadata[canidces] for canidces in sorted(metadata.keys())]
