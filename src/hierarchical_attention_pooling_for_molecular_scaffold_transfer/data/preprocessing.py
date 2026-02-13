"""Data preprocessing utilities for molecular graphs."""

import logging
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

logger = logging.getLogger(__name__)


def scaffold_split(
    smiles_list: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """Split molecules by scaffold to prevent data leakage.

    Groups molecules by their Murcko scaffolds and assigns entire scaffold
    groups to train/val/test splits. This ensures molecules with similar
    core structures don't appear in both training and test sets.

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)

    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES at index {i}: {smiles}")
                continue
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffold_to_indices[scaffold].append(i)
        except Exception as e:
            logger.warning(f"Error processing SMILES at index {i}: {e}")
            continue

    # Sort scaffolds by size (largest first) for balanced splits
    scaffolds = list(scaffold_to_indices.items())
    scaffolds.sort(key=lambda x: len(x[1]), reverse=True)

    # Assign scaffolds to splits
    train_indices, val_indices, test_indices = [], [], []
    train_size = val_size = test_size = 0
    total_size = len(smiles_list)

    for scaffold, indices in scaffolds:
        # Determine which split needs more data
        train_frac = train_size / total_size if total_size > 0 else 0
        val_frac = val_size / total_size if total_size > 0 else 0
        test_frac = test_size / total_size if total_size > 0 else 0

        if train_frac < train_ratio:
            train_indices.extend(indices)
            train_size += len(indices)
        elif val_frac < val_ratio:
            val_indices.extend(indices)
            val_size += len(indices)
        else:
            test_indices.extend(indices)
            test_size += len(indices)

    logger.info(f"Scaffold split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    return train_indices, val_indices, test_indices


def get_functional_groups(mol: Chem.Mol) -> List[Set[int]]:
    """Identify functional groups in a molecule for hierarchical pooling.

    Uses SMARTS patterns to identify common functional groups and returns
    sets of atom indices belonging to each group. Atoms can belong to
    multiple groups.

    Args:
        mol: RDKit molecule object

    Returns:
        List of sets, each containing atom indices for a functional group
    """
    # Common functional group SMARTS patterns
    functional_groups = {
        'carboxyl': '[CX3](=O)[OX2H1]',
        'carbonyl': '[CX3]=[OX1]',
        'hydroxyl': '[OX2H]',
        'amine': '[NX3;H2,H1;!$(NC=O)]',
        'amide': '[NX3][CX3](=[OX1])',
        'ester': '[CX3](=O)[OX2][CX4]',
        'ether': '[OD2]([#6])[#6]',
        'aromatic': 'a',  # Any aromatic atom
        'halogen': '[F,Cl,Br,I]',
        'nitro': '[N+](=O)[O-]',
        'sulfone': '[SX4](=O)(=O)',
        'phosphate': '[PX4](=O)([O,o])([O,o])[O,o]',
    }

    groups = []
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            groups.append(set(match))

    # Add ring systems as functional groups
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        groups.append(set(ring))

    # If no functional groups found, group by connected components
    if not groups:
        num_atoms = mol.GetNumAtoms()
        groups.append(set(range(num_atoms)))

    return groups


def one_hot_encoding(value: int, choices: List[int]) -> List[int]:
    """Create one-hot encoding for a value.

    Args:
        value: Value to encode
        choices: List of possible values

    Returns:
        One-hot encoded list
    """
    encoding = [0] * (len(choices) + 1)
    try:
        index = choices.index(value)
        encoding[index] = 1
    except ValueError:
        encoding[-1] = 1  # Unknown value
    return encoding


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract atom features for graph neural networks.

    Features include:
    - Atom type (one-hot)
    - Degree (one-hot)
    - Hybridization (one-hot)
    - Aromaticity (binary)
    - Formal charge

    Args:
        atom: RDKit atom object

    Returns:
        Feature vector as numpy array
    """
    features = []

    # Atom type
    atom_types = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # B, C, N, O, F, P, S, Cl, Br, I
    features.extend(one_hot_encoding(atom.GetAtomicNum(), atom_types))

    # Degree
    degrees = [0, 1, 2, 3, 4, 5]
    features.extend(one_hot_encoding(atom.GetDegree(), degrees))

    # Hybridization
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
    ]
    features.extend(one_hot_encoding(atom.GetHybridization(), hybridizations))

    # Aromaticity
    features.append(int(atom.GetIsAromatic()))

    # Formal charge
    features.append(atom.GetFormalCharge())

    return np.array(features, dtype=np.float32)


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Extract bond features for graph neural networks.

    Features include:
    - Bond type (one-hot)
    - Conjugation (binary)
    - Ring membership (binary)

    Args:
        bond: RDKit bond object

    Returns:
        Feature vector as numpy array
    """
    features = []

    # Bond type
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    features.extend(one_hot_encoding(bond.GetBondType(), bond_types))

    # Conjugation
    features.append(int(bond.GetIsConjugated()))

    # Ring membership
    features.append(int(bond.IsInRing()))

    return np.array(features, dtype=np.float32)


def mol_to_graph(
    smiles: str,
    include_functional_groups: bool = True
) -> Optional[Dict[str, torch.Tensor]]:
    """Convert SMILES to graph representation.

    Args:
        smiles: SMILES string
        include_functional_groups: Whether to identify functional groups

    Returns:
        Dictionary containing:
        - node_features: Node feature matrix [num_nodes, node_feat_dim]
        - edge_index: Edge connectivity [2, num_edges]
        - edge_features: Edge feature matrix [num_edges, edge_feat_dim]
        - functional_groups: List of functional group atom indices (optional)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Invalid SMILES: {smiles}")
        return None

    # Add hydrogens for more complete representation
    mol = Chem.AddHs(mol)

    # Node features
    num_atoms = mol.GetNumAtoms()
    node_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])

    # Edge features and connectivity
    edge_list = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Add both directions for undirected graph
        edge_list.extend([[i, j], [j, i]])
        edge_features.extend([bond_feat, bond_feat])

    if not edge_list:
        # Isolated atom - add self-loop
        edge_list = [[0, 0]]
        edge_features = [np.zeros(7, dtype=np.float32)]

    graph_data = {
        'node_features': torch.tensor(node_features, dtype=torch.float),
        'edge_index': torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
        'edge_features': torch.tensor(edge_features, dtype=torch.float),
        'num_nodes': num_atoms,
    }

    if include_functional_groups:
        groups = get_functional_groups(mol)
        graph_data['functional_groups'] = groups

    return graph_data


def featurize_molecule(smiles: str) -> Optional[Dict[str, torch.Tensor]]:
    """Featurize a molecule from SMILES string.

    Convenience wrapper around mol_to_graph.

    Args:
        smiles: SMILES string

    Returns:
        Graph data dictionary or None if invalid
    """
    return mol_to_graph(smiles, include_functional_groups=True)
