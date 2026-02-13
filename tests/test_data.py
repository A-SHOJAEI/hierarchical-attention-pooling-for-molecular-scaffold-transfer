"""Tests for data loading and preprocessing."""

import pytest
import numpy as np
import torch
from rdkit import Chem

from hierarchical_attention_pooling_for_molecular_scaffold_transfer.data.preprocessing import (
    scaffold_split,
    get_functional_groups,
    mol_to_graph,
    featurize_molecule,
    get_atom_features,
    get_bond_features
)


def test_scaffold_split(sample_smiles):
    """Test scaffold-based splitting."""
    train_idx, val_idx, test_idx = scaffold_split(
        sample_smiles,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )

    # Check that all indices are unique
    all_indices = set(train_idx + val_idx + test_idx)
    assert len(all_indices) == len(sample_smiles)

    # Check that splits are non-empty
    assert len(train_idx) > 0
    assert len(val_idx) >= 0  # Could be empty for small datasets
    assert len(test_idx) >= 0


def test_get_functional_groups(sample_smiles):
    """Test functional group identification."""
    mol = Chem.MolFromSmiles(sample_smiles[1])  # Acetic acid
    groups = get_functional_groups(mol)

    assert isinstance(groups, list)
    assert len(groups) > 0

    # Each group should be a set of atom indices
    for group in groups:
        assert isinstance(group, set)
        assert all(isinstance(idx, int) for idx in group)


def test_get_atom_features():
    """Test atom feature extraction."""
    mol = Chem.MolFromSmiles('CCO')
    atom = mol.GetAtomWithIdx(0)
    features = get_atom_features(atom)

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert len(features) > 0


def test_get_bond_features():
    """Test bond feature extraction."""
    mol = Chem.MolFromSmiles('CCO')
    bond = mol.GetBondWithIdx(0)
    features = get_bond_features(bond)

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert len(features) > 0


def test_mol_to_graph(sample_smiles):
    """Test molecule to graph conversion."""
    graph_data = mol_to_graph(sample_smiles[0])

    assert graph_data is not None
    assert 'node_features' in graph_data
    assert 'edge_index' in graph_data
    assert 'edge_features' in graph_data
    assert 'functional_groups' in graph_data

    # Check tensor types
    assert isinstance(graph_data['node_features'], torch.Tensor)
    assert isinstance(graph_data['edge_index'], torch.Tensor)
    assert isinstance(graph_data['edge_features'], torch.Tensor)

    # Check shapes
    num_nodes = graph_data['node_features'].size(0)
    num_edges = graph_data['edge_index'].size(1)
    assert num_nodes > 0
    assert num_edges > 0
    assert graph_data['edge_index'].size(0) == 2
    assert graph_data['edge_features'].size(0) == num_edges


def test_mol_to_graph_invalid_smiles():
    """Test handling of invalid SMILES."""
    graph_data = mol_to_graph('INVALID_SMILES')
    assert graph_data is None


def test_featurize_molecule(sample_smiles):
    """Test molecule featurization."""
    for smiles in sample_smiles:
        graph_data = featurize_molecule(smiles)
        assert graph_data is not None
        assert 'node_features' in graph_data
        assert 'edge_index' in graph_data
