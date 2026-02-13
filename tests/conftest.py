"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data as PyGData


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'dataset_name': 'BBBP',
            'split_type': 'scaffold',
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'batch_size': 2,
            'num_workers': 0,
            'seed': 42
        },
        'model': {
            'node_feat_dim': 38,
            'edge_feat_dim': 7,
            'hidden_dim': 64,
            'num_gnn_layers': 2,
            'gnn_type': 'gcn',
            'dropout': 0.2,
            'use_hierarchical_pooling': True,
            'num_attention_heads': 4,
            'functional_group_pooling': True,
            'scaffold_aware': True,
            'num_classes': 2,
            'task_type': 'classification'
        },
        'training': {
            'num_epochs': 5,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'gradient_clip': 1.0,
            'use_amp': False,
            'early_stopping': False,
            'save_dir': 'checkpoints',
            'log_interval': 1
        },
        'evaluation': {
            'metrics': ['roc_auc', 'accuracy', 'f1'],
            'scaffold_split_eval': True,
            'save_predictions': True,
            'results_dir': 'results'
        }
    }


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        'CCO',  # Ethanol
        'CC(=O)O',  # Acetic acid
        'c1ccccc1',  # Benzene
        'CCN(CC)CC',  # Triethylamine
        'C1CCCCC1',  # Cyclohexane
    ]


@pytest.fixture
def sample_graph_data():
    """Sample graph data for testing."""
    # Simple graph with 5 nodes
    x = torch.randn(5, 38)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    edge_attr = torch.randn(8, 7)
    y = torch.tensor([1], dtype=torch.long)

    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.batch = torch.zeros(5, dtype=torch.long)

    return data


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device('cpu')


@pytest.fixture
def set_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
