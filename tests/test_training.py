"""Tests for training utilities."""

import pytest
import torch
import tempfile
import os

from hierarchical_attention_pooling_for_molecular_scaffold_transfer.models.model import HierarchicalAttentionPoolingModel
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.training.trainer import Trainer, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset


def test_early_stopping():
    """Test early stopping mechanism."""
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    # Improving losses
    assert early_stopping(1.0) == False
    assert early_stopping(0.9) == False
    assert early_stopping(0.8) == False

    # No improvement
    assert early_stopping(0.81) == False
    assert early_stopping(0.82) == False
    assert early_stopping(0.83) == False
    assert early_stopping(0.84) == True  # Should trigger


def test_trainer_initialization(sample_config, device):
    """Test trainer initialization."""
    # Create dummy data loader
    x = torch.randn(10, 38)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)

    # Create model
    model = HierarchicalAttentionPoolingModel(sample_config)

    # Update config for short training
    sample_config['training']['num_epochs'] = 2
    sample_config['training']['use_amp'] = False
    sample_config['training']['early_stopping'] = False

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=sample_config,
        device=device
    )

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.criterion is not None


def test_trainer_checkpoint_saving(sample_config, device):
    """Test checkpoint saving."""
    # Create dummy data loader
    x = torch.randn(10, 38)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)

    # Create model
    model = HierarchicalAttentionPoolingModel(sample_config)

    # Use temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_config['training']['save_dir'] = tmpdir
        sample_config['training']['num_epochs'] = 1
        sample_config['training']['use_amp'] = False

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=sample_config,
            device=device
        )

        # Save checkpoint
        trainer.save_checkpoint(epoch=0, is_best=True)

        # Check if checkpoint exists
        checkpoint_path = os.path.join(tmpdir, 'best_model.pth')
        assert os.path.exists(checkpoint_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
