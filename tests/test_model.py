"""Tests for model components."""

import pytest
import torch

from hierarchical_attention_pooling_for_molecular_scaffold_transfer.models.model import (
    HierarchicalAttentionPoolingModel,
    GNNEncoder
)
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.models.components import (
    HierarchicalAttentionPooling,
    FunctionalGroupPooling,
    ScaffoldAwareLoss,
    CurriculumScheduler
)


def test_gnn_encoder(sample_config, sample_graph_data):
    """Test GNN encoder."""
    config = sample_config['model']
    encoder = GNNEncoder(
        node_feat_dim=config['node_feat_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_gnn_layers'],
        gnn_type=config['gnn_type'],
        dropout=config['dropout']
    )

    # Forward pass
    x = sample_graph_data.x
    edge_index = sample_graph_data.edge_index
    output = encoder(x, edge_index)

    assert output.size(0) == x.size(0)
    assert output.size(1) == config['hidden_dim']


def test_functional_group_pooling(sample_config):
    """Test functional group pooling."""
    hidden_dim = sample_config['model']['hidden_dim']
    pooling = FunctionalGroupPooling(hidden_dim=hidden_dim, num_heads=4, dropout=0.1)

    # Create dummy data
    x = torch.randn(10, hidden_dim)  # 10 nodes
    batch = torch.zeros(10, dtype=torch.long)  # Single graph

    # Functional groups
    functional_groups = [[{0, 1, 2}, {3, 4}, {5, 6, 7, 8, 9}]]

    # Forward pass
    output = pooling(x, batch, functional_groups)

    assert output.size(0) == 1  # Single graph
    assert output.size(1) == hidden_dim


def test_hierarchical_attention_pooling(sample_config):
    """Test hierarchical attention pooling."""
    hidden_dim = sample_config['model']['hidden_dim']
    pooling = HierarchicalAttentionPooling(hidden_dim=hidden_dim, num_heads=4, dropout=0.1)

    # Create dummy data
    x = torch.randn(10, hidden_dim)
    batch = torch.zeros(10, dtype=torch.long)

    # Forward pass
    output = pooling(x, batch, None)

    assert output.size(0) == 1
    assert output.size(1) == hidden_dim


def test_scaffold_aware_loss(sample_config):
    """Test scaffold-aware loss function."""
    num_classes = sample_config['model']['num_classes']
    loss_fn = ScaffoldAwareLoss(num_classes=num_classes, label_smoothing=0.1)

    # Create dummy data
    logits = torch.randn(4, num_classes)
    labels = torch.randint(0, num_classes, (4,))

    # Compute loss
    loss = loss_fn(logits, labels)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0


def test_curriculum_scheduler():
    """Test curriculum learning scheduler."""
    scheduler = CurriculumScheduler(start_epoch=0, end_epoch=100, num_complexity_levels=5)

    # Test threshold progression
    # At start: 1/5 = 0.2
    # At middle: 0.2 + 0.5 * (1.0 - 0.2) = 0.6
    # At end: 1.0
    assert scheduler.get_complexity_threshold(0) == 0.2  # 1/5
    assert abs(scheduler.get_complexity_threshold(50) - 0.6) < 1e-6  # Linear midpoint
    assert scheduler.get_complexity_threshold(100) == 1.0

    # Test sample inclusion
    assert scheduler.should_include_sample(0.1, epoch=0) == True
    assert scheduler.should_include_sample(0.5, epoch=0) == False
    assert scheduler.should_include_sample(0.5, epoch=50) == True


def test_model_forward(sample_config, sample_graph_data, device):
    """Test model forward pass."""
    model = HierarchicalAttentionPoolingModel(sample_config)
    model = model.to(device)
    model.eval()

    # Forward pass
    with torch.no_grad():
        logits = model(sample_graph_data)

    assert logits.size(0) == 1  # Batch size
    assert logits.size(1) == sample_config['model']['num_classes']


def test_model_predict(sample_config, sample_graph_data, device):
    """Test model prediction methods."""
    model = HierarchicalAttentionPoolingModel(sample_config)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Test predict (probabilities)
        probs = model.predict(sample_graph_data)
        assert probs.size(0) == 1
        assert probs.size(1) == sample_config['model']['num_classes']
        assert torch.allclose(probs.sum(dim=1), torch.ones(1))

        # Test predict_class
        pred_class = model.predict_class(sample_graph_data)
        assert pred_class.size(0) == 1
        assert 0 <= pred_class.item() < sample_config['model']['num_classes']


def test_baseline_model(sample_config, sample_graph_data, device):
    """Test baseline model without hierarchical pooling."""
    config = sample_config.copy()
    config['model']['use_hierarchical_pooling'] = False

    model = HierarchicalAttentionPoolingModel(config)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(sample_graph_data)

    assert logits.size(0) == 1
    assert logits.size(1) == config['model']['num_classes']


def test_model_parameter_count(sample_config):
    """Test parameter counting."""
    model = HierarchicalAttentionPoolingModel(sample_config)
    param_count = model.count_parameters()

    assert param_count > 0
    assert isinstance(param_count, int)
