#!/usr/bin/env python
"""Training script for hierarchical attention pooling model."""

import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import os
import json

import torch
import numpy as np

from hierarchical_attention_pooling_for_molecular_scaffold_transfer.utils.config import (
    load_config,
    save_config,
    set_seed,
    setup_logging
)
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.data.loader import get_dataloader
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.models.model import HierarchicalAttentionPoolingModel
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.training.trainer import Trainer
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.evaluation.analysis import plot_training_curves

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train hierarchical attention pooling model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting training script...")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

        # Set random seed
        seed = args.seed if args.seed is not None else config.get('data', {}).get('seed', 42)
        set_seed(seed)

        # Setup device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Create output directories
        results_dir = config.get('evaluation', {}).get('results_dir', 'results')
        checkpoints_dir = config.get('training', {}).get('save_dir', 'checkpoints')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Save configuration
        config_save_path = os.path.join(results_dir, 'config.yaml')
        save_config(config, config_save_path)

        # Initialize MLflow (optional)
        use_mlflow = config.get('training', {}).get('use_mlflow', False)
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_experiment("hierarchical_attention_pooling")
                mlflow.start_run()
                mlflow.log_params({
                    'dataset': config.get('data', {}).get('dataset_name', 'BBBP'),
                    'hidden_dim': config.get('model', {}).get('hidden_dim', 128),
                    'num_gnn_layers': config.get('model', {}).get('num_gnn_layers', 3),
                    'learning_rate': config.get('training', {}).get('learning_rate', 0.001),
                    'batch_size': config.get('data', {}).get('batch_size', 32),
                })
                logger.info("MLflow tracking enabled")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}")
                use_mlflow = False

        # Load data
        logger.info("Loading data...")
        train_loader, val_loader, test_loader, num_node_features, num_classes = get_dataloader(config)

        # Update config with actual feature dimensions
        config['model']['node_feat_dim'] = num_node_features
        config['model']['num_classes'] = num_classes

        logger.info(f"Data loaded: train={len(train_loader.dataset)}, "
                   f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

        # Create model
        logger.info("Initializing model...")
        model = HierarchicalAttentionPoolingModel(config)
        logger.info(f"Model has {model.count_parameters():,} parameters")

        # Create trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )

        # Train model
        logger.info("Starting training loop...")
        history = trainer.train()

        # Save training history
        history_path = os.path.join(results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        # Plot training curves
        plot_training_curves(history, save_dir=results_dir)

        # Log to MLflow
        if use_mlflow:
            try:
                mlflow.log_metrics({
                    'best_val_loss': trainer.best_val_loss,
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                })
                mlflow.log_artifacts(results_dir)
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Results saved to: {results_dir}")
        logger.info(f"Best model saved to: {os.path.join(checkpoints_dir, 'best_model.pth')}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
