#!/usr/bin/env python
"""Evaluation script for hierarchical attention pooling model."""

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
import pandas as pd

from hierarchical_attention_pooling_for_molecular_scaffold_transfer.utils.config import (
    load_config,
    setup_logging
)
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.data.loader import get_dataloader
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.models.model import HierarchicalAttentionPoolingModel
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.evaluation.metrics import (
    evaluate_model,
    plot_confusion_matrix,
    get_per_class_metrics
)
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.evaluation.analysis import analyze_results

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate hierarchical attention pooling model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (if None, loads from checkpoint)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Directory to save evaluation results'
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
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting evaluation script...")

    try:
        # Setup device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = checkpoint['config']
        logger.info("Loaded configuration")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load data
        logger.info("Loading data...")
        train_loader, val_loader, test_loader, num_node_features, num_classes = get_dataloader(config)

        # Select data loader based on split
        if args.split == 'train':
            data_loader = train_loader
        elif args.split == 'val':
            data_loader = val_loader
        else:
            data_loader = test_loader

        logger.info(f"Evaluating on {args.split} split with {len(data_loader.dataset)} samples")

        # Update config with actual feature dimensions
        config['model']['node_feat_dim'] = num_node_features
        config['model']['num_classes'] = num_classes

        # Create model
        logger.info("Initializing model...")
        model = HierarchicalAttentionPoolingModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        logger.info(f"Loaded model with {model.count_parameters():,} parameters")

        # Evaluate model
        logger.info("Running evaluation...")
        results = evaluate_model(
            model=model,
            data_loader=data_loader,
            device=device,
            return_predictions=True
        )

        # Print results
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS - {args.split.upper()} SET")
        print("="*60)

        metrics = results['metrics']
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:20s}: {metric_value:.4f}")

        print("="*60 + "\n")

        # Save metrics
        metrics_path = os.path.join(args.output_dir, f'{args.split}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

        # Get predictions
        predictions = results['predictions']
        y_true = predictions['y_true']
        y_pred = predictions['y_pred']
        y_prob = predictions['y_prob']

        # Save predictions
        pred_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'probability_class_0': y_prob[:, 0] if y_prob.shape[1] > 1 else 1 - y_prob.flatten(),
            'probability_class_1': y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.flatten(),
        })
        pred_path = os.path.join(args.output_dir, f'{args.split}_predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Saved predictions to {pred_path}")

        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, f'{args.split}_confusion_matrix.png')
        plot_confusion_matrix(
            y_true,
            y_pred,
            save_path=cm_path,
            class_names=['Class 0', 'Class 1']
        )

        # Get per-class metrics
        per_class = get_per_class_metrics(
            y_true,
            y_pred,
            class_names=['Class 0', 'Class 1']
        )

        print("\nPER-CLASS METRICS:")
        print("-" * 60)
        for class_name, class_metrics in per_class.items():
            print(f"\n{class_name}:")
            for metric_name, metric_value in class_metrics.items():
                print(f"  {metric_name:15s}: {metric_value:.4f}")
        print("-" * 60 + "\n")

        # Save per-class metrics
        per_class_path = os.path.join(args.output_dir, f'{args.split}_per_class_metrics.json')
        with open(per_class_path, 'w') as f:
            json.dump(per_class, f, indent=2)
        logger.info(f"Saved per-class metrics to {per_class_path}")

        # Comprehensive analysis with visualizations
        analyze_results(results, save_dir=args.output_dir)

        logger.info(f"Evaluation completed! Results saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
