#!/usr/bin/env python
"""Prediction script for hierarchical attention pooling model."""

import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging

import torch
import numpy as np

from hierarchical_attention_pooling_for_molecular_scaffold_transfer.utils.config import setup_logging
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.models.model import HierarchicalAttentionPoolingModel
from hierarchical_attention_pooling_for_molecular_scaffold_transfer.data.preprocessing import mol_to_graph
from torch_geometric.data import Data as PyGData

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions with hierarchical attention pooling model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--smiles',
        type=str,
        required=True,
        help='SMILES string to predict on'
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
    """Main prediction function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting prediction script...")

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
        config = checkpoint['config']

        # Create model
        logger.info("Initializing model...")
        model = HierarchicalAttentionPoolingModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        logger.info(f"Loaded model with {model.count_parameters():,} parameters")

        # Process input SMILES
        logger.info(f"Processing SMILES: {args.smiles}")
        graph_data = mol_to_graph(args.smiles, include_functional_groups=True)

        if graph_data is None:
            logger.error(f"Invalid SMILES string: {args.smiles}")
            sys.exit(1)

        # Convert to PyTorch Geometric Data
        data = PyGData(
            x=graph_data['node_features'],
            edge_index=graph_data['edge_index'],
            edge_attr=graph_data['edge_features'],
        )

        # Create batch
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        data = data.to(device)

        # Make prediction
        with torch.no_grad():
            logits = model(data)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        # Print results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"SMILES:           {args.smiles}")
        print(f"Predicted Class:  {pred_class}")
        print(f"Confidence:       {confidence:.4f}")
        print("\nClass Probabilities:")
        for i, prob in enumerate(probs[0]):
            print(f"  Class {i}: {prob.item():.4f}")
        print("="*60 + "\n")

        logger.info("Prediction completed successfully!")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
