"""Results analysis and visualization utilities."""

import logging
import os
from typing import Dict, List, Optional
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_dir: Optional[str] = None
) -> None:
    """Plot training curves for loss and learning rate.

    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots (optional)
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate
    ax2.plot(epochs, history['learning_rate'], label='Learning Rate', marker='o', markersize=3, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {path}")

    plt.close()


def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    save_dir: Optional[str] = None
) -> None:
    """Plot comparison of metrics across different configurations.

    Args:
        results: Dictionary mapping config names to metrics
        save_dir: Directory to save plot (optional)
    """
    # Extract metric names
    metric_names = list(next(iter(results.values())).keys())
    config_names = list(results.keys())

    # Prepare data for plotting
    data = {metric: [results[config][metric] for config in config_names]
            for metric in metric_names}

    # Create bar plot
    x = np.arange(len(config_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metric_names):
        offset = width * (i - len(metric_names) / 2)
        ax.bar(x + offset, data[metric], width, label=metric)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Metric Comparison Across Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'metric_comparison.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metric comparison to {path}")

    plt.close()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_dir: Optional[str] = None
) -> None:
    """Plot distribution of prediction probabilities by true class.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_dir: Directory to save plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get positive class probabilities for binary classification
    if y_prob.ndim == 1:
        pos_probs = y_prob
    elif y_prob.shape[1] == 2:
        pos_probs = y_prob[:, 1]
    else:
        # For multiclass, just use max probability
        pos_probs = y_prob.max(axis=1)

    # Plot distribution by true class
    for class_label in np.unique(y_true):
        mask = y_true == class_label
        axes[0].hist(
            pos_probs[mask],
            bins=30,
            alpha=0.6,
            label=f'Class {class_label}',
            density=True
        )

    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Prediction Distribution by True Class')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot calibration curve (predicted prob vs actual frequency)
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    true_freqs = []
    for i in range(len(bins) - 1):
        mask = (pos_probs >= bins[i]) & (pos_probs < bins[i+1])
        if mask.sum() > 0:
            true_freqs.append(y_true[mask].mean())
        else:
            true_freqs.append(np.nan)

    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[1].plot(bin_centers, true_freqs, 'o-', label='Model', markersize=8)
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Actual Frequency')
    axes[1].set_title('Calibration Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'prediction_distribution.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction distribution to {path}")

    plt.close()


def analyze_results(
    results: Dict,
    history: Optional[Dict] = None,
    save_dir: Optional[str] = None
) -> None:
    """Comprehensive analysis of results with visualizations.

    Args:
        results: Dictionary containing evaluation results
        history: Training history (optional)
        save_dir: Directory to save analysis (optional)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save metrics as JSON
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results.get('metrics', {}), f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

    # Plot training curves
    if history:
        plot_training_curves(history, save_dir)

    # Plot prediction distribution
    if 'predictions' in results:
        preds = results['predictions']
        plot_prediction_distribution(
            preds['y_true'],
            preds['y_prob'],
            save_dir
        )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    if 'metrics' in results:
        for metric, value in results['metrics'].items():
            print(f"{metric:20s}: {value:.4f}")

    print("="*60 + "\n")
