"""Evaluation metrics for molecular property prediction."""

import logging
from typing import Dict, List, Optional
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'binary'
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        y_prob: Predicted probabilities [N, num_classes] (optional)
        average: Averaging strategy for multiclass

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # ROC AUC (requires probabilities)
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                # Binary classification with single probability
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            elif y_prob.shape[1] == 2:
                # Binary classification with 2 class probabilities
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multiclass
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_prob, average=average, multi_class='ovr'
                )
        except ValueError as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
            metrics['roc_auc'] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None
) -> None:
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot (optional)
        class_names: Names of classes (optional)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names or 'auto',
        yticklabels=class_names or 'auto'
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")

    plt.close()


def get_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Compute per-class metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (optional)

    Returns:
        Dictionary mapping class names to their metrics
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    per_class = {}
    num_classes = len(np.unique(y_true))

    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class_{i}"
        class_key = str(i)

        if class_key in report:
            per_class[class_name] = {
                'precision': report[class_key]['precision'],
                'recall': report[class_key]['recall'],
                'f1': report[class_key]['f1-score'],
                'support': report[class_key]['support']
            }

    return per_class


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_predictions: bool = False
) -> Dict:
    """Evaluate model on a dataset.

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        return_predictions: Whether to return all predictions

    Returns:
        Dictionary containing metrics and optionally predictions
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            if batch.num_graphs == 0:
                continue

            batch = batch.to(device)

            # Get predictions
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.append(batch.y.squeeze().cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)

    results = {
        'metrics': metrics,
    }

    if return_predictions:
        results['predictions'] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    return results
