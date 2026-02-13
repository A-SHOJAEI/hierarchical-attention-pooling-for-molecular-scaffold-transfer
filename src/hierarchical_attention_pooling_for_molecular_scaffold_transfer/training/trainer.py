"""Training loop with early stopping and learning rate scheduling."""

import logging
import os
from typing import Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..models.components import ScaffoldAwareLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if should stop training.

        Args:
            val_loss: Current validation loss

        Returns:
            Whether to stop training
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


class Trainer:
    """Trainer class for molecular property prediction."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        config: Dict,
        device: torch.device
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_config = config.get('training', {})
        model_config = config.get('model', {})

        # Training parameters
        self.num_epochs = train_config.get('num_epochs', 200)
        self.learning_rate = train_config.get('learning_rate', 0.001)
        self.weight_decay = train_config.get('weight_decay', 0.00001)
        self.gradient_clip = train_config.get('gradient_clip', 1.0)
        self.log_interval = train_config.get('log_interval', 10)

        # Setup optimizer
        optimizer_name = train_config.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Setup learning rate scheduler
        scheduler_name = train_config.get('scheduler', 'cosine').lower()
        if scheduler_name == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate * 0.01
            )
        elif scheduler_name == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        elif scheduler_name == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=train_config.get('scheduler_factor', 0.5),
                patience=train_config.get('scheduler_patience', 10)
            )
        else:
            self.scheduler = None

        # Setup loss function
        label_smoothing = train_config.get('label_smoothing', 0.0)
        num_classes = model_config.get('num_classes', 2)

        self.criterion = ScaffoldAwareLoss(
            num_classes=num_classes,
            scaffold_weight=0.0,  # Disabled for now (no scaffold IDs in batch)
            label_smoothing=label_smoothing
        )

        # Mixed precision training
        self.use_amp = train_config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Early stopping
        self.use_early_stopping = train_config.get('early_stopping', True)
        if self.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=train_config.get('patience', 30),
                min_delta=train_config.get('min_delta', 0.0001)
            )

        # Checkpointing
        self.save_dir = train_config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_best_only = train_config.get('save_best_only', True)
        self.best_val_loss = float('inf')

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            if batch.num_graphs == 0:
                continue

            batch = batch.to(self.device)

            # Validate labels are in valid range
            labels = batch.y.squeeze()
            if labels.max() >= self.criterion.num_classes or labels.min() < 0:
                logger.error(f"Invalid labels detected: min={labels.min()}, max={labels.max()}, num_classes={self.criterion.num_classes}")
                continue

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    logits = self.model(batch)
                    loss = self.criterion(logits, labels)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(batch)
                loss = self.criterion(logits, labels)

                loss.backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Validate model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            if batch.num_graphs == 0:
                continue

            batch = batch.to(self.device)

            # Validate labels
            labels = batch.y.squeeze()
            if labels.max() >= self.criterion.num_classes or labels.min() < 0:
                logger.error(f"Invalid labels in validation: min={labels.min()}, max={labels.max()}")
                continue

            logits = self.model(batch)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }

        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            logger.info(f"Saved best model to {path}")

        if not self.save_best_only:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, path)

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            elif not self.save_best_only:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.use_early_stopping:
                if self.early_stopping(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.history
