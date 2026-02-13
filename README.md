# Hierarchical Attention Pooling for Molecular Scaffold Transfer

A deep learning approach for molecular property prediction that introduces hierarchical attention pooling to capture scaffold-aware representations. The model aggregates atom features within functional groups before global pooling, combined with scaffold-based curriculum learning for improved generalization across diverse chemical structures.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train the full model with hierarchical attention pooling:

```bash
python scripts/train.py --config configs/default.yaml
```

Train baseline model without hierarchical pooling (ablation study):

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate trained model on test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

### Prediction

Make predictions on new molecules:

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pth --smiles "CCO"
```

## Methodology

### Core Innovation

Traditional graph pooling methods (e.g., global mean/max pooling) treat all atoms equally, ignoring the hierarchical nature of molecular structure. This project introduces **hierarchical attention pooling** to capture scaffold-aware representations by exploiting the natural hierarchy of molecules:

1. **Atoms** → grouped into **Functional Groups** → aggregated to **Graph-level** representation

This hierarchy mirrors chemical intuition: functional groups determine reactivity and properties, while scaffolds provide the structural backbone that influences molecular behavior across similar compounds.

### Architecture

The model consists of three main components:

1. **GNN Encoder**: Graph neural network (GCN/GAT/GIN) to learn node embeddings from molecular structure
2. **Hierarchical Attention Pooling** (Core Innovation): Two-level pooling mechanism
   - **Level 1 - Functional Group Pooling**: Multi-head attention aggregates atom features within detected functional groups (e.g., hydroxyl, carboxyl, amine). This captures local chemical motifs.
   - **Level 2 - Graph-level Attention**: Self-attention over functional group representations produces the final graph embedding. This captures interactions between functional groups.
3. **MLP Classifier**: Final prediction head with dropout regularization

### Additional Novel Components

- **Scaffold-Aware Contrastive Loss**: Custom loss function that combines cross-entropy with a contrastive term encouraging molecules with similar scaffolds to have similar embeddings. This improves transfer learning across scaffold families.
- **Curriculum Learning Scheduler**: Progressively trains on molecules from simple (few functional groups) to complex (many functional groups), improving convergence and generalization.
- **SMARTS-based Functional Group Detection**: Identifies 10+ common functional groups using chemical patterns, enabling structured pooling.

### Why This Matters

Standard molecular property prediction models struggle with **scaffold transfer**: generalizing to molecules with new scaffolds not seen during training. This is critical for drug discovery, where novel scaffolds are the goal.

By explicitly modeling the scaffold-functional group hierarchy and training with scaffold-aware objectives, this approach:
- Learns representations that transfer across scaffold families
- Reduces overfitting to training scaffolds
- Better captures the chemical intuition that similar functional groups on different scaffolds often exhibit similar properties

## Key Features

- **Scaffold-aware splitting**: Prevents data leakage by grouping molecules with similar scaffolds
- **Functional group detection**: Identifies chemical functional groups using SMARTS patterns
- **Curriculum learning**: Progressive training from simple to complex molecular structures
- **Custom loss function**: Scaffold-aware contrastive loss for improved transfer learning
- **Mixed precision training**: Faster training with automatic mixed precision (AMP)

## Results

Run `python scripts/train.py` to reproduce. Results will be saved to `results/` directory.

| Model | ROC-AUC | Accuracy | F1 Score |
|-------|---------|----------|----------|
| Hierarchical Pooling | TBD | TBD | TBD |
| Baseline (Global Mean) | TBD | TBD | TBD |

Results are dataset-dependent. Default configuration uses BBBP (Blood-Brain Barrier Penetration) from MoleculeNet.

## Configuration

All hyperparameters are specified in YAML config files:

- `configs/default.yaml`: Full model with hierarchical attention pooling
- `configs/ablation.yaml`: Baseline model for ablation study

Key parameters:

- `model.use_hierarchical_pooling`: Enable/disable hierarchical pooling
- `model.functional_group_pooling`: Enable/disable functional group detection
- `training.use_curriculum`: Enable/disable curriculum learning
- `training.learning_rate`: Learning rate for optimizer
- `training.scheduler`: LR scheduler type (cosine, step, plateau)

## Project Structure

```
├── configs/               # Configuration files
├── src/                   # Source code
│   └── hierarchical_attention_pooling_for_molecular_scaffold_transfer/
│       ├── data/         # Data loading and preprocessing
│       ├── models/       # Model implementation
│       ├── training/     # Training loop
│       ├── evaluation/   # Metrics and analysis
│       └── utils/        # Utilities
├── scripts/              # Training, evaluation, prediction scripts
├── tests/                # Unit tests
└── requirements.txt      # Dependencies
```

## Testing

Run unit tests:

```bash
pytest tests/ -v --cov=src
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- RDKit >= 2023.3.1
- DGL >= 1.1.0

See `requirements.txt` for complete list.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
