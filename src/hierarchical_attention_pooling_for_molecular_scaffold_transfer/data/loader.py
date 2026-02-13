"""Data loading utilities for molecular datasets."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch
from rdkit import Chem
import requests
from io import StringIO

from .preprocessing import mol_to_graph, scaffold_split

logger = logging.getLogger(__name__)


class MoleculeDataset(Dataset):
    """PyTorch dataset for molecular graphs.

    Loads molecules from SMILES strings and converts them to graph
    representations suitable for GNN training.
    """

    def __init__(
        self,
        smiles_list: List[str],
        labels: np.ndarray,
        include_functional_groups: bool = True,
        cache_graphs: bool = True
    ):
        """Initialize dataset.

        Args:
            smiles_list: List of SMILES strings
            labels: Array of labels
            include_functional_groups: Whether to identify functional groups
            cache_graphs: Whether to cache processed graphs in memory
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.include_functional_groups = include_functional_groups
        self.cache_graphs = cache_graphs
        self._cache: Dict[int, Optional[PyGData]] = {}

        logger.info(f"Initialized dataset with {len(smiles_list)} molecules")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Optional[PyGData]:
        """Get a single graph.

        Args:
            idx: Index

        Returns:
            PyTorch Geometric Data object or None if invalid
        """
        if self.cache_graphs and idx in self._cache:
            return self._cache[idx]

        smiles = self.smiles_list[idx]
        label = self.labels[idx]

        graph_data = mol_to_graph(smiles, self.include_functional_groups)

        if graph_data is None:
            return None

        # Convert to PyTorch Geometric Data
        data = PyGData(
            x=graph_data['node_features'],
            edge_index=graph_data['edge_index'],
            edge_attr=graph_data['edge_features'],
            y=torch.tensor([label], dtype=torch.long),
            smiles=smiles,
        )

        if 'functional_groups' in graph_data:
            data.functional_groups = graph_data['functional_groups']

        if self.cache_graphs:
            self._cache[idx] = data

        return data


def collate_fn(batch: List[Optional[PyGData]]) -> PyGBatch:
    """Custom collate function that handles None values.

    Args:
        batch: List of PyG Data objects

    Returns:
        Batched PyG Data object
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]

    if not batch:
        # Return empty batch
        return PyGBatch()

    return PyGBatch.from_data_list(batch)


def download_moleculenet_dataset(dataset_name: str, save_dir: str = "data/raw") -> str:
    """Download a MoleculeNet dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'BBBP', 'Tox21', 'HIV')
        save_dir: Directory to save the dataset

    Returns:
        Path to the downloaded CSV file
    """
    os.makedirs(save_dir, exist_ok=True)

    # MoleculeNet dataset URLs
    base_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
    dataset_urls = {
        'BBBP': f"{base_url}/BBBP.csv",
        'Tox21': f"{base_url}/tox21.csv.gz",
        'HIV': f"{base_url}/HIV.csv",
        'BACE': f"{base_url}/bace.csv",
        'ClinTox': f"{base_url}/clintox.csv.gz",
    }

    if dataset_name not in dataset_urls:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_urls.keys())}")

    file_path = os.path.join(save_dir, f"{dataset_name}.csv")

    if os.path.exists(file_path):
        logger.info(f"Dataset already exists: {file_path}")
        return file_path

    logger.info(f"Downloading {dataset_name} dataset...")
    url = dataset_urls[dataset_name]

    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()

        # Handle gzipped files
        if url.endswith('.gz'):
            import gzip
            content = gzip.decompress(response.content).decode('utf-8')
        else:
            content = response.text

        with open(file_path, 'w') as f:
            f.write(content)

        logger.info(f"Downloaded dataset to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def load_moleculenet_dataset(
    dataset_name: str,
    split_type: str = "scaffold",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], np.ndarray, List[int], List[int], List[int]]:
    """Load and split a MoleculeNet dataset.

    Args:
        dataset_name: Name of the dataset
        split_type: Type of split ('scaffold' or 'random')
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        Tuple of (smiles_list, labels, train_idx, val_idx, test_idx)
    """
    file_path = download_moleculenet_dataset(dataset_name)

    # Load CSV
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} molecules from {dataset_name}")

    # Extract SMILES and labels
    # Identify SMILES column
    if 'smiles' in df.columns:
        smiles_col = 'smiles'
    elif 'Smiles' in df.columns:
        smiles_col = 'Smiles'
    elif 'SMILES' in df.columns:
        smiles_col = 'SMILES'
    else:
        # Assume last column is SMILES
        smiles_col = df.columns[-1]

    smiles_list = df[smiles_col].tolist()

    # Find label column - prioritize known label column names
    # Exclude identifier and SMILES columns
    exclude_cols = [smiles_col, 'mol_id', 'num', 'name', 'Name', 'ID', 'id']

    # Common label column names in MoleculeNet datasets
    label_candidates = ['p_np', 'activity', 'label', 'y', 'target', 'HIV_active', 'Class']

    label_col = None
    for candidate in label_candidates:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        # Fall back to first column that's not excluded
        label_cols = [col for col in df.columns if col not in exclude_cols]
        if not label_cols:
            raise ValueError("No label column found in dataset")
        label_col = label_cols[0]

    labels = df[label_col].values

    # Remove invalid entries
    valid_indices = []
    for i, (smi, label) in enumerate(zip(smiles_list, labels)):
        if pd.notna(smi) and pd.notna(label):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_indices.append(i)

    smiles_list = [smiles_list[i] for i in valid_indices]
    labels = labels[valid_indices].astype(np.int64)

    # Ensure labels are in valid range [0, num_classes-1]
    unique_labels = np.unique(labels)
    if len(unique_labels) > 0:
        # Map labels to contiguous range starting from 0
        label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
        labels = np.array([label_map[label] for label in labels], dtype=np.int64)

    logger.info(f"Filtered to {len(smiles_list)} valid molecules")
    logger.info(f"Original label values: {unique_labels}")
    logger.info(f"Mapped label distribution: {np.bincount(labels)}")

    # Split dataset
    if split_type == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(
            smiles_list, train_ratio, val_ratio, test_ratio, seed
        )
    else:  # random split
        np.random.seed(seed)
        indices = np.random.permutation(len(smiles_list))
        train_size = int(train_ratio * len(smiles_list))
        val_size = int(val_ratio * len(smiles_list))

        train_idx = indices[:train_size].tolist()
        val_idx = indices[train_size:train_size + val_size].tolist()
        test_idx = indices[train_size + val_size:].tolist()

    return smiles_list, labels, train_idx, val_idx, test_idx


class MoleculeDataLoader:
    """Data loader manager for molecular datasets."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data', {})

    def get_dataloaders(
        self
    ) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
        """Create train, validation, and test dataloaders.

        Returns:
            Tuple of (train_loader, val_loader, test_loader, num_node_features, num_classes)
        """
        # Load dataset
        smiles_list, labels, train_idx, val_idx, test_idx = load_moleculenet_dataset(
            dataset_name=self.data_config.get('dataset_name', 'BBBP'),
            split_type=self.data_config.get('split_type', 'scaffold'),
            train_ratio=self.data_config.get('train_ratio', 0.8),
            val_ratio=self.data_config.get('val_ratio', 0.1),
            test_ratio=self.data_config.get('test_ratio', 0.1),
            seed=self.data_config.get('seed', 42)
        )

        # Create datasets
        train_dataset = MoleculeDataset(
            [smiles_list[i] for i in train_idx],
            labels[train_idx],
            include_functional_groups=True,
            cache_graphs=True
        )

        val_dataset = MoleculeDataset(
            [smiles_list[i] for i in val_idx],
            labels[val_idx],
            include_functional_groups=True,
            cache_graphs=True
        )

        test_dataset = MoleculeDataset(
            [smiles_list[i] for i in test_idx],
            labels[test_idx],
            include_functional_groups=True,
            cache_graphs=True
        )

        # Create dataloaders
        batch_size = self.data_config.get('batch_size', 32)
        num_workers = self.data_config.get('num_workers', 4)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        # Get feature dimensions from first valid sample
        for data in train_dataset:
            if data is not None:
                num_node_features = data.x.size(1)
                break
        else:
            raise ValueError("No valid samples found in training set")

        num_classes = len(np.unique(labels))

        logger.info(f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        logger.info(f"Node features: {num_node_features}, Classes: {num_classes}")

        return train_loader, val_loader, test_loader, num_node_features, num_classes


def get_dataloader(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Convenience function to create dataloaders.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_node_features, num_classes)
    """
    loader = MoleculeDataLoader(config)
    return loader.get_dataloaders()
