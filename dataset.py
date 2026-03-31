"""
TransactionDataset — PyTorch Dataset / DataLoader for local silo data.

Supports two data sources:
  1. Pre-computed NumPy tensor files (.npy) produced by the ingestion pipeline.
  2. Raw CSV files (fraudTrain.csv / local_data.csv) as a fallback.

The dataset is kept intentionally flat (2-D, no temporal sequences) so it
is directly compatible with the FraudNet feed-forward architecture.  If you
need temporal sequences for the GRU model, use data/dataloader.py instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_CSV_FEATURE_COLS = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "unix_time",
]
_CSV_LABEL_COL = "is_fraud"

# Expected number of feature dimensions produced by the ingestion vectorizer.
# Column 0 is the PAN_HASH identifier; columns 1-26 are the real features.
_TENSOR_N_COLS_WITH_ID = 27
_TENSOR_FEATURE_START = 1  # skip PAN_HASH at index 0


class TransactionDataset:
    """
    PyTorch Dataset for local silo transaction data.

    Parameters
    ----------
    x_data:
        Feature matrix, shape (n_samples, n_features).
    y_data:
        Binary label vector, shape (n_samples,).
    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for TransactionDataset.  Install it with: pip install torch"
            ) from exc

        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                f"x_data rows ({x_data.shape[0]}) != y_data rows ({y_data.shape[0]})"
            )
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> "Tuple[torch.Tensor, torch.Tensor]":
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Loaders from different data sources
# ---------------------------------------------------------------------------

def _normalise(x: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance normalisation (column-wise)."""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def load_from_tensors(
    tensors_dir: Union[str, Path],
    *,
    shard_index: Optional[int] = None,
    n_shards: int = 4,
    max_rows: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load feature/label data from .npy batch files.

    Parameters
    ----------
    tensors_dir:
        Directory containing ``X_batch_*.npy`` and ``y_batch_*.npy`` files.
    shard_index:
        When set, keep only every *n_shards*-th row starting at *shard_index*
        (0-based).  Simulates heterogeneous silos from a shared tensor store.
    n_shards:
        Total number of shards (default 4).
    max_rows:
        Truncate to this many rows after sharding.

    Returns
    -------
    ``(X, y)`` arrays or ``None`` if no matching files were found.
    """
    tensors_dir = Path(tensors_dir)
    x_files = sorted(tensors_dir.glob("X_batch_*.npy"))
    y_files = sorted(tensors_dir.glob("y_batch_*.npy"))

    if not x_files or not y_files:
        return None

    x_map = {f.name.replace("X_batch_", ""): f for f in x_files}
    y_map = {f.name.replace("y_batch_", ""): f for f in y_files}
    common = sorted(set(x_map) & set(y_map))
    if not common:
        return None

    x_parts = [np.load(x_map[s]).astype(np.float32) for s in common]
    y_parts = [np.load(y_map[s]).astype(np.float32) for s in common]

    x_all = np.vstack(x_parts)
    y_all = np.vstack(y_parts).reshape(-1)

    # Strip the PAN_HASH identifier column if it is present
    if x_all.shape[1] == _TENSOR_N_COLS_WITH_ID:
        x_all = x_all[:, _TENSOR_FEATURE_START:]

    if shard_index is not None:
        mask = np.arange(x_all.shape[0]) % n_shards == shard_index
        x_all, y_all = x_all[mask], y_all[mask]

    if max_rows and x_all.shape[0] > max_rows:
        x_all, y_all = x_all[:max_rows], y_all[:max_rows]

    return x_all, y_all


def load_from_csv(
    csv_path: Union[str, Path],
    *,
    shard_index: Optional[int] = None,
    n_shards: int = 4,
    max_rows: Optional[int] = None,
    feature_cols: Optional[list] = None,
    label_col: str = _CSV_LABEL_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load transaction data from a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the CSV (e.g. ``data/fraudTrain.csv`` or a node-local
        ``data/node_1/local_data.csv``).
    shard_index / n_shards:
        Shard selection — same semantics as :func:`load_from_tensors`.
    max_rows:
        Hard limit on the number of rows returned.
    feature_cols:
        Column names to use as features.  Defaults to
        ``_CSV_FEATURE_COLS``.
    label_col:
        Name of the binary label column.
    """
    import pandas as pd  # lazy import — pandas is heavy

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    cols = feature_cols or _CSV_FEATURE_COLS
    usecols = cols + [label_col]
    df = pd.read_csv(csv_path, usecols=usecols).dropna()

    if shard_index is not None:
        df = df.iloc[shard_index::n_shards].reset_index(drop=True)

    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows]

    y = df[label_col].astype(np.float32).to_numpy()
    x = df[cols].astype(np.float32).to_numpy()
    x = _normalise(x)

    return x, y


def load_dataset(
    *,
    tensors_dir: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
    shard_index: Optional[int] = None,
    n_shards: int = 4,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified loader: tries tensor files first, falls back to CSV.

    Raises
    ------
    RuntimeError
        If neither source produces data.
    """
    if tensors_dir is not None:
        result = load_from_tensors(
            tensors_dir,
            shard_index=shard_index,
            n_shards=n_shards,
            max_rows=max_rows,
        )
        if result is not None:
            logger.info("Loaded %d rows from tensor files in %s", result[0].shape[0], tensors_dir)
            return result

    if csv_path is not None:
        x, y = load_from_csv(
            csv_path,
            shard_index=shard_index,
            n_shards=n_shards,
            max_rows=max_rows,
        )
        logger.info("Loaded %d rows from CSV %s", x.shape[0], csv_path)
        return x, y

    raise RuntimeError("No data source provided or found.  Supply tensors_dir or csv_path.")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
) -> "DataLoader":
    """
    Wrap a (X, y) pair into a PyTorch DataLoader.

    Parameters
    ----------
    x_data, y_data:
        Feature matrix and label vector (numpy arrays).
    batch_size:
        Mini-batch size (default 64).
    shuffle:
        Randomise sample order before each epoch (recommended for training).
    """
    import torch
    from torch.utils.data import DataLoader as _DataLoader

    dataset = TransactionDataset(x_data, y_data)
    return _DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,  # required by Opacus
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    import os

    rng = np.random.default_rng(0)
    x_dummy = rng.random((200, 26)).astype(np.float32)
    y_dummy = rng.integers(0, 2, size=(200,)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "X_batch_0001.npy"), x_dummy)
        np.save(os.path.join(tmp, "y_batch_0001.npy"), y_dummy)

        x_loaded, y_loaded = load_from_tensors(tmp)  # type: ignore[misc]
        print(f"Loaded X: {x_loaded.shape}, y: {y_loaded.shape}")

    loader = get_dataloader(x_dummy, y_dummy, batch_size=32)
    x_batch, y_batch = next(iter(loader))
    print(f"Batch X: {x_batch.shape}, y: {y_batch.shape}")
