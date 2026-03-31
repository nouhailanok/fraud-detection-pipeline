"""
federated/client.py — Flower federated-learning client.

This module contains two FL client implementations:

1. **FraudNetFLClient** (default) — PyTorch FraudNet model with Opacus
   Differential Privacy.  Used when torch is available.

2. **LogisticFraudClient** (fallback) — pure-NumPy logistic-regression
   client kept for environments without a PyTorch installation.

The ``main()`` entry-point selects ``FraudNetFLClient`` automatically
and falls back to ``LogisticFraudClient`` if PyTorch is absent.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import flwr as fl
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Ensure repo root is on the path so local modules (models, dataset, …) resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Shared data-loading helpers (unchanged from original implementation)
# ---------------------------------------------------------------------------

def _client_index(client_id: str) -> int:
    digits = "".join(ch for ch in client_id if ch.isdigit())
    if not digits:
        return 0
    return max(int(digits) - 1, 0)


def _load_from_tensors(client_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    tensors_dir = Path(__file__).parent.parent / "data" / "tensors"
    x_files = sorted(tensors_dir.glob("X_batch_*.npy"))
    y_files = sorted(tensors_dir.glob("y_batch_*.npy"))
    if not x_files or not y_files:
        return None

    x_map = {file.name.replace("X_batch_", ""): file for file in x_files}
    y_map = {file.name.replace("y_batch_", ""): file for file in y_files}
    common_suffixes = sorted(set(x_map.keys()).intersection(set(y_map.keys())))
    if not common_suffixes:
        return None

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for suffix in common_suffixes:
        x_parts.append(np.load(x_map[suffix]).astype(np.float32))
        y_parts.append(np.load(y_map[suffix]).astype(np.float32))

    x_all = np.vstack(x_parts)
    y_all = np.vstack(y_parts).reshape(-1)

    idx = _client_index(client_id)
    shard_idx = np.arange(x_all.shape[0]) % 4 == idx
    x_local = x_all[shard_idx]
    y_local = y_all[shard_idx]

    if x_local.shape[0] == 0:
        return None
    return x_local, y_local


def _load_from_csv(client_id: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    csv_path = Path(__file__).parent.parent / "data" / "fraudTrain.csv"
    usecols = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "unix_time", "is_fraud"]
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df.dropna()

    idx = _client_index(client_id)
    df = df.iloc[idx::4].reset_index(drop=True)

    y = df["is_fraud"].astype(np.float32).to_numpy()
    x = df.drop(columns=["is_fraud"]).astype(np.float32).to_numpy()

    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - x_mean) / x_std

    max_rows = int(os.getenv("FL_MAX_LOCAL_ROWS", "5000"))
    if x.shape[0] > max_rows:
        x = x[:max_rows]
        y = y[:max_rows]

    return x, y


def load_local_dataset(client_id: str) -> Tuple[np.ndarray, np.ndarray]:
    tensor_data = _load_from_tensors(client_id)
    if tensor_data is not None:
        return tensor_data
    return _load_from_csv(client_id)


def _read_bytes(path_str: Optional[str]) -> Optional[bytes]:
    if not path_str:
        return None
    file_path = Path(path_str)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


def start_client_compatible(client: fl.client.NumPyClient, server_address: str) -> None:
    root_cert = _read_bytes(os.getenv("FLOWER_CA_CERT"))
    client_cert = _read_bytes(os.getenv("FLOWER_CLIENT_CERT"))
    client_key = _read_bytes(os.getenv("FLOWER_CLIENT_KEY"))

    start_client_fn = getattr(fl.client, "start_client", None)
    if start_client_fn is not None:
        kwargs: dict[str, Any] = {"server_address": server_address, "client": client.to_client()}
        signature = inspect.signature(start_client_fn)

        if "root_certificates" in signature.parameters and root_cert is not None:
            kwargs["root_certificates"] = root_cert

        if "certificates" in signature.parameters and client_cert is not None and client_key is not None:
            kwargs["certificates"] = (client_cert, client_key)

        start_client_fn(**kwargs)
        return

    numpy_client_fn = getattr(fl.client, "start_numpy_client")
    kwargs = {"server_address": server_address, "client": client}
    signature = inspect.signature(numpy_client_fn)

    if "root_certificates" in signature.parameters and root_cert is not None:
        kwargs["root_certificates"] = root_cert

    if "certificates" in signature.parameters and client_cert is not None and client_key is not None:
        kwargs["certificates"] = (client_cert, client_key)

    numpy_client_fn(**kwargs)


# ---------------------------------------------------------------------------
# Client 1: PyTorch FraudNet + Differential Privacy (preferred)
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    _TORCH_AVAILABLE = True
except ImportError:
    pass

_SKLEARN_AVAILABLE = False
try:
    from sklearn.metrics import f1_score as _sklearn_f1

    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


class FraudNetFLClient(fl.client.NumPyClient):
    """
    Flower NumPyClient that trains a FraudNet model with Differential Privacy.

    Workflow per federated round
    ----------------------------
    1. Receive global model weights from the server (``set_parameters``).
    2. Attach the DP engine (Opacus) to the model, optimizer and DataLoader.
    3. Train for ``local_epochs`` with DP-noise on every gradient step.
    4. Check that ε has not exceeded the configured limit (< 1.0).
    5. Send updated weights + metrics back to the server (``fit``).
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        *,
        client_id: str = "client-1",
        dp_enabled: bool = True,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for FraudNetFLClient.  Install torch.")

        from models.model import FraudNet, get_model_parameters, set_model_parameters
        from dataset import get_dataloader
        from federated.federated_config import PipelineConfig

        self._get_model_params = get_model_parameters
        self._set_model_params = set_model_parameters
        self._get_dataloader = get_dataloader

        cfg = PipelineConfig()
        self.cfg = cfg
        self.client_id = client_id
        self.dp_enabled = dp_enabled

        self.model = FraudNet(
            input_dim=x_train.shape[1],
            latent_dim=cfg.model.latent_dim,
            hidden_dims=cfg.model.hidden_dims,
            dropout=cfg.model.dropout,
        )

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self._round_epsilons: list[float] = []

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return self._get_model_params(self.model)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self._set_model_params(self.model, parameters)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        self.set_parameters(parameters)

        lr = float(config.get("lr", self.cfg.fl.learning_rate))
        local_epochs = int(config.get("local_epochs", self.cfg.fl.local_epochs))
        batch_size = int(config.get("batch_size", self.cfg.fl.batch_size))

        train_loader = self._get_dataloader(self.x_train, self.y_train, batch_size=batch_size)
        optimizer = Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        epsilon_spent = 0.0

        if self.dp_enabled:
            from privacy_engine import DPPrivacyEngine, DPConfig

            dp_cfg = DPConfig(
                max_grad_norm=self.cfg.dp.max_grad_norm,
                target_epsilon=self.cfg.dp.target_epsilon,
                target_delta=self.cfg.dp.target_delta,
                noise_multiplier=(
                    self.cfg.dp.noise_multiplier if self.cfg.dp.noise_multiplier > 0 else None
                ),
                epochs=local_epochs,
            )
            dp_engine = DPPrivacyEngine(dp_cfg)
            self.model, optimizer, train_loader = dp_engine.attach(
                self.model, optimizer, train_loader
            )

        self.model.train()
        for _epoch in range(local_epochs):
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.model(x_batch).squeeze(1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                if self.dp_enabled:
                    dp_engine.step_end()

        if self.dp_enabled:
            budget = dp_engine.current_budget()
            epsilon_spent = budget.epsilon
            self._round_epsilons.append(epsilon_spent)
            logger.info(
                "[%s] Round complete | ε=%.4f (limit=%.2f) | δ=%.1e",
                self.client_id,
                epsilon_spent,
                self.cfg.dp.target_epsilon,
                self.cfg.dp.target_delta,
            )
            try:
                dp_engine.enforce_epsilon_limit()
            except RuntimeError as exc:
                logger.error("[%s] %s", self.client_id, exc)

        # Unwrap the Opacus GradSampleModule wrapper to get the original model.
        # Opacus stores the original module as `._module` (private API stable since v1.0).
        # As a fallback, use the model itself if not wrapped.
        actual_model = getattr(self.model, "_module", self.model)
        params_out = self._get_model_params(actual_model)

        train_metrics = self._compute_metrics(self.x_train, self.y_train)
        train_metrics["epsilon"] = float(epsilon_spent)

        return params_out, len(self.x_train), train_metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        self.set_parameters(parameters)
        metrics = self._compute_metrics(self.x_val, self.y_val)
        return float(metrics["loss"]), len(self.x_val), metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, x_data: np.ndarray, y_data: np.ndarray) -> dict[str, Any]:
        # Use the unwrapped model for evaluation
        eval_model = getattr(self.model, "_module", self.model)
        eval_model.eval()
        with torch.no_grad():
            logits = eval_model(
                torch.tensor(x_data, dtype=torch.float32)
            ).squeeze(1)
            probs = torch.sigmoid(logits).numpy()

        preds = (probs >= 0.5).astype(np.float32)
        eps = 1e-7
        loss = float(
            -np.mean(
                y_data * np.log(probs + eps) + (1.0 - y_data) * np.log(1.0 - probs + eps)
            )
        )
        accuracy = float((preds == y_data).mean())

        f1 = 0.0
        if _SKLEARN_AVAILABLE:
            try:
                f1 = float(_sklearn_f1(y_data, preds, zero_division=0))
            except Exception:
                pass

        fp = float(((preds == 1) & (y_data == 0)).sum())
        tn = float(((preds == 0) & (y_data == 0)).sum())
        fpr = fp / (fp + tn + 1e-7)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1_score": f1,
            "false_positive_rate": fpr,
        }


# ---------------------------------------------------------------------------
# Client 2: NumPy logistic-regression fallback (no PyTorch required)
# ---------------------------------------------------------------------------

def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))


class LogisticFraudClient(fl.client.NumPyClient):
    """Pure-NumPy logistic-regression fallback for environments without PyTorch."""

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.weights = np.zeros((x_train.shape[1],), dtype=np.float32)
        self.bias = np.zeros((1,), dtype=np.float32)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return [self.weights, self.bias]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.weights = parameters[0].astype(np.float32)
        self.bias = parameters[1].astype(np.float32)

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, float]]:
        self.set_parameters(parameters)
        lr = float(config.get("lr", os.getenv("FL_LR", "0.05")))
        local_epochs = int(config.get("local_epochs", os.getenv("FL_LOCAL_EPOCHS", "1")))

        for _ in range(local_epochs):
            logits = self.x_train @ self.weights + self.bias[0]
            probs = sigmoid(logits)
            error = probs - self.y_train

            grad_w = (self.x_train.T @ error) / max(self.x_train.shape[0], 1)
            grad_b = np.array([error.mean()], dtype=np.float32)

            self.weights -= lr * grad_w.astype(np.float32)
            self.bias -= lr * grad_b

        metrics = self._compute_metrics(self.x_train, self.y_train)
        return self.get_parameters(config), len(self.x_train), metrics

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[float, int, dict[str, float]]:
        self.set_parameters(parameters)
        metrics = self._compute_metrics(self.x_val, self.y_val)
        return float(metrics["loss"]), len(self.x_val), metrics

    def _compute_metrics(self, x_data: np.ndarray, y_data: np.ndarray) -> dict[str, float]:
        logits = x_data @ self.weights + self.bias[0]
        probs = sigmoid(logits)
        preds = (probs >= 0.5).astype(np.float32)

        eps = 1e-7
        loss = -np.mean(y_data * np.log(probs + eps) + (1.0 - y_data) * np.log(1.0 - probs + eps))
        acc = float((preds == y_data).mean())

        return {"loss": float(loss), "accuracy": acc}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    from federated.federated_config import PipelineConfig

    cfg = PipelineConfig()
    client_id = cfg.fl.client_id
    server_address = f"{os.getenv('FLOWER_SERVER_HOST', 'flower')}:{cfg.fl.server_port}"
    retry_seconds = cfg.fl.retry_seconds
    continuous = cfg.fl.continuous

    x_data, y_data = load_local_dataset(client_id)
    if x_data.shape[0] < 10:
        raise RuntimeError(f"Pas assez de données locales pour {client_id} (rows={x_data.shape[0]})")

    split = max(int(x_data.shape[0] * 0.8), 1)
    x_train, x_val = x_data[:split], x_data[split:]
    y_train, y_val = y_data[:split], y_data[split:]

    if x_val.shape[0] == 0:
        x_val, y_val = x_train, y_train

    logger.info("[FL-CLIENT] %s -> %s", client_id, server_address)
    logger.info("[FL-CLIENT] train=%d | val=%d | features=%d", len(x_train), len(x_val), x_train.shape[1])

    if _TORCH_AVAILABLE:
        logger.info("[FL-CLIENT] Using FraudNetFLClient (PyTorch + DP)")
        fl_client: fl.client.NumPyClient = FraudNetFLClient(
            x_train, y_train, x_val, y_val,
            client_id=client_id,
            dp_enabled=cfg.dp.enabled,
        )
    else:
        logger.warning("[FL-CLIENT] PyTorch unavailable — using LogisticFraudClient fallback")
        fl_client = LogisticFraudClient(x_train, y_train, x_val, y_val)

    while True:
        try:
            start_client_compatible(fl_client, server_address)
            logger.info("[FL-CLIENT] Session terminée proprement")
        except Exception as exc:
            logger.error("[FL-CLIENT] Erreur de connexion/transport: %s", exc)

        if not continuous:
            break

        logger.info("[FL-CLIENT] Nouvelle tentative dans %ds…", retry_seconds)
        time.sleep(retry_seconds)


if __name__ == "__main__":
    main()
