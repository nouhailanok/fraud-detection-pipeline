import inspect
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd


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


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))


class LogisticFraudClient(fl.client.NumPyClient):
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


def main() -> None:
    client_id = os.getenv("CLIENT_ID", "1")
    host = os.getenv("FLOWER_SERVER_HOST", "flower")
    port = int(os.getenv("FLOWER_SERVER_PORT", "8080"))
    server_address = f"{host}:{port}"
    retry_seconds = int(os.getenv("FL_CLIENT_RETRY_SECONDS", "10"))
    continuous = os.getenv("FL_CLIENT_CONTINUOUS", "true").lower() == "true"

    x_data, y_data = load_local_dataset(client_id)
    if x_data.shape[0] < 10:
        raise RuntimeError(f"Pas assez de données locales pour {client_id} (rows={x_data.shape[0]})")

    split = max(int(x_data.shape[0] * 0.8), 1)
    x_train, x_val = x_data[:split], x_data[split:]
    y_train, y_val = y_data[:split], y_data[split:]

    if x_val.shape[0] == 0:
        x_val = x_train
        y_val = y_train

    print(f"[FL-CLIENT] {client_id} -> {server_address}")
    print(f"[FL-CLIENT] train={len(x_train)} | val={len(x_val)} | features={x_train.shape[1]}")

    fl_client = LogisticFraudClient(x_train, y_train, x_val, y_val)
    while True:
        try:
            start_client_compatible(fl_client, server_address)
            print("[FL-CLIENT] Session terminée proprement")
        except Exception as exc:
            print(f"[FL-CLIENT] Erreur de connexion/transport: {exc}")

        if not continuous:
            break

        print(f"[FL-CLIENT] Nouvelle tentative dans {retry_seconds}s...")
        time.sleep(retry_seconds)


if __name__ == "__main__":
    main()
