"""
federated_config.py — centralised configuration for the federated learning pipeline.

All tuneable hyperparameters live here so that server.py, client.py, and
privacy_engine.py share a single source of truth.  Values can be overridden
via environment variables (useful for Docker/Kubernetes deployments).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Federated Learning parameters
# ---------------------------------------------------------------------------

@dataclass
class FLConfig:
    """Top-level Flower FL configuration."""

    # Server
    rounds: int = field(default_factory=lambda: _env_int("FL_ROUNDS", 3))
    min_clients: int = field(default_factory=lambda: _env_int("FL_MIN_CLIENTS", 3))
    server_host: str = field(default_factory=lambda: os.getenv("FLOWER_SERVER_HOST", "0.0.0.0"))
    server_port: int = field(default_factory=lambda: _env_int("FLOWER_SERVER_PORT", 8080))

    # Client
    client_id: str = field(default_factory=lambda: os.getenv("CLIENT_ID", "client-1"))
    local_epochs: int = field(default_factory=lambda: _env_int("FL_LOCAL_EPOCHS", 1))
    learning_rate: float = field(default_factory=lambda: _env_float("FL_LR", 0.001))
    batch_size: int = field(default_factory=lambda: _env_int("FL_BATCH_SIZE", 64))
    max_local_rows: int = field(default_factory=lambda: _env_int("FL_MAX_LOCAL_ROWS", 5000))
    retry_seconds: int = field(default_factory=lambda: _env_int("FL_CLIENT_RETRY_SECONDS", 10))
    continuous: bool = field(default_factory=lambda: _env_bool("FL_CLIENT_CONTINUOUS", True))

    @property
    def server_address(self) -> str:
        return f"{self.server_host}:{self.server_port}"


# ---------------------------------------------------------------------------
# Differential Privacy parameters
# ---------------------------------------------------------------------------

@dataclass
class DPConfigFL:
    """
    Differential Privacy configuration that satisfies ε < 1.0.

    Defaults are calibrated for a high-privacy scenario:
      - ε = 0.9  (< 1.0 as required)
      - δ = 1e-6
      - C = 1.0  (L2 clipping threshold)
    """

    enabled: bool = field(default_factory=lambda: _env_bool("DP_ENABLED", True))
    target_epsilon: float = field(default_factory=lambda: _env_float("DP_EPSILON", 0.9))
    target_delta: float = field(default_factory=lambda: _env_float("DP_DELTA", 1e-6))
    max_grad_norm: float = field(default_factory=lambda: _env_float("DP_MAX_GRAD_NORM", 1.0))
    noise_multiplier: float = field(
        default_factory=lambda: _env_float("DP_NOISE_MULTIPLIER", 0.0)
    )
    """
    When 0.0 (default), Opacus auto-calibrates σ to meet (ε, δ).
    Set to a positive value to override σ directly.
    """
    accountant: str = field(default_factory=lambda: os.getenv("DP_ACCOUNTANT", "rdp"))

    def __post_init__(self) -> None:
        if self.target_epsilon >= 1.0:
            raise ValueError(
                f"DP_EPSILON must be < 1.0 for GDPR/PCI-DSS compliance (got {self.target_epsilon})"
            )


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """FraudNet architecture configuration."""

    input_dim: int = field(default_factory=lambda: _env_int("MODEL_INPUT_DIM", 26))
    latent_dim: int = field(default_factory=lambda: _env_int("MODEL_LATENT_DIM", 128))
    hidden_dims: tuple = (64, 32)
    dropout: float = field(default_factory=lambda: _env_float("MODEL_DROPOUT", 0.3))


# ---------------------------------------------------------------------------
# TLS / mTLS parameters (mirrors MTLS_SETUP.md)
# ---------------------------------------------------------------------------

@dataclass
class TLSConfig:
    """Paths to TLS certificates for mTLS-secured FL communication."""

    ca_cert: str = field(default_factory=lambda: os.getenv("FLOWER_CA_CERT", ""))
    server_cert: str = field(default_factory=lambda: os.getenv("FLOWER_TLS_SERVER_CERT", ""))
    server_key: str = field(default_factory=lambda: os.getenv("FLOWER_TLS_SERVER_KEY", ""))
    client_cert: str = field(default_factory=lambda: os.getenv("FLOWER_CLIENT_CERT", ""))
    client_key: str = field(default_factory=lambda: os.getenv("FLOWER_CLIENT_KEY", ""))
    require_client_cert: bool = field(
        default_factory=lambda: _env_bool("FLOWER_TLS_REQUIRE_CLIENT_CERT", False)
    )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Aggregated configuration for the full FL pipeline."""

    fl: FLConfig = field(default_factory=FLConfig)
    dp: DPConfigFL = field(default_factory=DPConfigFL)
    model: ModelConfig = field(default_factory=ModelConfig)
    tls: TLSConfig = field(default_factory=TLSConfig)

    def summary(self) -> str:
        lines = [
            "=== FL Pipeline Configuration ===",
            f"  Rounds        : {self.fl.rounds}",
            f"  Min clients   : {self.fl.min_clients}",
            f"  Local epochs  : {self.fl.local_epochs}",
            f"  Learning rate : {self.fl.learning_rate}",
            f"  Batch size    : {self.fl.batch_size}",
            "--- Differential Privacy ---",
            f"  DP enabled    : {self.dp.enabled}",
            f"  ε (epsilon)   : {self.dp.target_epsilon} (limit < 1.0)",
            f"  δ (delta)     : {self.dp.target_delta}",
            f"  C (clip norm) : {self.dp.max_grad_norm}",
            f"  σ (noise)     : {'auto' if self.dp.noise_multiplier == 0.0 else self.dp.noise_multiplier}",
            "--- Model ---",
            f"  Input dim     : {self.model.input_dim}",
            f"  Latent dim    : {self.model.latent_dim}",
            f"  Hidden dims   : {self.model.hidden_dims}",
        ]
        return "\n".join(lines)


# Singleton-style default config (can be overridden at import time)
DEFAULT_CONFIG = PipelineConfig()
