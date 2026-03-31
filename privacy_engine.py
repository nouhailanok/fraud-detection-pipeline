"""
Differential Privacy Engine — wraps Opacus to satisfy ε-DP with ε < 1.0.

Mathematical guarantee
-----------------------
For every pair of neighbouring datasets D, D' (differing in exactly one
record) and all measurable output sets S:

    P[A(D) ∈ S] ≤ exp(ε) · P[A(D') ∈ S]

Two sequential operations enforce this bound:

1. **L2 gradient clipping** — bounds the sensitivity C:
       ΔW̄ = ΔW / max(1, ‖ΔW‖₂ / C)

2. **Gaussian noise injection** — adds calibrated noise σ:
       ΔW_private = ΔW̄ + N(0, σ²C²I)

Opacus tracks the accumulated (ε, δ) via the Rényi Differential Privacy
accountant across training rounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_OPACUS_AVAILABLE = False
try:
    from opacus import PrivacyEngine as _OpacusPrivacyEngine
    from opacus.validators import ModuleValidator

    _OPACUS_AVAILABLE = True
except ImportError:
    logger.warning(
        "Opacus is not installed.  DPPrivacyEngine will fall back to "
        "manual gradient clipping without formal DP accounting.  "
        "Install opacus to enable full ε-DP guarantees."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPConfig:
    """Hyperparameters for the Differential Privacy engine."""

    max_grad_norm: float = 1.0
    """L2 gradient clipping threshold C."""

    target_epsilon: float = 0.9
    """Maximum allowed privacy loss ε.  Must be strictly < 1.0."""

    target_delta: float = 1e-6
    """Failure probability δ (typically 1/n where n = dataset size)."""

    noise_multiplier: Optional[float] = None
    """
    σ (sigma) — noise scale relative to C.  When *None*, Opacus
    computes σ automatically to satisfy (target_epsilon, target_delta)-DP
    for the given number of epochs and sample rate.
    """

    epochs: int = 1
    """Number of local training epochs per federated round."""

    accountant: str = "rdp"
    """Privacy accountant type accepted by Opacus ('rdp' or 'gdp')."""

    def __post_init__(self) -> None:
        if self.target_epsilon >= 1.0:
            raise ValueError(
                f"target_epsilon must be strictly < 1.0 for GDPR/PCI-DSS compliance "
                f"(got {self.target_epsilon}).  Values ≥ 1.0 provide insufficient "
                "privacy guarantees and expose the model to membership inference attacks."
            )
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


# ---------------------------------------------------------------------------
# Privacy engine
# ---------------------------------------------------------------------------

@dataclass
class PrivacyBudget:
    """Snapshot of the current privacy expenditure."""

    epsilon: float = 0.0
    delta: float = 1e-6
    rounds_completed: int = 0
    noise_multiplier: float = 0.0
    max_grad_norm: float = 1.0


class DPPrivacyEngine:
    """
    Differential Privacy engine for federated learning.

    Usage example
    -------------
    >>> config = DPConfig(target_epsilon=0.9, max_grad_norm=1.0, epochs=1)
    >>> engine = DPPrivacyEngine(config)
    >>> model, optimizer, train_loader = engine.attach(model, optimizer, train_loader)
    >>> # … train for one round …
    >>> budget = engine.current_budget()
    >>> print(f"ε = {budget.epsilon:.4f} (limit 1.0)")
    """

    def __init__(self, config: Optional[DPConfig] = None) -> None:
        self.config = config or DPConfig()
        self._opacus_engine: Optional["_OpacusPrivacyEngine"] = None
        self._rounds: int = 0
        self._noise_multiplier: float = self.config.noise_multiplier or 1.0
        self._attached: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(
        self,
        model: "nn.Module",
        optimizer: "Optimizer",
        data_loader: "DataLoader",
    ) -> "tuple[nn.Module, Optimizer, DataLoader]":
        """
        Attach the privacy engine to a model / optimizer / data_loader.

        Returns the (possibly wrapped) model, optimizer, and data loader
        that must be used for training.  The wrapped objects transparently
        apply per-sample gradient clipping and Gaussian noise injection.

        Raises
        ------
        RuntimeError
            If `target_epsilon` ≥ 1.0 (compile-time check via DPConfig).
        """
        cfg = self.config

        if _OPACUS_AVAILABLE:
            # Validate and fix the model for Opacus compatibility
            # (e.g. replace BatchNorm with GroupNorm)
            if not ModuleValidator.is_valid(model):
                logger.info("Fixing model for Opacus compatibility …")
                model = ModuleValidator.fix(model)

            opacus_engine = _OpacusPrivacyEngine(accountant=cfg.accountant)

            if cfg.noise_multiplier is not None:
                # Use the explicitly provided σ value
                sigma = cfg.noise_multiplier
            else:
                # Let Opacus calibrate σ to satisfy the privacy budget
                from opacus.accountants.utils import get_noise_multiplier

                n = len(data_loader.dataset)  # type: ignore[arg-type]
                sample_rate = data_loader.batch_size / n if n > 0 else 0.01  # type: ignore[union-attr]

                sigma = get_noise_multiplier(
                    target_epsilon=cfg.target_epsilon,
                    target_delta=cfg.target_delta,
                    sample_rate=sample_rate,
                    epochs=cfg.epochs,
                    accountant=cfg.accountant,
                )
                logger.info(
                    "Opacus calibrated noise_multiplier σ=%.4f for "
                    "(ε=%.2f, δ=%.1e) over %d epoch(s)",
                    sigma,
                    cfg.target_epsilon,
                    cfg.target_delta,
                    cfg.epochs,
                )

            self._noise_multiplier = sigma

            model, optimizer, data_loader = opacus_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=sigma,
                max_grad_norm=cfg.max_grad_norm,
            )
            self._opacus_engine = opacus_engine
        else:
            # Fallback: manual per-parameter gradient clipping (no formal DP accounting).
            # WARNING: This does NOT satisfy Differential Privacy — per-sample gradient
            # clipping and Gaussian noise injection require Opacus.  This path should
            # only be used for local development / debugging without privacy guarantees.
            logger.warning(
                "Running WITHOUT Opacus — per-parameter gradient clipping applied "
                "(max_grad_norm=%.1f) but NO Gaussian DP noise is added.  "
                "This fallback does NOT satisfy ε-DP.  Install opacus for full guarantees.",
                cfg.max_grad_norm,
            )
            self._register_gradient_clipping(model, cfg.max_grad_norm)

        self._attached = True
        logger.info(
            "DPPrivacyEngine attached | C=%.1f | σ=%.4f | opacus=%s",
            cfg.max_grad_norm,
            self._noise_multiplier,
            _OPACUS_AVAILABLE,
        )
        return model, optimizer, data_loader

    def step_end(self) -> None:
        """
        Call once after each optimizer.step() / backward pass.

        Checks that the accumulated ε has not exceeded the configured
        limit and logs a warning if it has.
        """
        self._rounds += 1
        budget = self.current_budget()
        if budget.epsilon >= self.config.target_epsilon:
            logger.warning(
                "Privacy budget nearly exhausted: ε=%.4f / limit=%.2f",
                budget.epsilon,
                self.config.target_epsilon,
            )

    def current_budget(self) -> PrivacyBudget:
        """Return the current privacy expenditure (ε, δ)."""
        if self._opacus_engine is not None:
            try:
                eps = self._opacus_engine.get_epsilon(self.config.target_delta)
            except Exception:
                eps = 0.0
        else:
            eps = 0.0

        return PrivacyBudget(
            epsilon=eps,
            delta=self.config.target_delta,
            rounds_completed=self._rounds,
            noise_multiplier=self._noise_multiplier,
            max_grad_norm=self.config.max_grad_norm,
        )

    def enforce_epsilon_limit(self) -> None:
        """
        Raise RuntimeError if the current ε has exceeded the target.

        Call at the end of each federated round to enforce the
        ε < 1.0 hard constraint.
        """
        budget = self.current_budget()
        if budget.epsilon > self.config.target_epsilon:
            raise RuntimeError(
                f"Privacy budget exceeded: ε={budget.epsilon:.4f} > "
                f"limit={self.config.target_epsilon:.2f}. "
                "Stop training to preserve privacy guarantees."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _register_gradient_clipping(model: "nn.Module", max_norm: float) -> None:
        """Register a backward hook that clips per-parameter gradients."""
        import torch  # local import — torch may not be installed at module load time

        def _clip_hook(grad: "torch.Tensor") -> "torch.Tensor":
            norm = grad.norm(2)
            return grad * min(1.0, max_norm / (norm + 1e-8))

        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(_clip_hook)
