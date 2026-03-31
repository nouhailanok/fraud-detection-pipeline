"""
FraudNet — FraudDetection neural network for federated learning.

Architecture: 26 input features → 128-dim latent projection → hidden layers → binary output.
Designed to be compatible with Opacus DP training (no BatchNorm layers).
"""

import torch
import torch.nn as nn
from typing import Sequence


class FraudNet(nn.Module):
    """
    Binary fraud-detection classifier for federated learning.

    The first operation is always a linear projection from *input_dim*
    (default 26) to *latent_dim* (default 128) as required by the
    Privacy-Preserving MLOps specification.

    BatchNorm is intentionally omitted: Opacus (Differential Privacy)
    is incompatible with BatchNorm and requires GroupNorm or LayerNorm
    instead.  Since the hidden layers are small, no normalisation is
    used here — Dropout provides sufficient regularisation.
    """

    def __init__(
        self,
        input_dim: int = 26,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # --- Step 1: mandatory projection layer (26 → 128) ---
        self.projection = nn.Linear(input_dim, latent_dim)

        # --- Step 2: hidden layers with ReLU activation + Dropout ---
        hidden_layers: list[nn.Module] = [nn.ReLU(), nn.Dropout(dropout)]
        prev_dim = latent_dim
        for dim in hidden_dims:
            hidden_layers += [
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim
        self.hidden = nn.Sequential(*hidden_layers)

        # --- Step 3: binary output (raw logit — use BCEWithLogitsLoss) ---
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape (batch_size, input_dim).

        Returns:
            Raw logit tensor of shape (batch_size, 1).
        """
        x = self.projection(x)
        x = self.hidden(x)
        return self.output(x)


def get_model_parameters(model: nn.Module) -> list:
    """Extract model parameters as a list of NumPy arrays (for Flower)."""
    return [p.cpu().detach().numpy() for p in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: list) -> None:
    """Load a list of NumPy arrays back into the model (from Flower)."""
    import numpy as np

    with torch.no_grad():
        for param, array in zip(model.parameters(), parameters):
            param.copy_(torch.from_numpy(np.array(array, dtype=np.float32)))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dummy = torch.randn(32, 26)
    model = FraudNet()
    out = model(dummy)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Total params : {total_params:,}")
