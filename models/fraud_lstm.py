import torch
import torch.nn as nn

# Import conditionnel Opacus
try:
    from opacus.layers import DPLSTM
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

INPUT_DIM    = 26
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT_RATE = 0.2
USE_DPLSTM   = True


class FraudLSTM(nn.Module):
    """
    Modèle LSTM / DPLSTM pour la détection de fraude.

    Entrée : (batch_size, seq_len, 26)
    Sortie : (batch_size, 1) logit
    """
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout_rate: float = DROPOUT_RATE,
        use_dplstm: bool = USE_DPLSTM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_dplstm = use_dplstm and OPACUS_AVAILABLE

        if self.use_dplstm:
            self.lstm = DPLSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1, :, :]
        logits = self.classifier(final_hidden)
        return logits

    def info(self) -> str:
        model_type = "DPLSTM (Opacus)" if self.use_dplstm else "LSTM standard"
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"Modèle     : {model_type}\n"
            f"Entrée     : {INPUT_DIM} features\n"
            f"Hidden     : {self.hidden_dim} × {self.num_layers} couches\n"
            f"Dropout    : {self.dropout_rate}\n"
            f"Paramètres : {total_params:,}"
        )


def build_model(use_dplstm: bool = USE_DPLSTM) -> FraudLSTM:
    return FraudLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        use_dplstm=use_dplstm,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Test du modèle FraudLSTM")
    print("=" * 60)

    dummy_batch = torch.randn(32, 5, INPUT_DIM)

    for dplstm in [True, False]:
        if dplstm and not OPACUS_AVAILABLE:
            print("\n⚠️  Opacus non installé — test DPLSTM ignoré")
            print("   pip install opacus")
            continue

        model = build_model(use_dplstm=dplstm)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        print(f"\n{model.info()}")
        print(f"Entrée  : {dummy_batch.shape}  → (batch, seq_len, features)")
        print(f"Sortie  : {output.shape}        → (batch, 1)")
        print(f"Probs   : {torch.sigmoid(output).squeeze()[:5].tolist()}")
        print("-" * 60)

    print("\n✅ Test terminé — modèle LSTM créé avec succès")
