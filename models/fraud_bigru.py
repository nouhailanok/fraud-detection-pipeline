"""
fraud_bigru.py — Architecture BiGRU pour la détection de fraude
═════════════════════════════════════════════════════════════════
Modèle : BiGRU (Gated Recurrent Unit bidirectionnel) ou DPGRU (DP)

Configuration active :
  - USE_DPGRU = False → BiGRU standard (bidirectionnel, entraînement local)
  - USE_DPGRU = True  → DPGRU (Opacus-compatible, FL + DP)

Note : 
  - BiGRU classique traite la séquence dans DEUX directions (avant + arrière)
  - DPGRU est uni-directionnel mais compatible Opacus pour le DP
  - Basculer USE_DPGRU selon le mode (local vs federated)

Historique comparatif :
  GRU standard   : AUC 0.9984 (B2, référence)
  DPGRU (FL)     : AUC 0.9960 (D2, DP activé)
  BiGRU (local)  : À tester
  BiGRU + DPGRU  : À tester avec DP
"""

import torch
import torch.nn as nn

# Import conditionnel Opacus
try:
    from opacus.layers import DPGRU
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

INPUT_DIM    = 26
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT_RATE = 0.2
USE_DPGRU    = False  # False = BiGRU bidirectionnel | True = DPGRU (DP)


class FraudBiGRU(nn.Module):
    """
    Modèle BiGRU ou DPGRU pour la détection de fraude.
    
    Mode BiGRU (USE_DPGRU=False) :
    - GRU bidirectionnel (avant + arrière)
    - Capture patterns temporels dans les deux sens
    - État final concaténé : (batch, hidden_dim × 2)
    
    Mode DPGRU (USE_DPGRU=True) :
    - DPGRU uni-directionnel (compatible Opacus)
    - État final : (batch, hidden_dim)
    - Activable pour Federated Learning + DP

    Entrée : (batch_size, seq_len, 26)
    Sortie : (batch_size, 1) logit
    """
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout_rate: float = DROPOUT_RATE,
        use_dpgru: bool = USE_DPGRU,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_dpgru = use_dpgru and OPACUS_AVAILABLE

        # Couche RNN
        if self.use_dpgru:
            # DPGRU : compatible Opacus pour FL + DP (uni-directionnel)
            self.gru = DPGRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            final_dim = hidden_dim  # DPGRU uni-directionnel
        else:
            # BiGRU : bidirectionnel pour entraînement local (meilleure performance)
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
                bidirectional=True,  # Double la dimension de sortie
            )
            final_dim = hidden_dim * 2  # Bidirectional double la dimension

        # Classifieur MLP
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape : (batch_size, seq_len, input_dim)
        
        Mode BiGRU :
        1. Forward direction  : x[0] → x[1] → ... → x[4]
        2. Backward direction : x[4] → x[3] → ... → x[0]
        3. États finaux concaténés : (batch, hidden_dim*2)
        
        Mode DPGRU :
        1. Direction unique : x[0] → x[1] → ... → x[4]
        2. État final : (batch, hidden_dim)
        """
        _, h_n = self.gru(x)
        
        if self.use_dpgru:
            # DPGRU : h_n shape (num_layers, batch, hidden_dim)
            # Prendre l'état final de la dernière couche
            final_hidden = h_n[-1, :, :]
        else:
            # BiGRU : h_n shape (num_layers*2, batch, hidden_dim)
            # Concaténer les états finaux forward + backward
            forward_final = h_n[-2, :, :]     # Avant-dernière (forward)
            backward_final = h_n[-1, :, :]    # Dernière (backward)
            final_hidden = torch.cat([forward_final, backward_final], dim=1)
        
        logits = self.classifier(final_hidden)
        return logits

    def info(self) -> str:
        gru_type = "DPGRU (Opacus)" if self.use_dpgru else "BiGRU (Bidirectionnel)"
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"Modèle     : {gru_type}\n"
            f"Entrée     : {INPUT_DIM} features\n"
            f"Hidden     : {self.hidden_dim} × {self.num_layers} couches\n"
            f"Dropout    : {self.dropout_rate}\n"
            f"Paramètres : {total_params:,}"
        )


def build_model(use_dpgru: bool = USE_DPGRU) -> FraudBiGRU:
    """
    Crée et retourne le modèle BiGRU ou DPGRU.

    Args:
        use_dpgru : Si True et Opacus dispo → DPGRU (DP)
                    Si False → BiGRU bidirectionnel (local)

    Usage dans train_local.py :
        from models.fraud_bigru import build_model
        model = build_model(use_dpgru=False).to(DEVICE)  # BiGRU local
        
    Usage dans client.py (Flower) :
        from models.fraud_bigru import build_model
        model = build_model(use_dpgru=True)  # DPGRU pour FL+DP
    """
    return FraudBiGRU(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        use_dpgru=use_dpgru,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Test du modèle FraudBiGRU")
    print("=" * 70)

    dummy_batch = torch.randn(32, 5, INPUT_DIM)

    for dpgru in [True, False]:
        if dpgru and not OPACUS_AVAILABLE:
            print("\n⚠️  Opacus non installé — test DPGRU ignoré")
            print("   pip install opacus")
            continue

        model = build_model(use_dpgru=dpgru)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        print(f"\n{model.info()}")
        print(f"Entrée  : {dummy_batch.shape}  → (batch, seq_len, features)")
        print(f"Sortie  : {output.shape}        → (batch, 1)")
        print(f"Logits  : {output[:5].squeeze().tolist()}")
        print(f"Probs   : {torch.sigmoid(output[:5]).squeeze().tolist()}")
        print("-" * 70)

    print("\n✅ Test terminé — modèle BiGRU créé avec succès")
    print("\n📌 Recommandations :")
    print("   - use_dpgru=False : BiGRU bidirectionnel → meilleur pour entraînement local")
    print("   - use_dpgru=True  : DPGRU (Opacus) → requis pour Federated Learning + DP")

