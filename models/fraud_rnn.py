"""
fraud_rnn.py — Architecture du modèle de détection de fraude
═════════════════════════════════════════════════════════════
Variante A4 : DPGRU · 3 couches · seq_len=10 · dropout=0.3

Dropout strategy pour DPGRU 3 couches :
  - DPGRU Opacus ne supporte pas le dropout inter-couches natif
    (incompatible avec le calcul per-sample gradients)
  - Compensation : dropout post-GRU (0.3) + dropout dans classifier (0.3)
  - Dropout augmenté de 0.2 → 0.3 pour compenser la profondeur (3 couches)
"""

import torch
import torch.nn as nn

try:
    from opacus.layers import DPGRU
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

# ── Configuration A4 ─────────────────────────────────────────────────────────
USE_DPGRU    = True
INPUT_DIM    = 26
HIDDEN_DIM   = 128
NUM_LAYERS   = 3       # ← 2 → 3
DROPOUT_RATE = 0.3     # ← 0.2 → 0.3 (plus profond = plus de régularisation)


class FraudRNN(nn.Module):
    """
    Modèle GRU / DPGRU pour la détection de fraude.

    Dropout strategy (3 couches) :
      GRU couche 1 ──→ GRU couche 2 ──→ GRU couche 3
                                               ↓
                                    post_gru_dropout(0.3)   ← ajouté
                                               ↓
                               Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1)
    """

    def __init__(
        self,
        input_dim:    int   = INPUT_DIM,
        hidden_dim:   int   = HIDDEN_DIM,
        num_layers:   int   = NUM_LAYERS,
        dropout_rate: float = DROPOUT_RATE,
        use_dpgru:    bool  = USE_DPGRU,
    ):
        super().__init__()

        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.dropout_rate = dropout_rate
        self.use_dpgru    = use_dpgru and OPACUS_AVAILABLE

        # ── 1. Couche récurrente ─────────────────────────────────────────────
        if self.use_dpgru:
            # DPGRU Opacus — pas de dropout inter-couches (per-sample gradient)
            self.gru = DPGRU(
                input_size  = input_dim,
                hidden_size = hidden_dim,
                num_layers  = num_layers,
                batch_first = True,
            )
        else:
            # GRU standard — dropout inter-couches natif disponible
            self.gru = nn.GRU(
                input_size  = input_dim,
                hidden_size = hidden_dim,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = dropout_rate if num_layers > 1 else 0,
            )

        # ── 2. Dropout post-GRU ──────────────────────────────────────────────
        # Appliqué sur l'état caché final avant le classifier
        # Compense l'absence de dropout inter-couches dans DPGRU
        # Augmenté à 0.3 car 3 couches → plus de risque d'overfitting
        self.post_gru_dropout = nn.Dropout(dropout_rate)

        # ── 3. Classifieur MLP ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),   # dropout interne au classifier
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape : (batch_size, seq_len, input_dim)"""
        _, h_n = self.gru(x)

        # h_n : (num_layers, batch_size, hidden_dim)
        final_memory = h_n[-1, :, :]              # dernière couche
        final_memory = self.post_gru_dropout(final_memory)   # dropout post-GRU
        logits       = self.classifier(final_memory)
        return logits

    def info(self) -> str:
        gru_type     = "DPGRU (Opacus)" if self.use_dpgru else "GRU standard"
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"Modèle     : {gru_type}\n"
            f"Entrée     : {INPUT_DIM} features\n"
            f"Hidden     : {self.hidden_dim}  ×  {self.num_layers} couches\n"
            f"Dropout    : {self.dropout_rate} (post-GRU + classifier)\n"
            f"Paramètres : {total_params:,}"
        )


def build_model(use_dpgru: bool = USE_DPGRU) -> FraudRNN:
    return FraudRNN(
        input_dim    = INPUT_DIM,
        hidden_dim   = HIDDEN_DIM,
        num_layers   = NUM_LAYERS,
        dropout_rate = DROPOUT_RATE,
        use_dpgru    = use_dpgru,
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  Test FraudRNN — Variante A4")
    print("=" * 50)

    dummy_batch = torch.randn(32, 10, INPUT_DIM)   # seq_len=10

    for dpgru in [True, False]:
        if dpgru and not OPACUS_AVAILABLE:
            print("\n⚠️  Opacus non installé — test DPGRU ignoré")
            continue

        model = build_model(use_dpgru=dpgru)
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)

        print(f"\n{model.info()}")
        print(f"Entrée  : {dummy_batch.shape}  → (batch, seq=10, features)")
        print(f"Sortie  : {output.shape}        → (batch, 1) logits")
        print("-" * 50)

    print("\n✅ Test terminé")










# """
# fraud_rnn.py — Architecture du modèle de détection de fraude
# ═════════════════════════════════════════════════════════════
# Modèle : DPGRU (Opacus-compatible) ou GRU standard

# Configuration active : Run D2 (Approche B · 80/20 stratifié)
#   - DPGRU  : hidden=128, layers=2, dropout=0.2
#   - Entrée : 26 features (colonne 0 = PAN_ID retirée par le dataloader)
#   - Sortie : 1 logit (BCEWithLogitsLoss)

# Switcher entre GRU et DPGRU :
#   USE_DPGRU = True   → DPGRU (Opacus · FL + DP)
#   USE_DPGRU = False  → GRU standard (entraînement local uniquement)

# Historique des runs :
#   D2 : DPGRU · Approche B · 80/20  → AUC 0.9960 · FA calibrées 170  ← actif
#   D3 : DPGRU · Approche A · 70/10/20 → AUC 0.9935 · FA calibrées 86
#   B2 : GRU   · Approche B · 80/20  → AUC 0.9984 · (référence)
# """

# import torch
# import torch.nn as nn

# # ── Import conditionnel Opacus ────────────────────────────────────────────────
# try:
#     from opacus.layers import DPGRU
#     OPACUS_AVAILABLE = True
# except ImportError:
#     OPACUS_AVAILABLE = False


# # ============================================================================
# # ⚙️  CONFIGURATION DU MODÈLE
# #     Modifier ces valeurs pour tester différentes configurations
# # ============================================================================

# USE_DPGRU    = True    # True = DPGRU (FL + DP) | False = GRU standard

# INPUT_DIM    = 26      # Ne jamais modifier — fixé par le vectorizer (26 features)
# HIDDEN_DIM   = 128     # Neurones par couche GRU  — testé : 64, 128, 256
# NUM_LAYERS   = 2       # Couches GRU empilées     — testé : 1, 2
# DROPOUT_RATE = 0.2     # Dropout entre couches    — testé : 0.1, 0.2, 0.3


# # ============================================================================
# # 🧠 ARCHITECTURE
# # ============================================================================

# class FraudRNN(nn.Module):
#     """
#     Modèle GRU / DPGRU pour la détection de fraude.

#     Entrée attendue (depuis le dataloader)
#     ──────────────────────────────────────
#     x : (batch_size, seq_len, 26)
#         - batch_size : nombre de transactions dans le batch
#         - seq_len    : longueur de séquence (défaut 5)
#         - 26         : features métier (PAN_ID retiré par FraudSequenceDataset)

#     Sortie
#     ──────
#     logits : (batch_size, 1)
#         Logit brut — passer dans sigmoid() pour obtenir une probabilité.
#         La loss BCEWithLogitsLoss applique sigmoid() en interne.

#     Architecture interne
#     ────────────────────
#     GRU/DPGRU empilé (num_layers couches)
#         → état caché final de la dernière couche  (batch_size, hidden_dim)
#         → MLP classifieur : Linear(128→64) · ReLU · Dropout · Linear(64→1)
#     """

#     def __init__(
#         self,
#         input_dim:    int   = INPUT_DIM,
#         hidden_dim:   int   = HIDDEN_DIM,
#         num_layers:   int   = NUM_LAYERS,
#         dropout_rate: float = DROPOUT_RATE,
#         use_dpgru:    bool  = USE_DPGRU,
#     ):
#         super().__init__()

#         self.hidden_dim   = hidden_dim
#         self.num_layers   = num_layers
#         self.dropout_rate = dropout_rate
#         self.use_dpgru    = use_dpgru and OPACUS_AVAILABLE

#         # ── 1. Couche récurrente ─────────────────────────────────────────────
#         if self.use_dpgru:
#             # DPGRU : même interface que nn.GRU
#             # Implémentation non-fusionnée → compatible per-sample gradients Opacus
#             # Note : DPGRU ne supporte pas le paramètre dropout natif entre couches
#             #        → le dropout est géré par le classifieur MLP
#             self.gru = DPGRU(
#                 input_size  = input_dim,
#                 hidden_size = hidden_dim,
#                 num_layers  = num_layers,
#                 batch_first = True,
#             )
#         else:
#             # GRU standard PyTorch — plus rapide, entraînement local uniquement
#             self.gru = nn.GRU(
#                 input_size  = input_dim,
#                 hidden_size = hidden_dim,
#                 num_layers  = num_layers,
#                 batch_first = True,
#                 dropout     = dropout_rate if num_layers > 1 else 0,
#             )

#         # ── 2. Classifieur MLP ───────────────────────────────────────────────
#         # Transition douce 128 → 64 → 1
#         # Le Dropout ici compense l'absence de dropout interne au DPGRU
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(64, 1),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x shape : (batch_size, seq_len, input_dim)
#         """
#         _, h_n = self.gru(x)

#         # h_n shape : (num_layers, batch_size, hidden_dim)
#         # On extrait uniquement l'état caché de la DERNIÈRE couche
#         final_memory = h_n[-1, :, :]   # → (batch_size, hidden_dim)

#         logits = self.classifier(final_memory)   # → (batch_size, 1)
#         return logits

#     def info(self) -> str:
#         """Résumé lisible de la configuration."""
#         gru_type = "DPGRU (Opacus)" if self.use_dpgru else "GRU standard"
#         total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         return (
#             f"Modèle     : {gru_type}\n"
#             f"Entrée     : {INPUT_DIM} features\n"
#             f"Hidden     : {self.hidden_dim}  ×  {self.num_layers} couches\n"
#             f"Dropout    : {self.dropout_rate}\n"
#             f"Paramètres : {total_params:,}"
#         )


# # ============================================================================
# # 🏭 FACTORY — instanciation simple depuis train_local.py et client.py
# # ============================================================================

# def build_model(use_dpgru: bool = USE_DPGRU) -> FraudRNN:
#     """
#     Crée et retourne le modèle avec la configuration active.

#     Usage dans train_local.py :
#         from models.fraud_rnn import build_model
#         model = build_model().to(DEVICE)

#     Usage dans client.py (Flower) :
#         from models.fraud_rnn import build_model
#         model = build_model(use_dpgru=True)
#     """
#     return FraudRNN(
#         input_dim    = INPUT_DIM,
#         hidden_dim   = HIDDEN_DIM,
#         num_layers   = NUM_LAYERS,
#         dropout_rate = DROPOUT_RATE,
#         use_dpgru    = use_dpgru,
#     )


# # ============================================================================
# # 🧪 ZONE DE TEST  (python fraud_rnn.py)
# # ============================================================================

# if __name__ == "__main__":
#     print("=" * 50)
#     print("  Test du modèle FraudRNN")
#     print("=" * 50)

#     # Batch simulé : 32 clients, 5 transactions, 26 features
#     # Exactement ce que retourne FraudSequenceDataset.__getitem__
#     dummy_batch = torch.randn(32, 5, INPUT_DIM)

#     for dpgru in [True, False]:
#         if dpgru and not OPACUS_AVAILABLE:
#             print("\n⚠️  Opacus non installé — test DPGRU ignoré")
#             print("   pip install opacus")
#             continue

#         model = build_model(use_dpgru=dpgru)
#         model.eval()

#         with torch.no_grad():
#             output = model(dummy_batch)

#         print(f"\n{model.info()}")
#         print(f"Entrée  : {dummy_batch.shape}  → (batch, seq_len, features)")
#         print(f"Sortie  : {output.shape}        → (batch, 1) logits")
#         print(f"Probs   : {torch.sigmoid(output).squeeze()[:5].tolist()}")
#         print("-" * 50)

#     print("\n✅ Test terminé — modèle conforme au dataloader")