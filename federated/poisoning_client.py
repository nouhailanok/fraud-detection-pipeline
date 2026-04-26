"""
poisoning_client.py — Client FL malveillant pour tester le Behavioral Analysis
═══════════════════════════════════════════════════════════════════════════════
Simule un nœud FL qui effectue une attaque configurable sur les gradients.
La manipulation est appliquée sur les paramètres APRÈS l'entraînement normal,
garantissant que les features détectées par BehavioralAnalyzer correspondent
exactement aux seuils de classification.

TYPES D'ATTAQUE (configurer ATTACK_TYPE ci-dessous) :
  "FREE_RIDER"  → envoie des poids quasi-nuls   (norm_L2 < 1e-3)
  "SIGN_FLIP"   → inverse tous les delta W       (cos_sim < -0.5)
  "SCALE"       → amplifie les delta W × SCALE_FACTOR (norm_L2 >> moyenne × 10)
  "NOISE"       → remplace delta W par du bruit  (var_delta >> moyenne × 5)
  "BYZANTINE"   → combinaison SIGN_FLIP + SCALE  (score_IF < -0.3)
  "NORMAL"      → comportement honnête (test de référence)

ACTIVATION :
  ATTACK_START_ROUND = round à partir duquel l'attaque commence
  → permet de simuler un nœud qui se comporte honnêtement au début

Usage :
  Remplacer un des scripts client_X_run.ps1 par ce fichier
  $env:ATTACK_TYPE = "SIGN_FLIP"
  $env:ATTACK_START_ROUND = "3"
  $env:CLIENT_ID = "2"   ← nœud qui sera attaqué
  & python federated/poisoning_client.py
"""

import os
import time
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import flwr as fl
import torch
import torch.nn as nn

# ── Configuration de l'attaque ────────────────────────────────────────────────
ATTACK_TYPE        = os.getenv("ATTACK_TYPE",        "SIGN_FLIP")  # type d'attaque
ATTACK_START_ROUND = int(os.getenv("ATTACK_START_ROUND", "3"))     # round de début
SCALE_FACTOR       = float(os.getenv("SCALE_FACTOR",    "15.0"))   # ×15 pour SCALE
NOISE_STD          = float(os.getenv("NOISE_STD",       "5.0"))    # σ pour NOISE

# ── Config FL (identique à client.py) ────────────────────────────────────────
CLIENT_ID      = os.getenv("CLIENT_ID", "2")
SERVER_HOST    = os.getenv("FLOWER_SERVER_HOST", "127.0.0.1")
SERVER_PORT    = os.getenv("FLOWER_SERVER_PORT", "8080")
DATA_ROOT      = Path(os.getenv("FL_DATA_ROOT",  "data"))
SEQ_LEN        = int(os.getenv("FL_SEQ_LEN",     "5"))
BATCH_SIZE     = int(os.getenv("FL_BATCH_SIZE",  "64"))
LEARNING_RATE  = float(os.getenv("FL_LR",        "0.0005"))
POS_WEIGHT     = float(os.getenv("FL_POS_WEIGHT","167.0"))
DP_NOISE       = float(os.getenv("DP_NOISE",     "1.0"))
EPSILON_TARGET = float(os.getenv("DP_EPSILON_TARGET", "1.0"))
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ── Imports projet ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fraud_rnn import build_model
from data.dataloader  import get_split_dataloaders


# ============================================================================
# 🔧 Utilitaires paramètres
# ============================================================================

def get_model_parameters(model) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_parameters(model, parameters: List[np.ndarray]) -> None:
    for p, new_p in zip(model.parameters(), parameters):
        p.data = torch.tensor(new_p, dtype=torch.float32).to(DEVICE)


# ============================================================================
# ☠️  Manipulation des paramètres selon le type d'attaque
# ============================================================================

def apply_attack(
    original_params : List[np.ndarray],  # poids globaux reçus du serveur
    trained_params  : List[np.ndarray],  # poids après entraînement local
    attack_type     : str,
    server_round    : int,
) -> List[np.ndarray]:
    """
    Applique l'attaque sur les delta W = trained - original.
    Retourne les paramètres manipulés à envoyer au serveur.

    Chaque attaque est calibrée pour dépasser les seuils de classify_attack() :
      FREE_RIDER  → norm_L2 < 1e-3
      SIGN_FLIP   → cos_sim < -0.5  (inverse le delta)
      SCALE       → norm_L2 > moy × SCALE_FACTOR (≥10)
      NOISE       → var_delta > moy × 5
      BYZANTINE   → SIGN_FLIP + SCALE combinés
    """
    if attack_type == "NORMAL" or server_round < ATTACK_START_ROUND:
        return trained_params

    # Calculer les deltas (mise à jour locale)
    deltas = [t - o for t, o in zip(trained_params, original_params)]

    print(f"\n  ☠️  ATTAQUE [{attack_type}] — Round {server_round}")

    # ── FREE_RIDER — envoie les poids originaux (delta ≈ 0) ──────────────────
    if attack_type == "FREE_RIDER":
        print(f"     → Envoi des poids quasi-nuls (norm_L2 ≈ 0)")
        # Ajoute un bruit infinitésimal pour éviter exactement 0
        return [o + np.random.normal(0, 1e-6, o.shape).astype(np.float32)
                for o in original_params]

    # ── SIGN_FLIP — inverse la direction des deltas ───────────────────────────
    elif attack_type == "SIGN_FLIP":
        print(f"     → Inversion forte des deltas × -15 (force cos_sim < -0.5)")
        # Facteur -15 : nécessaire pour contrebalancer 3 nœuds honnêtes
        # dans le calcul de cos_sim vs moyenne des pairs
        # cos_sim = dot(delta_i, mean_delta) / (||delta_i|| * ||mean_delta||)
        # mean_delta = (3×honnête + 1×inversé×15) / 4
        # → mean_delta fortement tiré vers le négatif
        # → cos_sim de node_4 très négatif
        manipulated = [o + (-15.0 * d) for o, d in zip(original_params, deltas)]
        return manipulated

    # ── SCALE — amplifie les deltas × SCALE_FACTOR ───────────────────────────
    elif attack_type == "SCALE":
        print(f"     → Amplification des deltas × {SCALE_FACTOR} (norm_L2 >> moyenne)")
        manipulated = [o + (SCALE_FACTOR * d) for o, d in zip(original_params, deltas)]
        return manipulated

    # ── NOISE — remplace les deltas par du bruit gaussien ────────────────────
    elif attack_type == "NOISE":
        print(f"     → Injection de bruit gaussien (σ={NOISE_STD}, var_delta >> normale)")
        # Calculer la norme moyenne pour calibrer le bruit
        delta_flat  = np.concatenate([d.flatten() for d in deltas])
        delta_norm  = float(np.linalg.norm(delta_flat))
        noise_scale = delta_norm * NOISE_STD / max(len(delta_flat) ** 0.5, 1)

        manipulated = [
            o + np.random.normal(0, noise_scale, o.shape).astype(np.float32)
            for o in original_params
        ]
        return manipulated

    # ── BYZANTINE — SIGN_FLIP + SCALE (combinaison) ───────────────────────────
    elif attack_type == "BYZANTINE":
        print(f"     → Attaque combinée SIGN_FLIP + SCALE × {SCALE_FACTOR/2}")
        manipulated = [
            o + (-SCALE_FACTOR / 2.0 * d)
            for o, d in zip(original_params, deltas)
        ]
        return manipulated

    return trained_params


# ============================================================================
# 🌸 FLOWER CLIENT MALVEILLANT
# ============================================================================

class PoisoningClient(fl.client.NumPyClient):

    def __init__(self, model, train_loader, val_loader, client_id: str = "2"):
        self.model        = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.client_id    = client_id
        self.current_round= 0

        pos_weight     = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Mémoriser les poids globaux reçus pour calculer les deltas
        self._original_params: Optional[List[np.ndarray]] = None

        print(f"\n{'═'*55}")
        print(f"  ☠️  CLIENT MALVEILLANT — Nœud {client_id}")
        print(f"  Attaque   : {ATTACK_TYPE}")
        print(f"  Début     : round {ATTACK_START_ROUND}")
        print(f"  Device    : {DEVICE}")
        print(f"{'═'*55}\n")

    def get_parameters(self, config) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self._original_params = [p.copy() for p in parameters]
        set_model_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.current_round = int(config.get("server_round", self.current_round + 1))
        local_epochs = int(config.get("local_epochs", 3))

        # ── Entraînement normal (identique à client.py honnête) ──────────────
        self.model.train()
        avg_loss = 0.0

        for epoch in range(local_epochs):
            total_loss, n_batches = 0.0, 0
            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()
                logits = self.model(x).float()
                loss   = self.criterion(logits, y.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches  += 1
            avg_loss = total_loss / max(n_batches, 1)
            print(f"   [Epoch {epoch+1}/{local_epochs}] Loss: {avg_loss:.4f}")

        # ── Appliquer l'attaque sur les paramètres entraînés ─────────────────
        trained_params    = get_model_parameters(self.model)
        manipulated_params = apply_attack(
            self._original_params,
            trained_params,
            ATTACK_TYPE,
            self.current_round,
        )

        # Charger les paramètres manipulés dans le modèle
        set_model_parameters(self.model, manipulated_params)

        # Vérification diagnostique
        if ATTACK_TYPE != "NORMAL" and self.current_round >= ATTACK_START_ROUND:
            delta_flat = np.concatenate([
                (m - o).flatten()
                for m, o in zip(manipulated_params, self._original_params)
            ])
            norm_L2  = float(np.linalg.norm(delta_flat, ord=2))
            var_d    = float(np.var(delta_flat))
            print(f"     norm_L2={norm_L2:.4f}  var_delta={var_d:.4f}")

        # Normaliser l'ID pour correspondre au format des autres nœuds
        node_id = f"node_{self.client_id}" if not self.client_id.startswith("node_") else self.client_id

        return (
            manipulated_params,
            len(self.train_loader.dataset),
            {
                "train_loss": avg_loss,
                "client_id" : node_id,
                "epsilon"   : 0.0,
                "attack"    : ATTACK_TYPE,
            },
        )

    def evaluate(self, parameters, config):
        """Évaluation honnête — pour ne pas fausser les métriques globales."""
        set_model_parameters(self.model, parameters)
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        tp = fp = fn = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y   = x.to(DEVICE), y.to(DEVICE)
                logits = self.model(x).float()
                loss   = self.criterion(logits, y.float())
                total_loss += loss.item()
                preds  = (torch.sigmoid(logits) >= 0.5).squeeze().long()
                y_int  = y.long()
                correct += (preds == y_int).sum().item()
                total   += y_int.size(0)
                tp += ((preds == 1) & (y_int == 1)).sum().item()
                fp += ((preds == 1) & (y_int == 0)).sum().item()
                fn += ((preds == 0) & (y_int == 1)).sum().item()

        accuracy  = correct / max(total, 1)
        recall    = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return float(total_loss), len(self.val_loader.dataset), {
            "accuracy" : accuracy,
            "recall"   : recall,
            "precision": precision,
            "f1"       : f1,
            "client_id": self.client_id,
        }


# ============================================================================
# 🚀 MAIN
# ============================================================================

def main():
    print(f"\n{'═'*55}")
    print(f"  ☠️  POISONING CLIENT — {ATTACK_TYPE}")
    print(f"  Nœud {CLIENT_ID} | {SERVER_HOST}:{SERVER_PORT}")
    print(f"  Attaque active à partir du round {ATTACK_START_ROUND}")
    print(f"{'═'*55}\n")

    # Charger les données du nœud
    node_dir = DATA_ROOT / f"node_{CLIENT_ID}" / "tensors"
    print(f"📂 Chargement des tensors : {node_dir}")

    loaders = get_split_dataloaders(
        data_dir   = str(node_dir),
        batch_size = BATCH_SIZE,
        seq_len    = SEQ_LEN,
    )
    train_loader = loaders[0]
    val_loader   = loaders[1] if len(loaders) > 1 else loaders[0]

    # Construire le modèle
    model = build_model(use_dpgru=True)

    # Client malveillant
    client = PoisoningClient(model, train_loader, val_loader, CLIENT_ID)

    # Connexion Flower
    server_address = f"{SERVER_HOST}:{SERVER_PORT}"
    print(f"🔗 Connexion à {server_address}...")

    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
        )
    except Exception as e:
        print(f"❌ Erreur connexion : {e}")

    print(f"\n  ☠️  Session malveillante terminée — Attaque : {ATTACK_TYPE}")


if __name__ == "__main__":
    main()