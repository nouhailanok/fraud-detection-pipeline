"""
client.py — Nœud Flower pour le Federated Learning
════════════════════════════════════════════════════
Changements par rapport à la version originale :

  1. Import build_model() depuis fraud_rnn.py
       → Remplace FraudRNN(...) hardcodé
       → Utilise automatiquement DPGRU si USE_DPGRU=True dans fraud_rnn.py
       → Un seul endroit pour changer l'architecture

  2. ModuleValidator.fix() supprimé
       → DPGRU est nativement compatible Opacus — pas besoin de fix
       → GRU standard reste incompatible, mais on n'utilise plus GRU en FL

  3. pos_weight = 167.0 ajouté dans BCEWithLogitsLoss
       → Aligné avec train_local.py (ratio réel : 362 974 légit / 2 174 fraudes)
       → Sans ça, le modèle FL ignorait le déséquilibre des classes

  4. learning_rate : 0.001 → 0.0005
       → Aligné avec le meilleur run (D2/D3)
       → Configurable via variable d'environnement LR

  5. batch_size : 64 → 256
       → Aligné avec train_local.py pour une meilleure utilisation GPU
       → Configurable via variable d'environnement BATCH_SIZE

  6. gradient clipping ajouté dans fit()
       → clip_grad_norm_(max_norm=1.0)
       → Même protection que train_local.py contre les explosions de gradient

  7. logits.float() dans evaluate()
       → Même fix FP32 que train_local.py
       → Évite les NaN avec pos_weight=167 en FP16

  8. get_split_dataloaders → retourne 2 loaders (train + test)
       → val_loader = test_loader (users 100% inconnus = pas de biais en FL)
       → batch_size et seq_len alignés avec train_local.py
"""

import os
import time
import inspect
from pathlib import Path
from typing import Any, Optional, List

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from opacus import PrivacyEngine

# ── Imports du projet ────────────────────────────────────────────────────────
# CHANGEMENT 1 : build_model() au lieu de FraudRNN() hardcodé
from models.fraud_rnn import build_model
from data.dataloader import get_split_dataloaders


# ============================================================================
# ⚙️  CONFIGURATION
#     Les valeurs par défaut sont alignées avec les meilleurs runs (D2/D3).
#     Toutes sont surchargeables via variables d'environnement Docker.
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHANGEMENT 4 + 5 : LR et BATCH_SIZE alignés avec train_local.py
LEARNING_RATE = float(os.getenv("FL_LR",         "0.0005"))   # ← 0.001 → 0.0005
BATCH_SIZE    = int(os.getenv("FL_BATCH_SIZE",   "256"))      # ← 64 → 256
SEQ_LEN       = int(os.getenv("FL_SEQ_LEN",      "5"))
TRAIN_RATIO   = float(os.getenv("FL_TRAIN_RATIO","0.8"))

# CHANGEMENT 3 : pos_weight aligné avec train_local.py
POS_WEIGHT    = float(os.getenv("FL_POS_WEIGHT", "167.0"))    # ← manquait


# ============================================================================
# 🔧 UTILS
# ============================================================================

def get_model_parameters(model) -> List:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_parameters(model, parameters: List):
    for p, new_p in zip(model.parameters(), parameters):
        p.data = torch.tensor(new_p, dtype=torch.float32).to(DEVICE)


# ============================================================================
# 🌸 FLOWER CLIENT
# ============================================================================

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model        = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # CHANGEMENT 3 : pos_weight ajouté — aligné avec train_local.py
        pos_weight       = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
        self.criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # CHANGEMENT 2 : ModuleValidator.fix() supprimé
        # DPGRU est nativement compatible Opacus — pas besoin de réparer le modèle
        # self.model = ModuleValidator.fix(self.model)  ← supprimé

        # CHANGEMENT 4 : LR aligné avec train_local.py
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # 🔐 Differential Privacy
        self.privacy_engine = PrivacyEngine()

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module          = self.model,
            optimizer       = self.optimizer,
            data_loader     = self.train_loader,
            noise_multiplier= float(os.getenv("DP_NOISE", "1.5")),
            max_grad_norm   = 1.0,
        )

    # ── Paramètres FL ────────────────────────────────────────────────────────

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        set_model_parameters(self.model, parameters)

    # ── Entraînement local (avec DP) ─────────────────────────────────────────

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        local_epochs = int(config.get("local_epochs", 1))

        for epoch in range(local_epochs):
            total_loss = 0.0
            n_batches  = 0

            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                self.optimizer.zero_grad()

                # Forward en FP32 (même approche que train_local.py)
                logits      = self.model(x).float()
                loss        = self.criterion(logits, y.float())

                loss.backward()

                # CHANGEMENT 6 : gradient clipping — aligné avec train_local.py
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"   [FL-Epoch {epoch+1}/{local_epochs}] Loss: {avg_loss:.4f}")

        # 🔐 Calcul du budget ε
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        print(f"🔐 ε = {epsilon:.4f}  (target ≤ 1.0)")

        if epsilon > 1.0:
            print("⚠️  ε > 1.0 → privacy faible → augmenter DP_NOISE")

        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"epsilon": float(epsilon), "train_loss": avg_loss},
        )

    # ── Évaluation locale ─────────────────────────────────────────────────────

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                # CHANGEMENT 7 : .float() pour éviter NaN avec pos_weight FP16
                logits = self.model(x).float()
                loss   = self.criterion(logits, y.float())

                probs  = torch.sigmoid(logits)
                preds  = (probs > 0.5).float()

                correct    += (preds == y).sum().item()
                total      += y.size(0)
                total_loss += loss.item() * y.size(0)

        accuracy = correct / total    if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return avg_loss, total, {"accuracy": accuracy}


# ============================================================================
# 🔐 TLS
# ============================================================================

def _read_bytes(path_str: Optional[str]) -> Optional[bytes]:
    if not path_str:
        return None
    file_path = Path(path_str)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


def start_client_compatible(client: fl.client.NumPyClient, server_address: str) -> None:
    root_cert   = _read_bytes(os.getenv("FLOWER_CA_CERT"))
    client_cert = _read_bytes(os.getenv("FLOWER_CLIENT_CERT"))
    client_key  = _read_bytes(os.getenv("FLOWER_CLIENT_KEY"))

    start_client_fn = getattr(fl.client, "start_client", None)
    if start_client_fn is not None:
        kwargs: dict[str, Any] = {
            "server_address": server_address,
            "client": client.to_client(),
        }
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


# ============================================================================
# 🚀 MAIN
# ============================================================================

def main() -> None:
    client_id      = os.getenv("CLIENT_ID", "1")
    host           = os.getenv("FLOWER_SERVER_HOST", "flower")
    port           = int(os.getenv("FLOWER_SERVER_PORT", "8080"))
    server_address = f"{host}:{port}"
    retry_seconds  = int(os.getenv("FL_CLIENT_RETRY_SECONDS", "10"))
    continuous     = os.getenv("FL_CLIENT_CONTINUOUS", "true").lower() == "true"

    print(f"\n{'═'*55}")
    print(f"  🚀 BANQUE {client_id}  |  {DEVICE}")
    print(f"  LR={LEARNING_RATE}  Batch={BATCH_SIZE}  pos_weight={POS_WEIGHT}")
    print(f"{'═'*55}\n")

    # ── Données ───────────────────────────────────────────────────────────────
    data_dir = Path(f"data/node_{client_id}/tensors")

    if not data_dir.exists() or not any(data_dir.glob("X_batch_*.npy")):
        raise FileNotFoundError(
            f"❌ Aucun tensor trouvé dans {data_dir} pour le nœud {client_id}."
        )

    print(f"📂 Chargement des tensors depuis : {data_dir}")

    # CHANGEMENT 5 + 8 : batch_size=256, retourne 2 loaders (B2 : val=test)
    train_loader, val_loader = get_split_dataloaders(
        data_dir    = data_dir,
        train_ratio = TRAIN_RATIO,
        batch_size  = BATCH_SIZE,    # ← 64 → 256
        seq_len     = SEQ_LEN,
    )

    print(f"[FL-CLIENT] {client_id} → {server_address}")

    # CHANGEMENT 1 : build_model() au lieu de FraudRNN(...) hardcodé
    # Utilise automatiquement la config définie dans fraud_rnn.py (USE_DPGRU, etc.)
    model = build_model(use_dpgru=True)
    print(f"\n{model.info()}\n")

    # Client Flower
    client = FlowerClient(model, train_loader, val_loader)

    print(f"🚀 Connexion au serveur Flower : {server_address}")

    while True:
        try:
            print(f"🔗 Tentative de connexion à {server_address}...")
            start_client_compatible(client, server_address)
            print("[FL-CLIENT] Session terminée proprement")
            break
        except Exception as exc:
            print(f"[FL-CLIENT] Erreur de transport: {exc}")

        if not continuous:
            break

        print(f"[FL-CLIENT] Nouvelle tentative dans {retry_seconds}s... (nœud {client_id})")
        time.sleep(retry_seconds)


if __name__ == "__main__":
    main()