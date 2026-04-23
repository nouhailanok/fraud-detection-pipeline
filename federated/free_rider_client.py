"""
free_rider_client.py — Simulation d'un nœud FREE RIDER
════════════════════════════════════════════════════════
Un FREE RIDER est un nœud qui :
  - Reçoit le modèle global
  - NE s'entraîne PAS (ou presque pas)
  - Renvoie les poids INCHANGÉS (ou avec bruit minimal)
  - Déclare une loss et un ε artificiellement bas

Pourquoi c'est dangereux :
  - Le nœud profite du modèle global sans contribuer
  - Il dilue l'agrégation FedAvg (ses poids inchangés tirent le global vers 0)
  - Difficile à détecter car loss ≈ 0 et ε ≈ 0 semblent "bons"

Détection BehavioralAnalyzer :
  - norm_L2 ≈ 0  (∆W ≈ 0, poids non modifiés)
  - var_delta ≈ 0 (aucune variation)
  - train_loss = 0.0 (auto-déclarée)
  - epsilon = 0.0 (auto-déclarée)
  → Isolation Forest flagge ce profil comme anomalie

Variables d'environnement :
  CLIENT_ID            : ID du nœud (défaut: "4")
  FLOWER_SERVER_HOST   : adresse serveur Flower
  FLOWER_SERVER_PORT   : port serveur Flower
  FREE_RIDER_MODE      : "zero"   → renvoie poids inchangés (défaut)
                         "noise"  → ajoute bruit minimal σ=0.001
  FREE_RIDER_START_ROUND : round à partir duquel le nœud devient FREE_RIDER (défaut: 1)
  FL_CLIENT_CONTINUOUS : true/false

Usage :
  $env:CLIENT_ID = "4"
  $env:FREE_RIDER_MODE = "zero"
  $env:FREE_RIDER_START_ROUND = "3"
  python attack_sim/free_rider_client.py
"""

import os
import time
import inspect
import numpy as np
from pathlib import Path
from typing import Any, Optional, List

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from opacus import PrivacyEngine

# ── Import modèle ─────────────────────────────────────────────────────────────
try:
    from models.fraud_lstm import build_model
except ImportError:
    from models.fraud_rnn import build_model

from data.dataloader import get_split_dataloaders


# ============================================================================
# ⚙️  CONFIGURATION
# ============================================================================

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = float(os.getenv("FL_LR",         "0.0005"))
BATCH_SIZE    = int(os.getenv("FL_BATCH_SIZE",   "64"))
SEQ_LEN       = int(os.getenv("FL_SEQ_LEN",      "5"))
TRAIN_RATIO   = float(os.getenv("FL_TRAIN_RATIO","0.8"))
POS_WEIGHT    = float(os.getenv("FL_POS_WEIGHT", "167.0"))

FREE_RIDER_MODE        = os.getenv("FREE_RIDER_MODE",        "zero")   # "zero" | "noise"
FREE_RIDER_START_ROUND = int(os.getenv("FREE_RIDER_START_ROUND", "1"))
FREE_RIDER_NOISE_STD   = float(os.getenv("FREE_RIDER_NOISE_STD",  "0.001"))


# ============================================================================
# 🔧 UTILS
# ============================================================================

def get_model_parameters(model) -> List:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_parameters(model, parameters: List):
    for p, new_p in zip(model.parameters(), parameters):
        p.data = torch.tensor(new_p, dtype=torch.float32).to(DEVICE)


# ============================================================================
# 🕵️  FREE RIDER CLIENT
# ============================================================================

class FreeRiderClient(fl.client.NumPyClient):
    """
    Nœud FREE RIDER — hérite de NumPyClient et surcharge fit().

    En mode FREE RIDER :
      - fit() renvoie les poids reçus SANS les modifier
      - Déclare train_loss=0.0 et epsilon=0.0
      - num_examples=1 (pour ne pas être filtré par le serveur comme dp_exhausted)

    En mode légitime (avant FREE_RIDER_START_ROUND) :
      - Comportement normal avec DP
    """

    def __init__(self, model, train_loader, val_loader, client_id: str = "4"):
        self.model        = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.client_id    = client_id
        self.round_count  = 0

        pos_weight     = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Differential Privacy (utilisé uniquement en mode légitime)
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module           = self.model,
            optimizer        = self.optimizer,
            data_loader      = self.train_loader,
            noise_multiplier = float(os.getenv("DP_NOISE", "1.0")),
            max_grad_norm    = 1.0,
        )
        self.epsilon_target = float(os.getenv("DP_EPSILON_TARGET", "1.0"))
        self.dp_exhausted   = False

        print(f"\n{'═'*55}")
        print(f"  🕵️  FREE RIDER CLIENT — Nœud {client_id}")
        print(f"  Mode         : {FREE_RIDER_MODE}")
        print(f"  Activation   : à partir du round {FREE_RIDER_START_ROUND}")
        if FREE_RIDER_MODE == "noise":
            print(f"  Bruit σ      : {FREE_RIDER_NOISE_STD}")
        print(f"{'═'*55}\n")

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        set_model_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.round_count += 1
        global_round = config.get("round", self.round_count)

        # ── Mode FREE RIDER ───────────────────────────────────────────────────
        if self.round_count >= FREE_RIDER_START_ROUND:
            print(f"\n🕵️  Nœud {self.client_id} — FREE RIDER actif (round {global_round})")

            if FREE_RIDER_MODE == "noise":
                # Ajoute un bruit gaussien minimal — presque invisible
                params_with_noise = [
                    p + np.random.normal(0, FREE_RIDER_NOISE_STD, p.shape).astype(np.float32)
                    for p in parameters
                ]
                print(f"   Bruit gaussien σ={FREE_RIDER_NOISE_STD} ajouté")
                returned_params = params_with_noise
            else:
                # Mode "zero" : renvoie les poids EXACTEMENT tels que reçus
                print(f"   Poids inchangés renvoyés (∆W = 0)")
                returned_params = parameters

            print(f"   train_loss=0.0  epsilon=0.0  → détectable par Behavioral Analysis")

            return (
                returned_params,
                1,     # num_examples=1 → ne pas être filtré comme dp_exhausted=True
                {
                    "epsilon"    : 0.0,
                    "train_loss" : 0.0,
                    "dp_exhausted": False,
                    "client_id"  : self.client_id,
                    "free_rider" : True,
                },
            )

        # ── Mode LÉGITIME (avant FREE_RIDER_START_ROUND) ─────────────────────
        print(f"\n✅ Nœud {self.client_id} — Entraînement légitime (round {global_round})")

        if self.dp_exhausted:
            epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
            return (
                self.get_parameters(config),
                0,
                {"epsilon": float(epsilon), "train_loss": 0.0,
                 "dp_exhausted": True, "client_id": self.client_id},
            )

        self.model.train()
        local_epochs = int(config.get("local_epochs", 1))
        avg_loss = 0.0

        for epoch in range(local_epochs):
            total_loss = 0.0
            n_batches  = 0
            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()
                logits = self.model(x).float()
                loss   = self.criterion(logits, y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches  += 1
            avg_loss = total_loss / max(n_batches, 1)
            print(f"   [Epoch {epoch+1}/{local_epochs}] Loss: {avg_loss:.4f}")

        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        print(f"🔐 ε = {epsilon:.4f}")

        if epsilon > self.epsilon_target:
            self.dp_exhausted = True

        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"epsilon": float(epsilon), "train_loss": avg_loss,
             "dp_exhausted": self.dp_exhausted, "client_id": self.client_id},
        )

    def evaluate(self, parameters, config):
        """Évaluation identique au client légitime."""
        from sklearn.metrics import (
            recall_score, precision_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix,
        )

        self.set_parameters(parameters)
        self.model.eval()

        round_num  = config.get("round", "?")
        total_loss = 0.0
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = self.model(x).float()
                loss   = self.criterion(logits, y.float())
                probs  = torch.sigmoid(logits)
                preds  = (probs > 0.5).float()
                total_loss  += loss.item() * y.size(0)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        n           = len(all_targets)
        avg_loss    = total_loss / n if n > 0 else 0.0
        all_targets = np.array(all_targets)
        all_preds   = np.array(all_preds)
        all_probs   = np.array(all_probs)

        accuracy  = float((all_preds == all_targets).mean())
        recall    = float(recall_score(all_targets, all_preds,    pos_label=1, zero_division=0))
        precision = float(precision_score(all_targets, all_preds, pos_label=1, zero_division=0))
        f1        = float(f1_score(all_targets, all_preds,        pos_label=1, zero_division=0))

        try:
            roc_auc = float(roc_auc_score(all_targets, all_probs))
            pr_auc  = float(average_precision_score(all_targets, all_probs))
        except ValueError:
            roc_auc = pr_auc = 0.0

        print(f"\n  📊 Nœud {self.client_id} (FREE RIDER) — Eval Round {round_num}")
        print(f"  Loss={avg_loss:.4f}  AUC={roc_auc:.4f}  F1={f1:.4f}")

        return avg_loss, n, {
            "accuracy" : accuracy, "loss"     : avg_loss,
            "recall"   : recall,   "precision": precision,
            "f1"       : f1,       "roc_auc"  : roc_auc,
            "pr_auc"   : pr_auc,
        }


# ============================================================================
# 🔐 TLS
# ============================================================================

def _read_bytes(path_str: Optional[str]) -> Optional[bytes]:
    if not path_str:
        return None
    p = Path(path_str)
    return p.read_bytes() if p.exists() else None


def start_client_compatible(client, server_address: str) -> None:
    root_cert   = _read_bytes(os.getenv("FLOWER_CA_CERT"))
    client_cert = _read_bytes(os.getenv("FLOWER_CLIENT_CERT"))
    client_key  = _read_bytes(os.getenv("FLOWER_CLIENT_KEY"))

    start_fn = getattr(fl.client, "start_client", None)
    if start_fn is not None:
        kwargs    = {"server_address": server_address, "client": client.to_client()}
        sig       = inspect.signature(start_fn)
        if "root_certificates" in sig.parameters and root_cert:
            kwargs["root_certificates"] = root_cert
        if "certificates" in sig.parameters and client_cert and client_key:
            kwargs["certificates"] = (client_cert, client_key)
        start_fn(**kwargs)
        return

    numpy_fn = getattr(fl.client, "start_numpy_client")
    kwargs   = {"server_address": server_address, "client": client}
    sig      = inspect.signature(numpy_fn)
    if "root_certificates" in sig.parameters and root_cert:
        kwargs["root_certificates"] = root_cert
    if "certificates" in sig.parameters and client_cert and client_key:
        kwargs["certificates"] = (client_cert, client_key)
    numpy_fn(**kwargs)


# ============================================================================
# 🚀 MAIN
# ============================================================================

def main():
    client_id      = os.getenv("CLIENT_ID", "4")
    host           = os.getenv("FLOWER_SERVER_HOST", "127.0.0.1")
    port           = int(os.getenv("FLOWER_SERVER_PORT", "8080"))
    server_address = f"{host}:{port}"
    retry_seconds  = int(os.getenv("FL_CLIENT_RETRY_SECONDS", "10"))
    continuous     = os.getenv("FL_CLIENT_CONTINUOUS", "false").lower() == "true"

    data_dir = Path(f"data/node_{client_id}/tensors")
    if not data_dir.exists() or not any(data_dir.glob("X_batch_*.npy")):
        raise FileNotFoundError(f"❌ Aucun tensor dans {data_dir}")

    print(f"📂 Chargement tensors : {data_dir}")
    train_loader, val_loader = get_split_dataloaders(
        data_dir    = data_dir,
        train_ratio = TRAIN_RATIO,
        batch_size  = BATCH_SIZE,
        seq_len     = SEQ_LEN,
    )

    try:
        model = build_model(use_dplstm=True)
    except TypeError:
        model = build_model(use_dpgru=True)

    client = FreeRiderClient(model, train_loader, val_loader, client_id=client_id)

    print(f"🔗 Connexion → {server_address}")

    while True:
        try:
            start_client_compatible(client, server_address)
            print("[FREE RIDER] Session terminée")
            break
        except Exception as exc:
            print(f"[FREE RIDER] Erreur : {exc}")

        if not continuous:
            break

        print(f"[FREE RIDER] Retry dans {retry_seconds}s...")
        time.sleep(retry_seconds)


if __name__ == "__main__":
    main()
