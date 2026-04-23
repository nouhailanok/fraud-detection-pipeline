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

import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)
from opacus import PrivacyEngine

# ── Imports du projet ────────────────────────────────────────────────────────
# CHANGEMENT 1 : build_model() au lieu de FraudRNN() hardcodé
from models.fraud_rnn import build_model
# from models.fraud_lstm import build_model
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
    def __init__(self, model, train_loader, val_loader, client_id: str = "1"):
        self.model        = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.client_id    = client_id

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

        # 🔐 Budget DP — arrêt automatique quand ε dépasse le seuil
        self.epsilon_target  = float(os.getenv("DP_EPSILON_TARGET", "1.0"))
        self.dp_exhausted    = False   # True quand ε > epsilon_target

        # 🔐 Budget DP — arrêt automatique quand ε dépasse le seuil
        self.epsilon_target  = float(os.getenv("DP_EPSILON_TARGET", "1.0"))
        self.dp_exhausted    = False   # True quand ε > epsilon_target

    # ── Paramètres FL ────────────────────────────────────────────────────────

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        set_model_parameters(self.model, parameters)

    # ── Entraînement local (avec DP) ─────────────────────────────────────────

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # ── Vérification budget DP AVANT le round ────────────────────────────
        # Si le budget est déjà épuisé depuis le round précédent,
        # on renvoie les paramètres reçus sans entraîner
        if self.dp_exhausted:
            print(f"\n🔒 Nœud {self.client_id} — Budget DP épuisé (ε > {self.epsilon_target})")
            print(f"   Ce nœud ne participera plus à l'entraînement")
            print(f"   Les autres nœuds continuent...")
            epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
            return (
                self.get_parameters(config),
                0,   # 0 exemples → FedAvg ignore ce nœud (poids nul)
                {"epsilon": float(epsilon), "train_loss": 0.0, "dp_exhausted": True},
            )

        # ── Entraînement normal ───────────────────────────────────────────────
        self.model.train()
        local_epochs = int(config.get("local_epochs", 1))
        avg_loss = 0.0
        avg_loss = 0.0

        for epoch in range(local_epochs):
            total_loss = 0.0
            n_batches  = 0

            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                self.optimizer.zero_grad()

                logits      = self.model(x).float()
                loss        = self.criterion(logits, y.float())

                loss.backward()

                # torch.nn.utils.clip_grad_norm_(
                #     self.model.parameters(), max_norm=1.0
                # )

                self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"   [FL-Epoch {epoch+1}/{local_epochs}] Loss: {avg_loss:.4f}")

        # ── Calcul ε APRÈS le round complet ──────────────────────────────────
        # ── Calcul ε APRÈS le round complet ──────────────────────────────────
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        print(f"🔐 ε = {epsilon:.4f}  (target ≤ {self.epsilon_target})")

        # Vérification APRÈS le round (jamais pendant)
        if epsilon > self.epsilon_target:
            self.dp_exhausted = True
            print(f"⚠️  Nœud {self.client_id} : ε = {epsilon:.4f} > {self.epsilon_target}")
            print(f"   Budget DP épuisé — ce nœud arrêtera APRÈS ce round")
            print(f"   Le round actuel est complété normalement ✅")
        print(f"🔐 ε = {epsilon:.4f}  (target ≤ {self.epsilon_target})")

        # Vérification APRÈS le round (jamais pendant)
        if epsilon > self.epsilon_target:
            self.dp_exhausted = True
            print(f"⚠️  Nœud {self.client_id} : ε = {epsilon:.4f} > {self.epsilon_target}")
            print(f"   Budget DP épuisé — ce nœud arrêtera APRÈS ce round")
            print(f"   Le round actuel est complété normalement ✅")

        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"epsilon": float(epsilon), "train_loss": avg_loss,
             "dp_exhausted": self.dp_exhausted},
        )

    # ── Évaluation locale — identique à train_local.py ──────────────────────

    def evaluate(self, parameters, config):
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

        n          = len(all_targets)
        avg_loss   = total_loss / n if n > 0 else 0.0
        all_targets = np.array(all_targets)
        all_preds   = np.array(all_preds)
        all_probs   = np.array(all_probs)

        # ── Métriques détaillées (comme train_local.py) ───────────────────────
        accuracy  = (all_preds == all_targets).mean()
        recall    = recall_score(all_targets, all_preds,    pos_label=1, zero_division=0)
        precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
        f1        = f1_score(all_targets, all_preds,        pos_label=1, zero_division=0)

        try:
            roc_auc = roc_auc_score(all_targets, all_probs)
            pr_auc  = average_precision_score(all_targets, all_probs)
        except ValueError:
            roc_auc = pr_auc = 0.0

        # ── Matrice de confusion ──────────────────────────────────────────────
        cm = confusion_matrix(all_targets, all_preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        # ── Seuil optimal calibré sur val (max F1 fraude) ────────────────────
        best_f1_opt, best_thresh = 0.0, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            preds_t = (all_probs >= t).astype(float)
            f1_t    = f1_score(all_targets, preds_t, pos_label=1, zero_division=0)
            if f1_t > best_f1_opt:
                best_f1_opt, best_thresh = f1_t, t

        preds_opt = (all_probs >= best_thresh).astype(float)
        cm_opt    = confusion_matrix(all_targets, preds_opt)
        tn_o, fp_o, fn_o, tp_o = cm_opt.ravel() if cm_opt.shape == (2,2) else (0,0,0,0)

        # ── Rapport console (nœud) ────────────────────────────────────────────
        print(f"\n{'═'*52}")
        print(f"  📊  Nœud {self.client_id} — Évaluation Round {round_num}")
        print(f"{'═'*52}")
        print(f"  Transactions : {n:,}  |  Fraudes : {int(all_targets.sum())}")
        print(f"{'─'*52}")
        print(f"  Métriques globales (seuil 0.5)")
        print(f"    Accuracy  : {accuracy:.4f}")
        print(f"    Loss      : {avg_loss:.4f}")
        print(f"    Recall    : {recall:.4f}")
        print(f"    Precision : {precision:.4f}")
        print(f"    F1-fraude : {f1:.4f}")
        print(f"    AUC-ROC   : {roc_auc:.4f}")
        print(f"    PR-AUC    : {pr_auc:.4f}")
        print(f"{'─'*52}")
        print(f"  Matrice de confusion (seuil 0.5)")
        print(f"    VP (fraudes détectées) : {tp:>6,}  / {int(all_targets.sum())}")
        print(f"    FP (fausses alertes)   : {fp:>6,}  ⚠️")
        print(f"    FN (fraudes manquées)  : {fn:>6,}  🚨")
        print(f"    VN (légitimes OK)      : {tn:>6,}")
        print(f"{'─'*52}")
        print(f"  Seuil optimal : {best_thresh:.2f}  (F1={best_f1_opt:.4f})")
        print(f"    VP : {tp_o:,}  |  FP : {fp_o:,}  |  FN : {fn_o:,}")
        recall_opt = tp_o / max(tp_o + fn_o, 1)
        prec_opt   = tp_o / max(tp_o + fp_o, 1)
        print(f"    Rappel    : {recall_opt:.2%}")
        print(f"    Précision : {prec_opt:.2%}")
        print(f"{'═'*52}\n")

        # ── Sauvegarde locale du round ────────────────────────────────────────
        self._save_round_log(round_num, {
            "loss": avg_loss, "accuracy": accuracy,
            "recall": recall, "precision": precision, "f1": f1,
            "roc_auc": roc_auc, "pr_auc": pr_auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "best_thresh": best_thresh, "f1_opt": best_f1_opt,
            "tp_opt": int(tp_o), "fp_opt": int(fp_o), "fn_opt": int(fn_o),
            "recall_opt": round(recall_opt, 4), "prec_opt": round(prec_opt, 4),
            "n_examples": n,
        })

        return avg_loss, n, {
            "accuracy" : float(accuracy),
            "loss"     : float(avg_loss),
            "recall"   : float(recall),
            "precision": float(precision),
            "f1"       : float(f1),
            "roc_auc"  : float(roc_auc),
            "pr_auc"   : float(pr_auc),
        }

    def _save_round_log(self, round_num, metrics: dict) -> None:
        """Sauvegarde les métriques du round dans un fichier JSON par nœud."""
        import json
        log_dir  = Path(f"logs/node_{self.client_id}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "eval_history.json"

        history = []
        if log_file.exists():
            try:
                history = json.loads(log_file.read_text())
            except Exception:
                history = []

        history.append({"round": round_num, **metrics})
        log_file.write_text(json.dumps(history, indent=2))


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
    # server_address = "localhost:8080"
    # server_address = "localhost:8080"
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
    # train_loader, val_loader = get_split_dataloaders(
    #     data_dir    = data_dir,
    #     train_ratio = TRAIN_RATIO,
    #     batch_size  = BATCH_SIZE,    # ← 64 → 256
    #     seq_len     = SEQ_LEN,
    # )

    loaders = get_split_dataloaders(
        data_dir    = data_dir,
        train_ratio = TRAIN_RATIO,
        batch_size  = BATCH_SIZE,    # ← 64 → 256
        seq_len     = SEQ_LEN,
    )

    if len(loaders) == 2:
        train_loader, eval_loader = loaders
    elif len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
        # Prefer test when available to keep evaluation stable and less noisy.
        eval_loader = test_loader if len(test_loader.dataset) > 0 else val_loader
    else:
        raise ValueError(
            f"get_split_dataloaders() doit retourner 2 ou 3 loaders, reçu: {len(loaders)}"
        )

    print(f"[FL-CLIENT] {client_id} → {server_address}")

    # CHANGEMENT 1 : build_model() au lieu de FraudRNN(...) hardcodé
    # Utilise automatiquement la config définie dans fraud_rnn.py (USE_DPGRU, etc.)
    model = build_model(use_dpgru=True)
    # model = build_model(use_dplstm=True)
    print(f"\n{model.info()}\n")

    # Client Flower
    client = FlowerClient(model, train_loader, eval_loader, client_id=client_id)

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


# # Ajoute temporairement en haut de client.py pour déboguer :
# import torch
# print(f"CUDA disponible : {torch.cuda.is_available()}")
# print(f"GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'aucun'}")

if __name__ == "__main__":
    main()


# """
# client.py — Nœud Flower pour le Federated Learning
# ════════════════════════════════════════════════════
# Changements par rapport à la version originale :

#   1. Import build_model() depuis fraud_rnn.py
#        → Remplace FraudRNN(...) hardcodé
#        → Utilise automatiquement DPGRU si USE_DPGRU=True dans fraud_rnn.py
#        → Un seul endroit pour changer l'architecture

#   2. ModuleValidator.fix() supprimé
#        → DPGRU est nativement compatible Opacus — pas besoin de fix
#        → GRU standard reste incompatible, mais on n'utilise plus GRU en FL

#   3. pos_weight = 167.0 ajouté dans BCEWithLogitsLoss
#        → Aligné avec train_local.py (ratio réel : 362 974 légit / 2 174 fraudes)
#        → Sans ça, le modèle FL ignorait le déséquilibre des classes

#   4. learning_rate : 0.001 → 0.0005
#        → Aligné avec le meilleur run (D2/D3)
#        → Configurable via variable d'environnement LR

#   5. batch_size : 64 → 256
#        → Aligné avec train_local.py pour une meilleure utilisation GPU
#        → Configurable via variable d'environnement BATCH_SIZE

#   6. gradient clipping ajouté dans fit()
#        → clip_grad_norm_(max_norm=1.0)
#        → Même protection que train_local.py contre les explosions de gradient

#   7. logits.float() dans evaluate()
#        → Même fix FP32 que train_local.py
#        → Évite les NaN avec pos_weight=167 en FP16

#   8. get_split_dataloaders → retourne 2 loaders (train + test)
#        → val_loader = test_loader (users 100% inconnus = pas de biais en FL)
#        → batch_size et seq_len alignés avec train_local.py
# """

# import os
# import time
# import inspect
# from pathlib import Path
# from typing import Any, Optional, List

# import numpy as np
# import flwr as fl
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from sklearn.metrics import (
#     recall_score, precision_score, f1_score,
#     roc_auc_score, average_precision_score,
#     confusion_matrix,
# )
# from opacus import PrivacyEngine

# # ── Imports du projet ────────────────────────────────────────────────────────
# # CHANGEMENT 1 : build_model() au lieu de FraudRNN() hardcodé
# from models.fraud_rnn import build_model
# from data.dataloader import get_split_dataloaders


# # ============================================================================
# # ⚙️  CONFIGURATION
# #     Les valeurs par défaut sont alignées avec les meilleurs runs (D2/D3).
# #     Toutes sont surchargeables via variables d'environnement Docker.
# # ============================================================================

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # CHANGEMENT 4 + 5 : LR et BATCH_SIZE alignés avec train_local.py
# LEARNING_RATE = float(os.getenv("FL_LR",         "0.0005"))   # ← 0.001 → 0.0005
# BATCH_SIZE    = int(os.getenv("FL_BATCH_SIZE",   "256"))      # ← 64 → 256
# SEQ_LEN       = int(os.getenv("FL_SEQ_LEN",      "5"))
# TRAIN_RATIO   = float(os.getenv("FL_TRAIN_RATIO","0.8"))

# # CHANGEMENT 3 : pos_weight aligné avec train_local.py
# POS_WEIGHT    = float(os.getenv("FL_POS_WEIGHT", "167.0"))    # ← manquait


# # ============================================================================
# # 🔧 UTILS
# # ============================================================================

# def get_model_parameters(model) -> List:
#     return [p.detach().cpu().numpy() for p in model.parameters()]


# def set_model_parameters(model, parameters: List):
#     for p, new_p in zip(model.parameters(), parameters):
#         p.data = torch.tensor(new_p, dtype=torch.float32).to(DEVICE)


# # ============================================================================
# # 🌸 FLOWER CLIENT
# # ============================================================================

# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, model, train_loader, val_loader, client_id: str = "1"):
#         self.model        = model.to(DEVICE)
#         self.train_loader = train_loader
#         self.val_loader   = val_loader
#         self.client_id    = client_id

#         # CHANGEMENT 3 : pos_weight ajouté — aligné avec train_local.py
#         pos_weight       = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
#         self.criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         # CHANGEMENT 2 : ModuleValidator.fix() supprimé
#         # DPGRU est nativement compatible Opacus — pas besoin de réparer le modèle
#         # self.model = ModuleValidator.fix(self.model)  ← supprimé

#         # CHANGEMENT 4 : LR aligné avec train_local.py
#         self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

#         # 🔐 Differential Privacy
#         self.privacy_engine = PrivacyEngine()

#         self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
#             module          = self.model,
#             optimizer       = self.optimizer,
#             data_loader     = self.train_loader,
#             noise_multiplier= float(os.getenv("DP_NOISE", "1.5")),
#             max_grad_norm   = 1.0,
#         )

#     # ── Paramètres FL ────────────────────────────────────────────────────────

#     def get_parameters(self, config):
#         return get_model_parameters(self.model)

#     def set_parameters(self, parameters):
#         set_model_parameters(self.model, parameters)

#     # ── Entraînement local (avec DP) ─────────────────────────────────────────

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.train()

#         local_epochs = int(config.get("local_epochs", 1))

#         for epoch in range(local_epochs):
#             total_loss = 0.0
#             n_batches  = 0

#             for x, y in self.train_loader:
#                 x, y = x.to(DEVICE), y.to(DEVICE)

#                 self.optimizer.zero_grad()

#                 # Forward en FP32 (même approche que train_local.py)
#                 logits      = self.model(x).float()
#                 loss        = self.criterion(logits, y.float())

#                 loss.backward()

#                 # CHANGEMENT 6 : gradient clipping — aligné avec train_local.py
#                 torch.nn.utils.clip_grad_norm_(
#                     self.model.parameters(), max_norm=1.0
#                 )

#                 self.optimizer.step()

#                 total_loss += loss.item()
#                 n_batches  += 1

#             avg_loss = total_loss / max(n_batches, 1)
#             print(f"   [FL-Epoch {epoch+1}/{local_epochs}] Loss: {avg_loss:.4f}")

#         # 🔐 Calcul du budget ε
#         epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
#         print(f"🔐 ε = {epsilon:.4f}  (target ≤ 1.0)")

#         if epsilon > 1.0:
#             print("⚠️  ε > 1.0 → privacy faible → augmenter DP_NOISE")

#         return (
#             self.get_parameters(config),
#             len(self.train_loader.dataset),
#             {"epsilon": float(epsilon), "train_loss": avg_loss},
#         )

#     # ── Évaluation locale — identique à train_local.py ──────────────────────

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.eval()

#         round_num  = config.get("round", "?")
#         total_loss = 0.0
#         all_preds, all_targets, all_probs = [], [], []

#         with torch.no_grad():
#             for x, y in self.val_loader:
#                 x, y = x.to(DEVICE), y.to(DEVICE)

#                 logits = self.model(x).float()
#                 loss   = self.criterion(logits, y.float())

#                 probs  = torch.sigmoid(logits)
#                 preds  = (probs > 0.5).float()

#                 total_loss  += loss.item() * y.size(0)
#                 all_probs.extend(probs.cpu().numpy())
#                 all_preds.extend(preds.cpu().numpy())
#                 all_targets.extend(y.cpu().numpy())

#         n          = len(all_targets)
#         avg_loss   = total_loss / n if n > 0 else 0.0
#         all_targets = np.array(all_targets)
#         all_preds   = np.array(all_preds)
#         all_probs   = np.array(all_probs)

#         # ── Métriques détaillées (comme train_local.py) ───────────────────────
#         accuracy  = (all_preds == all_targets).mean()
#         recall    = recall_score(all_targets, all_preds,    pos_label=1, zero_division=0)
#         precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
#         f1        = f1_score(all_targets, all_preds,        pos_label=1, zero_division=0)

#         try:
#             roc_auc = roc_auc_score(all_targets, all_probs)
#             pr_auc  = average_precision_score(all_targets, all_probs)
#         except ValueError:
#             roc_auc = pr_auc = 0.0

#         # ── Matrice de confusion ──────────────────────────────────────────────
#         cm = confusion_matrix(all_targets, all_preds)
#         tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

#         # ── Seuil optimal calibré sur val (max F1 fraude) ────────────────────
#         best_f1_opt, best_thresh = 0.0, 0.5
#         for t in np.arange(0.05, 0.95, 0.01):
#             preds_t = (all_probs >= t).astype(float)
#             f1_t    = f1_score(all_targets, preds_t, pos_label=1, zero_division=0)
#             if f1_t > best_f1_opt:
#                 best_f1_opt, best_thresh = f1_t, t

#         preds_opt = (all_probs >= best_thresh).astype(float)
#         cm_opt    = confusion_matrix(all_targets, preds_opt)
#         tn_o, fp_o, fn_o, tp_o = cm_opt.ravel() if cm_opt.shape == (2,2) else (0,0,0,0)

#         # ── Rapport console (nœud) ────────────────────────────────────────────
#         print(f"\n{'═'*52}")
#         print(f"  📊  Nœud {self.client_id} — Évaluation Round {round_num}")
#         print(f"{'═'*52}")
#         print(f"  Transactions : {n:,}  |  Fraudes : {int(all_targets.sum())}")
#         print(f"{'─'*52}")
#         print(f"  Métriques globales (seuil 0.5)")
#         print(f"    Accuracy  : {accuracy:.4f}")
#         print(f"    Loss      : {avg_loss:.4f}")
#         print(f"    Recall    : {recall:.4f}")
#         print(f"    Precision : {precision:.4f}")
#         print(f"    F1-fraude : {f1:.4f}")
#         print(f"    AUC-ROC   : {roc_auc:.4f}")
#         print(f"    PR-AUC    : {pr_auc:.4f}")
#         print(f"{'─'*52}")
#         print(f"  Matrice de confusion (seuil 0.5)")
#         print(f"    VP (fraudes détectées) : {tp:>6,}  / {int(all_targets.sum())}")
#         print(f"    FP (fausses alertes)   : {fp:>6,}  ⚠️")
#         print(f"    FN (fraudes manquées)  : {fn:>6,}  🚨")
#         print(f"    VN (légitimes OK)      : {tn:>6,}")
#         print(f"{'─'*52}")
#         print(f"  Seuil optimal : {best_thresh:.2f}  (F1={best_f1_opt:.4f})")
#         print(f"    VP : {tp_o:,}  |  FP : {fp_o:,}  |  FN : {fn_o:,}")
#         recall_opt = tp_o / max(tp_o + fn_o, 1)
#         prec_opt   = tp_o / max(tp_o + fp_o, 1)
#         print(f"    Rappel    : {recall_opt:.2%}")
#         print(f"    Précision : {prec_opt:.2%}")
#         print(f"{'═'*52}\n")

#         # ── Sauvegarde locale du round ────────────────────────────────────────
#         self._save_round_log(round_num, {
#             "loss": avg_loss, "accuracy": accuracy,
#             "recall": recall, "precision": precision, "f1": f1,
#             "roc_auc": roc_auc, "pr_auc": pr_auc,
#             "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
#             "best_thresh": best_thresh, "f1_opt": best_f1_opt,
#             "tp_opt": int(tp_o), "fp_opt": int(fp_o), "fn_opt": int(fn_o),
#             "recall_opt": round(recall_opt, 4), "prec_opt": round(prec_opt, 4),
#             "n_examples": n,
#         })

#         return avg_loss, n, {
#             "accuracy" : float(accuracy),
#             "loss"     : float(avg_loss),
#             "recall"   : float(recall),
#             "precision": float(precision),
#             "f1"       : float(f1),
#             "roc_auc"  : float(roc_auc),
#             "pr_auc"   : float(pr_auc),
#         }

#     def _save_round_log(self, round_num, metrics: dict) -> None:
#         """Sauvegarde les métriques du round dans un fichier JSON par nœud."""
#         import json
#         log_dir  = Path(f"logs/node_{self.client_id}")
#         log_dir.mkdir(parents=True, exist_ok=True)
#         log_file = log_dir / "eval_history.json"

#         history = []
#         if log_file.exists():
#             try:
#                 history = json.loads(log_file.read_text())
#             except Exception:
#                 history = []

#         history.append({"round": round_num, **metrics})
#         log_file.write_text(json.dumps(history, indent=2))


# # ============================================================================
# # 🔐 TLS
# # ============================================================================

# def _read_bytes(path_str: Optional[str]) -> Optional[bytes]:
#     if not path_str:
#         return None
#     file_path = Path(path_str)
#     if not file_path.exists():
#         return None
#     return file_path.read_bytes()


# def start_client_compatible(client: fl.client.NumPyClient, server_address: str) -> None:
#     root_cert   = _read_bytes(os.getenv("FLOWER_CA_CERT"))
#     client_cert = _read_bytes(os.getenv("FLOWER_CLIENT_CERT"))
#     client_key  = _read_bytes(os.getenv("FLOWER_CLIENT_KEY"))

#     start_client_fn = getattr(fl.client, "start_client", None)
#     if start_client_fn is not None:
#         kwargs: dict[str, Any] = {
#             "server_address": server_address,
#             "client": client.to_client(),
#         }
#         signature = inspect.signature(start_client_fn)

#         if "root_certificates" in signature.parameters and root_cert is not None:
#             kwargs["root_certificates"] = root_cert

#         if "certificates" in signature.parameters and client_cert is not None and client_key is not None:
#             kwargs["certificates"] = (client_cert, client_key)

#         start_client_fn(**kwargs)
#         return

#     numpy_client_fn = getattr(fl.client, "start_numpy_client")
#     kwargs = {"server_address": server_address, "client": client}
#     signature = inspect.signature(numpy_client_fn)

#     if "root_certificates" in signature.parameters and root_cert is not None:
#         kwargs["root_certificates"] = root_cert

#     if "certificates" in signature.parameters and client_cert is not None and client_key is not None:
#         kwargs["certificates"] = (client_cert, client_key)

#     numpy_client_fn(**kwargs)


# # ============================================================================
# # 🚀 MAIN
# # ============================================================================

# def main() -> None:
#     client_id      = os.getenv("CLIENT_ID", "1")
#     host           = os.getenv("FLOWER_SERVER_HOST", "flower")
#     port           = int(os.getenv("FLOWER_SERVER_PORT", "8080"))
#     server_address = f"{host}:{port}"
#     # server_address = "localhost:8080"
#     retry_seconds  = int(os.getenv("FL_CLIENT_RETRY_SECONDS", "10"))
#     continuous     = os.getenv("FL_CLIENT_CONTINUOUS", "true").lower() == "true"

#     print(f"\n{'═'*55}")
#     print(f"  🚀 BANQUE {client_id}  |  {DEVICE}")
#     print(f"  LR={LEARNING_RATE}  Batch={BATCH_SIZE}  pos_weight={POS_WEIGHT}")
#     print(f"{'═'*55}\n")

#     # ── Données ───────────────────────────────────────────────────────────────
#     data_dir = Path(f"data/node_{client_id}/tensors")

#     if not data_dir.exists() or not any(data_dir.glob("X_batch_*.npy")):
#         raise FileNotFoundError(
#             f"❌ Aucun tensor trouvé dans {data_dir} pour le nœud {client_id}."
#         )

#     print(f"📂 Chargement des tensors depuis : {data_dir}")

#     # CHANGEMENT 5 + 8 : batch_size=256, retourne 2 loaders (B2 : val=test)
#     train_loader, val_loader = get_split_dataloaders(
#         data_dir    = data_dir,
#         train_ratio = TRAIN_RATIO,
#         batch_size  = BATCH_SIZE,    # ← 64 → 256
#         seq_len     = SEQ_LEN,
#     )

#     print(f"[FL-CLIENT] {client_id} → {server_address}")

#     # CHANGEMENT 1 : build_model() au lieu de FraudRNN(...) hardcodé
#     # Utilise automatiquement la config définie dans fraud_rnn.py (USE_DPGRU, etc.)
#     model = build_model(use_dpgru=True)
#     print(f"\n{model.info()}\n")

#     # Client Flower
#     client = FlowerClient(model, train_loader, val_loader, client_id=client_id)

#     print(f"🚀 Connexion au serveur Flower : {server_address}")

#     while True:
#         try:
#             print(f"🔗 Tentative de connexion à {server_address}...")
#             start_client_compatible(client, server_address)
#             print("[FL-CLIENT] Session terminée proprement")
#             break
#         except Exception as exc:
#             print(f"[FL-CLIENT] Erreur de transport: {exc}")

#         if not continuous:
#             break

#         print(f"[FL-CLIENT] Nouvelle tentative dans {retry_seconds}s... (nœud {client_id})")
#         time.sleep(retry_seconds)


# # # Ajoute temporairement en haut de client.py pour déboguer :
# # import torch
# # print(f"CUDA disponible : {torch.cuda.is_available()}")
# # print(f"GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'aucun'}")

# if __name__ == "__main__":
#     main()