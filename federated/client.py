import os
import time
import inspect
from pathlib import Path
from typing import Any, Optional, List, Tuple, Dict, Any


import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# 🔥 IMPORTS DE TON PROJET
from models.fraud_rnn import FraudRNN
from data.dataloader import get_dataloader


# ============================
# 🔧 UTILS
# ============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_parameters(model) -> List:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_parameters(model, parameters: List):
    for p, new_p in zip(model.parameters(), parameters):
        p.data = torch.tensor(new_p, dtype=torch.float32).to(DEVICE)



# ============================
#  FLOWER CLIENT
# ============================

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(DEVICE)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.BCEWithLogitsLoss()

        # ==========================================
        # 🛠️ 1. RÉPARATION DU MODÈLE POUR OPACUS
        # ==========================================
        # On fixe le modèle AVANT de créer l'optimiseur !
        self.model = ModuleValidator.fix(self.model)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 🔐 DIFFERENTIAL PRIVACY
        self.privacy_engine = PrivacyEngine()

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=float(os.getenv("DP_NOISE", "1.5")),
            max_grad_norm=1.0,
        )

    # ============================
    # 📡 PARAMÈTRES
    # ============================

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        set_model_parameters(self.model, parameters)

    # ============================
    # 🔥 TRAIN (AVEC DP)
    # ============================

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        local_epochs = int(config.get("local_epochs", 1))

        for _ in range(local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                self.optimizer.zero_grad()

                logits = self.model(x)
                loss = self.criterion(logits, y)

                loss.backward()
                self.optimizer.step()

        # 🔐 Calcul ε
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        print(f"🔐 ε = {epsilon:.4f}")
        target_epsilon = 1.0

        if epsilon > target_epsilon:
            print("⚠️ ε trop élevé → augmenter bruit")

        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"epsilon": float(epsilon)},
        )

    # ============================
    # 📊 EVALUATION
    # ============================

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                correct += (preds == y).sum().item()
                total += y.size(0)
                total_loss += loss.item() * y.size(0)

        accuracy = correct / total if total > 0 else 0
        loss = total_loss / total if total > 0 else 0

        return loss, total, {"accuracy": accuracy}

def _read_bytes(path_str: Optional[str]) -> Optional[bytes]:
    if not path_str:
        return None
    file_path = Path(path_str)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


def start_client_compatible(client: fl.client.NumPyClient, server_address: str) -> None:
    root_cert = _read_bytes(os.getenv("FLOWER_CA_CERT"))
    client_cert = _read_bytes(os.getenv("FLOWER_CLIENT_CERT"))
    client_key = _read_bytes(os.getenv("FLOWER_CLIENT_KEY"))

    start_client_fn = getattr(fl.client, "start_client", None)
    if start_client_fn is not None:
        kwargs: dict[str, Any] = {"server_address": server_address, "client": client.to_client()}
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


# ============================
# 🚀 MAIN
# ============================

def main():
    client_id = os.getenv("CLIENT_ID", "ingestion-1")
    host = os.getenv("FLOWER_SERVER_HOST", "flower")
    port = int(os.getenv("FLOWER_SERVER_PORT", "8080"))
    server_address = f"{host}:{port}"
    retry_seconds = int(os.getenv("FL_CLIENT_RETRY_SECONDS", "10"))
    continuous = os.getenv("FL_CLIENT_CONTINUOUS", "true").lower() == "true"
    #server_address = os.getenv("FLOWER_SERVER", "localhost:8080")

    
    print(f"🚀 Démarrage de la BANQUE {client_id} sur {DEVICE}")


    # chemins des données
    # data_dir = Path("data/tensors")

    x_train_path = f"data/node_{client_id}/X_train.npy"
    y_train_path = f"data/node_{client_id}/y_train.npy"
    x_test_path = f"data/node_{client_id}/X_test.npy"
    y_test_path = f"data/node_{client_id}/y_test.npy"


    # x_path = data_dir / f"X_{client_id}.npy"
    # y_path = data_dir / f"y_{client_id}.npy"



    if not x_train_path.exists() or not y_train_path.exists() or not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(f"❌ Données introuvables pour {client_id}")

    print(f"📂 Chargement des données pour {client_id}")

    # DataLoader
    train_loader = get_dataloader(x_train_path, y_train_path, batch_size=64, seq_len=5, shuffle=True)
    val_loader = get_dataloader(x_test_path, y_test_path, batch_size=64, seq_len=5, shuffle=False)

    print(f"[FL-CLIENT] {client_id} -> {server_address}")


    # Modèle
    model = FraudRNN(input_dim=26, hidden_dim=64)

    # Client Flower
    client = FlowerClient(model, train_loader, val_loader)


    print(f"🚀 Connexion au serveur Flower : {server_address}")


    while True:
        try:
            print(f"🔗 Tentative de connexion à {server_address}...")
            start_client_compatible(client, server_address)
            print("[FL-CLIENT] Session terminée proprement")
            break # Si on finit proprement, on sort de la boucle
        except Exception as exc:
            print(f"[FL-CLIENT] Erreur de transport: {exc}")
        if not continuous:
            break

        print(f"[FL-CLIENT] Nouvelle tentative dans {retry_seconds}s...")
        print(client_id)
        time.sleep(retry_seconds)

        


if __name__ == "__main__":
    main()