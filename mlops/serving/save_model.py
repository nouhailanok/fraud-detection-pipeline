# ============================================================================
# save_model.py — Sauvegarde du modèle promu dans le BentoML Store
# ============================================================================
# Usage :
#   python serving/save_model.py
#
# Prérequis :
#   - Le modèle doit avoir été promu via registry.py --promote
#   - best_model/model.npz et best_model/metadata.json doivent exister
#
# Ce script :
#   1. Charge le checkpoint .npz (ou .pt)
#   2. Reconstruit l'architecture DPGRU via build_model()
#   3. Injecte les poids dans le modèle
#   4. Sauvegarde dans le BentoML Model Store avec les métriques en tags
# ============================================================================

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Imports du projet. ──

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

try:
    from models.fraud_rnn import build_model
except ImportError:
    from models.fraud_lstm import build_model

# ── BentoML ──
try:
    import bentoml
except ImportError:
    print("[ERROR] BentoML n'est pas installé. Lancez : pip install bentoml>=1.2.0")
    sys.exit(1)

def resolve_best_model_dir(cli_value: str | None = None) -> Path:
    if cli_value:
        return Path(cli_value)

    env_value = os.getenv("BEST_MODEL_DIR")
    if env_value:
        return Path(env_value)

    candidates = [
        PROJECT_ROOT / "model_registry" / "best_model",
        PROJECT_ROOT / "mlops" / "model_registry" / "best_model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _metric_from_metadata(metadata: dict, key: str, default: float = 0.0) -> float:
    metrics = metadata.get("metrics", {}) if isinstance(metadata.get("metrics"), dict) else {}
    if key in metrics:
        return float(metrics.get(key, default))
    return float(metadata.get(key, default))


def load_checkpoint(checkpoint_path: str):
    """
    Charge un checkpoint .npz ou .pt/.pth et retourne le state_dict.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    if path.suffix == ".npz":
        data = dict(np.load(path, allow_pickle=True))
        # npz peut contenir directement le state_dict ou être wrappé
        if "state_dict" in data:
            state_dict = {k: torch.from_numpy(v) for k, v in data["state_dict"].items()}
        else:
            # Essayer de convertir tous les arrays
            state_dict = {}
            for k, v in data.items():
                try:
                    state_dict[k] = torch.from_numpy(v)
                except Exception:
                    pass  # Ignorer les entrées non-tensor
        return state_dict

    elif path.suffix in (".pt", ".pth"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    else:
        raise ValueError(f"Format de checkpoint non supporté : {path.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Save promoted FL model to BentoML store")
    parser.add_argument("--best-model-dir", help="Path to best_model directory", default=None)
    parser.add_argument("--model-name", help="BentoML model name", default=os.getenv("BENTOML_MODEL_NAME", "fraud_dpgru_v1"))
    args = parser.parse_args()

    best_model_dir = resolve_best_model_dir(args.best_model_dir)
    model_file = best_model_dir / "model.npz"
    meta_file = best_model_dir / "metadata.json"

    print("=" * 60)
    print("BentoML — Sauvegarde du modèle en production")
    print("=" * 60)

    # ── Vérifier les fichiers ──
    if not model_file.exists():
        print(f"[ERROR] {model_file} introuvable. Promouvez d'abord un run avec registry.py --promote")
        sys.exit(1)

    if not meta_file.exists():
        print(f"[ERROR] {meta_file} introuvable.")
        sys.exit(1)

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    run_name = metadata.get("run_name", metadata.get("run_id", "unknown"))
    recall = _metric_from_metadata(metadata, "recall")
    f1 = _metric_from_metadata(metadata, "f1")
    epsilon_final = _metric_from_metadata(metadata, "epsilon_final")
    trust_score = float(metadata.get("trust_score", 0.0))

    print(f"[INFO] Run promu    : {run_name}")
    print(f"[INFO] Recall       : {recall:.4f}")
    print(f"[INFO] F1           : {f1:.4f}")
    print(f"[INFO] epsilon      : {epsilon_final:.4f}")
    print(f"[INFO] Trust score  : {trust_score:.2f}")

    # ── Reconstruire le modèle ──
    print("\n[INFO] Reconstruction du modèle DPGRU...")
    model = build_model(use_dpgru=False)  # On utilise la même architecture que pour l'entraînement centralisé
    print(f"[INFO] Architecture : {type(model).__name__}")
    model.eval()

    # ── Charger les poids ──
    print(f"[INFO] Chargement du checkpoint : {model_file}")
    state_dict = load_checkpoint(str(model_file))
    # state_dict = model.state_dict()
    path = Path(str(model_file)) 


    
    if path.suffix == ".npz":
        print(f"[INFO] Checkpoint au format .npz détecté. Tentative de chargement...")
        mapping = {
            "param_0": "gru.weight_ih_l0",
            "param_1": "gru.bias_ih_l0",
            "param_2": "gru.weight_hh_l0",
            "param_3": "gru.bias_hh_l0",
            "param_4": "gru.weight_ih_l1",
            "param_5": "gru.bias_ih_l1",
            "param_6": "gru.weight_hh_l1",
            "param_7": "gru.bias_hh_l1",
            "param_8": "gru.weight_ih_l2",
            "param_9": "gru.bias_ih_l2",
            "param_10": "gru.weight_hh_l2",
            "param_11": "gru.bias_hh_l2",
            "param_12": "classifier.0.weight",
            "param_13": "classifier.0.bias",
            "param_14": "classifier.3.weight",
            "param_15": "classifier.3.bias",
            "param_0" : "gru.l0.ih.weight",
            "param_1" : "gru.l0.ih.bias",
            "param_2" : "gru.l0.hh.weight",
            "param_3" : "gru.l0.hh.bias",
            "param_4" : "gru.l1.ih.weight",
            "param_5" : "gru.l1.ih.bias",
            "param_6" : "gru.l1.hh.weight",
            "param_7" : "gru.l1.hh.bias",
            "param_8" : "gru.l2.ih.weight",
            "param_9" : "gru.l2.ih.bias",
            "param_10" : "gru.l2.hh.weight",
            "param_11" : "gru.l2.hh.bias",
            "param_12": "classifier.0.weight",
            "param_13": "classifier.0.bias",
            "param_14": "classifier.3.weight",
            "param_15": "classifier.3.bias"
        }
        data = dict(np.load(path, allow_pickle=True))
        print("\n===== CHECKPOINT PARAMS =====")
        state_dict = {}
        for ckpt_key, model_key in mapping.items():
            tensor = torch.from_numpy(np.array(data[ckpt_key]))

            print(f"[MAP] {ckpt_key} → {model_key} | shape={tensor.shape}")

            state_dict[model_key] = tensor

    elif path.suffix in (".pt", ".pth"):
        print(f"[INFO] Checkpoint .pt détecté — GRU standard (centralisé)...")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # train_central.py sauvegarde sous la clé "model_state"
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            epoch = ckpt.get("epoch", "?")
            val_f1 = ckpt.get("val_f1", 0.0)
            print(f"[INFO] Epoch={epoch} | val_F1={val_f1:.4f}")
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt  # state_dict direct

        print("🔍 Keys du state_dict :")
        print(list(state_dict.keys())[:6])
    else:
        print(f"[ERROR] Format de checkpoint non supporté : {path.suffix}")

    print("🔍 Keys du state_dict :")
    print(list(state_dict.keys())[:10])  

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Clés manquantes  : {missing}")
    if unexpected:
        print(f"[WARN] Clés inattendues : {unexpected}")

    # ── Sauvegarder dans BentoML ──
    # model_name = args.model_name
    model_name = f"{args.model_name}:{run_name}_time_{int(time.time())}"
    tags = {
        "recall": recall,
        "f1": f1,
        "epsilon_final": epsilon_final,
        "trust_score": trust_score,
        "composite_score": metadata.get("composite_score", 0.0),
        "run_name": run_name,
        "architecture": metadata.get("architecture", "DPGRU"),
        "promoted_at": metadata.get("promoted_at", "unknown"),
    }

    print(f"\n[INFO] Sauvegarde dans BentoML store sous '{model_name}' ...")
    saved = bentoml.pytorch.save_model(
        model_name,
        model,
        metadata=tags,
    )

    print(f"[OK] Modèle sauvegardé dans BentoML store")
    print(f"     Tag   : {saved.tag}")
    print(f"     Path  : {saved.path}")
    print("\n[INFO] Pour servir le modele :")
    print("     bentoml serve serving/bentoml_service.py:FraudDetectionService --port 3001")


if __name__ == "__main__":
    main()
