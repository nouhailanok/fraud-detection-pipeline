"""
verify_model.py — Vérifie que le modèle best_model/model.pt == BentoML store
Usage : python mlops/serving/verify_model.py --tag fraud_dpgru_v1:2026-05-01_00-18-13_mohamed_central_time_1777671357
"""
import os
import sys, argparse, torch, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.fraud_rnn import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="BentoML model tag")
    parser.add_argument("--best-model", default="mlops/model_registry/best_model/model.pt")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  🔍 Vérification modèle BentoML vs best_model")
    print(f"{'═'*60}")

    # ── 1. Charger depuis best_model/ ────────────────────────────
    best_path = Path(args.best_model)
    if not best_path.exists():
        # Essayer .npz
        best_path = best_path.with_suffix('.npz')
    
    if not best_path.exists():
        print(f"❌ best_model introuvable : {args.best_model}")
        sys.exit(1)

    print(f"\n📂 Chargement best_model : {best_path}")
    
    model_local = build_model(use_dpgru=False)
    
    if best_path.suffix == '.pt':
        ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        model_local.load_state_dict(sd, strict=False)
    else:
        data = dict(np.load(best_path, allow_pickle=True))
        params_sorted = sorted(data.items(), key=lambda x: int(x[0].split("_")[1]))
        with torch.no_grad():
            for (_, val), param in zip(params_sorted, model_local.parameters()):
                param.copy_(torch.from_numpy(np.array(val)).float())

    params_local = {n: p.detach().clone() for n, p in model_local.named_parameters()}
    print(f"  ✅ best_model chargé — {len(params_local)} paramètres")

    # ── 2. Charger depuis BentoML store ──────────────────────────
    print(f"\n📦 Chargement BentoML : {args.tag}")
    try:
        import bentoml
        from torch.nn import GRU, Linear, Dropout, Sequential, ReLU
        from models.fraud_rnn import FraudRNN

        torch.serialization.add_safe_globals([FraudRNN, GRU, Linear, Dropout, Sequential, ReLU])
        model_ref = bentoml.models.get(args.tag)
        try:
            model_bento = model_ref.load_model(weights_only=False)
        except TypeError:
            model_bento = model_ref.load_model()
        
        params_bento = {n: p.detach().clone() for n, p in model_bento.named_parameters()}
        print(f"  ✅ BentoML chargé — {len(params_bento)} paramètres")
    except Exception as e:
        print(f"  ❌ Erreur BentoML : {e}")
        sys.exit(1)

    # ── 3. Comparer les poids ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  🔬 Comparaison des poids...")
    
    all_match = True
    max_diff  = 0.0

    for name in params_local:
        if name not in params_bento:
            print(f"  ❌ Clé absente dans BentoML : {name}")
            all_match = False
            continue

        p_local = params_local[name]
        p_bento = params_bento[name]

        if p_local.shape != p_bento.shape:
            print(f"  ❌ Shape mismatch {name}: {p_local.shape} vs {p_bento.shape}")
            all_match = False
            continue

        diff = (p_local - p_bento).abs().max().item()
        max_diff = max(max_diff, diff)

        if diff > 1e-5:
            print(f"  ❌ {name} — diff max = {diff:.6f}")
            all_match = False

    print(f"\n{'═'*60}")
    if all_match:
        print(f"  ✅ IDENTIQUES — diff max = {max_diff:.2e}")
        print(f"     best_model == BentoML store ✅")
    else:
        print(f"  ❌ DIFFÉRENTS — les modèles ne correspondent pas")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()