"""
fl_hyperparameter_search.py
═══════════════════════════
Script d'analyse automatique du projet FL pour générer les 12 meilleures
combinaisons de paramètres d'entraînement.

Analyse :
  1. Données réelles de chaque nœud (pos_weight, taille dataset, fraudes)
  2. Architecture du modèle DPGRU (paramètres, complexité)
  3. Simulation de la montée de epsilon selon Opacus
  4. Estimation du drift client selon les epochs locaux
  5. Estimation du temps GPU sur RTX 4060 / 3050 Ti
  6. Score composite → top 12 combinaisons

Usage :
  python fl_hyperparameter_search.py
  python fl_hyperparameter_search.py --data_root data/ --budget_hours 8 --gpu rtx4060
"""

import os
import sys
import json
import argparse
import itertools
import math
from pathlib import Path
from datetime import datetime

import numpy as np

# ── Configuration du projet ──────────────────────────────────────────────────

MODEL_CONFIG = {
    "input_dim"   : 26,
    "hidden_dim"  : 128,
    "num_layers"  : 2,
    "dropout"     : 0.2,
    "seq_len"     : 5,
    "n_params"    : 167_297,   # paramètres DPGRU calculés
}

# Vitesse GPU estimée (secondes par batch de 128 sur DPGRU avec Opacus)
GPU_SPEED = {
    "rtx4060"  : 0.012,   # RTX 4060 8GB — plus rapide
    "rtx3050ti": 0.022,   # RTX 3050 Ti 4GB
    "cpu"      : 0.180,   # CPU fallback
}

# ── Grille de recherche ───────────────────────────────────────────────────────

SEARCH_GRID = {
    "fl_rounds"      : [15, 20, 25, 30],
    "local_epochs"   : [3, 5, 8],
    "dp_noise"       : [0.8, 0.9, 1.0, 1.2],
    "batch_size"     : [128, 256],
    "lr"             : [0.0005],    # fixé — validé par les runs précédents
}

# ── Analyse des données par nœud ─────────────────────────────────────────────

def analyze_node_data(data_root: Path, node_id: int) -> dict:
    """Lit les tenseurs d'un nœud et calcule ses statistiques réelles."""
    tensor_dir = data_root / f"node_{node_id}" / "tensors"

    if not tensor_dir.exists():
        return None

    y_files = sorted(tensor_dir.glob("y_batch_*.npy"))
    if not y_files:
        return None

    all_y = np.concatenate([np.load(f) for f in y_files])
    n_total  = len(all_y)
    n_fraud  = int(all_y.sum())
    n_legit  = n_total - n_fraud
    pos_weight = n_legit / max(n_fraud, 1)

    x_files = sorted(tensor_dir.glob("X_batch_*.npy"))
    n_tensors = len(x_files)

    # Taille du train set (80%)
    n_train_approx = int(n_total * 0.80)
    n_batches_per_epoch = {
        128: math.ceil(n_train_approx / 128),
        256: math.ceil(n_train_approx / 256),
    }

    return {
        "node_id"           : node_id,
        "n_total"           : n_total,
        "n_fraud"           : n_fraud,
        "n_legit"           : n_legit,
        "fraud_rate"        : n_fraud / n_total,
        "pos_weight"        : round(pos_weight, 1),
        "n_train_approx"    : n_train_approx,
        "n_batches_128"     : n_batches_per_epoch[128],
        "n_batches_256"     : n_batches_per_epoch[256],
        "n_tensor_files"    : n_tensors,
    }


# ── Simulation epsilon (formule Opacus RDP) ───────────────────────────────────

def estimate_epsilon(
    noise_multiplier: float,
    n_epochs: int,
    n_rounds: int,
    n_samples: int,
    batch_size: int,
    delta: float = 1e-5,
) -> float:
    """
    Estimation du budget DP epsilon via la borne RDP d'Opacus.
    Formule simplifiée basée sur le sampling Poisson :
      ε ≈ q * sqrt(2 * T * log(1/delta)) / sigma
    où q = batch/N (sampling rate), T = epochs * rounds (steps totaux)
    """
    q       = batch_size / max(n_samples, 1)
    T       = n_epochs * n_rounds
    sigma   = noise_multiplier
    epsilon = q * math.sqrt(2 * T * math.log(1 / delta)) / sigma
    return round(epsilon, 4)


def round_epsilon_exceeds(
    noise_multiplier: float,
    local_epochs: int,
    n_samples: int,
    batch_size: int,
    target_epsilon: float = 1.0,
    delta: float = 1e-5,
    max_rounds: int = 100,
) -> int:
    """Retourne le round auquel ε dépasse target_epsilon. -1 si jamais."""
    for r in range(1, max_rounds + 1):
        eps = estimate_epsilon(
            noise_multiplier, local_epochs, r, n_samples, batch_size, delta
        )
        if eps > target_epsilon:
            return r
    return -1


# ── Estimation du drift ───────────────────────────────────────────────────────

def estimate_drift_score(local_epochs: int, lr: float, n_clients: int) -> float:
    """
    Score de drift [0, 1] — plus proche de 1 = drift élevé = mauvais pour FL.
    Le drift augmente avec les epochs locaux, le LR et le nombre de clients.
    Basé sur l'analyse de McMahan et al. (FedAvg paper) :
      drift ∝ local_epochs * lr * sqrt(n_clients)
    """
    raw   = local_epochs * lr * math.sqrt(n_clients)
    score = 1 - math.exp(-raw * 10)   # normalise en [0, 1]
    return round(score, 4)


# ── Estimation du temps d'entraînement ───────────────────────────────────────

def estimate_duration_hours(
    fl_rounds: int,
    local_epochs: int,
    n_batches_per_epoch: int,
    n_nodes: int,
    gpu: str,
) -> float:
    """
    Estimation du temps total en heures.
    Les nœuds tournent en parallèle → temps = max(temps d'un nœud) × rounds
    + overhead communication (~5s par round par nœud)
    """
    sec_per_batch   = GPU_SPEED.get(gpu, GPU_SPEED["rtx4060"])
    # Opacus ajoute ~3x de overhead sur le temps de batch standard
    sec_per_batch_dp = sec_per_batch * 3.0

    secs_per_round = (local_epochs * n_batches_per_epoch * sec_per_batch_dp
                      + n_nodes * 5)   # overhead communication
    total_secs = fl_rounds * secs_per_round
    return round(total_secs / 3600, 2)


# ── Score composite ───────────────────────────────────────────────────────────

def compute_score(
    epsilon_final    : float,
    epsilon_exhaustion_round: int,
    drift_score      : float,
    duration_hours   : float,
    fl_rounds        : int,
    budget_hours     : float,
    pos_weight_mean  : float,
) -> dict:
    """
    Score composite [0, 1] — plus élevé = meilleure combinaison.

    Composantes :
      - privacy_score   : ε final < 1.0 fortement favorisé
      - endurance_score : le modèle ne s'épuise pas avant les 2/3 des rounds
      - antidrift_score : drift faible = FedAvg efficace
      - time_score      : durée raisonnable dans le budget
      - balance_score   : pos_weight proche du ratio réel
    """

    # Privacy — pénalise si ε > 1 et récompense si très bas
    if epsilon_final <= 1.0:
        privacy_score = 1.0 - (epsilon_final / 1.0) * 0.3
    else:
        privacy_score = max(0.0, 1.0 - (epsilon_final - 1.0) * 0.5)

    # Endurance — on veut que l'épuisement arrive après les 2/3 des rounds
    threshold_round = fl_rounds * 2 / 3
    if epsilon_exhaustion_round < 0:
        endurance_score = 1.0   # jamais épuisé
    elif epsilon_exhaustion_round >= fl_rounds:
        endurance_score = 0.95
    elif epsilon_exhaustion_round >= threshold_round:
        endurance_score = 0.7
    else:
        endurance_score = max(0.1, epsilon_exhaustion_round / fl_rounds)

    # Anti-drift
    antidrift_score = 1.0 - drift_score

    # Temps
    if duration_hours <= budget_hours:
        time_score = 1.0 - (duration_hours / budget_hours) * 0.2
    else:
        time_score = max(0.0, 1.0 - (duration_hours - budget_hours) / budget_hours)

    # Équilibre pos_weight — favorise les valeurs autour de 100-167
    if 80 <= pos_weight_mean <= 200:
        balance_score = 1.0
    else:
        balance_score = 0.7

    # Composite pondéré
    total = (
        privacy_score    * 0.30 +
        endurance_score  * 0.25 +
        antidrift_score  * 0.25 +
        time_score       * 0.15 +
        balance_score    * 0.05
    )

    return {
        "total"           : round(total, 4),
        "privacy"         : round(privacy_score, 3),
        "endurance"       : round(endurance_score, 3),
        "antidrift"       : round(antidrift_score, 3),
        "time"            : round(time_score, 3),
        "balance"         : round(balance_score, 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FL Hyperparameter Search")
    parser.add_argument("--data_root",     default="data",     type=str)
    parser.add_argument("--budget_hours",  default=10.0,        type=float,
                        help="Budget temps max pour un run (heures)")
    parser.add_argument("--gpu",           default="rtx4060",
                        choices=["rtx4060", "rtx3050ti", "cpu"])
    parser.add_argument("--n_nodes",       default=4,          type=int)
    parser.add_argument("--top_k",         default=12,         type=int)
    parser.add_argument("--target_epsilon",default=1.0,        type=float)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    print("=" * 65)
    print("  FL Hyperparameter Search — Analyse automatique du projet")
    print("=" * 65)
    print(f"  GPU         : {args.gpu}")
    print(f"  Budget max  : {args.budget_hours}h")
    print(f"  Nœuds       : {args.n_nodes}")
    print(f"  ε target    : {args.target_epsilon}")
    print(f"  Modèle      : DPGRU hidden={MODEL_CONFIG['hidden_dim']}"
          f"  layers={MODEL_CONFIG['num_layers']}"
          f"  params={MODEL_CONFIG['n_params']:,}")
    print()

    # ── 1. Analyse des données ───────────────────────────────────────────────
    print("── Analyse des nœuds ─────────────────────────────────────────")
    nodes = []
    for node_id in range(1, args.n_nodes + 1):
        info = analyze_node_data(data_root, node_id)
        if info:
            nodes.append(info)
            print(f"  Node {node_id} : {info['n_total']:>8,} txns | "
                  f"{info['n_fraud']:>5,} fraudes | "
                  f"pos_weight={info['pos_weight']:.1f} | "
                  f"train≈{info['n_train_approx']:,}")
        else:
            print(f"  Node {node_id} : tenseurs non trouvés dans {data_root}/node_{node_id}/tensors/")

    if not nodes:
        print("\n  Aucun tenseur trouvé. Assurez-vous d'avoir généré les tenseurs.")
        print("  python fast_generate_tensors.py")
        sys.exit(1)

    # Statistiques agrégées
    n_samples_min  = min(n["n_train_approx"] for n in nodes)
    n_samples_mean = int(sum(n["n_train_approx"] for n in nodes) / len(nodes))
    pos_weights    = [n["pos_weight"] for n in nodes]
    pos_weight_mean = sum(pos_weights) / len(pos_weights)

    print(f"\n  Résumé : {len(nodes)} nœuds | "
          f"train min={n_samples_min:,} | "
          f"pos_weight moyen={pos_weight_mean:.1f}")

    # ── 2. Génération et évaluation de toutes les combinaisons ───────────────
    print("\n── Évaluation des combinaisons ───────────────────────────────")

    keys   = list(SEARCH_GRID.keys())
    values = list(SEARCH_GRID.values())
    combos = list(itertools.product(*values))
    print(f"  {len(combos)} combinaisons à évaluer...")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))

        n_batches = (nodes[0]["n_batches_128"]
                     if params["batch_size"] == 128
                     else nodes[0]["n_batches_256"])

        # Calcul epsilon sur le nœud le plus petit (épuise le plus vite)
        eps_final = estimate_epsilon(
            noise_multiplier = params["dp_noise"],
            n_epochs         = params["local_epochs"],
            n_rounds         = params["fl_rounds"],
            n_samples        = n_samples_min,
            batch_size       = params["batch_size"],
        )

        eps_round = round_epsilon_exceeds(
            noise_multiplier = params["dp_noise"],
            local_epochs     = params["local_epochs"],
            n_samples        = n_samples_min,
            batch_size       = params["batch_size"],
            target_epsilon   = args.target_epsilon,
        )

        drift = estimate_drift_score(
            params["local_epochs"], params["lr"], args.n_nodes
        )

        duration = estimate_duration_hours(
            params["fl_rounds"],
            params["local_epochs"],
            n_batches,
            args.n_nodes,
            args.gpu,
        )

        score = compute_score(
            epsilon_final           = eps_final,
            epsilon_exhaustion_round= eps_round,
            drift_score             = drift,
            duration_hours          = duration,
            fl_rounds               = params["fl_rounds"],
            budget_hours            = args.budget_hours,
            pos_weight_mean         = pos_weight_mean,
        )

        results.append({
            "params"          : params,
            "eps_final"       : eps_final,
            "eps_round"       : eps_round,
            "drift"           : drift,
            "duration_hours"  : duration,
            "score"           : score,
            "pos_weights"     : {f"node_{n['node_id']}": n["pos_weight"]
                                 for n in nodes},
        })

    # ── 3. Top K ──────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r["score"]["total"], reverse=True)
    top = results[:args.top_k]

    print(f"\n── Top {args.top_k} combinaisons ─────────────────────────────────────")
    print()
    print(f"  {'#':>2}  {'Rounds':>6}  {'Epochs':>6}  {'Noise':>6}  "
          f"{'Batch':>6}  {'ε final':>8}  {'ε round':>8}  "
          f"{'Drift':>6}  {'Durée':>7}  {'Score':>6}")
    print("  " + "─" * 73)

    for i, r in enumerate(top, 1):
        p = r["params"]
        eps_round_str = str(r["eps_round"]) if r["eps_round"] > 0 else "jamais"
        marker = " ★" if i <= 3 else ""
        print(f"  {i:>2}  {p['fl_rounds']:>6}  {p['local_epochs']:>6}  "
              f"{p['dp_noise']:>6.2f}  {p['batch_size']:>6}  "
              f"{r['eps_final']:>8.3f}  {eps_round_str:>8}  "
              f"{r['drift']:>6.3f}  {r['duration_hours']:>6.1f}h  "
              f"{r['score']['total']:>6.3f}{marker}")

    # ── 4. Détail des 3 meilleures ────────────────────────────────────────────
    print()
    print("── Détail des 3 meilleures combinaisons ──────────────────────")
    for i, r in enumerate(top[:3], 1):
        p  = r["params"]
        sc = r["score"]
        eps_round_str = (f"round {r['eps_round']}"
                         if r["eps_round"] > 0 else "jamais épuisé")
        print(f"""
  [{i}] Score global : {sc['total']:.4f}
       Rounds={p['fl_rounds']}  Epochs={p['local_epochs']}  Noise={p['dp_noise']}  Batch={p['batch_size']}
       LR={p['lr']}
       ε final estimé   : {r['eps_final']:.4f}  (épuisement : {eps_round_str})
       Drift             : {r['drift']:.3f}  ({_drift_label(r['drift'])})
       Durée estimée     : {r['duration_hours']:.1f}h
       pos_weight / nœud : {r['pos_weights']}
       Scores détaillés  :
         privacy={sc['privacy']:.3f}  endurance={sc['endurance']:.3f}
         antidrift={sc['antidrift']:.3f}  time={sc['time']:.3f}""")

    # ── 5. Sauvegarde JSON ────────────────────────────────────────────────────
    output = {
        "generated_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu"          : args.gpu,
        "budget_hours" : args.budget_hours,
        "n_nodes"      : args.n_nodes,
        "target_epsilon": args.target_epsilon,
        "model"        : MODEL_CONFIG,
        "nodes"        : nodes,
        "top_combinations": [
            {
                "rank"           : i,
                "params"         : r["params"],
                "eps_final"      : r["eps_final"],
                "eps_round"      : r["eps_round"],
                "drift"          : r["drift"],
                "duration_hours" : r["duration_hours"],
                "score"          : r["score"],
                "pos_weights"    : r["pos_weights"],
            }
            for i, r in enumerate(top, 1)
        ],
    }

    out_path = Path("fl_search_results.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # ── 6. Script ps1 pour la meilleure combinaison ───────────────────────────
    best = top[0]["params"]
    ps1_path = Path("fl_best_run.ps1")
    ps1_content = f'''# fl_best_run.ps1 — Meilleure combinaison FL
# Generé automatiquement par fl_hyperparameter_search.py
# Score : {top[0]["score"]["total"]:.4f}  |  ε estimé : {top[0]["eps_final"]:.4f}

$env:FL_ROUNDS       = "{best['fl_rounds']}"
$env:FL_LOCAL_EPOCHS = "{best['local_epochs']}"
$env:FLOWER_PORT     = "8090"
$env:FL_MIN_CLIENTS  = "{args.n_nodes}"
$env:FL_PATIENCE     = "5"
$env:FL_RESUME       = "false"
$env:DP_NOISE        = "{best['dp_noise']}"
$env:DP_EPSILON_TARGET = "{args.target_epsilon}"
$env:FL_LR           = "{best['lr']}"
$env:FL_BATCH_SIZE   = "{best['batch_size']}"
# pos_weight calcule automatiquement par chaque client depuis ses donnees
'''
    ps1_path.write_text(ps1_content, encoding="utf-8")

    print(f"\n  Résultats sauvegardés → {out_path}")
    print(f"  Script ps1 meilleure config → {ps1_path}")
    print()
    print("=" * 65)


def _drift_label(drift: float) -> str:
    if drift < 0.3:   return "faible — FedAvg efficace"
    if drift < 0.6:   return "modéré"
    return "élevé — risque de stagnation"


if __name__ == "__main__":
    main()
