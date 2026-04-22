"""
server.py — Serveur Flower pour le Federated Learning
═══════════════════════════════════════════════════════
Changements par rapport à la version originale :

  1. Système de log des métriques par round
       → Sauvegarde un fichier JSON  : logs/fl_metrics.json
       → Sauvegarde un fichier CSV   : logs/fl_metrics.csv
       → Affiche un rapport console  à chaque round

  2. Métriques agrégées après chaque FedAvg :
       Évaluation  → accuracy, loss, recall, precision, f1 (pondérés par nœud)
       Training    → train_loss, epsilon moyen/max par round
       Privacy     → budget ε cumulé, statut (OK / WARNING)

  3. Early stopping FL
       → Surveille le F1-score agrégé après chaque round
       → Arrêt automatique si pas d'amélioration pendant FL_PATIENCE rounds (défaut: 5)
       → Sauvegarde logs/fl/checkpoints/early_stop.txt avec le meilleur round

  4. Rapport final à la fin de tous les rounds
       → Meilleur round (accuracy max)
       → Évolution de ε round par round
       → Fichier de résumé : logs/fl_summary.txt
"""

import os
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil

import numpy as np
import flwr as fl


# ============================================================================
# ⚙️  CONFIGURATION
# ============================================================================

LOGS_DIR       = Path(os.getenv("FL_LOGS_DIR",       "logs/fl"))
CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "logs/fl/checkpoints"))
FL_PATIENCE    = int(os.getenv("FL_PATIENCE",         "5"))   # rounds sans amélioration du F1 avant arrêt



# ============================================================================
# CRÉATION DU DOSSIER RUN (server.py)
# ============================================================================

def create_run_directory() -> tuple[Path, Path]:
    """Crée un dossier de run (logs + checkpoints) si aucun n'est fourni.

    Priorité user:
      - NOM_USER (demandé)
      - FL_USER (compat)

    Base dir configurable via FL_RUNS_BASE_DIR (défaut: logs/runs)
    """
    user = os.getenv("NOM_USER") or os.getenv("FL_USER") or "unknown_user"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{user}"

    base_dir = Path(os.getenv("FL_RUNS_BASE_DIR", "logs/runs"))
    run_dir = base_dir / run_name

    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, checkpoints_dir


def snapshot_server_env(logs_dir: Path) -> None:
    """Sauvegarde un snapshot des variables d'env liées au run."""
    keys = [
        "NOM_USER", "FL_USER",
        "FL_ROUNDS", "FL_MIN_CLIENTS", "FL_LOCAL_EPOCHS",
        "FLOWER_PORT", "FLOWER_TLS_REQUIRE_CLIENT_CERT",
        "FL_PATIENCE",
        "FL_RESUME", "FL_RESUME_FROM", "FL_CHECKPOINT_PATH",
        "FL_LOGS_DIR", "FL_CHECKPOINT_DIR",
        "PYTHONPATH",
    ]
    env_path = logs_dir / "server_env.txt"
    lines = []
    for k in keys:
        v = os.getenv(k, "")
        lines.append(f"{k}={v}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



# ============================================================================
# 💾 CHECKPOINT — sauvegarde et reprise du modèle global
# ============================================================================

def save_checkpoint(server_round: int, parameters) -> None:
    """Sauvegarde les poids globaux après chaque round."""
    if parameters is None:
        print(f"  ⚠️  Round {server_round} : paramètres None — checkpoint ignoré")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Convertir les paramètres Flower en numpy arrays
        import flwr.common
        ndarrays = flwr.common.parameters_to_ndarrays(parameters)

        # Sauvegarde du round courant
        ckpt_path = CHECKPOINT_DIR / f"global_model_round_{server_round:03d}.npz"
        arrays    = {f"param_{i}": arr for i, arr in enumerate(ndarrays)}
        np.savez(str(ckpt_path), **arrays)

        # Sauvegarde du dernier round
        last_path = CHECKPOINT_DIR / "global_model_last.npz"
        np.savez(str(last_path), **arrays)

        # Numéro de round
        meta_path = CHECKPOINT_DIR / "last_round.txt"
        meta_path.write_text(str(server_round), encoding="utf-8")

        print(f"  💾 Checkpoint sauvegardé → round {server_round} ({ckpt_path.name})")

    except Exception as e:
        print(f"  ⚠️  Erreur sauvegarde checkpoint round {server_round} : {e}")


def _infer_last_round_from_dir(checkpoint_dir: Path) -> int:
    meta_path = checkpoint_dir / "last_round.txt"
    if meta_path.exists():
        try:
            return int(meta_path.read_text(encoding="utf-8").strip())
        except Exception:
            pass

    # fallback: max global_model_round_XXX.npz
    best = 0
    for p in checkpoint_dir.glob("global_model_round_*.npz"):
        stem = p.stem
        try:
            n = int(stem.split("_")[-1])
            best = max(best, n)
        except Exception:
            continue
    return best


def load_checkpoint_from(path_or_dir: Optional[str]) -> tuple[Optional[list], int]:
    """Charge un checkpoint depuis un chemin (dir ou fichier .npz).

    - Si dir: utilise global_model_last.npz
    - Si fichier .npz: charge ce fichier
    - last_round: last_round.txt si dispo, sinon inféré
    """
    if not path_or_dir:
        return None, 0

    p = Path(path_or_dir)
    if p.is_dir():
        checkpoint_dir = p
        npz_path = checkpoint_dir / "global_model_last.npz"
    else:
        npz_path = p
        checkpoint_dir = p.parent

    if not npz_path.exists():
        return None, 0

    try:
        data = np.load(str(npz_path))
        parameters = [data[k] for k in sorted(data.files)]

        last_round = _infer_last_round_from_dir(checkpoint_dir)
        # si on charge explicitement global_model_round_XXX.npz
        if last_round == 0 and "global_model_round_" in npz_path.name:
            try:
                last_round = int(npz_path.stem.split("_")[-1])
            except Exception:
                pass

        print(f"  ✅ Checkpoint chargé → reprise depuis le round {last_round}")
        return parameters, last_round
    except Exception as e:
        print(f"  ⚠️  Erreur chargement checkpoint : {e}")
        return None, 0


def load_checkpoint() -> tuple[Optional[list], int]:
    """Compat: charge depuis CHECKPOINT_DIR (dossier courant du run)."""
    return load_checkpoint_from(str(CHECKPOINT_DIR))


# ============================================================================
# 📁 LOGGER — sauvegarde JSON + CSV
# ============================================================================

class MetricsLogger:
    """
    Centralise la sauvegarde des métriques FL à chaque round.

    Fichiers produits :
      logs/fl/fl_metrics.json  → historique complet (tous les rounds)
      logs/fl/fl_metrics.csv   → format tableur (un round par ligne)
      logs/fl/fl_summary.txt   → rapport final lisible
    """

    def __init__(self, logs_dir: Path):
        self.logs_dir   = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.history    : list = []
        self.start_time = time.time()

        self.json_path    = logs_dir / "fl_metrics.json"
        self.csv_path     = logs_dir / "fl_metrics.csv"
        self.summary_path = logs_dir / "fl_summary.txt"

        # Initialise le CSV avec les en-têtes
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writeheader()

        print(f"📁 Logs FL → {self.logs_dir.resolve()}")

    def _csv_fields(self) -> list:
        return [
            "round", "timestamp",
            "accuracy", "loss",
            "recall", "precision", "f1",
            "n_clients_eval", "n_examples_eval",
            "train_loss", "n_clients_fit",
            "avg_epsilon", "max_epsilon", "epsilon_status",
            "elapsed_sec",
        ]

    def log_round(self, round_num: int, eval_metrics: dict, fit_metrics: dict) -> None:
        """Enregistre toutes les métriques d'un round."""
        elapsed  = round(time.time() - self.start_time, 1)
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Statut privacy
        max_eps  = fit_metrics.get("max_epsilon", 0.0)
        eps_ok   = max_eps <= 1.0

        record = {
            "round"           : round_num,
            "timestamp"       : ts,
            # Métriques d'évaluation (après FedAvg)
            "accuracy"        : round(eval_metrics.get("accuracy",  0.0), 4),
            "loss"            : round(eval_metrics.get("loss",      0.0), 4),
            "recall"          : round(eval_metrics.get("recall",    0.0), 4),
            "precision"       : round(eval_metrics.get("precision", 0.0), 4),
            "f1"              : round(eval_metrics.get("f1",        0.0), 4),
            "n_clients_eval"  : eval_metrics.get("n_clients",    0),
            "n_examples_eval" : eval_metrics.get("n_examples",   0),
            # Métriques d'entraînement
            "train_loss"      : round(fit_metrics.get("train_loss",   0.0), 4),
            "n_clients_fit"   : fit_metrics.get("n_clients",      0),
            # Privacy
            "avg_epsilon"     : round(fit_metrics.get("avg_epsilon", 0.0), 4),
            "max_epsilon"     : round(max_eps, 4),
            "epsilon_status"  : "OK" if eps_ok else "WARNING",
            "elapsed_sec"     : elapsed,
        }

        self.history.append(record)

        # Sauvegarde JSON (réécriture complète à chaque round)
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        # Sauvegarde CSV (ajout de la ligne)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writerow(record)

    def print_round_report(self, round_num: int, total_rounds: int) -> None:
        """Affiche le rapport console du round courant."""
        if not self.history:
            return
        r = self.history[-1]

        eps_icon = "✅" if r["epsilon_status"] == "OK" else "⚠️"

        print(f"\n{'─'*58}")
        print(f"  📊  Round {round_num}/{total_rounds}  —  {r['timestamp']}")
        print(f"{'─'*58}")
        print(f"  Évaluation ({r['n_clients_eval']} nœuds · {r['n_examples_eval']:,} txns)")
        print(f"    Accuracy  : {r['accuracy']:.4f}")
        print(f"    Loss      : {r['loss']:.4f}")
        print(f"    Recall    : {r['recall']:.4f}")
        print(f"    Precision : {r['precision']:.4f}")
        print(f"    F1-score  : {r['f1']:.4f}")
        print(f"  Entraînement ({r['n_clients_fit']} nœuds)")
        print(f"    Train Loss : {r['train_loss']:.4f}")
        print(f"  Privacy {eps_icon}")
        print(f"    Avg ε  : {r['avg_epsilon']:.4f}")
        print(f"    Max ε  : {r['max_epsilon']:.4f}  [{r['epsilon_status']}]")
        print(f"  Temps écoulé : {r['elapsed_sec']}s")
        print(f"{'─'*58}")

    def save_summary(self, total_rounds: int) -> None:
        """Écrit le rapport final après tous les rounds."""
        if not self.history:
            return
        # j'ai changé la clé de "accuracy" à "f1" pour le meilleur round
        best_round = max(self.history, key=lambda r: r["f1"])
        total_time = round(time.time() - self.start_time, 1)

        lines = [
            "=" * 58,
            "  RAPPORT FINAL — Federated Learning",
            "=" * 58,
            f"  Rounds effectués  : {len(self.history)} / {total_rounds}",
            f"  Temps total       : {total_time}s",
            f"  Nœuds (min)       : {self.history[0].get('n_clients_eval', '?')}",
            "",
            "  Meilleur round :",
            f"    Round     : {best_round['round']}",
            f"    Accuracy  : {best_round['accuracy']:.4f}",
            f"    Recall    : {best_round['recall']:.4f}",
            f"    Precision : {best_round['precision']:.4f}",
            f"    F1-score  : {best_round['f1']:.4f}",
            f"    Loss      : {best_round['loss']:.4f}",
            "",
            "  Évolution ε par round :",
        ]

        for r in self.history:
            icon = "✅" if r["epsilon_status"] == "OK" else "⚠️"
            lines.append(
                f"    Round {r['round']:>2} : avg={r['avg_epsilon']:.4f}"
                f"  max={r['max_epsilon']:.4f}  {icon}"
            )

        lines += [
            "",
            "  Fichiers de log :",
            f"    JSON    : {self.json_path}",
            f"    CSV     : {self.csv_path}",
            "=" * 58,
        ]

        summary_text = "\n".join(lines)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        print(f"\n{summary_text}")
        print(f"\n📄 Résumé sauvegardé → {self.summary_path}")


# ============================================================================
# 📊 FONCTIONS D'AGRÉGATION DES MÉTRIQUES
# ============================================================================

# Instance globale du logger (initialisée dans main())
_logger: Optional[MetricsLogger] = None
_total_rounds: int = 0


def aggregate_eval_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Agrège les métriques d'évaluation de tous les nœuds (moyenne pondérée).

    Métriques reçues depuis client.evaluate() :
      - accuracy  : taux de bonnes prédictions
      - loss      : BCEWithLogitsLoss moyenne
      - recall    : taux de fraudes détectées      ← clé pour la fraude
      - precision : taux de précision sur fraudes  ← clé pour les FA
      - f1        : F1-score fraude

    Note : le client.py actuel ne renvoie que accuracy et loss.
    Les autres métriques seront ajoutées quand client.py sera mis à jour.
    """
    if not metrics:
        return {}

    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    def wavg(key: str) -> float:
        return sum(n * m.get(key, 0.0) for n, m in metrics) / total_examples

    aggregated = {
        "accuracy"  : wavg("accuracy"),
        "loss"      : wavg("loss"),
        "recall"    : wavg("recall"),
        "precision" : wavg("precision"),
        "f1"        : wavg("f1"),
        "n_clients" : len(metrics),
        "n_examples": total_examples,
    }

    return aggregated


def aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Agrège les métriques d'entraînement de tous les nœuds.

    Métriques reçues depuis client.fit() :
      - epsilon    : budget DP consommé par ce nœud
      - train_loss : loss d'entraînement local
    """
    if not metrics:
        return {}

    epsilons    = [m["epsilon"]    for _, m in metrics if "epsilon"    in m]
    train_losses= [m["train_loss"] for _, m in metrics if "train_loss" in m]

    aggregated = {
        "n_clients"  : len(metrics),
        "avg_epsilon": sum(epsilons) / len(epsilons)       if epsilons     else 0.0,
        "max_epsilon": max(epsilons)                        if epsilons     else 0.0,
        "train_loss" : sum(train_losses) / len(train_losses) if train_losses else 0.0,
    }

    # Affichage privacy
    if epsilons:
        eps_ok = aggregated["max_epsilon"] <= 1.0
        icon   = "✅" if eps_ok else "⚠️"
        print(f"\n🔐 Privacy {icon}  avg_ε={aggregated['avg_epsilon']:.4f}"
              f"  max_ε={aggregated['max_epsilon']:.4f}")

    return aggregated


# ============================================================================
# 🧠 STRATEGY CUSTOM — FedAvg + logging par round
# ============================================================================

class FedAvgWithLogging(fl.server.strategy.FedAvg):
    """FedAvg + logging.

    round_offset permet de reprendre un entraînement et de garder les numéros
    de rounds cohérents (ex: reprise après round 9 → prochain round global=10).
    """

    def __init__(self, total_rounds: int, patience: int = 5, round_offset: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.total_rounds      = total_rounds
        self.round_offset      = int(round_offset)
        self.current_round     = 0
        # ── Early stopping ───────────────────────────────────────────────────
        self.patience          = patience
        self.best_f1           = 0.0
        self.best_round        = 0
        self.rounds_no_improve = 0

    def aggregate_evaluate(self, server_round, results, failures):
        """Appelé après l'évaluation de chaque round."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        global_round = int(server_round) + self.round_offset
        self.current_round = global_round

        # Extraction des métriques depuis les résultats bruts
        eval_metrics_raw = [
            (res.num_examples, res.metrics)
            for _, res in results
            if res is not None
        ]
        eval_agg = aggregate_eval_metrics(eval_metrics_raw)

        # Récupère les métriques fit du round courant (déjà dans l'historique)
        fit_agg = {}
        if _logger and _logger.history:
            last = _logger.history[-1] if _logger.history else {}
            fit_agg = {
                "avg_epsilon": last.get("avg_epsilon", 0.0),
                "max_epsilon": last.get("max_epsilon", 0.0),
                "train_loss" : last.get("train_loss",  0.0),
                "n_clients"  : last.get("n_clients_fit", 0),
            }

        # Log du round
        if _logger:
            _logger.log_round(global_round, eval_agg, fit_agg)
            _logger.print_round_report(global_round, self.total_rounds)

        # ── Early stopping sur F1 agrégé ──────────────────────────────────────
        current_f1 = eval_agg.get("f1", 0.0)
        if current_f1 > self.best_f1 + 1e-4:
            self.best_f1           = current_f1
            self.best_round        = global_round
            self.rounds_no_improve = 0
            print(f"  ✅ Nouveau meilleur F1 global : {self.best_f1:.4f} (round {global_round})")

            # Alias "best" : copie le checkpoint du meilleur round
            try:
                import shutil
                src = CHECKPOINT_DIR / f"global_model_round_{global_round:03d}.npz"
                dst = CHECKPOINT_DIR / "global_model_best.npz"
                if src.exists():
                    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
            except Exception as e:
                print(f"  ⚠️  Impossible de sauvegarder global_model_best.npz : {e}")
        else:
            self.rounds_no_improve += 1
            print(f"  ⏳ F1 sans amélioration : {self.rounds_no_improve}/{self.patience}"
                  f"  (best={self.best_f1:.4f} @ round {self.best_round}"
                  f"  |  current={current_f1:.4f})")

            if self.rounds_no_improve >= self.patience:
                print(f"\n{'═'*58}")
                print(f"  🛑 EARLY STOPPING FL")
                print(f"  {self.patience} rounds consécutifs sans amélioration du F1")
                print(f"  Meilleur F1 : {self.best_f1:.4f} au round {self.best_round}")
                print(f"  Checkpoint  : {CHECKPOINT_DIR / f'global_model_round_{self.best_round:03d}.npz'}")
                print(f"{'═'*58}\n")
                # Sauvegarde un flag pour signaler l'arrêt
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                stop_path = CHECKPOINT_DIR / "early_stop.txt"
                stop_path.write_text(
                    f"Early stop au round {global_round}\n"
                    f"Meilleur F1 = {self.best_f1:.4f} au round {self.best_round}\n"
                    f"Meilleur checkpoint : global_model_round_{self.best_round:03d}.npz",
                    encoding="utf-8"
                )

        return aggregated

    def aggregate_fit(self, server_round, results, failures):
        """Appelé après l'entraînement de chaque round.

        Important: certains clients peuvent renvoyer num_examples=0 (ex: budget DP épuisé).
        Si TOUS les clients renvoient 0, FedAvg (Flower) divise par 0 → crash.

        Après modification:
          - on filtre les résultats à num_examples>0
          - si aucun client actif, on affiche un message final, on sauvegarde le meilleur checkpoint
            et on STOPPE proprement le serveur (exit 0)
        """

        global_round = int(server_round) + self.round_offset

        if not results:
            print(f"⚠️ Aucun résultat au round {global_round}")
            return None

        valid_results = []
        zero_results = []
        for client, fit_res in results:
            n = int(getattr(fit_res, "num_examples", 0) or 0)
            if n > 0:
                valid_results.append((client, fit_res))
            else:
                zero_results.append((client, fit_res))

        if not valid_results:  # tout le monde a 0
            exhausted = 0
            for _, r in zero_results:
                try:
                    if getattr(r, "metrics", {}).get("dp_exhausted") is True:
                        exhausted += 1
                except Exception:
                    pass

            # Prépare/assure un alias du meilleur checkpoint (si connu)
            best_alias = CHECKPOINT_DIR / "global_model_best.npz"
            try:
                import shutil
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                if self.best_round > 0:
                    src = CHECKPOINT_DIR / f"global_model_round_{self.best_round:03d}.npz"
                    if src.exists():
                        shutil.copyfile(src, best_alias)
                if not best_alias.exists():
                    last_src = CHECKPOINT_DIR / "global_model_last.npz"
                    if last_src.exists():
                        shutil.copyfile(last_src, best_alias)
            except Exception as e:
                print(f"  ⚠️  Sauvegarde alias best impossible : {e}")

            # Message final clair (pas de stacktrace)
            print(f"\n{'═'*58}")
            print("  🛑 ARRÊT FL — aucun client actif")
            print(f"  Raison : tous les clients ont num_examples=0 au round {global_round}")
            if exhausted:
                print(f"  Détails: budget DP épuisé sur {exhausted}/{len(zero_results)} clients")
            if self.best_round > 0:
                print(f"  Meilleur F1 connu : {self.best_f1:.4f} (round {self.best_round})")
                print(f"  Meilleur checkpoint : {CHECKPOINT_DIR / f'global_model_round_{self.best_round:03d}.npz'}")
            if best_alias.exists():
                print(f"  Alias best : {best_alias}")
            print(f"{'═'*58}\n")

            # Écrit un fichier "reason" pour audit
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            stop_path = CHECKPOINT_DIR / "dp_budget_exhausted.txt"
            stop_path.write_text(
                "Arrêt FL: tous les clients ont num_examples=0.\n"
                f"Round : {global_round}\n"
                f"Clients dp_exhausted (signalés) : {exhausted}/{len(zero_results)}\n"
                f"Best F1 connu : {self.best_f1:.6f} (round {self.best_round})\n"
                "Cause typique: DP_EPSILON_TARGET trop bas / DP_NOISE trop faible / trop de rounds / local_epochs trop haut.\n",
                encoding="utf-8",
            )

            # Sauvegarde le résumé même si on s'arrête avant la fin
            if _logger:
                try:
                    _logger.save_summary(self.total_rounds)
                except Exception as e:
                    print(f"  ⚠️  Impossible de sauver le résumé final : {e}")

            # STOP proprement (exit code 0)
            raise SystemExit(0)

        num_examples_total = sum(int(r.num_examples) for _, r in valid_results)
        if num_examples_total <= 0:
            print(f"🛑 Round {global_round}: num_examples_total<=0 après filtrage — skip")
            return None

        aggregated = super().aggregate_fit(server_round, valid_results, failures)

        # Log préliminaire des métriques fit (avant évaluation)
        fit_metrics_raw = [
            (res.num_examples, res.metrics)
            for _, res in valid_results
            if res is not None
        ]
        fit_agg = aggregate_fit_metrics(fit_metrics_raw)

        # Pré-enregistrement pour que aggregate_evaluate puisse le lire
        if _logger:
            _logger.history.append({
                "round"          : global_round,
                "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy"       : 0.0, "loss": 0.0,
                "recall"         : 0.0, "precision": 0.0, "f1": 0.0,
                "n_clients_eval" : 0,   "n_examples_eval": 0,
                "train_loss"     : fit_agg.get("train_loss",  0.0),
                "n_clients_fit"  : fit_agg.get("n_clients",   0),
                "avg_epsilon"    : fit_agg.get("avg_epsilon", 0.0),
                "max_epsilon"    : fit_agg.get("max_epsilon", 0.0),
                "epsilon_status" : "OK" if fit_agg.get("max_epsilon", 0) <= 1.0 else "WARNING",
                "elapsed_sec"    : round(time.time() - _logger.start_time, 1),
            })

        # 💾 Sauvegarde checkpoint après chaque round
        if aggregated is not None:
            parameters, _ = aggregated
            save_checkpoint(global_round, parameters)
        else:
            print(
                f"  ⚠️  Round {global_round} : agrégation échouée (0 résultats valides) — checkpoint non sauvegardé"
            )

        return aggregated


# ============================================================================
# 🔐 TLS
# ============================================================================

def _read_cert(path: Optional[str]) -> Optional[bytes]:
    if not path:
        return None
    cert_path = Path(path)
    if not cert_path.exists():
        return None
    return cert_path.read_bytes()


def _build_certificates() -> Optional[Tuple[bytes, bytes, bytes]]:
    ca_cert     = _read_cert(os.getenv("FLOWER_TLS_CA_CERT"))
    server_cert = _read_cert(os.getenv("FLOWER_TLS_SERVER_CERT"))
    server_key  = _read_cert(os.getenv("FLOWER_TLS_SERVER_KEY"))

    if server_cert is None or server_key is None:
        return None

    if os.getenv("FLOWER_TLS_REQUIRE_CLIENT_CERT", "false").lower() == "true":
        if ca_cert is None:
            raise RuntimeError(
                "FLOWER_TLS_REQUIRE_CLIENT_CERT=true mais FLOWER_TLS_CA_CERT est absent"
            )

    return (ca_cert, server_cert, server_key) if ca_cert is not None else None


# ============================================================================
# 🚀 MAIN
# ============================================================================

def main():
    global _logger, _total_rounds

    total_rounds_target = int(os.getenv("FL_ROUNDS", "5"))
    min_clients   = int(os.getenv("FL_MIN_CLIENTS",   "2"))
    local_epochs  = int(os.getenv("FL_LOCAL_EPOCHS",  "3"))
    port          = int(os.getenv("FLOWER_PORT",       "8080"))
    server_address = f"0.0.0.0:{port}"

    _total_rounds = total_rounds_target
    _logger       = MetricsLogger(LOGS_DIR)
    snapshot_server_env(LOGS_DIR)

    # ── Reprise depuis checkpoint (optionnelle) ───────────────────────────────
    resume_from = (os.getenv("FL_RESUME_FROM") or os.getenv("FL_CHECKPOINT_PATH") or "").strip()
    resume = os.getenv("FL_RESUME", "false").lower() == "true" or bool(resume_from)

    saved_params, last_round = (None, 0)
    if resume_from:
        saved_params, last_round = load_checkpoint_from(resume_from)
    elif resume:
        saved_params, last_round = load_checkpoint()

    round_offset = int(last_round or 0)
    rounds_to_run = total_rounds_target

    if saved_params is not None and round_offset > 0:
        remaining_rounds = total_rounds_target - round_offset
        if remaining_rounds <= 0:
            print(f"\n✅ Entraînement déjà complet ({round_offset} rounds).")
            return
        print(f"\n🔄 Reprise depuis le round {round_offset} → {remaining_rounds} rounds restants")
        rounds_to_run = remaining_rounds

        # Si l'ancien run avait déjà un global_model_best.npz, on le copie dans le nouveau run
        try:
            import shutil
            src_dir = Path(resume_from).parent if Path(resume_from).is_file() else Path(resume_from)
            src_best = src_dir / "global_model_best.npz"
            if src_best.exists():
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_best, CHECKPOINT_DIR / "global_model_best.npz")
        except Exception:
            pass
    else:
        saved_params = None
        round_offset = 0
        print(f"\n🆕 Nouvel entraînement — pas de reprise")

    print(f"\n{'═'*58}")
    print(f"  🚀 Serveur Flower")
    print(f"  Rounds : {rounds_to_run} (objectif={total_rounds_target})  |  Min clients : {min_clients}  |  Local epochs : {local_epochs}")
    if round_offset > 0:
        print(f"  Reprise : depuis round {round_offset} (offset)")
    print(f"  Logs   : {LOGS_DIR.resolve()}")
    print(f"{'═'*58}\n")

    # on_fit_config_fn : config envoyée aux clients avant chaque round d'entraînement
    # Le client lit config.get("local_epochs", 1) dans fit()
    def fit_config(server_round: int) -> dict:
        global_round = int(server_round) + round_offset
        return {
            "local_epochs" : local_epochs,   # nb d'epochs locaux par round
            "round"        : global_round,
        }

    # on_evaluate_config_fn : config envoyée aux clients avant chaque évaluation
    # Le client lit config.get("round", "?") dans evaluate()
    def eval_config(server_round: int) -> dict:
        return {"round": int(server_round) + round_offset}

    # Si reprise depuis checkpoint → fournir les poids initiaux au serveur
    initial_parameters = None
    if saved_params is not None:
        import flwr.common
        initial_parameters = flwr.common.ndarrays_to_parameters(saved_params)

    fl_patience = int(os.getenv("FL_PATIENCE", "5"))
    strategy = FedAvgWithLogging(
        total_rounds              = total_rounds_target,
        patience                  = fl_patience,
        round_offset              = round_offset,
        fraction_fit              = 1.0,
        fraction_evaluate         = 1.0,
        min_fit_clients           = min_clients,
        min_evaluate_clients      = min_clients,
        min_available_clients     = min_clients,
        on_fit_config_fn          = fit_config,
        on_evaluate_config_fn     = eval_config,
        initial_parameters        = initial_parameters,   # ← reprise checkpoint
        evaluate_metrics_aggregation_fn = aggregate_eval_metrics,
        fit_metrics_aggregation_fn      = aggregate_fit_metrics,
    )

    certificates = _build_certificates()

    fl.server.start_server(
        server_address = server_address,
        config         = fl.server.ServerConfig(num_rounds=rounds_to_run),
        strategy       = strategy,
        certificates   = certificates,
    )

    # Rapport final après tous les rounds
    if _logger:
        _logger.save_summary(total_rounds_target)


if __name__ == "__main__":
    main()















# """
# server.py — Serveur Flower pour le Federated Learning
# ═══════════════════════════════════════════════════════
# Changements par rapport à la version originale :

#   1. Système de log des métriques par round
#        → Sauvegarde un fichier JSON  : logs/fl_metrics.json
#        → Sauvegarde un fichier CSV   : logs/fl_metrics.csv
#        → Affiche un rapport console  à chaque round

#   2. Métriques agrégées après chaque FedAvg :
#        Évaluation  → accuracy, loss, recall, precision, f1 (pondérés par nœud)
#        Training    → train_loss, epsilon moyen/max par round
#        Privacy     → budget ε cumulé, statut (OK / WARNING)

#   3. Early stopping FL
#        → Surveille le F1-score agrégé après chaque round
#        → Arrêt automatique si pas d'amélioration pendant FL_PATIENCE rounds (défaut: 5)
#        → Sauvegarde logs/fl/checkpoints/early_stop.txt avec le meilleur round

#   4. Rapport final à la fin de tous les rounds
#        → Meilleur round (accuracy max)
#        → Évolution de ε round par round
#        → Fichier de résumé : logs/fl_summary.txt
# """

# import os
# import json
# import csv
# import time
# from datetime import datetime
# from pathlib import Path
# from typing import List, Tuple, Dict, Optional

# import numpy as np
# import flwr as fl


# # ============================================================================
# # ⚙️  CONFIGURATION
# # ============================================================================

# LOGS_DIR       = Path(os.getenv("FL_LOGS_DIR",       "logs/fl"))
# CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "logs/fl/checkpoints"))
# FL_PATIENCE    = int(os.getenv("FL_PATIENCE",         "5"))   # rounds sans amélioration du F1 avant arrêt




# # ============================================================================
# # 💾 CHECKPOINT — sauvegarde et reprise du modèle global
# # ============================================================================

# def save_checkpoint(server_round: int, parameters) -> None:
#     """Sauvegarde les poids globaux après chaque round."""
#     if parameters is None:
#         print(f"  ⚠️  Round {server_round} : paramètres None — checkpoint ignoré")
#         return

#     CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

#     try:
#         # Convertir les paramètres Flower en numpy arrays
#         import flwr.common
#         ndarrays = flwr.common.parameters_to_ndarrays(parameters)

#         # Sauvegarde du round courant
#         ckpt_path = CHECKPOINT_DIR / f"global_model_round_{server_round:03d}.npz"
#         arrays    = {f"param_{i}": arr for i, arr in enumerate(ndarrays)}
#         np.savez(str(ckpt_path), **arrays)

#         # Sauvegarde du dernier round
#         last_path = CHECKPOINT_DIR / "global_model_last.npz"
#         np.savez(str(last_path), **arrays)

#         # Numéro de round
#         meta_path = CHECKPOINT_DIR / "last_round.txt"
#         meta_path.write_text(str(server_round), encoding="utf-8")

#         print(f"  💾 Checkpoint sauvegardé → round {server_round} ({ckpt_path.name})")

#     except Exception as e:
#         print(f"  ⚠️  Erreur sauvegarde checkpoint round {server_round} : {e}")


# def load_checkpoint() -> tuple[Optional[list], int]:
#     """
#     Charge le dernier checkpoint disponible.
#     Retourne (parameters_as_list, last_round) ou (None, 0) si aucun checkpoint.
#     """
#     last_path = CHECKPOINT_DIR / "global_model_last.npz"
#     meta_path = CHECKPOINT_DIR / "last_round.txt"

#     if not last_path.exists():
#         return None, 0

#     try:
#         data       = np.load(str(last_path))
#         parameters = [data[k] for k in sorted(data.files)]
#         last_round = int(meta_path.read_text(encoding="utf-8").strip()) if meta_path.exists() else 0
#         print(f"  ✅ Checkpoint chargé → reprise depuis le round {last_round}")
#         return parameters, last_round
#     except Exception as e:
#         print(f"  ⚠️  Erreur chargement checkpoint : {e}")
#         return None, 0


# # ============================================================================
# # 📁 LOGGER — sauvegarde JSON + CSV
# # ============================================================================

# class MetricsLogger:
#     """
#     Centralise la sauvegarde des métriques FL à chaque round.

#     Fichiers produits :
#       logs/fl/fl_metrics.json  → historique complet (tous les rounds)
#       logs/fl/fl_metrics.csv   → format tableur (un round par ligne)
#       logs/fl/fl_summary.txt   → rapport final lisible
#     """

#     def __init__(self, logs_dir: Path):
#         self.logs_dir   = logs_dir
#         self.logs_dir.mkdir(parents=True, exist_ok=True)
#         self.history    : list = []
#         self.start_time = time.time()

#         self.json_path    = logs_dir / "fl_metrics.json"
#         self.csv_path     = logs_dir / "fl_metrics.csv"
#         self.summary_path = logs_dir / "fl_summary.txt"

#         # Initialise le CSV avec les en-têtes
#         with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=self._csv_fields())
#             writer.writeheader()

#         print(f"📁 Logs FL → {self.logs_dir.resolve()}")

#     def _csv_fields(self) -> list:
#         return [
#             "round", "timestamp",
#             "accuracy", "loss",
#             "recall", "precision", "f1",
#             "n_clients_eval", "n_examples_eval",
#             "train_loss", "n_clients_fit",
#             "avg_epsilon", "max_epsilon", "epsilon_status",
#             "elapsed_sec",
#         ]

#     def log_round(self, round_num: int, eval_metrics: dict, fit_metrics: dict) -> None:
#         """Enregistre toutes les métriques d'un round."""
#         elapsed  = round(time.time() - self.start_time, 1)
#         ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         # Statut privacy
#         max_eps  = fit_metrics.get("max_epsilon", 0.0)
#         eps_ok   = max_eps <= 1.0

#         record = {
#             "round"           : round_num,
#             "timestamp"       : ts,
#             # Métriques d'évaluation (après FedAvg)
#             "accuracy"        : round(eval_metrics.get("accuracy",  0.0), 4),
#             "loss"            : round(eval_metrics.get("loss",      0.0), 4),
#             "recall"          : round(eval_metrics.get("recall",    0.0), 4),
#             "precision"       : round(eval_metrics.get("precision", 0.0), 4),
#             "f1"              : round(eval_metrics.get("f1",        0.0), 4),
#             "n_clients_eval"  : eval_metrics.get("n_clients",    0),
#             "n_examples_eval" : eval_metrics.get("n_examples",   0),
#             # Métriques d'entraînement
#             "train_loss"      : round(fit_metrics.get("train_loss",   0.0), 4),
#             "n_clients_fit"   : fit_metrics.get("n_clients",      0),
#             # Privacy
#             "avg_epsilon"     : round(fit_metrics.get("avg_epsilon", 0.0), 4),
#             "max_epsilon"     : round(max_eps, 4),
#             "epsilon_status"  : "OK" if eps_ok else "WARNING",
#             "elapsed_sec"     : elapsed,
#         }

#         self.history.append(record)

#         # Sauvegarde JSON (réécriture complète à chaque round)
#         with open(self.json_path, "w") as f:
#             json.dump(self.history, f, indent=2, ensure_ascii=False)

#         # Sauvegarde CSV (ajout de la ligne)
#         with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=self._csv_fields())
#             writer.writerow(record)

#     def print_round_report(self, round_num: int, total_rounds: int) -> None:
#         """Affiche le rapport console du round courant."""
#         if not self.history:
#             return
#         r = self.history[-1]

#         eps_icon = "✅" if r["epsilon_status"] == "OK" else "⚠️"

#         print(f"\n{'─'*58}")
#         print(f"  📊  Round {round_num}/{total_rounds}  —  {r['timestamp']}")
#         print(f"{'─'*58}")
#         print(f"  Évaluation ({r['n_clients_eval']} nœuds · {r['n_examples_eval']:,} txns)")
#         print(f"    Accuracy  : {r['accuracy']:.4f}")
#         print(f"    Loss      : {r['loss']:.4f}")
#         print(f"    Recall    : {r['recall']:.4f}")
#         print(f"    Precision : {r['precision']:.4f}")
#         print(f"    F1-score  : {r['f1']:.4f}")
#         print(f"  Entraînement ({r['n_clients_fit']} nœuds)")
#         print(f"    Train Loss : {r['train_loss']:.4f}")
#         print(f"  Privacy {eps_icon}")
#         print(f"    Avg ε  : {r['avg_epsilon']:.4f}")
#         print(f"    Max ε  : {r['max_epsilon']:.4f}  [{r['epsilon_status']}]")
#         print(f"  Temps écoulé : {r['elapsed_sec']}s")
#         print(f"{'─'*58}")

#     def save_summary(self, total_rounds: int) -> None:
#         """Écrit le rapport final après tous les rounds."""
#         if not self.history:
#             return

#         best_round = max(self.history, key=lambda r: r["accuracy"])
#         total_time = round(time.time() - self.start_time, 1)

#         lines = [
#             "=" * 58,
#             "  RAPPORT FINAL — Federated Learning",
#             "=" * 58,
#             f"  Rounds effectués  : {len(self.history)} / {total_rounds}",
#             f"  Temps total       : {total_time}s",
#             f"  Nœuds (min)       : {self.history[0].get('n_clients_eval', '?')}",
#             "",
#             "  Meilleur round :",
#             f"    Round     : {best_round['round']}",
#             f"    Accuracy  : {best_round['accuracy']:.4f}",
#             f"    Recall    : {best_round['recall']:.4f}",
#             f"    Precision : {best_round['precision']:.4f}",
#             f"    F1-score  : {best_round['f1']:.4f}",
#             f"    Loss      : {best_round['loss']:.4f}",
#             "",
#             "  Évolution ε par round :",
#         ]

#         for r in self.history:
#             icon = "✅" if r["epsilon_status"] == "OK" else "⚠️"
#             lines.append(
#                 f"    Round {r['round']:>2} : avg={r['avg_epsilon']:.4f}"
#                 f"  max={r['max_epsilon']:.4f}  {icon}"
#             )

#         lines += [
#             "",
#             "  Fichiers de log :",
#             f"    JSON    : {self.json_path}",
#             f"    CSV     : {self.csv_path}",
#             "=" * 58,
#         ]

#         summary_text = "\n".join(lines)
#         with open(self.summary_path, "w", encoding="utf-8") as f:
#             f.write(summary_text)

#         print(f"\n{summary_text}")
#         print(f"\n📄 Résumé sauvegardé → {self.summary_path}")


# # ============================================================================
# # 📊 FONCTIONS D'AGRÉGATION DES MÉTRIQUES
# # ============================================================================

# # Instance globale du logger (initialisée dans main())
# _logger: Optional[MetricsLogger] = None
# _total_rounds: int = 0


# def aggregate_eval_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
#     """
#     Agrège les métriques d'évaluation de tous les nœuds (moyenne pondérée).

#     Métriques reçues depuis client.evaluate() :
#       - accuracy  : taux de bonnes prédictions
#       - loss      : BCEWithLogitsLoss moyenne
#       - recall    : taux de fraudes détectées      ← clé pour la fraude
#       - precision : taux de précision sur fraudes  ← clé pour les FA
#       - f1        : F1-score fraude

#     Note : le client.py actuel ne renvoie que accuracy et loss.
#     Les autres métriques seront ajoutées quand client.py sera mis à jour.
#     """
#     if not metrics:
#         return {}

#     total_examples = sum(n for n, _ in metrics)
#     if total_examples == 0:
#         return {}

#     def wavg(key: str) -> float:
#         return sum(n * m.get(key, 0.0) for n, m in metrics) / total_examples

#     aggregated = {
#         "accuracy"  : wavg("accuracy"),
#         "loss"      : wavg("loss"),
#         "recall"    : wavg("recall"),
#         "precision" : wavg("precision"),
#         "f1"        : wavg("f1"),
#         "n_clients" : len(metrics),
#         "n_examples": total_examples,
#     }

#     return aggregated


# def aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
#     """
#     Agrège les métriques d'entraînement de tous les nœuds.

#     Métriques reçues depuis client.fit() :
#       - epsilon    : budget DP consommé par ce nœud
#       - train_loss : loss d'entraînement local
#     """
#     if not metrics:
#         return {}

#     epsilons    = [m["epsilon"]    for _, m in metrics if "epsilon"    in m]
#     train_losses= [m["train_loss"] for _, m in metrics if "train_loss" in m]

#     aggregated = {
#         "n_clients"  : len(metrics),
#         "avg_epsilon": sum(epsilons) / len(epsilons)       if epsilons     else 0.0,
#         "max_epsilon": max(epsilons)                        if epsilons     else 0.0,
#         "train_loss" : sum(train_losses) / len(train_losses) if train_losses else 0.0,
#     }

#     # Affichage privacy
#     if epsilons:
#         eps_ok = aggregated["max_epsilon"] <= 1.0
#         icon   = "✅" if eps_ok else "⚠️"
#         print(f"\n🔐 Privacy {icon}  avg_ε={aggregated['avg_epsilon']:.4f}"
#               f"  max_ε={aggregated['max_epsilon']:.4f}")

#     return aggregated


# # ============================================================================
# # 🧠 STRATEGY CUSTOM — FedAvg + logging par round
# # ============================================================================

# class FedAvgWithLogging(fl.server.strategy.FedAvg):
#     """
#     Extension de FedAvg qui intercepte les résultats de chaque round
#     pour les logger via MetricsLogger.
#     """

#     def __init__(self, total_rounds: int, patience: int = 5, **kwargs):
#         super().__init__(**kwargs)
#         self.total_rounds      = total_rounds
#         self.current_round     = 0
#         # ── Early stopping ───────────────────────────────────────────────────
#         self.patience          = patience
#         self.best_f1           = 0.0
#         self.best_round        = 0
#         self.rounds_no_improve = 0

#     def aggregate_evaluate(self, server_round, results, failures):
#         """Appelé après l'évaluation de chaque round."""
#         aggregated = super().aggregate_evaluate(server_round, results, failures)
#         self.current_round = server_round

#         # Extraction des métriques depuis les résultats bruts
#         eval_metrics_raw = [
#             (res.num_examples, res.metrics)
#             for _, res in results
#             if res is not None
#         ]
#         eval_agg = aggregate_eval_metrics(eval_metrics_raw)

#         # Récupère les métriques fit du round courant (déjà dans l'historique)
#         fit_agg = {}
#         if _logger and _logger.history:
#             last = _logger.history[-1] if _logger.history else {}
#             fit_agg = {
#                 "avg_epsilon": last.get("avg_epsilon", 0.0),
#                 "max_epsilon": last.get("max_epsilon", 0.0),
#                 "train_loss" : last.get("train_loss",  0.0),
#                 "n_clients"  : last.get("n_clients_fit", 0),
#             }

#         # Log du round
#         if _logger:
#             _logger.log_round(server_round, eval_agg, fit_agg)
#             _logger.print_round_report(server_round, self.total_rounds)

#         # ── Early stopping sur F1 agrégé ──────────────────────────────────────
#         current_f1 = eval_agg.get("f1", 0.0)
#         if current_f1 > self.best_f1 + 1e-4:
#             self.best_f1           = current_f1
#             self.best_round        = server_round
#             self.rounds_no_improve = 0
#             print(f"  ✅ Nouveau meilleur F1 global : {self.best_f1:.4f} (round {server_round})")
#         else:
#             self.rounds_no_improve += 1
#             print(f"  ⏳ F1 sans amélioration : {self.rounds_no_improve}/{self.patience}"
#                   f"  (best={self.best_f1:.4f} @ round {self.best_round}"
#                   f"  |  current={current_f1:.4f})")

#             if self.rounds_no_improve >= self.patience:
#                 print(f"\n{'═'*58}")
#                 print(f"  🛑 EARLY STOPPING FL")
#                 print(f"  {self.patience} rounds consécutifs sans amélioration du F1")
#                 print(f"  Meilleur F1 : {self.best_f1:.4f} au round {self.best_round}")
#                 print(f"  Checkpoint  : logs/fl/checkpoints/global_model_round_{self.best_round:03d}.npz")
#                 print(f"{'═'*58}\n")
#                 # Sauvegarde un flag pour signaler l'arrêt
#                 CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
#                 stop_path = CHECKPOINT_DIR / "early_stop.txt"
#                 stop_path.write_text(
#                     f"Early stop au round {server_round}\n"
#                     f"Meilleur F1 = {self.best_f1:.4f} au round {self.best_round}\n"
#                     f"Meilleur checkpoint : global_model_round_{self.best_round:03d}.npz",
#                     encoding="utf-8"
#                 )

#         return aggregated

#     def aggregate_fit(self, server_round, results, failures):
#         """Appelé après l'entraînement de chaque round."""
#         aggregated = super().aggregate_fit(server_round, results, failures)

#         # Log préliminaire des métriques fit (avant évaluation)
#         fit_metrics_raw = [
#             (res.num_examples, res.metrics)
#             for _, res in results
#             if res is not None
#         ]
#         fit_agg = aggregate_fit_metrics(fit_metrics_raw)

#         # Pré-enregistrement pour que aggregate_evaluate puisse le lire
#         if _logger:
#             # Entrée temporaire dans l'historique (sera complétée par aggregate_evaluate)
#             _logger.history.append({
#                 "round"          : server_round,
#                 "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "accuracy"       : 0.0, "loss": 0.0,
#                 "recall"         : 0.0, "precision": 0.0, "f1": 0.0,
#                 "n_clients_eval" : 0,   "n_examples_eval": 0,
#                 "train_loss"     : fit_agg.get("train_loss",  0.0),
#                 "n_clients_fit"  : fit_agg.get("n_clients",   0),
#                 "avg_epsilon"    : fit_agg.get("avg_epsilon", 0.0),
#                 "max_epsilon"    : fit_agg.get("max_epsilon", 0.0),
#                 "epsilon_status" : "OK" if fit_agg.get("max_epsilon", 0) <= 1.0 else "WARNING",
#                 "elapsed_sec"    : round(time.time() - _logger.start_time, 1),
#             })

#         # 💾 Sauvegarde checkpoint après chaque round
#         if aggregated is not None:
#             parameters, _ = aggregated
#             save_checkpoint(server_round, parameters)
#         else:
#             print(f"  ⚠️  Round {server_round} : agrégation échouée (0 résultats) — checkpoint non sauvegardé")

#         return aggregated


# # ============================================================================
# # 🔐 TLS
# # ============================================================================

# def _read_cert(path: Optional[str]) -> Optional[bytes]:
#     if not path:
#         return None
#     cert_path = Path(path)
#     if not cert_path.exists():
#         return None
#     return cert_path.read_bytes()


# def _build_certificates() -> Optional[Tuple[bytes, bytes, bytes]]:
#     ca_cert     = _read_cert(os.getenv("FLOWER_TLS_CA_CERT"))
#     server_cert = _read_cert(os.getenv("FLOWER_TLS_SERVER_CERT"))
#     server_key  = _read_cert(os.getenv("FLOWER_TLS_SERVER_KEY"))

#     if server_cert is None or server_key is None:
#         return None

#     if os.getenv("FLOWER_TLS_REQUIRE_CLIENT_CERT", "false").lower() == "true":
#         if ca_cert is None:
#             raise RuntimeError(
#                 "FLOWER_TLS_REQUIRE_CLIENT_CERT=true mais FLOWER_TLS_CA_CERT est absent"
#             )

#     return (ca_cert, server_cert, server_key) if ca_cert is not None else None


# # ============================================================================
# # 🚀 MAIN
# # ============================================================================

# def main():
#     global _logger, _total_rounds

#     rounds        = int(os.getenv("FL_ROUNDS",        "5"))
#     min_clients   = int(os.getenv("FL_MIN_CLIENTS",   "2"))
#     local_epochs  = int(os.getenv("FL_LOCAL_EPOCHS",  "3"))
#     port          = int(os.getenv("FLOWER_PORT",       "8080"))
#     server_address = f"0.0.0.0:{port}"

#     _total_rounds = rounds
#     _logger       = MetricsLogger(LOGS_DIR)

#     # ── Reprise depuis checkpoint si disponible ───────────────────────────────
#     resume      = os.getenv("FL_RESUME", "true").lower() == "true"
#     saved_params, last_round = load_checkpoint() if resume else (None, 0)

#     if saved_params is not None and last_round > 0:
#         remaining_rounds = rounds - last_round
#         if remaining_rounds <= 0:
#             print(f"\n✅ Entraînement déjà complet ({last_round} rounds). Supprimer les checkpoints pour relancer.")
#             return
#         print(f"\n🔄 Reprise depuis le round {last_round} → {remaining_rounds} rounds restants")
#         rounds = remaining_rounds
#     else:
#         saved_params = None
#         print(f"\n🆕 Nouvel entraînement — aucun checkpoint trouvé")

#     print(f"\n{'═'*58}")
#     print(f"  🚀 Serveur Flower")
#     print(f"  Rounds : {rounds}  |  Min clients : {min_clients}  |  Local epochs : {local_epochs}")
#     if saved_params is not None:
#         print(f"  Reprise : depuis round {last_round}")
#     print(f"  Logs   : {LOGS_DIR.resolve()}")
#     print(f"{'═'*58}\n")

#     # on_fit_config_fn : config envoyée aux clients avant chaque round d'entraînement
#     # Le client lit config.get("local_epochs", 1) dans fit()
#     def fit_config(server_round: int) -> dict:
#         return {
#             "local_epochs" : local_epochs,   # nb d'epochs locaux par round
#             "round"        : server_round,
#         }

#     # on_evaluate_config_fn : config envoyée aux clients avant chaque évaluation
#     # Le client lit config.get("round", "?") dans evaluate()
#     def eval_config(server_round: int) -> dict:
#         return {"round": server_round}

#     # Si reprise depuis checkpoint → fournir les poids initiaux au serveur
#     initial_parameters = None
#     if saved_params is not None:
#         import flwr.common
#         initial_parameters = flwr.common.ndarrays_to_parameters(saved_params)

#     fl_patience = int(os.getenv("FL_PATIENCE", "5"))
#     strategy = FedAvgWithLogging(
#         total_rounds              = rounds,
#         patience                  = fl_patience,
#         fraction_fit              = 1.0,
#         fraction_evaluate         = 1.0,
#         min_fit_clients           = min_clients,
#         min_evaluate_clients      = min_clients,
#         min_available_clients     = min_clients,
#         on_fit_config_fn          = fit_config,
#         on_evaluate_config_fn     = eval_config,
#         initial_parameters        = initial_parameters,   # ← reprise checkpoint
#         evaluate_metrics_aggregation_fn = aggregate_eval_metrics,
#         fit_metrics_aggregation_fn      = aggregate_fit_metrics,
#     )

#     certificates = _build_certificates()

#     fl.server.start_server(
#         server_address = server_address,
#         config         = fl.server.ServerConfig(num_rounds=rounds),
#         strategy       = strategy,
#         certificates   = certificates,
#     )

#     # Rapport final après tous les rounds
#     if _logger:
#         _logger.save_summary(rounds)


# if __name__ == "__main__":
#     main()











# """
# server.py — Serveur Flower pour le Federated Learning
# ═══════════════════════════════════════════════════════
# Changements par rapport à la version originale :

#   1. Système de log des métriques par round
#        → Sauvegarde un fichier JSON  : logs/fl_metrics.json
#        → Sauvegarde un fichier CSV   : logs/fl_metrics.csv
#        → Affiche un rapport console  à chaque round

#   2. Métriques agrégées après chaque FedAvg :
#        Évaluation  → accuracy, loss, recall, precision, f1 (pondérés par nœud)
#        Training    → train_loss, epsilon moyen/max par round
#        Privacy     → budget ε cumulé, statut (OK / WARNING)

#   3. Early stopping FL
#        → Surveille le F1-score agrégé après chaque round
#        → Arrêt automatique si pas d'amélioration pendant FL_PATIENCE rounds (défaut: 5)
#        → Sauvegarde logs/fl/checkpoints/early_stop.txt avec le meilleur round
#
#   4. Rapport final à la fin de tous les rounds
#        → Meilleur round (accuracy max)
#        → Évolution de ε round par round
#        → Fichier de résumé : logs/fl_summary.txt
# """

# import os
# import json
# import csv
# import time
# from datetime import datetime
# from pathlib import Path
# from typing import List, Tuple, Dict, Optional

# import flwr as fl

# import torch
# from models.fraud_rnn import build_model


# # ============================================================================
# # ⚙️  CONFIGURATION
# # ============================================================================

# LOGS_DIR = Path(os.getenv("FL_LOGS_DIR", "logs/fl"))


# # ============================================================================
# # 📁 LOGGER — sauvegarde JSON + CSV
# # ============================================================================

# class MetricsLogger:
#     """
#     Centralise la sauvegarde des métriques FL à chaque round.

#     Fichiers produits :
#       logs/fl/fl_metrics.json  → historique complet (tous les rounds)
#       logs/fl/fl_metrics.csv   → format tableur (un round par ligne)
#       logs/fl/fl_summary.txt   → rapport final lisible
#     """

#     def __init__(self, logs_dir: Path):
#         self.logs_dir   = logs_dir
#         self.logs_dir.mkdir(parents=True, exist_ok=True)
#         self.history    : list = []
#         self.start_time = time.time()

#         self.json_path    = logs_dir / "fl_metrics.json"
#         self.csv_path     = logs_dir / "fl_metrics.csv"
#         self.summary_path = logs_dir / "fl_summary.txt"

#         # Initialise le CSV avec les en-têtes
#         with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=self._csv_fields())
#             writer.writeheader()

#         print(f"📁 Logs FL → {self.logs_dir.resolve()}")

#     def _csv_fields(self) -> list:
#         return [
#             "round", "timestamp",
#             "accuracy", "loss",
#             "recall", "precision", "f1",
#             "n_clients_eval", "n_examples_eval",
#             "train_loss", "n_clients_fit",
#             "avg_epsilon", "max_epsilon", "epsilon_status",
#             "elapsed_sec",
#         ]

#     def log_round(self, round_num: int, eval_metrics: dict, fit_metrics: dict) -> None:
#         """Enregistre toutes les métriques d'un round."""
#         elapsed  = round(time.time() - self.start_time, 1)
#         ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         # Statut privacy
#         max_eps  = fit_metrics.get("max_epsilon", 0.0)
#         eps_ok   = max_eps <= 1.0

#         record = {
#             "round"           : round_num,
#             "timestamp"       : ts,
#             # Métriques d'évaluation (après FedAvg)
#             "accuracy"        : round(eval_metrics.get("accuracy",  0.0), 4),
#             "loss"            : round(eval_metrics.get("loss",      0.0), 4),
#             "recall"          : round(eval_metrics.get("recall",    0.0), 4),
#             "precision"       : round(eval_metrics.get("precision", 0.0), 4),
#             "f1"              : round(eval_metrics.get("f1",        0.0), 4),
#             "n_clients_eval"  : eval_metrics.get("n_clients",    0),
#             "n_examples_eval" : eval_metrics.get("n_examples",   0),
#             # Métriques d'entraînement
#             "train_loss"      : round(fit_metrics.get("train_loss",   0.0), 4),
#             "n_clients_fit"   : fit_metrics.get("n_clients",      0),
#             # Privacy
#             "avg_epsilon"     : round(fit_metrics.get("avg_epsilon", 0.0), 4),
#             "max_epsilon"     : round(max_eps, 4),
#             "epsilon_status"  : "OK" if eps_ok else "WARNING",
#             "elapsed_sec"     : elapsed,
#         }

#         self.history.append(record)

#         # Sauvegarde JSON (réécriture complète à chaque round)
#         with open(self.json_path, "w") as f:
#             json.dump(self.history, f, indent=2, ensure_ascii=False)

#         # Sauvegarde CSV (ajout de la ligne)
#         with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=self._csv_fields())
#             writer.writerow(record)

#     def print_round_report(self, round_num: int, total_rounds: int) -> None:
#         """Affiche le rapport console du round courant."""
#         if not self.history:
#             return
#         r = self.history[-1]

#         eps_icon = "✅" if r["epsilon_status"] == "OK" else "⚠️"

#         print(f"\n{'─'*58}")
#         print(f"  📊  Round {round_num}/{total_rounds}  —  {r['timestamp']}")
#         print(f"{'─'*58}")
#         print(f"  Évaluation ({r['n_clients_eval']} nœuds · {r['n_examples_eval']:,} txns)")
#         print(f"    Accuracy  : {r['accuracy']:.4f}")
#         print(f"    Loss      : {r['loss']:.4f}")
#         print(f"    Recall    : {r['recall']:.4f}")
#         print(f"    Precision : {r['precision']:.4f}")
#         print(f"    F1-score  : {r['f1']:.4f}")
#         print(f"  Entraînement ({r['n_clients_fit']} nœuds)")
#         print(f"    Train Loss : {r['train_loss']:.4f}")
#         print(f"  Privacy {eps_icon}")
#         print(f"    Avg ε  : {r['avg_epsilon']:.4f}")
#         print(f"    Max ε  : {r['max_epsilon']:.4f}  [{r['epsilon_status']}]")
#         print(f"  Temps écoulé : {r['elapsed_sec']}s")
#         print(f"{'─'*58}")

#     def save_summary(self, total_rounds: int) -> None:
#         """Écrit le rapport final après tous les rounds."""
#         if not self.history:
#             return

#         best_round = max(self.history, key=lambda r: r["accuracy"])
#         total_time = round(time.time() - self.start_time, 1)

#         lines = [
#             "=" * 58,
#             "  RAPPORT FINAL — Federated Learning",
#             "=" * 58,
#             f"  Rounds effectués  : {len(self.history)} / {total_rounds}",
#             f"  Temps total       : {total_time}s",
#             f"  Nœuds (min)       : {self.history[0].get('n_clients_eval', '?')}",
#             "",
#             "  Meilleur round :",
#             f"    Round     : {best_round['round']}",
#             f"    Accuracy  : {best_round['accuracy']:.4f}",
#             f"    Recall    : {best_round['recall']:.4f}",
#             f"    Precision : {best_round['precision']:.4f}",
#             f"    F1-score  : {best_round['f1']:.4f}",
#             f"    Loss      : {best_round['loss']:.4f}",
#             "",
#             "  Évolution ε par round :",
#         ]

#         for r in self.history:
#             icon = "✅" if r["epsilon_status"] == "OK" else "⚠️"
#             lines.append(
#                 f"    Round {r['round']:>2} : avg={r['avg_epsilon']:.4f}"
#                 f"  max={r['max_epsilon']:.4f}  {icon}"
#             )

#         lines += [
#             "",
#             "  Fichiers de log :",
#             f"    JSON    : {self.json_path}",
#             f"    CSV     : {self.csv_path}",
#             "=" * 58,
#         ]

#         summary_text = "\n".join(lines)
#         with open(self.summary_path, "w", encoding="utf-8") as f:
#             f.write(summary_text)

#         print(f"\n{summary_text}")
#         print(f"\n📄 Résumé sauvegardé → {self.summary_path}")




# def set_model_parameters(model, parameters):
#     for p, new_p in zip(model.parameters(), parameters):
#         p.data = torch.tensor(new_p, dtype=torch.float32)


# # ============================================================================
# # 📊 FONCTIONS D'AGRÉGATION DES MÉTRIQUES
# # ============================================================================

# # Instance globale du logger (initialisée dans main())
# _logger: Optional[MetricsLogger] = None
# _total_rounds: int = 0


# def aggregate_eval_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
#     """
#     Agrège les métriques d'évaluation de tous les nœuds (moyenne pondérée).

#     Métriques reçues depuis client.evaluate() :
#       - accuracy  : taux de bonnes prédictions
#       - loss      : BCEWithLogitsLoss moyenne
#       - recall    : taux de fraudes détectées      ← clé pour la fraude
#       - precision : taux de précision sur fraudes  ← clé pour les FA
#       - f1        : F1-score fraude

#     Note : le client.py actuel ne renvoie que accuracy et loss.
#     Les autres métriques seront ajoutées quand client.py sera mis à jour.
#     """
#     if not metrics:
#         return {}

#     total_examples = sum(n for n, _ in metrics)
#     if total_examples == 0:
#         return {}

#     def wavg(key: str) -> float:
#         return sum(n * m.get(key, 0.0) for n, m in metrics) / total_examples

#     aggregated = {
#         "accuracy"  : wavg("accuracy"),
#         "loss"      : wavg("loss"),
#         "recall"    : wavg("recall"),
#         "precision" : wavg("precision"),
#         "f1"        : wavg("f1"),
#         "n_clients" : len(metrics),
#         "n_examples": total_examples,
#     }

#     return aggregated


# def aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
#     """
#     Agrège les métriques d'entraînement de tous les nœuds.

#     Métriques reçues depuis client.fit() :
#       - epsilon    : budget DP consommé par ce nœud
#       - train_loss : loss d'entraînement local
#     """
#     if not metrics:
#         return {}

#     epsilons    = [m["epsilon"]    for _, m in metrics if "epsilon"    in m]
#     train_losses= [m["train_loss"] for _, m in metrics if "train_loss" in m]

#     aggregated = {
#         "n_clients"  : len(metrics),
#         "avg_epsilon": sum(epsilons) / len(epsilons)       if epsilons     else 0.0,
#         "max_epsilon": max(epsilons)                        if epsilons     else 0.0,
#         "train_loss" : sum(train_losses) / len(train_losses) if train_losses else 0.0,
#     }

#     # Affichage privacy
#     if epsilons:
#         eps_ok = aggregated["max_epsilon"] <= 1.0
#         icon   = "✅" if eps_ok else "⚠️"
#         print(f"\n🔐 Privacy {icon}  avg_ε={aggregated['avg_epsilon']:.4f}"
#               f"  max_ε={aggregated['max_epsilon']:.4f}")

#     return aggregated


# # ============================================================================
# # 🧠 STRATEGY CUSTOM — FedAvg + logging par round
# # ============================================================================

# class FedAvgWithLogging(fl.server.strategy.FedAvg):
#     """
#     Extension de FedAvg qui intercepte les résultats de chaque round
#     pour les logger via MetricsLogger.
#     """

#     def __init__(self, total_rounds: int, **kwargs):
#         super().__init__(**kwargs)
#         self.total_rounds  = total_rounds
#         self.current_round = 0
#         self.final_parameters = None 

#     def aggregate_evaluate(self, server_round, results, failures):
#         """Appelé après l'évaluation de chaque round."""
#         aggregated = super().aggregate_evaluate(server_round, results, failures)
#         self.current_round = server_round

#         # Extraction des métriques depuis les résultats bruts
#         eval_metrics_raw = [
#             (res.num_examples, res.metrics)
#             for _, res in results
#             if res is not None
#         ]
#         eval_agg = aggregate_eval_metrics(eval_metrics_raw)

#         # Récupère les métriques fit du round courant (déjà dans l'historique)
#         fit_agg = {}
#         if _logger and _logger.history:
#             last = _logger.history[-1] if _logger.history else {}
#             fit_agg = {
#                 "avg_epsilon": last.get("avg_epsilon", 0.0),
#                 "max_epsilon": last.get("max_epsilon", 0.0),
#                 "train_loss" : last.get("train_loss",  0.0),
#                 "n_clients"  : last.get("n_clients_fit", 0),
#             }

#         # Log du round
#         if _logger:
#             _logger.log_round(server_round, eval_agg, fit_agg)
#             _logger.print_round_report(server_round, self.total_rounds)

#         return aggregated

#     def aggregate_fit(self, server_round, results, failures):
#         """Appelé après l'entraînement de chaque round."""
#         aggregated = super().aggregate_fit(server_round, results, failures)

#             # 👇 Sauvegarde des paramètres globaux
#         if aggregated is not None:
#             self.final_parameters = aggregated[0]

#         # Log préliminaire des métriques fit (avant évaluation)
#         fit_metrics_raw = [
#             (res.num_examples, res.metrics)
#             for _, res in results
#             if res is not None
#         ]
#         fit_agg = aggregate_fit_metrics(fit_metrics_raw)

#         # Pré-enregistrement pour que aggregate_evaluate puisse le lire
#         if _logger:
#             # Entrée temporaire dans l'historique (sera complétée par aggregate_evaluate)
#             _logger.history.append({
#                 "round"          : server_round,
#                 "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "accuracy"       : 0.0, "loss": 0.0,
#                 "recall"         : 0.0, "precision": 0.0, "f1": 0.0,
#                 "n_clients_eval" : 0,   "n_examples_eval": 0,
#                 "train_loss"     : fit_agg.get("train_loss",  0.0),
#                 "n_clients_fit"  : fit_agg.get("n_clients",   0),
#                 "avg_epsilon"    : fit_agg.get("avg_epsilon", 0.0),
#                 "max_epsilon"    : fit_agg.get("max_epsilon", 0.0),
#                 "epsilon_status" : "OK" if fit_agg.get("max_epsilon", 0) <= 1.0 else "WARNING",
#                 "elapsed_sec"    : round(time.time() - _logger.start_time, 1),
#             })

#         return aggregated


# # ============================================================================
# # 🔐 TLS
# # ============================================================================

# def _read_cert(path: Optional[str]) -> Optional[bytes]:
#     if not path:
#         return None
#     cert_path = Path(path)
#     if not cert_path.exists():
#         return None
#     return cert_path.read_bytes()


# def _build_certificates() -> Optional[Tuple[bytes, bytes, bytes]]:
#     ca_cert     = _read_cert(os.getenv("FLOWER_TLS_CA_CERT"))
#     server_cert = _read_cert(os.getenv("FLOWER_TLS_SERVER_CERT"))
#     server_key  = _read_cert(os.getenv("FLOWER_TLS_SERVER_KEY"))

#     if server_cert is None or server_key is None:
#         return None

#     if os.getenv("FLOWER_TLS_REQUIRE_CLIENT_CERT", "false").lower() == "true":
#         if ca_cert is None:
#             raise RuntimeError(
#                 "FLOWER_TLS_REQUIRE_CLIENT_CERT=true mais FLOWER_TLS_CA_CERT est absent"
#             )

#     return (ca_cert, server_cert, server_key) if ca_cert is not None else None


# # ============================================================================
# # 🚀 MAIN
# # ============================================================================

# def main():
#     global _logger, _total_rounds

#     rounds        = int(os.getenv("FL_ROUNDS",        "20"))
#     min_clients   = int(os.getenv("FL_MIN_CLIENTS",   "2"))
#     local_epochs  = int(os.getenv("FL_LOCAL_EPOCHS",  "10"))
#     port          = int(os.getenv("FLOWER_PORT",       "8080"))
#     server_address = f"0.0.0.0:{port}"
#     # server_address = "localhost:8080"

#     _total_rounds = rounds
#     _logger       = MetricsLogger(LOGS_DIR)

#     print(f"\n{'═'*58}")
#     print(f"  🚀 Serveur Flower")
#     print(f"  Rounds : {rounds}  |  Min clients : {min_clients}  |  Local epochs : {local_epochs}")
#     print(f"  Logs   : {LOGS_DIR.resolve()}")
#     print(f"{'═'*58}\n")

#     # on_fit_config_fn : config envoyée aux clients avant chaque round d'entraînement
#     # Le client lit config.get("local_epochs", 1) dans fit()
#     def fit_config(server_round: int) -> dict:
#         return {
#             "local_epochs" : local_epochs,   # nb d'epochs locaux par round
#             "round"        : server_round,
#         }

#     # on_evaluate_config_fn : config envoyée aux clients avant chaque évaluation
#     # Le client lit config.get("round", "?") dans evaluate()
#     def eval_config(server_round: int) -> dict:
#         return {"round": server_round}

#     strategy = FedAvgWithLogging(
#         total_rounds              = rounds,
#         fraction_fit              = 1.0,
#         fraction_evaluate         = 1.0,
#         min_fit_clients           = min_clients,
#         min_evaluate_clients      = min_clients,
#         min_available_clients     = min_clients,
#         on_fit_config_fn          = fit_config,      # ← envoie local_epochs aux clients
#         on_evaluate_config_fn     = eval_config,     # ← envoie round num aux clients
#         evaluate_metrics_aggregation_fn = aggregate_eval_metrics,
#         fit_metrics_aggregation_fn      = aggregate_fit_metrics,
#     )

#     certificates = _build_certificates()

#     fl.server.start_server(
#         server_address = server_address,
#         config         = fl.server.ServerConfig(num_rounds=rounds),
#         strategy       = strategy,
#         certificates   = certificates,
#     )

#     # Rapport final après tous les rounds
#     if _logger:
#         _logger.save_summary(rounds)


#     # Sauvegarde du modèle final
#     if strategy.final_parameters is not None:
#         print("💾 Sauvegarde du modèle global...")

#         model = build_model(use_dpgru=True)
#         params_ndarrays = fl.common.parameters_to_ndarrays(strategy.final_parameters)

#         set_model_parameters(model, params_ndarrays)

#         save_path = Path("saved_models")
#         save_path.mkdir(exist_ok=True)

#         torch.save(model.state_dict(), save_path / "global_model.pth")

#         print(f"✅ Modèle sauvegardé → {save_path / 'global_model.pth'}")


# if __name__ == "__main__":
#     main()













# import os
# from pathlib import Path
# from typing import List, Tuple, Dict,Optional

# import flwr as fl


# # ============================
# # 📊 AGGREGATION METRICS
# # ============================

# def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
#     """Moyenne pondérée des accuracies"""
#     if not metrics:
#         return {}

#     total_examples = sum(num_examples for num_examples, _ in metrics)

#     accuracy = sum(
#         num_examples * m.get("accuracy", 0.0)
#         for num_examples, m in metrics
#     ) / total_examples

#     return {"accuracy": accuracy}


# # ============================
# # 🔐 AGGREGATION PRIVACY ε
# # ============================

# def aggregate_epsilon(metrics: List[Tuple[int, Dict]]) -> Dict:
#     """Affiche le epsilon moyen"""
#     epsilons = [m["epsilon"] for _, m in metrics if "epsilon" in m]

#     if epsilons:
#         avg_eps = sum(epsilons) / len(epsilons)
#         max_eps = max(epsilons)

#         print("\n🔐 ===== Privacy Report =====")
#         print(f"📊 Clients: {len(epsilons)}")
#         print(f"📉 Avg ε: {avg_eps:.4f}")
#         print(f"📈 Max ε: {max_eps:.4f}")

#         if max_eps > 1.0:
#             print("⚠️ WARNING: ε > 1.0 (privacy faible)")
#         else:
#             print("✅ Privacy OK (ε < 1.0)")
#         print("===========================\n")

#     return {"avg_epsilon": avg_eps, "max_epsilon": max_eps} if epsilons else {}


# # ============================
# # 🔐 TLS 
# # ============================

# def _read_cert(path: Optional[str]) -> Optional[bytes]:
# 	if not path:
# 		return None
# 	cert_path = Path(path)
# 	if not cert_path.exists():
# 		return None
# 	return cert_path.read_bytes()

# # def load_certificates():
# #     """Charge les certificats TLS si fournis"""
# #     ca_cert = os.getenv("FLOWER_CA_CERT")
# #     server_cert = os.getenv("FLOWER_SERVER_CERT")
# #     server_key = os.getenv("FLOWER_SERVER_KEY")

# #     if ca_cert and server_cert and server_key:
# #         try:
# #             return (
# #                 open(ca_cert, "rb").read(),
# #                 open(server_cert, "rb").read(),
# #                 open(server_key, "rb").read(),
# #             )
# #         except Exception as e:
# #             print(f"⚠️ Erreur chargement TLS: {e}")

# #     return None


# def _build_certificates() -> Optional[Tuple[bytes, bytes, bytes]]:
# 	ca_cert = _read_cert(os.getenv("FLOWER_TLS_CA_CERT"))
# 	server_cert = _read_cert(os.getenv("FLOWER_TLS_SERVER_CERT"))
# 	server_key = _read_cert(os.getenv("FLOWER_TLS_SERVER_KEY"))

# 	if server_cert is None or server_key is None:
# 		return None

# 	require_client_cert = os.getenv("FLOWER_TLS_REQUIRE_CLIENT_CERT", "false").lower() == "true"
# 	if require_client_cert and ca_cert is None:
# 		raise RuntimeError("FLOWER_TLS_REQUIRE_CLIENT_CERT=true mais FLOWER_TLS_CA_CERT est absent")

# 	if ca_cert is not None:
# 		return (ca_cert, server_cert, server_key)

# 	return None


# # ============================
# # 🚀 MAIN SERVER
# # ============================

# def main():
#     # ⚙️ Config dynamique
#     rounds = int(os.getenv("FL_ROUNDS", "5"))
#     min_clients = int(os.getenv("FL_MIN_CLIENTS", "2"))
#     port = int(os.getenv("FLOWER_PORT", "8080"))
#     server_address = f"0.0.0.0:{port}"

#     print("🚀 Lancement du serveur Flower")
#     print(f"🔁 Rounds: {rounds}")
#     print(f"👥 Min clients: {min_clients}")

#     # 🧠 STRATEGY FedAvg
#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_fit_clients=min_clients,
#         min_evaluate_clients=min_clients,
#         min_available_clients=min_clients,

#         evaluate_metrics_aggregation_fn=weighted_average,
#         fit_metrics_aggregation_fn=aggregate_epsilon,
#     )

#     # 🔐 TLS
#     certificates = _build_certificates()

#     # 🚀 START SERVER
#     fl.server.start_server(
#         server_address=server_address,
#         config=fl.server.ServerConfig(num_rounds=rounds),
#         strategy=strategy,
#         certificates=certificates,
#     )


# if __name__ == "__main__":
#     main()