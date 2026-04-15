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

  3. Rapport final à la fin de tous les rounds
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

import flwr as fl


# ============================================================================
# ⚙️  CONFIGURATION
# ============================================================================

LOGS_DIR = Path(os.getenv("FL_LOGS_DIR", "logs/fl"))


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
        with open(self.csv_path, "w", newline="") as f:
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
        with open(self.csv_path, "a", newline="") as f:
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

        best_round = max(self.history, key=lambda r: r["accuracy"])
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
        with open(self.summary_path, "w") as f:
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
    """
    Extension de FedAvg qui intercepte les résultats de chaque round
    pour les logger via MetricsLogger.
    """

    def __init__(self, total_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.total_rounds  = total_rounds
        self.current_round = 0

    def aggregate_evaluate(self, server_round, results, failures):
        """Appelé après l'évaluation de chaque round."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        self.current_round = server_round

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
            _logger.log_round(server_round, eval_agg, fit_agg)
            _logger.print_round_report(server_round, self.total_rounds)

        return aggregated

    def aggregate_fit(self, server_round, results, failures):
        """Appelé après l'entraînement de chaque round."""
        aggregated = super().aggregate_fit(server_round, results, failures)

        # Log préliminaire des métriques fit (avant évaluation)
        fit_metrics_raw = [
            (res.num_examples, res.metrics)
            for _, res in results
            if res is not None
        ]
        fit_agg = aggregate_fit_metrics(fit_metrics_raw)

        # Pré-enregistrement pour que aggregate_evaluate puisse le lire
        if _logger:
            # Entrée temporaire dans l'historique (sera complétée par aggregate_evaluate)
            _logger.history.append({
                "round"          : server_round,
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

    rounds      = int(os.getenv("FL_ROUNDS",      "5"))
    min_clients = int(os.getenv("FL_MIN_CLIENTS", "2"))
    port        = int(os.getenv("FLOWER_PORT",    "8080"))
    server_address = f"0.0.0.0:{port}"

    _total_rounds = rounds
    _logger       = MetricsLogger(LOGS_DIR)

    print(f"\n{'═'*58}")
    print(f"  🚀 Serveur Flower")
    print(f"  Rounds : {rounds}  |  Min clients : {min_clients}")
    print(f"  Logs   : {LOGS_DIR.resolve()}")
    print(f"{'═'*58}\n")

    strategy = FedAvgWithLogging(
        total_rounds           = rounds,
        fraction_fit           = 1.0,
        fraction_evaluate      = 1.0,
        min_fit_clients        = min_clients,
        min_evaluate_clients   = min_clients,
        min_available_clients  = min_clients,
        evaluate_metrics_aggregation_fn = aggregate_eval_metrics,
        fit_metrics_aggregation_fn      = aggregate_fit_metrics,
    )

    certificates = _build_certificates()

    fl.server.start_server(
        server_address = server_address,
        config         = fl.server.ServerConfig(num_rounds=rounds),
        strategy       = strategy,
        certificates   = certificates,
    )

    # Rapport final après tous les rounds
    if _logger:
        _logger.save_summary(rounds)


if __name__ == "__main__":
    main()













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