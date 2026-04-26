"""
mlops/monitoring/fl_metrics_exporter.py

Expose les métriques Federated Learning au format Prometheus.
BentoML expose ses métriques automatiquement sur /metrics.
Ce script expose les métriques FL custom sur localhost:8001/metrics.

Lancement :
    python mlops/monitoring/fl_metrics_exporter.py

Vérification :
    curl http://localhost:8001/metrics
"""

import json
import time
import glob
from pathlib import Path
from prometheus_client import start_http_server, Gauge, REGISTRY
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily


# ================================================================
# CONFIGURATION
# ================================================================

PORT          = 8001
REFRESH_SEC   = 30
RUNS_DIR      = Path("logs/runs")          # dossier contenant les runs/
BEHAVIORAL_F  = Path("logs/fl/behavioral_analysis.json")


# ================================================================
# MÉTRIQUES PROMETHEUS
# ================================================================
# Chaque métrique porte un label run="{run_name}" pour distinguer
# les différentes expériences (2026-04-19_19-41-52_Imane, etc.)

fl_recall      = Gauge("fl_recall",          "Recall FL du run",              ["run"])
fl_f1          = Gauge("fl_f1",              "F1 score FL du run",            ["run"])
fl_precision   = Gauge("fl_precision",       "Precision FL du run",           ["run"])
fl_epsilon_max = Gauge("fl_epsilon_max",     "Epsilon DP max atteint",        ["run"])
fl_rounds      = Gauge("fl_rounds_total",    "Nombre de rounds FL",           ["run"])
fl_ba_alerts   = Gauge("fl_ba_alerts_total", "Alertes behavioral analysis",   ["run"])
fl_trust_score = Gauge("fl_trust_score",     "Trust score du run (0 à 1)",    ["run"])


# ================================================================
# FONCTIONS DE LECTURE
# ================================================================

def read_fl_metrics(run_dir: Path) -> dict:
    """
    Lit fl_metrics.json d'un run et retourne les dernières métriques.
    Retourne un dict vide si le fichier est absent ou corrompu.
    """
    metrics_path = run_dir / "fl_metrics.json"
    if not metrics_path.exists():
        return {}

    try:
        with open(metrics_path) as f:
            data = json.load(f)

        # fl_metrics.json peut être une liste de rounds ou un dict final
        if isinstance(data, list) and len(data) > 0:
            last = data[-1]               # dernier round disponible
        elif isinstance(data, dict):
            last = data
        else:
            return {}

        return {
            "recall":        float(last.get("recall",        0.0)),
            "f1":            float(last.get("f1",            0.0)),
            "precision":     float(last.get("precision",     0.0)),
            "epsilon_final": float(last.get("epsilon_final", 0.0)),
            "round_number":  int(last.get("round_number",    0)),
        }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"⚠️  Erreur lecture {metrics_path}: {e}")
        return {}


def read_behavioral(run_name: str) -> dict:
    """
    Lit behavioral_analysis.json et retourne les infos du run.
    Calcule le trust_score = 1 - (n_rounds_suspects / total_rounds).
    """
    if not BEHAVIORAL_F.exists():
        return {"ba_alerts": 0, "trust_score": 1.0}

    try:
        with open(BEHAVIORAL_F) as f:
            data = json.load(f)

        # behavioral_analysis.json peut contenir plusieurs runs
        # Structure attendue : liste de dicts avec run_name + rounds
        if isinstance(data, list):
            for entry in data:
                if entry.get("run_name") == run_name:
                    return _compute_trust(entry)

        elif isinstance(data, dict):
            if data.get("run_name") == run_name:
                return _compute_trust(data)

        return {"ba_alerts": 0, "trust_score": 1.0}

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"⚠️  Erreur lecture behavioral: {e}")
        return {"ba_alerts": 0, "trust_score": 1.0}


def _compute_trust(entry: dict) -> dict:
    """
    Calcule le trust_score à partir des rounds suspects.
    trust_score = 1 - (n_alerts / total_rounds)
    """
    total_rounds = int(entry.get("total_rounds", 1))
    ba_suspects  = int(entry.get("ba_suspects",  0))    # rounds avec anomalie

    n_alerts    = ba_suspects
    trust_score = max(0.0, 1.0 - (n_alerts / max(total_rounds, 1)))

    return {
        "ba_alerts":   n_alerts,
        "trust_score": round(trust_score, 4),
    }


# ================================================================
# BOUCLE DE MISE À JOUR
# ================================================================

def update_metrics():
    """
    Scanne tous les runs disponibles dans logs/runs/
    et met à jour les métriques Prometheus.
    """
    if not RUNS_DIR.exists():
        print(f"⚠️  Dossier {RUNS_DIR} introuvable — en attente...")
        return

    run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]

    if not run_dirs:
        print("⚠️  Aucun run trouvé dans logs/runs/")
        return

    for run_dir in run_dirs:
        run_name = run_dir.name
        metrics  = read_fl_metrics(run_dir)

        if not metrics:
            continue

        behavioral = read_behavioral(run_name)

        # ── Mise à jour des gauges Prometheus ──────────────────────
        fl_recall.labels(run=run_name).set(metrics["recall"])
        fl_f1.labels(run=run_name).set(metrics["f1"])
        fl_precision.labels(run=run_name).set(metrics["precision"])
        fl_epsilon_max.labels(run=run_name).set(metrics["epsilon_final"])
        fl_rounds.labels(run=run_name).set(metrics["round_number"])
        fl_ba_alerts.labels(run=run_name).set(behavioral["ba_alerts"])
        fl_trust_score.labels(run=run_name).set(behavioral["trust_score"])

        print(
            f"✅ {run_name} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"ε={metrics['epsilon_final']:.4f} | "
            f"trust={behavioral['trust_score']:.4f}"
        )


# ================================================================
# POINT D'ENTRÉE
# ================================================================

if __name__ == "__main__":
    print(f"🚀 FL Metrics Exporter démarré sur le port {PORT}")
    print(f"📂 Lecture des runs depuis : {RUNS_DIR.resolve()}")
    print(f"📊 Métriques disponibles sur : http://localhost:{PORT}/metrics")
    print(f"🔄 Rafraîchissement toutes les {REFRESH_SEC}s\n")

    # Démarrer le serveur HTTP Prometheus
    start_http_server(PORT)

    # Première mise à jour immédiate
    update_metrics()

    # Boucle infinie de rafraîchissement
    while True:
        time.sleep(REFRESH_SEC)
        print(f"\n🔄 Mise à jour des métriques...")
        update_metrics()
