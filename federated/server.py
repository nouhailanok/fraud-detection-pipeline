"""
server.py — Serveur Flower FL + Behavioral Analysis
═══════════════════════════════════════════════════
v2 — Intégration BehavioralAnalyzer (Isolation Forest)
  2a : Instanciation analyzer dans __init__
  2b : analyzer.analyze() dans aggregate_fit AVANT super()
  2c : set_prev_params() APRÈS super()
  2d : rapport BA dans _push_temp_entry + print_round_report

Fix double-log : aggregate_fit → _push_temp_entry (métriques fit + BA)
                 aggregate_evaluate → _complete_entry (métriques eval)
                 → 1 seule entrée par round dans JSON/CSV
"""

import os
import json
import csv
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import flwr as fl

# ── Import Behavioral Analysis ────────────────────────────────────────────────
try:
    from federated.behavioral_analysis import BehavioralAnalyzer
    BA_AVAILABLE = True
except ImportError:
    try:
        from behavioral_analysis import BehavioralAnalyzer
        BA_AVAILABLE = True
    except ImportError:
        BA_AVAILABLE = False
        print("⚠️  BehavioralAnalyzer non disponible — mettre behavioral_analysis.py dans federated/")


# ============================================================================
# ⚙️  CONFIGURATION
# ============================================================================

LOGS_DIR       = Path(os.getenv("FL_LOGS_DIR",       "logs/fl"))
CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "logs/fl/checkpoints"))
FL_PATIENCE    = int(os.getenv("FL_PATIENCE",         "5"))

BA_ACTIVATION_ROUND = int(os.getenv("BA_ACTIVATION_ROUND", "3"))
# BA_CONTAMINATION    = float(os.getenv("BA_CONTAMINATION",  "0.1"))

# BA_CONTAMINATION : si non défini, calculé depuis FL_MIN_CLIENTS (1 suspect max sur N)
# Avec 4 nœuds → 1/4 = 0.25 | Avec 2 nœuds → 1/2 = 0.5
# Isolation Forest exige contamination > 0 et <= 0.5
_default_contamination = round(
    1.0 / max(int(os.getenv("FL_MIN_CLIENTS", "4")), 2), 4
)
BA_CONTAMINATION = float(os.getenv("BA_CONTAMINATION", str(_default_contamination)))


# ============================================================================
# SNAPSHOT ENV
# ============================================================================

def snapshot_server_env(logs_dir: Path) -> None:
    keys = [
        "NOM_USER", "FL_USER", "FL_ROUNDS", "FL_MIN_CLIENTS", "FL_LOCAL_EPOCHS",
        "FLOWER_PORT", "FLOWER_TLS_REQUIRE_CLIENT_CERT", "FL_PATIENCE",
        "FL_RESUME", "FL_RESUME_FROM", "FL_CHECKPOINT_PATH",
        "FL_LOGS_DIR", "FL_CHECKPOINT_DIR", "PYTHONPATH",
        "BA_ACTIVATION_ROUND", "BA_CONTAMINATION",
    ]
    env_path = logs_dir / "server_env.txt"
    env_path.write_text("\n".join(f"{k}={os.getenv(k, '')}" for k in keys) + "\n",
                        encoding="utf-8")


# ============================================================================
# 💾 CHECKPOINT
# ============================================================================

def save_checkpoint(server_round: int, parameters) -> None:
    if parameters is None:
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import flwr.common
        ndarrays = flwr.common.parameters_to_ndarrays(parameters)
        arrays   = {f"param_{i}": arr for i, arr in enumerate(ndarrays)}
        np.savez(str(CHECKPOINT_DIR / f"global_model_round_{server_round:03d}.npz"), **arrays)
        np.savez(str(CHECKPOINT_DIR / "global_model_last.npz"), **arrays)
        (CHECKPOINT_DIR / "last_round.txt").write_text(str(server_round), encoding="utf-8")
        print(f"  💾 Checkpoint round {server_round}")
    except Exception as e:
        print(f"  ⚠️  Checkpoint error round {server_round} : {e}")

    
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
        
        # FIX : Tri numérique (0, 1, 2... 10) et non alphabétique (0, 1, 10... 2)
        sorted_keys = sorted(data.files, key=lambda x: int(x.split('_')[1]))
        parameters = [data[k] for k in sorted_keys]

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
    return load_checkpoint_from(str(CHECKPOINT_DIR))


# ============================================================================
# 📁 LOGGER — Fix double-log : _push_temp_entry + _complete_entry
# ============================================================================

class MetricsLogger:
    def __init__(self, logs_dir: Path):
        self.logs_dir   = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.history    : list = []
        self.start_time = time.time()
        self.json_path    = logs_dir / "fl_metrics.json"
        self.csv_path     = logs_dir / "fl_metrics.csv"
        self.summary_path = logs_dir / "fl_summary.txt"

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields()).writeheader()
        print(f"📁 Logs FL → {self.logs_dir.resolve()}")

    def _csv_fields(self) -> list:
        return [
            "round", "timestamp",
            "accuracy", "loss", "recall", "precision", "f1",
            "n_clients_eval", "n_examples_eval",
            "train_loss", "n_clients_fit",
            "avg_epsilon", "max_epsilon", "epsilon_status",
            "ba_suspects", "ba_active",
            "elapsed_sec",
        ]

    def _push_temp_entry(self, round_num: int, fit_agg: dict, ba_report: dict) -> None:
        """
        Étape 1/2 : enregistre fit + behavioral dans l'historique en mémoire.
        Les métriques eval sont à 0 — elles seront complétées par _complete_entry().
        """
        entry = {
            "round"          : round_num,
            "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy"       : 0.0, "loss": 0.0,
            "recall"         : 0.0, "precision": 0.0, "f1": 0.0,
            "n_clients_eval" : 0,   "n_examples_eval": 0,
            "train_loss"     : round(fit_agg.get("train_loss",  0.0), 4),
            "n_clients_fit"  : fit_agg.get("n_clients", 0),
            "avg_epsilon"    : round(fit_agg.get("avg_epsilon", 0.0), 4),
            "max_epsilon"    : round(fit_agg.get("max_epsilon", 0.0), 4),
            "epsilon_status" : "OK" if fit_agg.get("max_epsilon", 0) <= 1.0 else "WARNING",
            "ba_suspects"    : ";".join(ba_report.get("suspects", [])),
            "ba_active"      : ba_report.get("active", False),
            "elapsed_sec"    : round(time.time() - self.start_time, 1),
        }
        self.history.append(entry)

    def _complete_entry(self, round_num: int, eval_agg: dict) -> None:
        """
        Étape 2/2 : met à jour l'entrée temporaire avec les métriques eval,
        puis sauvegarde JSON + CSV.
        """
        entry = None
        for e in reversed(self.history):
            if e["round"] == round_num:
                entry = e
                break
        if entry is None:
            entry = {"round": round_num,
                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            self.history.append(entry)

        entry.update({
            "accuracy"       : round(eval_agg.get("accuracy",  0.0), 4),
            "loss"           : round(eval_agg.get("loss",      0.0), 4),
            "recall"         : round(eval_agg.get("recall",    0.0), 4),
            "precision"      : round(eval_agg.get("precision", 0.0), 4),
            "f1"             : round(eval_agg.get("f1",        0.0), 4),
            "n_clients_eval" : eval_agg.get("n_clients",  0),
            "n_examples_eval": eval_agg.get("n_examples", 0),
            "elapsed_sec"    : round(time.time() - self.start_time, 1),
        })

        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writerow({k: entry.get(k, "") for k in self._csv_fields()})

    def print_round_report(self, round_num: int, total_rounds: int,
                           ba_report: dict) -> None:
        entry = next((e for e in reversed(self.history) if e["round"] == round_num), None)
        if not entry:
            return

        eps_icon = "✅" if entry.get("epsilon_status") == "OK" else "⚠️"
        print(f"\n{'─'*58}")
        print(f"  📊  Round {round_num}/{total_rounds}  —  {entry.get('timestamp', '')}")
        print(f"{'─'*58}")
        print(f"  Évaluation ({entry.get('n_clients_eval', 0)} nœuds · "
              f"{entry.get('n_examples_eval', 0):,} txns)")
        print(f"    Accuracy  : {entry.get('accuracy', 0):.4f}")
        print(f"    Loss      : {entry.get('loss', 0):.4f}")
        print(f"    Recall    : {entry.get('recall', 0):.4f}")
        print(f"    Precision : {entry.get('precision', 0):.4f}")
        print(f"    F1-score  : {entry.get('f1', 0):.4f}")
        print(f"  Entraînement ({entry.get('n_clients_fit', 0)} nœuds)")
        print(f"    Train Loss : {entry.get('train_loss', 0):.4f}")
        print(f"  Privacy {eps_icon}")
        print(f"    Avg ε  : {entry.get('avg_epsilon', 0):.4f}")
        print(f"    Max ε  : {entry.get('max_epsilon', 0):.4f}  [{entry.get('epsilon_status', '')}]")

        # Rapport Behavioral
        suspects  = ba_report.get("suspects", [])
        ba_active = ba_report.get("active", False)
        if ba_active:
            ba_icon = "🚨" if suspects else "✅"
            print(f"  Behavioral Analysis {ba_icon}")
            if suspects:
                scores = ba_report.get("scores", {})
                for cid in suspects:
                    feats = ba_report.get("features", {}).get(cid, {})
                    print(f"    ⚠️  {cid} → norm_L2={feats.get('norm_L2', 0):.3f}  "
                          f"cos_sim={feats.get('cos_sim', 0):.3f}  "
                          f"score_IF={scores.get(cid, 0):.3f}")
            else:
                print(f"    Tous les nœuds OK ({len(ba_report.get('features', {}))} analysés)")
        else:
            print(f"  Behavioral Analysis 🔬 calibration (round {round_num} < {BA_ACTIVATION_ROUND})")

        print(f"  Temps : {entry.get('elapsed_sec', 0)}s")
        print(f"{'─'*58}")

    def save_summary(self, total_rounds: int) -> None:
        if not self.history:
            return
        complete   = [r for r in self.history if r.get("n_clients_eval", 0) > 0] or self.history
        best_round = max(complete, key=lambda r: r.get("f1", 0.0))
        total_time = round(time.time() - self.start_time, 1)

        lines = [
            "=" * 58, "  RAPPORT FINAL — Federated Learning", "=" * 58,
            f"  Rounds effectués  : {len(complete)} / {total_rounds}",
            f"  Temps total       : {total_time}s",
            "", "  Meilleur round :",
            f"    Round     : {best_round.get('round')}",
            f"    F1-score  : {best_round.get('f1', 0):.4f}",
            f"    Accuracy  : {best_round.get('accuracy', 0):.4f}",
            f"    Recall    : {best_round.get('recall', 0):.4f}",
            "", "  Évolution ε par round :",
        ]
        for r in complete:
            icon    = "✅" if r.get("epsilon_status") == "OK" else "⚠️"
            ba_flag = " 🚨" if r.get("ba_suspects") else ""
            lines.append(f"    Round {r.get('round'):>2} : "
                         f"avg={r.get('avg_epsilon', 0):.4f}  "
                         f"max={r.get('max_epsilon', 0):.4f}  {icon}{ba_flag}")

        lines += ["", "  Fichiers de log :",
                  f"    JSON       : {self.json_path}",
                  f"    CSV        : {self.csv_path}",
                  f"    Behavioral : {self.logs_dir / 'behavioral_analysis.json'}",
                  "=" * 58]

        summary_text = "\n".join(lines)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"\n{summary_text}")
        print(f"\n📄 Résumé → {self.summary_path}")


# ============================================================================
# 📊 AGRÉGATION MÉTRIQUES
# ============================================================================

_logger       : Optional[MetricsLogger] = None
_total_rounds : int = 0


def aggregate_eval_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    wavg = lambda k: sum(n * m.get(k, 0.0) for n, m in metrics) / total
    return {
        "accuracy" : wavg("accuracy"), "loss"     : wavg("loss"),
        "recall"   : wavg("recall"),   "precision": wavg("precision"),
        "f1"       : wavg("f1"),
        "n_clients": len(metrics),     "n_examples": total,
    }


def aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    if not metrics:
        return {}
    eps    = [m["epsilon"]    for _, m in metrics if "epsilon"    in m]
    losses = [m["train_loss"] for _, m in metrics if "train_loss" in m]
    agg = {
        "n_clients"  : len(metrics),
        "avg_epsilon": sum(eps) / len(eps)       if eps    else 0.0,
        "max_epsilon": max(eps)                   if eps    else 0.0,
        "train_loss" : sum(losses) / len(losses) if losses else 0.0,
    }
    if eps:
        icon = "✅" if agg["max_epsilon"] <= 1.0 else "⚠️"
        print(f"\n🔐 Privacy {icon}  avg_ε={agg['avg_epsilon']:.4f}  max_ε={agg['max_epsilon']:.4f}")
    return agg


# ============================================================================
# 🧠 STRATEGY — FedAvg + BA + Logging
# ============================================================================

class FedAvgWithLogging(fl.server.strategy.FedAvg):

    def __init__(self, total_rounds: int, patience: int = 5,
                 round_offset: int = 0, min_clients: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.total_rounds      = total_rounds
        self.round_offset      = int(round_offset)
        self.current_round     = 0
        self.patience          = patience
        self.best_f1           = 0.0
        self.best_round        = 0
        self.rounds_no_improve = 0

        # ── 2a : BehavioralAnalyzer ───────────────────────────────────────────
        self._last_ba_report  : Dict = {}
        self._prev_agg_params = None  # poids globaux N-1 pour calcul ∆W
        self._blacklisted     : set  = set()  # nœuds blacklistés définitivement

        if BA_AVAILABLE and os.getenv("BA_ENABLED", "true").lower() == "true":
            self.analyzer = BehavioralAnalyzer(
                n_clients        = min_clients,
                activation_round = BA_ACTIVATION_ROUND,
                contamination    = BA_CONTAMINATION,
                n_estimators     = 100,
                logs_dir         = str(LOGS_DIR),
            )
            print(f"🔍 BehavioralAnalyzer initialisé "
                  f"(activation round {BA_ACTIVATION_ROUND}, contamination={BA_CONTAMINATION})")
        else:
            self.analyzer = None

    # ── aggregate_evaluate ────────────────────────────────────────────────────

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated   = super().aggregate_evaluate(server_round, results, failures)
        global_round = int(server_round) + self.round_offset
        self.current_round = global_round

        eval_metrics_raw = [
            (res.num_examples, res.metrics)
            for _, res in results if res is not None
        ]
        eval_agg = aggregate_eval_metrics(eval_metrics_raw)

        if _logger:
            # Complète l'entrée temporaire créée dans aggregate_fit (fix double-log)
            _logger._complete_entry(global_round, eval_agg)
            _logger.print_round_report(global_round, self.total_rounds, self._last_ba_report)

        # Early stopping F1
        current_f1 = eval_agg.get("f1", 0.0)
        if current_f1 > self.best_f1 + 1e-4:
            self.best_f1           = current_f1
            self.best_round        = global_round
            self.rounds_no_improve = 0
            print(f"  ✅ Nouveau meilleur F1 : {self.best_f1:.4f} (round {global_round})")
            try:
                src = CHECKPOINT_DIR / f"global_model_round_{global_round:03d}.npz"
                dst = CHECKPOINT_DIR / "global_model_best.npz"
                if src.exists():
                    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
            except Exception as e:
                print(f"  ⚠️  global_model_best.npz : {e}")
        else:
            self.rounds_no_improve += 1
            print(f"  ⏳ F1 sans amélioration : {self.rounds_no_improve}/{self.patience} "
                  f"(best={self.best_f1:.4f} @ {self.best_round} | current={current_f1:.4f})")

            if self.rounds_no_improve >= self.patience:
                print(f"\n{'═'*58}")
                print(f"  🛑 EARLY STOPPING FL — {self.patience} rounds sans amélioration")
                print(f"  Meilleur F1 : {self.best_f1:.4f} au round {self.best_round}")
                print(f"{'═'*58}\n")
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                (CHECKPOINT_DIR / "early_stop.txt").write_text(
                    f"Early stop round {global_round}\nBest F1={self.best_f1:.4f} @ {self.best_round}\n",
                    encoding="utf-8"
                )

        return aggregated

    # ── aggregate_fit ─────────────────────────────────────────────────────────

    def aggregate_fit(self, server_round, results, failures):
        global_round = int(server_round) + self.round_offset

        if not results:
            print(f"⚠️ Aucun résultat round {global_round}")
            return None

        valid_results = [(c, r) for c, r in results if int(getattr(r, "num_examples", 0) or 0) > 0]
        zero_results  = [(c, r) for c, r in results if int(getattr(r, "num_examples", 0) or 0) == 0]

        if not valid_results:
            exhausted = sum(1 for _, r in zero_results
                            if getattr(r, "metrics", {}).get("dp_exhausted") is True)
            print(f"\n{'═'*58}")
            print("  🛑 ARRÊT FL — aucun client actif")
            if exhausted:
                print(f"  Budget DP épuisé : {exhausted}/{len(zero_results)} clients")
            print(f"{'═'*58}\n")
            try:
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                best_alias = CHECKPOINT_DIR / "global_model_best.npz"
                if self.best_round > 0:
                    src = CHECKPOINT_DIR / f"global_model_round_{self.best_round:03d}.npz"
                    if src.exists():
                        shutil.copyfile(src, best_alias)
                (CHECKPOINT_DIR / "dp_budget_exhausted.txt").write_text(
                    f"Arrêt FL round {global_round}\n", encoding="utf-8"
                )
            except Exception:
                pass
            if _logger:
                try:
                    _logger.save_summary(self.total_rounds)
                except Exception:
                    pass
            raise SystemExit(0)

        # ── 2b : Behavioral Analysis AVANT super() ───────────────────────────
        ba_report = {}
        if self.analyzer is not None:
            try:
                ba_report = self.analyzer.analyze(
                    server_round = global_round,
                    results      = valid_results,
                    prev_params  = self._prev_agg_params,
                )
            except Exception as e:
                print(f"  ⚠️  BehavioralAnalyzer.analyze() : {e}")

        self._last_ba_report = ba_report

        # ── Appliquer les décisions BA (EXCLUDE / BLACKLIST) ─────────────────
        # Les décisions sont calculées dans behavioral_analysis.py
        # On filtre valid_results AVANT FedAvg pour exclure les nœuds suspects
        if ba_report and ba_report.get("decisions"):
            decisions = ba_report["decisions"]   # dict : client_id → décision
 
            # Mettre à jour le blacklist permanent
            for cid, decision in decisions.items():
                if decision == "BLACKLIST" and cid not in self._blacklisted:
                    self._blacklisted.add(cid)
                    print(f"  🚫 Node {cid} ajouté au BLACKLIST permanent")
 
            # Filtrer les résultats selon les décisions
            filtered_results = []
            for client, fit_res in valid_results:
                metrics   = getattr(fit_res, "metrics", {}) or {}
                client_id = str(metrics.get("client_id", "?"))
 
                if client_id in self._blacklisted:
                    print(f"  🚫 Node {client_id} BLACKLISTÉ — exclu de FedAvg")
                    continue
 
                decision = decisions.get(client_id, "NORMAL")
                if decision == "EXCLUDE":
                    attack = ba_report.get("attack_types", {}).get(client_id, "?")
                    print(f"  ⛔ Node {client_id} EXCLU ce round ({attack})")
                    continue
 
                filtered_results.append((client, fit_res))
 
            # Si tous les nœuds sont exclus → garder les valides originaux
            # pour éviter un FedAvg vide (cas extrême)
            if not filtered_results:
                print(f"  ⚠️  Tous les nœuds exclus — FedAvg sur résultats originaux")
                filtered_results = valid_results
 
            valid_results = filtered_results

        # ── FedAvg standard ──────────────────────────────────────────────────
        aggregated = super().aggregate_fit(server_round, valid_results, failures)

        # ── 2c : Mémorisation poids agrégés APRÈS super() ───────────────────
        if aggregated is not None:
            try:
                import flwr.common
                parameters, _ = aggregated
                self._prev_agg_params = flwr.common.parameters_to_ndarrays(parameters)
            except Exception as e:
                print(f"  ⚠️  prev_params : {e}")

            if self.analyzer is not None:
                try:
                    self.analyzer.set_prev_params(aggregated)
                except Exception:
                    pass

        # ── Métriques fit ─────────────────────────────────────────────────────
        fit_metrics_raw = [
            (res.num_examples, res.metrics)
            for _, res in valid_results if res is not None
        ]
        fit_agg = aggregate_fit_metrics(fit_metrics_raw)

        # ── 2d : _push_temp_entry avec BA ────────────────────────────────────
        if _logger:
            _logger._push_temp_entry(global_round, fit_agg, ba_report)

        # Checkpoint
        if aggregated is not None:
            parameters, _ = aggregated
            save_checkpoint(global_round, parameters)

        return aggregated


# ============================================================================
# 🔐 TLS
# ============================================================================

def _read_cert(path: Optional[str]) -> Optional[bytes]:
    if not path:
        return None
    p = Path(path)
    return p.read_bytes() if p.exists() else None


def _build_certificates() -> Optional[Tuple[bytes, bytes, bytes]]:
    ca   = _read_cert(os.getenv("FLOWER_TLS_CA_CERT"))
    cert = _read_cert(os.getenv("FLOWER_TLS_SERVER_CERT"))
    key  = _read_cert(os.getenv("FLOWER_TLS_SERVER_KEY"))
    if cert is None or key is None:
        return None
    if os.getenv("FLOWER_TLS_REQUIRE_CLIENT_CERT", "false").lower() == "true" and ca is None:
        raise RuntimeError("FLOWER_TLS_REQUIRE_CLIENT_CERT=true mais FLOWER_TLS_CA_CERT absent")
    return (ca, cert, key) if ca is not None else None


# ============================================================================
# 🚀 MAIN
# ============================================================================

def main():
    global _logger, _total_rounds

    total_rounds_target = int(os.getenv("FL_ROUNDS",       "5"))
    min_clients         = int(os.getenv("FL_MIN_CLIENTS",  "2"))
    local_epochs        = int(os.getenv("FL_LOCAL_EPOCHS", "3"))
    port                = int(os.getenv("FLOWER_PORT",     "8080"))

    _total_rounds = total_rounds_target
    _logger       = MetricsLogger(LOGS_DIR)
    snapshot_server_env(LOGS_DIR)

    # Reprise checkpoint
    resume_from = (os.getenv("FL_RESUME_FROM") or os.getenv("FL_CHECKPOINT_PATH") or "").strip()
    resume      = os.getenv("FL_RESUME", "false").lower() == "true" or bool(resume_from)
    saved_params, last_round = ((None, 0))
    if resume_from:
        saved_params, last_round = load_checkpoint_from(resume_from)
    elif resume:
        saved_params, last_round = load_checkpoint()

    round_offset  = int(last_round or 0)
    rounds_to_run = total_rounds_target

    if saved_params is not None and round_offset > 0:
        remaining = total_rounds_target - round_offset
        if remaining <= 0:
            print(f"\n✅ Entraînement déjà complet ({round_offset} rounds).")
            return
        print(f"\n🔄 Reprise depuis round {round_offset} → {remaining} rounds restants")
        rounds_to_run = remaining
    else:
        saved_params = None
        round_offset = 0
        print(f"\n🆕 Nouvel entraînement")

    print(f"\n{'═'*58}")
    print(f"  🚀 Serveur Flower")
    print(f"  Rounds : {rounds_to_run} (objectif={total_rounds_target})  "
          f"|  Clients : {min_clients}  |  Epochs : {local_epochs}")
    print(f"  Behavioral : activation round {BA_ACTIVATION_ROUND}  contamination={BA_CONTAMINATION}")
    print(f"  Logs : {LOGS_DIR.resolve()}")
    print(f"{'═'*58}\n")

    def fit_config(server_round: int) -> dict:
        return {"local_epochs": local_epochs, "round": int(server_round) + round_offset}

    def eval_config(server_round: int) -> dict:
        return {"round": int(server_round) + round_offset}

    initial_parameters = None
    if saved_params is not None:
        import flwr.common
        initial_parameters = flwr.common.ndarrays_to_parameters(saved_params)

    strategy = FedAvgWithLogging(
        total_rounds              = total_rounds_target,
        patience                  = int(os.getenv("FL_PATIENCE", "5")),
        round_offset              = round_offset,
        min_clients               = min_clients,
        fraction_fit              = 1.0,
        fraction_evaluate         = 1.0,
        min_fit_clients           = min_clients,
        min_evaluate_clients      = min_clients,
        min_available_clients     = min_clients,
        on_fit_config_fn          = fit_config,
        on_evaluate_config_fn     = eval_config,
        initial_parameters        = initial_parameters,
        evaluate_metrics_aggregation_fn = aggregate_eval_metrics,
        fit_metrics_aggregation_fn      = aggregate_fit_metrics,
    )

    fl.server.start_server(
        server_address = f"0.0.0.0:{port}",
        config         = fl.server.ServerConfig(num_rounds=rounds_to_run),
        strategy       = strategy,
        certificates   = _build_certificates(),
    )

    if _logger:
        _logger.save_summary(total_rounds_target)

    # Résumé behavioral final
    if BA_AVAILABLE and strategy.analyzer is not None:
        ba_sum = strategy.analyzer.get_summary()
        print(f"\n🔍 Behavioral — {ba_sum['total_alerts']} alertes / {ba_sum['total_rounds']} rounds")
        for rnd, suspects in ba_sum.get("suspects_by_round", {}).items():
            print(f"   Round {rnd} : {', '.join(suspects)}")
        print(f"   Rapport : {LOGS_DIR / 'behavioral_analysis.json'}")


if __name__ == "__main__":
    main()

