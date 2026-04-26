"""
behavioral_analysis.py — Analyse comportementale des nœuds FL
Isolation Forest sur 6 features de gradient pour détecter les nœuds suspects.

Cas détectés : FREE_RIDER, SIGN_FLIP, SCALE, NOISE, BYZANTINE
6 features : norm_L2, norm_L1, cos_sim, var_delta, train_loss, epsilon
Politique : ALERT uniquement (pas d'exclusion)
Activation : round >= activation_round (défaut 3)
Sauvegarde : logs/fl/behavioral_analysis.json
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn non installé → BehavioralAnalyzer désactivé")


class BehavioralAnalyzer:
    def __init__(self, n_clients=4, activation_round=3,
                 contamination=0.1, n_estimators=100, logs_dir="logs/fl"):
        self.n_clients        = n_clients
        self.activation_round = activation_round
        self.contamination    = contamination
        self.n_estimators     = n_estimators
        self.logs_dir         = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._prev_params : Optional[List[np.ndarray]] = None
        self._history     : List[Dict] = []
        self._reports     : List[Dict] = []
        self._log_path    = self.logs_dir / "behavioral_analysis.json"

        # Décisions et blacklist
        self._blacklist          : set             = set()
        self._consecutive_alerts : Dict[str, int]  = {}
        self._attack_counts      : Dict[str, Dict] = {}

    def set_prev_params(self, aggregated_params) -> None:
        """Mémorise les poids globaux après aggregate_fit."""
        if aggregated_params is None:
            return
        try:
            import flwr.common
            parameters, _ = aggregated_params
            self._prev_params = flwr.common.parameters_to_ndarrays(parameters)
        except Exception as e:
            print(f"  ⚠️  BA.set_prev_params() : {e}")

    def analyze(self, server_round: int, results: list,
                prev_params: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Calcule les 6 features par nœud et détecte les suspects via IF.
        Appelé AVANT super().aggregate_fit().
        """
        if not SKLEARN_AVAILABLE:
            return self._empty_report(server_round, "sklearn_missing")

        if prev_params is None:
            prev_params = self._prev_params

        node_features = self._extract_features(results, prev_params)
        if not node_features:
            return self._empty_report(server_round, "no_results")

        for client_id, feats in node_features.items():
            self._history.append({"round": server_round, "client_id": client_id, **feats})

        report = {
            "round"     : server_round,
            "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active"    : server_round >= self.activation_round,
            "features"  : node_features,
            "suspects"  : [],
            "scores"    : {},
            "n_points"  : len(self._history),
        }

        if server_round >= self.activation_round and len(self._history) >= self.n_clients:
            suspects, scores = self._run_isolation_forest(node_features)
            report["suspects"] = suspects
            report["scores"]   = scores

            # ── Classification + Décisions ────────────────────────────────
            attack_types = {}
            decisions    = {}

            for cid in list(node_features.keys()):
                feats    = node_features[cid]
                if_score = scores.get(cid, 0.0)
                attack   = self.classify_attack(cid, feats, if_score) if cid in suspects else "NORMAL"
                decision = self.get_decision(cid, attack)
                attack_types[cid] = attack
                decisions[cid]    = decision

            report["attack_types"] = attack_types
            report["decisions"]    = decisions

            if suspects:
                print(f"\n  🚨 BEHAVIORAL ALERT — Round {server_round}")
                for cid in suspects:
                    feats    = node_features.get(cid, {})
                    attack   = attack_types.get(cid, "?")
                    decision = decisions.get(cid, "?")
                    print(f"     Nœud {cid} → {attack} → {decision}"
                          f"  norm_L2={feats.get('norm_L2', 0):.4f}"
                          f"  cos_sim={feats.get('cos_sim', 0):.4f}"
                          f"  score_IF={scores.get(cid, 0):.4f}")
            else:
                print(f"  ✅ Behavioral OK — Round {server_round} "
                      f"({len(node_features)} nœuds)")
        else:
            reason = (f"calibration (round {server_round} < {self.activation_round})"
                      if server_round < self.activation_round
                      else "pas assez de points")
            report["reason"]       = reason
            report["attack_types"] = {}
            report["decisions"]    = {}
            print(f"  🔬 Behavioral — {reason}")

        self._reports.append(report)
        self._save_reports()
        return report

    def get_summary(self) -> Dict:
        if not self._reports:
            return {"total_rounds": 0, "total_alerts": 0, "suspects_by_round": {}}
        alerts = [r for r in self._reports if r.get("suspects")]
        return {
            "total_rounds"      : len(self._reports),
            "total_alerts"      : len(alerts),
            "suspects_by_round" : {r["round"]: r["suspects"] for r in alerts},
        }

    def _extract_features(self, results, prev_params) -> Dict[str, Dict]:
        import flwr.common
        node_features = {}
        delta_weights = {}

        for _, fit_res in results:
            if fit_res is None:
                continue
            metrics   = getattr(fit_res, "metrics", {}) or {}
            client_id = str(metrics.get("client_id", f"node_{len(delta_weights)+1}"))
            try:
                params = flwr.common.parameters_to_ndarrays(fit_res.parameters)
            except Exception:
                continue

            if prev_params is not None and len(params) == len(prev_params):
                delta = [p - g for p, g in zip(params, prev_params)]
            else:
                delta = params

            delta_flat = np.concatenate([d.flatten() for d in delta])
            delta_flat = np.nan_to_num(delta_flat, nan=0.0, posinf=1e6, neginf=-1e6)

            delta_weights[client_id] = delta_flat
            node_features[client_id] = {
                "norm_L2"    : float(np.linalg.norm(delta_flat, ord=2)),
                "norm_L1"    : float(np.linalg.norm(delta_flat, ord=1)),
                "var_delta"  : float(np.var(delta_flat)),
                "train_loss" : float(metrics.get("train_loss", 0.0)),
                "epsilon"    : float(metrics.get("epsilon", 0.0)),
                "cos_sim"    : 0.0,
            }

        if len(delta_weights) > 1:
            ids     = list(delta_weights.keys())
            deltas  = list(delta_weights.values())
            min_len = min(len(d) for d in deltas)
            deltas  = [d[:min_len] for d in deltas]
            mean_d  = np.mean(deltas, axis=0)
            for i, cid in enumerate(ids):
                d    = deltas[i]
                norm = np.linalg.norm(d) * np.linalg.norm(mean_d)
                node_features[cid]["cos_sim"] = float(np.dot(d, mean_d) / norm) if norm > 1e-8 else 0.0

        return node_features

    def _run_isolation_forest(self, node_features) -> Tuple[List[str], Dict[str, float]]:
        KEYS = ["norm_L2", "norm_L1", "cos_sim", "var_delta", "train_loss", "epsilon"]

        X_history = np.array(
            [[p.get(k, 0.0) for k in KEYS] for p in self._history], dtype=np.float32
        )
        X_history = np.nan_to_num(X_history, nan=0.0, posinf=1e6, neginf=-1e6)

        iso = IsolationForest(n_estimators=self.n_estimators,
                              contamination=self.contamination, random_state=42)
        iso.fit(X_history)

        ids     = list(node_features.keys())
        X_round = np.array(
            [[node_features[cid].get(k, 0.0) for k in KEYS] for cid in ids], dtype=np.float32
        )
        X_round     = np.nan_to_num(X_round, nan=0.0, posinf=1e6, neginf=-1e6)
        predictions = iso.predict(X_round)
        scores      = iso.score_samples(X_round)

        return ([ids[i] for i, p in enumerate(predictions) if p == -1],
                {ids[i]: float(scores[i]) for i in range(len(ids))})

    def _empty_report(self, server_round, reason=""):
        return {"round": server_round, "active": False, "features": {},
                "suspects": [], "scores": {}, "n_points": 0, "reason": reason,
                "attack_types": {}, "decisions": {},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    def _save_reports(self):
        try:
            with open(self._log_path, "w", encoding="utf-8") as f:
                json.dump(self._reports, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  ⚠️  BA save error : {e}")

    def _get_mean_norm(self) -> float:
        norms = [p.get("norm_L2", 0.0) for p in self._history[-20:]]
        return float(np.mean(norms)) if norms else 1.0

    def _get_mean_var(self) -> float:
        vars_ = [p.get("var_delta", 0.0) for p in self._history[-20:]]
        return float(np.mean(vars_)) if vars_ else 1.0

    def classify_attack(self, client_id, features, if_score):
        norm_L2   = features["norm_L2"]
        cos_sim   = features["cos_sim"]
        var_delta = features["var_delta"]
        mean_norm = self._get_mean_norm()

        if norm_L2 < 1e-3:
            return "FREE_RIDER"
        elif cos_sim < -0.5:
            return "SIGN_FLIP"
        elif norm_L2 > mean_norm * 10:
            return "SCALE"
        elif var_delta > self._get_mean_var() * 5:
            return "NOISE"
        elif if_score < -0.50:
            return "BYZANTINE"
        else:
            return "NORMAL"
        
    def get_decision(self, client_id, attack_type):
        """
        Retourne : NORMAL | ALERT | EXCLUDE | BLACKLIST
        """
        # Nœud déjà blacklisté
        if client_id in self._blacklist:
            return "BLACKLIST"

        # Initialiser les compteurs si premier contact
        if client_id not in self._consecutive_alerts:
            self._consecutive_alerts[client_id] = 0
        if client_id not in self._attack_counts:
            self._attack_counts[client_id] = {}

        if attack_type == "SIGN_FLIP":
            # SIGN_FLIP → attaque intentionnelle prouvée → BLACKLIST immédiat
            self._blacklist.add(client_id)
            print(f"  🚫 Node {client_id} BLACKLISTÉ — SIGN_FLIP")
            return "BLACKLIST"

        elif attack_type == "NORMAL":
            self._consecutive_alerts[client_id] = 0
            return "NORMAL"

        # Incrémenter le compteur par type d'attaque
        counts = self._attack_counts[client_id]
        counts[attack_type] = counts.get(attack_type, 0) + 1
        n = counts[attack_type]

        if attack_type == "FREE_RIDER":
            if n >= 3:
                self._blacklist.add(client_id)
                print(f"  🚫 Node {client_id} BLACKLISTÉ — FREE_RIDER ×{n}")
                return "BLACKLIST"
            return "EXCLUDE"

        elif attack_type == "SCALE":
            if n >= 3:
                self._blacklist.add(client_id)
                print(f"  🚫 Node {client_id} BLACKLISTÉ — SCALE ×{n}")
                return "BLACKLIST"
            return "EXCLUDE"

        elif attack_type == "BYZANTINE":
            if n >= 2:
                self._blacklist.add(client_id)
                print(f"  🚫 Node {client_id} BLACKLISTÉ — BYZANTINE ×{n}")
                return "BLACKLIST"
            return "EXCLUDE"

        elif attack_type == "NOISE":
            self._consecutive_alerts[client_id] += 1
            if self._consecutive_alerts[client_id] >= 3:
                return "EXCLUDE"
            return "ALERT"

        return "NORMAL"