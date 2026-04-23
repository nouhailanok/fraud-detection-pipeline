"""
behavioral_analysis.py — Détection comportementale de nœuds malveillants (FL)
═══════════════════════════════════════════════════════════════════════════════
Détecte les attaques suivantes via Isolation Forest sur 6 features gradient :

  FREE_RIDER   : nœud qui n'entraîne pas (ΔW ≈ 0, loss ≈ 0, ε ≈ 0)
  SIGN_FLIP    : inversion gradients (cos_sim très négatif)
  SCALE        : amplification poids (norm_L2 >> normale)
  NOISE        : bruit gaussien (norm_L2 légèrement élevée)
  BYZANTINE    : nœud défaillant (gradients aléatoires / NaN)
  DATA_POISON  : labels corrompus (détection partielle via comportement)

6 features par nœud :
  - norm_L2    : norme L2 du vecteur ΔW
  - norm_L1    : norme L1 du vecteur ΔW
  - cos_sim    : similarité cosinus vs moyenne des autres nœuds
  - var_dw     : variance du vecteur ΔW
  - train_loss : loss d'entraînement auto-déclarée par le client
  - epsilon    : budget DP consommé auto-déclaré

Paramètres Isolation Forest :
  - n_estimators  = 100   (stable, standard)
  - contamination = 0.1   (conservateur → évite faux positifs)
  - random_state  = 42    (reproductibilité)
  - Activation    à partir du round 3 (rounds 1-2 = calibration)

Action : ALERT uniquement (pas d'exclusion — avec 4 nœuds, exclure = -25% FL)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional


# ─── Import conditionnel scikit-learn ────────────────────────────────────────
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn non installé → BehavioralAnalyzer désactivé")
    print("   pip install scikit-learn")


# ============================================================================
# 🔍 BEHAVIORAL ANALYZER
# ============================================================================

class BehavioralAnalyzer:
    """
    Analyse comportementale des nœuds FL via Isolation Forest.

    Usage dans server.py (FedAvgWithLogging) :
        # Dans __init__ :
        self.analyzer = BehavioralAnalyzer(logs_dir=LOGS_DIR)

        # Dans aggregate_fit, AVANT super() :
        self.analyzer.analyze(global_round, results)

        # Dans aggregate_fit, APRÈS super() :
        if aggregated is not None:
            params, _ = aggregated
            self.analyzer.set_prev_params(params)
    """

    # Seuils de détection heuristique (fallback si IF pas encore calibré)
    HEURISTIC_THRESHOLDS = {
        "cos_sim_flip"   : -0.5,    # en dessous → probable SIGN_FLIP
        "norm_ratio_high": 10.0,    # au dessus de 10× la médiane → probable SCALE
        "norm_l2_zero"   : 1e-6,    # en dessous → probable FREE_RIDER
        "loss_zero"      : 1e-4,    # en dessous → probable FREE_RIDER
    }

    def __init__(
        self,
        logs_dir        : Path,
        min_rounds_if   : int   = 3,       # rounds de calibration avant IF
        contamination   : float = 0.1,     # fraction attendue de nœuds suspects
        n_estimators    : int   = 100,
        random_state    : int   = 42,
    ):
        self.logs_dir      = Path(logs_dir)
        self.min_rounds_if = min_rounds_if
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state

        # Historique de toutes les features (accumulation multi-rounds)
        self._history      : List[Dict] = []   # liste de records {round, client_id, features...}

        # Modèle Isolation Forest (entraîné à chaque round dès min_rounds_if atteint)
        self._if_model     = None

        # Poids globaux du round précédent (pour calculer ΔW)
        self._prev_params  : Optional[List[np.ndarray]] = None

        # Log path
        self._log_path = self.logs_dir / "behavioral_analysis.json"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔍 BehavioralAnalyzer initialisé")
        print(f"   IF activation : round {self.min_rounds_if}+")
        print(f"   Contamination : {self.contamination}")
        print(f"   Log           : {self._log_path}")

    # ─── Interface publique ───────────────────────────────────────────────────

    def set_prev_params(self, parameters) -> None:
        """
        Mémorise les poids globaux agrégés du round courant.
        À appeler APRÈS super().aggregate_fit() → ces poids seront le
        point de référence pour calculer ΔW au prochain round.
        """
        if parameters is None:
            return
        try:
            import flwr.common
            self._prev_params = flwr.common.parameters_to_ndarrays(parameters)
        except Exception as e:
            print(f"  ⚠️  BehavioralAnalyzer.set_prev_params() erreur : {e}")

    def analyze(self, round_num: int, results) -> Dict:
        """
        Analyse les résultats de fit d'un round.
        À appeler AVANT super().aggregate_fit() dans aggregate_fit().

        Paramètres
        ----------
        round_num : numéro du round global (avec offset)
        results   : liste de (ClientProxy, FitRes) retournée par Flower

        Retourne
        --------
        dict contenant les alertes et features par nœud
        """
        if not SKLEARN_AVAILABLE:
            return {}

        if not results:
            return {}

        # ── 1. Extraction des features ────────────────────────────────────────
        round_records = []
        client_params_list = []

        for client_proxy, fit_res in results:
            try:
                record = self._extract_features(
                    client_proxy, fit_res, round_num
                )
                round_records.append(record)
                # Stocker les params bruts pour calcul cos_sim croisé
                import flwr.common
                params = flwr.common.parameters_to_ndarrays(fit_res.parameters)
                client_params_list.append(params)
            except Exception as e:
                print(f"  ⚠️  BehavioralAnalyzer extract erreur client : {e}")
                continue

        if not round_records:
            return {}

        # ── 2. Calcul cos_sim croisé (chaque nœud vs moyenne des autres) ─────
        self._compute_cross_cos_sim(round_records, client_params_list)

        # ── 3. Accumulation dans l'historique global ──────────────────────────
        self._history.extend(round_records)

        # ── 4. Détection (heuristique ou Isolation Forest) ───────────────────
        alerts = self._detect(round_records, round_num)

        # ── 5. Affichage console ──────────────────────────────────────────────
        self._print_report(round_num, round_records, alerts)

        # ── 6. Sauvegarde JSON ────────────────────────────────────────────────
        self._save_log(round_num, round_records, alerts)

        return {
            "records": round_records,
            "alerts" : alerts,
        }

    def get_summary(self) -> Dict:
        """Retourne un résumé des alertes cumulées sur tous les rounds."""
        if not self._history:
            return {"total_rounds": 0, "total_alerts": 0, "suspects": {}}

        alerts_by_client = {}
        for record in self._history:
            cid = record.get("client_id", "?")
            if record.get("alert"):
                alerts_by_client[cid] = alerts_by_client.get(cid, 0) + 1

        return {
            "total_rounds"   : len(set(r["round"] for r in self._history)),
            "total_alerts"   : sum(alerts_by_client.values()),
            "suspects"       : alerts_by_client,
        }

    # ─── Extraction features ──────────────────────────────────────────────────

    def _extract_features(self, client_proxy, fit_res, round_num: int) -> Dict:
        """Calcule les 6 features pour un nœud donné."""
        import flwr.common

        # Identifiant client (depuis metrics si dispo, sinon hash du proxy)
        metrics    = fit_res.metrics or {}
        client_id  = str(metrics.get("client_id", id(client_proxy)))
        train_loss = float(metrics.get("train_loss", 0.0))
        epsilon    = float(metrics.get("epsilon",    0.0))

        # Poids envoyés par le client (après entraînement local)
        client_params = flwr.common.parameters_to_ndarrays(fit_res.parameters)

        # ΔW = poids_client - poids_global_précédent
        if self._prev_params is not None and len(self._prev_params) == len(client_params):
            delta_w = np.concatenate([
                (cp - pp).flatten()
                for cp, pp in zip(client_params, self._prev_params)
            ])
        else:
            # Premier round : pas de poids précédents → ΔW = poids bruts
            delta_w = np.concatenate([p.flatten() for p in client_params])

        # Features
        norm_l2 = float(np.linalg.norm(delta_w, ord=2))
        norm_l1 = float(np.linalg.norm(delta_w, ord=1))
        var_dw  = float(np.var(delta_w))

        # cos_sim sera rempli après (calcul croisé)
        return {
            "round"      : round_num,
            "client_id"  : client_id,
            "norm_l2"    : norm_l2,
            "norm_l1"    : norm_l1,
            "cos_sim"    : 0.0,   # placeholder
            "var_dw"     : var_dw,
            "train_loss" : train_loss,
            "epsilon"    : epsilon,
            "alert"      : False,
            "alert_type" : None,
            "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Stockage temporaire pour cos_sim
            "_delta_w"   : delta_w,
        }

    def _compute_cross_cos_sim(self, records: List[Dict], params_list: List) -> None:
        """
        Calcule la similarité cosinus de chaque nœud vs la moyenne des autres.
        Modifie les records en place.
        """
        if len(records) < 2:
            for r in records:
                r["cos_sim"] = 1.0
            return

        delta_ws = [r["_delta_w"] for r in records]
        mean_all = np.mean(delta_ws, axis=0)

        for i, record in enumerate(records):
            # Moyenne des autres nœuds
            others = [delta_ws[j] for j in range(len(delta_ws)) if j != i]
            mean_others = np.mean(others, axis=0)

            dw   = record["_delta_w"]
            norm_dw     = np.linalg.norm(dw)
            norm_others = np.linalg.norm(mean_others)

            if norm_dw < 1e-10 or norm_others < 1e-10:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(dw, mean_others) / (norm_dw * norm_others))

            record["cos_sim"] = round(cos_sim, 4)

        # Nettoyage du champ temporaire
        for r in records:
            r.pop("_delta_w", None)

    # ─── Détection ────────────────────────────────────────────────────────────

    def _detect(self, round_records: List[Dict], round_num: int) -> List[Dict]:
        """
        Détecte les nœuds suspects.
        - Rounds < min_rounds_if : heuristiques simples
        - Rounds >= min_rounds_if : Isolation Forest sur historique complet
        """
        alerts = []

        if round_num >= self.min_rounds_if and len(self._history) >= 4:
            alerts = self._detect_isolation_forest(round_records)
        else:
            alerts = self._detect_heuristic(round_records)

        # Marquer les records
        suspect_ids = {a["client_id"] for a in alerts}
        for record in round_records:
            if record["client_id"] in suspect_ids:
                record["alert"] = True
                alert = next(a for a in alerts if a["client_id"] == record["client_id"])
                record["alert_type"] = alert.get("type", "UNKNOWN")

        return alerts

    def _detect_heuristic(self, records: List[Dict]) -> List[Dict]:
        """Détection heuristique simple (calibration rounds 1-2)."""
        alerts = []
        th = self.HEURISTIC_THRESHOLDS

        for r in records:
            alert_type = None

            if r["norm_l2"] < th["norm_l2_zero"] and r["train_loss"] < th["loss_zero"]:
                alert_type = "FREE_RIDER"
            elif r["cos_sim"] < th["cos_sim_flip"]:
                alert_type = "SIGN_FLIP"
            elif r["norm_l2"] > 0:
                # Comparer vs médiane des autres nœuds
                other_norms = [x["norm_l2"] for x in records if x["client_id"] != r["client_id"]]
                if other_norms:
                    median_norm = float(np.median(other_norms))
                    if median_norm > 1e-6 and r["norm_l2"] / median_norm > th["norm_ratio_high"]:
                        alert_type = "SCALE"

            if alert_type:
                alerts.append({
                    "client_id": r["client_id"],
                    "round"    : r["round"],
                    "type"     : alert_type,
                    "method"   : "HEURISTIC",
                    "score"    : None,
                })

        return alerts

    def _detect_isolation_forest(self, round_records: List[Dict]) -> List[Dict]:
        """
        Détection via Isolation Forest entraîné sur tout l'historique.
        """
        alerts = []

        # Construction de la matrice de features depuis l'historique complet
        feature_keys = ["norm_l2", "norm_l1", "cos_sim", "var_dw", "train_loss", "epsilon"]
        X_history = np.array([
            [r.get(k, 0.0) for k in feature_keys]
            for r in self._history
            if not any(np.isnan(r.get(k, 0.0)) for k in feature_keys)
        ])

        if len(X_history) < 4:
            return self._detect_heuristic(round_records)

        # Entraînement IF sur tout l'historique
        try:
            self._if_model = IsolationForest(
                n_estimators = self.n_estimators,
                contamination= self.contamination,
                random_state = self.random_state,
            )
            self._if_model.fit(X_history)
        except Exception as e:
            print(f"  ⚠️  Isolation Forest fit erreur : {e}")
            return self._detect_heuristic(round_records)

        # Scoring des nœuds du round courant
        X_current = np.array([
            [r.get(k, 0.0) for k in feature_keys]
            for r in round_records
        ])

        try:
            scores     = self._if_model.score_samples(X_current)   # plus négatif = plus suspect
            predictions= self._if_model.predict(X_current)         # -1 = anomalie, 1 = normal
        except Exception as e:
            print(f"  ⚠️  Isolation Forest predict erreur : {e}")
            return self._detect_heuristic(round_records)

        for i, (record, score, pred) in enumerate(zip(round_records, scores, predictions)):
            if pred == -1:
                # Détermination du type d'attaque probable
                alert_type = self._classify_attack(record)
                alerts.append({
                    "client_id": record["client_id"],
                    "round"    : record["round"],
                    "type"     : alert_type,
                    "method"   : "ISOLATION_FOREST",
                    "score"    : round(float(score), 4),
                })

        return alerts

    def _classify_attack(self, record: Dict) -> str:
        """Heuristique post-IF pour typer l'attaque probable."""
        th = self.HEURISTIC_THRESHOLDS

        if record["norm_l2"] < th["norm_l2_zero"] and record["train_loss"] < th["loss_zero"]:
            return "FREE_RIDER"
        if record["cos_sim"] < th["cos_sim_flip"]:
            return "SIGN_FLIP"
        if record["norm_l2"] > 0 and record["epsilon"] < 1e-6:
            return "BYZANTINE"
        if record["norm_l2"] > 0:
            return "SCALE_or_NOISE"
        return "UNKNOWN"

    # ─── Affichage et logs ────────────────────────────────────────────────────

    def _print_report(self, round_num: int, records: List[Dict], alerts: List[Dict]) -> None:
        """Affiche le rapport comportemental dans la console serveur."""
        alert_ids = {a["client_id"] for a in alerts}
        method    = "IF" if round_num >= self.min_rounds_if else "HEURISTIC"

        print(f"\n{'─'*58}")
        print(f"  🔍  Analyse comportementale — Round {round_num}  [{method}]")
        print(f"{'─'*58}")
        print(f"  {'Client':>8}  {'norm_L2':>9}  {'cos_sim':>8}  {'ε':>7}  {'loss':>8}  {'Statut'}")
        print(f"  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*10}")

        for r in records:
            status = "⚠️  SUSPECT" if r["client_id"] in alert_ids else "✅ OK"
            if r["client_id"] in alert_ids:
                alert = next(a for a in alerts if a["client_id"] == r["client_id"])
                status += f" [{alert['type']}]"
            print(
                f"  {r['client_id']:>8}  "
                f"{r['norm_l2']:>9.4f}  "
                f"{r['cos_sim']:>8.4f}  "
                f"{r['epsilon']:>7.4f}  "
                f"{r['train_loss']:>8.4f}  "
                f"{status}"
            )

        if alerts:
            print(f"\n  🚨 {len(alerts)} nœud(s) suspect(s) détecté(s) — ALERT ONLY (pas d'exclusion)")
        else:
            print(f"\n  ✅ Aucune anomalie détectée")
        print(f"{'─'*58}")

    def _save_log(self, round_num: int, records: List[Dict], alerts: List[Dict]) -> None:
        """Sauvegarde le log JSON complet."""
        # Chargement de l'existant
        existing = []
        if self._log_path.exists():
            try:
                existing = json.loads(self._log_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []

        # Nouvelle entrée
        entry = {
            "round"    : round_num,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_clients": len(records),
            "n_alerts" : len(alerts),
            "nodes"    : [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in records
            ],
            "alerts"   : alerts,
            "summary"  : self.get_summary(),
        }
        existing.append(entry)

        try:
            self._log_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"  ⚠️  BehavioralAnalyzer log erreur : {e}")


# ============================================================================
# 🧪 ZONE DE TEST
# ============================================================================

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("  Test BehavioralAnalyzer — 4 rounds simulés")
    print("=" * 60)

    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn requis : pip install scikit-learn")
        exit(1)

    analyzer = BehavioralAnalyzer(logs_dir=Path("logs/fl"), min_rounds_if=3)

    # Simulation de 4 rounds avec nœud 3 = SIGN_FLIP round 3, nœud 4 = FREE_RIDER round 4
    for rnd in range(1, 5):
        print(f"\n{'═'*60}")
        print(f"  Simulation round {rnd}")
        print(f"{'═'*60}")

        # Création de faux résultats
        class FakeFitRes:
            def __init__(self, params, metrics):
                import flwr.common
                self.parameters = flwr.common.ndarrays_to_parameters(params)
                self.metrics    = metrics
                self.num_examples = 1000

        class FakeProxy:
            pass

        results = []
        base_params = [np.random.randn(128, 26).astype(np.float32),
                       np.random.randn(128).astype(np.float32)]

        for node_id in range(1, 5):
            params = [p + np.random.randn(*p.shape).astype(np.float32) * 0.01
                      for p in base_params]

            metrics = {
                "client_id" : str(node_id),
                "train_loss": random.uniform(1.5, 3.0),
                "epsilon"   : random.uniform(0.1, 0.4),
            }

            # Attaques simulées
            if rnd == 3 and node_id == 3:
                # SIGN_FLIP : inverser les gradients
                params = [-p for p in params]
                metrics["train_loss"] = 2.5

            if rnd == 4 and node_id == 4:
                # FREE_RIDER : gradients quasi-nuls
                params = [np.zeros_like(p) for p in params]
                metrics["train_loss"] = 0.0
                metrics["epsilon"]    = 0.0

            results.append((FakeProxy(), FakeFitRes(params, metrics)))

        # Analyse
        result = analyzer.analyze(rnd, results)

        # Mise à jour poids précédents
        import flwr.common
        mean_params = [
            np.mean([flwr.common.parameters_to_ndarrays(res.parameters)[i]
                     for _, res in results], axis=0)
            for i in range(len(base_params))
        ]
        analyzer.set_prev_params(flwr.common.ndarrays_to_parameters(mean_params))

    print(f"\n{'═'*60}")
    print("  Résumé final :")
    print(json.dumps(analyzer.get_summary(), indent=2))
    print(f"{'═'*60}")
    print("\n✅ Test terminé — voir logs/fl/behavioral_analysis.json")


















# """
# behavioral_analysis.py — Analyse comportementale des nœuds FL
# ══════════════════════════════════════════════════════════════
# Détecte les nœuds malveillants ou défaillants en analysant leurs
# mises à jour de poids (∆W) à chaque round, via Isolation Forest.

# Cas détectés :
#   FREE_RIDER    → nœud qui ne s'entraîne pas (∆W ≈ 0, loss=0, ε≈0)
#   SIGN_FLIP     → nœud qui inverse ses gradients (cos_sim très négatif)
#   SCALE         → nœud qui amplifie ses poids (norm_L2 >> normale)
#   NOISE         → nœud qui ajoute du bruit (norm_L2 légèrement élevée)
#   BYZANTINE     → nœud défaillant (gradients aléatoires/NaN)

# Features analysées par nœud (6 dimensions) :
#   1. norm_l2    : ∥∆W∥₂  — taille globale de la mise à jour
#   2. norm_l1    : ∥∆W∥₁  — somme des valeurs absolues
#   3. cos_sim    : similarité cosinus avec la moyenne des autres nœuds
#   4. variance   : dispersion des valeurs de ∆W
#   5. train_loss : loss déclarée par le nœud (auto-déclaré)
#   6. epsilon    : budget DP consommé (auto-déclaré)

# Stratégie d'activation :
#   - Rounds 1-2  : calibration — features loggées, aucune décision
#   - Round 3+    : Isolation Forest activé sur l'historique complet

# Action sur nœud suspect :
#   - Mode ALERT  : log uniquement, FedAvg inchangé

# Fichiers produits :
#   logs/fl/behavioral_analysis.json  → historique complet round par round
# """

# import os
# import json
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple

# import numpy as np

# try:
#     from sklearn.ensemble import IsolationForest
#     SKLEARN_AVAILABLE = True
# except ImportError:
#     SKLEARN_AVAILABLE = False
#     print("⚠️  sklearn non disponible — Isolation Forest désactivé")
#     print("   pip install scikit-learn")


# # ============================================================================
# # ⚙️  CONFIGURATION
# # ============================================================================

# # Round à partir duquel l'IF prend des décisions (avant : calibration)
# MIN_ROUNDS_FOR_IF = int(os.getenv("BA_MIN_ROUNDS",      "3"))

# # Proportion maximale de nœuds supposés malveillants
# # 0.1 = au max 1 nœud sur 10 malveillant → prudent, peu de faux positifs
# IF_CONTAMINATION  = float(os.getenv("BA_CONTAMINATION", "0.1"))

# # Nombre d'arbres dans l'Isolation Forest
# IF_N_ESTIMATORS   = int(os.getenv("BA_N_ESTIMATORS",    "100"))

# # Seuil score IF en dessous duquel un nœud est SUSPECT
# # Scores IF : proche de 0 = normal, proche de -1 = très anormal
# SUSPECT_THRESHOLD = float(os.getenv("BA_SUSPECT_THRESHOLD", "-0.1"))

# # Dossier de logs
# LOGS_DIR          = Path(os.getenv("FL_LOGS_DIR", "logs/fl"))


# # ============================================================================
# # 🧠 BEHAVIORAL ANALYZER
# # ============================================================================

# class BehavioralAnalyzer:
#     """
#     Analyse le comportement de chaque nœud FL à chaque round.

#     Utilisation dans server.py :

#         # Dans FedAvgWithLogging.__init__
#         self.analyzer = BehavioralAnalyzer()

#         # Dans aggregate_fit, AVANT super().aggregate_fit()
#         report = self.analyzer.analyze(
#             server_round = server_round,
#             results      = results,
#             prev_params  = self.prev_params,
#         )

#         # Après super().aggregate_fit(), mettre à jour prev_params
#         params, _ = aggregated
#         ndarrays = flwr.common.parameters_to_ndarrays(params)
#         self.analyzer.set_prev_params(ndarrays)
#     """

#     def __init__(self):
#         self.prev_params     : Optional[List[np.ndarray]] = None
#         self.feature_history : List[np.ndarray]           = []
#         self.node_ids_history: List[str]                  = []

#         self.log_path = LOGS_DIR / "behavioral_analysis.json"
#         LOGS_DIR.mkdir(parents=True, exist_ok=True)

#         with open(self.log_path, "w", encoding="utf-8") as f:
#             json.dump([], f)

#         print(f"\n🛡️  BehavioralAnalyzer initialisé")
#         print(f"   Activation IF : round ≥ {MIN_ROUNDS_FOR_IF}")
#         print(f"   Contamination : {IF_CONTAMINATION}")
#         print(f"   Seuil suspect : score < {SUSPECT_THRESHOLD}")
#         print(f"   Logs          : {self.log_path}\n")

#     # ── API principale ────────────────────────────────────────────────────────

#     def analyze(
#         self,
#         server_round : int,
#         results      : list,
#         prev_params  : Optional[List[np.ndarray]] = None,
#     ) -> Dict:
#         """
#         Point d'entrée principal — appelé depuis aggregate_fit().

#         Args:
#             server_round : numéro du round courant
#             results      : [(ClientProxy, FitRes), ...] depuis Flower
#             prev_params  : poids globaux du round précédent

#         Returns:
#             rapport dict avec statut par nœud
#         """
#         print(f"\n🛡️  Behavioral Analysis — Round {server_round}")

#         # Utilise prev_params passé en argument ou celui stocké en mémoire
#         params_ref = prev_params if prev_params is not None else self.prev_params

#         # ── 1. Extraction des données brutes ──────────────────────────────────
#         nodes_data = self._extract_nodes_data(results, params_ref)

#         if not nodes_data:
#             print("  ⚠️  Aucune donnée exploitable — analyse ignorée")
#             return {"round": server_round, "status": "NO_DATA", "nodes": {}}

#         # ── 2. Calcul des 6 features par nœud ────────────────────────────────
#         nodes_features = self._compute_features(nodes_data)

#         # ── 3. Mise à jour de l'historique ────────────────────────────────────
#         self._update_history(server_round, nodes_features)

#         # ── 4. Isolation Forest ───────────────────────────────────────────────
#         if server_round < MIN_ROUNDS_FOR_IF:
#             print(f"  ⏳ Calibration (IF actif à partir du round {MIN_ROUNDS_FOR_IF})")
#             if_scores   = {nid: 0.0           for nid in nodes_features}
#             if_statuses = {nid: "CALIBRATION" for nid in nodes_features}
#         else:
#             if_scores, if_statuses = self._run_isolation_forest(nodes_features)

#         # ── 5. Construction + sauvegarde + affichage du rapport ───────────────
#         report = self._build_report(
#             server_round, nodes_features, if_scores, if_statuses
#         )
#         self._save_report(report)
#         self._print_report(report)

#         return report

#     def set_prev_params(self, params: List[np.ndarray]) -> None:
#         """
#         Mémorise les poids globaux après chaque round.
#         À appeler depuis aggregate_fit() après super().aggregate_fit().
#         """
#         self.prev_params = [p.copy() for p in params]

#     # ── Extraction des données brutes ─────────────────────────────────────────

#     def _extract_nodes_data(
#         self,
#         results    : list,
#         prev_params: Optional[List[np.ndarray]],
#     ) -> Dict[str, Dict]:
#         """
#         Extrait depuis les résultats Flower :
#           - poids mis à jour de chaque nœud
#           - métriques déclarées (epsilon, train_loss, client_id)
#           - ∆W = poids_après − poids_avant
#         """
#         import flwr.common

#         nodes_data = {}

#         for i, (proxy, fit_res) in enumerate(results):
#             if fit_res is None:
#                 continue

#             metrics   = fit_res.metrics or {}
#             client_id = str(metrics.get("client_id", f"node_{i+1}"))

#             # Récupération des poids mis à jour
#             try:
#                 updated_params = flwr.common.parameters_to_ndarrays(
#                     fit_res.parameters
#                 )
#             except Exception as e:
#                 print(f"  ⚠️  Impossible de lire les poids de {client_id} : {e}")
#                 continue

#             # Calcul de ∆W
#             if prev_params is not None and len(prev_params) == len(updated_params):
#                 delta_w = [
#                     u.astype(np.float32) - p.astype(np.float32)
#                     for u, p in zip(updated_params, prev_params)
#                 ]
#             else:
#                 # Premier round : ∆W = poids eux-mêmes
#                 delta_w = [p.astype(np.float32) for p in updated_params]

#             nodes_data[client_id] = {
#                 "delta_w"    : delta_w,
#                 "train_loss" : float(metrics.get("train_loss", 0.0)),
#                 "epsilon"    : float(metrics.get("epsilon",    0.0)),
#                 "n_examples" : fit_res.num_examples,
#             }

#         return nodes_data

#     # ── Calcul des 6 features ─────────────────────────────────────────────────

#     def _compute_features(self, nodes_data: Dict[str, Dict]) -> Dict[str, Dict]:
#         """
#         Calcule les 6 features pour chaque nœud.

#         Chaque ∆W est d'abord aplati en un vecteur 1D pour les calculs.
#         """
#         # Aplatissement des ∆W
#         flat_deltas = {
#             node_id: np.concatenate([dw.flatten() for dw in data["delta_w"]])
#             for node_id, data in nodes_data.items()
#         }
#         node_ids = list(nodes_data.keys())
#         features = {}

#         for node_id in node_ids:
#             dw   = flat_deltas[node_id]
#             data = nodes_data[node_id]

#             # Feature 1 — Norme L2
#             norm_l2 = float(np.linalg.norm(dw, ord=2))

#             # Feature 2 — Norme L1
#             norm_l1 = float(np.linalg.norm(dw, ord=1))

#             # Feature 3 — Similarité cosinus avec la moyenne des autres nœuds
#             others = [
#                 flat_deltas[oid] for oid in node_ids if oid != node_id
#             ]
#             if others:
#                 mean_others = np.mean(others, axis=0)
#                 cos_sim     = self._cosine_similarity(dw, mean_others)
#             else:
#                 cos_sim = 1.0

#             # Feature 4 — Variance de ∆W
#             variance = float(np.var(dw))

#             # Feature 5 — Train loss (auto-déclaré)
#             train_loss = data["train_loss"]

#             # Feature 6 — Epsilon DP (auto-déclaré)
#             epsilon = data["epsilon"]

#             features[node_id] = {
#                 "norm_l2"    : round(norm_l2,    6),
#                 "norm_l1"    : round(norm_l1,    6),
#                 "cos_sim"    : round(cos_sim,    6),
#                 "variance"   : round(variance,   6),
#                 "train_loss" : round(train_loss, 6),
#                 "epsilon"    : round(epsilon,    6),
#             }

#         return features

#     @staticmethod
#     def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#         """Similarité cosinus robuste aux normes nulles."""
#         norm_a = np.linalg.norm(a)
#         norm_b = np.linalg.norm(b)
#         if norm_a < 1e-10 or norm_b < 1e-10:
#             return 0.0
#         return float(np.dot(a, b) / (norm_a * norm_b))

#     # ── Historique multi-rounds ───────────────────────────────────────────────

#     def _update_history(
#         self, server_round: int, nodes_features: Dict[str, Dict]
#     ) -> None:
#         """
#         Accumule les features de chaque nœud à travers les rounds.
#         L'IF s'entraîne sur cet historique complet → meilleure détection
#         au fil des rounds.
#         """
#         for node_id, feats in nodes_features.items():
#             vector = np.array([
#                 feats["norm_l2"],
#                 feats["norm_l1"],
#                 feats["cos_sim"],
#                 feats["variance"],
#                 feats["train_loss"],
#                 feats["epsilon"],
#             ], dtype=np.float32)

#             self.feature_history.append(vector)
#             self.node_ids_history.append(f"r{server_round}_{node_id}")

#     # ── Isolation Forest ──────────────────────────────────────────────────────

#     def _run_isolation_forest(
#         self,
#         nodes_features: Dict[str, Dict],
#     ) -> Tuple[Dict[str, float], Dict[str, str]]:
#         """
#         Entraîne IF sur l'historique complet et calcule le score
#         d'anomalie pour chaque nœud du round courant.

#         score IF proche de  0  → NORMAL  (point dans la masse)
#         score IF proche de -1  → SUSPECT (point facilement isolable)
#         """
#         if not SKLEARN_AVAILABLE:
#             return (
#                 {nid: 0.0                  for nid in nodes_features},
#                 {nid: "SKLEARN_UNAVAILABLE" for nid in nodes_features},
#             )

#         if len(self.feature_history) < 4:
#             return (
#                 {nid: 0.0               for nid in nodes_features},
#                 {nid: "INSUFFICIENT_DATA" for nid in nodes_features},
#             )

#         # Entraînement sur tout l'historique
#         X = np.array(self.feature_history, dtype=np.float32)
#         X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

#         clf = IsolationForest(
#             n_estimators  = IF_N_ESTIMATORS,
#             contamination = IF_CONTAMINATION,
#             random_state  = 42,
#         )
#         clf.fit(X)

#         # Score pour chaque nœud du round courant
#         scores   = {}
#         statuses = {}

#         for node_id, feats in nodes_features.items():
#             vec = np.array([[
#                 feats["norm_l2"],
#                 feats["norm_l1"],
#                 feats["cos_sim"],
#                 feats["variance"],
#                 feats["train_loss"],
#                 feats["epsilon"],
#             ]], dtype=np.float32)
#             vec = np.nan_to_num(vec, nan=0.0)

#             score = float(clf.score_samples(vec)[0])
#             scores[node_id] = round(score, 4)

#             if score < SUSPECT_THRESHOLD:
#                 statuses[node_id] = self._classify_attack(feats)
#             else:
#                 statuses[node_id] = "NORMAL"

#         return scores, statuses

#     def _classify_attack(self, feats: Dict) -> str:
#         """
#         Heuristique pour identifier le TYPE d'attaque.
#         Appelé seulement quand l'IF a déjà marqué le nœud SUSPECT.

#         Signatures connues :
#           FREE_RIDER : norm_l2 ≈ 0, train_loss ≈ 0, epsilon ≈ 0
#           SIGN_FLIP  : cos_sim très négatif
#           SCALE      : norm_l2 très élevée
#           BYZANTINE  : variance explosive
#         """
#         if feats["norm_l2"] < 0.05 and feats["train_loss"] < 0.01 and feats["epsilon"] < 0.01:
#             return "SUSPECT ⚠️ [FREE_RIDER]"
#         if feats["cos_sim"] < -0.3:
#             return "SUSPECT ⚠️ [SIGN_FLIP]"
#         if feats["norm_l2"] > 20.0:
#             return "SUSPECT ⚠️ [SCALE]"
#         if feats["variance"] > 1.0:
#             return "SUSPECT ⚠️ [BYZANTINE]"
#         return "SUSPECT ⚠️ [UNKNOWN]"

#     # ── Rapport ───────────────────────────────────────────────────────────────

#     def _build_report(
#         self,
#         server_round   : int,
#         nodes_features : Dict[str, Dict],
#         if_scores      : Dict[str, float],
#         if_statuses    : Dict[str, str],
#     ) -> Dict:
#         """Construit le dict de rapport complet pour ce round."""
#         nodes_report = {}
#         n_suspects   = 0

#         for node_id, feats in nodes_features.items():
#             status = if_statuses.get(node_id, "UNKNOWN")
#             score  = if_scores.get(node_id, 0.0)

#             if "SUSPECT" in status:
#                 n_suspects += 1

#             nodes_report[node_id] = {
#                 "norm_l2"    : feats["norm_l2"],
#                 "norm_l1"    : feats["norm_l1"],
#                 "cos_sim"    : feats["cos_sim"],
#                 "variance"   : feats["variance"],
#                 "train_loss" : feats["train_loss"],
#                 "epsilon"    : feats["epsilon"],
#                 "if_score"   : score,
#                 "status"     : status,
#             }

#         return {
#             "round"      : server_round,
#             "n_nodes"    : len(nodes_features),
#             "n_suspects" : n_suspects,
#             "if_active"  : server_round >= MIN_ROUNDS_FOR_IF,
#             "nodes"      : nodes_report,
#         }

#     def _save_report(self, report: Dict) -> None:
#         """Ajoute le rapport du round courant dans le JSON."""
#         try:
#             if self.log_path.exists():
#                 with open(self.log_path, "r", encoding="utf-8") as f:
#                     history = json.load(f)
#             else:
#                 history = []

#             history.append(report)

#             with open(self.log_path, "w", encoding="utf-8") as f:
#                 json.dump(history, f, indent=2, ensure_ascii=False)

#         except Exception as e:
#             print(f"  ⚠️  Erreur sauvegarde behavioral_analysis.json : {e}")

#     def _print_report(self, report: Dict) -> None:
#         """Affiche le rapport dans la console."""
#         print(f"  {'─'*54}")

#         if not report["if_active"]:
#             print(f"  ⏳ Calibration — décisions actives au round {MIN_ROUNDS_FOR_IF}")

#         for node_id, data in report["nodes"].items():
#             status     = data["status"]
#             score      = data["if_score"]
#             is_suspect = "SUSPECT" in status
#             icon       = "⚠️ " if is_suspect else "✅"

#             print(
#                 f"  {icon} {node_id:<10} "
#                 f"score={score:+.3f}  "
#                 f"L2={data['norm_l2']:.3f}  "
#                 f"cos={data['cos_sim']:+.3f}  "
#                 f"loss={data['train_loss']:.3f}  "
#                 f"ε={data['epsilon']:.3f}"
#             )
#             if is_suspect:
#                 print(f"     └─ {status}")

#         n_suspects = report["n_suspects"]
#         if n_suspects > 0:
#             print(f"\n  🚨 {n_suspects} nœud(s) SUSPECT(s) ce round !")
#         else:
#             print(f"\n  ✅ Tous les nœuds semblent normaux")

#         print(f"  {'─'*54}")

#     def get_summary(self) -> Dict:
#         """Retourne un résumé de toute la session. Appelé en fin de FL."""
#         try:
#             with open(self.log_path, "r", encoding="utf-8") as f:
#                 history = json.load(f)
#         except Exception:
#             return {}

#         total_suspects       = sum(r.get("n_suspects", 0) for r in history)
#         rounds_with_suspects = [r["round"] for r in history if r.get("n_suspects", 0) > 0]

#         return {
#             "total_rounds"         : len(history),
#             "total_suspect_events" : total_suspects,
#             "rounds_with_suspects" : rounds_with_suspects,
#             "log_path"             : str(self.log_path),
#         }


# # ============================================================================
# # 🧪 ZONE DE TEST  (python behavioral_analysis.py)
# # ============================================================================

# if __name__ == "__main__":
#     import flwr.common

#     print("=" * 60)
#     print("  Test BehavioralAnalyzer — 4 rounds simulés")
#     print("=" * 60)

#     analyzer = BehavioralAnalyzer()

#     N = 167_297  # taille réelle des paramètres DPGRU

#     def fake_fitres(delta, loss, eps, node_id, n=50_000):
#         params = flwr.common.ndarrays_to_parameters([delta])
#         class R:
#             parameters  = params
#             num_examples = n
#             metrics     = {"train_loss": loss, "epsilon": eps, "client_id": node_id}
#         return R()

#     class P: pass

#     prev = [np.zeros(N, dtype=np.float32)]
#     base = np.random.normal(0, 0.5, N).astype(np.float32)

#     # Round 1 & 2 : calibration, tous honnêtes
#     for rnd in [1, 2]:
#         results = [
#             (P(), fake_fitres(base * np.random.normal(1, 0.1, N).astype(np.float32),
#                               3.5 - rnd * 0.3, 0.15 + rnd * 0.05, f"node_{i}"))
#             for i in range(1, 5)
#         ]
#         analyzer.analyze(rnd, results, prev)

#     # Round 3 : node_4 = SIGN_FLIP (IF activé)
#     print("\n--- Round 3 : node_4 attaque SIGN_FLIP ---")
#     results_r3 = [
#         (P(), fake_fitres(base * np.random.normal(1, 0.1, N).astype(np.float32), 2.5, 0.27, "node_1")),
#         (P(), fake_fitres(base * np.random.normal(1, 0.1, N).astype(np.float32), 2.6, 0.26, "node_2")),
#         (P(), fake_fitres(base * np.random.normal(1, 0.1, N).astype(np.float32), 2.4, 0.28, "node_3")),
#         (P(), fake_fitres(-base,                                                  2.5, 0.27, "node_4")),  # ← SIGN_FLIP
#     ]
#     analyzer.analyze(3, results_r3, prev)

#     # Round 4 : node_2 = FREE_RIDER
#     print("\n--- Round 4 : node_2 = FREE_RIDER ---")
#     results_r4 = [
#         (P(), fake_fitres(base * np.random.normal(1, 0.1, N).astype(np.float32), 2.2, 0.32, "node_1")),
#         (P(), fake_fitres(np.zeros(N, dtype=np.float32), 0.0, 0.001, "node_2")),  # ← FREE_RIDER
#         (P(), fake_fitres(base * np.random.normal(1, 0.1, N).astype(np.float32), 2.1, 0.33, "node_3")),
#         (P(), fake_fitres(base * np.random.normal(1, 0.1, N).astype(np.float32), 2.2, 0.32, "node_4")),
#     ]
#     analyzer.analyze(4, results_r4, prev)

#     print("\n" + "=" * 60)
#     summary = analyzer.get_summary()
#     for k, v in summary.items():
#         print(f"  {k:<30} : {v}")
#     print(f"\n✅ Test terminé")