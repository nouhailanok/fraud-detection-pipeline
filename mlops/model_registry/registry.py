"""
registry.py — Model Registry pour le projet FL Fraud Detection
══════════════════════════════════════════════════════════════
Centralise tous les runs FL, calcule les scores et sélectionne
le meilleur modèle à promouvoir en production.

Usage :
  python mlops/model_registry/registry.py --scan logs/runs/
  python mlops/model_registry/registry.py --list
  python mlops/model_registry/registry.py --propose
  python mlops/model_registry/registry.py --promote V4_noise1.00_batch128
  python mlops/model_registry/registry.py --status
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


# ── Chemins par défaut ────────────────────────────────────────────────────────
REGISTRY_DIR  = Path(__file__).parent
REGISTRY_JSON = REGISTRY_DIR / "registry.json"
BEST_MODEL_DIR= REGISTRY_DIR / "best_model"


# ============================================================================
# 📦 MODEL REGISTRY
# ============================================================================

class ModelRegistry:

    def __init__(self, registry_path: Path = REGISTRY_JSON):
        self.registry_path = Path(registry_path)
        self._data = self._load()

    # ── Lecture / écriture ────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path, encoding="utf-8") as f:
                return json.load(f)
        return {"production": None, "last_updated": "", "models": []}

    def _save(self) -> None:
        self._data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    # ── Calcul du trust score ─────────────────────────────────────────────────

    # Poids de pénalité par type d'attaque — alignés avec behavioral_analysis.py
    ATTACK_WEIGHTS = {
        "NORMAL"    : 0.00,
        "NOISE"     : 0.20,
        "FREE_RIDER": 0.50,
        "SCALE"     : 0.70,
        "BYZANTINE" : 0.90,
        "SIGN_FLIP" : 1.00,
    }

    def _compute_trust_score(self, run_dir: Path) -> float:
        """
        Lit behavioral_analysis.json et calcule le trust score (Option B).

        Formule :
          penalite    = somme des poids des attaques détectées sur tous les rounds
          trust_score = max(0, 1 - penalite / total_rounds)

        Poids :
          NOISE=0.20  FREE_RIDER=0.50  SCALE=0.70  BYZANTINE=0.90  SIGN_FLIP=1.00

        Exemple :
          15 rounds, 1 NOISE + 2 SIGN_FLIP → pénalité = 0.20 + 2×1.00 = 2.20
          trust = max(0, 1 - 2.20/15) = 0.853

        Retourne 1.0 si behavioral_analysis.json non disponible (bénéfice du doute).
        """
        ba_path = run_dir / "behavioral_analysis.json"
        if not ba_path.exists():
            return 1.0

        try:
            with open(ba_path, encoding="utf-8") as f:
                reports = json.load(f)

            if not reports:
                return 1.0

            total_rounds = len(reports)
            penalite     = 0.0

            for report in reports:
                # Lire les attack_types du round (nouveau format)
                attack_types = report.get("attack_types", {})
                if attack_types:
                    for attack in attack_types.values():
                        penalite += self.ATTACK_WEIGHTS.get(attack, 0.0)
                else:
                    # Fallback ancien format — juste compter les suspects
                    if report.get("suspects"):
                        penalite += 0.50   # pénalité par défaut si pas de type

            trust_score = max(0.0, 1.0 - penalite / total_rounds)
            return round(trust_score, 4)

        except Exception:
            return 1.0

    # ── Calcul du score composite ─────────────────────────────────────────────

    def _compute_composite_score(self, recall: float, f1: float,
                                  epsilon: float, trust: float) -> float:
        """
        Score composite [0, 1] — plus élevé = meilleur candidat.
          recall  × 0.35  ← priorité absolue (détecter les fraudes)
          f1      × 0.30  ← équilibre précision/recall
          privacy × 0.15  ← récompense un ε faible
          trust   × 0.20  ← entraînement propre (behavioral)
        """
        privacy_score = max(0.0, 1.0 - epsilon)
        score = (
            recall        * 0.35 +
            f1            * 0.30 +
            privacy_score * 0.15 +
            trust         * 0.20
        )
        return round(score, 4)

    # ── Scan des runs ─────────────────────────────────────────────────────────

    def scan_runs(self, runs_dir: str = "logs/runs") -> int:
        """
        Parcourt tous les sous-dossiers de runs_dir.
        Pour chaque run trouve : fl_metrics.json + best checkpoint.
        Enregistre les nouveaux runs dans registry.json.
        Retourne le nombre de nouveaux runs ajoutés.
        """
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            print(f"  ❌ Dossier introuvable : {runs_path.resolve()}")
            return 0

        existing_ids = {m["id"] for m in self._data["models"]}
        added = 0

        for run_dir in sorted(runs_path.iterdir()):
            if not run_dir.is_dir():
                continue

            run_id = run_dir.name

            # Ignorer les runs déjà enregistrés
            if run_id in existing_ids:
                print(f"  ↩  {run_id} — déjà enregistré")
                continue

            # Chercher fl_metrics.json
            metrics_path = run_dir / "fl_metrics.json"
            if not metrics_path.exists():
                print(f"  ⚠️  {run_id} — fl_metrics.json introuvable, ignoré")
                continue

            # Chercher le meilleur checkpoint
            ckpt_dir  = run_dir / "checkpoints"
            ckpt_path = None
            if ckpt_dir.exists():
                # Préférer global_model_best.npz
                last = ckpt_dir / "global_model_best.npz"
                if last.exists():
                    ckpt_path = last
                else:
                    # Prendre le dernier round disponible
                    npz_files = sorted(ckpt_dir.glob("global_model_round_*.npz"))
                    if npz_files:
                        ckpt_path = npz_files[-1]

            if ckpt_path is None:
                print(f"  ⚠️  {run_id} — aucun checkpoint trouvé, ignoré")
                continue

            # Lire les métriques FL
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    fl_data = json.load(f)
            except Exception as e:
                print(f"  ⚠️  {run_id} — erreur lecture métriques : {e}")
                continue

            # Extraire les meilleures métriques (meilleur round selon F1)
            all_rounds = fl_data if isinstance(fl_data, list) else fl_data.get("rounds", [])
            if not all_rounds:
                print(f"  ⚠️  {run_id} — aucun round dans fl_metrics.json")
                continue

            # Chaque round a 2 entrées : fit (f1=0) + eval (f1 réel)
            # On garde uniquement les entrées d'évaluation réelles
            eval_rounds = [
                r for r in all_rounds
                if r.get("n_clients_eval", 0) > 0 and r.get("f1", 0.0) > 0.0
            ]
            rounds = eval_rounds if eval_rounds else all_rounds

            best_round = max(rounds, key=lambda r: r.get("f1", 0.0))
            recall     = float(best_round.get("recall",    0.0))
            f1         = float(best_round.get("f1",        0.0))
            precision  = float(best_round.get("precision", 0.0))
            # max_epsilon est la clé utilisée dans server.py
            epsilon    = float(
                best_round.get("max_epsilon",
                best_round.get("epsilon_max",
                best_round.get("avg_epsilon",
                best_round.get("epsilon", 0.0))))
            )
            round_num  = int(best_round.get("round", 0))

            # Lire les paramètres du run depuis le nom du dossier ou server_env.txt
            params = self._parse_run_params(run_dir)

            # Calcul trust score et composite score
            trust_score = self._compute_trust_score(run_dir)
            comp_score  = self._compute_composite_score(recall, f1, epsilon, trust_score)

            # Compter les alertes behavioral
            ba_alerts = self._count_ba_alerts(run_dir)

            entry = {
                "id"              : run_id,
                "checkpoint"      : str(ckpt_path),
                "best_round"      : round_num,
                "rounds_total"    : len(rounds),
                "f1"              : round(f1, 4),
                "recall"          : round(recall, 4),
                "precision"       : round(precision, 4),
                "epsilon_final"   : round(epsilon, 4),
                "ba_alerts"       : ba_alerts,
                "trust_score"     : trust_score,
                "composite_score" : comp_score,
                "status"          : "candidate",
                "registered_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "params"          : params,
            }

            self._data["models"].append(entry)
            existing_ids.add(run_id)
            added += 1
            print(f"  ✅ {run_id} — F1={f1:.4f}  recall={recall:.4f}"
                  f"  ε={epsilon:.4f}  trust={trust_score:.3f}"
                  f"  score={comp_score:.4f}")

        if added > 0:
            self._save()
            print(f"\n  💾 {added} run(s) ajouté(s) dans registry.json")
        else:
            print(f"\n  ℹ️  Aucun nouveau run trouvé")

        return added

    def _parse_run_params(self, run_dir: Path) -> dict:
        """Essaie de lire les paramètres depuis server_env.txt ou le nom du dossier."""
        params = {}

        # Essayer server_env.txt d'abord
        env_path = run_dir / "server_env.txt"
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        params[k.strip()] = v.strip()
                return params
            except Exception:
                pass

        # Sinon parser le nom du dossier (ex: V4_noise1.00_batch128)
        name = run_dir.name
        if "noise" in name:
            try:
                parts = name.split("_")
                for i, p in enumerate(parts):
                    if p.startswith("noise"):
                        params["dp_noise"] = p.replace("noise", "")
                    if p.startswith("batch"):
                        params["batch_size"] = p.replace("batch", "")
            except Exception:
                pass

        return params

    def _count_ba_alerts(self, run_dir: Path) -> int:
        """Compte le nombre de rounds avec au moins 1 suspect behavioral."""
        ba_path = run_dir / "behavioral_analysis.json"
        if not ba_path.exists():
            return 0
        try:
            with open(ba_path, encoding="utf-8") as f:
                reports = json.load(f)
            return sum(1 for r in reports if r.get("suspects"))
        except Exception:
            return 0

    # ── Enregistrement manuel ─────────────────────────────────────────────────

    def register_run(self, run_name: str, fl_metrics_path: str,
                     checkpoint_path: str, behavioral_path: str = None) -> bool:
        """Enregistre manuellement un run."""
        if any(m["id"] == run_name for m in self._data["models"]):
            print(f"  ⚠️  {run_name} déjà enregistré")
            return False

        try:
            with open(fl_metrics_path, encoding="utf-8") as f:
                fl_data = json.load(f)
        except Exception as e:
            print(f"  ❌ Erreur lecture {fl_metrics_path} : {e}")
            return False

        rounds     = fl_data if isinstance(fl_data, list) else fl_data.get("rounds", [])
        best_round = max(rounds, key=lambda r: r.get("f1", 0.0))
        recall     = float(best_round.get("recall",    0.0))
        f1         = float(best_round.get("f1",        0.0))
        precision  = float(best_round.get("precision", 0.0))
        epsilon    = float(best_round.get("epsilon_max", best_round.get("epsilon", 1.0)))

        ba_dir    = Path(behavioral_path).parent if behavioral_path else Path(fl_metrics_path).parent
        trust     = self._compute_trust_score(ba_dir)
        score     = self._compute_composite_score(recall, f1, epsilon, trust)
        ba_alerts = self._count_ba_alerts(ba_dir)

        self._data["models"].append({
            "id"              : run_name,
            "checkpoint"      : str(checkpoint_path),
            "best_round"      : int(best_round.get("round", 0)),
            "rounds_total"    : len(rounds),
            "f1"              : round(f1, 4),
            "recall"          : round(recall, 4),
            "precision"       : round(precision, 4),
            "epsilon_final"   : round(epsilon, 4),
            "ba_alerts"       : ba_alerts,
            "trust_score"     : trust,
            "composite_score" : score,
            "status"          : "candidate",
            "registered_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params"          : {},
        })
        self._save()
        print(f"  ✅ {run_name} enregistré — score={score:.4f}")
        return True

    # ── Affichage ─────────────────────────────────────────────────────────────

    def list_models(self) -> None:
        """Affiche tous les modèles enregistrés sous forme de tableau."""
        models = self._data["models"]
        if not models:
            print("  Aucun modèle enregistré. Lancer --scan d'abord.")
            return

        prod = self._data.get("production")

        # Trier par composite_score décroissant
        models_sorted = sorted(models, key=lambda m: m["composite_score"], reverse=True)

        print(f"\n{'═'*95}")
        print(f"  {'#':>2}  {'Run ID':<35}  {'F1':>6}  {'Recall':>7}  "
              f"{'ε':>6}  {'Trust':>6}  {'Score':>6}  {'Status'}")
        print(f"  {'─'*2}  {'─'*35}  {'─'*6}  {'─'*7}  "
              f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*12}")

        for i, m in enumerate(models_sorted, 1):
            marker = " ★" if m["id"] == prod else ""
            status = m.get("status", "candidate")
            if m["id"] == prod:
                status = "production"
            print(f"  {i:>2}  {m['id']:<35}  {m['f1']:>6.4f}  {m['recall']:>7.4f}  "
                  f"{m['epsilon_final']:>6.4f}  {m['trust_score']:>6.3f}  "
                  f"{m['composite_score']:>6.4f}  {status}{marker}")

        print(f"{'═'*95}")
        print(f"  Production actuelle : {prod or 'aucune'}")
        print(f"  Total : {len(models)} modèle(s)\n")

    # ── Proposition du meilleur ───────────────────────────────────────────────

    def propose_best(self) -> str | None:
        """Propose le meilleur candidat selon le composite_score."""
        models = self._data["models"]
        if not models:
            print("  Aucun modèle enregistré.")
            return None

        best      = max(models, key=lambda m: m["composite_score"])
        current   = self._data.get("production")
        current_m = next((m for m in models if m["id"] == current), None)

        print(f"\n{'═'*60}")
        print(f"  🏆 Candidat proposé : {best['id']}")
        print(f"     F1      : {best['f1']:.4f}")
        print(f"     Recall  : {best['recall']:.4f}")
        print(f"     ε       : {best['epsilon_final']:.4f}")
        print(f"     Trust   : {best['trust_score']:.3f}")
        print(f"     Score   : {best['composite_score']:.4f}")

        if current_m and current != best["id"]:
            delta_recall = (best["recall"] - current_m["recall"]) * 100
            delta_f1     = (best["f1"]     - current_m["f1"])     * 100
            print(f"\n  📊 vs production ({current}) :")
            print(f"     Recall : {current_m['recall']:.4f} → {best['recall']:.4f}"
                  f"  ({delta_recall:+.1f}%)")
            print(f"     F1     : {current_m['f1']:.4f} → {best['f1']:.4f}"
                  f"  ({delta_f1:+.1f}%)")
        elif current == best["id"]:
            print(f"\n  ✅ Le modèle en production est déjà le meilleur")

        print(f"{'═'*60}\n")
        return best["id"]

    # ── Promotion ─────────────────────────────────────────────────────────────

    def promote(self, run_id: str, confirm: bool = False) -> bool:
        """
        Promeut un run en production :
        1. Copie le checkpoint dans best_model/model.npz
        2. Écrit best_model/metadata.json
        3. Met à jour registry.json
        """
        model = next((m for m in self._data["models"] if m["id"] == run_id), None)
        if model is None:
            print(f"  ❌ Run '{run_id}' non trouvé dans le registry.")
            print(f"     Lancer --scan d'abord.")
            return False

        if not confirm:
            rep = input(f"\n  Promouvoir '{run_id}' en production ? [o/n] : ").strip().lower()
            if rep not in ("o", "oui", "y", "yes"):
                print("  Annulé.")
                return False

        # Vérifier que le checkpoint existe
        ckpt_src = Path(model["checkpoint"])
        if not ckpt_src.exists():
            print(f"  ❌ Checkpoint introuvable : {ckpt_src}")
            return False

        # Copier vers best_model/
        BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dest_model = BEST_MODEL_DIR / "model.npz"
        shutil.copy2(ckpt_src, dest_model)

        # Écrire metadata.json
        metadata = {
            "run_id"           : run_id,
            "promoted_at"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "f1"               : model["f1"],
            "recall"           : model["recall"],
            "precision"        : model["precision"],
            "epsilon_final"    : model["epsilon_final"],
            "trust_score"      : model["trust_score"],
            "composite_score"  : model["composite_score"],
            "ba_alerts"        : model["ba_alerts"],
            "best_round"       : model["best_round"],
            "checkpoint_origin": str(ckpt_src),
            "params"           : model.get("params", {}),
        }
        dest_meta = BEST_MODEL_DIR / "metadata.json"
        with open(dest_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Mettre à jour le statut dans registry
        for m in self._data["models"]:
            if m["id"] == self._data.get("production"):
                m["status"] = "archived"
            if m["id"] == run_id:
                m["status"] = "production"

        self._data["production"] = run_id
        self._save()

        print(f"\n  ✅ '{run_id}' promu en production")
        print(f"     Checkpoint → {dest_model}")
        print(f"     Metadata   → {dest_meta}\n")
        return True

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> None:
        """Affiche le modèle actuellement en production."""
        prod = self._data.get("production")
        if not prod:
            print("  Aucun modèle en production.")
            return

        model = next((m for m in self._data["models"] if m["id"] == prod), None)
        if not model:
            print(f"  ⚠️  Production '{prod}' introuvable dans les modèles.")
            return

        meta_path = BEST_MODEL_DIR / "metadata.json"
        promoted  = "inconnue"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    promoted = json.load(f).get("promoted_at", "inconnue")
            except Exception:
                pass

        print(f"\n{'═'*55}")
        print(f"  🚀 Modèle en production : {prod}")
        print(f"     F1        : {model['f1']:.4f}")
        print(f"     Recall    : {model['recall']:.4f}")
        print(f"     Precision : {model['precision']:.4f}")
        print(f"     ε final   : {model['epsilon_final']:.4f}")
        print(f"     Trust     : {model['trust_score']:.3f}")
        print(f"     BA alerts : {model['ba_alerts']}")
        print(f"     Score     : {model['composite_score']:.4f}")
        print(f"     Promu le  : {promoted}")
        print(f"     Checkpoint: {BEST_MODEL_DIR / 'model.npz'}")
        print(f"{'═'*55}\n")

    # ── Accesseur pour BentoML (étape 2) ─────────────────────────────────────

    def get_production_model(self) -> dict | None:
        """
        Retourne les infos du modèle en production.
        Utilisé par BentoML serving à l'étape 2.
        """
        prod = self._data.get("production")
        if not prod:
            return None
        return next((m for m in self._data["models"] if m["id"] == prod), None)


# ============================================================================
# 🖥️  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model Registry — Fraud Detection FL+DP",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scan",    metavar="RUNS_DIR",
                        help="Scanner un dossier de runs et enregistrer les nouveaux\n"
                             "Ex: --scan logs/runs/")
    parser.add_argument("--list",    action="store_true",
                        help="Lister tous les modèles enregistrés")
    parser.add_argument("--propose", action="store_true",
                        help="Proposer le meilleur candidat")
    parser.add_argument("--promote", metavar="RUN_ID",
                        help="Promouvoir un run en production\n"
                             "Ex: --promote V4_noise1.00_batch128")
    parser.add_argument("--status",  action="store_true",
                        help="Afficher le modèle en production")
    parser.add_argument("--yes",     action="store_true",
                        help="Confirmer la promotion sans demander")

    args = parser.parse_args()

    registry = ModelRegistry()

    if args.scan:
        print(f"\n── Scan de {args.scan} ──")
        registry.scan_runs(args.scan)

    elif args.list:
        registry.list_models()

    elif args.propose:
        registry.propose_best()

    elif args.promote:
        registry.promote(args.promote, confirm=args.yes)

    elif args.status:
        registry.status()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()