from email.utils import format_datetime

import bentoml
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
import torch
from pydantic import BaseModel
from typing import Optional


import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Imports du projet ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.enricher import TransactionEnricher
from features.vectorizer import TransactionVectorizer

# ── Imports du projet2 ──

from collections import defaultdict
from pydantic import BaseModel
from torch.nn import GRU, LSTM, Linear, Dropout, Embedding,Sequential,ReLU

# from models.fraud_lstm import build_model

try:
    from opacus.layers import  DPGRU , DPLSTM
    OPACUS_AVAILABLE = True
    print("[OK]Opacus found")
except ImportError:
    OPACUS_AVAILABLE = False
    print("[ERROR]Opacus not found")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

try:
    from models.fraud_rnn import FraudRNN
    from models.fraud_lstm import FraudLSTM
    print("[OK] FraudLSTM chargé")

except ImportError:
    
    print("[ERROR] Impossible d'importer FraudLSTM")

# ── Configuration ──
MODEL_NAME = os.getenv("BENTOML_MODEL_NAME", "fraud_dpgru_v1")
SEQ_LEN    = int(os.getenv("FL_SEQ_LEN",        "5"))
THRESHOLD  = float(os.getenv("FRAUD_THRESHOLD",  "0.5"))

# ── Node Configuration ──
# Chaque banque lance le service avec son propre NODE_ID
# → Le service charge automatiquement data/node_X/local_data.csv
NODE_ID    = os.getenv("CLIENT_ID", "1")
DATA_ROOT  = Path(os.getenv("FL_DATA_ROOT", "data"))
TRAIN_CSV  = Path(os.getenv(
    "FRAUD_TRAIN_CSV",
    str(DATA_ROOT / f"node_{NODE_ID}" / "local_data.csv")
))

# -------------------------
# Chargement modèle (UNE SEULE FOIS)
# -------------------------

# ============================================================================
# torch.serialization.add_safe_globals([
#     FraudLSTM,
#     LSTM,
#     Linear,
#     Dropout,
#     Embedding,
#     Sequential,ReLU
# ])

torch.serialization.add_safe_globals([
    FraudLSTM,
    FraudRNN,
    GRU,
    LSTM,
    DPGRU,
    DPLSTM,
    Linear,
    Dropout,
    Embedding,
    Sequential,ReLU
])



# ── Charger le modèle BentoML ──
try:
    # model_ref = bentoml.models.get(MODEL_NAME)
    model_ref = bentoml.models.get("fraud_dpgru_v1:2026-04-21_01-06-30_saad_time_1777243082")
    # model_ref = bentoml.pytorch.get("fraud_dpgru_v1:latest")
    with torch.serialization.safe_globals([FraudLSTM,FraudRNN,DPGRU , DPLSTM,GRU,LSTM,Linear,Dropout,Embedding,Sequential,ReLU]):
        try:
            # PyTorch 2.6+ uses a safer default that can block custom classes.
            model = model_ref.load_model(weights_only=False)
        except TypeError:
            # Backward compatibility for BentoML versions that do not expose this kwarg.
            model = model_ref.load_model()
    model.eval()
    model_metadata = model_ref.info.metadata if model_ref else {}
    print("[OK] Modèle chargé avec BentoML")
except Exception as e:
    print(f"[WARN] Modèle BentoML '{MODEL_NAME}' non trouvé : {e}")
    print("[WARN] Lancez d'abord : python serving/save_model.py")
    model_ref = None
    model = None
    model_metadata = {}

# ── Enricher / Vectorizer globaux ──
enricher = TransactionEnricher(time_window_hours=24)
vectorizer = TransactionVectorizer()



# ── Cache d'historique par utilisateur ──
# Format: {pan_id: [vector_1, vector_2, ..., vector_N]}
# user_history: Dict[str, List[np.ndarray]] = {}

def _meta_float(key: str, default: float = 0.0) -> float:
    if key in model_metadata:
        return float(model_metadata.get(key, default))
    metrics = model_metadata.get("metrics", {}) if isinstance(model_metadata.get("metrics"), dict) else {}
    return float(metrics.get(key, default))


def _meta_str(key: str, default: str = "unknown") -> str:
    return str(model_metadata.get(key, default))


# -------------------------
# Service moderne
# -------------------------
@bentoml.service
class FraudDetectionService:

    def __init__(self):
        self.user_history = defaultdict(list)

        self.stats = {
            "n_predictions"  : 0,
            "n_fraud_detected": 0,
            "n_legit_detected": 0,
            "false_alerts"   : 0,
            "latency_ms"     : [],
            "model_version"  : MODEL_NAME,
        }

        # Pré-charger l'historique depuis local_data.csv du node
        print(f"\n  🏦 Service BentoML — Node {NODE_ID}")
        print(f"  📂 Fichier train : {TRAIN_CSV}")
        self._load_history_from_csv(TRAIN_CSV)

        # Si CSV échoue → fallback sur tenseurs .npy
        if len(self.user_history) == 0:
            print(f"  ⚠️  CSV failed → fallback sur tenseurs .npy")
            tensors_dir = DATA_ROOT / f"node_{NODE_ID}" / "tensors"
            self._load_history_from_tensors(tensors_dir)

    # -------------------------
    # INPUT STRUCTURE
    # -------------------------

    class InputData(BaseModel):
        pan_id: str
        amount: float
        lat: float = 0.0
        long: float = 0.0
        merch_lat: Optional[float] = None
        merch_long: Optional[float] = None
        merchant: str = "UNKNOWN"
        trans_date_trans_time: str
        mcc: str = "5999"
        dob: Optional[str] = None


    # ─────────────────────────────────────────────────────────────────────
    # HISTORIQUE — chargement depuis local_data.csv
    # ─────────────────────────────────────────────────────────────────────

    def _load_history_from_csv(self, csv_path: Path) -> None:
        """
        Pré-charge l'historique des utilisateurs depuis data/node_X/local_data.csv.

        Pour chaque utilisateur :
          - Lit ses transactions dans l'ordre chronologique (unix_time)
          - Les envoie dans le pipeline enricher → vectorizer
          - Stocke les SEQ_LEN-1 derniers vecteurs dans user_history

        La transaction de test sera ajoutée par build_sequence() au moment
        du /predict → séquence complète de SEQ_LEN sans zéros pour les users connus.

        Conformité FL ✅ :
          Node 1 charge data/node_1/local_data.csv → users Banque 1 uniquement
          Node 2 charge data/node_2/local_data.csv → users Banque 2 uniquement
          Les données ne quittent jamais le PC de la banque.
        """
        import pandas as pd

        def format_datetime(date_str):
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%m%d%H%M%S")

        if not csv_path.exists():
            print(f"  ⚠️  local_data.csv introuvable : {csv_path}")
            print(f"      Chemin attendu : data/node_{NODE_ID}/local_data.csv")
            print(f"      Les prédictions démarreront sans historique (padding zéros)")
            return

        print(f"  📖 Lecture de {csv_path.name}...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ⚠️  Erreur lecture CSV : {e}")
            return

        print(f"  📊 {len(df):,} transactions | {df['cc_num'].nunique()} utilisateurs")

        # Trier chronologiquement par utilisateur
        df = df.sort_values(["cc_num", "unix_time"])

        n_users   = 0
        n_vectors = 0
        n_errors  = 0

        for pan_id, group in df.groupby("cc_num"):
            pan_str = str(pan_id)
            vectors = []

            for _, row in group.iterrows():
                try:
                    ml_input = {
                        "DE002_PAN_HASH"    : pan_str,
                        "DE004_Amount"      : f"{float(row['amt']):012.0f}",
                        "DE007_DateTime"    : format_datetime(row["trans_date_trans_time"]),
                        "DE018_MCC"         : str(row.get("category", "5999")),
                        "Client_Lat"        : float(row.get("lat",       0.0)),
                        "Client_Long"       : float(row.get("long",      0.0)),
                        "Merch_Lat"         : float(row.get("merch_lat", row.get("lat",  0.0))),
                        "Merch_Long"        : float(row.get("merch_long",row.get("long", 0.0))),
                        "Merchant_Name"     : str(row.get("merchant",   "UNKNOWN")),
                        "DOB"               : str(row.get("dob",        "1990-01-01")),
                        "Metadata_is_fraud" : int(row.get("is_fraud",   0)),
                    }
                    enriched = enricher.enrich(ml_input)
                    X, _     = vectorizer.vectorize(enriched)
                    vectors.append(X)
                except Exception as e:
                    if n_errors == 0:
                        print(f"  ❌ Première erreur : {e}")
                        print(f"     Colonnes CSV : {list(row.index)}")
                        print(f"     Valeurs : {dict(row)}")
                    n_errors += 1
                    continue

            if vectors:
                # Stocker SEQ_LEN-1 dernières transactions dans l'historique
                # Pourquoi SEQ_LEN-1 et pas SEQ_LEN ?
                # → build_sequence() va ajouter la transaction de test
                # → on veut une séquence finale de SEQ_LEN exactement
                self.user_history[pan_str] = vectors[-(SEQ_LEN - 1):]
                n_users   += 1
                n_vectors += len(self.user_history[pan_str])

        print(f"  ✅ Historique Node {NODE_ID} chargé :")
        print(f"     {n_users} utilisateurs connus")
        print(f"     {n_vectors} vecteurs en mémoire")
        if n_errors > 0:
            print(f"     {n_errors} transactions ignorées (erreurs pipeline)")

        # Rapport de couverture
        known = len(self.user_history)
        print(f"     Couverture : {known} users avec historique complet")
        print(f"historique exemple (pan_id={38859492057661}): {self.user_history['38859492057661']}\n")

    def _load_history_from_tensors(self, tensors_dir: Path) -> None:
        """
        Fallback : charge l'historique depuis les tenseurs .npy.
        col0 = PAN_ID, col1-26 = features déjà enrichies + normalisées.
        Utilisé quand local_data.csv échoue (erreur pipeline enricher/vectorizer).

        Limitation : l'enricher.client_state reste vide → les features
        temporelles des nouvelles transactions seront calculées sans historique.
        """
        import glob

        x_files = sorted(glob.glob(str(tensors_dir / "X_batch_*.npy")))
        if not x_files:
            print(f"  ❌ Aucun tenseur trouvé dans {tensors_dir}")
            return

        print(f"  📂 Chargement tenseurs depuis {tensors_dir}...")
        try:
            X_all = np.concatenate([np.load(f) for f in x_files], axis=0)
        except Exception as e:
            print(f"  ❌ Erreur lecture tenseurs : {e}")
            return

        # col0 = PAN_ID, col1-26 = features
        pan_ids  = X_all[:, 0].astype(np.int64)
        features = X_all[:, 1:].astype(np.float32)

        # Normaliser si scaler disponible
        if self.scaler is not None:
            features = self.scaler.transform(features)
            features = np.clip(features, -5.0, 5.0).astype(np.float32)

        # Grouper par utilisateur
        from collections import defaultdict as _dd
        user_vecs = _dd(list)
        for i, pid in enumerate(pan_ids):
            user_vecs[str(pid)].append(features[i])

        # Stocker SEQ_LEN-1 derniers vecteurs
        n_users = 0
        for pan_str, vecs in user_vecs.items():
            self.user_history[pan_str] = vecs[-(SEQ_LEN - 1):]
            n_users += 1

        print(f"  ✅ Tenseurs chargés : {n_users} utilisateurs")
        print(f"  ⚠️  Note : enricher.client_state vide")
        print(f"      Features temporelles des NV transactions = 0")
        print(f"      → Corriger l'erreur CSV pour une détection optimale")

    # ─────────────────────────────────────────────────────────────────────
    # SÉQUENCE — fenêtre glissante
    # ─────────────────────────────────────────────────────────────────────

    def build_sequence(self, pan_id: str, current_vector: np.ndarray) -> np.ndarray:
        """
        Construit une séquence de SEQ_LEN transactions pour l'utilisateur.

        Logique identique à FraudSequenceDataset dans dataloader.py :
          - Ajoute la transaction courante à l'historique de l'utilisateur
          - Si historique < SEQ_LEN → padding par ZÉROS au début
          - Si historique >= SEQ_LEN → fenêtre glissante sur les SEQ_LEN dernières

        Exemple avec SEQ_LEN=5 :
          txn test 1 (user connu)   → [v_train_2, v_train_3, v_train_4, v_train_5, v_test_1]
          txn test 1 (user inconnu) → [0, 0, 0, 0, v_test_1]
        """
        self.user_history[pan_id].append(current_vector.copy())
        history = self.user_history[pan_id][-SEQ_LEN:]
        self.user_history[pan_id] = history

        n_pad    = SEQ_LEN - len(history)
        zero_pad = [np.zeros_like(current_vector) for _ in range(n_pad)]
        seq      = zero_pad + list(history)

        return np.stack(seq, axis=0)  # (SEQ_LEN, n_features)
    
    def _make_prediction(self, sequence: np.ndarray) -> tuple:
        """
        Passe la séquence dans le modèle DPGRU et retourne (prediction, probability).
        """
        if model is None:
            raise RuntimeError("Modèle non chargé. Vérifiez le BentoML store.")

        # [seq_len, n_features] → [1, seq_len, n_features]
        x = torch.from_numpy(sequence).unsqueeze(0).float()

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()
            #prob = torch.sigmoid(logits).view(-1).item()

        prediction = 1 if prob >= THRESHOLD else 0
        return prediction, float(prob)

    # -------------------------
    # API PREDICT
    # -------------------------
    @bentoml.api(route="/predict")
    def predict(self, input_data: InputData) -> dict:
        """
        POST /predict

        Input JSON exemple :
        {
            "input_data": {
                "pan_id": "DE123_abc",
                "amount": 250.00,
                "merchant": "AMAZON",
                "lat": 48.8566,
                "long": 2.3522,
                "trans_date_trans_time": "2019-01-01 06:48:36"
            }
        }

        Output JSON :
        {
            "prediction": 0,
            "probability": 0.12,
            "threshold": 0.5,
            "latency_ms": 8.4,
            "model_version": "fraud_dpgru_v1",
            "sequence_length": 5
        }
        """
        def format_datetime(date_str):
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%m%d%H%M%S")
        
        t0 = time.perf_counter()
        try:
            # ── 1. Préparation du dictionnaire pour l'enrichisseur ──
            # ml_input = {
            #     "DE002_PAN_HASH": str(input_data.get("pan_id", "")),
            #     "DE004_Amount": f"{float(input_data.get('amount', 0)):012.0f}",
            #     "DE007_DateTime": str(input_data.get("unix_time", "")),
            #     "DE018_MCC": str(input_data.get("mcc", "5999")),
            #     "Client_Lat": float(input_data.get("lat", 0.0)),
            #     "Client_Long": float(input_data.get("long", 0.0)),
            #     "Merch_Lat": float(input_data.get("merch_lat", input_data.get("lat", 0.0))),
            #     "Merch_Long": float(input_data.get("merch_long", input_data.get("long", 0.0))),
            #     "Merchant_Name": str(input_data.get("merchant", "UNKNOWN")),
            #     "DOB": str(input_data.get("dob", "1990-01-01")),
            # }
            
            # ─────────────────────────────────────────
            # 1. SAFE INPUT CLEANING
            # ─────────────────────────────────────────

            pan_id = input_data.pan_id
            amount = float(input_data.amount)
            lat = float(input_data.lat)
            lon = float(input_data.long)

            merch_lat = input_data.merch_lat if input_data.merch_lat is not None else lat
            merch_lon = input_data.merch_long if input_data.merch_long is not None else lon

            merchant = input_data.merchant or "UNKNOWN"
            dob = input_data.dob if input_data.dob is not None else "1990-01-01"
            mcc = input_data.mcc or "5999"

            # IMPORTANT : format compatible enricher
            dt_encoded = format_datetime(input_data.trans_date_trans_time)

            # ─────────────────────────────────────────
            # 2. BUILD ML INPUT (STRICT ENRICHER CONTRACT)
            # ─────────────────────────────────────────

            ml_input = {
                "DE002_PAN_HASH": pan_id,
                "DE004_Amount": f"{amount:012.0f}",
                "DE007_DateTime": dt_encoded,
                "DE018_MCC": mcc,

                "Client_Lat": lat,
                "Client_Long": lon,
                "Merch_Lat": merch_lat,
                "Merch_Long": merch_lon,

                "Merchant_Name": merchant,
                "DOB": dob,

                # optionnel debug (NE PAS casser si absent)
                "Metadata_is_fraud": 0
            }

            # ── 2. Enrichissement ──
            enriched_json = enricher.enrich(ml_input)

            # ── 3. Vectorisation ──
            X, _ = vectorizer.vectorize(enriched_json)

            # ── 4. Construction de la séquence utilisateur ──
            # pan_id = input_data.pan_id
            # pan_id = str(input_data.get("pan_id", "unknown"))
            sequence = self.build_sequence(pan_id, X)

            # ── 5. Prédiction ──
            prediction, probability = self._make_prediction(sequence)

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            # ── 6. Mise à jour des stats ──
            self.stats["n_predictions"] += 1
            if prediction == 1:
                self.stats["n_fraud_detected"] += 1
            else:
                self.stats["n_legit_detected"] += 1
            self.stats["latency_ms"].append(latency_ms)
            # Garder seulement les 1000 dernières latences
            if len(self.stats["latency_ms"]) > 1000:
                self.stats["latency_ms"] = self.stats["latency_ms"][-1000:]

            return {
                "prediction": prediction,
                "probability": round(probability, 4),
                "threshold": THRESHOLD,
                "latency_ms": latency_ms,
                "model_version": self.stats["model_version"],
                "sequence_length": SEQ_LEN,
                "pan_id": pan_id,
            }

        except Exception as e:
            return {
                "error": str(e),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

    # -------------------------
    # HEALTH
    # -------------------------
    @bentoml.api(route="/health")
    def health(self) -> dict:
        """
        GET /health

        Retourne la santé du service et l'état du modèle.
        """
        model_loaded = model is not None
        return {
            "status": "ok" if model_loaded else "degraded",
            "model": MODEL_NAME,
            "model_loaded": model_loaded,
            "threshold": THRESHOLD,
            "seq_len": SEQ_LEN,
        }

    # -------------------------
    # METRICS
    # -------------------------
    @bentoml.api(route="/metrics")
    def metrics(self) -> dict:
        """
        GET /service_metrics

        Retourne les métriques internes du service (JSON).
        Les métriques Prometheus restent exposées par BentoML sur /metrics.
        """
        latencies = self.stats["latency_ms"]
        p50 = round(np.percentile(latencies, 50), 2) if latencies else 0.0
        p99 = round(np.percentile(latencies, 99), 2) if latencies else 0.0

        return {
            "n_predictions": self.stats["n_predictions"],
            "n_fraud_detected": self.stats["n_fraud_detected"],
            "n_legit_detected": self.stats["n_legit_detected"],
            "false_alerts": self.stats["false_alerts"],
            "latency_p50_ms": p50,
            "latency_p99_ms": p99,
            "model_version": self.stats["model_version"],
        }

    # -------------------------
    # MODEL INFO
    # -------------------------
    @bentoml.api(route="/model/info")
    def model_info(self) -> dict:
        """
        GET /model_info

        Retourne les métadonnées du modèle en production
        (architecture, métriques FL, privacy, trust).
        """
        meta = model_metadata if model_metadata else {}

        return {
            "model_name": MODEL_NAME,
            "architecture": _meta_str("architecture", "DPGRU"),
            "recall": _meta_float("recall", 0.0),
            "f1": _meta_float("f1", 0.0),
            "epsilon_final": _meta_float("epsilon_final", 0.0),
            "trust_score": _meta_float("trust_score", 0.0),
            "composite_score": _meta_float("composite_score", 0.0),
            "run_name": _meta_str("run_name", _meta_str("run_id", "unknown")),
            "promoted_at": _meta_str("promoted_at", "unknown"),
            "threshold": THRESHOLD,
            "seq_len": SEQ_LEN,
        }
















# import bentoml
# import numpy as np
# import time
# from datetime import datetime
# from collections import defaultdict
# import torch
# from pydantic import BaseModel
# from typing import Optional


# import json
# import os
# import sys
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# # ── Imports du projet ──
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))

# from features.enricher import TransactionEnricher
# from features.vectorizer import TransactionVectorizer

# # ── Imports du projet2 ──

# from collections import defaultdict
# from pydantic import BaseModel
# from torch.nn import LSTM, Linear, Dropout, Embedding,Sequential,ReLU


# try:
#     from opacus.layers import  DPGRU , DPLSTM
#     OPACUS_AVAILABLE = True
#     print("[OK]Opacus found")
# except ImportError:
#     OPACUS_AVAILABLE = False
#     print("[ERROR]Opacus not found")
    

# # from models.fraud_lstm import build_model

# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(BASE_DIR))

# try:
#     from models.fraud_lstm import FraudLSTM
#     print("[OK] FraudLSTM chargé")

# except ImportError:
#     from models.fraud_rnn import FraudRNN
#     print("[ERROR] Impossible d'importer FraudLSTM")

# # ── Configuration ──
# MODEL_NAME = os.getenv("BENTOML_MODEL_NAME", "fraud_dplstm_v1:latest")
# SEQ_LEN = int(os.getenv("FL_SEQ_LEN", "5"))
# THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

# # -------------------------
# # Chargement modèle (UNE SEULE FOIS)
# # -------------------------

# # ============================================================================
# torch.serialization.add_safe_globals([
#     FraudLSTM,
#     LSTM,
#     DPGRU , 
#     DPLSTM,
#     Linear,
#     Dropout,
#     Embedding,
#     Sequential,
#     ReLU
# ])

# # ── Charger le modèle BentoML ──
# try:
#     # model_ref = bentoml.models.get(MODEL_NAME)
#     # model_ref = bentoml.models.get("fraud_lstm_v1:latest")
    
#     model_ref = bentoml.models.get("fraud_dpgru_v1:2026-04-21_01-06-30_saad_time_1777243082")
#     # model_ref = bentoml.pytorch.get("fraud_dplstm_v1:latest")
#     with torch.serialization.safe_globals([FraudLSTM,DPLSTM,DPGRU,LSTM,Linear,Dropout,Embedding,Sequential,ReLU]):
#         try:
#             # PyTorch 2.6+ uses a safer default that can block custom classes.
#             model = model_ref.load_model(weights_only=False)
#         except TypeError:
#             # Backward compatibility for BentoML versions that do not expose this kwarg.
#             model = model_ref.load_model()
#     model.eval()
#     model_metadata = model_ref.info.metadata if model_ref else {}
#     print("[OK] Modèle chargé avec BentoML")
# except Exception as e:
#     print(f"[WARN] Modèle BentoML '{MODEL_NAME}' non trouvé : {e}")
#     print("[WARN] Lancez d'abord : python serving/save_model.py")
#     model_ref = None
#     model = None
#     model_metadata = {}

# # ── Enricher / Vectorizer globaux ──
# enricher = TransactionEnricher(time_window_hours=24)
# vectorizer = TransactionVectorizer()



# # ── Cache d'historique par utilisateur ──
# # Format: {pan_id: [vector_1, vector_2, ..., vector_N]}
# # user_history: Dict[str, List[np.ndarray]] = {}

# def _meta_float(key: str, default: float = 0.0) -> float:
#     if key in model_metadata:
#         return float(model_metadata.get(key, default))
#     metrics = model_metadata.get("metrics", {}) if isinstance(model_metadata.get("metrics"), dict) else {}
#     return float(metrics.get(key, default))


# def _meta_str(key: str, default: str = "unknown") -> str:
#     return str(model_metadata.get(key, default))


# # -------------------------
# # Service moderne
# # -------------------------
# @bentoml.service
# class FraudDetectionService:

#     def __init__(self):
#         self.user_history = defaultdict(list)
#         # self.MAX_SEQ = 20

#         self.stats = {
#             "n_predictions": 0,
#             "n_fraud_detected": 0,
#             "n_legit_detected": 0,
#             "false_alerts": 0,
#             "latency_ms": [],
#             "model_version": MODEL_NAME,
#         }

#     # -------------------------
#     # INPUT STRUCTURE
#     # -------------------------

#     class InputData(BaseModel):
#         pan_id: str
#         amount: float
#         lat: float = 0.0
#         long: float = 0.0
#         merch_lat: Optional[float] = None
#         merch_long: Optional[float] = None
#         merchant: str = "UNKNOWN"
#         trans_date_trans_time: str
#         mcc: str = "5999"
#         dob: Optional[str] = None


#     # def build_sequence(self,pan_id: str, current_vector: np.ndarray) -> np.ndarray:
#     #     """
#     #     Construit une séquence de SEQ_LEN transactions pour l'utilisateur.
#     #     Si l'historique est insuffisant, padding par répétition du vecteur courant.
#     #     """
#     #     # global user_history

#     #     if pan_id not in self.user_history:
#     #         self.user_history[pan_id] = []

#     #     history = self.user_history[pan_id]
#     #     history.append(current_vector)

#     #     # Garder uniquement SEQ_LEN dernières transactions
#     #     if len(history) > SEQ_LEN:
#     #         history = history[-SEQ_LEN:]
#     #         self.user_history[pan_id] = history

#     #     # Padding si nécessaire
#     #     seq = history.copy()
#     #     while len(seq) < SEQ_LEN:
#     #         seq.insert(0, current_vector)

#     #     return np.stack(seq, axis=0)  # [seq_len, n_features]


#     def build_sequence(self, pan_id: str, current_vector: np.ndarray) -> np.ndarray:
#         """
#         Construit une séquence de SEQ_LEN transactions pour l'utilisateur.
 
#         Logique identique à FraudSequenceDataset dans dataloader.py :
#           - Ajoute la transaction courante à l'historique de l'utilisateur
#           - Si historique < SEQ_LEN → padding par ZÉROS au début
#           - Si historique >= SEQ_LEN → fenêtre glissante sur les SEQ_LEN dernières
 
#         Exemple avec SEQ_LEN=5 :
#           txn 1 → [0, 0, 0, 0, v1]
#           txn 2 → [0, 0, 0, v1, v2]
#           txn 5 → [v1, v2, v3, v4, v5]  ← séquence complète
#           txn 6 → [v2, v3, v4, v5, v6]  ← fenêtre glissante
#         """
#         # Ajouter la transaction courante à l'historique
#         self.user_history[pan_id].append(current_vector.copy())
 
#         # Garder uniquement les SEQ_LEN dernières transactions
#         history = self.user_history[pan_id][-SEQ_LEN:]
#         self.user_history[pan_id] = history
 
#         # Padding par zéros si historique insuffisant (même logique que dataloader.py)
#         n_pad = SEQ_LEN - len(history)
#         zero_pad = [np.zeros_like(current_vector) for _ in range(n_pad)]
#         seq = zero_pad + list(history)
 
#         return np.stack(seq, axis=0)  # (SEQ_LEN, n_features)
    
    
#     def _make_prediction(self, sequence: np.ndarray) -> tuple:
#         """
#         Passe la séquence dans le modèle DPGRU et retourne (prediction, probability).
#         """
#         if model is None:
#             raise RuntimeError("Modèle non chargé. Vérifiez le BentoML store.")

#         # [seq_len, n_features] → [1, seq_len, n_features]
#         x = torch.from_numpy(sequence).unsqueeze(0).float()

#         with torch.no_grad():
#             logits = model(x)
#             prob = torch.sigmoid(logits).item()
#             #prob = torch.sigmoid(logits).view(-1).item()

#         prediction = 1 if prob >= THRESHOLD else 0
#         return prediction, float(prob)

#     # -------------------------
#     # API PREDICT
#     # -------------------------
#     @bentoml.api(route="/predict")
#     def predict(self, input_data: InputData) -> dict:
#         """
#         POST /predict

#         Input JSON exemple :
#         {
#             "pan_id": "DE123_abc",
#             "amount": 250.00,
#             "merchant": "AMAZON",
#             "lat": 48.8566,
#             "long": 2.3522,
#             "trans_date_trans_time": "2019-01-01 06:48:36"
#         }

#         Output JSON :
#         {
#             "prediction": 0,
#             "probability": 0.12,
#             "threshold": 0.5,
#             "latency_ms": 8.4,
#             "model_version": "fraud_dplstm_v1",
#             "sequence_length": 5
#         }
#         """
#         def format_datetime(date_str):
#             dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
#             return dt.strftime("%m%d%H%M%S")
        
#         t0 = time.perf_counter()
#         try:
#             # ── 1. Préparation du dictionnaire pour l'enrichisseur ──
#             # ml_input = {
#             #     "DE002_PAN_HASH": str(input_data.get("pan_id", "")),
#             #     "DE004_Amount": f"{float(input_data.get('amount', 0)):012.0f}",
#             #     "DE007_DateTime": str(input_data.get("unix_time", "")),
#             #     "DE018_MCC": str(input_data.get("mcc", "5999")),
#             #     "Client_Lat": float(input_data.get("lat", 0.0)),
#             #     "Client_Long": float(input_data.get("long", 0.0)),
#             #     "Merch_Lat": float(input_data.get("merch_lat", input_data.get("lat", 0.0))),
#             #     "Merch_Long": float(input_data.get("merch_long", input_data.get("long", 0.0))),
#             #     "Merchant_Name": str(input_data.get("merchant", "UNKNOWN")),
#             #     "DOB": str(input_data.get("dob", "1990-01-01")),
#             # }
            
#             # ─────────────────────────────────────────
#             # 1. SAFE INPUT CLEANING
#             # ─────────────────────────────────────────

#             pan_id = input_data.pan_id
#             amount = float(input_data.amount)
#             lat = float(input_data.lat)
#             lon = float(input_data.long)

#             merch_lat = input_data.merch_lat if input_data.merch_lat is not None else lat
#             merch_lon = input_data.merch_long if input_data.merch_long is not None else lon

#             merchant = input_data.merchant or "UNKNOWN"
#             dob = input_data.dob if input_data.dob is not None else "1990-01-01"
#             mcc = input_data.mcc or "5999"

#             # IMPORTANT : format compatible enricher
#             dt_encoded = format_datetime(input_data.trans_date_trans_time)

#             # ─────────────────────────────────────────
#             # 2. BUILD ML INPUT (STRICT ENRICHER CONTRACT)
#             # ─────────────────────────────────────────

#             ml_input = {
#                 "DE002_PAN_HASH": pan_id,
#                 "DE004_Amount": f"{amount:012.0f}",
#                 "DE007_DateTime": dt_encoded,
#                 "DE018_MCC": mcc,

#                 "Client_Lat": lat,
#                 "Client_Long": lon,
#                 "Merch_Lat": merch_lat,
#                 "Merch_Long": merch_lon,

#                 "Merchant_Name": merchant,
#                 "DOB": dob,

#                 # optionnel debug (NE PAS casser si absent)
#                 "Metadata_is_fraud": 0
#             }

#             # ── 2. Enrichissement ──
#             enriched_json = enricher.enrich(ml_input)

#             # ── 3. Vectorisation ──
#             X, _ = vectorizer.vectorize(enriched_json)

#             # ── 4. Construction de la séquence utilisateur ──
#             # pan_id = input_data.pan_id
#             # pan_id = str(input_data.get("pan_id", "unknown"))
#             sequence = self.build_sequence(pan_id, X)

#             # ── 5. Prédiction ──
#             prediction, probability = self._make_prediction(sequence)

#             latency_ms = round((time.perf_counter() - t0) * 1000, 2)

#             # ── 6. Mise à jour des stats ──
#             self.stats["n_predictions"] += 1
#             if prediction == 1:
#                 self.stats["n_fraud_detected"] += 1
#             else:
#                 self.stats["n_legit_detected"] += 1
#             self.stats["latency_ms"].append(latency_ms)
#             # Garder seulement les 1000 dernières latences
#             if len(self.stats["latency_ms"]) > 1000:
#                 self.stats["latency_ms"] = self.stats["latency_ms"][-1000:]

#             return {
#                 "prediction": prediction,
#                 "probability": round(probability, 4),
#                 "threshold": THRESHOLD,
#                 "latency_ms": latency_ms,
#                 "model_version": self.stats["model_version"],
#                 "sequence_length": SEQ_LEN,
#                 "pan_id": pan_id,
#             }

#         except Exception as e:
#             return {
#                 "error": str(e),
#                 "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
#             }

#     # -------------------------
#     # HEALTH
#     # -------------------------
#     @bentoml.api(route="/health")
#     def health(self) -> dict:
#         """
#         GET /health

#         Retourne la santé du service et l'état du modèle.
#         """
#         model_loaded = model is not None
#         return {
#             "status": "ok" if model_loaded else "degraded",
#             "model": MODEL_NAME,
#             "model_loaded": model_loaded,
#             "threshold": THRESHOLD,
#             "seq_len": SEQ_LEN,
#         }

#     # -------------------------
#     # METRICS
#     # -------------------------
#     @bentoml.api(route="/metrics")
#     def metrics(self) -> dict:
#         """
#         GET /service_metrics

#         Retourne les métriques internes du service (JSON).
#         Les métriques Prometheus restent exposées par BentoML sur /metrics.
#         """
#         latencies = self.stats["latency_ms"]
#         p50 = round(np.percentile(latencies, 50), 2) if latencies else 0.0
#         p99 = round(np.percentile(latencies, 99), 2) if latencies else 0.0

#         return {
#             "n_predictions": self.stats["n_predictions"],
#             "n_fraud_detected": self.stats["n_fraud_detected"],
#             "n_legit_detected": self.stats["n_legit_detected"],
#             "false_alerts": self.stats["false_alerts"],
#             "latency_p50_ms": p50,
#             "latency_p99_ms": p99,
#             "model_version": self.stats["model_version"],
#         }

#     # -------------------------
#     # MODEL INFO
#     # -------------------------
#     @bentoml.api(route="/model/info")
#     def model_info(self) -> dict:
#         """
#         GET /model_info

#         Retourne les métadonnées du modèle en production
#         (architecture, métriques FL, privacy, trust).
#         """
#         meta = model_metadata if model_metadata else {}

#         return {
#             "model_name": MODEL_NAME,
#             "architecture": _meta_str("architecture", "DPGRU"),
#             "recall": _meta_float("recall", 0.0),
#             "f1": _meta_float("f1", 0.0),
#             "epsilon_final": _meta_float("epsilon_final", 0.0),
#             "trust_score": _meta_float("trust_score", 0.0),
#             "composite_score": _meta_float("composite_score", 0.0),
#             "run_name": _meta_str("run_name", _meta_str("run_id", "unknown")),
#             "promoted_at": _meta_str("promoted_at", "unknown"),
#             "threshold": THRESHOLD,
#             "seq_len": SEQ_LEN,
#         }




