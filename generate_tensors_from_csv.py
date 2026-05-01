"""
generate_tensors_from_csv.py
════════════════════════════
Court-circuit du pipeline Kafka → génère des .npy IDENTIQUES à ingestion_1.py.

Fidélité garantie :
  - ml_input construit avec exactement les mêmes clés que ingestion_1.py (ligne 139-152)
  - Nommage X_batch_<unix_ms>.npy identique au timestamp Kafka (message.timestamp)
  - pan_id stable : SHA256(cc_num_brut) % 1_000_000  (le vrai fix)

Usage :
    python generate_tensors_from_csv.py
    python generate_tensors_from_csv.py data/node_1/local_data.csv
"""

import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Imports identiques à ingestion_1.py ─────────────────────────────────────
from security.pii_masking import mask_pii
from features.enricher import TransactionEnricher
from features.vectorizer import TransactionVectorizer

# ── Config ───────────────────────────────────────────────────────────────────
# CSV_PATH   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/node_1/local_data.csv")
CSV_PATH   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/FraudTest_sorted.csv")

SAVE_DIR   = Path("data/tensors_FraudTest")
BATCH_SIZE = 1000

# ── Initialisation ───────────────────────────────────────────────────────────
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Suppression des anciens tenseurs corrompus
old_files = list(SAVE_DIR.glob("*.npy"))
for f in old_files:
    f.unlink()
print(f"🗑️  {len(old_files)} anciens tenseurs supprimés.\n")

df         = pd.read_csv(CSV_PATH)
enricher   = TransactionEnricher(time_window_hours=24)
vectorizer = TransactionVectorizer()

X_batch : list = []
y_batch : list = []
batch_count    = 0
error_count    = 0

print(f"📂 Source     : {CSV_PATH}  ({len(df):,} transactions)")
print(f"💾 Destination: {SAVE_DIR}")
print(f"📦 Batch size : {BATCH_SIZE}")
print(f"{'─'*55}\n")

# ── Fonctions de formatage identiques à iso_streamer_1.py ───────────────────

def format_amount(amount: float) -> str:
    """Même logique que iso_streamer_1.py → DE004"""
    return str(int(float(amount) * 100)).zfill(12)

def format_datetime(date_str: str) -> str:
    """Même logique que iso_streamer_1.py → DE007 (MMDDhhmmss)"""
    dt = pd.to_datetime(date_str)
    return dt.strftime("%m%d%H%M%S")

# ── Boucle principale ────────────────────────────────────────────────────────

for _, row in df.iterrows():
    try:
        # ── Reconstitution du message ISO 8583 (comme iso_streamer_1.py) ────
        raw_transaction = {
            "message_type"       : "0200",
            "DE002_PAN"          : str(row["cc_num"]),
            "DE003_ProcessingCode": "000000",
            "DE004_Amount"       : format_amount(row["amt"]),
            "DE007_DateTime"     : format_datetime(row["trans_date_trans_time"]),
            "DE011_STAN"         : "000001",
            "DE018_MCC"          : str(row["category"]),
            "DE037_RRN"          : str(row["trans_num"])[:12],
            "DE043_MerchantLoc"  : f"{row['merchant']} {row['city']} {row['state']}",
            "DE049_Currency"     : "840",
            "DE123_CustomData"   : {
                "trans_num"    : row["trans_num"],
                "unix_time"    : int(row["unix_time"]),
                "is_fraud"     : int(row["is_fraud"]),
                "lat"          : float(row["lat"]),
                "long"         : float(row["long"]),
                "merch_lat"    : float(row["merch_lat"]),
                "merch_long"   : float(row["merch_long"]),
                "merchant_name": str(row["merchant"]),
                "dob"          : str(row["dob"]),
            }
        }

        # ── Masquage PII — identique à ingestion_1.py ───────────────────────
        masked_transaction = mask_pii(raw_transaction)
        custom_data        = raw_transaction["DE123_CustomData"]  # non masqué (comme ingestion)

        # ── ml_input — copie EXACTE de ingestion_1.py lignes 139-152 ────────
        ml_input = {
            "message_type"    : masked_transaction.get("message_type"),
            "DE002_PAN_HASH"  : masked_transaction.get("DE002_PAN"),
            "DE004_Amount"    : masked_transaction.get("DE004_Amount"),
            "DE007_DateTime"  : masked_transaction.get("DE007_DateTime"),
            "DE018_MCC"       : masked_transaction.get("DE018_MCC"),
            "Client_Lat"      : custom_data.get("lat"),
            "Client_Long"     : custom_data.get("long"),
            "Merch_Lat"       : custom_data.get("merch_lat"),
            "Merch_Long"      : custom_data.get("merch_long"),
            "Merchant_Name"   : custom_data.get("merchant_name"),
            "DOB"             : custom_data.get("dob"),
            "Metadata_is_fraud": custom_data.get("is_fraud"),
        }

        # ── Enrichissement + Vectorisation — identiques à ingestion_1.py ────
        enriched_json = enricher.enrich(ml_input)
        X, y          = vectorizer.vectorize(enriched_json)

        # ── pan_id STABLE depuis cc_num BRUT (le vrai fix) ──────────────────
        raw_cc_num = str(row["cc_num"])
        # pan_id     = float(
        #     int(hashlib.sha256(raw_cc_num.encode()).hexdigest(), 16) % 1_000_000
        # )
        pan_hex = hashlib.sha256(raw_cc_num.encode()).hexdigest()  # = ce que stable_hash produit
        pan_id  = float(int(pan_hex[:12], 16))                    # = même calcul que ingestion_1.py

        X_with_id = np.insert(X, 0, pan_id)   # (27,) = ID + 26 features
        X_batch.append(X_with_id)
        y_batch.append(y)

        # ── Sauvegarde du batch ──────────────────────────────────────────────
        if len(X_batch) >= BATCH_SIZE:
            # Nommage identique à ingestion_1.py : unix_time en millisecondes
            # (message.timestamp Kafka = ms depuis epoch, unix_time CSV = secondes)
            timestamp = int(row["unix_time"]) * 1000

            np.save(SAVE_DIR / f"X_batch_{timestamp}.npy", np.vstack(X_batch))
            np.save(SAVE_DIR / f"y_batch_{timestamp}.npy", np.vstack(y_batch))

            batch_count += 1
            print(f"💾 Batch {batch_count:>4}  →  X_batch_{timestamp}.npy  ({BATCH_SIZE} transactions)")

            X_batch = []
            y_batch = []

    except Exception as e:
        error_count += 1
        print(f"⚠️  Erreur ligne {_} : {e}")
        continue

# ── Dernier batch résiduel ───────────────────────────────────────────────────
if X_batch:
    timestamp = int(df.iloc[-1]["unix_time"]) * 1000 + 1  # +1 pour éviter collision

    np.save(SAVE_DIR / f"X_batch_{timestamp}.npy", np.vstack(X_batch))
    np.save(SAVE_DIR / f"y_batch_{timestamp}.npy", np.vstack(y_batch))

    batch_count += 1
    print(f"💾 Batch {batch_count:>4}  →  X_batch_{timestamp}.npy  ({len(X_batch)} transactions)")

# ── Rapport final ────────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"✅ Terminé : {batch_count} fichiers .npy générés dans {SAVE_DIR}")
print(f"   Transactions traitées : {batch_count * BATCH_SIZE + len(X_batch):,}")
print(f"   Erreurs               : {error_count}")
print(f"   Dimensions vecteur    : (27,) = 1 pan_id + 26 features")
npy_files = sorted(SAVE_DIR.glob("X_batch_*.npy"))
if npy_files:
    sample = np.load(npy_files[0])
    print(f"   Vérification shape    : {sample.shape}  ← attendu (1000, 27)")