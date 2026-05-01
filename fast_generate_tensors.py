"""
generate_tensors_from_csv.py  (version multi-core)
════════════════════════════════════════════════════
Court-circuit du pipeline Kafka → génère des .npy IDENTIQUES à ingestion_1.py.

Optimisation : ProcessPoolExecutor avec découpage PAR UTILISATEUR.

Pourquoi PAR UTILISATEUR et pas par lignes ?
  → TransactionEnricher a une mémoire d'historique par client.
  → Couper par lignes = chaque worker repart de zéro pour le même client
    = Velocity_Count_24H, Prev_Txn_Distance_Km, etc. tous faux.
  → Couper par users = chaque worker voit l'historique COMPLET de ses clients.

Usage :
    python generate_tensors_from_csv.py
    python generate_tensors_from_csv.py data/node_1/local_data.csv
"""

import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/FraudTest_sorted.csv")
SAVE_DIR   = Path("data/tensors_FraudTest")
BATCH_SIZE = 1000
NUM_CORES  = max(1, mp.cpu_count() - 2)  # Laisse 1 core libre pour l'OS


# ── Formatage identique à iso_streamer_1.py ───────────────────────────────────

def format_amount(amount: float) -> str:
    return str(int(float(amount) * 100)).zfill(12)

def format_datetime(date_str: str) -> str:
    return pd.to_datetime(date_str).strftime("%m%d%H%M%S")


# ── Worker — traite un groupe d'utilisateurs COMPLETS ────────────────────────

def process_user_chunk(args):
    """
    Reçoit un sous-DataFrame contenant des utilisateurs COMPLETS.
    L'enrichisseur voit l'historique entier de chaque client → features correctes.
    """
    chunk_df, chunk_id = args

    # Import ici pour éviter les problèmes de pickling entre processus
    from security.pii_masking import mask_pii
    from features.enricher import TransactionEnricher
    from features.vectorizer import TransactionVectorizer

    enricher   = TransactionEnricher(time_window_hours=24)
    vectorizer = TransactionVectorizer()

    X_batch         : list = []
    y_batch         : list = []
    saved_files     : list = []
    error_count            = 0
    batch_local_idx        = 0

    records = chunk_df.to_dict("records")  # ~10x plus rapide que iterrows()

    for row in records:
        try:
            # ── ISO 8583 ──────────────────────────────────────────────────────
            raw_transaction = {
                "message_type"        : "0200",
                "DE002_PAN"           : str(row["cc_num"]),
                "DE003_ProcessingCode": "000000",
                "DE004_Amount"        : format_amount(row["amt"]),
                "DE007_DateTime"      : format_datetime(row["trans_date_trans_time"]),
                "DE011_STAN"          : "000001",
                "DE018_MCC"           : str(row["category"]),
                "DE037_RRN"           : str(row["trans_num"])[:12],
                "DE043_MerchantLoc"   : f"{row['merchant']} {row['city']} {row['state']}",
                "DE049_Currency"      : "840",
                "DE123_CustomData"    : {
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

            # ── Masquage PII ──────────────────────────────────────────────────
            masked_transaction = mask_pii(raw_transaction)
            custom_data        = raw_transaction["DE123_CustomData"]

            # ── ml_input — copie exacte de ingestion_1.py ────────────────────
            ml_input = {
                "message_type"     : masked_transaction.get("message_type"),
                "DE002_PAN_HASH"   : masked_transaction.get("DE002_PAN"),
                "DE004_Amount"     : masked_transaction.get("DE004_Amount"),
                "DE007_DateTime"   : masked_transaction.get("DE007_DateTime"),
                "DE018_MCC"        : masked_transaction.get("DE018_MCC"),
                "Client_Lat"       : custom_data.get("lat"),
                "Client_Long"      : custom_data.get("long"),
                "Merch_Lat"        : custom_data.get("merch_lat"),
                "Merch_Long"       : custom_data.get("merch_long"),
                "Merchant_Name"    : custom_data.get("merchant_name"),
                "DOB"              : custom_data.get("dob"),
                "Metadata_is_fraud": custom_data.get("is_fraud"),
            }

            # ── Enrichissement + Vectorisation ────────────────────────────────
            enriched_json = enricher.enrich(ml_input)
            X, y          = vectorizer.vectorize(enriched_json)

            # ── pan_id stable ─────────────────────────────────────────────────
            pan_hex = hashlib.sha256(str(row["cc_num"]).encode()).hexdigest()
            pan_id  = float(int(pan_hex[:12], 16))

            X_batch.append(np.insert(X, 0, pan_id))
            y_batch.append(y)

            # ── Sauvegarde du batch ───────────────────────────────────────────
            if len(X_batch) >= BATCH_SIZE:
                # Nom unique : chunk_id + index local → jamais de collision entre workers
                timestamp = int(row["unix_time"]) * 1000
                x_name    = f"X_batch_{chunk_id:03d}_{batch_local_idx:04d}_{timestamp}.npy"
                y_name    = f"y_batch_{chunk_id:03d}_{batch_local_idx:04d}_{timestamp}.npy"

                np.save(SAVE_DIR / x_name, np.vstack(X_batch))
                np.save(SAVE_DIR / y_name, np.vstack(y_batch))

                saved_files.append(x_name)
                X_batch, y_batch  = [], []
                batch_local_idx  += 1

        except Exception as e:
            error_count += 1
            continue

    # ── Batch résiduel du worker ──────────────────────────────────────────────
    if X_batch:
        timestamp = int(records[-1]["unix_time"]) * 1000 + chunk_id
        x_name    = f"X_batch_{chunk_id:03d}_{batch_local_idx:04d}_{timestamp}.npy"
        y_name    = f"y_batch_{chunk_id:03d}_{batch_local_idx:04d}_{timestamp}.npy"

        np.save(SAVE_DIR / x_name, np.vstack(X_batch))
        np.save(SAVE_DIR / y_name, np.vstack(y_batch))
        saved_files.append(x_name)

    return {
        "chunk_id" : chunk_id,
        "n_users"  : chunk_df["cc_num"].nunique(),
        "n_txns"   : len(records),
        "n_files"  : len(saved_files),
        "errors"   : error_count,
    }


# ── Découpage PAR UTILISATEUR ─────────────────────────────────────────────────

def split_by_users(df: pd.DataFrame, num_cores: int) -> list:
    """
    Divise le DataFrame en `num_cores` groupes d'utilisateurs COMPLETS.
    Chaque groupe contient des clients entiers → l'enrichisseur voit
    l'historique complet de chaque client qu'il traite.
    """
    unique_users = df["cc_num"].unique()
    user_chunks  = np.array_split(unique_users, num_cores)

    return [
        (df[df["cc_num"].isin(user_group)].reset_index(drop=True), chunk_id)
        for chunk_id, user_group in enumerate(user_chunks)
        if len(user_group) > 0
    ]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Nettoyage des anciens tenseurs
    old = list(SAVE_DIR.glob("*.npy"))
    for f in old:
        f.unlink()
    print(f"🗑️  {len(old)} anciens tenseurs supprimés.\n")

    # Chargement du CSV
    print(f"📂 Chargement : {CSV_PATH}")
    df      = pd.read_csv(CSV_PATH)
    n_users = df["cc_num"].nunique()
    print(f"   {len(df):,} transactions  |  {n_users} utilisateurs uniques\n")

    # Découpage par utilisateurs (jamais plus de cores que d'users)
    cores_to_use = min(NUM_CORES, n_users)
    chunks       = split_by_users(df, cores_to_use)

    print(f"🚀 {cores_to_use} workers  |  ~{n_users // cores_to_use} users/worker")
    print(f"{'─' * 55}")

    # Exécution parallèle
    total_txns   = 0
    total_files  = 0
    total_errors = 0

    with ProcessPoolExecutor(max_workers=cores_to_use) as executor:
        futures = {
            executor.submit(process_user_chunk, chunk): chunk[1]
            for chunk in chunks
        }

        with tqdm(total=len(chunks), desc="⚙️  Workers", unit="chunk") as pbar:
            for future in as_completed(futures):
                result        = future.result()
                total_txns   += result["n_txns"]
                total_files  += result["n_files"]
                total_errors += result["errors"]
                pbar.set_postfix({
                    "txns"  : f"{total_txns:,}",
                    "files" : total_files,
                    "errors": total_errors,
                })
                pbar.update(1)

    # Rapport final
    npy_files = sorted(SAVE_DIR.glob("X_batch_*.npy"))
    print(f"\n{'─' * 55}")
    print(f"✅ Terminé !")
    print(f"   Transactions traitées : {total_txns:,}")
    print(f"   Fichiers .npy générés : {len(npy_files)}")
    print(f"   Erreurs               : {total_errors}")

    if npy_files:
        sample = np.load(npy_files[0])
        print(f"   Vérification shape    : {sample.shape}  ← attendu (1000, 27)")
        print(f"   Dossier               : {SAVE_DIR}")


if __name__ == "__main__":
    main()













# import sys
# import hashlib
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm

# # ── Imports ──────────────────────────────────────────────────────────────────
# from security.pii_masking import mask_pii
# from features.enricher import TransactionEnricher
# from features.vectorizer import TransactionVectorizer

# # ── Config ───────────────────────────────────────────────────────────────────
# CSV_PATH   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/node_1/local_data.csv")
# SAVE_DIR   = Path("data/node_1/tensors")
# BATCH_SIZE = 1000
# # NUM_CORES  = mp.cpu_count()  # Use all available cores
# NUM_CORES  = 20  # Use all available cores

# def process_chunk(chunk_data):
#     """
#     Worker function to process a slice of the dataframe.
#     """
#     # Initialize objects inside the worker to avoid pickling issues
#     enricher = TransactionEnricher(time_window_hours=24)
#     vectorizer = TransactionVectorizer()
    
#     X_batch = []
#     y_batch = []
    
#     # Convert chunk to list of dicts (much faster than iterrows)
#     records = chunk_data.to_dict('records')
    
#     for row in records:
#         try:
#             # ── Reconstitution logic (Identical to your original) ──
#             raw_transaction = {
#                 "message_type": "0200",
#                 "DE002_PAN": str(row["cc_num"]),
#                 "DE003_ProcessingCode": "000000",
#                 "DE004_Amount": str(int(float(row["amt"]) * 100)).zfill(12),
#                 "DE007_DateTime": pd.to_datetime(row["trans_date_trans_time"]).strftime("%m%d%H%M%S"),
#                 "DE011_STAN": "000001",
#                 "DE018_MCC": str(row["category"]),
#                 "DE037_RRN": str(row["trans_num"])[:12],
#                 "DE043_MerchantLoc": f"{row['merchant']} {row['city']} {row['state']}",
#                 "DE049_Currency": "840",
#                 "DE123_CustomData": {
#                     "trans_num": row["trans_num"],
#                     "unix_time": int(row["unix_time"]),
#                     "is_fraud": int(row["is_fraud"]),
#                     "lat": float(row["lat"]),
#                     "long": float(row["long"]),
#                     "merch_lat": float(row["merch_lat"]),
#                     "merch_long": float(row["merch_long"]),
#                     "merchant_name": str(row["merchant"]),
#                     "dob": str(row["dob"]),
#                 }
#             }

#             masked_transaction = mask_pii(raw_transaction)
#             custom_data = raw_transaction["DE123_CustomData"]

#             ml_input = {
#                 "message_type": masked_transaction.get("message_type"),
#                 "DE002_PAN_HASH": masked_transaction.get("DE002_PAN"),
#                 "DE004_Amount": masked_transaction.get("DE004_Amount"),
#                 "DE007_DateTime": masked_transaction.get("DE007_DateTime"),
#                 "DE018_MCC": masked_transaction.get("DE018_MCC"),
#                 "Client_Lat": custom_data.get("lat"),
#                 "Client_Long": custom_data.get("long"),
#                 "Merch_Lat": custom_data.get("merch_lat"),
#                 "Merch_Long": custom_data.get("merch_long"),
#                 "Merchant_Name": custom_data.get("merchant_name"),
#                 "DOB": custom_data.get("dob"),
#                 "Metadata_is_fraud": custom_data.get("is_fraud"),
#             }

#             enriched_json = enricher.enrich(ml_input)
#             X, y = vectorizer.vectorize(enriched_json)

#             # pan_id logic
#             pan_hex = hashlib.sha256(str(row["cc_num"]).encode()).hexdigest()
#             pan_id = float(int(pan_hex[:12], 16))

#             X_batch.append(np.insert(X, 0, pan_id))
#             y_batch.append(y)

#             # Save when batch size reached
#             if len(X_batch) >= BATCH_SIZE:
#                 timestamp = int(row["unix_time"]) * 1000
#                 np.save(SAVE_DIR / f"X_batch_{timestamp}.npy", np.vstack(X_batch))
#                 np.save(SAVE_DIR / f"y_batch_{timestamp}.npy", np.vstack(y_batch))
#                 X_batch, y_batch = [], []

#         except Exception as e:
#             continue
    
#     # Final residual batch for this worker
#     if X_batch:
#         timestamp = int(records[-1]["unix_time"]) * 1000 + 1
#         np.save(SAVE_DIR / f"X_batch_{timestamp}.npy", np.vstack(X_batch))
#         np.save(SAVE_DIR / f"y_batch_{timestamp}.npy", np.vstack(y_batch))

# def main():
#     SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
#     # Clean old files
#     for f in SAVE_DIR.glob("*.npy"): f.unlink()
    
#     print(f"📂 Loading data from {CSV_PATH}...")
#     df = pd.read_csv(CSV_PATH)
    
#     # On an i9-14900HX, 20 is great. 
#     # If you notice RAM issues, drop this to 12-14.
#     print(f"🚀 Splitting workload into {NUM_CORES} chunks...")
#     chunks = np.array_split(df, NUM_CORES)
    
#     # FIXED: Only one executor block
#     print(f"⚙️  Processing starting on {NUM_CORES} cores...")
#     with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#         # Wrap the map in list(tqdm(...)) to see progress in real-time
#         results = list(tqdm(
#             executor.map(process_chunk, chunks), 
#             total=len(chunks), 
#             desc="Overall Progress"
#         ))

#     print(f"\n✅ Processing Complete!")
#     print(f"💾 Tensors saved in: {SAVE_DIR}")
#     print(f"📊 Total files: {len(list(SAVE_DIR.glob('*.npy')))}")

# if __name__ == "__main__":
#     main()