import json
import sys
from pathlib import Path
from kafka import KafkaConsumer
from pydantic import BaseModel, Field, ValidationError
import numpy as np

# Ajouter le chemin du dossier parent pour importer les modules security
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.pii_masking import mask_pii
from security.audit_logger import audit_logger

# --- AJOUT 2 : VOS IMPORTS MLOPS ---
from features.enricher import TransactionEnricher
from features.vectorizer import TransactionVectorizer


# Schéma de validation pour les données custom (DE123)
class CustomDataSchema(BaseModel):
    trans_num: str
    unix_time: int
    is_fraud: int
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    merchant_name: str
    dob: str


# Schéma principal ISO 8583 avec validation
class ISOTransaction(BaseModel):
    message_type: str = Field(...,pattern="^0200$")
    DE002_PAN: str = Field(..., min_length=8, max_length=19, pattern=r"^\d+$")
    DE003_ProcessingCode: str = "000000"
    DE004_Amount: str = Field(..., min_length=12, max_length=12)
    DE007_DateTime: str = Field(..., min_length=10, max_length=10)
    DE011_STAN: str
    DE018_MCC: str
    DE037_RRN: str
    DE043_MerchantLoc: str
    DE049_Currency: str = "840"
    DE123_CustomData: CustomDataSchema

TOPIC_NAME = 'topic_raw_transactions_2'
BOOTSTRAP_SERVERS_SSL = ['kafka:9093']
CONSUMER_GROUP = 'fraud-detection-group'

# Chemins vers les certificats mTLS
BASE_CERT_PATH = Path(__file__).parent.parent / "security" / "certs"
CA_CERT = str(BASE_CERT_PATH / "ca.crt")
CLIENT_CERT = str(BASE_CERT_PATH / "client.crt")
CLIENT_KEY = str(BASE_CERT_PATH / "client.key")


def consume_and_process():
    """
    Consomme les messages ISO 8583 depuis Kafka (mTLS sécurisé), applique le masquage PII
    et enregistre les transactions traitées.
    """
    
    consumer = None
    transaction_count = 0

    # --- AJOUT 5 : INITIALISATION MLOPS ---
    enricher = TransactionEnricher(time_window_hours=24)
    vectorizer = TransactionVectorizer()
    X_batch = []
    y_batch = []
    BATCH_SIZE = 1000
    # --------------------------------------

    try:
        for attempt in range(1, 61):
            try:
                consumer = KafkaConsumer(
                    TOPIC_NAME,
                    bootstrap_servers=BOOTSTRAP_SERVERS_SSL,
                    security_protocol='SSL',
                    ssl_cafile=CA_CERT,
                    ssl_certfile=CLIENT_CERT,
                    ssl_keyfile=CLIENT_KEY,
                    ssl_check_hostname=False,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    group_id=CONSUMER_GROUP,
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    max_poll_records=500,
                    session_timeout_ms=30000
                )
                break
            except Exception as error:
                print(f"[INGESTION] Kafka indisponible (tentative {attempt}/60): {error}")
                audit_logger.warning(f"Kafka indisponible (tentative {attempt}/60): {error}")
                import time
                time.sleep(2)

        if consumer is None:
            raise RuntimeError("Impossible de se connecter à Kafka après plusieurs tentatives")
        
        audit_logger.info(f"✅ Consumer connecté au topic '{TOPIC_NAME}' avec mTLS (port 9093)")
        print(f"[INGESTION] Connecté via mTLS au topic '{TOPIC_NAME}'...")
        
        # Boucle de consommation
        for message in consumer:
            try:
                transaction = message.value
                transaction_count += 1
                
                # 1. Validation Pydantic
                try:
                    ISOTransaction.model_validate(transaction)
                    audit_logger.info(f"✅ Transaction #{transaction_count} validée contre le schéma ISO 8583")
                except ValidationError as e:
                    audit_logger.error(
                        f"❌ Erreur validation schéma ISO 8583 - Transaction #{transaction_count}: {e}"
                    )
                    continue
                
                # 2. Masquage PII
                masked_transaction = mask_pii(transaction)
                custom_data = transaction.get('DE123_CustomData', {})
                
                # 3. Préparation du dictionnaire pour l'Enrichisseur
                ml_input = {
                    "message_type": masked_transaction.get('message_type'),
                    "DE002_PAN_HASH": masked_transaction.get('DE002_PAN'),
                    "DE004_Amount": masked_transaction.get('DE004_Amount'),
                    "DE007_DateTime": masked_transaction.get('DE007_DateTime'),
                    "DE018_MCC": masked_transaction.get('DE018_MCC'),
                    "Client_Lat": custom_data.get('lat'),
                    "Client_Long": custom_data.get('long'),
                    "Merch_Lat": custom_data.get('merch_lat'),
                    "Merch_Long": custom_data.get('merch_long'),
                    "Merchant_Name": custom_data.get('merchant_name'),
                    "DOB": custom_data.get('dob'),
                    "Metadata_is_fraud": custom_data.get('is_fraud')
                }

                # 4. Enrichissement
                enriched_json = enricher.enrich(ml_input)

                # 5. Vectorisation (Sortie : X = 26 dimensions)
                X, y = vectorizer.vectorize(enriched_json)

                # --- 🛠️ MODIFICATION : AJOUT DE L'IDENTIFIANT ---
                pan_hex = enriched_json["PAN_HASH"]
                pan_id = float(int(pan_hex[:12], 16))

                # On insère l'ID à l'index 0. X passe de 26 à 27 colonnes.
                X_with_id = np.insert(X, 0, pan_id) 
                # -----------------------------------------------

                # 6. Ajout au lot
                X_batch.append(X_with_id)
                y_batch.append(y)

                # 7. Sauvegarde sur disque
                if len(X_batch) >= BATCH_SIZE:
                    save_dir = Path(__file__).parent.parent / "data" / "node_2" / "tensors"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = message.timestamp
                    np.save(save_dir / f"X_batch_{timestamp}.npy", np.vstack(X_batch))
                    np.save(save_dir / f"y_batch_{timestamp}.npy", np.vstack(y_batch))
                    
                    print(f"💾 [MLOps] Lot de {BATCH_SIZE} sauvegardé. (Col 0: ID, Col 1-27: Features)")
                    
                    X_batch = []
                    y_batch = []
                
            except Exception as e:
                audit_logger.error(f"Erreur lors du traitement: {e}")
                continue
    
    except KeyboardInterrupt:
        audit_logger.info(f"\n⛔ Consumer arrêté (KeyboardInterrupt) - {transaction_count} transactions traitées")
        print(f"\n[INGESTION] Consumer arrêté - {transaction_count} transactions traitées")
    
    except Exception as e:
        audit_logger.critical(f"Erreur critique du consumer: {e}")
        print(f"[❌] Erreur critique: {e}")
    
    finally:
        if consumer is not None:
            consumer.close()
            audit_logger.info("Consumer Kafka fermé")


if __name__ == "__main__":
    consume_and_process()
