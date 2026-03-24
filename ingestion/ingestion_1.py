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
    merchant_name: str     # <-- AJOUT : Nom du magasin
    dob: str


# Schéma principal ISO 8583 avec validation
class ISOTransaction(BaseModel):
    message_type: str = Field(...,pattern="^0200$")  # Doit être "0200"
    DE002_PAN: str = Field(..., min_length=8, max_length=19, pattern=r"^\d+$")  # minimum 8 et macimum 9 chiffres
    DE003_ProcessingCode: str = "000000"
    DE004_Amount: str = Field(..., min_length=12, max_length=12)  # Strictement 12 digits
    DE007_DateTime: str = Field(..., min_length=10, max_length=10)  # MMDDhhmmss
    DE011_STAN: str
    DE018_MCC: str
    DE037_RRN: str
    DE043_MerchantLoc: str
    DE049_Currency: str = "840"
    DE123_CustomData: CustomDataSchema  # Validation imbriquée

TOPIC_NAME = 'topic_raw_transactions_1'
BOOTSTRAP_SERVERS_SSL = ['kafka:9093']  # ⚠️ Port SSL pour mTLS
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
                    # auto_offset_reset='earliest',
                    auto_offset_reset='latest',  # Commence à consommer à partir des nouveaux messages
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
        print(f"[SÉCURITÉ] Certificats : CA={CA_CERT}, Client={CLIENT_CERT}")
        
        # Boucle de consommation
        for message in consumer:
            try:
                transaction = message.value
                transaction_count += 1
                
                # Valider le schéma ISO 8583 avec Pydantic
                try:
                    validated_transaction = ISOTransaction.model_validate(transaction)
                    audit_logger.info(f"✅ Transaction #{transaction_count} validée contre le schéma ISO 8583")
                except ValidationError as e:
                    audit_logger.error(
                        f"❌ Erreur validation schéma ISO 8583 - Transaction #{transaction_count}: {e}"
                    )
                    print(f"\n[❌ VALIDATION ERROR - Transaction #{transaction_count}]")
                    print(f"   Erreurs détectées:")
                    for error in e.errors():
                        print(f"   - {error['loc']}: {error['msg']}")
                    continue
                
                # Appliquer le masquage PII (sur les données validées)
                masked_transaction = mask_pii(transaction)
                
                # Extraction des infos pour le log
                pan_masked = masked_transaction.get('DE002_PAN', 'N/A')[:16]
                amount = masked_transaction.get('DE004_Amount', 'N/A')
                is_fraud = transaction.get('DE123_CustomData', {}).get('is_fraud', 'N/A')
                
                # Log d'audit et affichage
                audit_logger.info(
                    f"Transaction #{transaction_count} | PAN masqué: {pan_masked}... | "
                    f"Montant: {amount} | Fraude: {is_fraud}"
                )
                
                print(f"\n[✅ TRANSACTION #{transaction_count}]")
                print(f"   Message Type: {masked_transaction.get('message_type')}")
                print(f"   PAN (masqué): {pan_masked}...")
                print(f"   Montant: {masked_transaction.get('DE004_Amount')}")
                print(f"   DateTime: {masked_transaction.get('DE007_DateTime')}")
                print(f"   STAN: {masked_transaction.get('DE011_STAN')}")
                print(f"   Fraude: {is_fraud}")
                print(f"   Statut: ✅ PII masquée avec succès")
                
                
                # --- LE PONT (MAPPING) VERS VOTRE CODE ---
                custom_data = transaction.get('DE123_CustomData', {})
                
                # 1. On prépare le dictionnaire pour votre Enrichisseur
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
                    "Merchant_Name": custom_data.get('merchant_name'), # Maintenant on est sûr de l'avoir !
                    "DOB": custom_data.get('dob'),                # Maintenant on est sûr de l'avoir !
                    "Metadata_is_fraud": custom_data.get('is_fraud')
                }

                # 2. Vos calculs mathématiques
                enriched_json = enricher.enrich(ml_input)

                # # --- 🛑 AJOUT TEMPORAIRE POUR LE TEST VISUEL ---
                # # Affiche seulement les 3 premières transactions pour ne pas spammer votre terminal
                # if transaction_count <= 3:
                #     print(f"\n[🔍 TEST VISUEL - JSON ENRICHI #{transaction_count}]")
                #     # json.dumps avec indent=4 permet d'afficher le dictionnaire de façon très lisible
                #     print(json.dumps(enriched_json, indent=4))
                # # -----------------------------------------------

                # --- 🛑 TEST VISUEL CONTINU ---
                print(f"\n[🔍 TEST VISUEL - JSON ENRICHI #{transaction_count}]")
                print(json.dumps(enriched_json, indent=4))
                print("-" * 50) # Une petite ligne pour séparer chaque transaction
                # -----------------------------------------------

                X, y = vectorizer.vectorize(enriched_json)
                
                # 3. Ajout au lot (Batching)
                X_batch.append(X)
                y_batch.append(y)

                # 4. Sauvegarde sur disque tous les 1000 logs
                if len(X_batch) >= BATCH_SIZE:
                    save_dir = Path(__file__).parent.parent / "data" / "tensors"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    np.save(save_dir / f"X_batch_{message.timestamp}.npy", np.vstack(X_batch))
                    np.save(save_dir / f"y_batch_{message.timestamp}.npy", np.vstack(y_batch))
                    
                    print(f"💾 [MLOps] Lot de {BATCH_SIZE} tenseurs (26 dimensions) sauvegardé.")
                    
                    X_batch = []
                    y_batch = []
                # ----------------------------------------------------

                # Optionnel : sauvegarder ou envoyer vers un autre système
                # (base de données, topic Kafka de sortie, fichier, etc.)
                
            except json.JSONDecodeError as e:
                audit_logger.error(f"Erreur décodage JSON: {e}")
            except Exception as e:
                audit_logger.error(f"Erreur lors du traitement de la transaction: {e}")
                print(f"[❌] Erreur: {e}")
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
