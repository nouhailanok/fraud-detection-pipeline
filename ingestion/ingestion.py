import json
import sys
from pathlib import Path
from kafka import KafkaConsumer

# Ajouter le chemin du dossier parent pour importer les modules security
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.pii_masking import mask_pii
from security.audit_logger import audit_logger

TOPIC_NAME = 'topic_raw_transactions'
BOOTSTRAP_SERVERS = ['localhost:9092']
CONSUMER_GROUP = 'fraud-detection-group'


def consume_and_process():
    """
    Consomme les messages ISO 8583 depuis Kafka, applique le masquage PII
    et enregistre les transactions traitées.
    """
    
    try:
        # Initialisation du KafkaConsumer
        consumer = KafkaConsumer(
            TOPIC_NAME,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=CONSUMER_GROUP,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            max_poll_records=500,
            session_timeout_ms=30000
        )
        
        audit_logger.info(f"✅ Consumer connecté au topic '{TOPIC_NAME}'")
        print(f"[INGESTION] En attente de messages du topic '{TOPIC_NAME}'...")
        
        transaction_count = 0
        
        # Boucle de consommation
        for message in consumer:
            try:
                transaction = message.value
                transaction_count += 1
                
                # Appliquer le masquage PII
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
        consumer.close()
        audit_logger.info("Consumer Kafka fermé")


if __name__ == "__main__":
    consume_and_process()
