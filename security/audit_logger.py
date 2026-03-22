import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
AUDIT_LOG_FILE = os.path.join(LOG_DIR, "audit.log")

def setup_audit_logger():
    """
    Configure le logger d'audit conforme aux exigences de sécurité.
    Génère le fichier audit.log localement.
    """
    logger = logging.getLogger("audit_logger")
    logger.setLevel(logging.INFO)
    
    # Eviter d'ajouter plusieurs handlers si la fonction est appelée plusieurs fois
    if not logger.handlers:
        file_handler = logging.FileHandler(AUDIT_LOG_FILE, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - [AUDIT SEC] - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

audit_logger = setup_audit_logger()

# Compteur rudimentaire (peut être étendu avec Redis ou une base de données)
class TransactionAuditor:
    def __init__(self):
        self.processed_transactions = 0
        self.leak_detected = 0

    def log_success(self, stan="N/A", rrn="N/A"):
        self.processed_transactions += 1
        if self.processed_transactions % 100 == 0:
            audit_logger.info(f"{self.processed_transactions} transactions traitées, {self.leak_detected} fuite détectée. Dernière transaction RRN: {rrn}, STAN: {stan}")
        
        # Log détaillé (niveau DEBUG pour ne pas saturer audit.log, ou INFO si strict)
        audit_logger.debug(f"Transaction validée - STAN: {stan} | RRN: {rrn}")

    def log_leak_warning(self, detail: str):
        self.leak_detected += 1
        audit_logger.critical(f"ALERTE DE SÉCURITÉ : Fuite potentielle de données détectée - {detail}")
