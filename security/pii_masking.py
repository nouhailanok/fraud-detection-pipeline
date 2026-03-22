import hashlib
import secrets
import gc

from .audit_logger import audit_logger, TransactionAuditor

# Initialisation de l'auditeur de transactions
auditor = TransactionAuditor()

def generate_salt(length: int = 16) -> str:
    """Génère un sel dynamique cryptographique."""
    return secrets.token_hex(length)

def mask_pii(transaction_data: dict) -> dict:
    """
    Masque les données sensibles (PII) d'une transaction JSON (dictionnaire).
    - Hache le PAN (Primary Account Number) - par défaut DE002_PAN
    - Supprime les données en clair (Noms, Prénoms)
    Note: Les champs ISO ne sont pas figés, on vérifie leur présence d'abord.
    """
    masked_data = transaction_data.copy()
    
    # 1. Hachage du PAN (ex: DE002_PAN)
    pan_field = "DE002_PAN"
    if pan_field in masked_data:
        raw_pan = str(masked_data[pan_field])
        salt = generate_salt()
        
        # Concaténation et hachage (SHA-256)
        pan_with_salt = (salt + raw_pan).encode('utf-8')
        hashed_pan = hashlib.sha256(pan_with_salt).hexdigest()
        
        # Remplacement par la version hachée
        masked_data[pan_field] = hashed_pan
        # On pourrait aussi stocker le sel de manière sécurisée si on doit un jour vérifier le hash.
        masked_data["DE002_PAN_SALT"] = salt 
        
        # Nettoyage de la mémoire (suppression explicite de la variable en clair)
        del raw_pan
        del pan_with_salt

    # 2. Suppression des champs PII nominatifs (Noms, Prénoms, etc.)
    # Liste flexible des champs potentiellement identifiants pouvant fuiter du dataset source
    pii_fields_to_remove = ["first", "last", "nom", "prenom", "customer_name", "cardholder_name"]
    
    for field in pii_fields_to_remove:
        if field in masked_data:
            del masked_data[field]
            
    # Récupération des identifiants (STAN / RRN) pour le registre d'audit
    stan = masked_data.get("DE011_STAN", "N/A")
    rrn = masked_data.get("DE037_RRN", "N/A")
    
    # Forcer le garbage collector pour éliminer les traces en RAM
    gc.collect()
    
    auditor.log_success(stan=stan, rrn=rrn)
    # Log plus fin (optionnel, pour debug)
    # audit_logger.debug("Transaction anonymisée avec succès.")
    
    return masked_data
