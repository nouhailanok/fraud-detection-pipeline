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

    Deux hashes coexistent pour deux usages distincts :

    ┌─────────────────────────────────────────────────────────────────┐
    │  DE002_PAN      → SHA256(cc_num)            STABLE             │
    │                   Utilisé par l'enrichisseur et le ML.         │
    │                   Permet de reconnaître le même client          │
    │                   d'une transaction à l'autre.                  │
    │                   Ne contient pas le vrai numéro de carte.      │
    ├─────────────────────────────────────────────────────────────────┤
    │  DE002_PAN_AUDIT → SHA256(sel_aléatoire + cc_num)  SALÉ        │
    │                    Utilisé uniquement pour les logs d'audit.    │
    │                    Irréversible, résiste aux Rainbow Tables.     │
    │                    Change à chaque transaction → non traçable.  │
    └─────────────────────────────────────────────────────────────────┘
    """
    masked_data = transaction_data.copy()

    pan_field = "DE002_PAN"
    if pan_field in masked_data:
        raw_pan = str(masked_data[pan_field])

        # ── 1. HASH STABLE pour le ML ────────────────────────────────────────
        # SHA256 du cc_num brut, sans sel → toujours identique pour le même client
        # Le GRU peut ainsi regrouper les transactions d'un même utilisateur
        stable_hash = hashlib.sha256(raw_pan.encode("utf-8")).hexdigest()

        # ── 2. HASH SALÉ pour l'audit sécurité ──────────────────────────────
        # Sel aléatoire cryptographique → résultat différent à chaque appel
        # Protège contre les attaques Rainbow Table sur les logs d'audit
        salt        = generate_salt()
        audit_hash  = hashlib.sha256(
            (salt + raw_pan).encode("utf-8")
        ).hexdigest()

        # ── 3. Remplacement dans le dictionnaire ─────────────────────────────
        masked_data[pan_field]         = stable_hash   # ← lu par enrichisseur + ML
        masked_data["DE002_PAN_SALT"]  = salt          # ← conservé pour vérification audit
        masked_data["DE002_PAN_AUDIT"] = audit_hash    # ← utilisé uniquement dans les logs

        # ── 4. Nettoyage mémoire ─────────────────────────────────────────────
        del raw_pan

    # ── Suppression des champs PII nominatifs ────────────────────────────────
    pii_fields_to_remove = [
        "first", "last", "nom", "prenom", "customer_name", "cardholder_name"
    ]
    for field in pii_fields_to_remove:
        if field in masked_data:
            del masked_data[field]

    # ── Audit ─────────────────────────────────────────────────────────────────
    stan = masked_data.get("DE011_STAN", "N/A")
    rrn  = masked_data.get("DE037_RRN",  "N/A")

    gc.collect()
    auditor.log_success(stan=stan, rrn=rrn)

    return masked_data