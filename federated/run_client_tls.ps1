# --- CONFIGURATION DU NŒUD ---
$CLIENT_ID = "1" # À changer par 2, 3 ou 4 selon l'ami
$SERVER_IP = "100.108.251.54"
$SERVER_PORT = "8080"

# --- PARAMÈTRES D'ENTRAÎNEMENT ---
$env:CLIENT_ID = $CLIENT_ID
$env:FLOWER_SERVER_HOST = $SERVER_IP
$env:FLOWER_SERVER_PORT = $SERVER_PORT
$env:DP_NOISE = "1.20"
$env:FL_LR = "0.0005"
$env:FL_BATCH_SIZE = "256"
$env:FL_POS_WEIGHT = "167.0"

# --- CONFIGURATION TLS (PATHS À MODIFIER) ---
# Chaque ami doit avoir ses propres fichiers bankX dans ce dossier
$env:FLOWER_TLS_CA_CERT = "../certs_prod/ca.crt"
$env:FLOWER_TLS_CLIENT_CERT = "../certs_prod/bank$CLIENT_ID.crt"
$env:FLOWER_TLS_CLIENT_KEY = "../certs_prod/bank$CLIENT_ID.key"

# --- EXÉCUTION ---
$DIR = "C:\Users\SAAD\OneDrive\Desktop\CSCC_S4\Projet metier\fraud-detection-pipeline"
$PY = "$DIR\venv\Scripts\python.exe"

Write-Host "🔐 Connexion sécurisée mTLS vers $SERVER_IP..." -ForegroundColor Cyan
& $PY federated/client.py