#!/bin/bash

# --- CONFIGURATION RÉSEAU ---
export FLOWER_PORT=8080

# --- CONFIGURATION FLOWER ---
export FL_ROUNDS=15
export FL_MIN_CLIENTS=4

# --- CONFIGURATION TLS (PATHS À MODIFIER) ---
# Ces fichiers doivent être dans ton dossier ~/flower/certs/ sur l'EC2
export FLOWER_TLS_CA_CERT="./certs/ca.crt"
export FLOWER_TLS_SERVER_CERT="./certs/server.crt"
export FLOWER_TLS_SERVER_KEY="./certs/server.key"
export FLOWER_TLS_REQUIRE_CLIENT_CERT="true" # Exige mTLS des banques

# --- VARIABLES STEP 05 (Behavioral Analysis) ---
export TRUST_REJECT_THRESHOLD=0.3
export IF_CONTAMINATION=0.1

echo "🚀 Lancement du serveur Flower sécurisé sur le port $FLOWER_PORT..."
python3 federated/server.py