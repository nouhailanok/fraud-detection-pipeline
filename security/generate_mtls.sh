#!/bin/bash
# Script de génération des certificats mTLS (Mutual TLS) corrigé pour Git Bash / Windows.

# Cette variable empêche Git Bash de transformer les "/" en chemins Windows
export MSYS_NO_PATHCONV=1

CERTS_DIR="certs"
mkdir -p $CERTS_DIR
cd $CERTS_DIR

echo "1. Génération de la clé et du certificat de l'Autorité de Certification (CA)..."
# On utilise -nodes pour ne pas avoir à taper de mot de passe à chaque fois en dev
openssl req -new -x509 -keyout ca.key -out ca.crt -days 365 -nodes -subj "/CN=Secured-Data-Pipeline-CA"

echo "2. Génération de la clé privée et du certificat pour le Serveur (Kafka)..."
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr -subj "/CN=kafka"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365

echo "3. Génération de la clé privée et du certificat pour le Client (Scripts Python)..."
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr -subj "/CN=Data-Ingestion-Client"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 365

echo "Nettoyage des fichiers temporaires (CSR)..."
rm *.csr *.srl

echo "Certificats générés avec succès dans le dossier security/certs/."
chmod 600 *.key