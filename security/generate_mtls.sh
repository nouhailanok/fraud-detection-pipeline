#!/bin/bash
# Script de génération des certificats mTLS (Mutual TLS) pour Docker/Git Bash (Windows compatible).

set -euo pipefail

# Cette variable empêche Git Bash de transformer les "/" en chemins Windows
export MSYS_NO_PATHCONV=1

CERTS_DIR="certs"
mkdir -p "$CERTS_DIR"
cd "$CERTS_DIR"

create_server_cert() {
	local cert_name="$1"
	local common_name="$2"
	local san="$3"

	cat > "${cert_name}_ext.cnf" <<EOF
subjectAltName=${san}
extendedKeyUsage=serverAuth
keyUsage=digitalSignature,keyEncipherment
basicConstraints=CA:FALSE
EOF

	openssl genrsa -out "${cert_name}.key" 2048
	openssl req -new -key "${cert_name}.key" -out "${cert_name}.csr" -subj "/CN=${common_name}"
	openssl x509 -req -in "${cert_name}.csr" -CA ca.crt -CAkey ca.key -CAcreateserial -out "${cert_name}.crt" -days 365 -extfile "${cert_name}_ext.cnf"
}

create_client_cert() {
	local cert_name="$1"
	local common_name="$2"

	cat > "${cert_name}_ext.cnf" <<EOF
extendedKeyUsage=clientAuth
keyUsage=digitalSignature,keyEncipherment
basicConstraints=CA:FALSE
EOF

	openssl genrsa -out "${cert_name}.key" 2048
	openssl req -new -key "${cert_name}.key" -out "${cert_name}.csr" -subj "/CN=${common_name}"
	openssl x509 -req -in "${cert_name}.csr" -CA ca.crt -CAkey ca.key -CAcreateserial -out "${cert_name}.crt" -days 365 -extfile "${cert_name}_ext.cnf"
}

echo "1. Génération de la clé et du certificat de l'Autorité de Certification (CA)..."
openssl req -new -x509 -keyout ca.key -out ca.crt -days 365 -nodes -subj "/CN=Secured-Data-Pipeline-CA"

echo "2. Génération du certificat serveur Kafka..."
create_server_cert "server" "kafka" "DNS:kafka,DNS:localhost,IP:127.0.0.1"
cat server.key server.crt > server.pem

echo "3. Génération du certificat serveur Flower..."
create_server_cert "flower_server" "flower_server" "DNS:flower_server,DNS:localhost,IP:127.0.0.1"

echo "4. Génération des certificats clients des générateurs (4 nœuds)..."
for i in 1 2 3 4; do
	create_client_cert "gen_${i}" "generator-${i}"
done

echo "5. Génération des certificats clients des nœuds d'ingestion (4 nœuds)..."
for i in 1 2 3 4; do
	create_client_cert "ing_${i}" "ingestion-${i}"
done

echo "Nettoyage des fichiers temporaires..."
rm -f *.csr *.srl *_ext.cnf

echo "Certificats générés avec succès dans security/certs/."
chmod 600 *.key || true