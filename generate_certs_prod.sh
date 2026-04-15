#!/bin/bash
OUTPUT_DIR="certs_prod"
SERVER_IP="100.108.251.54"
DAYS=365

mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

echo "🛡️ Correction de la génération PKI pour Windows..."

# 1. CA - Note le double // au début
echo "👉 1. Création de la CA..."
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days $DAYS -out ca.crt \
    -subj "//C=MA/ST=Casablanca/L=Casablanca/O=ENSAM/CN=FraudDetection-CA"

# 2. SERVEUR
echo "👉 2. Création du certificat Serveur..."
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr \
    -subj "//C=MA/ST=Casablanca/L=Casablanca/O=ENSAM/CN=$SERVER_IP"

cat > server_ext.cnf <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = IP:$SERVER_IP
EOF

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out server.crt -days $DAYS -sha256 -extfile server_ext.cnf

# 3. CLIENTS
for i in {1..4}
do
    echo "👉 3.$i. Création du certificat pour Banque_$i..."
    openssl genrsa -out bank$i.key 2048
    openssl req -new -key bank$i.key -out bank$i.csr \
        -subj "//C=MA/ST=Casablanca/L=Casablanca/O=ENSAM/CN=bank$i.local"
    openssl x509 -req -in bank$i.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
        -out bank$i.crt -days $DAYS -sha256
done

rm *.csr server_ext.cnf ca.srl
echo "✅ Terminé ! Vérifie la présence des fichiers .crt"