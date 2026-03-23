# 🔒 Configuration mTLS - Guide d'Exécution

Ce document explique comment déployer le pipeline avec **Mutual TLS (mTLS)** activé pour sécuriser la communication Kafka.

---

## 📋 Prérequis

- **Docker & Docker Compose** installés
- **Python 3.8+** avec les dépendances : `pip install -r requirements.txt`
- Un dataset CSV dans `data/fraudTrain.csv` (ou généré)

---

## 🔐 Étape 1 : Générer les Certificats mTLS

Les certificats mTLS garantissent une authentification mutuelle entre le client et le serveur.

### Exécution du script de génération :

```bash
cd security
chmod +x generate_mtls.sh          # Rendre l'script exécutable
./generate_mtls.sh                  # Générer CA, serveur et certificats clients
cd ..
```

### Résultat attendu :
```
security/certs/
├── ca.crt              # Autorité de Certification
├── ca.key              # Clé CA (à garder secrète)
├── server.crt          # Certificat serveur Kafka
├── server.key          # Clé privée serveur
├── client.crt          # Certificat client (Producer/Consumer)
└── client.key          # Clé privée client
```

---

## 🐳 Étape 2 : Lancer l'Infrastructure Kafka (mTLS)

Le fichier `security/docker-compose.yml` est configuré pour :
- Écouter sur **port 9093** (SSL/mTLS)
- Exiger les certificats clients (`KAFKA_SSL_CLIENT_AUTH: required`)

```bash
cd security
docker-compose up -d
cd ..
```

### Vérifier l'état :
```bash
docker ps                           # Voir les conteneurs
docker logs kafka-broker            # Voir les logs Kafka
```

---

## 📤 Étape 3 : Lancer le Producer (ISO Streamer)

Le Producer envoie les transactions CSV au topic Kafka via **mTLS**.

```bash
python generator/iso_streamer.py
```

### Résultat attendu :
```
[📤 PRODUCER] Démarrage du streaming ISO 8583 vers Kafka...
[🔒 SÉCURITÉ] Connexion mTLS activée (port 9093)
[📂 DATASET] Fichier: data/fraudTrain.csv

<ISO transaction JSON>
<ISO transaction JSON>
...
```

### Vérifier la connexion :
```bash
# Terminal séparé, vérifier les messages dans Kafka
docker exec kafka-broker kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic topic_raw_transactions \
  --from-beginning
```

---

## 📥 Étape 4 : Lancer le Consumer (Ingestion)

Le Consumer reçoit les transactions, les valide (Pydantic), masque les PII et les enregistre.

```bash
python ingestion/ingestion.py
```

### Résultat attendu :
```
[INGESTION] Connecté via mTLS au topic 'topic_raw_transactions'...
[SÉCURITÉ] Certificats : CA=.../security/certs/ca.crt, Client=.../security/certs/client.crt

[✅ TRANSACTION #1]
   Message Type: 0200
   PAN (masqué): 5425233070632...
   Montant: 000000015250
   DateTime: 0915143521
   STAN: 000001
   Fraude: 1
   Statut: ✅ PII masquée avec succès

[✅ TRANSACTION #2]
...
```

---

## 🔍 Vérification de la Sécurité

### 1. Audit Logs
Les logs d'audit sont enregistrés dans `security/logs/audit.log` :
```bash
tail -f security/logs/audit.log
```

### 2. Validation Pydantic
Les transactions malformées sont rejetées avec détails :
```
[❌ VALIDATION ERROR - Transaction #5]
   Erreurs détectées:
   - ('DE004_Amount',): ensure this value has exactly 12 characters
```

### 3. Masquage PII
Les PANs sont hachés (SHA-256 + salt) :
```json
{
  "DE002_PAN": "a1b2c3d4e5f6...",  // Haché
  "DE002_PAN_SALT": "7f8e9d0a..."    // Salt utilisé
}
```

---

## 🚨 Dépannage

### Erreur : "certificats non trouvés"
```
ssl.SSLError: [Errno 1] _ssl.c:... certificate_verify_failed
```
**Solution :** Vérifier que `security/certs/` existe et contient tous les fichiers `.crt` et `.key`.

### Erreur : "connexion refusée"
```
KafkaError: Unable to connect to ...
```
**Solution :** Vérifier que Docker Kafka est bien lancé :
```bash
docker-compose -f security/docker-compose.yml logs kafka-broker
```

### Erreur : "fichier fraudTrain.csv non trouvé"
**Solution :** Créer ou ajouter un dataset dans `data/fraudTrain.csv`.

---

## 📊 Architecture Sécurisée (3 Couches)

```
┌─────────────────────────────────────────────────────────────┐
│                   Pipeline Fraude Detection                   │
├─────────────────────────────────────────────────────────────┤
│                                                                 │
│  DATA (csv) → PRODUCER (iso_streamer.py, mTLS) → KAFKA (SSL/mTLS)
│                            ↓                              ↓
│                     [🔒 SÉCURITÉ 1]              [🔒 SÉCURITÉ 3]
│                     - Schema Validation          - mTLS
│                     - ISO 8583 Mapping           - Client Auth
│                                                                 │
│  ← KAFKA (SSL/mTLS) ← CONSUMER (ingestion.py, mTLS) ← LOGS
│           ↓                  ↓
│      [🔒 SÉCURITÉ 3]  [🔒 SÉCURITÉ 2]
│      - mTLS              - PII Masking
│      - Client Auth       - Audit Logger
│                          - Schema Validation
│
└─────────────────────────────────────────────────────────────┘
```

### Les 3 Couches de Protection :
1. **Validation Pydantic** → Intégrité des données
2. **PII Masking** → Confidentialité (PCI-DSS)
3. **mTLS (SSL)** → Authentification mutuelle + Chiffrement transport

---

## ✅ Validation Complète

Pour valider la chaîne complète, exécute dans 3 terminaux différents :

**Terminal 1 : Infrastructure**
```bash
cd security
docker-compose up
```

**Terminal 2 : Producer**
```bash
python generator/iso_streamer.py
```

**Terminal 3 : Consumer**
```bash
python ingestion/ingestion.py
```

**Terminal 4 (optionnel) : Monitoring**
```bash
tail -f security/logs/audit.log
```

---

## 🎯 Résumé

✅ **Générateur ISO 8583 :** Streaming sécurisé (mTLS)
✅ **Validation Pydantic :** Intégrité des données
✅ **PII Masking :** Conformité PCI-DSS
✅ **Audit Logger :** Traçabilité complète
✅ **mTLS:** Authentification & Chiffrement

La pipeline est maintenant **sécurisée de bout en bout** ! 🎉
