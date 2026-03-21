# GUIDE ET POLITIQUE DE SÉCURITÉ (OFFICIER DE SÉCURITÉ)

Ce dossier rassemble l'ensemble des modules responsables de l'authentification forte, de l'anonymisation de la donnée, et de l'audit réglementaire (PCI-DSS) du projet de détection de fraude.

## 1. Sécuriser le Transport (mTLS - Mutual TLS) 🔒

Le "Tuyau" (Apache Kafka implémenté sous Docker) ne transporte que des données bancaires chiffrées de bout en bout et refuse les connexions d'un composant qui ne dispose pas d'un certificat explicite signé par notre autorité (CA).

### Comment ça marche ?
1. **Génération locale :** Le script `security/generate_mtls.sh` crée une Autorité de Certification locale (CA) et génère des paires de clés (Privée/Publique) pour les serveurs et les clients. Tout atterrit sous format `.crt` et `.key` (PEM) dans `security/certs/`.
2. **Distribution via Docker Compose :** Le fichier `docker-compose.yml` crée un **Volume** (read-only) pointant vers `./security/certs` et le mappe dans le répertoire `/etc/kafka/secrets/` du nœud Kafka.
3. **Imposition du Client (Client Auth) :** Le noeud Kafka a été configuré avec la variable `KAFKA_SSL_CLIENT_AUTH: required`. Cela force les producteurs et les consommateurs Python à se présenter avec le `client.crt` sous peine d'un rejet sec !

### Comment l'exécuter ?
Dans un terminal (Git Bash conseillé pour Windows) :
```bash
# Etape 1 : Générer les certificats
cd security
chmod +x generate_mtls.sh
./generate_mtls.sh

# Etape 2 : Lancer l'infrastructure (Kafka mTLS paramétré)
cd ..
docker-compose up -d
```
*Le port SSL 9093 sera désormais ouvert et prêt.*

---

## 2. Protection des PII et Masking (Anonymisation) 🛡️

Le module `security/pii_masking.py` transforme la donnée reçue par les flux producteurs avant de l'envoyer dans le modèle IA. 
Cela répond à l'exigence : **Rendre les données bancaires sensibles illisibles tout en restant exploitables (Federated Learning)**.

* **Hachage du PAN :** Le Primary Account Number (`DE002_PAN`) ne sera pas laissé en clair. Il combinera un SEL cryptographique dynamique (généré via `haslib` et `secrets`) pour contrer les attaques de type "dictionnaire" (Rainbow tables).
* **Purge en mémoire (RAM) :** Aussitôt l'information modifiée, le module utilise les primitives Python `del` et appelle explicitement le ramasse-miettes (`gc.collect()`) pour évincer l'empreinte mémoire d'une donnée sensible transitoire.
* **Retrait des Noms :** Les entités identifiantes (Noms, Prénoms) sont écartées des dictionnaires.

---

## 3. Audit et Gouvernance (Logs et Traçabilité) 📋

Afin de garantir et justifier la conformité du pipeline (surtout lors d'un audit de conformité local sur chaque nœud) :
* Le module `security/audit_logger.py` fournit une classe technique et un logger sécurisé écrivant dans `security/logs/audit.log`.
* Il ne stocke jamais de données PII, mais va archiver au compte goutte le rythme de traitement par lot (ex: "X transactions vectorisées"). 
* En cas d'anomalie ou de détection potentiellement risquée pour la conformité (Data Leak), il émettra des évènements textuels taggués `[CRITICAL]`.

Ces traces sont destinées à garantir qu'en environnement "Federated Learning", la donnée d'origine modifiée n'a jamais quitté le sous-réseau.
