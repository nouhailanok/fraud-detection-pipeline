# Rapport de Conformité et Gouvernance des Données

**Date :** 21 Mars 2026
**Rôle :** Responsable Sécurité (Security Officer)
**Projet :** Fraud Detection Pipeline (Distributed / Federated Learning)

## 1. Sécurité du Transport (mTLS) 🔒
Les échanges entre les agents d'ingestion et les serveurs de traitement (Kafka / APIs internes) sont chiffrés et authentifiés mutuellement à l'aide de certificats mTLS générés par notre CA interne.
- **Délivrables :** Script `generate_mtls.sh` et clés privées restreintes.

## 2. PII Masking & Minimisation des Données 🛡️
Conformément aux directives PCI-DSS, les numéros de carte (PAN) ne sont jamais tracés en clair dans les logs, ni stockés en clair sur le disque.
- **Hachage Fort :** Chaque PAN est couplé à un sel cryptographique dynamique avant d'être traité en SHA-256.
- **Effacement mémoire :** Aussitôt la conversion effectuée, l'instruction `del` couplée au Garbage Collector force l'éviction de l'information brute de la mémoire RAM du processus (`pii_masking.py`).
- **Suppression nominative :** Les champs contextuels relatifs à l'identité du porteur (`nom`, `prenom`, etc.) sont purgés de la structure JSON.

## 3. Conformité Federated Learning (Architectural Guarantee) 📍
Les transactions vectorisées (ex: `.parquet` / `.npy`) utilisées par les modèles d'Intelligence Artificielle sont stockées strictement dans les environnements de nœuds locaux du client. 
- **Garantie :** Aucune donnée PII ou données sources en clair ne sort du nœud d'exécution. Les vecteurs mathématiques ne permettent pas l'ingénierie inverse (re-identification).
- Les mécanismes d'agrégation FL ne partagent que les poids synaptiques des modèles, ne contenant aucune trace direct de transaction.

## 4. Audit Trail 📋
Un journal technique immuable `audit.log` tourne sur chaque nœud pour recenser :
- Le volume de données anonymisées et traitées.
- Les potentielles tentatives d'accès ou fuites détectées en transit.
Ce journal sert de preuve administrative pour tout contrôle réglementaire interne.
