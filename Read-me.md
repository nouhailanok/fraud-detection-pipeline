Projet : Pipeline de Détection de Fraude (MLOps & Apprentissage Fédéré)
1. Description du Projet
Ce projet vise à concevoir une plateforme de détection de fraude scalable et respectueuse de la vie privée (Federated Learning). Le système traite des transactions financières au format ISO 8583 en temps réel. Le pipeline est conçu pour être déployé sur 4 nœuds distincts, simulant des entités bancaires décentralisées.

2. Architecture Système (Data Flow)
Le flux de données suit une architecture "Stream-to-Tensor" optimisée pour l'entraînement de réseaux de neurones récurrents (RNN).

[ SOURCE ]
    |--- local_data.csv (Dataset brut : cc_num, amt, lat, long, etc.)
    |
[ GÉNÉRATEUR (Streamer) ]
    |--- Script : generator/iso_streamer_X.py
    |--- Rôle : Simule un flux de transactions. Mappe le CSV vers JSON ISO 8583.
    |--- Transport : Kafka (mTLS sécurisé sur port 9093). Partitionnement par PAN (cc_num).
    |
[ INGESTION & PRÉ-TRAITEMENT ]
    |--- Script : ingestion/ingestion_X.py
    |--- Rôle : Consommation asynchrone par paquets (poll).
    |--- Étapes : 1. Validation Pydantic | 2. Masquage PII | 3. Enrichissement | 4. Vectorisation.
    |--- Sortie : Tenseurs NumPy (.npy) de 27 colonnes (ID + 26 features métier).
    |
[ ENTRAÎNEMENT LOCAL ]
    |--- Script : train_local.py & dataloader.py
    |--- Modèle : GRU RNN (Gated Recurrent Unit).
    |--- Séquençage : Fenêtre glissante de 5 transactions avec Padding par utilisateur.



3. Architecture Technique (Docker)
Le projet est orchestré via Docker Compose, isolant les services de messagerie et les services applicatifs.

Zookeeper : Coordination de Kafka.

Kafka : Broker de messages (SSL/mTLS activé).

Node_Generator : Simulateurs de terminaux de paiement.

Node_Ingestion : Processeurs de données et générateurs de tenseurs.

Flower Server (Prévu) : Agrégateur pour l'apprentissage fédéré.

4. Arborescence du Projet

fraud-detection-pipeline/
├── data/
│   ├── node_1/
│   │   ├── local_data.csv       # Dataset source du nœud
│   │   └── tensors/             # Sortie : Fichiers X_batch_*.npy et y_batch_*.npy
├── federated/                   # Logique Flower (Client/Server)
├── features/
│   ├── enricher.py              # Calcul des features temporelles/géographiques
│   └── vectorizer.py            # Transformation en vecteurs numériques
├── generator/
│   └── iso_streamer_1.py        # Producteur Kafka
├── ingestion/
│   └── ingestion_1.py           # Consommateur et processeur de tenseurs
├── models/
│   └── fraud_rnn.py             # Architecture du modèle GRU
├── security/
│   ├── certs/                   # Certificats CA, Client et Key (mTLS)
│   ├── audit_logger.py          # Journalisation des événements de sécurité
│   └── pii_masking.py           # Masquage des données sensibles (PAN hashing)
├── data/
│   └── dataloader.py            # Logique de split (A/B) et de séquençage
├── docker-compose.yml           # Orchestration des conteneurs
├── requirements.txt             # Dépendances (PyTorch, Pandas, Kafka-python, Pydantic)
└── train_local.py               # Script d'entraînement baseline GPU



5. Spécifications des Données et du Modèle
Identifiant Utilisateur : Extrait du PAN (cc_num), stocké en colonne 0 des fichiers .npy pour le routage dans le Dataloader.

Amnésie du Modèle : L'ID utilisateur est retiré par le Dataloader avant l'entrée dans le modèle. Le GRU ne voit que 26 features.

Séquençage : Longueur de séquence = 5. Un padding de zéros est appliqué si l'historique d'un utilisateur est inférieur à 5 ou si la frontière d'un autre utilisateur est atteinte.

Split Stratégie A : Split temporel intra-utilisateur (80% passé / 20% futur).

Split Stratégie B : Split par population (80% des utilisateurs connus / 20% d'utilisateurs inconnus).

Hardware de référence : GPU NVIDIA RTX 3050 Ti (CUDA activé).