# Pipeline de Détection de Fraude — Federated Learning + Differential Privacy

## 1. Description du Projet

Ce projet implémente une plateforme de détection de fraude décentralisée, respectueuse de la vie privée, basée sur le **Federated Learning (FL)** avec **Differential Privacy (DP)**. Le système traite des transactions financières au format ISO 8583 en temps réel, réparties sur **4 nœuds bancaires indépendants**.

**Points clés :**
- Chaque banque garde ses données localement — aucune donnée client ne quitte le nœud
- Le modèle global est construit par agrégation des mises à jour locales via FedAvg
- La Differential Privacy (Opacus) garantit qu'aucun client individuel n'est identifiable dans les gradients
- Budget DP mesuré : ε = 0.26 après 3 rounds (bien en dessous du seuil ε ≤ 1.0)

---

## 2. Architecture Système

```
[ SOURCE ]
    local_data.csv (cc_num, amt, lat, long, merchant, category...)
         |
[ GÉNÉRATEUR ]
    iso_streamer_X.py
    → Simule un flux ISO 8583 via Kafka (mTLS port 9093)
    → Partitionnement par PAN (cc_num)
         |
[ INGESTION ]
    ingestion_X.py
    → Validation Pydantic
    → Masquage PII (SHA-256 stable pour ML, aléatoire pour logs)
    → Enrichissement (vélocité, distance, time_delta)
    → Vectorisation (12 features numériques + 14 OHE catégorielles)
    → Sortie : tenseurs .npy (col0=PAN_ID, col1-26=features)
         |
[ GÉNÉRATION TENSEURS ]
    fast_generate_tensors.py
    → Multi-core par utilisateur
    → Produit X_batch_*.npy et y_batch_*.npy
         |
[ ENTRAÎNEMENT LOCAL ]        [ FEDERATED LEARNING ]
    train_local.py                 federated/
    → DPGRU hidden=128             → client.py (FlowerClient + Opacus DP)
    → Approche A ou B              → server.py (FedAvgWithLogging)
    → AUC-ROC 0.9984               → 4 nœuds × FedAvg → modèle global
    → FA calibrées : 86-170        → Checkpoint automatique par round
```

---

## 3. Modèle — DPGRU

Le modèle retenu après 15 runs de comparaison est le **DPGRU** (Differential Privacy GRU).

| Paramètre | Valeur | Justification |
|---|---|---|
| Architecture | DPGRU (Opacus) | Compatible per-sample gradients pour DP |
| input_dim | 26 | Fixé par le vectorizer (12 num + 14 OHE) |
| hidden_dim | 128 | Optimal (64=underfit, 256=overfit) |
| num_layers | 2 | Patterns locaux + séquence globale |
| dropout | 0.2 | Régularisation standard |
| seq_len | 5 | ~2-3 derniers jours de transactions |
| Paramètres totaux | 167,297 | Léger pour la communication FL |

**Classifieur MLP après GRU :** `Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→1)`

---

## 4. Données et Splits

**Dataset :** 4 nœuds bancaires, chacun avec ~250 000 à 365 000 transactions

| Nœud | Transactions | Users | Fraudes |
|---|---|---|---|
| node_1 | 365,148 | 294 | 2,174 (0.60%) |
| node_2 | 334,661 | 245 | 2,040 (0.61%) |
| node_3 | 342,837 | 245 | 1,760 (0.51%) |
| node_4 | 252,448 | 196 | 1,513 (0.60%) |

**Normalisation :** RobustScaler fitté uniquement sur le train + clipping [-5, 5]
→ Résout les NaN causés par Time_Delta_Sec (~31M) et Velocity_Sum_24H (~2.7M)

**Approche B — Split stratifié 80/20 (retenu pour FL) :**
- 80% des users → Train (toutes leurs transactions)
- 20% des users → Test (jamais vus pendant l'entraînement)
- Stratification en 4 groupes : petit/grand × peu/très frauduleux
- Déterministe — pas de seed aléatoire

---

## 5. Résultats d'Entraînement Local

| Run | Modèle | Approche | AUC-ROC | PR-AUC | FA calibrées |
|---|---|---|---|---|---|
| B2 | GRU | B 80/20 | **0.9984** | **0.9489** | — |
| D2 | DPGRU | B 80/20 | 0.9960 | 0.9144 | 170 |
| D3 | DPGRU | A 70/10/20 | 0.9935 | 0.8820 | **86** |

**Modèle retenu pour FL :** DPGRU, Approche B stratifiée 80/20

---

## 6. Résultats FL (3 rounds, 2 nœuds, 3 epochs locaux — test initial)

| Round | Loss | AUC-ROC N1 | Recall N1 | ε max |
|---|---|---|---|---|
| 1 | 4.73 | 0.81 | 0% | 0.148 |
| 2 | 3.48 | 0.91 | 1.3% | 0.210 |
| 3 | 2.80 | **0.94** | 31% | **0.258** |

Convergence confirmée — ε bien inférieur à 1.0 ✅

---

## 7. Paramètres FL

| Paramètre | Valeur | Où configurer |
|---|---|---|
| FL_ROUNDS | 40 | server_run.ps1 |
| FL_LOCAL_EPOCHS | 20 | server_run.ps1 |
| FL_MIN_CLIENTS | 4 | server_run.ps1 |
| FLOWER_PORT | 8090 | server_run.ps1 |
| FL_LR | 0.0005 | client_X_run.ps1 |
| FL_BATCH_SIZE | 128 | client_X_run.ps1 |
| FL_POS_WEIGHT | 167.0 | client_X_run.ps1 |
| DP_NOISE | 1.5 | client_X_run.ps1 |

---

## 8. Arborescence du Projet

```
fraud-detection-pipeline/
├── data/
│   ├── node_1/
│   │   ├── local_data.csv          # Dataset source du nœud
│   │   └── tensors/                # X_batch_*.npy et y_batch_*.npy
│   ├── node_2/ ... node_4/
├── federated/
│   ├── client.py                   # FlowerClient + Opacus DP + évaluation complète
│   ├── server.py                   # FedAvgWithLogging + MetricsLogger + Checkpoint
│   ├── server_run.ps1              # Lancement serveur (rounds, epochs, port)
│   ├── client1_run.ps1             # Lancement nœud 1
│   ├── client2_run.ps1             # Lancement nœud 2
│   ├── client3_run.ps1             # Lancement nœud 3
│   └── client4_run.ps1             # Lancement nœud 4
├── features/
│   ├── enricher.py                 # Vélocité, distance, time_delta
│   └── vectorizer.py               # 12 features num + 14 OHE
├── generator/
│   └── iso_streamer_X.py           # Producteur Kafka ISO 8583
├── ingestion/
│   └── ingestion_X.py              # Consommateur Kafka + pipeline
├── models/
│   └── fraud_rnn.py                # FraudRNN (GRU/DPGRU) + build_model()
├── data/
│   └── dataloader.py               # FraudSequenceDataset + Approche A/B
├── logs/
│   └── fl/
│       ├── fl_metrics.json         # Historique complet des rounds
│       ├── fl_metrics.csv          # Métriques par round (tableur)
│       ├── fl_summary.txt          # Rapport final
│       └── checkpoints/            # Poids globaux par round (.npz)
├── fast_generate_tensors.py        # Génération tenseurs multi-core
├── train_local.py                  # Entraînement local (GRU ou DPGRU)
├── federated_test_Mohamed.ps1      # Lancement FL complet (1 commande)
└── requirements.txt                # Dépendances Python
```

---

## 9. Lancement Rapide

### Prérequis
- Python 3.12.x
- CUDA 12.x (RTX 4060 ou équivalent)
- 8 GB VRAM minimum pour 4 nœuds simultanés

### Installation

```powershell
# 1. PyTorch avec CUDA
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Reste des dépendances
python -m pip install -r requirements.txt
```

### Générer les tenseurs (une fois)

```powershell
python fast_generate_tensors.py --node 1
python fast_generate_tensors.py --node 2
python fast_generate_tensors.py --node 3
python fast_generate_tensors.py --node 4
```

### Entraînement local (validation)

```powershell
# Dans train_local.py : APPROACH = "B", USE_DPGRU = True
python train_local.py
```

### Lancement FL complet

```powershell
.\federated_test_Mohamed.ps1
# Ouvre automatiquement 5 fenêtres : 1 serveur + 4 banques
```

### Arrêt et reprise

```powershell
# Arrêter : fermer les fenêtres PowerShell
# Reprendre : relancer federated_test_Mohamed.ps1
# Le serveur reprend automatiquement depuis le dernier checkpoint

# Repartir de zéro :
Remove-Item -Recurse -Force "logs\fl\checkpoints"
```

---

## 10. Métriques Suivies par Round

Le serveur enregistre automatiquement après chaque FedAvg :

**Évaluation (agrégée sur tous les nœuds) :**
- AUC-ROC, PR-AUC
- Recall, Précision, F1-score
- Fausses alertes (seuil 0.5 et seuil calibré)
- Loss

**Entraînement :**
- Train Loss par nœud
- ε moyen et max (budget Differential Privacy)

**Par nœud** (logs individuels) :
```
logs/node_1/eval_history.json
logs/node_2/eval_history.json
...
```

---

## 11. Justification des Choix Techniques

**Pourquoi DPGRU et pas GRU standard ?**
Le GRU PyTorch utilise des kernels CUDA fusionnés incompatibles avec le calcul per-sample gradients requis par Opacus. Le DPGRU est la réimplémentation non-fusionnée avec une interface identique.

**Pourquoi pos_weight=167 ?**
Ratio exact du déséquilibre : 362 974 légitimes / 2 174 fraudes = 167. Sans correction, le modèle apprend à tout classifier comme légitime pour atteindre 99.4% d'accuracy sans détecter aucune fraude.

**Pourquoi RobustScaler + clip[-5,5] ?**
StandardScaler est sensible aux outliers extrêmes (Time_Delta_Sec = 31M secondes). RobustScaler utilise la médiane et l'IQR. Le clipping à ±5 coupe les valeurs qui font saturer les fonctions tanh/sigmoid du GRU.

**Pourquoi Approche B (split par population) pour le FL ?**
En FL, le modèle global sera déployé sur des clients que chaque nœud n'a jamais vus localement. L'Approche B simule exactement ce scénario — les 20% de clients test n'ont jamais participé au gradient, garantissant une évaluation non biaisée.

**Pourquoi ε ≤ 1.0 ?**
C'est le seuil standard de Differential Privacy forte dans la littérature. Nos résultats montrent ε = 0.258 après 3 rounds — bien en dessous, prouvant que la DP forte est compatible avec de bonnes performances de détection.

---
