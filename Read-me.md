# Fraud Detection Pipeline — Federated Learning with Differential Privacy

> **Run promu :** `2026-04-23_01-00-54_imane` | **Modèle principal :** DPGRU 3 couches (variante A4) | **Branche :** `main`

---

## Table des Matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du Projet](#2-architecture-du-projet)
3. [Modèle Principal — DPGRU 3 Couches](#3-modèle-principal--dpgru-3-couches)
4. [Pipeline de Données](#4-pipeline-de-données)
5. [Apprentissage Fédéré (Flower)](#5-apprentissage-fédéré-flower)
6. [Confidentialité Différentielle (Opacus)](#6-confidentialité-différentielle-opacus)
7. [Analyse Comportementale (Behavioral Analysis)](#7-analyse-comportementale-behavioral-analysis)
8. [Résultats du Run Promu](#8-résultats-du-run-promu)
9. [Déploiement et Infrastructure](#9-déploiement-et-infrastructure)
10. [Problèmes Connus / À Corriger](#10-problèmes-connus--à-corriger)
11. [Justification des Choix Techniques](#11-justification-des-choix-techniques)
12. [Références Internes](#12-références-internes)

---

## 1. Vue d'Ensemble

Ce projet implémente un système de détection de fraude bancaire basé sur :

- **Federated Learning** (framework Flower, FedAvg + FedProx μ=0.01) — 4 nœuds bancaires indépendants collaborent sans partager leurs données brutes.
- **Differential Privacy** (Opacus, per-sample gradients, ε budget ≤ 1.0) — garantie formelle de confidentialité des transactions.
- **DPGRU 3 couches** — modèle principal, séquences de 5 transactions, 26 features.
- **Behavioral Analysis** (Isolation Forest) — détection des nœuds malveillants / empoisonneurs.
- **Approche B** — split stratifié par population (80/20 unseen users) pour une évaluation réaliste en FL.

### Performances du Run Promu (`2026-04-23_01-00-54_imane`)

| Métrique | Valeur (Best Round 30) |
|---|---|
| Accuracy | 0.9977 |
| Recall | 0.6252 |
| Precision | 0.8035 |
| F1-score | **0.7029** |
| Loss | 2.3911 |
| max ε (DP) | **0.9542** ✅ |
| Durée totale | 36 093 s (~10 h) |

---

## 2. Architecture du Projet

```
fraud-detection-pipeline/
├── models/
│   └── fraud_rnn.py            # Modèle DPGRU 3 couches — source de vérité
├── data/
│   └── dataloader.py           # FraudSequenceDataset + Approche A / Approche B
├── federated/
│   ├── client.py               # FlowerClient + Opacus DP + FedProx
│   ├── server.py               # FedAvgWithLogging + BehavioralAnalyzer + MetricsLogger
│   ├── behavioral_analysis.py  # Isolation Forest sur gradients
│   ├── poisoning_client.py     # Client malveillant (tests BA)
│   ├── run_all_saad.ps1        # Script de lancement FL complet
│   └── run_all_saad_poison_test.ps1
├── train_local.py              # Entraînement local (désynchronisé — voir §10)
├── train_central.py            # Baseline centralisée (nouveau)
├── mlops/
│   ├── serving/
│   │   ├── bentoml_service.py  # Service BentoML (port 3001)
│   │   └── save_model.py       # Export checkpoint → BentoML
│   └── monitoring/             # Prometheus + Grafana
├── api_gateway/
│   └── fastapi_app.py          # Gateway FastAPI (port 8000)
├── infrastructure/
│   └── terraform/              # AWS IaC
├── security/
│   ├── MTLS_SETUP.md
│   ├── README_SECURITY.md
│   └── compliance_report.md
├── logs/
│   └── runs/
│       └── 2026-04-23_01-00-54_imane/   # Run promu
│           ├── fl_summary.txt
│           ├── fl_metrics.json
│           ├── fl_metrics.csv           # Git LFS pointer
│           ├── behavioral_analysis.json
│           ├── server_run_env.txt
│           ├── run_all_saad.ps1         # Snapshot des params du run
│           └── checkpoints/
│               └── global_model_best.npz
├── docker-compose.yml          # En chantier — voir §10
├── requirements.txt            # ATTENTION : packages critiques commentés — voir §10
└── Read-me.md
```

---

## 3. Modèle Principal — DPGRU 3 Couches

**Fichier :** `models/fraud_rnn.py`
**Variante :** A4 (configuration finale retenue)

### Configuration

| Paramètre | Valeur |
|---|---|
| Architecture | DPGRU (GRU non-fusionné compatible Opacus) |
| `NUM_LAYERS` | **3** |
| `HIDDEN_SIZE` | 128 |
| `DROPOUT` | **0.3** |
| `INPUT_DIM` | 26 features |
| `SEQ_LEN` | 5 transactions |
| `USE_DPGRU` | True |

### Factory Function

```python
from models.fraud_rnn import build_model

model = build_model(
    input_dim=26,
    hidden_size=128,
    num_layers=3,
    dropout_rate=0.3,
    use_dpgru=True
)
```

`build_model()` est la **seule** façon canonique d'instancier le modèle. Elle est utilisée par `federated/client.py` et `train_central.py`.

> **Note :** `train_local.py` instancie encore `FraudRNN(num_layers=2, dropout_rate=0.2)` en dur — bug connu, voir §10.

### Pourquoi DPGRU ?

Le GRU PyTorch standard utilise des kernels CUDA fusionnés incompatibles avec le calcul de per-sample gradients requis par Opacus. Le DPGRU est la réimplémentation non-fusionnée avec une interface identique. En entraînement centralisé sans DP (`train_central.py`), on revient à `nn.GRU` standard (plus rapide).

---

## 4. Pipeline de Données

**Fichier :** `data/dataloader.py`

### Dataset

```python
FraudSequenceDataset(data_path, seq_len=5, scaler=RobustScaler(), clip_val=5.0)
```

- **SEQ_LEN = 5** transactions glissantes par séquence
- **RobustScaler** (médiane/IQR) + clip à ±5 σ
- **26 features** après prétraitement

### Split Utilisé : Approche B (FL Production)

Split **stratifié par population** : 80% clients train, 20% clients test (unseen users). Déterministe (seed fixe).

```python
from data.dataloader import get_split_dataloaders  # alias → Approche B

train_loader, val_loader, test_loader = get_split_dataloaders(
    data_path="data/node_1/",
    seq_len=5,
    batch_size=128
)
```

**Justification :** En FL, le modèle global sera déployé sur des clients inconnus de chaque nœud. L'Approche B simule exactement ce scénario — les 20 % de clients test n'ont jamais participé à un gradient local.

### Approche A (référence)

Split intra-utilisateur temporel 70/10/20 — utilisé pour comparaison dans les expériences ablation.

---

## 5. Apprentissage Fédéré (Flower)

### Configuration du Run Promu

| Paramètre | Valeur |
|---|---|
| `FL_ROUNDS` | 30 |
| `FL_LOCAL_EPOCHS` | 3 |
| `FL_MIN_CLIENTS` | 4 |
| `FLOWER_PORT` | 8080 |
| `FL_PATIENCE` (early stop) | 5 |
| `FL_LR` | 0.0005 |
| `FL_BATCH_SIZE` | 128 |
| `DP_NOISE` | 1.20 |

### Stratégie : FedAvg + FedProx

**Serveur :** `federated/server.py`

- `FedAvgWithLogging` — agrégation pondérée par nombre d'exemples
- Terme proximal FedProx : `(μ/2) · ‖w_local − w_global‖²` avec μ = 0.01
- Early stopping sur F1 (patience = 5)
- Checkpoint automatique du meilleur modèle global
- `MetricsLogger` — enregistre `fl_metrics.json` / `fl_metrics.csv` par round
- `BehavioralAnalyzer` — analyse des gradients de chaque nœud (actif dès round 3)

**Client :** `federated/client.py`

- `FlowerClient` avec Opacus DP
- `pos_weight` par nœud : 167 / 163 / 193.8 / 165.8 (ratio déséquilibre local)
- Arrêt automatique si `max_epsilon > epsilon_target`

### Lancement FL

```powershell
# Lancement complet (1 serveur + 4 banques)
.\federated\run_all_saad.ps1

# Lancement avec test d'empoisonnement
.\federated\run_all_saad_poison_test.ps1
```

### Reprise depuis Checkpoint

```powershell
$env:FL_RESUME_FROM = "logs\runs\2026-04-23_01-00-54_imane\checkpoints"
.\federated\run_all_saad.ps1
```

---

## 6. Confidentialité Différentielle (Opacus)

### Paramètres

| Paramètre | Valeur |
|---|---|
| Mécanisme | Gaussian (Opacus GradSampleModule) |
| `noise_multiplier` | 1.20 |
| `max_grad_norm` | 1.0 (clipping) |
| `epsilon_target` | 1.0 |
| `delta` | 1e-5 |

### Budget ε — Run Promu

```
max ε (round 30) = 0.9542  ✅  (< 1.0)
```

L'entraînement s'arrête automatiquement si ε dépasse `epsilon_target`. Le budget de 30 rounds × 3 epochs locales a été calibré pour rester sous ε = 1.0 avec `noise_multiplier = 1.20`.

### Interprétation

ε < 1.0 garantit une **Differential Privacy forte** (cf. Dwork & Roth). Cela signifie qu'un adversaire observant le modèle global ne peut pas distinguer avec confiance si une transaction individuelle a participé à l'entraînement.

---

## 7. Analyse Comportementale (Behavioral Analysis)

**Fichier :** `federated/behavioral_analysis.py`

### Principe

Après chaque round FL (dès round 3), le serveur extrait un vecteur de 6 features pour chaque nœud participant :

| Feature | Description |
|---|---|
| `norm_L2` | Norme L2 de la mise à jour de gradient |
| `norm_L1` | Norme L1 de la mise à jour |
| `cos_sim` | Similarité cosinus avec le modèle global |
| `var_delta` | Variance des deltas de poids |
| `train_loss` | Loss d'entraînement local |
| `epsilon` | Budget DP consommé |

### Algorithme

**Isolation Forest** avec :
- `contamination = 1 / FL_MIN_CLIENTS` = 0.25 (1 nœud sur 4)
- Décisions : `NORMAL` / `EXCLUDE` (round) / `BLACKLIST` (permanent)

### Tests d'Empoisonnement

**Fichier :** `federated/poisoning_client.py`

Types d'attaques simulées :
- `FREE_RIDER` — envoie le modèle global sans entraînement local
- `SIGN_FLIP` — inverse le signe des gradients
- `SCALE` — amplifie les mises à jour (×10)
- `NOISE` — ajoute du bruit gaussien massif
- `BYZANTINE` — mises à jour aléatoires
- `NORMAL` — nœud légitime (baseline)

Résultats : `logs/runs/<run>/behavioral_analysis.json`

---

## 8. Résultats du Run Promu

**Run :** `logs/runs/2026-04-23_01-00-54_imane/`

### Configuration d'Exécution

```
FL_ROUNDS=30, FL_LOCAL_EPOCHS=3, FL_MIN_CLIENTS=4
FLOWER_PORT=8080, FL_PATIENCE=5
DP_NOISE=1.20, FL_LR=0.0005, FL_BATCH_SIZE=128
pos_weight : node_1=167, node_2=163, node_3=193.8, node_4=165.8
```

### Meilleur Round (Round 30)

```
Accuracy   : 0.9977
Recall     : 0.6252
Precision  : 0.8035
F1-score   : 0.7029   ← critère de sélection
Loss       : 2.3911
max ε      : 0.9542   ✅ (< 1.0)
Durée totale : 36 093.5 s (~10 h)
```

### Checkpoint Sauvegardé

```
logs/runs/2026-04-23_01-00-54_imane/checkpoints/global_model_best.npz
```

> **Note sur fl_summary.txt :** Le fichier indique « 60 / 30 rounds ». C'est un artefact du double-log historique (chaque round créait 1 entrée fit + 1 entrée eval). Le fix `_push_temp_entry` / `_complete_entry` est en place dans la version actuelle de `server.py`, mais le run promu a été produit avant ce fix.

---

## 9. Déploiement et Infrastructure

### Stack Technologique

| Composant | Technologie |
|---|---|
| Serving | BentoML (port 3001) |
| API Gateway | FastAPI (port 8000) |
| Monitoring | Prometheus + Grafana |
| Infrastructure | Terraform AWS |
| Sécurité transport | mTLS (Kafka + Flower) |
| Conteneurisation | Docker Compose (en chantier — voir §10) |

### Déploiement BentoML + API Gateway

```powershell
# Sauver le meilleur modèle FL en bento
python mlops/serving/save_model.py --checkpoint logs/runs/2026-04-23_01-00-54_imane/checkpoints/global_model_best.npz

# Lancer le service BentoML (port 3001)
bentoml serve mlops/serving/bentoml_service.py:svc

# Lancer l'API Gateway (port 8000)
uvicorn api_gateway.fastapi_app:app --port 8000
```

### Lancement FL + Test d'Empoisonnement

```powershell
.\federated_test_Mohamed.ps1
# Ouvre automatiquement 5 fenêtres : 1 serveur + 4 banques

.\federated\run_all_saad_poison_test.ps1
# Un des nœuds est remplacé par poisoning_client.py
# Vérifier la détection : logs/runs/<run>/behavioral_analysis.json
```

### Métriques par Round (MetricsLogger)

Le serveur enregistre automatiquement après chaque round dans `fl_metrics.json` / `fl_metrics.csv` :

**Évaluation (agrégée pondérée) :**
- Accuracy, Loss, Recall, Precision, F1
- `n_clients_eval`, `n_examples_eval`

**Entraînement (agrégé) :**
- `train_loss` moyen
- `avg_epsilon`, `max_epsilon`, `epsilon_status` (OK / WARNING)

**Behavioral Analysis :**
- `ba_active`, `ba_suspects`

**Par nœud :**
```
logs/node_<id>/eval_history.json   # Métriques détaillées + matrices de confusion
```

---

## 10. Problèmes Connus / À Corriger

Audit du code effectué le 02/05/2026. Les points suivants méritent attention avant la rédaction du rapport final :

### 1. `requirements.txt` — Packages Critiques Commentés

`torch`, `flwr`, `opacus`, `scikit-learn` sont **commentés**. Un nouvel utilisateur qui lance `pip install -r requirements.txt` n'obtient pas un environnement fonctionnel.

**Action :** Décommenter ou ajouter ces lignes.

### 2. `train_local.py` — Désynchronisé avec `fraud_rnn.py`

Le fichier instancie `FraudRNN(num_layers=2, dropout_rate=0.2)` en dur, alors que la config canonique du modèle (variante A4 / run promu) est `num_layers=3, dropout=0.3`.

**Action :** Remplacer par :
```python
from models.fraud_rnn import build_model
model = build_model(input_dim=26, hidden_size=128, num_layers=3, dropout_rate=0.3)
```

### 3. `train_local.py` — Label Approche B Trompeur

Affiche `"Population Unseen Users 80/20"` mais appelle `get_dataloaders_approach_B(train_ratio=0.70, val_ratio=0.10)` — c'est en réalité le mode B3 (70/10/20).

**Action :** Harmoniser le label ou les ratios.

### 4. `federated/client.py` — Code Dupliqué

Les blocs `self.epsilon_target / self.dp_exhausted`, `avg_loss = 0.0`, `loss = self.criterion(...)`, le print `ε = ...` et le contrôle `dp_exhausted` apparaissent **deux fois** consécutivement. Sans effet de bord (réécriture identique) mais source de confusion.

**Action :** Supprimer les blocs en double.

### 5. `fraud_rnn.py` — Header Trompeur

Le docstring annonce `seq_len=10` (pour le test `__main__`) mais tout le pipeline utilise `SEQ_LEN=5`. Le modèle est agnostique au seq_len, mais le commentaire prête à confusion.

**Action :** Corriger le commentaire → `seq_len=5`.

### 6. `docker-compose.yml` — Désynchronisé

- Image `python:3.9-slim` alors que le projet exige Python 3.11+.
- `ingestion_processor_2/3/4` ont `command: ["python", "federated/client.py"]` qui **override** le CMD du Dockerfile inline — un seul processeur d'ingestion (node_1) tourne réellement.
- `FL_ROUNDS=50` dans le compose vs 30 dans le run promu.
- Workflow actuel = scripts PowerShell locaux ; le compose est en chantier.

**Action :** Mettre à jour ou documenter explicitement comme WIP.

### 7. `dataloader.py` — Code Mort

`_stratified_user_split()` n'est plus appelée (remplacée par la logique inline dans `get_dataloaders_approach_B`).

**Action :** Supprimer ou marquer `@deprecated`.

### 8. `fl_summary.txt` — "60 / 30 rounds"

Le run promu indique « Rounds effectués 60 / 30 ». Cause : double-log historique (1 entrée fit + 1 entrée eval par round). Le fix est en place dans la version actuelle de `server.py` mais le run a été produit avant ce fix.

**Impact :** Cosmétique uniquement. Les métriques sont correctes.

### 9. `fl_metrics.csv` — Git LFS Pointer

Le fichier CSV du run promu est un pointer Git LFS (3 160 octets). Le contenu réel est dans `fl_metrics.json`.

**Action :** Vérifier `.gitattributes` si on veut versionner les CSV directement.

---

## 11. Justification des Choix Techniques

### Pourquoi DPGRU et pas GRU standard ?

Le GRU PyTorch utilise des kernels CUDA fusionnés incompatibles avec le calcul de per-sample gradients requis par Opacus. Le DPGRU est la réimplémentation non-fusionnée avec une interface identique. En centralisé sans DP (`train_central.py`), on revient à `nn.GRU` (plus rapide).

### Pourquoi 3 couches et dropout 0.3 ?

Tests internes (variantes A1-A4) ont montré qu'une 3ᵉ couche améliore le rappel sur les patterns frauduleux multi-jours, à condition d'augmenter le dropout (0.2 → 0.3) pour compenser le risque de sur-apprentissage.

### Pourquoi `pos_weight ≈ 167` ?

Ratio exact du déséquilibre node_1 : 362 974 légitimes / 2 174 fraudes ≈ 167. Chaque nœud utilise son propre ratio (167 / 163 / 193.8 / 165.8). Sans correction, le modèle apprend tout-légitime (accuracy 99.4 %, recall 0 %).

### Pourquoi RobustScaler + clip[-5, 5] ?

StandardScaler sature sur `Time_Delta_Sec` (~31 M secondes). RobustScaler (médiane/IQR) résiste aux outliers ; le clip à ±5 σ coupe les valeurs qui font diverger les fonctions tanh/sigmoid du GRU.

### Pourquoi Approche B (split par population) pour le FL ?

En production, le modèle global sera évalué sur des clients **inconnus** de chaque nœud. L'Approche B simule exactement ce scénario : les 20 % de clients test n'ont participé à aucun gradient — évaluation non biaisée.

### Pourquoi ε ≤ 1.0 ?

Seuil standard de Differential Privacy forte (cf. Dwork & Roth). Notre run principal atteint max ε = 0.9542 après 30 rounds — limite respectée, performances exploitables.

### Pourquoi FedProx (μ=0.01) en plus de FedAvg ?

Le terme proximal `(μ/2)·‖w_local − w_global‖²` stabilise l'apprentissage en présence d'**hétérogénéité statistique** (les 4 nœuds ont des distributions de fraude différentes). μ faible (0.01) suffit à éviter la divergence sans étouffer l'apprentissage local.

### Pourquoi Behavioral Analysis avec Isolation Forest ?

Le FL est intrinsèquement vulnérable aux attaques par empoisonnement. L'Isolation Forest est non supervisé (pas besoin d'attaque labellisée) et fonctionne sur peu d'échantillons (4 nœuds × 1 vecteur de features par round). Les seuils ont été calibrés via `poisoning_client.py` sur les 5 types d'attaques.

---

## 12. Références Internes

- **Run principal promu :** `logs/runs/2026-04-23_01-00-54_imane/`
- **Modèle principal :** `models/fraud_rnn.py` (variante A4, DPGRU 3 couches)
- **Sécurité mTLS :** `security/MTLS_SETUP.md` et `security/README_SECURITY.md`
- **Conformité :** `security/compliance_report.md`
- **Baseline centralisée :** `train_central.py` (GRU sans DP, TRAIN_RATIO=0.90)
- **Tests d'attaque :** `federated/poisoning_client.py`
