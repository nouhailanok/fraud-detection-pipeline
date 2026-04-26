# INTEGRATION GUIDE — Test complet de l'infra FL + MLOps

Ce guide est un mode operatoire de test, pas seulement une description technique.
Objectif: verifier que tout le flux fonctionne de bout en bout, depuis le streaming ISO 8583 jusqu'au serving BentoML, avec monitoring client et serveur.

## 1) Vue globale de l'architecture

```text
Client banque (machine locale)
  Docker: zookeeper + kafka + streamer + prometheus + grafana + pushgateway + node_exporter + cadvisor
  Local process: ingestion_1.py + federated/client.py (+ bentoml_service.py optionnel)

Serveur AWS Flower
  Flower server (FedAvg)
  model_registry (choix best model)
  Docker monitoring serveur: prometheus + grafana + flower_state_exporter + node_exporter + cadvisor
```

## 2) Role de chaque fichier (important pour comprendre les tests)

### 2.1 Orchestration

- `docker-compose.local.yml`
  - lance l'infra Docker du noeud client
  - ne lance PAS ingestion/client FL (ils tournent en local pour CUDA)
  - expose Prometheus local et Grafana local

- `docker-compose.server.yml`
  - lance le monitoring cote serveur Flower
  - scrape la sante Flower + artefacts FL + model registry

### 2.2 Monitoring client

- `monitoring/prometheus.yml`
  - configure les jobs scrape cote client
  - scrape `streamer` (container) + `ingestion_local` + `federated_client_local` + `bentoml_service`

- `monitoring/alert_rules.yml`
  - regles d'alerte client
  - detecte process down, rounds FL bloques, baisse F1/AUC, debit ingestion faible

- `monitoring/grafana/dashboards/node-dashboard.json`
  - dashboard client "FL Client Node Observability"

### 2.3 Monitoring serveur

- `server_monitoring/prometheus.yml`
  - scrape `flower_state_exporter`, endpoint metrics Flower, ressources host

- `server_monitoring/alert_rules.yml`
  - alertes serveur: exporter down, stagnation rounds, suspects behavioral, absence de production model

- `server_monitoring/flower_state_exporter.py`
  - lit les fichiers `logs/runs/*/fl_metrics.json`, `behavioral_analysis.json`, `model_registry/registry.json`
  - convertit ces informations en metriques Prometheus exploitables dans Grafana

- `server_monitoring/grafana/dashboards/flower-server-dashboard.json`
  - dashboard serveur "Flower Server Observability"

### 2.4 Model lifecycle (registry + serving)

- `model_registry/registry.py`
  - scan des runs
  - calcul score composite/trust
  - promotion du meilleur modele vers `model_registry/best_model/`

- `serving/sync_best_model.py`
  - copie `model.npz` + `metadata.json` depuis le serveur vers le client via SSH/SCP

- `serving/save_model.py`
  - prend le `best_model` local
  - reconstruit le modele PyTorch
  - sauvegarde dans le store BentoML

- `serving/bentoml_service.py`
  - expose l'API inference
  - endpoint prediction: `/predict`
  - endpoint metriques internes JSON: `/service_metrics`
  - endpoint Prometheus Bento: `/metrics`

### 2.5 Telemetrie applicative

- `scripts/metrics_exporter.py`
  - librairie Python commune pour exposer des metriques Prometheus depuis streamer/ingestion/client
  - c'est elle qui alimente les panels Grafana et les alertes Prometheus

## 3) Explication claire de metrics_exporter.py

### 3.1 Comment ca marche

Le module `MetricsExporter` fait 4 choses:

1. cree un registre Prometheus local (CollectorRegistry)
2. declare des metriques typées:
   - Counter: compte cumulatif (ex: transactions consommees)
   - Gauge: valeur instantanee (ex: F1 courante)
   - Histogram: distribution de latence (ex: duree fit/eval)
3. attache des labels standards a toutes les metriques:
   - `node`, `client_id`, `service`
4. demarre un mini serveur HTTP `/metrics` via `start_http_server(port)`

Prometheus ne "recoit" pas les metriques automatiquement:
- c'est Prometheus qui vient lire (`scrape`) le endpoint `/metrics`
- donc il faut que:
  - le process soit lance
  - le port soit correct
  - le job soit declare dans `prometheus.yml`

### 3.2 Ports utilises (dans votre setup)

- streamer: 8002
- ingestion local: 8003
- client FL local: 8001

### 3.3 Mappage metriques -> code

- streamer:
  - `inc_tx_generated()`
  - `inc_tx_kafka_sent()`
  - `inc_streamer_error()`

- ingestion:
  - `inc_tx_consumed()`
  - `inc_tx_validated()`
  - `inc_tx_rejected()`
  - `inc_batches_saved()`
  - `inc_ingestion_error()`

- client FL:
  - `inc_fl_round()`
  - `set_fl_loss()`
  - `observe_fl_duration()`
  - `set_eval_f1()`, `set_eval_auc()`, `set_eval_precision()`, `set_eval_recall()`
  - `observe_eval_duration()`

### 3.4 Regle pratique

A chaque event metier important, appelez la methode metrics correspondante.
Si vous oubliez d'incrementer/mettre a jour une metrique dans le code, Grafana restera vide pour ce signal.

## 4) Pre-requis avant test

- Docker Engine + Docker Compose
- Python env local avec dependances projet
- certificats TLS disponibles dans `security/certs`
- `.env` cree depuis `.env.example`
- acces SSH au serveur AWS Flower (pour sync model)

## 5) Procedure de test complete (pas a pas)

## Etape A - Verification base locale

1. preparer l'environnement

```bash
cp .env.example .env
```

Sous Windows PowerShell, si `cp` ne marche pas:

```powershell
Copy-Item .env.example .env
```

2. demarrer la stack docker locale

```bash
docker compose -f docker-compose.local.yml --env-file .env up -d
```

3. verifier les containers

```bash
docker compose -f docker-compose.local.yml ps
```

Resultat attendu:
- zookeeper, kafka, streamer, prometheus, grafana, pushgateway, node_exporter, cadvisor en `running`

4. verifier endpoints monitoring docker

- Prometheus: `http://localhost:9090/-/healthy`
- Grafana: `http://localhost:3000/login`
- Streamer metrics: `http://localhost:8002/metrics`

## Etape B - Lancer process locaux (hors Docker)

5. lancer ingestion localement

```powershell
$env:METRICS_ENABLED="true"
$env:METRICS_PORT="8003"
python ingestion/ingestion_1.py
```

6. lancer client FL localement

```powershell
$env:METRICS_ENABLED="true"
$env:METRICS_PORT="8001"
python federated/client.py
```

7. optionnel: lancer BentoML local

```bash
bentoml serve serving/bentoml_service.py:svc --port 3001 --reload
```

8. verifier endpoints metrics des process locaux

- `http://localhost:8003/metrics`
- `http://localhost:8001/metrics`
- `http://localhost:3001/metrics` (si Bento lance)

Resultat attendu:
- texte Prometheus visible (lignes `# HELP`, `# TYPE`, puis valeurs)

## Etape C - Verification Prometheus/Grafana cote client

9. verifier targets Prometheus

- ouvrir `http://localhost:9090/targets`
- verifier que ces jobs sont `UP`:
  - `streamer`
  - `ingestion_local`
  - `federated_client_local`
  - `bentoml_service` (si Bento lance)

10. tester quelques requetes PromQL

- `rate(streamer_transactions_generated_total[1m])`
- `rate(ingestion_transactions_consumed_total[1m])`
- `fl_evaluation_f1_score`
- `increase(fl_round_total[30m])`

11. verifier dashboard Grafana client

- ouvrir dashboard `FL Client Node Observability`
- confirmer:
  - status panels UP
  - throughput non nul
  - metriques FL evoluent au fil des rounds

## Etape D - Monitoring serveur Flower

12. sur le serveur AWS, lancer stack monitoring serveur

```bash
docker compose -f docker-compose.server.yml --env-file .env up -d
```

13. verifier endpoints serveur

- `http://localhost:9190/-/healthy`
- `http://localhost:9108/metrics` (flower_state_exporter)
- `http://localhost:3300/login`

14. verifier targets serveur dans Prometheus

- ouvrir `http://localhost:9190/targets`
- jobs attendus UP:
  - `flower_state_exporter`
  - `node_exporter`
  - `cadvisor`
  - `flower_server` (si endpoint metrics Flower expose)

15. verifier dashboard serveur

- ouvrir `Flower Server Observability`
- confirmer presence des signaux:
  - `fl_server_last_round`
  - `fl_server_last_f1`
  - `fl_server_behavioral_suspects_last_round`
  - `fl_server_registry_production_composite_score`

## Etape E - Test model registry + sync + serving

16. promouvoir un run sur serveur

```bash
python model_registry/registry.py --scan logs/runs/
python model_registry/registry.py --propose
python model_registry/registry.py --promote <RUN_ID> --yes
python model_registry/registry.py --status
```

17. sync best_model vers client

```bash
python serving/sync_best_model.py \
  --ssh-user <USER> \
  --ssh-host <FLOWER_HOST> \
  --remote-best-model /path/server/model_registry/best_model \
  --local-best-model model_registry/best_model \
  --save-to-bentoml
```

18. demarrer BentoML (si pas deja fait)

```bash
bentoml serve serving/bentoml_service.py:svc --port 3001
```

19. tester inference

```bash
curl -X POST http://localhost:3001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pan_id":"test_pan_001",
    "amount":250.0,
    "merchant":"SHOP",
    "lat":48.8566,
    "long":2.3522,
    "unix_time":1714051200
  }'
```

Resultat attendu:
- JSON avec `prediction`, `probability`, `latency_ms`, `model_version`

20. verifier metriques serving

- endpoint JSON: `POST http://localhost:3001/service_metrics`
- endpoint Prometheus: `GET http://localhost:3001/metrics`

## 6) Scenarios de test recommandes

- test disponibilite:
  - arretez ingestion local, verifier alerte `IngestionLocalDown`
- test FL stop:
  - stop client FL, verifier `FederatedClientLocalDown`
- test throughput:
  - baisser `STREAM_RATE_TPS`, verifier impact sur `rate(...)`
- test behavioral:
  - lancer run avec noud suspect simule et verifier signaux serveur

## 7) Troubleshooting rapide

- targets `DOWN` dans Prometheus:
  - verifier process lance
  - verifier port correct
  - verifier firewall local

- pas de metriques FL dans Grafana:
  - `metrics.start_server()` absent ou non execute
  - fonctions `set_eval_*`/`inc_fl_round` non appelees dans le code

- dashboard serveur vide:
  - `logs/runs` non monte/non accessible pour `flower_state_exporter`
  - fichiers `fl_metrics.json` ou `behavioral_analysis.json` absents

## 8) Checklist finale (go/no-go)

- client docker infra UP
- ingestion local + client FL local + metrics UP
- Prometheus client targets UP
- Grafana client affiche throughput + FL metrics
- monitoring serveur UP
- registry promotion fonctionne
- sync best_model fonctionne
- endpoint `/predict` fonctionne
- metriques serving visibles dans Prometheus
