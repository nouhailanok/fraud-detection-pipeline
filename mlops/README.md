# MLOps for Federated Learning Fraud Detection

This repository now supports a split architecture:

- client node: Kafka + streamer + monitoring in Docker
- client node: ingestion + FL training + Bento serving as local processes (outside Docker, CUDA-friendly)
- server side (AWS Flower): dedicated monitoring stack for FedAvg, behavioral analysis, and model registry status

## 1) Final workflow

```text
iso_streamer_x.py (Docker)
  -> Kafka local (Docker)
  -> ingestion_x.py (local process, GPU/CUDA friendly)
  -> client.py (local FL process)
  -> Flower Server AWS (FedAvg)
  -> model_registry on server promotes best_model
  -> sync_best_model.py copies best_model by SSH to client
  -> save_model.py stores model in BentoML
  -> bentoml_service.py serves /predict for new transactions
```

## 2) What changed

### Local node stack (`docker-compose.local.yml`)

- removed `ingestion` and `federated_client` containers
- kept `zookeeper`, `kafka`, `streamer`, `prometheus`, `pushgateway`, `grafana`, `node_exporter`, `cadvisor`
- `prometheus` now scrapes:
  - `streamer` (container)
  - `ingestion_local` on `host.docker.internal:8003`
  - `federated_client_local` on `host.docker.internal:8001`
  - `bentoml_service` on `host.docker.internal:3001`

### Client monitoring (`monitoring/`)

- updated `monitoring/prometheus.yml` for host-based ingestion/training/serving
- updated `monitoring/alert_rules.yml` with local-process aware alerts
- replaced client dashboard with a cleaner one aligned to new jobs

### Server monitoring (`docker-compose.server.yml` + `server_monitoring/`)

Created:

- `docker-compose.server.yml`
- `server_monitoring/prometheus.yml`
- `server_monitoring/alert_rules.yml`
- `server_monitoring/flower_state_exporter.py`
- `server_monitoring/grafana/provisioning/...`
- `server_monitoring/grafana/dashboards/flower-server-dashboard.json`

This stack gives visibility on:

- Flower server metrics endpoint status
- FL round history (latest round, rounds total, latest F1/recall/precision/epsilon)
- behavioral analysis (suspects, attack counts, blacklist)
- model registry production state and composite/trust/privacy metrics
- server host resources (CPU/RAM)

### Registry/serving compatibility

- `model_registry/registry.py` now writes promotion metadata in a backward + forward compatible format
  - flat keys: `f1`, `recall`, `epsilon_final`, ...
  - nested keys: `metrics.{f1, recall, precision, epsilon_final}`
  - shared keys: `run_name`, `run_id`, `architecture`, `promoted_at`
- `serving/save_model.py` now supports both metadata formats
- `serving/save_model.py` now resolves `best_model` path robustly for this repo
- `serving/bentoml_service.py`:
  - fixed project root resolution
  - switched JSON metrics endpoint to `/service_metrics`
  - kept Prometheus `/metrics` endpoint free for scraping
  - made model metadata reads robust for old/new schemas
- new `serving/sync_best_model.py` to copy `model.npz` and `metadata.json` from server via SSH/SCP

## 3) Run client node (local bank machine)

### 3.1 Start Docker infra + monitoring

```bash
docker compose -f docker-compose.local.yml --env-file .env up -d
```

### 3.2 Run ingestion locally (outside Docker)

Example (adapt to your venv/conda and paths):

```bash
set METRICS_ENABLED=true
set METRICS_PORT=8003
python ingestion/ingestion_1.py
```

### 3.3 Run FL client locally (outside Docker)

```bash
set METRICS_ENABLED=true
set METRICS_PORT=8001
python federated/client.py
```

### 3.4 Optional: run Bento service locally (outside Docker)

```bash
bentoml serve serving/bentoml_service.py:svc --port 3001 --reload
```

## 4) Run Flower server monitoring (AWS side)

On the server host:

```bash
docker compose -f docker-compose.server.yml --env-file .env up -d
```

Default ports:

- Prometheus server: `http://localhost:9190`
- Grafana server: `http://localhost:3300`
- Flower state exporter: `http://localhost:9108/metrics`

If Flower exposes metrics on another port than `9095`, update `server_monitoring/prometheus.yml` target `flower_server`.

## 5) Promote best model on server

```bash
python model_registry/registry.py --scan logs/runs/
python model_registry/registry.py --propose
python model_registry/registry.py --promote <RUN_ID> --yes
python model_registry/registry.py --status
```

The promoted artifacts are written to:

- `model_registry/best_model/model.npz`
- `model_registry/best_model/metadata.json`

## 6) Sync best model to client and serve

On the client node:

```bash
python serving/sync_best_model.py \
  --ssh-user ubuntu \
  --ssh-host <flower-aws-host> \
  --remote-best-model /path/on/server/model_registry/best_model \
  --local-best-model model_registry/best_model \
  --save-to-bentoml
```

Then start the API:

```bash
bentoml serve serving/bentoml_service.py:svc --port 3001
```

Prediction endpoint:

- `POST /predict`

Service health endpoints:

- `POST /health`
- `POST /service_metrics`
- `POST /model_info`
- `GET /metrics` (Prometheus exposition)

## 7) Monitoring views

### Client Grafana (`http://localhost:3000`)

Dashboard: `FL Client Node Observability`

Main signals:

- streamer / ingestion local / fl client local / bentoml up
- generated vs consumed tx throughput
- FL rounds per hour
- local eval metrics (F1, precision, recall)
- pipeline errors and host resource usage

### Server Grafana (`http://localhost:3300`)

Dashboard: `Flower Server Observability`

Main signals:

- Flower metrics endpoint availability
- latest FL round and quality metrics
- epsilon/trust/composite score trends
- behavioral suspects and attack counts
- production model status from registry

## 8) Notes and best practices

- Keep ingestion and FL training outside Docker when CUDA/GPU drivers are a concern.
- Keep only stable infra in Docker on client machines.
- Keep model promotion centralized on server (`model_registry`).
- Automate `sync_best_model.py` with a scheduler (Task Scheduler/Cron) if you want periodic pull.
- If needed, add Alertmanager later for Slack/Email paging.

## 9) Quick health checklist

Client node:

- `docker compose -f docker-compose.local.yml ps`
- `curl http://localhost:9090/-/healthy`
- `curl http://localhost:8001/metrics`
- `curl http://localhost:8003/metrics`
- `curl http://localhost:3001/metrics`

Server node:

- `docker compose -f docker-compose.server.yml ps`
- `curl http://localhost:9190/-/healthy`
- `curl http://localhost:9108/metrics`
- `curl http://localhost:3300/api/health`
