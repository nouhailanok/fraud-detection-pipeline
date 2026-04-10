## Plan: Hybrid FL Deployment (4 on-prem + AWS aggregator)

Deploy one Flower aggregator on AWS EC2 and four Flower clients on physical machines with mTLS enabled, while aligning current code/config mismatches that would otherwise break startup. The recommended approach is: (1) pre-flight fixes/decisions, (2) certificate and network hardening, (3) deterministic data placement per node, (4) staged launch and validation.

**Steps**
1. Phase 0 - Scope and architecture lock: confirm this run mode is federated-training only (Flower traffic between clients and AWS server), and whether Kafka ingestion remains local/preprocessing-only.
2. Phase 1 - Pre-flight blockers (must be resolved first):
   - Align server port env naming because server reads FLOWER_PORT while compose/server env currently uses FLOWER_SERVER_PORT.
   - Fix client dataset path checks and loading path format: client uses string wildcards with .exists(), and loader expects concrete existing file paths.
   - Choose dataset format contract for clients: either single train/test files per node or explicit aggregation of batch files into canonical train/test files.
3. Phase 2 - PKI/mTLS preparation:
   - Regenerate Flower server cert with SAN including the real AWS endpoint (public DNS and/or Elastic IP), not only localhost/internal names.
   - Keep one CA, issue one server cert for aggregator and four unique client cert/key pairs for nodes.
   - Distribute certs: each client receives CA + its own client cert/key only; AWS receives CA + server cert/key.
4. Phase 3 - AWS aggregator host provisioning:
   - Create EC2 (Ubuntu or Amazon Linux), attach Elastic IP, and optionally Route53 DNS.
   - Security Group inbound: Flower port from only the four node public IPs (or VPN CIDR); deny broad 0.0.0.0/0 exposure.
   - Install Python + dependencies and deploy project (git clone/rsync/container image).
   - Configure runtime env: FL_ROUNDS, FL_MIN_CLIENTS=4, FLOWER_PORT, FLOWER_TLS_CA_CERT, FLOWER_TLS_SERVER_CERT, FLOWER_TLS_SERVER_KEY, FLOWER_TLS_REQUIRE_CLIENT_CERT=true.
5. Phase 4 - Physical client node preparation (repeat for 4 machines):
   - Install Python/runtime deps and copy project code.
   - Place local node dataset files in agreed client format/path.
   - Install certs on each machine (CA + client_i cert/key).
   - Configure client env: CLIENT_ID, FLOWER_SERVER_HOST=<AWS DNS/IP>, FLOWER_SERVER_PORT, FLOWER_CA_CERT, FLOWER_CLIENT_CERT, FLOWER_CLIENT_KEY, FL_CLIENT_CONTINUOUS=true, FL_CLIENT_RETRY_SECONDS.
6. Phase 5 - Data readiness and consistency gate:
   - Verify each node can load local train/test tensors and model input shape remains compatible with FraudRNN input_dim.
   - Confirm all four clients have data before server launch to avoid min-client round blocking.
7. Phase 6 - Launch sequence:
   - Start AWS Flower server first and verify listening socket + TLS materials loaded.
   - Start clients one by one (1->4), validate mTLS handshake and successful registration.
   - Start a full federated run and observe round progress until at least one fit/evaluate cycle completes on all clients.
8. Phase 7 - Operations hardening:
   - Add process supervision (systemd or container restart policy) for server and clients.
   - Add centralized logging/monitoring for server + node logs.
   - Rotate certs and restrict file permissions for private keys.

**Relevant files**
- c:/Users/SAAD/OneDrive/Desktop/CSCC_S4/Projet metier/fraud-detection-pipeline/federated/server.py — Flower server startup, TLS certificate loading, min-client strategy and server port env.
- c:/Users/SAAD/OneDrive/Desktop/CSCC_S4/Projet metier/fraud-detection-pipeline/federated/client.py — client connection env vars, retry behavior, TLS client cert loading, data path contract.
- c:/Users/SAAD/OneDrive/Desktop/CSCC_S4/Projet metier/fraud-detection-pipeline/data/dataloader.py — file existence and data loading contract (expects concrete files, not wildcard globs).
- c:/Users/SAAD/OneDrive/Desktop/CSCC_S4/Projet metier/fraud-detection-pipeline/security/generate_mtls.sh — CA/server/client certificate generation and SAN definitions to adapt for AWS endpoint.
- c:/Users/SAAD/OneDrive/Desktop/CSCC_S4/Projet metier/fraud-detection-pipeline/docker-compose.yml — current local compose assumptions and env naming mismatch reference.
- c:/Users/SAAD/OneDrive/Desktop/CSCC_S4/Projet metier/fraud-detection-pipeline/ingestion/ingestion_1.py — node tensor generation pattern and save paths (same pattern in ingestion_2/3/4).

**Verification**
1. PKI check: verify cert chain and SAN for AWS endpoint before rollout.
2. Network check: from each physical machine, confirm TCP reachability to AWS Flower port.
3. mTLS check: start one client and verify successful certificate-authenticated connection.
4. Data check: on each node, confirm train/test tensors exist and are loadable by the client loader contract.
5. Federated check: run with FL_MIN_CLIENTS=4 and verify all four clients participate in fit/evaluate each round.
6. Resilience check: restart one client during continuous mode and confirm reconnect behavior.
7. Privacy/metrics check: verify server logs include aggregated accuracy and epsilon metrics each round.

**Decisions**
- Included: secure hybrid FL deployment (AWS server + 4 physical clients), TLS/mTLS, networking, data placement, launch and validation.
- Excluded: full CI/CD, autoscaling, multi-region HA, Kubernetes migration.
- Required decision: public internet exposure vs private connectivity (VPN/WireGuard/Tailscale). Recommendation: private overlay/VPN for production.
- Required decision: keep Kafka in the hybrid path or pre-generate tensors and run pure FL training. Recommendation: pre-generate tensors first for deployment simplicity.

**Further Considerations**
1. If you want strict zero-trust networking, prefer VPN overlay and keep Flower port private; this reduces certificate/SAN and firewall complexity.
2. If rounds stall frequently, set operational policy for partial participation versus strict FL_MIN_CLIENTS=4.
3. Add a runbook page documenting exact per-node env files and certificate inventory to reduce on-call errors.
