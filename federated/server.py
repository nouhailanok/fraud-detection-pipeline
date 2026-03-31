"""
federated/server.py — Flower federated-learning server.

Implements a FedAvg aggregation strategy that:
- Collects convergence metrics (Loss, Accuracy, F1-Score) per round.
- Tracks the privacy budget (ε) reported by each client.
- Logs a compliance summary for GDPR/PCI-DSS audit trails.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _read_cert(path: Optional[str]) -> Optional[bytes]:
    if not path:
        return None
    cert_path = Path(path)
    if not cert_path.exists():
        return None
    return cert_path.read_bytes()


def _build_certificates() -> Optional[Tuple[bytes, bytes, bytes]]:
    ca_cert = _read_cert(os.getenv("FLOWER_TLS_CA_CERT"))
    server_cert = _read_cert(os.getenv("FLOWER_TLS_SERVER_CERT"))
    server_key = _read_cert(os.getenv("FLOWER_TLS_SERVER_KEY"))

    if server_cert is None or server_key is None:
        return None

    require_client_cert = os.getenv("FLOWER_TLS_REQUIRE_CLIENT_CERT", "false").lower() == "true"
    if require_client_cert and ca_cert is None:
        raise RuntimeError("FLOWER_TLS_REQUIRE_CLIENT_CERT=true mais FLOWER_TLS_CA_CERT est absent")

    if ca_cert is not None:
        return (ca_cert, server_cert, server_key)

    return None


# ---------------------------------------------------------------------------
# Instrumented FedAvg strategy with metrics + privacy budget tracking
# ---------------------------------------------------------------------------

class FraudDetectionFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg strategy augmented with:
    - Per-round convergence logging (loss, accuracy, F1-score).
    - Privacy budget (ε) aggregation across clients.
    - Compliance audit trail for GDPR/PCI-DSS.
    """

    def __init__(self, target_epsilon: float = 0.9, total_rounds: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.target_epsilon = target_epsilon
        self._round_metrics: list[dict[str, Any]] = []
        self._total_rounds: int = total_rounds

    def aggregate_fit(
        self,
        server_round: int,
        results: list[Tuple[ClientProxy, FitRes]],
        failures: list[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate client updates and log convergence + privacy metrics."""
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Collect per-client metrics
        losses, accuracies, f1_scores, epsilons = [], [], [], []
        total_samples = 0
        for _, fit_res in results:
            n = fit_res.num_examples
            total_samples += n
            m = fit_res.metrics or {}
            if "loss" in m:
                losses.append(float(m["loss"]) * n)
            if "accuracy" in m:
                accuracies.append(float(m["accuracy"]) * n)
            if "f1_score" in m:
                f1_scores.append(float(m["f1_score"]) * n)
            if "epsilon" in m:
                epsilons.append(float(m["epsilon"]))

        n = max(total_samples, 1)
        round_summary: dict[str, Any] = {
            "round": server_round,
            "num_clients": len(results),
            "num_failures": len(failures),
        }

        if losses:
            round_summary["train_loss"] = sum(losses) / n
        if accuracies:
            round_summary["train_accuracy"] = sum(accuracies) / n
        if f1_scores:
            round_summary["train_f1"] = sum(f1_scores) / n
        if epsilons:
            max_eps = max(epsilons)
            round_summary["max_epsilon"] = max_eps
            if max_eps > self.target_epsilon:
                logger.warning(
                    "[ROUND %d] Privacy budget exceeded on at least one client: "
                    "ε_max=%.4f > limit=%.2f",
                    server_round, max_eps, self.target_epsilon,
                )

        self._round_metrics.append(round_summary)

        # Build human-readable log line
        parts = [f"[ROUND {server_round}/{self._get_total_rounds()}]"]
        for key in ("train_loss", "train_accuracy", "train_f1", "max_epsilon"):
            if key in round_summary:
                parts.append(f"{key}={round_summary[key]:.4f}")
        logger.info(" | ".join(parts))

        return aggregated_params, aggregated_metrics

    def _get_total_rounds(self) -> str:
        return str(self._total_rounds) if self._total_rounds > 0 else "?"

    def convergence_report(self) -> str:
        """Return a formatted convergence summary for the audit trail."""
        lines = ["=== Convergence Report ==="]
        for m in self._round_metrics:
            row = f"  Round {m['round']:>2}"
            for key in ("train_loss", "train_accuracy", "train_f1", "max_epsilon"):
                if key in m:
                    row += f" | {key}: {m[key]:.4f}"
            row += f" | clients: {m['num_clients']}"
            if m.get("num_failures"):
                row += f" | failures: {m['num_failures']}"
            lines.append(row)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    from federated.federated_config import PipelineConfig

    cfg = PipelineConfig()

    rounds = cfg.fl.rounds
    min_clients = cfg.fl.min_clients
    server_address = f"0.0.0.0:{cfg.fl.server_port}"

    strategy = FraudDetectionFedAvg(
        target_epsilon=cfg.dp.target_epsilon,
        total_rounds=rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )

    certificates = _build_certificates()

    logger.info("[FLOWER] Démarrage du serveur")
    logger.info("[FLOWER] Adresse: %s", server_address)
    logger.info("[FLOWER] Rounds: %d | Min clients: %d", rounds, min_clients)
    logger.info("[FLOWER] TLS actif: %s", certificates is not None)
    logger.info(cfg.summary())

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        certificates=certificates,
    )

    logger.info("\n%s", strategy.convergence_report())


if __name__ == "__main__":
    main()
