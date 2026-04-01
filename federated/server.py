import os
from pathlib import Path
from typing import List, Tuple, Dict,Optional

import flwr as fl


# ============================
# 📊 AGGREGATION METRICS
# ============================

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Moyenne pondérée des accuracies"""
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)

    accuracy = sum(
        num_examples * m.get("accuracy", 0.0)
        for num_examples, m in metrics
    ) / total_examples

    return {"accuracy": accuracy}


# ============================
# 🔐 AGGREGATION PRIVACY ε
# ============================

def aggregate_epsilon(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Affiche le epsilon moyen"""
    epsilons = [m["epsilon"] for _, m in metrics if "epsilon" in m]

    if epsilons:
        avg_eps = sum(epsilons) / len(epsilons)
        max_eps = max(epsilons)

        print("\n🔐 ===== Privacy Report =====")
        print(f"📊 Clients: {len(epsilons)}")
        print(f"📉 Avg ε: {avg_eps:.4f}")
        print(f"📈 Max ε: {max_eps:.4f}")

        if max_eps > 1.0:
            print("⚠️ WARNING: ε > 1.0 (privacy faible)")
        else:
            print("✅ Privacy OK (ε < 1.0)")
        print("===========================\n")

    return {"avg_epsilon": avg_eps, "max_epsilon": max_eps} if epsilons else {}


# ============================
# 🔐 TLS 
# ============================

def _read_cert(path: Optional[str]) -> Optional[bytes]:
	if not path:
		return None
	cert_path = Path(path)
	if not cert_path.exists():
		return None
	return cert_path.read_bytes()

# def load_certificates():
#     """Charge les certificats TLS si fournis"""
#     ca_cert = os.getenv("FLOWER_CA_CERT")
#     server_cert = os.getenv("FLOWER_SERVER_CERT")
#     server_key = os.getenv("FLOWER_SERVER_KEY")

#     if ca_cert and server_cert and server_key:
#         try:
#             return (
#                 open(ca_cert, "rb").read(),
#                 open(server_cert, "rb").read(),
#                 open(server_key, "rb").read(),
#             )
#         except Exception as e:
#             print(f"⚠️ Erreur chargement TLS: {e}")

#     return None


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


# ============================
# 🚀 MAIN SERVER
# ============================

def main():
    # ⚙️ Config dynamique
    rounds = int(os.getenv("FL_ROUNDS", "5"))
    min_clients = int(os.getenv("FL_MIN_CLIENTS", "2"))
    port = int(os.getenv("FLOWER_PORT", "8080"))
    server_address = f"0.0.0.0:{port}"

    print("🚀 Lancement du serveur Flower")
    print(f"🔁 Rounds: {rounds}")
    print(f"👥 Min clients: {min_clients}")

    # 🧠 STRATEGY FedAvg
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,

        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=aggregate_epsilon,
    )

    # 🔐 TLS
    certificates = _build_certificates()

    # 🚀 START SERVER
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        certificates=certificates,
    )


if __name__ == "__main__":
    main()