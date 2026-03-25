import os
from pathlib import Path
from typing import Optional, Tuple

import flwr as fl


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


def main() -> None:
	rounds = int(os.getenv("FL_ROUNDS", "3"))
	min_clients = int(os.getenv("FL_MIN_CLIENTS", "4"))
	port = int(os.getenv("FLOWER_SERVER_PORT", "8080"))
	server_address = f"0.0.0.0:{port}"

	strategy = fl.server.strategy.FedAvg(
		fraction_fit=1.0,
		fraction_evaluate=1.0,
		min_fit_clients=min_clients,
		min_evaluate_clients=min_clients,
		min_available_clients=min_clients,
	)

	certificates = _build_certificates()

	print("[FLOWER] Démarrage du serveur")
	print(f"[FLOWER] Adresse: {server_address}")
	print(f"[FLOWER] Rounds: {rounds} | Min clients: {min_clients}")
	print(f"[FLOWER] TLS actif: {certificates is not None}")

	fl.server.start_server(
		server_address=server_address,
		config=fl.server.ServerConfig(num_rounds=rounds),
		strategy=strategy,
		certificates=certificates,
	)


if __name__ == "__main__":
	main()
