resource "aws_security_group" "flower_server_sg" {
  name        = "flower-server-sg"
  description = "Security group for Flower Federated Learning Aggregator"
  vpc_id      = aws_vpc.project.id

  ingress {
    description = "Flower gRPC port"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_flower_cidrs # TODO: USER_DATA # TODO: MODIFY_WITH_YOUR_DATA
  }

  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs # TODO: USER_DATA # TODO: MODIFY_WITH_YOUR_DATA
  }

  # Autoriser le trafic UDP pour le tunnel Tailscale
  ingress {
    description = "Tailscale UDP Discovery"
    from_port   = 41641
    to_port     = 41641
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"] # Nécessaire pour l'initialisation du tunnel
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "flower-server-sg"
    Project     = var.project_name
    Environment = "dev"
  }
}
