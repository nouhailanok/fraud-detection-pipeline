resource "aws_instance" "flower_server" {
  # Region-specific Ubuntu 24.04 AMI.
  # Replace this with the AMI ID for your selected region before apply.
  ami = var.ubuntu_2404_ami_id # TODO: MODIFY_WITH_YOUR_DATA

  # Good baseline for model aggregation workloads.
  instance_type = var.instance_type

  # Existing EC2 key pair in your AWS account.
  key_name = var.key_pair_name # TODO: MODIFY_WITH_YOUR_DATA

  subnet_id                   = local.project_subnet_id
  vpc_security_group_ids      = [aws_security_group.flower_server_sg.id]
  associate_public_ip_address = true

  tags = {
    Name        = "flower-server"
    Project     = var.project_name
    Role        = "federated-aggregator"
    Environment = "dev"
  }
}
