output "flower_server_public_ip" {
  description = "Public IP address of the Flower server EC2 instance"
  value       = aws_instance.flower_server.public_ip
}

output "flower_server_private_ip" {
  description = "Private IP address of the Flower server EC2 instance"
  value       = aws_instance.flower_server.private_ip
}

output "project_vpc_id" {
  description = "ID of the dedicated project VPC"
  value       = aws_vpc.project.id
}

output "project_public_subnet_id" {
  description = "ID of the public subnet used by the Flower server"
  value       = aws_subnet.project_public.id
}
