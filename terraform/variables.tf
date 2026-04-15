variable "aws_region" {
  description = "AWS region where infrastructure will be provisioned"
  type        = string
  default     = "eu-west-3" # TODO: MODIFY_WITH_YOUR_DATA
}

variable "project_name" {
  description = "Project name used for tagging cloud resources"
  type        = string
  default     = "fraud-detection-pipeline"
}

variable "vpc_cidr" {
  description = "CIDR block for the dedicated project VPC"
  type        = string
  default     = "10.42.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for the public subnet where the EC2 instance is placed"
  type        = string
  default     = "10.42.1.0/24"
}

variable "instance_type" {
  description = "EC2 instance type for the Flower Aggregator"
  type        = string
  default     = "c7i-flex.large"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
  default     = "saad-aggregator-key" # TODO: MODIFY_WITH_YOUR_DATA
}

variable "ubuntu_2404_ami_id" {
  description = "Ubuntu 24.04 LTS AMI ID for the selected region"
  type        = string
  default     = "ami-007dcf089b8078f1a" # TODO: MODIFY_WITH_YOUR_DATA
}

variable "allowed_flower_cidrs" {
  description = "CIDR blocks allowed to access Flower gRPC (port 8080)"
  type        = list(string)
  default     = ["100.64.0.0/10"] # TODO: MODIFY_WITH_YOUR_DATA
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH (port 22)"
  type        = list(string)
  default     = ["100.95.110.63/32", "100.127.240.19/32", "100.71.252.79/32", "100.87.187.93/32"] # TODO: MODIFY_WITH_YOUR_DATA
}