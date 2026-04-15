resource "aws_vpc" "project" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name        = "${var.project_name}-vpc"
    Project     = var.project_name
    Environment = "dev"
  }
}

resource "aws_internet_gateway" "project_igw" {
  vpc_id = aws_vpc.project.id

  tags = {
    Name        = "${var.project_name}-igw"
    Project     = var.project_name
    Environment = "dev"
  }
}

resource "aws_subnet" "project_public" {
  vpc_id                  = aws_vpc.project.id
  cidr_block              = var.public_subnet_cidr
  map_public_ip_on_launch = true

  tags = {
    Name        = "${var.project_name}-public-subnet"
    Project     = var.project_name
    Environment = "dev"
  }
}

resource "aws_route_table" "project_public_rt" {
  vpc_id = aws_vpc.project.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.project_igw.id
  }

  tags = {
    Name        = "${var.project_name}-public-rt"
    Project     = var.project_name
    Environment = "dev"
  }
}

resource "aws_route_table_association" "project_public_assoc" {
  subnet_id      = aws_subnet.project_public.id
  route_table_id = aws_route_table.project_public_rt.id
}

locals {
  project_subnet_id = aws_subnet.project_public.id
}
