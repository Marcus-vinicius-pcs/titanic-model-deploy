#!/bin/bash
# Simulação de deploy AWS com Terraform
# Uso: ./deploy_aws_simulado.sh <ambiente>

set -e

AMBIENTE=${1:-dev}
INFRA_DIR="$(dirname "$0")/../infra"
INVENTORY_DIR="$INFRA_DIR/inventories/$AMBIENTE"

if [ ! -d "$INVENTORY_DIR" ]; then
  echo "Ambiente '$AMBIENTE' não encontrado em $INFRA_DIR/inventories."
  exit 1
fi

cd "$INFRA_DIR"
echo "==> Inicializando Terraform..."
terraform init

echo "==> Validando Terraform..."
terraform validate

echo "==> Aplicando infraestrutura para o ambiente: $AMBIENTE"
echo "Comando real seria: terraform apply -var-file=\"inventories/$AMBIENTE/terraform.tfvars\""
echo "(Simulação: não será executado de verdade)"
# terraform apply -var-file="inventories/$AMBIENTE/terraform.tfvars"

echo "==> Deploy simulado finalizado com sucesso!"
