#!/usr/bin/env bash
# Construye la imagen Docker de la app Streamlit y la sube a ECR.
#
# Pasos:
#   1. Obtiene account ID y region de las credenciales AWS activas
#   2. Crea el repositorio en ECR si no existe
#   3. Autentica Docker contra ECR
#   4. Construye, tagea y hace push de la imagen
#
# Uso (desde la raíz del repo):
#   bash infra/scripts/build_and_push.sh
#
# Si estás en SageMaker Studio, usa --network sagemaker en el docker build.

IMAGE_NAME="1c-app"
DOCKERFILE_PATH="./app"

account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]; then
    echo "Error: no se pudo obtener el account ID. Verifica tus credenciales AWS."
    exit 255
fi

region=$(aws configure get region)
region=${region:-us-east-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${IMAGE_NAME}:latest"

echo "Creando repositorio ECR si no existe..."
aws ecr describe-repositories --repository-names "${IMAGE_NAME}" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    aws ecr create-repository --repository-name "${IMAGE_NAME}" > /dev/null
    echo "Repositorio ${IMAGE_NAME} creado."
fi

echo "Autenticando Docker contra ECR..."
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"

echo "Construyendo imagen..."
docker build -t ${IMAGE_NAME} ${DOCKERFILE_PATH}

echo "Subiendo imagen a ECR: ${fullname}"
docker tag ${IMAGE_NAME} ${fullname}
docker push ${fullname}

echo ""
echo "Imagen publicada: ${fullname}"
echo "Usa esta URI en el parámetro ImageUri del CloudFormation."
