#!/bin/bash

REPO_PATH=$(pwd)/triton_models

echo "Model Repository: $REPO_PATH"

if [ ! -d "$REPO_PATH" ]; then
    echo "Ошибка: Папка triton_models не найдена."
    echo "Запустите сначала: poetry run python src/pneumonia_detect/commands.py setup_triton"
    exit 1
fi

docker run --rm \
    --name triton_server \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $REPO_PATH:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
