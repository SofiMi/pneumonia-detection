#!/bin/bash
set -e

ONNX_PATH=${1:-"models/model.onnx"}
OUTPUT_DIR=${2:-"models"}
RESULT_NAME=${3:-"model.engine"}

echo "=== TensorRT Conversion ==="
echo "Input: $ONNX_PATH"
echo "Output: $OUTPUT_DIR/$RESULT_NAME"

mkdir -p "$OUTPUT_DIR"

if ! command -v trtexec &> /dev/null; then
    TRT_EXEC="/usr/src/tensorrt/bin/trtexec"
    if [ ! -f "$TRT_EXEC" ]; then
        echo "Ошибка: trtexec не найден ни в PATH, ни в /usr/src/tensorrt/bin/"
        exit 1
    fi
else
    TRT_EXEC="trtexec"
fi

$TRT_EXEC \
    --onnx="$ONNX_PATH" \
    --saveEngine="$OUTPUT_DIR/$RESULT_NAME" \
    --fp16 \
    --explicitBatch \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:1x3x224x224 \
    --maxShapes=input:8x3x224x224

echo "Готово"
