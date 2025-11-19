#!/usr/bin/bash

set -euo pipefail

# run from root
ONNX_INPUT="gtcrn_micro/models/onnx/"
# ONNX_FILE=gtcrn_micro.onnx
ONNX_FILE=gtcrn_micro.onnx # testing lowered opset 16
OUTPUT_PATH="gtcrn_micro/models/tflite/"
JSON_FILE=replace_gtcrn_micro.json

# run onnx conversion
uv run onnx2tf \
	\
	-i "${ONNX_INPUT}${ONNX_FILE}" \
	-o "${OUTPUT_PATH}" \
	-kat mix conv_cache tra_cache inter_cache \
	-prf ${OUTPUT_PATH}${JSON_FILE} \
	-cotof
