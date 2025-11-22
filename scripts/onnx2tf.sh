#!/usr/bin/bash

set -euo pipefail

# run from root
ONNX_INPUT="gtcrn_micro/models/onnx/"
ONNX_FILE=gtcrn_micro.onnx # testing lowered opset 16
# ONNX_FILE=gtcrn_s3.onnx # testing lowered opset 16
OUTPUT_PATH="gtcrn_micro/models/tflite/"
JSON_FILE=replace_gtcrn_micro.json
# JSON_FILE=replace_gtcrn_s3.json
CALIB_DATA="${OUTPUT_PATH}tflite_calibration.npy"

# double check file exists
if [ -e "$CALIB_DATA" ]; then
	echo "$CALIB_DATA exists."
else
	echo "$CALIB_DATA does not exist"
fi

# run onnx conversion
uv run onnx2tf \
	\
	-i "${ONNX_INPUT}${ONNX_FILE}" \
	-o "${OUTPUT_PATH}" \
	-kat mix conv_cache tra_cache inter_cache \
	-prf ${OUTPUT_PATH}${JSON_FILE} \
	-cotof \
	-oiqt \
	-qt per-channel \
	-cind "audio" "$CALIB_DATA" "[[[[0.]]]]" "[[[[1.]]]]" \
	-rtpo PReLU \
	-osd \
	-b 1 \
	-v debug \
	-ofgd
