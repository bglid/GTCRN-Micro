#!/usr/bin/bash

set -euo pipefail

# run from root
ONNX_INPUT="gtcrn_micro/streaming/onnx/"
ONNX_FILE=gtcrn_micro.onnx
OUTPUT_PATH="gtcrn_micro/streaming/tflite/"
JSON_FILE=replace_gtcrn_micro.json
CALIB_DATA="${OUTPUT_PATH}tflite_calibration.npy"

# getting the scale for STD in -cind
SCALE_FILE="${OUTPUT_PATH}calib_scale.txt"
if [[ ! -f "$SCALE_FILE" ]]; then
	echo "Missing $SCALE_FILE" >&2
	exit 1
fi

SCALE="$(tr -d ' \n\r\t' <"$SCALE_FILE")"
STD=$(python -c "print(1.0/float('$SCALE'))")

echo "SCALE=$SCALE"
echo "STD=$STD"
python - <<PY
import numpy as np
x = np.load("${CALIB_DATA}")
print("calib shape:", x.shape, "dtype:", x.dtype)
print("calib min/max:", float(x.min()), float(x.max()))
print("calib p0.1/p99.9:", float(np.percentile(x,0.1)), float(np.percentile(x,99.9)))
PY

# firstly convert the model from PyTorch ->> ONNX
if [ -e "$ONNX_INPUT$ONNX_FILE" ]; then
	echo "$ONNX_INPUT$ONNX_FILE exists..."
else
	echo "$ONNX_INPUT$ONNX_FILE doesn't exist..."
	echo "Running Streaming Torch -> ONNX conversion"
	uv run -m gtcrn_micro.streaming.conversion.stream_onnx
	echo "$ONNX_FILE created in $ONNX_INPUT"
fi

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
	\
	-prf ${OUTPUT_PATH}${JSON_FILE} \
	-cotof \
	-oiqt \
	-qt per-channel \
	-cind "audio" "$CALIB_DATA" "[[[[0.5], [0.5]]]]" "[[[[$STD], [$STD]]]]" \
	-rtpo PReLU \
	-osd \
	-b 1 \
	-v debug \
	-ofgd # -kat audio conv_cache \
