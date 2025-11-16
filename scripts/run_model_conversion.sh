#!/usr/bin/bash
set -euo pipefail

cd ./python

uv run -m gtcrn_micro.utils.torch_converter
