#!/usr/bin/bash
set -euo pipefail

# Meant for running eval metrics from CL
# Due to UV being setup within python dir, cd in first is required

cd ./python

uv run -m gtcrn_micro.utils.torch_converter
