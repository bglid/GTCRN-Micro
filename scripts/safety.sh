#!/usr/bin/bash

set -euo pipefail

# running test code
uv run pytest --cov=gtcrn_micro --cov-report=term-missing

# doing checks with uv
uvx bandit -ll -r src
# can't run for workflow
uvx uv-secure
