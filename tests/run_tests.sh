#!/bin/bash

# Exit immediately if any command fails
set -e

# Get the root directory of the git repository
WORKSPACE=$(git rev-parse --show-toplevel)

# Change to workspace root so paths are consistent
cd "$WORKSPACE"

# Run pytest with coverage and generate HTML report
pytest tests/ -v \
    --cov=deesseia \
    --cov-report=html:tests/htmlcov
