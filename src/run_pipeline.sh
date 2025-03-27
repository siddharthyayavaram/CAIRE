#!/bin/bash
set -e

echo "Running VEL pipeline..."
python -m src.main_VEL

echo "Running cultural relevance scoring..."
python -m src.main_culture

echo "Pipeline execution completed successfully."