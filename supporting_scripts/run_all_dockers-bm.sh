#!/bin/bash

# Deleting All dataset, output files

echo "--------------------"
echo "Emotion-classification-bm"
echo "--------------------"
docker compose run --rm emotion-classification-bm
echo "--------------------"
echo "Fusion-layer"
echo "--------------------"
docker compose run --rm fusion-layer
echo "--------------------"
echo "Ed-evaluation"
echo "--------------------"
docker compose run --rm ed-evaluation
echo "--------------------"

