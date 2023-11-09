#!/bin/bash

# Deleting All dataset, output files

echo "--------------------"
echo "Emotion-classification-audio"
echo "--------------------"
docker compose run --rm emotion-classification-audio
echo "--------------------"
echo "Fusion-layer"
echo "--------------------"
docker compose run --rm fusion-layer
echo "--------------------"
echo "Ed-evaluation"
echo "--------------------"
docker compose run --rm ed-evaluation
echo "--------------------"

