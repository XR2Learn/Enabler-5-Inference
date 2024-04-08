#!/bin/bash

echo "--------------------"
echo "Emotion-classification-audio"
echo "--------------------"
CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm emotion-classification-audio
echo "--------------------"
echo "Fusion-layer"
echo "--------------------"
CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm fusion-layer
echo "--------------------"
echo "Ed-evaluation"
echo "--------------------"
CONFIG_FILE_PATH=$CONFIG_FILE_PATH compose run --rm ed-evaluation
echo "--------------------"

