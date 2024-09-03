#!/bin/bash

# Deleting All dataset, output files

echo "--------------------"
docker compose down mock-xroom-writer
echo "--------------------"
docker compose down inference-data-processing-bm
echo "--------------------"
docker compose down inference-data-processing-body-tracking
echo "--------------------"
sudo rm datasets/test_data/*
echo "--------------------"


