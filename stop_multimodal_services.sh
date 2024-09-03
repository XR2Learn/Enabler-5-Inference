#!/bin/bash

# Deleting All dataset, output files

echo "--------------------"
docker compose stop mock-xroom-writer
echo "--------------------"
#docker compose stop inference-data-processing-bm
echo "--------------------"
docker compose stop inference-data-processing-body-tracking
echo "--------------------"
sudo rm datasets/test_data/*
echo "--------------------"


