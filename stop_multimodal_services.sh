#!/bin/bash

# Deleting All dataset, output files

echo "--------------------"
docker compose stop mock-xroom-writer
echo "--------------------"
docker compose stop inference-data-processing-bm
echo "--------------------"
sudo rm datasets/test_data/*
echo "--------------------"
#docker compose run --rm inference-data-processing-bm
#echo "--------------------"
#docker compose run --rm mock-xroom-writer

