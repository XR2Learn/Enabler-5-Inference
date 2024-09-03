#!/bin/bash


echo "--------------------"
docker compose start dashboard
echo "--------------------"
CONFIG_FILE_PATH="./configs/example.configuration_bm_end2end_streaming.json" docker compose up emotion-classification-bm -d
echo "--------------------"
CONFIG_FILE_PATH="./configs/example.configuration_body_tracking_end2end_streaming.json" docker compose up emotion-classification-body-tracking -d
echo "--------------------"
CONFIG_FILE_PATH="./configs/example.configuration_bm_end2end_streaming.json" docker compose up inference-data-processing-bm -d
echo "--------------------"
CONFIG_FILE_PATH="./configs/example.configuration_body_tracking_end2end_streaming.json" docker compose up inference-data-processing-body-tracking -d
#echo "--------------------"
docker compose up mock-xroom-writer -d