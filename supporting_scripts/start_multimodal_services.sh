#!/bin/bash


echo "--------------------"
docker compose start dashboard
echo "--------------------"
CONFIG_FILE_PATH="./configs/example.configuration_multimodal_bm_body_tracking_streaming.json" docker compose up emotion-classification-bm -d
echo "--------------------"
CONFIG_FILE_PATH="./configs/example.configuration_multimodal_bm_body_tracking_streaming.json" docker compose up emotion-classification-body-tracking -d
echo "--------------------"
CONFIG_FILE_PATH="./configs/example.configuration_multimodal_bm_body_tracking_streaming.json" docker compose up inference-data-processing-bm -d
echo "--------------------"
CONFIG_FILE_PATH="./configs/example.configuration_multimodal_bm_body_tracking_streaming.json" docker compose up inference-data-processing-body-tracking -d
echo "--------------------"
CONFIG_FILE_PATH="./configs/example.configuration_multimodal_bm_body_tracking_streaming.json" docker compose up fusion-layer -d
docker compose up mock-xroom-writer -d