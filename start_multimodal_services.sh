#!/bin/bash


echo "--------------------"
docker compose start dashboard
echo "--------------------"
docker compose start emotion-classification-bm
echo "--------------------"
docker compose start inference-data-processing-bm
echo "--------------------"
docker compose start mock-xroom-writer