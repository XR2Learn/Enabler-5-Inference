# Inference Data Processing

Data processing component for the inference with bio-measurements

# Including component in Docker-compose.yml file as a service

```yaml
inference-data-processing-bm:
    image: some.registry.com/xr2learn-enablers/inference-data-processing-bm:latest
    build:
      context: 'Inference_Data_Processing/Inference_Data_Processing_BM_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./Inference_Data_Processing/Inference_Data_Processing_BM_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python inference_data_processing_bm_modality/process_data.py

```
