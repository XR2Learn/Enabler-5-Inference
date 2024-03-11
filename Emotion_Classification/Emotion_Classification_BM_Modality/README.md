# Emotion Classification using bm modality

Emotion Classification using bm modality

# Including component in Docker-compose.yml file as a service

```yaml
emotion-classification-bm:
    image: some.registry.com/xr2learn-enablers/emotion-classification-bm:latest
    build:
      context: 'Emotion_Classification/Emotion_Classification_BM_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./Emotion_Classification/Emotion_Classification_BM_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python emotion_classification_bm_modality/predict.py

```
