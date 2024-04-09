# Emotion Classification Body Tracking Modality

Emotion Classification component for body tracking modality.

# Including component in Docker-compose.yml file as a service

```yaml
emotion-classification-body-tracking:
    image: some.registry.com/xr2learn-enablers/emotion-classification-body-tracking:latest
    build:
      context: 'Emotion_Classification/Emotion_Classification_Body_Tracking_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./Emotion_Classification/Emotion_Classification_Body_Tracking_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python emotion_classification_body_tracking_modality/predict.py

```
