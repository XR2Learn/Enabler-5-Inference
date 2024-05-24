import logging
import os

import numpy as np
import pandas as pd
import redis
from tqdm import tqdm

from multimodal_fusion_layer.conf import (ID_TO_LABEL, REDIS_HOST, REDIS_PORT,
                                          PUBLISHER_ON, OUTPUT_MODALITY_FOLDER,
                                          DATA_TO_FUSION, DATASET, CUSTOM_SETTINGS,
                                          EXPERIMENT_ID, MODALITY)
from multimodal_fusion_layer.fusion_pub_sub import FusionPublisherSubscriber
from multimodal_fusion_layer.fusion_schema import get_majority_voting_index


def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def multimodal_prediction():
    meta_data = pd.read_csv(os.path.join(OUTPUT_MODALITY_FOLDER, 'test.csv'))

    if PUBLISHER_ON:
        publish_predicted_emotion()
    else:
        write_predicted_emotion(meta_data, DATA_TO_FUSION, DATASET)


def write_predicted_emotion(meta_data, modalities, dataset="RAVDESS"):
    files = meta_data['files']
    all_predictions = []
    for f in tqdm(files):
        all_predictions_for_file = extract_predictions(modalities, f)
        majority_index = get_majority_voting_index(all_predictions_for_file)
        prediction_label = ID_TO_LABEL[dataset][majority_index]
        all_predictions.append(prediction_label)
    meta_data['prediction'] = all_predictions
    meta_data.to_csv(os.path.join(OUTPUT_MODALITY_FOLDER, 'predictions.csv'))


def publish_predicted_emotion():
    redis_cli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    logger = init_logger()
    emotion_publisher = FusionPublisherSubscriber(redis_cli, logger)
    print('Listening to the Emotion Detection Channel')
    emotion_publisher.subscribe_unimodal_emotion_classification()


def ckpt_name(data_to_fusion):
    folder_structure_prediction = (
        f"{EXPERIMENT_ID}_"
        f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
        f"{MODALITY}_"
        f"{data_to_fusion}_"
        f"{CUSTOM_SETTINGS['encoder_config']['class_name']}"
    )
    return folder_structure_prediction


def extract_predictions(modalities, filename):
    all_predictions = None
    for mod in modalities:
        single_prediction = np.load(os.path.join(OUTPUT_MODALITY_FOLDER,
                                                 'prediction-' + ckpt_name(mod),
                                                 filename))
        all_predictions = single_prediction if all_predictions is None else np.vstack(
            (all_predictions, single_prediction))
    return all_predictions


def get_majority_voting_index_with_weights(predictions, weights):
    return np.argmax(np.dot(weights, predictions))


if __name__ == '__main__':
    multimodal_prediction()
