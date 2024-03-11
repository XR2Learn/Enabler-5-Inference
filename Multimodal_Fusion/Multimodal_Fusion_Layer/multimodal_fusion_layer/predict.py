import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import redis

from multimodal_fusion_layer.conf import OUTPUTS_FOLDER, CUSTOM_SETTINGS, ID_TO_LABEL, REDIS_HOST, REDIS_PORT, \
    MAPPING_RAVDESS_TO_THEORY_FLOW_DUMMY, PUBLISHER_ON
from multimodal_fusion_layer.emotion_publisher import EmotionPublisher


def multimodal_prediction():
    # modalities = ['eGeMAPs','MFCC']
    # weights = [0.6,0.4]
    modalities = [CUSTOM_SETTINGS["encoder_config"]["input_type"]]
    dataset = CUSTOM_SETTINGS["dataset_config"]["dataset_name"] if (
        "dataset_config" in CUSTOM_SETTINGS and "dataset_name" in CUSTOM_SETTINGS["dataset_config"]
    ) else "RAVDESS"
    meta_data = pd.read_csv(os.path.join(OUTPUTS_FOLDER, 'test.csv'))

    if PUBLISHER_ON:
        publish_predicted_emotion(meta_data, modalities, dataset)
    else:
        write_predicted_emotion(meta_data, modalities, dataset)


def write_predicted_emotion(meta_data, modalities, dataset="RAVDESS"):
    files = meta_data['files']
    all_predictions = []
    for f in tqdm(files):
        all_predictions_for_file = extract_predictions(modalities, f)
        majority_index = get_majority_voting_index(all_predictions_for_file)
        prediction_label = ID_TO_LABEL[dataset][majority_index]
        # print(prediction_label)
        all_predictions.append(prediction_label)
    meta_data['prediction'] = all_predictions
    meta_data.to_csv(os.path.join(OUTPUTS_FOLDER, 'predictions.csv'))


def publish_predicted_emotion(meta_data, modalities, dataset="RAVDESS"):
    files = meta_data['files']
    print('Publishing the predicted emotions')
    redis_cli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    emotion_publisher = EmotionPublisher(redis_cli)
    for file in files:
        all_predictions_for_file = extract_predictions(modalities, file)
        majority_index = get_majority_voting_index(all_predictions_for_file)
        prediction_label = ID_TO_LABEL[dataset][majority_index]
        prediction_label_to_publish = MAPPING_RAVDESS_TO_THEORY_FLOW_DUMMY[prediction_label]
        emotion_publisher.publish_emotion(prediction_label_to_publish)
        print(prediction_label_to_publish)


def extract_predictions(modalities, filename):
    all_predictions = None
    for mod in modalities:
        single_prediction = np.load(os.path.join(OUTPUTS_FOLDER, f"prediction-{mod}", filename))
        all_predictions = single_prediction if all_predictions is None else np.vstack(
            (all_predictions, single_prediction))
    return all_predictions


def get_majority_voting_index(predictions):
    if len(predictions.shape) <= 1:
        return np.argmax(predictions)
    return np.argmax(np.sum(predictions, axis=0))


def get_majority_voting_index_with_weights(predictions, weights):
    return np.argmax(np.dot(weights, predictions))


if __name__ == '__main__':
    multimodal_prediction()
