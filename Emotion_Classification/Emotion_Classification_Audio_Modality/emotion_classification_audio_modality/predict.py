# Write Python code here
import json
import os
import pathlib

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from classifiers.linear import LinearClassifier
from conf import OUTPUTS_FOLDER, CUSTOM_SETTINGS, EXPERIMENT_ID


def predict():
    print(json.dumps(CUSTOM_SETTINGS, indent=4))
    split_paths = {'test': "test.csv"}
    features_size = np.load(os.path.join(OUTPUTS_FOLDER, CUSTOM_SETTINGS['inference_config']['features'], os.listdir(
        os.path.join(OUTPUTS_FOLDER, CUSTOM_SETTINGS['inference_config']['features']))[0])).size
    classifier = LinearClassifier(features_size, CUSTOM_SETTINGS['dataset_config']['number_of_labels'])
    classifier.load_state_dict(
        torch.load(os.path.join(OUTPUTS_FOLDER, 'supervised_training', f'{EXPERIMENT_ID}_classifier.pt')))
    classifier.eval()
    predict_and_save(classifier, split_paths['test'])


def predict_and_save(classifier, csv_path):
    """
    Given the classifier, predict emotion and save it to .npy files

    Parameters
    ----------
        classifier:
            the pytorch trained classifier model to do inference of emotions
        csv_path: str
            path to the csv containing the path files for features to be used to classify emotions
    Returns
    -------
        none
    """

    meta_data = pd.read_csv(os.path.join(OUTPUTS_FOLDER, csv_path), index_col=0)
    # all_predictions = []
    pathlib.Path(os.path.join(OUTPUTS_FOLDER, f'prediction-{CUSTOM_SETTINGS["encoder_config"]["input_type"]}')).mkdir(
        parents=True,
        exist_ok=True)
    for data_path in tqdm(meta_data['files']):
        x = np.load(
            os.path.join(OUTPUTS_FOLDER, 'SSL_features', data_path))
        x_tensor = torch.tensor(np.expand_dims(x, axis=0) if len(x.shape) <= 1 else x)
        prediction = classifier(torch.nn.Flatten(start_dim=0)(x_tensor))
        np.save(os.path.join(OUTPUTS_FOLDER, f'prediction-{CUSTOM_SETTINGS["encoder_config"]["input_type"]}',
                             f"{data_path}"), prediction.detach().numpy())


if __name__ == '__main__':
    predict()
