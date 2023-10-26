# Write Python code here
import json
import os
import pathlib

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from classifiers.linear import LinearClassifier
from conf import OUTPUTS_FOLDER, CUSTOM_SETTINGS


def predict():
    print(json.dumps(CUSTOM_SETTINGS, indent=4))
    split_paths = {'test': "test.csv"}
    features_size = np.load(os.path.join(OUTPUTS_FOLDER, CUSTOM_SETTINGS['inference_config']['features'], os.listdir(
        os.path.join(OUTPUTS_FOLDER, CUSTOM_SETTINGS['inference_config']['features']))[0])).size
    classifier = LinearClassifier(features_size, CUSTOM_SETTINGS['dataset_config']['number_of_labels'])
    classifier.load_state_dict(
        torch.load(os.path.join(OUTPUTS_FOLDER, 'supervised_training', 'dev_model_classifier.pt')))
    classifier.eval()
    predict_and_save_classifier(classifier, split_paths['test'], '')


def predict_and_save_classifier(classifier, csv_path, out_path):
    """
    generate_and_save : given the encoder, extract the features and save to .npy files

    Parameters
    ----------
        classifier:
            the pytorch encoder model to extract features from
        csv_path: str
            csv containing the paths to the files for which features have to be extracted and saved
        out_path: str
            output path to save the features to
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
        # TODO : find replacement for .replace('\\','/')) to have a seperator that works on all OS
        x = np.load(
            os.path.join(OUTPUTS_FOLDER, 'SSL_features', data_path).replace('\\', '/'))
        x_tensor = torch.tensor(np.expand_dims(x, axis=0) if len(x.shape) <= 1 else x)
        prediction = classifier(torch.nn.Flatten(start_dim=0)(x_tensor))
        np.save(os.path.join(OUTPUTS_FOLDER, f'prediction-{CUSTOM_SETTINGS["encoder_config"]["input_type"]}',
                             f"{data_path}"), prediction.detach().numpy())


if __name__ == '__main__':
    predict()
