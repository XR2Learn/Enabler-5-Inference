# Write Python code here
import pandas as pd
import numpy as np
import os
import torch

from tqdm import tqdm
from conf import DATASETS_FOLDER, OUTPUTS_FOLDER, CUSTOM_SETTINGS,EXPERIMENT_ID,ID_TO_LABEL
from utils.init_utils import (init_encoder)
from classifiers.linear import LinearClassifier
from classification_model import classification_model

from sklearn.metrics import accuracy_score,classification_report


def testing_component():
    print(f'Running Docker for Emotion Classification - Audio Modality')

    print(CUSTOM_SETTINGS)
    splith_paths = {'test': "test.csv"}

    encoder = init_encoder(CUSTOM_SETTINGS["encoder_config"],
                           CUSTOM_SETTINGS['encoder_config']['pretrained'] if "pretrained_path" in CUSTOM_SETTINGS[
                               'encoder_config'].keys()  else f"{OUTPUTS_FOLDER}/SSL_Training/{EXPERIMENT_ID}_encoder.pt" if "pretrained_same_experiment" in CUSTOM_SETTINGS[
                               'encoder_config'].keys() and CUSTOM_SETTINGS['encoder_config']["pretrained_same_experiment"] else None
                           )
    classifier = LinearClassifier(encoder.out_size, CUSTOM_SETTINGS['dataset_config']['number_of_labels'])

    model = classification_model(encoder=encoder, classifier=classifier, **CUSTOM_SETTINGS['sup_config']['kwargs'])
    model.load_state_dict(torch.load(os.path.join(OUTPUTS_FOLDER,'supervised_training','dev_model_model.pt')))
    model.eval()
    print(model)

    predict_and_save(model,splith_paths['test'],'')

def predict_and_save(model, csv_path, out_path):
    """
    generate_and_save : given the encoder, extract the features and save to .npy files

    Args:
        encoder: the pytorch encoder model to extract features from
        csv_path: csv containing the paths to the files for which features have to be extracted and saved
        out_path: output path to save the features to
    Returns:
        none
    """

    meta_data = pd.read_csv(os.path.join(OUTPUTS_FOLDER, csv_path))
    all_predictions = []
    for data_path in tqdm(meta_data['files']):
        # TODO : find replacement for .replace('\\','/')) to have a seperator that works on all OS
        x = np.load(
            os.path.join(OUTPUTS_FOLDER, CUSTOM_SETTINGS['encoder_config']['input_type'], data_path).replace('\\', '/'))
        x_tensor = torch.tensor(np.expand_dims(x, axis=0) if len(x.shape) <= 1 else x)
        prediction = model(x_tensor.T)
        prediction_label = ID_TO_LABEL['RAVDESS'][torch.argmax(prediction).item()]
        all_predictions.append(prediction_label)
    meta_data['prediction'] = all_predictions
    meta_data.to_csv(os.path.join(OUTPUTS_FOLDER,'predictions.csv'))
        # print(data_path.split(os.path.sep))
        #np.save(os.path.join(out_path, data_path.split(os.path.sep)[-1]), features.detach().numpy())
    evaluate_predictions(meta_data['labels'],all_predictions)

def evaluate_predictions(gt,prediction):
    #print(accuracy_score(gt,prediction))
    print(classification_report(gt,prediction))





if __name__ == '__main__':
    testing_component()
