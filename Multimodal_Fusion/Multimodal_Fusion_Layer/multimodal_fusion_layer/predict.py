import os
import numpy as np
import pandas as pd
from conf import DATASETS_FOLDER, OUTPUTS_FOLDER, CUSTOM_SETTINGS,ID_TO_LABEL



def testing_multimodal_layer():
    print(f'Running Docker for Multimodal Layer!')
    #modalities = ['eGeMAPs','MFCC']
    #weights = [0.6,0.4]
    modalities = [CUSTOM_SETTINGS["encoder_config"]["input_type"]]
    meta_data = pd.read_csv(os.path.join(OUTPUTS_FOLDER,'test.csv'))
    files = meta_data['files']
    all_predictions = []
    for f in files:
        all_predictions_for_file = extract_predictions(modalities,f)
        majority_index = get_majority_voting_index(all_predictions_for_file)
        prediction_label = ID_TO_LABEL['RAVDESS'][majority_index]
        print(prediction_label)
        all_predictions.append(prediction_label)
    meta_data['prediction'] = all_predictions
    meta_data.to_csv(os.path.join(OUTPUTS_FOLDER,'predictions.csv'))


def extract_predictions(modalities,filename):
    all_predictions = None
    for mod in modalities:
        single_prediction = np.load(os.path.join(OUTPUTS_FOLDER,f"prediction-{mod}",filename))
        all_predictions = single_prediction if all_predictions is None else np.vstack((all_predictions,single_prediction))
    return all_predictions

def get_majority_voting_index(predictions):
    if len(predictions.shape)<=1:
        return np.argmax(predictions)
    return np.argmax(np.sum(predictions,axis=0))

def get_majority_voting_index_with_weights(predictions,weights):
    return np.argmax(np.dot(weights,predictions))

if __name__ == '__main__':
    testing_multimodal_layer()
