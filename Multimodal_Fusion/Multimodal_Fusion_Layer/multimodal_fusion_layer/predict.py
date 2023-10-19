import os
import numpy as np
import pandas as pd
from conf import DATASETS_FOLDER, OUTPUTS_FOLDER, CUSTOM_SETTINGS,ID_TO_LABEL



def testing_multimodal_layer():
    print(f'Running Docker for Multimodal Layer!')
    modalities = ['eGeMAPs','MFCC']
    weights = [0.6,0.4]
    meta_data = pd.read_csv(os.path.join(OUTPUTS_FOLDER,'test.csv'))
    files = meta_data['files']
    for f in files:
        all_predictions_for_file = extract_predictions(modalities,f)
        majority_index = get_majority_voting_index(all_predictions_for_file)
        print(ID_TO_LABEL['RAVDESS'][majority_index])


def extract_predictions(modalities,filename):
    all_predictions = None
    for mod in modalities:
        single_prediction = np.load(os.path.join(OUTPUTS_FOLDER,f"prediction-{mod}",filename))
        all_predictions = single_prediction if all_predictions is None else np.vstack((all_predictions,single_prediction))
    return all_predictions

def get_majority_voting_index(predictions):
    return np.argmax(np.sum(predictions,axis=0))

def get_majority_voting_index_with_weights(predictions,weights):
    return np.argmax(np.dot(weights,predictions))

if __name__ == '__main__':
    testing_multimodal_layer()
