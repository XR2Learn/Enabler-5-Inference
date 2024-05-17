import numpy as np

from multimodal_fusion_layer.conf import ID_TO_LABEL, MAPPING_RAVDESS_TO_THEORY_FLOW_DUMMY


def process_prediction(dataset, prediction_vector):
    prediction_vector = np.array(prediction_vector)
    majority_index = get_majority_voting_index(prediction_vector)
    prediction_label = int(majority_index)
    if dataset == 'RAVDESS':
        prediction_label = ID_TO_LABEL[dataset][majority_index]
        prediction_label = MAPPING_RAVDESS_TO_THEORY_FLOW_DUMMY[prediction_label]
    return prediction_label


def get_majority_voting_index(predictions):
    if len(predictions.shape) <= 1:
        return np.argmax(predictions)
    return np.argmax(np.sum(predictions, axis=0))
