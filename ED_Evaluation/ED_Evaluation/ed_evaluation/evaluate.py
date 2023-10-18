# Write Python code here
import pandas as pd
import os

from sklearn.metrics import classification_report
from conf import OUTPUTS_FOLDER


def evaluate():
    print(f'Running Docker for Emotion Classification - Audio Modality')

    meta_data = pd.read_csv(os.path.join(OUTPUTS_FOLDER,'predictions.csv'))
    gt = meta_data['labels']
    predictions = meta_data['prediction']
    evaluate_predictions(gt,predictions)

def evaluate_predictions(gt,prediction):
    #print(accuracy_score(gt,prediction))
    print(classification_report(gt,prediction))





if __name__ == '__main__':
    evaluate()
