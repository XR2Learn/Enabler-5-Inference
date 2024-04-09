# Write Python code here
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from conf import OUTPUT_MODALITY_FOLDER


def evaluate():
    meta_data = pd.read_csv(os.path.join(OUTPUT_MODALITY_FOLDER, 'predictions.csv'))
    gt = meta_data['labels']
    predictions = meta_data['prediction']
    evaluate_predictions(gt, predictions)


def evaluate_predictions(gt, prediction):
    # Compute confusion matrix
    unique_labels = gt.unique()
    cm = confusion_matrix(gt, prediction, labels=unique_labels)
    print("Confusion Matrix:")
    print("Actual\\Predicted   ", end="")
    for label in unique_labels:
        print(f"{label:<10s}", end="")
    print()  # New line
    for i, row in enumerate(cm):
        print(f"{unique_labels[i]:<10s}", end="")
        for value in row:
            print(f"{value:10d}", end="")
        print()  # New line

    print(classification_report(gt, prediction, zero_division=np.nan))


if __name__ == '__main__':
    evaluate()
