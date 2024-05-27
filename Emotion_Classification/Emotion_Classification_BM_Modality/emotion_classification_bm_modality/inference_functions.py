import os
import pathlib
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from classification_model import SupervisedModel


def predict_and_save(
        model: Union[torch.nn.Module, pl.LightningModule],
        destination_path: str,
        input_type: str,
        metadata_csv_path: Optional[str] = None,
        inference_data_path: Optional[str] = None,
        prefix_path: Optional[str] = None,
        transforms: Optional[Dict[str, Any]] = None,
):
    """
    Given the model, predict emotion and save it to .npy files

    Parameters
    ----------
        model:
            the pytorch trained model to for emotion recognition inference
        csv_path: str
            path to the csv containing the path files for features to be used to classify emotions
    Returns
    -------
        none
    """
    assert metadata_csv_path is not None or inference_data_path is not None, """
        Either inference path or metadata csv should be provided
    """

    if prefix_path is None:
        prefix_path = ""

    if metadata_csv_path is not None:
        files = pd.read_csv(os.path.join(metadata_csv_path), index_col=0)["files"]
    elif inference_data_path is not None:
        files = [
            os.path.join(
                inference_data_path, file_
            ) for file_ in os.listdir(inference_data_path) if file_.endswith(".npy")
        ]

    pathlib.Path(os.path.join(destination_path, f'prediction-{input_type}')).mkdir(parents=True, exist_ok=True)
    for data_path in tqdm(files):
        x = np.load(
            os.path.join(prefix_path, data_path)
        )
        prediction = make_prediction_from_numpy(x, model, transforms)
        np.save(
            os.path.join(destination_path, f'prediction-{input_type}', f"{data_path}"),
            prediction.detach().numpy()
        )


def make_prediction_from_numpy(
        x: np.ndarray,
        model: Union[torch.nn.Module, pl.LightningModule],
        transforms: Optional[Dict[str, Any]] = None,
):
    if len(x.shape) <= 1:
        x = np.expand_dims(x, axis=-1)
    if transforms is not None:
        x = transforms(x)

    # add batch dimension for the supervised model
    prediction = model(torch.nn.Flatten(start_dim=0)(torch.tensor(x)).unsqueeze(0)) if (
        not isinstance(model, SupervisedModel)
    ) else model(x.unsqueeze(0))
    prediction = prediction.squeeze(0)
    return prediction
