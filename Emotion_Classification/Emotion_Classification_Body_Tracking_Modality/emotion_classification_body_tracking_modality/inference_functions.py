from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import pytorch_lightning as pl

from classification_model import SupervisedModel


def make_prediction_from_numpy(
        x: np.ndarray,
        model: Union[torch.nn.Module, pl.LightningModule],
        transforms: Optional[Dict[str, Any]] = None,
):
    if len(x.shape) <= 1:
        x = np.expand_dims(x, axis=-1)
    if transforms is not None:
        x = transforms(x)

    x = torch.tensor(x).permute(0, 2, 1)
    prediction = model(x)
    prediction = prediction.squeeze(0)
    return prediction.detach().numpy()
