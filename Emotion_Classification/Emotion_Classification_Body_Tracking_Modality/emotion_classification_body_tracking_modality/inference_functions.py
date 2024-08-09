from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf


def make_prediction_from_numpy(
        x,
        model
):
    if len(x.shape) <= 1:
        x = np.expand_dims(x, axis=0)

    prediction = tf.squeeze(model(x)).numpy()
    return prediction
