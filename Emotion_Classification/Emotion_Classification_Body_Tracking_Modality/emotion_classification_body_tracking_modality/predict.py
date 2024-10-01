# Write Python code here
import json
import logging
import os

import tensorflow as tf

from conf import (
    CUSTOM_SETTINGS,
    MODALITY,
    MODALITY_FOLDER,
    EXPERIMENT_ID,
    ID_TO_LABEL,
    REDIS_HOST,
    REDIS_PORT,
    PUBLISHER_ON
)
from emotionpubsub import init_redis_emocl_pubsub


def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def predict():
    logger = init_logger()

    logger.info("User configurations:")
    logger.info(json.dumps(CUSTOM_SETTINGS, indent=4))

    ckpt_name = (
        f"{EXPERIMENT_ID}_"
        f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
        f"{MODALITY}"
    )

    if (
            CUSTOM_SETTINGS[MODALITY]["inference_config"]["mode"] == "features" and
            PUBLISHER_ON
    ):
        raise ValueError("""
                         Mode 'features' in not supported for inference with pub/sub protocol
                         and running inference processing. Please, use 'end-to-end' mode".
                         """)

    # Initialize models:
    # mode == end-to-end: use fine-tuned model from pre-processed data
    if CUSTOM_SETTINGS[MODALITY]["inference_config"]["mode"] == "end-to-end":
        # get checkpoint path
        if "model_path" not in CUSTOM_SETTINGS[MODALITY]["inference_config"]:
            supervised_model_checkpoint_path = os.path.join(
                MODALITY_FOLDER,
                "supervised_training",
                f"{ckpt_name}_model.ckpt"
            )
        model = tf.keras.models.load_model(supervised_model_checkpoint_path)
    else:
        raise ValueError("Unexpected inference mode.")

    if PUBLISHER_ON:
        logging.info("Initializing emotion classification pub/sub")
        emocl_pubsub = init_redis_emocl_pubsub(
            REDIS_HOST,
            REDIS_PORT,
            MODALITY,
            f"{MODALITY}_data_stream",
            f"{MODALITY}_data",
            model,
            logger,
            ID_TO_LABEL
        )
        logging.info(f"Listening to {MODALITY}_data_stream channel...")
        emocl_pubsub.start_processing()
    else:
        raise ValueError("Only pub/sub inference is implemented.")


if __name__ == "__main__":
    predict()
