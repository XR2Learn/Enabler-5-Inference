# Write Python code here
import json
import logging
import os

import numpy as np
import torch

from classification_model import SupervisedModel
from classifiers.linear import LinearClassifier
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
# from emotion_classification_bm_modality.conf import PUBLISHER_ON
from emotionpubsub import init_redis_emocl_pubsub
from inference_functions import predict_and_save
from utils.init_utils import init_encoder, init_transforms

# # Since we included modality in the hierarchy of the conf file
# CUSTOM_SETTINGS = CUSTOM_SETTINGS[MODALITY]


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
    split_paths = {"test": "test.csv"}

    # modality = CUSTOM_SETTINGS['dataset_config']['modality'] if (
    #         'modality' in CUSTOM_SETTINGS['dataset_config']
    # ) else 'default_modality'

    ckpt_name = (
        f"{EXPERIMENT_ID}_"
        f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
        f"{MODALITY}_"
        f"{CUSTOM_SETTINGS[MODALITY]['sup_config']['input_type']}_"
        f"{CUSTOM_SETTINGS[MODALITY]['encoder_config']['class_name']}"
    )

    if (
            CUSTOM_SETTINGS[MODALITY]["inference_config"]["mode"] == "features" and
            PUBLISHER_ON
    ):
        raise ValueError("""
                         Mode 'features' in not supported for inference with pub/sub protocol
                         and running inference processing. Please, use 'end-to-end' mode".
                         """)

    num_classes = CUSTOM_SETTINGS['dataset_config'].get("number_of_labels", 3)
    if isinstance(num_classes, dict):
        num_classes = num_classes.get(MODALITY, 3)
    # Initialize models:
    # mode == features: use SSL features saved to npy and pass them through classifier
    if CUSTOM_SETTINGS[MODALITY]["inference_config"]["mode"] == "features":
        # Find features generated by the relevant encoder
        prefix_path = os.path.join(
            MODALITY_FOLDER,
            f"ssl_features_{ckpt_name}_encoder")
        assert os.path.isdir(prefix_path), f"The path to features does not exist: {prefix_path}"
        # check dimensionality of features
        features_size = np.load(
            os.path.join(
                prefix_path,
                os.listdir(prefix_path)[0]
            )
        ).size
        # initialize classifier from supervised model
        model = LinearClassifier(features_size, num_classes)
        model.load_state_dict(
            torch.load(os.path.join(MODALITY_FOLDER, "supervised_training", f"{ckpt_name}_classifier.pt")))
        transforms = None
    # mode == end-to-end: use fine-tuned model from pre-processed data
    elif CUSTOM_SETTINGS[MODALITY]["inference_config"]["mode"] == "end-to-end":
        prefix_path = os.path.join(MODALITY_FOLDER, CUSTOM_SETTINGS[MODALITY]["encoder_config"]["input_type"])
        # get checkpoint path
        if "model_path" not in CUSTOM_SETTINGS[MODALITY]["inference_config"]:
            supervised_model_checkpoint_path = os.path.join(
                MODALITY_FOLDER,
                "supervised_training",
                f"{ckpt_name}_model"
            )
        else:
            supervised_model_checkpoint_path = CUSTOM_SETTINGS[MODALITY]["inference_config"]["model_path"]
        # initialize the model
        encoder = init_encoder(model_cfg=CUSTOM_SETTINGS[MODALITY]["encoder_config"])
        classifier = LinearClassifier(encoder.out_size, num_classes)
        model = SupervisedModel.load_from_checkpoint(
            supervised_model_checkpoint_path + ".ckpt",
            encoder=encoder,
            classifier=classifier,
        )
        if "transforms" in CUSTOM_SETTINGS[MODALITY]:
            transforms, _ = init_transforms(CUSTOM_SETTINGS[MODALITY]["transforms"])
        else:
            transforms = None
    else:
        raise ValueError("Unexpected inference mode.")

    model.eval()
    use_inference_path = "inference_path" in CUSTOM_SETTINGS[MODALITY]["inference_config"]

    if PUBLISHER_ON:
        logging.info("Initializing emotion classification pub/sub")
        emocl_pubsub = init_redis_emocl_pubsub(
            REDIS_HOST,
            REDIS_PORT,
            MODALITY,
            f"{MODALITY}_data_stream",
            f"{MODALITY}_data",
            model,
            transforms,
            logger,
            ID_TO_LABEL
        )
        logging.info(f"Listening to {MODALITY}_data_stream channel...")
        emocl_pubsub.start_processing()
    else:
        predict_and_save(
            model,
            MODALITY_FOLDER,
            ckpt_name,
            os.path.join(MODALITY_FOLDER, split_paths["test"]) if not use_inference_path else None,
            None if not use_inference_path else CUSTOM_SETTINGS[MODALITY]["inference_config"]["inference_path"],
            prefix_path,
            transforms
        )


if __name__ == "__main__":
    predict()
