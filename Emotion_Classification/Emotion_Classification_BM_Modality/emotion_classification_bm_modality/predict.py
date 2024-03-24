# Write Python code here
import json
import os

import numpy as np
import torch

from classifiers.linear import LinearClassifier
from classification_model import SupervisedModel
from inference_functions import predict_and_save
from utils.init_utils import init_encoder, init_transforms
from conf import MODALITY_FOLDER, CUSTOM_SETTINGS, EXPERIMENT_ID


def predict():
    print(json.dumps(CUSTOM_SETTINGS, indent=4))
    split_paths = {"test": "test.csv"}
    # Initialize models:
    # mode == features: use SSL features saved to npy and pass them through classifier
    if CUSTOM_SETTINGS["inference_config"]["mode"] == "features":
        prefix_path = os.path.join(MODALITY_FOLDER, "SSL_features")
        features_size = np.load(
            os.path.join(
                MODALITY_FOLDER,
                CUSTOM_SETTINGS["inference_config"]["features"],
                os.listdir(os.path.join(MODALITY_FOLDER, CUSTOM_SETTINGS["inference_config"]["features"]))[0]
            )
        ).size
        model = LinearClassifier(features_size, CUSTOM_SETTINGS["dataset_config"]["number_of_labels"])
        model.load_state_dict(
            torch.load(os.path.join(MODALITY_FOLDER, "supervised_training", f"{EXPERIMENT_ID}_classifier.pt")))
        transforms = None
    # mode == end-to-end: use fine-tuned model from pre-processed data
    elif CUSTOM_SETTINGS["inference_config"]["mode"] == "end-to-end":
        prefix_path = os.path.join(MODALITY_FOLDER, CUSTOM_SETTINGS["encoder_config"]["input_type"])
        supervised_model_checkpoint_path = os.path.join(
            MODALITY_FOLDER,
            "supervised_training",
            f"{EXPERIMENT_ID}_model"
        ) if (
            "model_path" not in CUSTOM_SETTINGS["inference_config"]
        ) else CUSTOM_SETTINGS["inference_config"]["model_path"]
        encoder = init_encoder(model_cfg=CUSTOM_SETTINGS["encoder_config"])
        classifier = LinearClassifier(encoder.out_size, CUSTOM_SETTINGS["dataset_config"]["number_of_labels"])
        model = SupervisedModel.load_from_checkpoint(
            supervised_model_checkpoint_path + ".ckpt",
            encoder=encoder,
            classifier=classifier,
        )
        if "transforms" in CUSTOM_SETTINGS:
            transforms, _ = init_transforms(CUSTOM_SETTINGS["transforms"])
        else:
            transforms = None
    else:
        raise ValueError("Unexpected inference mode.")

    model.eval()
    use_inference_path = "inference_path" in CUSTOM_SETTINGS["inference_config"]

    predict_and_save(
        model,
        MODALITY_FOLDER,
        CUSTOM_SETTINGS["encoder_config"]["input_type"],
        os.path.join(MODALITY_FOLDER, split_paths["test"]) if not use_inference_path else None,
        None if not use_inference_path else CUSTOM_SETTINGS["inference_config"]["inference_path"],
        prefix_path,
        transforms
    )


if __name__ == "__main__":
    predict()
