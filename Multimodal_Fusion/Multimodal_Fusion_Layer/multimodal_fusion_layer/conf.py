"""
File to include global variables across the python package and configuration.
All the other files inside the python package can access these variables.
"""
import json
import os
import pathlib

from decouple import config

EMOTIONS_RAVDESS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                    5: 'angry', 6: 'fear', 7: 'disgust',
                    8: 'surprise'}
EMOTION_INTENSITY_RAVDESS = {1: 'normal', 2: "strong"}

BM_LABEL_TO_EMOTION = {
    "01": "BORED",
    "02": "ENGAGED",
    "03": "FRUSTRATED"
}
BM_EMOTION_TO_LABEL = {
    "BORED": "01",
    "ENGAGED": "02",
    "FRUSTRATED": "03"
}

LABEL_TO_ID = {
    'RAVDESS': {
        'neutral': 0,
        'calm': 1,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fearful': 5,
        'disgust': 6,
        'surprised': 7
    },
    "BM": {
        "BORED": 0,
        "ENGAGED": 1,
        "FRUSTRATED": 2
    }
}

ID_TO_LABEL = {
    'RAVDESS': {
        0: 'neutral',
        1: 'calm',
        2: 'happy',
        3: 'sad',
        4: 'angry',
        5: 'fearful',
        6: 'disgust',
        7: 'surprised'
    },
    "XRoom": {
        0: "BORED",
        1: "ENGAGED",
        2: "FRUSTRATED"
    }
}

# A dummy mapping from RAVDESS emotions to the theory of flow
MAPPING_RAVDESS_TO_THEORY_FLOW_DUMMY = {
    'neutral': 1, 'calm': 1, 'happy': 1, 'sad': 0,
    'angry': 2, 'fearful': 2, 'disgust': 0,
    'surprised': 2}

MAIN_FOLDER_DEFAULT = pathlib.Path(__file__).parent.parent.absolute()
MAIN_FOLDER = config('MAIN_FOLDER', default=MAIN_FOLDER_DEFAULT)
outputs_folder = os.path.join(MAIN_FOLDER, 'outputs')
OUTPUTS_FOLDER = config('OUTPUTS_FOLDER', default=outputs_folder)
datasets_folder = os.path.join(MAIN_FOLDER, 'outputs')
DATASETS_FOLDER = config('DATASETS_FOLDER', default=datasets_folder)
DATA_PATH = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'RAVDESS', 'audio_speech_actors_01-24')
RAVDESS_DATA_PATH = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'RAVDESS')
EXPERIMENT_ID = config('EXPERIMENT_ID', default='development-model')

# COMPONENT_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, 'ssl_training')

REDIS_PORT = config('REDIS_PORT', default='6379')
REDIS_HOST = config('REDIS_HOST', default='localhost')

# Yet to check if this is really necessary, maybe only for cases where passing values as ENV VARS is too cumbersome
# e.g. [[1, 'a', ],['789', 'o', 9]] would be very annoying to write and parse.
CUSTOM_SETTINGS = {
    'key': {
        'default': 'value',
    },
    'pre_processing': {
        'some_config_preprocessing': 'values',
    }
}
path_custom_settings = os.path.join(MAIN_FOLDER, 'configuration.json')
PATH_CUSTOM_SETTINGS = config('PATH_CUSTOM_SETTINGS', default=path_custom_settings)
print(PATH_CUSTOM_SETTINGS)
if os.path.exists(PATH_CUSTOM_SETTINGS):
    with open(PATH_CUSTOM_SETTINGS, 'r') as f:
        CUSTOM_SETTINGS = json.load(f)

PUBLISHER_ON = config('PUBLISHER_ON', default=CUSTOM_SETTINGS['inference_config'].get('publisher', False), cast=bool)

DATA_TO_FUSION = CUSTOM_SETTINGS['inference_config'].get('data_to_fusion',
                                                         [CUSTOM_SETTINGS["encoder_config"]["input_type"]])
DATASET = CUSTOM_SETTINGS["dataset_config"]["dataset_name"]

MODALITY = CUSTOM_SETTINGS["dataset_config"].get("modality", "default_modality")

OUTPUT_MODALITY_FOLDER = os.path.join(OUTPUTS_FOLDER, DATASET, MODALITY)


# CKPT_NAME = (
#         f"{EXPERIMENT_ID}_"
#         f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
#         f"{MODALITY}_"
#         f"{CUSTOM_SETTINGS['sup_config']['input_type']}_"
#         f"{CUSTOM_SETTINGS['encoder_config']['class_name']}"
#     )