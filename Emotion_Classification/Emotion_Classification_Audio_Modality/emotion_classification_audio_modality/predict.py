# Write Python code here
from emotion_classification_audio_modality.conf import DATASETS_FOLDER, OUTPUTS_FOLDER, PATH_CUSTOM_SETTINGS


def testing_component():
    print(f'Running Docker for Emotion Classification - Audio Modality')
    print(DATASETS_FOLDER)
    print(OUTPUTS_FOLDER)
    print(PATH_CUSTOM_SETTINGS)


if __name__ == '__main__':
    testing_component()
