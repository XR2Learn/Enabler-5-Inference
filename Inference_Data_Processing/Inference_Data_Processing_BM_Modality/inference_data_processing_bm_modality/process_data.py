import logging
import os
import time

import numpy as np

from conf import (
    CUSTOM_SETTINGS,
    MODALITY,
    REDIS_HOST,
    REDIS_PORT
)
from data_publisher import init_redis_data_publisher
from utils.processing_utils import (
    construct_windows,
    normalize,
    no_preprocessing,
    read_head,
    read_new_data,
    resample_bm,
    standardize
)
from utils.utils import init_logger, get_files


def launch_processing():
    logger = init_logger()
    monitor_directory = CUSTOM_SETTINGS[MODALITY]["inference_config"]["data_processing"]["monitor_directory"]

    delay = CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["seq_len"] + \
            CUSTOM_SETTINGS[MODALITY]["inference_config"]["data_processing"]["reading_delay"]

    window_size = CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["seq_len"] * \
                  CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["frequency"]

    data_publisher = init_redis_data_publisher(
        REDIS_HOST,
        REDIS_PORT,
        MODALITY,
        logger
    )

    processed_files = set()
    # infinite loop:
    #   checks if there are new files (sessions) that have not been processed before
    while True:
        # Get the list of files with the specified suffix
        files = get_files(
            monitor_directory,
            CUSTOM_SETTINGS[MODALITY]["inference_config"]["data_processing"]["modality_suffix"]
        )

        files = [
            os.path.join(monitor_directory, file) for file in files if (
                    os.path.join(monitor_directory, file) not in processed_files
            )
        ]
        files.sort()

        if not files:
            logger.info("Waiting for new files")
        else:
            logger.info(f"Files to process: {files}")
        # loop over target files in the directory
        for filename in files:
            session = os.path.basename(filename).split("_")[2]
            running_params = None
            logging.info(f"Processing {filename} ...")
            header = read_head(filename)
            # ts_idx = [i for i, col in enumerate(header) if col == "timestamp"]
            col_idx = [
                i for i, col in enumerate(header) if (
                        col in CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["use_sensors"]
                )
            ]
            last_position = 0
            buffer = []
            while True:
                file_size = os.stat(filename).st_size
                new_data, last_position = read_new_data(filename, last_position)

                # pre-processing and segmentation into time-windows of the specified length and frequency
                if new_data:
                    new_data = np.stack(new_data)[:, col_idx].astype(float)

                    if CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["resample_freq"] != \
                            CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["frequency"]:
                        resample_bm(
                            new_data,
                            CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["frequency"],
                            CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["resample_freq"]
                        )

                    preprocessing_functions = {
                        "normalize": normalize,
                        'standardize': standardize,
                        'raw': no_preprocessing
                    }

                    windows, buffer = construct_windows(new_data, window_size, buffer)

                    if windows is not None:
                        logging.info(
                            f"Number of extracted windows: {len(windows)}. Dimensionality: {windows.shape[1:]}. "
                            f"Buffer size {len(buffer)}"
                        )
                        preprocessed, running_params = preprocessing_functions[
                            CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["process"]
                        ](windows, running_params)

                        logging.info(
                            f"Pre-processing: '{CUSTOM_SETTINGS[MODALITY]['pre_processing_config']['process']}' applied."
                            f"Obtained batches shape: {preprocessed.shape}"
                        )
                        for window in preprocessed:
                            data_publisher.publish_data(session, window)
                    else:
                        logging.info(
                            f"No full temporal windows found. "
                            f"Buffer size: {len(buffer) if buffer is not None else 0}"
                        )
                else:
                    logging.info(
                        f"No new data found in {filename}."
                    )

                # delay before checking if the file was updated
                time.sleep(delay)

                # Check if file has been modified externally (e.g., new lines added)
                new_file_size = os.stat(filename).st_size
                if new_file_size <= file_size:
                    processed_files.add(filename)
                    break

        time.sleep(CUSTOM_SETTINGS[MODALITY]["inference_config"]["data_processing"]["reading_delay"])


if __name__ == "__main__":
    launch_processing()
