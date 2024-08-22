import logging
import os
import time

import numpy as np
import pandas as pd

from conf import (
    CUSTOM_SETTINGS,
    REDIS_HOST,
    REDIS_PORT
)

from data_publisher import init_redis_data_publisher
from utils.processing_utils import (
    construct_windows,
    correct_row,
    read_head,
    read_new_data,
    standardize
)
from utils.utils import init_logger, get_files


def launch_processing():
    logger = init_logger()
    monitor_directory = CUSTOM_SETTINGS["inference_config"]["data_processing"]["monitor_directory"]

    delay = CUSTOM_SETTINGS["inference_config"]["data_processing"]["reading_delay"]

    window_size = CUSTOM_SETTINGS["pre_processing_config"]["seq_len"] *\
        CUSTOM_SETTINGS["pre_processing_config"]["frequency"]

    data_publisher = init_redis_data_publisher(
        REDIS_HOST,
        REDIS_PORT,
        CUSTOM_SETTINGS["dataset_config"]["modality"],
        logger
    )

    processed_files = set()
    # infinite loop:
    #   checks if there are new files (sessions) that have not been processed before
    while True:
        # Get the list of files with the specified suffix
        files = get_files(
            monitor_directory,
            CUSTOM_SETTINGS["inference_config"]["data_processing"]["modality_suffix"]
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
            last_position = 0
            buffer = []
            while True:
                file_size = os.stat(filename).st_size
                new_data, last_position = read_new_data(filename, last_position)

                # pre-processing and segmentation into time-windows of the specified length and frequency
                if new_data:
                    # Pre-processing logic from SUPSI
                    # One of the columns might have 2 values due to the error in Magic XRoom
                    if len(new_data[0]) == 21:
                        new_data = [row + [np.nan] for row in new_data]
                    new_data = pd.DataFrame(new_data, dtype=str, columns=header)
                    new_data = new_data.apply(correct_row, axis=1).astype(float)
                    new_data['head_rotW'].fillna(new_data['head_rotW'].mean(), inplace=True)
                    new_data['lcontroller_rotW'].fillna(new_data['lcontroller_rotW'].mean(), inplace=True)

                    # drop the timestamp from input data
                    new_data = np.array(new_data.iloc[:, 1:]).astype(float)

                    windows, buffer = construct_windows(new_data, window_size, buffer)

                    if windows is not None:
                        logging.info(
                            f"Number of extracted windows: {len(windows)}. Dimensionality: {windows.shape[1:]}. "
                            f"Buffer size {len(buffer)}"
                        )
                        # use normalize function with running parameters instead of StandardScaler
                        preprocessed, running_params = standardize(windows, running_params)

                        logging.info(
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

        time.sleep(CUSTOM_SETTINGS["inference_config"]["data_processing"]["reading_delay"])


if __name__ == "__main__":
    launch_processing()
