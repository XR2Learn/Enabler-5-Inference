import logging
import os


def get_files(directory, suffix):
    return [filename for filename in os.listdir(directory) if filename.endswith(suffix)]


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
