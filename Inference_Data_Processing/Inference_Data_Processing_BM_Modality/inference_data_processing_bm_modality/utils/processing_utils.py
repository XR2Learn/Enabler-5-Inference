import csv

import numpy as np
import scipy


def read_head(filename):
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
    return header


def read_new_data(filename, last_position):
    """ Read all data available in the file starting from the last position (line number) provided

        Args:
            filename: file to open
            last_position: row number of the position to read from

        Returns:
            new_data: data read from the file
            last_position: last recorded position in the file
    """
    new_data = []
    with open(filename, 'r') as file:
        file.seek(last_position)
        new_chunk = file.read()
        if new_chunk:
            reader = csv.reader(new_chunk.splitlines())
            # skip header
            if last_position == 0:
                next(reader)
            for row in reader:
                if row:
                    new_data.append(row)

        # Update last position
        last_position = file.tell()

    return new_data, last_position


def construct_windows(data, window_size, buffer=[]):
    if len(buffer) != 0:
        data = np.concatenate([buffer, data], axis=0)

    if len(data) >= window_size:
        windows = np.array(
            [np.array(
                data[i: i + window_size]
            ) for i in range(0, len(data) - window_size + 1, window_size)]
        )
        buffer = data[len(windows) * window_size:]
    else:
        windows = None
        buffer = data

    return windows, buffer


def normalize(bm_segments, running_params=None):
    """
    normalize: transformed into a range between -1 and 1 by normalization for each speaker (min-max scaling)

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with audio from a subject
    """

    # stack segments for the whole session and compute per-channel statistics
    stacked_segments = bm_segments.reshape(-1, bm_segments.shape[-1])
    min_channel = stacked_segments.min(axis=0)
    max_channel = stacked_segments.max(axis=0)
    if running_params is not None:
        min_channel = np.stack([min_channel, running_params["min"]]).min(axis=0)
        min_channel = np.stack([min_channel, running_params["max"]]).max(axis=0)

    segments_min_max = (bm_segments - min_channel) / (max_channel - min_channel)

    return (
        segments_min_max,
        {
            "min": min_channel,
            "max": max_channel
        }
    )


def standardize(bm_segments: np.ndarray, running_params=None):
    """
    z-normalization to zero mean and unit variance for each segment with bio-measurements

    Args:
        bm_segment: 3D-array containing bio-measurement signals from one session (multiple segments per session)

    Returns:
        standardized_signal: a list containing the standardized numpy arrays with audio from a subject
    """

    # stack segments for the whole session and compute per-channel statistics
    stacked_segments = bm_segments.reshape(-1, bm_segments.shape[-1])
    sum_channel = stacked_segments.sum(axis=0)
    sum_square_channel = np.power(stacked_segments, 2).sum(axis=0)

    if running_params is not None:
        sum_channel += running_params["sum"]
        sum_square_channel += running_params["sum_square"]
        count_ts = running_params["count"] + len(stacked_segments)
        mean_channel = sum_channel / count_ts
        std_channel = np.sqrt((sum_square_channel / count_ts) - (np.power(mean_channel, 2)))
    else:
        mean_channel = stacked_segments.mean(axis=0)
        std_channel = stacked_segments.std(axis=0)

    bm_z_normalized = (bm_segments - mean_channel) / std_channel

    return (
        bm_z_normalized,
        {
            "sum": sum_channel,
            "sum_square": sum_square_channel,
            "count": count_ts if running_params is not None else len(stacked_segments)
        }
    )


def no_preprocessing(bm_segment, running_params=None):
    """
    No pre-processing applied

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
    """
    return bm_segment, None


def resample_bm(
        bm_segment: np.ndarray,
        sample_rate: int,
        target_rate: int
):
    """
    Resample stacked signals to a target frequency

    Args:
        bm_segment: 3D numpy array with the bio-measurement data to resample
        sample_rate: the sample rate of the original signal
        target_rate: the target sample rate
    Returns:
        resampled: resampled signals
    """
    number_of_samples = round(len(bm_segment) * float(target_rate) / sample_rate)
    resampled = scipy.signal.resample(bm_segment, number_of_samples, axis=0)
    return resampled
