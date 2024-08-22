import csv
import re

import numpy as np


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


def standardize(bt_segments: np.ndarray, running_params=None):
    """
    z-normalization to zero mean and unit variance for each segment with body-tracking

    Args:
        bt_segment: 3D-array containing body-tracking signals from one session (multiple segments per session)

    Returns:
        standardized_signal: a list containing the standardized numpy arrays with audio from a subject
    """

    # stack segments for the whole session and compute per-channel statistics
    stacked_segments = bt_segments.reshape(-1, bt_segments.shape[-1])
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

    bt_z_normalized = (bt_segments - mean_channel) / std_channel

    return (
        bt_z_normalized,
        {
            "sum": sum_channel,
            "sum_square": sum_square_channel,
            "count": count_ts if running_params is not None else len(stacked_segments)
        }
    )


def correct_row(row):
    head_rotW_value = str(row['head_rotW'])
    matched_values = re.findall(r'-?\d+\.\d+|-?\d+', head_rotW_value)
    if len(matched_values) > 1:
        row['head_rotW'] = matched_values[0]
        row['rcontroller_rotW'] = row['lcontroller_rotW']
        row['lcontroller_rotW'] = matched_values[1]
    elif len(matched_values) == 1:
        row['head_rotW'] = matched_values[0]
    return row
