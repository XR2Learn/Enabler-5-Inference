import time
import csv
import os
import pathlib


def read_csv(input_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            yield row


def write_csv(output_file, data):
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


def main(input_file, output_file):
    if not os.path.exists(output_file):
        # Create the output CSV file if it does not exist
        with open(output_file, 'w', newline='') as _:
            pass  # Empty file

    for row in read_csv(input_file):
        if row:
            write_csv(output_file, row)
            time.sleep(0.1)  # Reading delay


if __name__ == "__main__":
    input_files = [
        "./datasets/XRoom/V1_BK1/data_collection_638461160214938655_SHIMMER_.csv",
        "./datasets/XRoom/V1_BK1/data_collection_638461162609580603_SHIMMER_.csv"
    ]
    MAIN_FOLDER_DEFAULT = pathlib.Path(__file__).parent.parent.absolute()
    output_folder = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'test_data')
    # output_folder = './test_data'
    output_files = [os.path.join(output_folder, os.path.basename(input_file)) for input_file in input_files]
    for input_file, output_file in zip(input_files, output_files):
        main(input_file, output_file)
