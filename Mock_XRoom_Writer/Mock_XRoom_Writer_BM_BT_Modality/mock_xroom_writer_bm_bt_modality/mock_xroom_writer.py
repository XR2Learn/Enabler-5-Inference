import time
import csv
import itertools
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


def main(input_files, output_files):
    for output_file in output_files:
        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='') as _:
                pass  # Empty file

    for row1, row2 in tuple(itertools.zip_longest(read_csv(input_files[0]), read_csv(input_files[1]))):
        if row1:
            write_csv(output_files[0], row1)
        if row2:
            write_csv(output_files[1], row2)
        time.sleep(0.1)  # Reading delay


if __name__ == "__main__":
    input_files = [
        ["./datasets/XRoom/P7_M/data_collection_638677960757871436_VR_.csv",
         "./datasets/XRoom/P7_M/data_collection_638677960757871436_SHIMMER_.csv"],
        ["./datasets/XRoom/P7_M/data_collection_638677962396853370_VR_.csv",
         "./datasets/XRoom/P7_M/data_collection_638677962396853370_SHIMMER_.csv"]
    ]
    MAIN_FOLDER_DEFAULT = pathlib.Path(__file__).parent.parent.absolute()
    output_folder = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'test_data')
    output_files = [
        [
            os.path.join(output_folder, os.path.basename(input_file[i]))
            for i in range(len(input_file))
        ] for input_file in input_files
    ]
    for input_file, output_file in zip(input_files, output_files):
        main(input_file, output_file)
