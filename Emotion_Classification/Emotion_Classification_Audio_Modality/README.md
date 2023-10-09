# Emotion Classification - Audio Modality

Component to run inference - classify examples using audio data input type.

`Input`: CSV file with path to features files + timestamp.

- CSV Input file should be in the folder `/outputs/predict/<input_CSV_file.csv>`

`Output`: File with emotions identified.

CSV file with labels + path to features files + timestamp.

- Output file with predicted emotions is saved in the folder `/outputs/predict/<output_CSV_file.csv>`

# Development

## Development Environment Setup

1. Clone the repository
2. Prepare your virtual environment (e.g. VirtualEnv, Pip env, Conda)

## Development cycle (simplified version)

1. Crete a new branch to develop the task
2. Commit to remote to the new branch as needed
3. After finishing developing test docker image
4. When is everything done and tested, merge task branch to master branch

## Development Notes


## Configuration

1. For local run create `configuration.json` file in the same level as `example.configuration.json`

## Environment Variables (Env Vars)

1. For local run create `.env` file in the same level as `example.env` to load environment variables.
2. Add on docker compose the environment variable name under the service `ssl-audio`

# TODO
