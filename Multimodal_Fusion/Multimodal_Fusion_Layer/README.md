# Multimodal Fusion Component

Component to combine multiple modalities predictions into a single final prediction.

`Input`: .npy file with predicted emotions labels.

- Input file should be in the folder `/outputs/prediction-<input-type>/<npy_file>.npy`

`Output`: CSV file with the final prediction.

- Output file with the final prediction is saved in the folder `/outputs/predictions.csv`

# Installation (Local Run)
1. Clone or download the repository
2. Prepare your virtual environment (e.g. VirtualEnv, Pip env, Conda)
3. Install requirements

`pip install -r requirements.txt`

## Configuration

1. For local run create `configuration.json` file in the same level as `example.configuration.json`.

## Environment Variables (Env Vars)

1. For local run create `.env` file in the same level as `example.env` to load environment variables.
2. Add on docker compose the environment variable name under the service name
