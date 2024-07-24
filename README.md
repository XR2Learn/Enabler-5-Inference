# [XR2Learn Personalization Enablers] Inference Tools

These tools are designed for the unimodal and multimodal emotion classification and evaluation in XR2Learn. This set of
tools includes Emotion Classification (per modality), Multimodal Fusion and Evaluation components. Each tool is a
modularized component with an isolated environment and dependencies that can be used separately, in combination, or as
an end-to-end system (together with the Command-Line Interface – CLI).

Each component is deployed using Docker to ensure easy-to-use components, reproducible development and deployment
environments, and consistent results.

**Pre-processing**: a component to process raw input data into format and time windows to be used as input for the **Emotion Classification** component.

**Emotion Classification**: a component to recognize emotions. Each modality has a separated emotion classification
component.

**Multimodal Fusion**: a component to execute a decision-level emotion detection multimodal fusion, i.e., to compute the
combination of emotions from different modalities.

**Evaluation**: a component to evaluate a uni/multimodal emotion detection model according to different evaluation
metrics.

## Dependencies

- Docker
- Python 3.10

## Basic User Manual

The enablers were designed to be used with [Enablers-CLI](https://github.com/XR2Learn/Enablers-CLI), a command-line
interface that simplifies the use of enablers,
so the easiest way to access the enablers’ functionalities is by using Enablers-CLI.

However, if changing or expanding the enablers’ functionalities is required, it is possible to access each component
using docker commands, as exemplified below. Thus, the instructions described below are focused on running the enablers
for a development environment.

A `configuration.json` file is required to provide the enablers with the necessary specifications for running. A default
version of “configuration.json” is provided and can be changed by the user.

1. Run a docker image:

   `docker compose run --rm <<service-name>>`

**Note 1**: Service names can be found in the “docker-compose.yml” file in the project’s root folder. Each modality,
i.e., audio, bio-measurements (bm), body movements, are deployed in separated docker containers and their service name
follow the structure:

1. emotion-classification-MODALITY
2. fusion-layer
3. ed-evaluation

**Note 2**: Some additional services can be found in the Inference domain “docker-compose.yml” file, namely:

- redis
- personalisation-tool
- demo-ui

These services are present in the Inference domain to facilitate development and are explained in detail in the
following section, Personalization Tool.

There is an additional script to run all the docker images from a given modality, which will use the available
`configuration.json` file:

1. For Unix-based OS, MacOS and Linux:

`./run_all_dockers.sh`

2. For Windows:

`./run_all_dockers.ps1`

## Additional Useful Commands:

1. Build a specific docker image

`docker compose build <service-name>`

3. Run a specific docker image

`docker compose run --rm <service-name>`

5. Run a specific docker image providing EnvVars in the format KEY=VALUE:

`KEY=VALUE docker compose run --rm <service-name>`

7. Run a specific docker image with shell entry point

`docker compose run --rm <service-name> \bin\bash`

All the outputs produced by any component in the Inference domain are saved and can be accessed in the
folder `./outputs`.

## License

Copyright © 2024, Maastricht University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Pre-trained and fine-tuned models created using the RAVDESS dataset are shared under
the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license to
comply with the RAVDESS license, as the models are derivative works from this dataset.

Please refer to [LICENSE.md](LICENSE.md) document for more details.

## Changelog

A description of the main changes in the project’s versions can be found at [CHANGELOG.md].

[CHANGELOG.md]: https://github.com/um-xr2learn-enablers/XR2Learn-Inference/blob/master/CHANGELOG.md
