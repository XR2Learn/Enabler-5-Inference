# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## Added

- "modality" configuration required in addition to dataset name, under "dataset_configuration"
  on ```configuration.json``` file.
- MLP encoder for the audio modality (intended for eGeMAPS).
- Emotion classification component architecture for the body tracking modality.

## Changed

- Bio-Measurements (BM) dataset name changed to 'XRoom', for data coming
  from [Magic XRoom](https://github.com/XR2Learn/magic-xroom).
- Output directory structure: Two additional directory levels to indicate the dataset/modality that produced the
  output (changes applied for audio and BM modalities).
- Output directory structure for reading trained encoders and models, according to Training Tools structure.

## Fixed

- Multimodal Layer component now supports BM modality (when acting as a publisher).

## [0.4.0] - 2024 - 03 - 12

### Added

- Support for bio-measurements (BM) modality using data format
  by [Magic XRoom](https://github.com/XR2Learn/magic-xroom).
- License Update to Apache 2.0.

### Known Issues

- Multimodal Layer component does not support BM modality (when acting as a publisher).

## [0.3.1] - 2024 - 02 - 15

### Added

- License
- More documentation

## [0.3.0] - 2024 - 01 - 19

### Added

- Multimodal fusion layer can be a publisher, i.e., the predicted emotion will be published to be read by
  the personalisation tool (optional).

## [0.2.0] - 2024 - 01 - 09

### Added

- End2End mode: use fine-tuned model (encoder + classifier) to make predictions from pre-processed data
- Unit tests for audio prediction components and github workflow for automated testing

### Changed

- Inference config structure. Required field: "mode"

## [0.1.1] - 2023 - 12 - 06

### Added

- Docker compose receives EXPERIMENT_ID and CONFIG_FILE_PATH as env vars, with default values.
- More Documentation.

## [0.1.0] - 2023 - 10 - 27

### Added

- Basic version of Enabler 5 and related components for audio modality using RAVDESS dataset, supporting:
    - Evaluation metrics: confusion matrix, accuracy and f1-scores.
    - Multimodal fusion (Enabler 5): implementing late fusion (decision level fusion).
- Changelog

<!-- 
Example of Categories to use in each release

### Added
- Just an example of how to use changelog.

### Changed
- Just an example of how to use changelog.

### Fixed
- Just an example of how to use changelog.

### Removed
- Just an example of how to use changelog.

### Deprecated
- Just an example of how to use changelog. -->


[unreleased]: https://github.com/um-xr2learn-enablers/XR2Learn-Inference/compare/v0.4.0...master

[0.1.0]: https://github.com/um-xr2learn-enablers/XR2Learn-Inference/releases/tag/v0.1.0

[0.1.1]: https://github.com/um-xr2learn-enablers/XR2Learn-Inference/releases/tag/v0.1.1

[0.2.0]: https://github.com/um-xr2learn-enablers/XR2Learn-Inference/releases/tag/v0.2.0

[0.3.0]: https://github.com/um-xr2learn-enablers/XR2Learn-Inference/releases/tag/v0.3.0

[0.3.1]: https://github.com/um-xr2learn-enablers/XR2Learn-Inference/releases/tag/v0.3.1

[0.4.0]: https://github.com/um-xr2learn-enablers/XR2Learn-Inference/releases/tag/v0.4.0