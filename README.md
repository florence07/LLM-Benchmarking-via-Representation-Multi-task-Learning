# LLM Benchmarking via Representation Multi-task Learning

This repository is about the benchmarking LLMs with a representation multi-task learning framework. It collects the main code, real-data inputs, and exported result tables for the MMLU application in the paper.

## Repository structure

The repo currently uses a four-folder layout:

```text
LLM-Benchmarking-via-Representation-Multi-task-Learning/
├── application/
├── real_data/
├── results/
└── simulation/
```

### `application/`

This folder contains the code used to fit the methods in the paper on the MMLU item-response matrix.

- `run_mmlu_real_data_ai_measurement.py`
  Main Python script for the real-data estimation pipeline. In the paper's notation, this is the driver that fits the proposed estimator, the global benchmark, and related baselines on the model-by-item binary response matrix.
- `mmlu_initialization.R`
  R script that reproduces and inspects the initialization procedure used before optimization.

### `real_data/`

This folder contains the real-data inputs used by the application section of the paper.

- `item_contents.csv`
  Item-level content file. It maps each benchmark item to its subject.
- `item_response_matrix.part01.csv`
  First half of the model-by-item binary response matrix.
- `item_response_matrix.part02.csv`
  Second half of the model-by-item binary response matrix.
- `item_index_map.csv`
  Mapping file that links rows in `item_contents.csv` to the item-column names used in the response matrix files.
- `metadata.csv`
  Model-level metadata used for model descriptions and follow-up analysis.

### `results/`

This folder contains exported outputs that correspond directly to the paper's ranking objects.

The key files are:

- `general_results.csv`
  One row per model. This is the main general-ranking summary table and contains:
  - the proposed general trait estimate `theta_G`
  - the proposed general ranking
  - the global benchmark general trait estimate `theta_G`
  - the global benchmark ranking
  - the raw overall accuracy score on a 0-100 scale
  - the raw accuracy ranking

- `Domain-specific traits/`
  One CSV per subject. Each file stores, for every model:
  - the proposed domain-specific trait estimate `theta_t` and its ranking
  - the local benchmark domain-specific trait estimate and its ranking
  - the subject-specific raw accuracy on a 0-100 scale and its ranking

This matches the paper's comparison logic:

- `proposed`
  The representation multi-task learning estimator, which jointly estimates subject traits and a general trait.
- `global`
  The shared-general-trait benchmark that forces one common trait across subjects.
- `local`
  The subject-by-subject benchmark that fits each subject separately. In this repo it is used for domain-specific comparisons, not for the main general summary.
- `accuracy`
  The direct empirical score benchmark computed from the binary response matrix.

### `simulation/`

This folder is reserved for the simulation side of the project. In the current public repo it is only kept as a placeholder.


## Reading Guide

If you want to understand the application section quickly:

1. Start with `results/general_results.csv` to see how the proposed general ranking differs from the global benchmark and from raw overall accuracy.
2. Then open a few files in `results/Domain-specific traits/` to see how the proposed subject-specific traits compare with the local benchmark and raw subject accuracy.
3. Use `real_data/item_contents.csv` together with `real_data/item_index_map.csv` to map response-matrix columns back to benchmark items and subjects.
4. Use `application/run_mmlu_real_data_ai_measurement.py` if you want to trace how the estimates were produced.
