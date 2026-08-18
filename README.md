# LLM Benchmarking via Representation Multi-task Learning

This repository is the clean public-facing version of our case study on benchmarking LLMs with a representation multi-task learning framework.

## Repository structure

The repo currently uses a simple four-folder layout:

```text
LLM-Benchmarking-via-Representation-Multi-task-Learning/
├── application/
├── real_data/
├── results/
└── simulation/
```

### `application/`

This folder contains the main code for the real-data application.

- `run_mmlu_real_data_ai_measurement.py`
  Main Python script for the real-data estimation pipeline.
- `mmlu_initialization.R`
  R script for initialization and setup checks for the MMLU application.

### `real_data/`

This folder contains the public real-data inputs used by the application.

- `item_contents.csv`
  Item-level content file for the benchmark items.
- `item_level_matrix.part01.csv`
  First half of the model-by-item binary response matrix.
- `item_level_matrix.part02.csv`
  Second half of the model-by-item binary response matrix.
- `metadata.csv`
  Model-level metadata used in the application analysis.

### `results/`

This folder contains generated outputs from the project, such as:

- trait estimates
- ranking tables
- comparison summaries
- figures

For example, the `Domain-specific traits/` subfolder stores subject-by-subject CSV outputs with:

- proposed domain-specific trait and rank
- local domain-specific trait and rank
- subject accuracy on a 0-100 scale and its rank

### `simulation/`

This folder is reserved for simulation code and simulation outputs tied to the methodological part of the project.

At the moment, it is also kept as a clean placeholder in this version of the repo.

## Design principle

The structure is intentionally minimal:

- put empirical and modeling scripts in `application/`
- put public input data in `real_data/`
- put generated outputs in `results/`
- put simulation work in `simulation/`

This keeps the repository easy to read, easy to maintain, and easy to extend without adding extra nested README files or unnecessary subfolder complexity.
