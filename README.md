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

This folder contains the real-data file currently included in the public repo.

- `item_contents_prompt_dedup.csv`
  Prompt-deduplicated item-level content used in the MMLU application.

### `results/`

This folder is reserved for generated outputs from the project, such as:

- trait estimates
- ranking tables
- comparison summaries
- figures

At the moment, it is kept as a clean placeholder in this version of the repo.

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
