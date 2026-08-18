#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


_CV_PROCESS_STATE = None
_FULL_DATA_PROCESS_STATE = None
_POST_CV_PROCESS_STATE = None
_GLOBAL_PROCESS_STATE = None
ESTIMATOR_VERSION = "paper_consistent_subject_weighting_score_logistic_global_init_v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AI-measurement unequal-J real-data analysis on MMLU matrices."
    )
    parser.add_argument(
        "--matrix-dir",
        default="real_data/raw/harness_hendrycksTest_5",
        help="Directory containing the prompt-deduplicated MMLU matrix files.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional legacy per-subject matrix summary CSV.",
    )
    parser.add_argument(
        "--matrix-parts",
        nargs="+",
        default=[
            "real_data/raw/harness_hendrycksTest_5/"
            "item_level_matrix_prompt_dedup.part01.csv",
            "real_data/raw/harness_hendrycksTest_5/"
            "item_level_matrix_prompt_dedup.part02.csv",
        ],
        help="Prompt-deduplicated model-by-item CSV parts.",
    )
    parser.add_argument(
        "--item-metadata-csv",
        default=(
            "real_data/metadata/"
            "item_contents_prompt_dedup.csv"
        ),
        help="Prompt-deduplicated item metadata with a subject column.",
    )
    parser.add_argument(
        "--item-index-map-csv",
        default=(
            "real_data/metadata/"
            "item_prompt_dedup_index_map.csv"
        ),
        help="Mapping that aligns matrix column names with deduplicated items.",
    )
    parser.add_argument(
        "--python-script",
        default="simulation/code/compare_five_estimators_simulation_AI_item_cv.py",
        help="Path to the latest AI measurement implementation.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/estimation/reproduced_full_data_grid",
        help="Directory where outputs will be written.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional list of CV seeds. If omitted, the script runs two seeds: "
            "--seed and --seed+1."
        ),
    )
    parser.add_argument("--theta-bound", type=float, default=4.0)
    parser.add_argument("--a-lower-bound", type=float, default=0.0)
    parser.add_argument("--a-bound", type=float, default=4.0)
    parser.add_argument("--d-bound", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--learning-rate-decay-factor",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--learning-rate-decay-steps",
        type=int,
        default=200,
    )
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--c-lambda-grid",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional C_lambda grid shared by CV mode and --full-data-grid. "
            "If omitted, the script uses the default grid."
        ),
    )
    parser.add_argument(
        "--cv-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of concurrent candidate fits.",
    )
    parser.add_argument(
        "--full-data-grid",
        action="store_true",
        help=(
            "Skip cross-validation and fit every C_lambda candidate on the "
            "complete real-data matrix, followed by full-data local and global fits."
        ),
    )
    parser.add_argument(
        "--proposed-only",
        action="store_true",
        help=(
            "Run only the proposed estimator's CV/grid selection, skipping "
            "the full-data proposed refit and local/global/naive baselines."
        ),
    )
    return parser.parse_args()


def load_ai_measurement_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(
        "ai_measurement_unequal_j", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_as_domain_blocks(module, Y):
    converter = getattr(module, "_as_domain_blocks", None)
    if converter is not None:
        return converter(Y)
    legacy_converter = getattr(module, "_as_J_blocks", None)
    if legacy_converter is not None:
        return legacy_converter(Y)
    raise AttributeError(
        "Loaded module does not expose _as_domain_blocks or _as_J_blocks."
    )


def module_prepare_observation_masks(module, Y_blocks, observation_mask):
    mask_preparer = getattr(module, "_prepare_observation_masks", None)
    if mask_preparer is not None:
        return mask_preparer(Y_blocks, observation_mask)
    legacy_preparer = getattr(module, "_prepare_observation_mask", None)
    if legacy_preparer is not None:
        return legacy_preparer(Y_blocks, observation_mask)
    raise AttributeError(
        "Loaded module does not expose _prepare_observation_masks or "
        "_prepare_observation_mask."
    )


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def atomic_to_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pd.to_pickle(obj, tmp_path)
    tmp_path.replace(path)


def slim_pickle_payload(obj):
    """Drop bulky transient fields before checkpointing to disk."""
    if isinstance(obj, dict):
        dropped_keys = {
            "history",
            "cv_trace_records",
        }
        return {
            key: slim_pickle_payload(value)
            for key, value in obj.items()
            if key not in dropped_keys
        }
    if isinstance(obj, list):
        return [slim_pickle_payload(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(slim_pickle_payload(value) for value in obj)
    return obj


def atomic_checkpoint_pickle(obj, path: Path) -> None:
    atomic_to_pickle(slim_pickle_payload(obj), path)


def load_pickle_checkpoint(path: Path):
    if not path.exists():
        return None
    print(f"Resuming from checkpoint: {path.name}")
    return pd.read_pickle(path)


def validate_resume_config(current: pd.DataFrame, path: Path) -> None:
    """Refuse to combine checkpoints produced by different estimations."""
    if not path.exists():
        return
    previous = pd.read_csv(path, keep_default_na=False)
    ignored = {"cv_workers", "output_dir"}
    columns = [column for column in current.columns if column not in ignored]
    if len(previous) != 1 or any(column not in previous.columns for column in columns):
        raise RuntimeError(
            f"Existing resume configuration is incompatible: {path}. "
            "Use a different --output-dir for a new run."
        )
    mismatches = []
    for column in columns:
        new_value = current.iloc[0][column]
        old_value = previous.iloc[0][column]
        if isinstance(new_value, (float, np.floating)):
            equal = bool(np.isclose(float(old_value), float(new_value)))
        else:
            equal = str(old_value) == str(new_value)
        if not equal:
            mismatches.append(column)
    if mismatches:
        raise RuntimeError(
            "Existing checkpoints were created with different settings for: "
            f"{', '.join(mismatches)}. Use a different --output-dir for a new run."
        )


def read_subject_matrix(path: Path) -> dict:
    dat = pd.read_csv(path)
    if "source" not in dat.columns:
        raise ValueError(f"Matrix file {path} does not contain a 'source' column.")
    return {
        "models": dat["source"].tolist(),
        "Y": dat.drop(columns=["source"]).to_numpy(dtype=np.int8),
        "item_names": dat.columns[1:].tolist(),
    }


def load_mmlu_blocks(summary_csv: Path) -> tuple[list[dict], list[str]]:
    summary_df = pd.read_csv(summary_csv)
    required = {"subject", "matrix_path"}
    missing = required.difference(summary_df.columns)
    if missing:
        raise ValueError(
            f"Summary CSV is missing required columns: {', '.join(sorted(missing))}."
        )

    blocks = []
    for row in summary_df.itertuples(index=False):
        matrix_info = read_subject_matrix(Path(row.matrix_path))
        blocks.append(
            {
                "subject": row.subject,
                "matrix_path": row.matrix_path,
                "models": matrix_info["models"],
                "Y": matrix_info["Y"],
                "item_names": matrix_info["item_names"],
            }
        )

    common_models = sorted(set.intersection(*(set(block["models"]) for block in blocks)))
    if not common_models:
        raise ValueError("No common models were found across the MMLU subject matrices.")

    aligned_blocks = []
    for block in blocks:
        row_index = pd.Index(block["models"]).get_indexer(common_models)
        if np.any(row_index < 0):
            raise ValueError(f"Failed to align common models for subject {block['subject']}.")
        aligned_blocks.append(
            {
                **block,
                "models": common_models,
                "Y": block["Y"][row_index, :],
            }
        )

    return aligned_blocks, common_models


def load_mmlu_matrix_parts(
    matrix_parts: list[Path],
    item_metadata_csv: Path,
    item_index_map_csv: Path,
) -> tuple[list[dict], list[str]]:
    item_metadata = pd.read_csv(item_metadata_csv, usecols=["subject"])
    index_map = pd.read_csv(item_index_map_csv)
    required_map_columns = {"new_row_index", "old_col_name"}
    missing = required_map_columns.difference(index_map.columns)
    if missing:
        raise ValueError(
            "Item index map is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )
    expected_rows = np.arange(len(index_map))
    if not np.array_equal(index_map["new_row_index"].to_numpy(), expected_rows):
        raise ValueError("Item index map new_row_index must be consecutive and ordered.")
    if len(item_metadata) != len(index_map):
        raise ValueError("Item metadata and prompt-dedup index map have different lengths.")

    expected_item_columns = index_map["old_col_name"].astype(str).tolist()
    model_ids: list[str] = []
    response_parts: list[np.ndarray] = []
    response_dtypes = {column: np.int8 for column in expected_item_columns}
    for path in matrix_parts:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if not columns or columns[0] != "model_id":
            raise ValueError(f"Matrix part {path} must start with a model_id column.")
        if columns[1:] != expected_item_columns:
            raise ValueError(
                f"Matrix part {path} columns do not match the prompt-dedup index map."
            )
        part = pd.read_csv(path, dtype=response_dtypes)
        part_models = part.pop("model_id").astype(str).tolist()
        responses = part.to_numpy(dtype=np.int8, copy=False)
        if not np.isin(responses, [0, 1]).all():
            raise ValueError(f"Matrix part {path} contains responses outside 0/1.")
        model_ids.extend(part_models)
        response_parts.append(responses)

    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Matrix parts contain duplicate model_id values.")
    response_matrix = np.concatenate(response_parts, axis=0)
    subject_values = item_metadata["subject"].astype(str).to_numpy()
    subjects = item_metadata["subject"].drop_duplicates().astype(str).tolist()
    matrix_label = ";".join(path.as_posix() for path in matrix_parts)
    blocks = []
    for subject in subjects:
        item_positions = np.flatnonzero(subject_values == subject)
        blocks.append(
            {
                "subject": subject.replace(" ", "_"),
                "matrix_path": matrix_label,
                "models": model_ids,
                "Y": response_matrix[:, item_positions],
                "item_names": [expected_item_columns[index] for index in item_positions],
            }
        )
    return blocks, model_ids


def drop_miscellaneous_blocks(blocks: list[dict]) -> list[dict]:
    filtered_blocks = [
        block
        for block in blocks
        if str(block["subject"]).replace(" ", "_").strip().lower() != "miscellaneous"
    ]
    if not filtered_blocks:
        raise ValueError("All subject blocks were removed after excluding miscellaneous.")
    if len(filtered_blocks) == len(blocks):
        print("No miscellaneous subject block found; keeping all loaded subjects.")
    else:
        print(
            "Excluded miscellaneous subject block before fitting: "
            f"{len(blocks)} -> {len(filtered_blocks)} subjects."
        )
    return filtered_blocks


def filter_constant_items(blocks: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    filtered_blocks = []
    manifest_rows = []

    for block in blocks:
        Y = block["Y"]
        item_names = np.asarray(block["item_names"], dtype=object)
        col_sums = Y.sum(axis=0)
        n_models = Y.shape[0]
        keep_mask = np.logical_and(col_sums > 0, col_sums < n_models)

        removed_all_zero = int(np.sum(col_sums == 0))
        removed_all_one = int(np.sum(col_sums == n_models))
        kept_items = int(np.sum(keep_mask))

        filtered_blocks.append(
            {
                **block,
                "Y": Y[:, keep_mask],
                "item_names": item_names[keep_mask].tolist(),
            }
        )
        manifest_rows.append(
            {
                "subject": block["subject"],
                "models": int(n_models),
                "items_before_constant_filter": int(Y.shape[1]),
                "all_zero_items_removed": removed_all_zero,
                "all_one_items_removed": removed_all_one,
                "items_after_constant_filter": kept_items,
            }
        )

    return filtered_blocks, pd.DataFrame(manifest_rows)


def result_summary_row(method: str, result: dict) -> dict:
    return {
        "method": method,
        "lambda_value": float(result["lambda_value"]) if "lambda_value" in result else np.nan,
        "selected_C_lambda": float(result["selected_C_lambda"])
        if "selected_C_lambda" in result
        else np.nan,
        "objective": float(result["objective"]),
        "negative_log_likelihood": float(result["negative_log_likelihood"]),
        "penalty": float(result["penalty"]),
        "cv_mean_test_negative_log_likelihood": float(
            result["cv_mean_test_negative_log_likelihood"]
        )
        if "cv_mean_test_negative_log_likelihood" in result
        else np.nan,
        "cv_fold_standard_error": float(result["cv_fold_standard_error"])
        if "cv_fold_standard_error" in result
        else np.nan,
        "cv_n_folds": int(result["cv_n_folds"]) if "cv_n_folds" in result else np.nan,
        "cv_runtime_seconds": float(result["cv_runtime_seconds"])
        if "cv_runtime_seconds" in result
        else np.nan,
    }


def raw_mean_training_negative_log_likelihood(result: dict) -> float:
    raw_nll = float(
        result.get("raw_negative_log_likelihood", result["negative_log_likelihood"])
    )
    return raw_nll / float(result["total_entries"])


def theta_t_long(
    theta_hat: np.ndarray,
    model_ids: list[str],
    subjects: list[str],
    value_column: str = "theta_t_hat",
) -> pd.DataFrame:
    rows = []
    for subject, values in zip(subjects, theta_hat):
        rows.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "model_id": model_ids,
                    value_column: np.asarray(values, dtype=float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def item_parameter_long(
    result: dict,
    subjects: list[str],
    a_key: str = "a_hat",
    d_key: str = "d_hat",
) -> pd.DataFrame:
    rows = []
    a_hat = np.asarray(result[a_key])
    d_hat = np.asarray(result[d_key])
    J_vec = np.asarray(result["J_vec"], dtype=int)
    for subject, a_row, d_row, J_t in zip(subjects, a_hat, d_hat, J_vec):
        active = np.arange(int(J_t))
        rows.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "item_index": active + 1,
                    a_key: a_row[: int(J_t)].astype(float),
                    d_key: d_row[: int(J_t)].astype(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def maybe_write_records(result: dict, key: str, out_path: Path) -> None:
    if key not in result:
        return
    atomic_to_csv(pd.DataFrame(result[key]), out_path)


def maybe_write_pickle(result: dict, out_path: Path) -> None:
    if result is None:
        return
    atomic_checkpoint_pickle(result, out_path)


def pad_parameter_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    J_max = max(block.size for block in blocks)
    padded = np.zeros((len(blocks), J_max), dtype=float)
    for index, block in enumerate(blocks):
        padded[index, : block.size] = np.asarray(block, dtype=float)
    return padded


def write_initialization_bundle(
    initialization,
    output_prefix: Path,
    model_ids: list[str],
    subjects: list[str],
) -> None:
    theta_init, a_init_blocks, d_init_blocks, theta_G_init = initialization
    init_result = {
        "initial_theta_hat": np.asarray(theta_init, dtype=float),
        "initial_a_hat": pad_parameter_blocks(
            [np.asarray(block, dtype=float) for block in a_init_blocks]
        ),
        "initial_d_hat": pad_parameter_blocks(
            [np.asarray(block, dtype=float) for block in d_init_blocks]
        ),
        "initial_theta_G": np.asarray(theta_G_init, dtype=float),
        "J_vec": np.asarray([len(block) for block in a_init_blocks], dtype=int),
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(
        pd.DataFrame(
            {
                "model_id": model_ids,
                "initial_theta_G": np.asarray(
                    init_result["initial_theta_G"], dtype=float
                ),
            }
        ),
        output_prefix.with_name(output_prefix.name + "_initial_theta_G.csv"),
    )
    atomic_to_csv(
        theta_t_long(
            np.asarray(init_result["initial_theta_hat"], dtype=float),
            model_ids,
            subjects,
            value_column="initial_theta_t_hat",
        ),
        output_prefix.with_name(output_prefix.name + "_initial_theta_t_long.csv"),
    )
    atomic_to_csv(
        item_parameter_long(
            init_result,
            subjects,
            a_key="initial_a_hat",
            d_key="initial_d_hat",
        ),
        output_prefix.with_name(output_prefix.name + "_initial_item_parameters.csv"),
    )


def write_result_bundle(
    result: dict,
    output_prefix: Path,
    model_ids: list[str],
    subjects: list[str],
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not {"theta_G_hat", "theta_hat", "a_hat", "d_hat"}.issubset(result):
        maybe_write_records(
            result,
            "cv_fold_records",
            output_prefix.with_name(output_prefix.name + "_cv_folds.csv"),
        )
        return

    atomic_to_csv(
        pd.DataFrame(
            {
                "model_id": model_ids,
                "theta_G_hat": np.asarray(result["theta_G_hat"], dtype=float),
            }
        ),
        output_prefix.with_name(output_prefix.name + "_theta_G.csv"),
    )

    atomic_to_csv(
        theta_t_long(
            np.asarray(result["theta_hat"], dtype=float),
            model_ids,
            subjects,
        ),
        output_prefix.with_name(output_prefix.name + "_theta_t_long.csv"),
    )

    atomic_to_csv(
        item_parameter_long(result, subjects),
        output_prefix.with_name(output_prefix.name + "_item_parameters.csv"),
    )
    if {
        "initial_theta_G",
        "initial_theta_hat",
        "initial_a_hat",
        "initial_d_hat",
    }.issubset(result):
        atomic_to_csv(
            pd.DataFrame(
                {
                    "model_id": model_ids,
                    "initial_theta_G": np.asarray(
                        result["initial_theta_G"], dtype=float
                    ),
                }
            ),
            output_prefix.with_name(output_prefix.name + "_initial_theta_G.csv"),
        )
        atomic_to_csv(
            theta_t_long(
                np.asarray(result["initial_theta_hat"], dtype=float),
                model_ids,
                subjects,
                value_column="initial_theta_t_hat",
            ),
            output_prefix.with_name(output_prefix.name + "_initial_theta_t_long.csv"),
        )
        atomic_to_csv(
            item_parameter_long(
                result,
                subjects,
                a_key="initial_a_hat",
                d_key="initial_d_hat",
            ),
            output_prefix.with_name(
                output_prefix.name + "_initial_item_parameters.csv"
            ),
        )

    maybe_write_records(
        result,
        "cv_path_records",
        output_prefix.with_name(output_prefix.name + "_cv_path.csv"),
    )
    maybe_write_records(
        result,
        "cv_fold_records",
        output_prefix.with_name(output_prefix.name + "_cv_folds.csv"),
    )


def make_trace_records(history: list[dict], fold: int, c_lambda: float, lambda_value: float) -> list[dict]:
    records = []
    for row in history:
        records.append(
            {
                "fold": int(fold),
                "C_lambda": float(c_lambda),
                "training_lambda_value": float(lambda_value),
                **{key: row[key] for key in row},
            }
        )
    return records


def summarize_cv_path(
    fold_records: list[dict],
    constants: list[float],
    n_folds: int,
) -> tuple[list[dict], float | None]:
    if not fold_records:
        return [], None

    path_records = []
    complete_candidates = []
    for constant in constants:
        all_candidate = [
            record for record in fold_records if np.isclose(record["C_lambda"], constant)
        ]
        candidate = [
            record for record in all_candidate if bool(record["converged"])
        ]
        if not candidate:
            continue
        total_test_nll = float(sum(record["test_negative_log_likelihood"] for record in candidate))
        fold_means = np.array(
            [record["mean_test_negative_log_likelihood"] for record in candidate],
            dtype=float,
        )
        row = {
            "C_lambda": float(constant),
            "attempted_folds": int(len(all_candidate)),
            "completed_folds": int(len(candidate)),
            "mean_test_negative_log_likelihood": float(total_test_nll / len(candidate)),
            "fold_standard_error": float(
                fold_means.std(ddof=1) / np.sqrt(len(candidate))
            )
            if len(candidate) >= 2
            else np.nan,
            "mean_training_negative_log_likelihood": float(
                np.mean([record["training_mean_negative_log_likelihood"] for record in candidate])
            ),
            "mean_training_lambda_value": float(
                np.mean([record["training_lambda_value"] for record in candidate])
            ),
            "converged_rate": float(np.mean([record["converged"] for record in all_candidate]))
            if all_candidate
            else np.nan,
            "runtime_seconds": float(sum(record["runtime_seconds"] for record in candidate)),
            "selected": False,
        }
        path_records.append(row)
        if len(candidate) == n_folds:
            complete_candidates.append(row)

    selected_c = None
    if complete_candidates:
        selected_c = min(
            complete_candidates,
            key=lambda row: (row["mean_test_negative_log_likelihood"], row["C_lambda"]),
        )["C_lambda"]
        for row in path_records:
            row["selected"] = bool(np.isclose(row["C_lambda"], selected_c))

    path_records.sort(key=lambda row: row["C_lambda"])
    return path_records, selected_c


def persist_cv_progress(
    output_dir: Path,
    fold_records: list[dict],
    trace_records: list[dict],
    constants: list[float],
    n_folds: int,
) -> tuple[list[dict], float | None]:
    fold_df = pd.DataFrame(fold_records).sort_values(["fold", "C_lambda"], na_position="last")
    atomic_to_csv(fold_df, output_dir / "proposed_cv_folds_live.csv")

    trace_df = pd.DataFrame(trace_records)
    if not trace_df.empty:
      trace_df = trace_df.sort_values(["fold", "C_lambda", "iteration"], na_position="last")
    atomic_to_csv(trace_df, output_dir / "proposed_cv_trace_live.csv")

    path_records, selected_c = summarize_cv_path(fold_records, constants, n_folds)
    atomic_to_csv(pd.DataFrame(path_records), output_dir / "proposed_cv_path_live.csv")

    progress_row = pd.DataFrame(
        [
            {
                "completed_tasks": len(fold_records),
                "total_tasks": n_folds * len(constants),
                "completed_folds": len({record["fold"] for record in fold_records}),
                "total_folds": n_folds,
                "selected_C_lambda_if_complete": selected_c,
                "last_update_utc": pd.Timestamp.utcnow().isoformat(),
            }
        ]
    )
    atomic_to_csv(progress_row, output_dir / "proposed_cv_progress.csv")
    return path_records, selected_c


def probability_nll_from_blocks(
    Y_blocks: list[np.ndarray],
    masks: list[np.ndarray],
    probability_blocks: list[np.ndarray],
) -> dict:
    raw_domain_nll = np.empty(len(Y_blocks), dtype=float)
    weighted_domain_nll = np.empty(len(Y_blocks), dtype=float)
    domain_entries = np.empty(len(Y_blocks), dtype=int)
    for t, (Y_t, mask_t, prob_t) in enumerate(zip(Y_blocks, masks, probability_blocks)):
        J_t = int(Y_t.shape[1])
        clipped = np.clip(prob_t, 1e-6, 1.0 - 1e-6)
        raw_domain_nll[t] = float(
            np.sum(
                mask_t
                * (
                    np.logaddexp(0.0, np.log(clipped / (1.0 - clipped)))
                    - Y_t * np.log(clipped / (1.0 - clipped))
                )
            )
        )
        weighted_domain_nll[t] = raw_domain_nll[t] / float(J_t)
        domain_entries[t] = int(np.count_nonzero(mask_t))
    total_nll = float(weighted_domain_nll.sum())
    raw_total_nll = float(raw_domain_nll.sum())
    total_entries = int(domain_entries.sum())
    return {
        "test_negative_log_likelihood": total_nll,
        "mean_test_negative_log_likelihood": total_nll,
        "raw_test_negative_log_likelihood": raw_total_nll,
        "test_entries": total_entries,
        "weighted_domain_test_negative_log_likelihood": weighted_domain_nll,
        "domain_test_negative_log_likelihood": raw_domain_nll,
        "domain_test_entries": domain_entries,
    }


def fit_local_method_with_mask(module, Y_blocks: list[np.ndarray], observation_mask, fit_kwargs: dict) -> dict:
    Y_blocks, J_vec = module_as_domain_blocks(module, Y_blocks)
    masks = module_prepare_observation_masks(module, Y_blocks, observation_mask)
    T = len(Y_blocks)
    N = Y_blocks[0].shape[0]
    J_max = int(max(J_vec))
    theta_hat = np.empty((T, N))
    a_hat = np.zeros((T, J_max))
    d_hat = np.zeros((T, J_max))
    initial_theta_hat = np.empty((T, N))
    initial_a_hat = np.zeros((T, J_max))
    initial_d_hat = np.zeros((T, J_max))
    weighted_domain_nll = np.empty(T)
    raw_domain_nll = np.empty(T)
    iterations = np.empty(T, dtype=int)
    best_iterations = np.empty(T, dtype=int)
    converged = np.empty(T, dtype=bool)

    for t, (Y_t, mask_t) in enumerate(zip(Y_blocks, masks)):
        J_t = int(J_vec[t])
        block_list = [Y_t]
        mask_list = [mask_t]
        initialization = module._spectral_initialization(
            block_list,
            mask_list,
            fit_kwargs.get("theta_bound", 4.0),
            fit_kwargs.get("a_lower_bound", 0.0),
            fit_kwargs.get("a_bound", 4.0),
            fit_kwargs.get("d_bound", 4.0),
        )
        initial_theta_hat[t] = initialization[0][0]
        initial_a_hat[t, :J_t] = initialization[1][0]
        initial_d_hat[t, :J_t] = initialization[2][0]
        best = module._single_adam_fit(
            block_list,
            observation_masks=mask_list,
            lambda_value=0.0,
            theta_bound=fit_kwargs.get("theta_bound", 4.0),
            a_lower_bound=fit_kwargs.get("a_lower_bound", 0.0),
            a_bound=fit_kwargs.get("a_bound", 4.0),
            d_bound=fit_kwargs.get("d_bound", 4.0),
            learning_rate=fit_kwargs.get("learning_rate", 0.02),
            learning_rate_decay_factor=fit_kwargs.get(
                "learning_rate_decay_factor", 0.5
            ),
            learning_rate_decay_steps=fit_kwargs.get(
                "learning_rate_decay_steps", 200
            ),
            max_iter=fit_kwargs.get("max_iter", 1000),
            tolerance=fit_kwargs.get("tolerance", 1e-3),
            verbose=fit_kwargs.get("verbose", False),
            initialization=initialization,
        )
        theta_hat[t] = best["theta_hat"][0]
        a_hat[t, :J_t] = best["a_hat_blocks"][0]
        d_hat[t, :J_t] = best["d_hat_blocks"][0]
        weighted_domain_nll[t] = float(best["negative_log_likelihood"])
        raw_domain_nll[t] = float(
            best.get("raw_negative_log_likelihood", best["negative_log_likelihood"])
        )
        iterations[t] = int(best["iterations"])
        best_iterations[t] = int(best["best_iteration"])
        converged[t] = bool(best["converged"])

    common = module.estimate_general_trait(
        theta_hat,
        initial_theta_G=fit_kwargs.get("initial_theta_G"),
        learning_rate=fit_kwargs.get("learning_rate", 0.03),
        learning_rate_decay_factor=fit_kwargs.get(
            "learning_rate_decay_factor", 0.5
        ),
        learning_rate_decay_steps=fit_kwargs.get(
            "learning_rate_decay_steps", 200
        ),
        max_iter=fit_kwargs.get("max_iter", 1000),
        tolerance=fit_kwargs.get("tolerance", 1e-7),
    )
    initial_common = module.estimate_general_trait(
        initial_theta_hat,
        initial_theta_G=fit_kwargs.get("initial_theta_G"),
        learning_rate=fit_kwargs.get("learning_rate", 0.03),
        learning_rate_decay_factor=fit_kwargs.get(
            "learning_rate_decay_factor", 0.5
        ),
        learning_rate_decay_steps=fit_kwargs.get(
            "learning_rate_decay_steps", 200
        ),
        max_iter=fit_kwargs.get("max_iter", 1000),
        tolerance=fit_kwargs.get("tolerance", 1e-7),
    )
    return {
        "method": "Local",
        "theta_hat": theta_hat,
        "a_hat": a_hat,
        "d_hat": d_hat,
        "theta_G_hat": common["theta_G_hat"],
        "initial_theta_hat": initial_theta_hat,
        "initial_a_hat": initial_a_hat,
        "initial_d_hat": initial_d_hat,
        "initial_theta_G": initial_common["theta_G_hat"],
        "objective": float(weighted_domain_nll.sum()),
        "negative_log_likelihood": float(weighted_domain_nll.sum()),
        "penalty": 0.0,
        "raw_negative_log_likelihood": float(raw_domain_nll.sum()),
        "weighted_domain_negative_log_likelihood": weighted_domain_nll,
        "domain_negative_log_likelihood": raw_domain_nll,
        "iterations": iterations,
        "best_iteration": best_iterations,
        "converged": converged,
        "total_entries": int(sum(np.count_nonzero(mask) for mask in masks)),
        "J_vec": np.asarray(J_vec, dtype=int),
    }


def fit_global_method_with_mask(module, Y_blocks: list[np.ndarray], observation_mask, fit_kwargs: dict) -> dict:
    return module.fit_global_method_with_observation_mask(
        Y_blocks,
        observation_mask=observation_mask,
        theta_bound=fit_kwargs.get("theta_bound", 4.0),
        a_lower_bound=fit_kwargs.get("a_lower_bound", 0.0),
        a_bound=fit_kwargs.get("a_bound", 4.0),
        d_bound=fit_kwargs.get("d_bound", 4.0),
        learning_rate=fit_kwargs.get("learning_rate", 0.02),
        learning_rate_decay_factor=fit_kwargs.get(
            "learning_rate_decay_factor", 0.5
        ),
        learning_rate_decay_steps=fit_kwargs.get(
            "learning_rate_decay_steps", 200
        ),
        max_iter=fit_kwargs.get("max_iter", 1000),
        tolerance=fit_kwargs.get("tolerance", 1e-3),
        verbose=fit_kwargs.get("verbose", False),
    )


def fit_naive_domain_mean_baseline(module, Y_blocks: list[np.ndarray], observation_mask) -> dict:
    Y_blocks, J_vec = module_as_domain_blocks(module, Y_blocks)
    masks = module_prepare_observation_masks(module, Y_blocks, observation_mask)
    probability_blocks: list[np.ndarray] = []
    for Y_t, mask_t in zip(Y_blocks, masks):
        row_counts = mask_t.sum(axis=1)
        domain_mean = float((mask_t * Y_t).sum() / max(1, mask_t.sum()))
        row_means = np.divide(
            (mask_t * Y_t).sum(axis=1),
            row_counts,
            out=np.full(Y_t.shape[0], domain_mean, dtype=float),
            where=row_counts > 0,
        )
        row_means = np.clip(row_means, 1e-6, 1.0 - 1e-6)
        probability_blocks.append(np.repeat(row_means[:, None], Y_t.shape[1], axis=1))
    train_score = probability_nll_from_blocks(Y_blocks, masks, probability_blocks)
    return {
        "method": "NaiveDomainMean",
        "objective": float(train_score["test_negative_log_likelihood"]),
        "negative_log_likelihood": float(train_score["test_negative_log_likelihood"]),
        "penalty": 0.0,
        "domain_negative_log_likelihood": np.asarray(
            train_score["domain_test_negative_log_likelihood"],
            dtype=float,
        ),
        "total_entries": int(train_score["test_entries"]),
        "probability_hat": probability_blocks,
        "J_vec": np.asarray(J_vec, dtype=int),
    }


def score_naive_domain_mean_baseline(module, Y_blocks: list[np.ndarray], test_mask, fitted_result: dict) -> dict:
    Y_blocks, _ = module_as_domain_blocks(module, Y_blocks)
    masks = module_prepare_observation_masks(module, Y_blocks, test_mask)
    return probability_nll_from_blocks(
        Y_blocks,
        masks,
        fitted_result["probability_hat"],
    )


def summarize_baseline_cv(method: str, fold_records: list[dict], cv_runtime: float, n_folds: int) -> dict:
    fold_means = np.asarray(
        [record["mean_test_negative_log_likelihood"] for record in fold_records],
        dtype=float,
    )
    total_test_nll = float(sum(record["test_negative_log_likelihood"] for record in fold_records))
    return {
        "method": method,
        "cv_fold_records": fold_records,
        "cv_mean_test_negative_log_likelihood": float(total_test_nll / len(fold_records)),
        "cv_fold_standard_error": float(fold_means.std(ddof=1) / np.sqrt(len(fold_means)))
        if len(fold_means) >= 2
        else np.nan,
        "cv_n_folds": int(n_folds),
        "cv_runtime_seconds": float(cv_runtime),
    }


def cross_validate_fixed_method(
    method_name: str,
    Y_blocks: list[np.ndarray],
    folds: list[dict],
    fit_fn,
    score_fn,
    *,
    resume_path: Path | None = None,
) -> dict:
    fold_records: list[dict] = []
    if resume_path is not None and resume_path.exists():
        saved = pd.read_csv(resume_path)
        if {"method", "fold"}.issubset(saved.columns):
            saved = saved[
                (saved["method"] == method_name)
                & saved["fold"].isin(range(1, len(folds) + 1))
            ].drop_duplicates("fold", keep="last")
            fold_records = saved.to_dict("records")
            print(
                f"Resuming {method_name} CV with {len(fold_records)}/"
                f"{len(folds)} completed folds."
            )
    cv_start = perf_counter()
    for fold_index, fold in enumerate(folds, start=1):
        if any(int(record["fold"]) == fold_index for record in fold_records):
            print(f"Skipping completed {method_name} fold {fold_index}/{len(folds)}.")
            continue
        fold_start = perf_counter()
        fitted = fit_fn(Y_blocks, fold["training_mask"])
        score = score_fn(Y_blocks, fold["test_mask"], fitted)
        runtime = perf_counter() - fold_start
        training_entries_by_domain = np.asarray(
            [mask_t.sum() for mask_t in fold["training_mask"]]
        )
        test_entries_by_domain = np.asarray(
            [mask_t.sum() for mask_t in fold["test_mask"]]
        )
        converged = fitted.get("converged", True)
        iterations = fitted.get("iterations", np.nan)
        if isinstance(converged, np.ndarray):
            converged_rate = float(np.mean(converged))
        else:
            converged_rate = float(bool(converged))
        if isinstance(iterations, np.ndarray):
            iterations_mean = float(np.mean(iterations))
            iterations_max = int(np.max(iterations))
        elif np.isfinite(iterations):
            iterations_mean = float(iterations)
            iterations_max = int(iterations)
        else:
            iterations_mean = np.nan
            iterations_max = np.nan
        fold_record = {
            "method": method_name,
            "fold": int(fold_index),
            "training_entries": int(training_entries_by_domain.sum()),
            "minimum_domain_training_entries": int(training_entries_by_domain.min()),
            "maximum_domain_training_entries": int(training_entries_by_domain.max()),
            "test_entries": int(score["test_entries"]),
            "minimum_domain_test_entries": int(test_entries_by_domain.min()),
            "maximum_domain_test_entries": int(test_entries_by_domain.max()),
            "test_negative_log_likelihood": float(score["test_negative_log_likelihood"]),
            "mean_test_negative_log_likelihood": float(score["mean_test_negative_log_likelihood"]),
            "training_mean_negative_log_likelihood": float(
                raw_mean_training_negative_log_likelihood(fitted)
            ),
            "converged_rate": converged_rate,
            "iterations_mean": iterations_mean,
            "iterations_max": iterations_max,
            "runtime_seconds": float(runtime),
        }
        fold_records.append(fold_record)
        if resume_path is not None:
            atomic_to_csv(
                pd.DataFrame(fold_records).sort_values("fold"),
                resume_path,
            )
        print(
            f"Completed {method_name} fold {fold_index}/{len(folds)}, "
            f"test_nll={fold_record['mean_test_negative_log_likelihood']:.6f}, "
            f"fit_time={runtime:.3f}s"
        )
    return summarize_baseline_cv(
        method_name,
        fold_records,
        float(sum(record["runtime_seconds"] for record in fold_records))
        if resume_path is not None
        else perf_counter() - cv_start,
        len(folds),
    )


def fit_candidate_task(
    fold_index: int,
    c_lambda: float,
    fit_kwargs: dict,
) -> dict:
    if _CV_PROCESS_STATE is None:
        raise RuntimeError("CV worker process was not initialized.")
    (
        module,
        Y_blocks,
        fold_assignments,
        fold_initializations,
    ) = _CV_PROCESS_STATE
    test_mask = [assignment == fold_index for assignment in fold_assignments]
    training_mask = [
        (assignment != 0) & (assignment != fold_index)
        for assignment in fold_assignments
    ]
    precomputed_initialization = fold_initializations[fold_index - 1]
    fit_start = perf_counter()
    fitted = module.fit_proposed_method(
        Y_blocks,
        observation_mask=training_mask,
        lambda_constant=float(c_lambda),
        precomputed_initialization=precomputed_initialization,
        **fit_kwargs,
    )
    score = module.held_out_negative_log_likelihood(Y_blocks, test_mask, fitted)
    runtime = perf_counter() - fit_start
    training_entries_by_domain = np.asarray(
        [mask_t.sum() for mask_t in training_mask]
    )
    test_entries_by_domain = np.asarray(
        [mask_t.sum() for mask_t in test_mask]
    )
    fold_record = {
        "fold": int(fold_index),
        "C_lambda": float(c_lambda),
        "training_lambda_value": float(fitted["lambda_value"]),
        "training_entries": int(training_entries_by_domain.sum()),
        "minimum_domain_training_entries": int(training_entries_by_domain.min()),
        "maximum_domain_training_entries": int(training_entries_by_domain.max()),
        "test_entries": int(score["test_entries"]),
        "minimum_domain_test_entries": int(test_entries_by_domain.min()),
        "maximum_domain_test_entries": int(test_entries_by_domain.max()),
        "test_negative_log_likelihood": float(score["test_negative_log_likelihood"]),
        "mean_test_negative_log_likelihood": float(score["mean_test_negative_log_likelihood"]),
        "training_mean_negative_log_likelihood": float(
            raw_mean_training_negative_log_likelihood(fitted)
        ),
        "converged": bool(fitted["converged"]),
        "iterations": int(fitted["iterations"]),
        "runtime_seconds": float(runtime),
    }
    trace_records = make_trace_records(
        fitted.get("history", []),
        fold=fold_index,
        c_lambda=c_lambda,
        lambda_value=fitted["lambda_value"],
    )
    artifact_stub = {
        "fold": int(fold_index),
        "C_lambda": float(c_lambda),
        "lambda_value": float(fitted["lambda_value"]),
        "objective": float(fitted["objective"]),
        "negative_log_likelihood": float(fitted["negative_log_likelihood"]),
        "penalty": float(fitted["penalty"]),
        "iterations": int(fitted["iterations"]),
        "converged": bool(fitted["converged"]),
        "theta_G_hat": np.asarray(fitted["theta_G_hat"], dtype=float),
    }
    return {
        "fold_record": fold_record,
        "trace_records": trace_records,
        "artifact_stub": artifact_stub,
    }


def initialize_cv_worker(
    module_path: str,
    Y_blocks: list[np.ndarray],
    fold_assignments: list[np.ndarray],
    fold_initializations: list,
) -> None:
    """Initialize CV state in workers started with the spawn method."""
    global _CV_PROCESS_STATE
    module = load_ai_measurement_module(Path(module_path))
    _CV_PROCESS_STATE = (
        module,
        Y_blocks,
        fold_assignments,
        fold_initializations,
    )


def load_proposed_cv_progress(
    output_dir: Path,
    constants: list[float],
    n_folds: int,
) -> tuple[list[dict], list[dict]]:
    fold_path = output_dir / "proposed_cv_folds_live.csv"
    if not fold_path.exists():
        return [], []

    fold_df = pd.read_csv(fold_path)
    required = {"fold", "C_lambda", "converged"}
    if not required.issubset(fold_df.columns):
        raise RuntimeError(f"Invalid CV resume file: {fold_path}.")
    fold_df = fold_df[fold_df["fold"].isin(range(1, n_folds + 1))].copy()
    valid_constant = np.zeros(len(fold_df), dtype=bool)
    for constant in constants:
        matches = np.isclose(fold_df["C_lambda"].to_numpy(dtype=float), constant)
        fold_df.loc[matches, "C_lambda"] = constant
        valid_constant |= matches
    fold_df = fold_df.loc[valid_constant]
    fold_df = fold_df.drop_duplicates(["fold", "C_lambda"], keep="last")
    if fold_df["converged"].dtype != bool:
        fold_df["converged"] = (
            fold_df["converged"].astype(str).str.strip().str.lower().eq("true")
        )
    fold_records = fold_df.to_dict("records")

    trace_records: list[dict] = []
    trace_path = output_dir / "proposed_cv_trace_live.csv"
    if trace_path.exists() and trace_path.stat().st_size > 0:
        try:
            trace_df = pd.read_csv(trace_path)
        except EmptyDataError:
            trace_df = pd.DataFrame()
        if {"fold", "C_lambda"}.issubset(trace_df.columns):
            completed_keys = {
                (int(record["fold"]), float(record["C_lambda"]))
                for record in fold_records
            }
            keep = [
                any(
                    int(fold) == completed_fold and np.isclose(c_lambda, completed_c)
                    for completed_fold, completed_c in completed_keys
                )
                for fold, c_lambda in zip(trace_df["fold"], trace_df["C_lambda"])
            ]
            trace_df = trace_df.loc[keep]
            if "iteration" in trace_df.columns:
                trace_df = trace_df.drop_duplicates(
                    ["fold", "C_lambda", "iteration"], keep="last"
                )
            trace_records = trace_df.to_dict("records")

    print(
        f"Resuming proposed CV with {len(fold_records)}/"
        f"{n_folds * len(constants)} completed candidate fits."
    )
    return fold_records, trace_records


def cross_validate_proposed_method_multiprocess(
    module,
    Y_blocks: list[np.ndarray],
    constants: list[float],
    output_dir: Path,
    model_ids: list[str],
    subjects: list[str],
    *,
    n_folds: int,
    cv_workers: int,
    seed: int,
    fit_kwargs: dict,
    folds: list[dict] | None = None,
    refit_full_data: bool = True,
) -> dict:
    global _CV_PROCESS_STATE
    if "fork" in mp.get_all_start_methods():
        process_context = mp.get_context("fork")
        uses_spawn = False
    else:
        process_context = mp.get_context("spawn")
        uses_spawn = True

    Y_blocks, _ = module_as_domain_blocks(module, Y_blocks)
    if folds is None:
        full_mask = module_prepare_observation_masks(module, Y_blocks, None)
        folds = module.make_entrywise_cv_folds(
            full_mask,
            n_folds=n_folds,
            seed=seed + 100_003,
        )

    fold_records, trace_records = load_proposed_cv_progress(
        output_dir, constants, n_folds
    )
    cv_start = perf_counter()

    completed_keys = {
        (int(record["fold"]), float(record["C_lambda"]))
        for record in fold_records
    }
    pending_tasks = [
        (fold_index, constant)
        for constant in constants
        for fold_index in range(1, n_folds + 1)
        if not any(
            completed_fold == fold_index and np.isclose(completed_c, constant)
            for completed_fold, completed_c in completed_keys
        )
    ]

    if pending_tasks:
        print(
            f"Preparing all fold initializations before scheduling "
            f"{len(pending_tasks)} remaining grid tasks..."
        )
        assignment_dtype = np.uint8 if n_folds <= 255 else np.uint16
        fold_assignments = [
            np.zeros_like(Y_t, dtype=assignment_dtype) for Y_t in Y_blocks
        ]
        fold_initializations = [None] * n_folds
        pending_folds = {fold_index for fold_index, _ in pending_tasks}
        for fold_index, fold in enumerate(folds, start=1):
            for assignment, test_mask in zip(
                fold_assignments, fold["test_mask"]
            ):
                if np.any(assignment[test_mask] != 0):
                    raise RuntimeError("CV test folds overlap.")
                assignment[test_mask] = fold_index
            if fold_index in pending_folds:
                fold_initializations[fold_index - 1] = module._spectral_initialization(
                    [
                        np.where(mask_t, Y_t, 0.0)
                        for Y_t, mask_t in zip(Y_blocks, fold["training_mask"])
                    ],
                    fold["training_mask"],
                    fit_kwargs.get("theta_bound", 4.0),
                    fit_kwargs.get("a_lower_bound", 0.0),
                    fit_kwargs.get("a_bound", 4.0),
                    fit_kwargs.get("d_bound", 4.0),
                )
                write_initialization_bundle(
                    fold_initializations[fold_index - 1],
                    output_dir / f"proposed_cv_fold_{fold_index:02d}",
                    model_ids,
                    subjects,
                )

        print(
            f"Scheduling the full fold x candidate grid with "
            f"{min(cv_workers, len(pending_tasks))} workers."
        )
        _CV_PROCESS_STATE = (
            module,
            Y_blocks,
            fold_assignments,
            fold_initializations,
        )
        try:
            executor_kwargs = {}
            if uses_spawn:
                executor_kwargs = {
                    "initializer": initialize_cv_worker,
                    "initargs": (
                        str(Path(module.__file__).resolve()),
                        Y_blocks,
                        fold_assignments,
                        fold_initializations,
                    ),
                }
            with ProcessPoolExecutor(
                max_workers=min(cv_workers, len(pending_tasks)),
                mp_context=process_context,
                **executor_kwargs,
            ) as executor:
                futures = {
                    executor.submit(
                        fit_candidate_task,
                        fold_index,
                        constant,
                        fit_kwargs,
                    ): (fold_index, constant)
                    for fold_index, constant in pending_tasks
                }

                for future in as_completed(futures):
                    fold_index, constant = futures[future]
                    result = future.result()
                    fold_records.append(result["fold_record"])
                    trace_records.extend(result["trace_records"])

                    path_records, selected_c = persist_cv_progress(
                        output_dir,
                        fold_records,
                        trace_records,
                        constants,
                        n_folds,
                    )
                    print(
                        f"Completed fold {fold_index}/{n_folds}, C_lambda={constant:.4g}, "
                        f"test_nll={result['fold_record']['mean_test_negative_log_likelihood']:.6f}, "
                        f"completed_tasks={len(fold_records)}/{n_folds * len(constants)}, "
                        f"current_selected={selected_c}"
                    )
        finally:
            _CV_PROCESS_STATE = None
    else:
        print("All proposed CV grid tasks are already complete; skipping the pool.")

    path_records, selected_c = summarize_cv_path(fold_records, constants, n_folds)
    if selected_c is None:
        raise RuntimeError(
            "Cross-validation did not produce any candidate with "
            f"{n_folds} converged folds."
        )

    cv_summary = {
        "method": "proposed",
        "selected_C_lambda": float(selected_c),
        "cv_mean_test_negative_log_likelihood": float(
            next(
                row["mean_test_negative_log_likelihood"]
                for row in path_records
                if np.isclose(row["C_lambda"], selected_c)
            )
        ),
        "cv_fold_standard_error": float(
            next(
                row["fold_standard_error"]
                for row in path_records
                if np.isclose(row["C_lambda"], selected_c)
            )
        ),
        "cv_n_folds": int(n_folds),
        "cv_runtime_seconds": float(perf_counter() - cv_start),
        "cv_path_records": path_records,
        "cv_fold_records": sorted(
            fold_records,
            key=lambda row: (int(row["fold"]), float(row["C_lambda"])),
        ),
        "cv_trace_records": trace_records,
    }
    if not refit_full_data:
        return cv_summary

    refit_start = perf_counter()
    final_fit = module.fit_proposed_method(
        Y_blocks,
        lambda_constant=float(selected_c),
        **fit_kwargs,
    )
    final_fit.update(cv_summary)
    final_fit["refit_runtime_seconds"] = float(perf_counter() - refit_start)
    return final_fit


def candidate_lambda_values() -> list[float]:
    return [
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.075,
        0.09,
        0.1,
        0.11,
        0.125,
        0.2,
        0.5,
        1.0,
        1.5,
    ]


def resolved_c_lambda_grid(
    c_lambda_grid: list[float] | None,
) -> list[float]:
    values = candidate_lambda_values() if c_lambda_grid is None else c_lambda_grid
    constants = np.asarray(values, dtype=float)
    if constants.ndim != 1 or constants.size == 0:
        raise ValueError("C_lambda grid must be a nonempty one-dimensional sequence.")
    if not np.all(np.isfinite(constants)) or np.any(constants < 0.0):
        raise ValueError("Every C_lambda value must be finite and nonnegative.")
    unique = np.unique(constants)
    if unique.size != constants.size:
        raise ValueError("C_lambda grid must not contain duplicates.")
    return sorted(float(value) for value in constants)


@dataclass(frozen=True)
class PostCVTask:
    task_type: str
    method_name: str
    fold_index: int | None = None
    selected_c_lambda: float | None = None


@dataclass(frozen=True)
class EstimationTask:
    kind: str
    method: str | None = None
    fold_index: int | None = None
    c_lambda: float | None = None


def proposed_final_fit_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "proposed_final_fit_checkpoint.pkl"


def baseline_cv_checkpoint_path(output_dir: Path, method_name: str) -> Path:
    return output_dir / f"{method_name}_cv_checkpoint.pkl"


def baseline_full_data_only_checkpoint_path(output_dir: Path, method_name: str) -> Path:
    return output_dir / f"{method_name}_full_data_checkpoint.pkl"


def baseline_final_fit_checkpoint_path(output_dir: Path, method_name: str) -> Path:
    return output_dir / f"{method_name}_fit_checkpoint.pkl"


def baseline_live_csv_path(output_dir: Path, method_name: str) -> Path:
    return output_dir / f"{method_name}_cv_folds_live.csv"


def load_baseline_cv_progress(
    method_name: str,
    resume_path: Path,
    n_folds: int,
) -> list[dict]:
    if not resume_path.exists():
        return []
    saved = pd.read_csv(resume_path)
    required = {"method", "fold"}
    if not required.issubset(saved.columns):
        raise RuntimeError(f"Invalid baseline CV resume file: {resume_path}.")
    saved = saved[
        (saved["method"] == method_name)
        & saved["fold"].isin(range(1, n_folds + 1))
    ].drop_duplicates("fold", keep="last")
    records = saved.sort_values("fold").to_dict("records")
    print(
        f"Resuming {method_name} CV with {len(records)}/{n_folds} completed folds."
    )
    return records


def persist_baseline_cv_progress(
    method_name: str,
    fold_records: list[dict],
    resume_path: Path,
) -> None:
    if not fold_records:
        return
    method_records = [
        record for record in fold_records if str(record["method"]) == method_name
    ]
    atomic_to_csv(
        pd.DataFrame(method_records).sort_values("fold"),
        resume_path,
    )


def extract_cv_summary_fields(result: dict) -> dict:
    keys = {
        "method",
        "cv_fold_records",
        "cv_mean_test_negative_log_likelihood",
        "cv_fold_standard_error",
        "cv_n_folds",
        "cv_runtime_seconds",
    }
    return {key: result[key] for key in keys if key in result}


def merge_fit_with_cv_summary(full_fit: dict, cv_summary: dict) -> dict:
    merged = slim_pickle_payload(full_fit)
    merged.update(extract_cv_summary_fields(cv_summary))
    return merged


def fit_baseline_fold_record(
    method_name: str,
    Y_blocks: list[np.ndarray],
    fold: dict,
    module,
    fit_kwargs: dict,
) -> dict:
    fit_start = perf_counter()
    if method_name == "local":
        fitted = fit_local_method_with_mask(
            module, Y_blocks, fold["training_mask"], fit_kwargs
        )
        score = module.held_out_negative_log_likelihood(
            Y_blocks, fold["test_mask"], fitted
        )
    elif method_name == "global":
        fitted = fit_global_method_with_mask(
            module, Y_blocks, fold["training_mask"], fit_kwargs
        )
        score = module.held_out_negative_log_likelihood(
            Y_blocks, fold["test_mask"], fitted
        )
    elif method_name == "naive_domain_mean":
        fitted = fit_naive_domain_mean_baseline(
            module, Y_blocks, fold["training_mask"]
        )
        score = score_naive_domain_mean_baseline(
            module, Y_blocks, fold["test_mask"], fitted
        )
    else:
        raise ValueError(f"Unsupported baseline method: {method_name}.")

    runtime = perf_counter() - fit_start
    training_entries_by_domain = np.asarray(
        [mask_t.sum() for mask_t in fold["training_mask"]]
    )
    test_entries_by_domain = np.asarray(
        [mask_t.sum() for mask_t in fold["test_mask"]]
    )
    converged = fitted.get("converged", True)
    iterations = fitted.get("iterations", np.nan)
    if isinstance(converged, np.ndarray):
        converged_rate = float(np.mean(converged))
    else:
        converged_rate = float(bool(converged))
    if isinstance(iterations, np.ndarray):
        iterations_mean = float(np.mean(iterations))
        iterations_max = int(np.max(iterations))
    elif np.isfinite(iterations):
        iterations_mean = float(iterations)
        iterations_max = int(iterations)
    else:
        iterations_mean = np.nan
        iterations_max = np.nan
    return {
        "method": method_name,
        "training_mean_negative_log_likelihood": float(
            raw_mean_training_negative_log_likelihood(fitted)
        ),
        "converged_rate": converged_rate,
        "iterations_mean": iterations_mean,
        "iterations_max": iterations_max,
        "runtime_seconds": float(runtime),
        "test_negative_log_likelihood": float(score["test_negative_log_likelihood"]),
        "mean_test_negative_log_likelihood": float(
            score["mean_test_negative_log_likelihood"]
        ),
        "test_entries": int(score["test_entries"]),
        "training_entries": int(training_entries_by_domain.sum()),
        "minimum_domain_training_entries": int(training_entries_by_domain.min()),
        "maximum_domain_training_entries": int(training_entries_by_domain.max()),
        "minimum_domain_test_entries": int(test_entries_by_domain.min()),
        "maximum_domain_test_entries": int(test_entries_by_domain.max()),
    }


def fit_baseline_full_data_result(
    method_name: str,
    Y_blocks: list[np.ndarray],
    module,
    fit_kwargs: dict,
) -> dict:
    start = perf_counter()
    if method_name == "local":
        fitted = module.fit_local_method(Y_blocks, **fit_kwargs)
    elif method_name == "global":
        fitted = module.fit_global_method(Y_blocks, **fit_kwargs)
    elif method_name == "naive_domain_mean":
        fitted = fit_naive_domain_mean_baseline(module, Y_blocks, None)
    else:
        raise ValueError(f"Unsupported full-data method: {method_name}.")
    fitted["runtime_seconds"] = float(perf_counter() - start)
    return fitted


def initialize_post_cv_worker(
    module_path: str,
    Y_blocks: list[np.ndarray],
    folds: list[dict],
    fit_kwargs: dict,
) -> None:
    global _POST_CV_PROCESS_STATE
    module = load_ai_measurement_module(Path(module_path))
    _POST_CV_PROCESS_STATE = (module, Y_blocks, folds, fit_kwargs)


def run_post_cv_task(task: PostCVTask) -> dict:
    if _POST_CV_PROCESS_STATE is None:
        raise RuntimeError("Post-CV worker process was not initialized.")
    module, Y_blocks, folds, fit_kwargs = _POST_CV_PROCESS_STATE

    if task.task_type == "proposed_full_refit":
        start = perf_counter()
        fitted = module.fit_proposed_method(
            Y_blocks,
            lambda_constant=float(task.selected_c_lambda),
            **fit_kwargs,
        )
        fitted["runtime_seconds"] = float(perf_counter() - start)
        return {
            "task_type": task.task_type,
            "method_name": task.method_name,
            "selected_c_lambda": float(task.selected_c_lambda),
            "result": fitted,
        }

    if task.task_type == "baseline_cv_fold":
        fold = folds[int(task.fold_index) - 1]
        record = fit_baseline_fold_record(
            task.method_name,
            Y_blocks,
            fold,
            module,
            fit_kwargs,
        )
        record["fold"] = int(task.fold_index)
        return {
            "task_type": task.task_type,
            "method_name": task.method_name,
            "fold_index": int(task.fold_index),
            "fold_record": record,
        }

    if task.task_type == "baseline_full_fit":
        result = fit_baseline_full_data_result(
            task.method_name,
            Y_blocks,
            module,
            fit_kwargs,
        )
        return {
            "task_type": task.task_type,
            "method_name": task.method_name,
            "result": result,
        }

    raise ValueError(f"Unknown PostCVTask type: {task.task_type}.")


def initialize_global_worker(
    module_path: str,
    Y_blocks: list[np.ndarray],
    folds: list[dict],
    fit_kwargs: dict,
    fold_assignments: list[np.ndarray] | None,
    fold_initializations: list | None,
) -> None:
    global _GLOBAL_PROCESS_STATE
    module = load_ai_measurement_module(Path(module_path))
    _GLOBAL_PROCESS_STATE = (
        module,
        Y_blocks,
        folds,
        fit_kwargs,
        fold_assignments,
        fold_initializations,
    )


def run_estimation_task(task: EstimationTask) -> dict:
    if _GLOBAL_PROCESS_STATE is None:
        raise RuntimeError("Global worker process was not initialized.")
    (
        module,
        Y_blocks,
        folds,
        fit_kwargs,
        fold_assignments,
        fold_initializations,
    ) = _GLOBAL_PROCESS_STATE

    if task.kind == "proposed_cv":
        if fold_assignments is None or fold_initializations is None:
            raise RuntimeError("Proposed CV worker state is missing.")
        test_mask = [assignment == task.fold_index for assignment in fold_assignments]
        training_mask = [
            (assignment != 0) & (assignment != task.fold_index)
            for assignment in fold_assignments
        ]
        precomputed_initialization = fold_initializations[int(task.fold_index) - 1]
        fit_start = perf_counter()
        fitted = module.fit_proposed_method(
            Y_blocks,
            observation_mask=training_mask,
            lambda_constant=float(task.c_lambda),
            precomputed_initialization=precomputed_initialization,
            **fit_kwargs,
        )
        score = module.held_out_negative_log_likelihood(Y_blocks, test_mask, fitted)
        runtime = perf_counter() - fit_start
        training_entries_by_domain = np.asarray([mask_t.sum() for mask_t in training_mask])
        test_entries_by_domain = np.asarray([mask_t.sum() for mask_t in test_mask])
        fold_record = {
            "fold": int(task.fold_index),
            "C_lambda": float(task.c_lambda),
            "training_lambda_value": float(fitted["lambda_value"]),
            "training_entries": int(training_entries_by_domain.sum()),
            "minimum_domain_training_entries": int(training_entries_by_domain.min()),
            "maximum_domain_training_entries": int(training_entries_by_domain.max()),
            "test_entries": int(score["test_entries"]),
            "minimum_domain_test_entries": int(test_entries_by_domain.min()),
            "maximum_domain_test_entries": int(test_entries_by_domain.max()),
            "test_negative_log_likelihood": float(score["test_negative_log_likelihood"]),
            "mean_test_negative_log_likelihood": float(score["mean_test_negative_log_likelihood"]),
            "training_mean_negative_log_likelihood": float(
                raw_mean_training_negative_log_likelihood(fitted)
            ),
            "converged": bool(fitted["converged"]),
            "iterations": int(fitted["iterations"]),
            "runtime_seconds": float(runtime),
        }
        trace_records = make_trace_records(
            fitted.get("history", []),
            fold=int(task.fold_index),
            c_lambda=float(task.c_lambda),
            lambda_value=fitted["lambda_value"],
        )
        return {
            "kind": task.kind,
            "fold_record": fold_record,
            "trace_records": trace_records,
        }

    if task.kind in {"local_cv", "global_cv", "naive_cv"}:
        method_name = (
            "local"
            if task.kind == "local_cv"
            else "global"
            if task.kind == "global_cv"
            else "naive_domain_mean"
        )
        fold = folds[int(task.fold_index) - 1]
        record = fit_baseline_fold_record(
            method_name,
            Y_blocks,
            fold,
            module,
            fit_kwargs,
        )
        record["fold"] = int(task.fold_index)
        return {
            "kind": task.kind,
            "method_name": method_name,
            "fold_record": record,
        }

    if task.kind in {"local_full", "global_full", "naive_full"}:
        method_name = (
            "local"
            if task.kind == "local_full"
            else "global"
            if task.kind == "global_full"
            else "naive_domain_mean"
        )
        result = fit_baseline_full_data_result(
            method_name,
            Y_blocks,
            module,
            fit_kwargs,
        )
        return {
            "kind": task.kind,
            "method_name": method_name,
            "result": result,
        }

    if task.kind == "proposed_full":
        start = perf_counter()
        fitted = module.fit_proposed_method(
            Y_blocks,
            lambda_constant=float(task.c_lambda),
            **fit_kwargs,
        )
        fitted["runtime_seconds"] = float(perf_counter() - start)
        return {
            "kind": task.kind,
            "result": fitted,
            "selected_c_lambda": float(task.c_lambda),
        }

    raise ValueError(f"Unknown EstimationTask kind: {task.kind}.")


def initialize_full_data_worker(
    module_path: str,
    Y_blocks: list[np.ndarray],
    precomputed_initialization,
) -> None:
    global _FULL_DATA_PROCESS_STATE
    module = load_ai_measurement_module(Path(module_path))
    _FULL_DATA_PROCESS_STATE = (module, Y_blocks, precomputed_initialization)


def fit_full_data_candidate_task(
    candidate_index: int,
    c_lambda: float,
    fit_kwargs: dict,
) -> tuple[int, float, dict]:
    if _FULL_DATA_PROCESS_STATE is None:
        raise RuntimeError("Full-data worker process was not initialized.")
    module, Y_blocks, precomputed_initialization = _FULL_DATA_PROCESS_STATE
    start = perf_counter()
    fitted = module.fit_proposed_method(
        Y_blocks,
        lambda_constant=float(c_lambda),
        precomputed_initialization=precomputed_initialization,
        **fit_kwargs,
    )
    fitted["C_lambda"] = float(c_lambda)
    fitted["runtime_seconds"] = float(perf_counter() - start)
    return candidate_index, float(c_lambda), fitted


def fit_full_data_baseline_task(
    method_name: str,
    fit_kwargs: dict,
) -> tuple[str, dict]:
    if _FULL_DATA_PROCESS_STATE is None:
        raise RuntimeError("Full-data worker process was not initialized.")
    module, Y_blocks, _ = _FULL_DATA_PROCESS_STATE
    result = fit_baseline_full_data_result(
        method_name,
        Y_blocks,
        module,
        fit_kwargs,
    )
    return method_name, result


def maybe_finalize_proposed_checkpoint(
    output_dir: Path,
    proposed_cv: dict,
    proposed_full_fit: dict | None,
) -> dict | None:
    final_path = output_dir / "proposed_checkpoint.pkl"
    if final_path.exists():
        return load_pickle_checkpoint(final_path)
    if proposed_full_fit is None:
        return None
    finalized = slim_pickle_payload(proposed_full_fit)
    finalized.update(proposed_cv)
    finalized["refit_runtime_seconds"] = float(
        proposed_full_fit.get("runtime_seconds", np.nan)
    )
    atomic_checkpoint_pickle(finalized, final_path)
    return finalized


def maybe_finalize_baseline_checkpoint(
    output_dir: Path,
    method_name: str,
    cv_summary: dict | None,
    full_fit: dict | None,
) -> dict | None:
    final_path = baseline_final_fit_checkpoint_path(output_dir, method_name)
    if final_path.exists():
        return load_pickle_checkpoint(final_path)
    if cv_summary is None or full_fit is None:
        return None
    finalized = merge_fit_with_cv_summary(full_fit, cv_summary)
    atomic_checkpoint_pickle(finalized, final_path)
    return finalized


def run_post_cv_tasks_multiprocess(
    module,
    Y_blocks: list[np.ndarray],
    folds: list[dict],
    output_dir: Path,
    fit_kwargs: dict,
    proposed_cv: dict,
    cv_workers: int,
) -> tuple[dict | None, dict[str, dict | None], dict[str, dict | None]]:
    global _POST_CV_PROCESS_STATE
    baseline_methods = ("local", "global", "naive_domain_mean")
    n_folds = int(proposed_cv["cv_n_folds"])

    baseline_cv_summaries: dict[str, dict | None] = {
        method_name: None for method_name in baseline_methods
    }
    baseline_full_fits: dict[str, dict | None] = {
        method_name: None for method_name in baseline_methods
    }
    baseline_fold_records: dict[str, list[dict]] = {}

    for method_name in baseline_methods:
        cv_checkpoint = baseline_cv_checkpoint_path(output_dir, method_name)
        final_fit_checkpoint = baseline_final_fit_checkpoint_path(output_dir, method_name)
        if cv_checkpoint.exists():
            baseline_cv_summaries[method_name] = load_pickle_checkpoint(cv_checkpoint)
            baseline_fold_records[method_name] = list(
                baseline_cv_summaries[method_name]["cv_fold_records"]
            )
            print(f"Skipping {method_name} CV summary: checkpoint already exists.")
        elif final_fit_checkpoint.exists():
            fitted = load_pickle_checkpoint(final_fit_checkpoint)
            baseline_cv_summaries[method_name] = extract_cv_summary_fields(fitted)
            baseline_fold_records[method_name] = list(
                baseline_cv_summaries[method_name]["cv_fold_records"]
            )
            atomic_checkpoint_pickle(
                baseline_cv_summaries[method_name],
                cv_checkpoint,
            )
            print(f"Skipping {method_name} CV summary: found final fit checkpoint.")
        else:
            live_path = baseline_live_csv_path(output_dir, method_name)
            baseline_fold_records[method_name] = load_baseline_cv_progress(
                method_name,
                live_path,
                n_folds,
            )

    proposed_full_fit = None
    final_proposed_checkpoint = output_dir / "proposed_checkpoint.pkl"
    proposed_refit_checkpoint = proposed_final_fit_checkpoint_path(output_dir)
    if final_proposed_checkpoint.exists():
        proposed_full_fit = load_pickle_checkpoint(final_proposed_checkpoint)
        print("Skipping proposed full-data refit: final checkpoint already exists.")
    elif proposed_refit_checkpoint.exists():
        proposed_full_fit = load_pickle_checkpoint(proposed_refit_checkpoint)
        print("Skipping proposed full-data refit: intermediate checkpoint already exists.")

    for method_name in baseline_methods:
        final_path = baseline_final_fit_checkpoint_path(output_dir, method_name)
        intermediate_path = baseline_full_data_only_checkpoint_path(output_dir, method_name)
        if final_path.exists():
            baseline_full_fits[method_name] = load_pickle_checkpoint(final_path)
            print(f"Skipping {method_name} full-data fit: final checkpoint already exists.")
        elif intermediate_path.exists():
            baseline_full_fits[method_name] = load_pickle_checkpoint(intermediate_path)
            print(
                f"Skipping {method_name} full-data fit: intermediate checkpoint already exists."
            )

    pending_tasks: list[PostCVTask] = []
    if proposed_full_fit is None:
        pending_tasks.append(
            PostCVTask(
                task_type="proposed_full_refit",
                method_name="proposed",
                selected_c_lambda=float(proposed_cv["selected_C_lambda"]),
            )
        )

    for method_name in baseline_methods:
        if baseline_cv_summaries[method_name] is None:
            completed_folds = {
                int(record["fold"]) for record in baseline_fold_records[method_name]
            }
            for fold_index in range(1, n_folds + 1):
                if fold_index not in completed_folds:
                    pending_tasks.append(
                        PostCVTask(
                            task_type="baseline_cv_fold",
                            method_name=method_name,
                            fold_index=fold_index,
                        )
                    )
                else:
                    print(
                        f"Skipping completed {method_name} fold {fold_index}/{n_folds}."
                    )
        if baseline_full_fits[method_name] is None:
            pending_tasks.append(
                PostCVTask(
                    task_type="baseline_full_fit",
                    method_name=method_name,
                )
            )

    if pending_tasks:
        print(
            f"Scheduling {len(pending_tasks)} post-CV tasks with "
            f"{min(cv_workers, len(pending_tasks))} workers:"
        )
        print("  proposed full-data refit: 1" if proposed_full_fit is None else "  proposed full-data refit: 0 (skipped)")
        print(
            f"  local CV folds: "
            f"{sum(task.method_name == 'local' and task.task_type == 'baseline_cv_fold' for task in pending_tasks)}"
        )
        print(
            f"  global CV folds: "
            f"{sum(task.method_name == 'global' and task.task_type == 'baseline_cv_fold' for task in pending_tasks)}"
        )
        print(
            "  naive CV folds: "
            f"{sum(task.method_name == 'naive_domain_mean' and task.task_type == 'baseline_cv_fold' for task in pending_tasks)}"
        )
        print(
            f"  local/global/naive full-data fits: "
            f"{sum(task.task_type == 'baseline_full_fit' for task in pending_tasks)}"
        )

        if "fork" in mp.get_all_start_methods():
            process_context = mp.get_context("fork")
            uses_spawn = False
        else:
            process_context = mp.get_context("spawn")
            uses_spawn = True
        _POST_CV_PROCESS_STATE = (module, Y_blocks, folds, fit_kwargs)
        completed_tasks = 0
        try:
            executor_kwargs = {}
            if uses_spawn:
                executor_kwargs = {
                    "initializer": initialize_post_cv_worker,
                    "initargs": (
                        str(Path(module.__file__).resolve()),
                        Y_blocks,
                        folds,
                        fit_kwargs,
                    ),
                }
            with ProcessPoolExecutor(
                max_workers=min(cv_workers, len(pending_tasks)),
                mp_context=process_context,
                **executor_kwargs,
            ) as executor:
                futures = {
                    executor.submit(run_post_cv_task, task): task
                    for task in pending_tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    result = future.result()
                    completed_tasks += 1

                    if task.task_type == "proposed_full_refit":
                        proposed_full_fit = result["result"]
                        atomic_checkpoint_pickle(
                            proposed_full_fit,
                            proposed_refit_checkpoint,
                        )
                        print("Completed proposed full-data refit")
                    elif task.task_type == "baseline_cv_fold":
                        method_name = task.method_name
                        baseline_fold_records[method_name].append(result["fold_record"])
                        baseline_fold_records[method_name].sort(
                            key=lambda row: int(row["fold"])
                        )
                        persist_baseline_cv_progress(
                            method_name,
                            baseline_fold_records[method_name],
                            baseline_live_csv_path(output_dir, method_name),
                        )
                        print(
                            f"Completed {method_name} CV fold {task.fold_index}/{n_folds}"
                        )
                        if len(baseline_fold_records[method_name]) == n_folds:
                            summary = summarize_baseline_cv(
                                method_name,
                                baseline_fold_records[method_name],
                                float(
                                    sum(
                                        row["runtime_seconds"]
                                        for row in baseline_fold_records[method_name]
                                    )
                                ),
                                n_folds,
                            )
                            baseline_cv_summaries[method_name] = summary
                            atomic_checkpoint_pickle(
                                summary,
                                baseline_cv_checkpoint_path(output_dir, method_name),
                            )
                            print(
                                f"Completed {method_name} CV summary checkpoint."
                            )
                    elif task.task_type == "baseline_full_fit":
                        method_name = task.method_name
                        baseline_full_fits[method_name] = result["result"]
                        atomic_checkpoint_pickle(
                            baseline_full_fits[method_name],
                            baseline_full_data_only_checkpoint_path(output_dir, method_name),
                        )
                        print(f"Completed {method_name} full-data fit")

                    print(
                        f"post-CV completed_tasks={completed_tasks}/{len(pending_tasks)}"
                    )
        finally:
            _POST_CV_PROCESS_STATE = None
    else:
        print("All post-CV tasks are already complete; skipping the pool.")

    for method_name in baseline_methods:
        if (
            baseline_cv_summaries[method_name] is None
            and len(baseline_fold_records[method_name]) == n_folds
        ):
            summary = summarize_baseline_cv(
                method_name,
                baseline_fold_records[method_name],
                float(sum(row["runtime_seconds"] for row in baseline_fold_records[method_name])),
                n_folds,
            )
            baseline_cv_summaries[method_name] = summary
            atomic_checkpoint_pickle(
                summary,
                baseline_cv_checkpoint_path(output_dir, method_name),
            )
        if baseline_full_fits[method_name] is None:
            final_path = baseline_final_fit_checkpoint_path(output_dir, method_name)
            intermediate_path = baseline_full_data_only_checkpoint_path(output_dir, method_name)
            if final_path.exists():
                baseline_full_fits[method_name] = load_pickle_checkpoint(final_path)
            elif intermediate_path.exists():
                baseline_full_fits[method_name] = load_pickle_checkpoint(intermediate_path)

    proposed = maybe_finalize_proposed_checkpoint(
        output_dir,
        proposed_cv,
        proposed_full_fit,
    )
    finalized_baseline_fits: dict[str, dict | None] = {}
    for method_name in baseline_methods:
        finalized_baseline_fits[method_name] = maybe_finalize_baseline_checkpoint(
            output_dir,
            method_name,
            baseline_cv_summaries[method_name],
            baseline_full_fits[method_name],
        )

    return proposed, baseline_cv_summaries, finalized_baseline_fits


def run_all_estimations_multiprocess(
    module,
    Y_blocks: list[np.ndarray],
    folds: list[dict],
    output_dir: Path,
    model_ids: list[str],
    subjects: list[str],
    fit_kwargs: dict,
    constants: list[float],
    n_folds: int,
    cv_workers: int,
    run_baselines: bool = True,
    run_proposed_final: bool = True,
) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    global _GLOBAL_PROCESS_STATE
    baseline_methods = (
        ("local", "global", "naive_domain_mean") if run_baselines else tuple()
    )
    baseline_kind_map = {
        "local": ("local_cv", "local_full"),
        "global": ("global_cv", "global_full"),
        "naive_domain_mean": ("naive_cv", "naive_full"),
    }

    proposed_final_checkpoint = output_dir / "proposed_checkpoint.pkl"
    proposed_intermediate_checkpoint = proposed_final_fit_checkpoint_path(output_dir)
    proposed_final = None
    if run_proposed_final and proposed_final_checkpoint.exists():
        proposed_final = load_pickle_checkpoint(proposed_final_checkpoint)
        print("Skipping proposed final result: final checkpoint already exists.")

    proposed_fold_records, trace_records = load_proposed_cv_progress(
        output_dir,
        constants,
        n_folds,
    )
    proposed_completed_keys = {
        (int(record["fold"]), float(record["C_lambda"]))
        for record in proposed_fold_records
    }
    proposed_cv_complete = len(proposed_completed_keys) == n_folds * len(constants)
    proposed_path_records: list[dict] = []
    selected_c_lambda: float | None = None
    proposed_cv_summary: dict | None = None
    if proposed_cv_complete:
        proposed_path_records, selected_c_lambda = summarize_cv_path(
            proposed_fold_records,
            constants,
            n_folds,
        )
        if selected_c_lambda is None:
            raise RuntimeError("Completed proposed CV records did not select a C_lambda.")
        proposed_cv_summary = {
            "method": "proposed",
            "selected_C_lambda": float(selected_c_lambda),
            "cv_mean_test_negative_log_likelihood": float(
                next(
                    row["mean_test_negative_log_likelihood"]
                    for row in proposed_path_records
                    if np.isclose(row["C_lambda"], selected_c_lambda)
                )
            ),
            "cv_fold_standard_error": float(
                next(
                    row["fold_standard_error"]
                    for row in proposed_path_records
                    if np.isclose(row["C_lambda"], selected_c_lambda)
                )
            ),
            "cv_n_folds": int(n_folds),
            "cv_runtime_seconds": float(
                sum(float(record["runtime_seconds"]) for record in proposed_fold_records)
            ),
            "cv_path_records": proposed_path_records,
            "cv_fold_records": sorted(
                proposed_fold_records,
                key=lambda row: (int(row["fold"]), float(row["C_lambda"])),
            ),
            "cv_trace_records": trace_records,
        }
        if (
            run_proposed_final
            and proposed_final is None
            and proposed_intermediate_checkpoint.exists()
        ):
            proposed_full = load_pickle_checkpoint(proposed_intermediate_checkpoint)
            proposed_final = maybe_finalize_proposed_checkpoint(
                output_dir,
                proposed_cv_summary,
                proposed_full,
            )

    baseline_cv_summaries: dict[str, dict | None] = {
        method_name: None for method_name in baseline_methods
    }
    baseline_full_fits: dict[str, dict | None] = {
        method_name: None for method_name in baseline_methods
    }
    baseline_fold_records: dict[str, list[dict]] = {}
    for method_name in baseline_methods:
        cv_checkpoint = baseline_cv_checkpoint_path(output_dir, method_name)
        final_fit_checkpoint = baseline_final_fit_checkpoint_path(output_dir, method_name)
        intermediate_fit_checkpoint = baseline_full_data_only_checkpoint_path(
            output_dir, method_name
        )
        if cv_checkpoint.exists():
            baseline_cv_summaries[method_name] = load_pickle_checkpoint(cv_checkpoint)
            baseline_fold_records[method_name] = list(
                baseline_cv_summaries[method_name]["cv_fold_records"]
            )
            print(f"Skipping {method_name} CV summary: checkpoint already exists.")
        elif final_fit_checkpoint.exists():
            fitted = load_pickle_checkpoint(final_fit_checkpoint)
            baseline_cv_summaries[method_name] = extract_cv_summary_fields(fitted)
            baseline_fold_records[method_name] = list(
                baseline_cv_summaries[method_name]["cv_fold_records"]
            )
            atomic_checkpoint_pickle(baseline_cv_summaries[method_name], cv_checkpoint)
            print(f"Skipping {method_name} CV summary: found final fit checkpoint.")
        else:
            baseline_fold_records[method_name] = load_baseline_cv_progress(
                method_name,
                baseline_live_csv_path(output_dir, method_name),
                n_folds,
            )

        if final_fit_checkpoint.exists():
            baseline_full_fits[method_name] = load_pickle_checkpoint(final_fit_checkpoint)
            print(f"Skipping {method_name} full-data fit: final checkpoint already exists.")
        elif intermediate_fit_checkpoint.exists():
            baseline_full_fits[method_name] = load_pickle_checkpoint(intermediate_fit_checkpoint)
            print(
                f"Skipping {method_name} full-data fit: intermediate checkpoint already exists."
            )

    # Prepare reusable proposed CV initialization state once.
    assignment_dtype = np.uint8 if n_folds <= 255 else np.uint16
    fold_assignments = [np.zeros_like(Y_t, dtype=assignment_dtype) for Y_t in Y_blocks]
    fold_initializations = [None] * n_folds
    need_proposed_state = not proposed_cv_complete or (
        run_proposed_final and proposed_final is None
    )
    if need_proposed_state:
        print("Preparing proposed CV fold initializations for the shared global pool...")
    for fold_index, fold in enumerate(folds, start=1):
        for assignment, test_mask in zip(fold_assignments, fold["test_mask"]):
            if np.any(assignment[test_mask] != 0):
                raise RuntimeError("CV test folds overlap.")
            assignment[test_mask] = fold_index
        if need_proposed_state:
            fold_initializations[fold_index - 1] = module._spectral_initialization(
                [
                    np.where(mask_t, Y_t, 0.0)
                    for Y_t, mask_t in zip(Y_blocks, fold["training_mask"])
                ],
                fold["training_mask"],
                fit_kwargs.get("theta_bound", 4.0),
                fit_kwargs.get("a_lower_bound", 0.0),
                fit_kwargs.get("a_bound", 4.0),
                fit_kwargs.get("d_bound", 4.0),
            )
            write_initialization_bundle(
                fold_initializations[fold_index - 1],
                output_dir / f"proposed_cv_fold_{fold_index:02d}",
                model_ids,
                subjects,
            )

    pending_tasks: list[EstimationTask] = []
    if not proposed_cv_complete:
        for constant in constants:
            for fold_index in range(1, n_folds + 1):
                key = (fold_index, float(constant))
                if key not in proposed_completed_keys:
                    pending_tasks.append(
                        EstimationTask(
                            kind="proposed_cv",
                            fold_index=fold_index,
                            c_lambda=float(constant),
                        )
                    )
                else:
                    print(
                        f"Skipping completed proposed CV fold {fold_index}/{n_folds}, "
                        f"C_lambda={constant:.4g}."
                    )

    for method_name in baseline_methods:
        cv_kind, full_kind = baseline_kind_map[method_name]
        if baseline_cv_summaries[method_name] is None:
            completed_folds = {
                int(record["fold"]) for record in baseline_fold_records[method_name]
            }
            for fold_index in range(1, n_folds + 1):
                if fold_index not in completed_folds:
                    pending_tasks.append(
                        EstimationTask(
                            kind=cv_kind,
                            method=method_name,
                            fold_index=fold_index,
                        )
                    )
                else:
                    print(
                        f"Skipping completed {method_name} fold {fold_index}/{n_folds}."
                    )
        if baseline_full_fits[method_name] is None:
            pending_tasks.append(EstimationTask(kind=full_kind, method=method_name))

    proposed_full_submitted = (not run_proposed_final) or (proposed_final is not None)
    initial_task_count = len(pending_tasks) + (
        0 if proposed_full_submitted else 1
    )
    worker_count = min(
        cv_workers,
        max(1, len(pending_tasks) + (0 if proposed_full_submitted else 1)),
    )
    print(
        f"Scheduling one persistent global pool with up to {worker_count} workers."
    )
    print(
        f"Initially eligible tasks: proposed CV={sum(task.kind == 'proposed_cv' for task in pending_tasks)}, "
        f"local CV={sum(task.kind == 'local_cv' for task in pending_tasks)}, "
        f"global CV={sum(task.kind == 'global_cv' for task in pending_tasks)}, "
        f"naive CV={sum(task.kind == 'naive_cv' for task in pending_tasks)}, "
        f"baseline full fits={sum(task.kind in {'local_full', 'global_full', 'naive_full'} for task in pending_tasks)}"
    )

    cv_runtime_start = perf_counter()
    completed_task_counter = 0
    if pending_tasks or (
        run_proposed_final and proposed_cv_complete and proposed_final is None
    ):
        if "fork" in mp.get_all_start_methods():
            process_context = mp.get_context("fork")
            uses_spawn = False
        else:
            process_context = mp.get_context("spawn")
            uses_spawn = True
        _GLOBAL_PROCESS_STATE = (
            module,
            Y_blocks,
            folds,
            fit_kwargs,
            fold_assignments if need_proposed_state else None,
            fold_initializations if need_proposed_state else None,
        )
        try:
            executor_kwargs = {}
            if uses_spawn:
                executor_kwargs = {
                    "initializer": initialize_global_worker,
                    "initargs": (
                        str(Path(module.__file__).resolve()),
                        Y_blocks,
                        folds,
                        fit_kwargs,
                        fold_assignments if need_proposed_state else None,
                        fold_initializations if need_proposed_state else None,
                    ),
                }
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=process_context,
                **executor_kwargs,
            ) as executor:
                futures: dict = {}

                def submit_task(task: EstimationTask) -> None:
                    futures[executor.submit(run_estimation_task, task)] = task

                for task in pending_tasks:
                    submit_task(task)

                if (
                    run_proposed_final
                    and proposed_cv_complete
                    and proposed_final is None
                    and selected_c_lambda is not None
                ):
                    submit_task(
                        EstimationTask(kind="proposed_full", c_lambda=float(selected_c_lambda))
                    )
                    proposed_full_submitted = True

                while futures:
                    future = next(as_completed(futures))
                    task = futures.pop(future)
                    result = future.result()
                    completed_task_counter += 1

                    if task.kind == "proposed_cv":
                        proposed_fold_records.append(result["fold_record"])
                        trace_records.extend(result["trace_records"])
                        proposed_path_records, selected_c_lambda = persist_cv_progress(
                            output_dir,
                            proposed_fold_records,
                            trace_records,
                            constants,
                            n_folds,
                        )
                        print(
                            f"Completed proposed CV fold {task.fold_index}/{n_folds}, "
                            f"C_lambda={float(task.c_lambda):.4g}"
                        )
                        if (
                            run_proposed_final
                            and not proposed_full_submitted
                            and len(proposed_fold_records) == n_folds * len(constants)
                        ):
                            if selected_c_lambda is None:
                                raise RuntimeError("Proposed CV completed without selecting C_lambda.")
                            proposed_cv_summary = {
                                "method": "proposed",
                                "selected_C_lambda": float(selected_c_lambda),
                                "cv_mean_test_negative_log_likelihood": float(
                                    next(
                                        row["mean_test_negative_log_likelihood"]
                                        for row in proposed_path_records
                                        if np.isclose(row["C_lambda"], selected_c_lambda)
                                    )
                                ),
                                "cv_fold_standard_error": float(
                                    next(
                                        row["fold_standard_error"]
                                        for row in proposed_path_records
                                        if np.isclose(row["C_lambda"], selected_c_lambda)
                                    )
                                ),
                                "cv_n_folds": int(n_folds),
                                "cv_runtime_seconds": float(perf_counter() - cv_runtime_start),
                                "cv_path_records": proposed_path_records,
                                "cv_fold_records": sorted(
                                    proposed_fold_records,
                                    key=lambda row: (int(row["fold"]), float(row["C_lambda"])),
                                ),
                                "cv_trace_records": trace_records,
                            }
                            submit_task(
                                EstimationTask(
                                    kind="proposed_full",
                                    c_lambda=float(selected_c_lambda),
                                )
                            )
                            proposed_full_submitted = True
                            print(
                                f"Unlocked proposed full-data refit at selected C_lambda={selected_c_lambda:.4g}"
                            )
                    elif task.kind in {"local_cv", "global_cv", "naive_cv"}:
                        method_name = result["method_name"]
                        baseline_fold_records[method_name].append(result["fold_record"])
                        baseline_fold_records[method_name].sort(key=lambda row: int(row["fold"]))
                        persist_baseline_cv_progress(
                            method_name,
                            baseline_fold_records[method_name],
                            baseline_live_csv_path(output_dir, method_name),
                        )
                        print(
                            f"Completed {method_name} CV fold {task.fold_index}/{n_folds}"
                        )
                        if len(baseline_fold_records[method_name]) == n_folds:
                            summary = summarize_baseline_cv(
                                method_name,
                                baseline_fold_records[method_name],
                                float(
                                    sum(
                                        row["runtime_seconds"]
                                        for row in baseline_fold_records[method_name]
                                    )
                                ),
                                n_folds,
                            )
                            baseline_cv_summaries[method_name] = summary
                            atomic_checkpoint_pickle(
                                summary,
                                baseline_cv_checkpoint_path(output_dir, method_name),
                            )
                            print(f"Completed {method_name} CV summary checkpoint.")
                    elif task.kind in {"local_full", "global_full", "naive_full"}:
                        method_name = result["method_name"]
                        baseline_full_fits[method_name] = result["result"]
                        atomic_checkpoint_pickle(
                            baseline_full_fits[method_name],
                            baseline_full_data_only_checkpoint_path(output_dir, method_name),
                        )
                        print(f"Completed {method_name} full-data fit")
                    elif task.kind == "proposed_full":
                        proposed_full = result["result"]
                        atomic_checkpoint_pickle(
                            proposed_full,
                            proposed_intermediate_checkpoint,
                        )
                        if proposed_cv_summary is None:
                            if selected_c_lambda is None:
                                raise RuntimeError("Proposed full fit completed before CV summary existed.")
                            proposed_cv_summary = {
                                "method": "proposed",
                                "selected_C_lambda": float(selected_c_lambda),
                                "cv_mean_test_negative_log_likelihood": float(
                                    next(
                                        row["mean_test_negative_log_likelihood"]
                                        for row in proposed_path_records
                                        if np.isclose(row["C_lambda"], selected_c_lambda)
                                    )
                                ),
                                "cv_fold_standard_error": float(
                                    next(
                                        row["fold_standard_error"]
                                        for row in proposed_path_records
                                        if np.isclose(row["C_lambda"], selected_c_lambda)
                                    )
                                ),
                                "cv_n_folds": int(n_folds),
                                "cv_runtime_seconds": float(perf_counter() - cv_runtime_start),
                                "cv_path_records": proposed_path_records,
                                "cv_fold_records": sorted(
                                    proposed_fold_records,
                                    key=lambda row: (int(row["fold"]), float(row["C_lambda"])),
                                ),
                                "cv_trace_records": trace_records,
                            }
                        proposed_final = maybe_finalize_proposed_checkpoint(
                            output_dir,
                            proposed_cv_summary,
                            proposed_full,
                        )
                        print("Completed proposed full-data refit")

                    print(
                        f"global-pool completed_tasks={completed_task_counter}/"
                        f"{max(initial_task_count, completed_task_counter)}"
                    )
        finally:
            _GLOBAL_PROCESS_STATE = None

    if proposed_cv_summary is None:
        proposed_path_records, selected_c_lambda = summarize_cv_path(
            proposed_fold_records,
            constants,
            n_folds,
        )
        if selected_c_lambda is None:
            raise RuntimeError("Proposed CV did not produce a selected C_lambda.")
        proposed_cv_summary = {
            "method": "proposed",
            "selected_C_lambda": float(selected_c_lambda),
            "cv_mean_test_negative_log_likelihood": float(
                next(
                    row["mean_test_negative_log_likelihood"]
                    for row in proposed_path_records
                    if np.isclose(row["C_lambda"], selected_c_lambda)
                )
            ),
            "cv_fold_standard_error": float(
                next(
                    row["fold_standard_error"]
                    for row in proposed_path_records
                    if np.isclose(row["C_lambda"], selected_c_lambda)
                )
            ),
            "cv_n_folds": int(n_folds),
            "cv_runtime_seconds": float(perf_counter() - cv_runtime_start),
            "cv_path_records": proposed_path_records,
            "cv_fold_records": sorted(
                proposed_fold_records,
                key=lambda row: (int(row["fold"]), float(row["C_lambda"])),
            ),
            "cv_trace_records": trace_records,
        }

    if (
        run_proposed_final
        and proposed_final is None
        and proposed_intermediate_checkpoint.exists()
    ):
        proposed_full = load_pickle_checkpoint(proposed_intermediate_checkpoint)
        proposed_final = maybe_finalize_proposed_checkpoint(
            output_dir,
            proposed_cv_summary,
            proposed_full,
        )
    if run_proposed_final and proposed_final is None:
        raise RuntimeError("Unified scheduler finished without a proposed final fit.")

    finalized_baseline_fits: dict[str, dict] = {}
    for method_name in baseline_methods:
        if (
            baseline_cv_summaries[method_name] is None
            and len(baseline_fold_records[method_name]) == n_folds
        ):
            summary = summarize_baseline_cv(
                method_name,
                baseline_fold_records[method_name],
                float(sum(row["runtime_seconds"] for row in baseline_fold_records[method_name])),
                n_folds,
            )
            baseline_cv_summaries[method_name] = summary
            atomic_checkpoint_pickle(
                summary,
                baseline_cv_checkpoint_path(output_dir, method_name),
            )
        if baseline_full_fits[method_name] is None:
            final_path = baseline_final_fit_checkpoint_path(output_dir, method_name)
            intermediate_path = baseline_full_data_only_checkpoint_path(output_dir, method_name)
            if final_path.exists():
                baseline_full_fits[method_name] = load_pickle_checkpoint(final_path)
            elif intermediate_path.exists():
                baseline_full_fits[method_name] = load_pickle_checkpoint(intermediate_path)
        finalized = maybe_finalize_baseline_checkpoint(
            output_dir,
            method_name,
            baseline_cv_summaries[method_name],
            baseline_full_fits[method_name],
        )
        if finalized is None:
            raise RuntimeError(f"Unified scheduler finished without {method_name} final fit.")
        finalized_baseline_fits[method_name] = finalized

    return (
        proposed_final if run_proposed_final else proposed_cv_summary,
        {
            method_name: baseline_cv_summaries[method_name]
            for method_name in baseline_methods
        },
        finalized_baseline_fits,
    )


def candidate_checkpoint_path(
    output_dir: Path,
    candidate_index: int,
    c_lambda: float,
) -> Path:
    label = f"{c_lambda:g}".replace(".", "p")
    return output_dir / f"proposed_candidate_{candidate_index:02d}_C_{label}.pkl"


def run_full_data_grid(
    args: argparse.Namespace,
    module,
    Y_blocks: list[np.ndarray],
    subjects: list[str],
    common_models: list[str],
    constant_filter_manifest: pd.DataFrame,
    blocks: list[dict],
    root_output_dir: Path,
) -> None:
    global _FULL_DATA_PROCESS_STATE
    output_dir = root_output_dir / "full_data_grid"
    output_dir.mkdir(parents=True, exist_ok=True)
    constants = resolved_c_lambda_grid(args.c_lambda_grid)
    J_vec = np.array([block["Y"].shape[1] for block in blocks], dtype=int)
    run_config = pd.DataFrame(
        [
            {
                "analysis_mode": "full_data_grid_no_cv",
                "matrix_dir": args.matrix_dir,
                "summary_csv": args.summary_csv or "",
                "matrix_parts": ";".join(args.matrix_parts),
                "item_metadata_csv": args.item_metadata_csv,
                "item_index_map_csv": args.item_index_map_csv,
                "python_script": args.python_script,
                "output_dir": output_dir.as_posix(),
                "estimator_version": ESTIMATOR_VERSION,
                "common_models": len(common_models),
                "domains": len(Y_blocks),
                "total_items": int(J_vec.sum()),
                "C_lambda_grid": ";".join(f"{value:g}" for value in constants),
                "theta_bound": args.theta_bound,
                "a_lower_bound": args.a_lower_bound,
                "a_bound": args.a_bound,
                "d_bound": args.d_bound,
                "learning_rate": args.learning_rate,
                "learning_rate_decay_factor": args.learning_rate_decay_factor,
                "learning_rate_decay_steps": args.learning_rate_decay_steps,
                "max_iter": args.max_iter,
                "tolerance": args.tolerance,
                "cv_workers": args.cv_workers,
            }
        ]
    )
    config_path = output_dir / "run_config.csv"
    validate_resume_config(run_config, config_path)
    completion_path = output_dir / "run_complete.csv"
    if completion_path.exists():
        print(f"Skipping completed full-data grid: {output_dir.resolve()}")
        return
    atomic_to_csv(run_config, config_path)
    atomic_to_csv(pd.DataFrame({"C_lambda": constants}), output_dir / "candidate_lambda_grid.csv")
    atomic_to_csv(pd.DataFrame({"model_id": common_models}), output_dir / "model_manifest.csv")
    atomic_to_csv(
        pd.DataFrame({"subject": subjects, "items": J_vec}),
        output_dir / "domain_manifest.csv",
    )
    atomic_to_csv(
        constant_filter_manifest,
        output_dir / "constant_item_filter_manifest.csv",
    )

    fit_kwargs = {
        "theta_bound": args.theta_bound,
        "a_lower_bound": args.a_lower_bound,
        "a_bound": args.a_bound,
        "d_bound": args.d_bound,
        "learning_rate": args.learning_rate,
        "learning_rate_decay_factor": args.learning_rate_decay_factor,
        "learning_rate_decay_steps": args.learning_rate_decay_steps,
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
    }
    candidate_results: dict[int, dict] = {}
    pending: list[tuple[int, float, Path]] = []
    for candidate_index, c_lambda in enumerate(constants, start=1):
        checkpoint = candidate_checkpoint_path(output_dir, candidate_index, c_lambda)
        fitted = load_pickle_checkpoint(checkpoint)
        if fitted is None:
            pending.append((candidate_index, c_lambda, checkpoint))
        else:
            candidate_results[candidate_index] = fitted

    baseline_results = {}
    pending_baselines: list[tuple[str, Path]] = []
    for method_name in ("local", "global"):
        checkpoint = output_dir / f"{method_name}_fit_checkpoint.pkl"
        fitted = load_pickle_checkpoint(checkpoint)
        if fitted is None:
            pending_baselines.append((method_name, checkpoint))
        else:
            baseline_results[method_name] = fitted

    initialization = None
    if pending:
        print("Computing one shared full-data score/logistic initialization...")
        full_mask = module_prepare_observation_masks(module, Y_blocks, None)
        initialization = module._spectral_initialization(
            Y_blocks,
            full_mask,
            fit_kwargs.get("theta_bound", 4.0),
            fit_kwargs.get("a_lower_bound", 0.0),
            fit_kwargs.get("a_bound", 4.0),
            fit_kwargs.get("d_bound", 4.0),
        )
        write_initialization_bundle(
            initialization,
            output_dir / "proposed_full_data",
            common_models,
            subjects,
        )
    elif not pending_baselines:
        print("All complete-data C_lambda fits are already checkpointed.")

    total_pending_jobs = len(pending) + len(pending_baselines)
    if total_pending_jobs:
        if "fork" in mp.get_all_start_methods():
            process_context = mp.get_context("fork")
            uses_spawn = False
        else:
            process_context = mp.get_context("spawn")
            uses_spawn = True
        _FULL_DATA_PROCESS_STATE = (module, Y_blocks, initialization)
        try:
            executor_kwargs = {}
            if uses_spawn:
                executor_kwargs = {
                    "initializer": initialize_full_data_worker,
                    "initargs": (
                        str(Path(module.__file__).resolve()),
                        Y_blocks,
                        initialization,
                    ),
                }
            worker_count = min(args.cv_workers, total_pending_jobs)
            print(
                f"Scheduling {total_pending_jobs} full-data jobs with "
                f"{worker_count} workers: proposed candidates={len(pending)}, "
                f"baselines={len(pending_baselines)}."
            )
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=process_context,
                **executor_kwargs,
            ) as executor:
                futures = {}
                for candidate_index, c_lambda, checkpoint in pending:
                    futures[
                        executor.submit(
                            fit_full_data_candidate_task,
                            candidate_index,
                            c_lambda,
                            fit_kwargs,
                        )
                    ] = ("proposed", candidate_index, c_lambda, checkpoint)
                for method_name, checkpoint in pending_baselines:
                    futures[
                        executor.submit(
                            fit_full_data_baseline_task,
                            method_name,
                            fit_kwargs,
                        )
                    ] = ("baseline", method_name, checkpoint)

                for future in as_completed(futures):
                    metadata = futures[future]
                    if metadata[0] == "proposed":
                        _, candidate_index, c_lambda, checkpoint = metadata
                        _, _, fitted = future.result()
                        candidate_results[candidate_index] = fitted
                        atomic_checkpoint_pickle(fitted, checkpoint)
                        print(
                            f"Completed full-data candidate {candidate_index}/{len(constants)}, "
                            f"C_lambda={c_lambda:g}, objective={fitted['objective']:.6f}."
                        )
                    else:
                        _, method_name, checkpoint = metadata
                        _, fitted = future.result()
                        baseline_results[method_name] = fitted
                        atomic_checkpoint_pickle(fitted, checkpoint)
                        print(f"Completed full-data {method_name} baseline.")
        finally:
            _FULL_DATA_PROCESS_STATE = None

    summary_rows = []
    for candidate_index, c_lambda in enumerate(constants, start=1):
        fitted = candidate_results[candidate_index]
        prefix = candidate_checkpoint_path(
            output_dir, candidate_index, c_lambda
        ).with_suffix("")
        write_result_bundle(fitted, prefix, common_models, subjects)
        summary_rows.append(
            {
                "candidate_index": candidate_index,
                "C_lambda": c_lambda,
                "lambda_value": float(fitted["lambda_value"]),
                "objective": float(fitted["objective"]),
                "negative_log_likelihood": float(fitted["negative_log_likelihood"]),
                "mean_training_negative_log_likelihood": float(
                    raw_mean_training_negative_log_likelihood(fitted)
                ),
                "penalty": float(fitted["penalty"]),
                "iterations": int(fitted["iterations"]),
                "converged": bool(fitted["converged"]),
                "runtime_seconds": float(fitted["runtime_seconds"]),
            }
        )
    atomic_to_csv(pd.DataFrame(summary_rows), output_dir / "full_data_candidate_summary.csv")

    for method_name in ("local", "global"):
        fitted = baseline_results[method_name]
        write_result_bundle(
            fitted,
            output_dir / method_name,
            common_models,
            subjects,
        )
    atomic_to_csv(
        pd.DataFrame(
            [
                {
                    **result_summary_row(method_name, fitted),
                    "mean_training_negative_log_likelihood": float(
                        raw_mean_training_negative_log_likelihood(fitted)
                    ),
                    "runtime_seconds": float(fitted["runtime_seconds"]),
                }
                for method_name, fitted in baseline_results.items()
            ]
        ),
        output_dir / "local_global_summary.csv",
    )
    atomic_to_csv(
        pd.DataFrame([{"completed_utc": pd.Timestamp.utcnow().isoformat()}]),
        completion_path,
    )
    print(f"Finished full-data grid. Outputs written to: {output_dir.resolve()}")


def run_one_seed(
    args: argparse.Namespace,
    module,
    Y_blocks: list[np.ndarray],
    subjects: list[str],
    common_models: list[str],
    constant_filter_manifest: pd.DataFrame,
    blocks: list[dict],
    seed: int,
    output_dir: Path,
) -> None:
    J_vec = np.array([block["Y"].shape[1] for block in blocks], dtype=int)
    c_lambda_values = resolved_c_lambda_grid(args.c_lambda_grid)

    run_config = pd.DataFrame(
        [
            {
                "matrix_dir": args.matrix_dir,
                "summary_csv": args.summary_csv or "",
                "matrix_parts": ";".join(args.matrix_parts),
                "item_metadata_csv": args.item_metadata_csv,
                "item_index_map_csv": args.item_index_map_csv,
                "python_script": args.python_script,
                "output_dir": output_dir.as_posix(),
                "estimator_version": ESTIMATOR_VERSION,
                "common_models": len(common_models),
                "domains": len(Y_blocks),
                "total_items": int(J_vec.sum()),
                "constant_all_zero_removed": int(
                    constant_filter_manifest["all_zero_items_removed"].sum()
                ),
                "constant_all_one_removed": int(
                    constant_filter_manifest["all_one_items_removed"].sum()
                ),
                "C_lambda_grid": ";".join(f"{value:g}" for value in c_lambda_values),
                "n_folds": args.n_folds,
                "seed": seed,
                "theta_bound": args.theta_bound,
                "a_lower_bound": args.a_lower_bound,
                "a_bound": args.a_bound,
                "d_bound": args.d_bound,
                "learning_rate": args.learning_rate,
                "learning_rate_decay_factor": args.learning_rate_decay_factor,
                "learning_rate_decay_steps": args.learning_rate_decay_steps,
                "max_iter": args.max_iter,
                "tolerance": args.tolerance,
                "cv_workers": args.cv_workers,
            }
        ]
    )
    config_path = output_dir / "run_config.csv"
    validate_resume_config(run_config, config_path)
    completion_path = output_dir / "run_complete.csv"
    if completion_path.exists():
        print(f"Skipping completed seed={seed}: {output_dir.resolve()}")
        return
    atomic_to_csv(run_config, config_path)

    atomic_to_csv(
        pd.DataFrame(
            {
                "subject": subjects,
                "items": J_vec,
                "matrix_path": [block["matrix_path"] for block in blocks],
            }
        ),
        output_dir / "domain_manifest.csv",
    )
    atomic_to_csv(
        constant_filter_manifest,
        output_dir / "constant_item_filter_manifest.csv",
    )
    atomic_to_csv(
        pd.DataFrame({"model_id": common_models}),
        output_dir / "model_manifest.csv",
    )
    print(
        f"Loaded {len(Y_blocks)} MMLU domains with {len(common_models)} common models "
        f"and {int(J_vec.sum())} total items."
    )
    print(
        "Removed constant items before estimation: "
        f"all-zero={int(constant_filter_manifest['all_zero_items_removed'].sum())}, "
        f"all-one={int(constant_filter_manifest['all_one_items_removed'].sum())}."
    )

    atomic_to_csv(
        pd.DataFrame({"C_lambda": c_lambda_values}),
        output_dir / "candidate_lambda_grid.csv",
    )

    fit_kwargs = {
        "theta_bound": args.theta_bound,
        "a_lower_bound": args.a_lower_bound,
        "a_bound": args.a_bound,
        "d_bound": args.d_bound,
        "learning_rate": args.learning_rate,
        "learning_rate_decay_factor": args.learning_rate_decay_factor,
        "learning_rate_decay_steps": args.learning_rate_decay_steps,
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
    }
    full_mask = module_prepare_observation_masks(module, Y_blocks, None)
    folds = module.make_entrywise_cv_folds(
        full_mask,
        n_folds=args.n_folds,
        seed=seed + 100_003,
    )

    print("Running multiprocess cross-validated proposed estimator...")
    proposed_runtime_start = perf_counter()
    (
        proposed,
        baseline_cv_summaries,
        baseline_final_fits,
    ) = run_all_estimations_multiprocess(
        module,
        Y_blocks,
        folds,
        output_dir,
        common_models,
        subjects,
        fit_kwargs,
        c_lambda_values,
        args.n_folds,
        args.cv_workers,
        run_baselines=not args.proposed_only,
        run_proposed_final=not args.proposed_only,
    )
    proposed_runtime = perf_counter() - proposed_runtime_start

    cv_fold_frames = [pd.DataFrame(proposed["cv_fold_records"]).assign(method="proposed")]
    fit_summary_rows = []
    if not args.proposed_only:
        fit_summary_rows.append(
            {**result_summary_row("proposed", proposed), "runtime_seconds": proposed_runtime}
        )

    if not args.proposed_only:
        local_cv = baseline_cv_summaries["local"]
        global_cv = baseline_cv_summaries["global"]
        naive_cv = baseline_cv_summaries["naive_domain_mean"]
        local_fit = baseline_final_fits["local"]
        global_fit = baseline_final_fits["global"]
        naive_fit = baseline_final_fits["naive_domain_mean"]

        local_runtime = float(local_fit.get("runtime_seconds", np.nan))
        global_runtime = float(global_fit.get("runtime_seconds", np.nan))
        naive_runtime = float(naive_fit.get("runtime_seconds", np.nan))

        cv_fold_frames.extend(
            [
                pd.DataFrame(local_cv["cv_fold_records"]),
                pd.DataFrame(global_cv["cv_fold_records"]),
                pd.DataFrame(naive_cv["cv_fold_records"]),
            ]
        )
        fit_summary_rows.extend(
            [
                {**result_summary_row("local", local_fit), "runtime_seconds": local_runtime},
                {
                    **result_summary_row("global", global_fit),
                    "runtime_seconds": global_runtime,
                },
                {
                    **result_summary_row("naive_domain_mean", naive_fit),
                    "runtime_seconds": naive_runtime,
                },
            ]
        )

    cv_fold_comparison = pd.concat(
        cv_fold_frames,
        ignore_index=True,
    )
    cv_fold_comparison = cv_fold_comparison.sort_values(
        ["method", "fold", "C_lambda"], na_position="last"
    )
    atomic_to_csv(cv_fold_comparison, output_dir / "baseline_cv_folds.csv")

    if fit_summary_rows:
        fit_summary = pd.DataFrame(fit_summary_rows)
        atomic_to_csv(fit_summary, output_dir / "fit_summary.csv")

    write_result_bundle(proposed, output_dir / "proposed", common_models, subjects)
    if not args.proposed_only:
        write_result_bundle(local_fit, output_dir / "local", common_models, subjects)
        write_result_bundle(global_fit, output_dir / "global", common_models, subjects)
        write_result_bundle(
            naive_fit,
            output_dir / "naive_domain_mean",
            common_models,
            subjects,
        )

    atomic_to_csv(
        pd.DataFrame(
            [{"seed": seed, "completed_utc": pd.Timestamp.utcnow().isoformat()}]
        ),
        completion_path,
    )

    print(f"Finished. Outputs written to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    module = load_ai_measurement_module(Path(args.python_script))
    if args.summary_csv is None:
        blocks, common_models = load_mmlu_matrix_parts(
            [Path(path) for path in args.matrix_parts],
            Path(args.item_metadata_csv),
            Path(args.item_index_map_csv),
        )
    else:
        blocks, common_models = load_mmlu_blocks(Path(args.summary_csv))
    blocks = drop_miscellaneous_blocks(blocks)
    blocks, constant_filter_manifest = filter_constant_items(blocks)

    Y_blocks = [block["Y"] for block in blocks]
    subjects = [block["subject"] for block in blocks]
    if args.full_data_grid:
        run_full_data_grid(
            args,
            module,
            Y_blocks,
            subjects,
            common_models,
            constant_filter_manifest,
            blocks,
            root_output_dir,
        )
        return

    seeds = args.seeds if args.seeds is not None else [args.seed, args.seed + 1]
    atomic_to_csv(
        pd.DataFrame(
            {
                "run_index": np.arange(1, len(seeds) + 1, dtype=int),
                "seed": seeds,
                "output_subdir": [f"seed_{seed}" for seed in seeds],
            }
        ),
        root_output_dir / "seed_manifest.csv",
    )

    for run_index, seed in enumerate(seeds, start=1):
        output_dir = root_output_dir / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting seed run {run_index}/{len(seeds)} with seed={seed}")
        run_one_seed(
            args,
            module,
            Y_blocks,
            subjects,
            common_models,
            constant_filter_manifest,
            blocks,
            seed,
            output_dir,
        )


if __name__ == "__main__":
    main()
