#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


DEFAULT_C_LAMBDAS = [
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
_CV_PROCESS_STATE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public MMLU application pipeline: cross-validate the proposed "
            "estimator over C_lambda, then refit it on the full matrix."
        )
    )
    parser.add_argument(
        "--matrix-parts",
        nargs="+",
        default=[
            "real_data/item_response_matrix.part01.csv",
            "real_data/item_response_matrix.part02.csv",
        ],
        help="Model-by-item binary response matrix parts.",
    )
    parser.add_argument(
        "--item-metadata-csv",
        default="real_data/item_contents.csv",
        help="Item metadata with a subject column.",
    )
    parser.add_argument(
        "--item-index-map-csv",
        default="real_data/item_index_map.csv",
        help="Map from item metadata rows to matrix column names.",
    )
    parser.add_argument(
        "--estimator-script",
        required=True,
        help=(
            "Path to the estimator implementation that exposes fit_proposed_method, "
            "held_out_negative_log_likelihood, make_entrywise_cv_folds, and the "
            "initialization helpers used in the paper."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/proposed_cv_refit",
        help="Directory where CV summaries and final proposed-fit outputs are written.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--theta-bound", type=float, default=4.0)
    parser.add_argument("--a-lower-bound", type=float, default=0.0)
    parser.add_argument("--a-bound", type=float, default=4.0)
    parser.add_argument("--d-bound", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--learning-rate-decay-factor", type=float, default=0.5)
    parser.add_argument("--learning-rate-decay-steps", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--c-lambda-grid",
        nargs="+",
        type=float,
        default=None,
        help="Optional C_lambda grid. Defaults to the grid used in the paper.",
    )
    parser.add_argument(
        "--cv-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of concurrent CV fits.",
    )
    return parser.parse_args()


def load_estimator_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(
        "representation_multitask_estimator",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load estimator module from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def prepare_observation_masks(module, y_blocks, observation_mask):
    mask_preparer = getattr(module, "_prepare_observation_masks", None)
    if mask_preparer is not None:
        return mask_preparer(y_blocks, observation_mask)
    legacy_preparer = getattr(module, "_prepare_observation_mask", None)
    if legacy_preparer is not None:
        return legacy_preparer(y_blocks, observation_mask)
    raise AttributeError(
        "Estimator module must expose _prepare_observation_masks or "
        "_prepare_observation_mask."
    )


def load_mmlu_matrix_parts(
    matrix_parts: list[Path],
    item_metadata_csv: Path,
    item_index_map_csv: Path,
) -> tuple[list[dict], list[str]]:
    item_metadata = pd.read_csv(item_metadata_csv, usecols=["subject"])
    index_map = pd.read_csv(item_index_map_csv)
    required = {
        "item_content_row_index",
        "item_response_matrix_col_name",
    }
    missing = required.difference(index_map.columns)
    if missing:
        raise ValueError(
            "Item index map is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )

    expected_rows = np.arange(len(index_map))
    observed_rows = index_map["item_content_row_index"].to_numpy()
    if not np.array_equal(observed_rows, expected_rows):
        raise ValueError(
            "item_content_row_index must be consecutive and aligned with item_contents.csv."
        )
    if len(item_metadata) != len(index_map):
        raise ValueError("item_contents.csv and item_index_map.csv have different lengths.")

    expected_item_columns = (
        index_map["item_response_matrix_col_name"].astype(str).tolist()
    )
    model_ids: list[str] = []
    response_parts: list[np.ndarray] = []
    response_dtypes = {column: np.int8 for column in expected_item_columns}

    for path in matrix_parts:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if not columns or columns[0] != "model_id":
            raise ValueError(f"{path} must start with a model_id column.")
        if columns[1:] != expected_item_columns:
            raise ValueError(f"{path} columns do not match item_index_map.csv.")

        part = pd.read_csv(path, dtype=response_dtypes)
        model_ids.extend(part.pop("model_id").astype(str).tolist())
        response_matrix = part.to_numpy(dtype=np.int8, copy=False)
        if not np.isin(response_matrix, [0, 1]).all():
            raise ValueError(f"{path} contains responses outside 0/1.")
        response_parts.append(response_matrix)

    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Matrix parts contain duplicate model_id values.")

    response_matrix = np.concatenate(response_parts, axis=0)
    subject_values = item_metadata["subject"].astype(str).to_numpy()
    subjects = item_metadata["subject"].drop_duplicates().astype(str).tolist()

    blocks = []
    for subject in subjects:
        item_positions = np.flatnonzero(subject_values == subject)
        blocks.append(
            {
                "subject": subject.replace(" ", "_"),
                "models": model_ids,
                "Y": response_matrix[:, item_positions],
                "item_names": [expected_item_columns[index] for index in item_positions],
            }
        )
    return blocks, model_ids


def drop_miscellaneous_blocks(blocks: list[dict]) -> list[dict]:
    filtered = [
        block
        for block in blocks
        if str(block["subject"]).strip().lower() != "miscellaneous"
    ]
    if not filtered:
        raise ValueError("All subject blocks were removed after excluding miscellaneous.")
    return filtered


def filter_constant_items(blocks: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    filtered_blocks = []
    manifest_rows = []
    for block in blocks:
        y_block = block["Y"]
        item_names = np.asarray(block["item_names"], dtype=object)
        col_sums = y_block.sum(axis=0)
        n_models = y_block.shape[0]
        keep_mask = (col_sums > 0) & (col_sums < n_models)

        filtered_blocks.append(
            {
                **block,
                "Y": y_block[:, keep_mask],
                "item_names": item_names[keep_mask].tolist(),
            }
        )
        manifest_rows.append(
            {
                "subject": block["subject"],
                "models": int(n_models),
                "items_before_constant_filter": int(y_block.shape[1]),
                "all_zero_items_removed": int(np.sum(col_sums == 0)),
                "all_one_items_removed": int(np.sum(col_sums == n_models)),
                "items_after_constant_filter": int(np.sum(keep_mask)),
            }
        )
    return filtered_blocks, pd.DataFrame(manifest_rows)


def resolved_c_lambda_grid(c_lambda_grid: list[float] | None) -> list[float]:
    values = DEFAULT_C_LAMBDAS if c_lambda_grid is None else c_lambda_grid
    constants = np.asarray(values, dtype=float)
    if constants.ndim != 1 or constants.size == 0:
        raise ValueError("C_lambda grid must be a nonempty one-dimensional sequence.")
    if not np.all(np.isfinite(constants)) or np.any(constants < 0.0):
        raise ValueError("Every C_lambda value must be finite and nonnegative.")
    if np.unique(constants).size != constants.size:
        raise ValueError("C_lambda grid must not contain duplicates.")
    return sorted(float(value) for value in constants)


def theta_t_long(theta_hat: np.ndarray, model_ids: list[str], subjects: list[str]) -> pd.DataFrame:
    rows = []
    for subject, values in zip(subjects, theta_hat):
        rows.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "model_id": model_ids,
                    "theta_t_hat": np.asarray(values, dtype=float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def item_parameter_long(result: dict, subjects: list[str]) -> pd.DataFrame:
    rows = []
    a_hat = np.asarray(result["a_hat"], dtype=float)
    d_hat = np.asarray(result["d_hat"], dtype=float)
    j_vec = np.asarray(result["J_vec"], dtype=int)
    for subject, a_row, d_row, j_t in zip(subjects, a_hat, d_hat, j_vec):
        active = np.arange(int(j_t))
        rows.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "item_index": active + 1,
                    "a_hat": a_row[: int(j_t)],
                    "d_hat": d_row[: int(j_t)],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def write_result_bundle(
    result: dict,
    output_prefix: Path,
    model_ids: list[str],
    subjects: list[str],
) -> None:
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
        theta_t_long(np.asarray(result["theta_hat"], dtype=float), model_ids, subjects),
        output_prefix.with_name(output_prefix.name + "_theta_t_long.csv"),
    )
    atomic_to_csv(
        item_parameter_long(result, subjects),
        output_prefix.with_name(output_prefix.name + "_item_parameters.csv"),
    )


def raw_mean_training_negative_log_likelihood(result: dict) -> float:
    raw_nll = float(
        result.get("raw_negative_log_likelihood", result["negative_log_likelihood"])
    )
    return raw_nll / float(result["total_entries"])


def summarize_full_fit(method: str, result: dict, runtime_seconds: float) -> dict:
    iterations = result.get("iterations", np.nan)
    converged = result.get("converged", np.nan)
    if isinstance(iterations, np.ndarray):
        iterations_value = float(np.mean(iterations))
    else:
        iterations_value = float(iterations)
    if isinstance(converged, np.ndarray):
        converged_value = float(np.mean(converged))
    else:
        converged_value = float(bool(converged))
    return {
        "method": method,
        "lambda_value": float(result["lambda_value"]) if "lambda_value" in result else np.nan,
        "objective": float(result["objective"]),
        "negative_log_likelihood": float(result["negative_log_likelihood"]),
        "mean_training_negative_log_likelihood": float(
            raw_mean_training_negative_log_likelihood(result)
        ),
        "penalty": float(result["penalty"]),
        "iterations": iterations_value,
        "converged": converged_value,
        "runtime_seconds": float(runtime_seconds),
    }


def initialize_cv_worker(
    module_path: str,
    y_blocks: list[np.ndarray],
    fold_assignments: list[np.ndarray],
    fold_initializations: list,
) -> None:
    global _CV_PROCESS_STATE
    module = load_estimator_module(Path(module_path))
    _CV_PROCESS_STATE = (module, y_blocks, fold_assignments, fold_initializations)


def fit_candidate_task(fold_index: int, c_lambda: float, fit_kwargs: dict) -> dict:
    if _CV_PROCESS_STATE is None:
        raise RuntimeError("CV worker process was not initialized.")
    module, y_blocks, fold_assignments, fold_initializations = _CV_PROCESS_STATE
    test_mask = [assignment == fold_index for assignment in fold_assignments]
    training_mask = [
        (assignment != 0) & (assignment != fold_index)
        for assignment in fold_assignments
    ]
    fitted = module.fit_proposed_method(
        y_blocks,
        observation_mask=training_mask,
        lambda_constant=float(c_lambda),
        precomputed_initialization=fold_initializations[fold_index - 1],
        **fit_kwargs,
    )
    score = module.held_out_negative_log_likelihood(y_blocks, test_mask, fitted)

    training_entries_by_domain = np.asarray([mask.sum() for mask in training_mask])
    test_entries_by_domain = np.asarray([mask.sum() for mask in test_mask])
    return {
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
        "mean_test_negative_log_likelihood": float(
            score["mean_test_negative_log_likelihood"]
        ),
        "training_mean_negative_log_likelihood": float(
            raw_mean_training_negative_log_likelihood(fitted)
        ),
        "converged": bool(fitted["converged"]),
        "iterations": int(fitted["iterations"]),
    }


def summarize_cv_results(
    fold_records: list[dict],
    constants: list[float],
    n_folds: int,
) -> tuple[pd.DataFrame, float]:
    summary_rows = []
    for constant in constants:
        candidate_records = [
            row for row in fold_records if np.isclose(row["C_lambda"], constant)
        ]
        converged_records = [row for row in candidate_records if bool(row["converged"])]
        if len(converged_records) != n_folds:
            continue

        fold_means = np.asarray(
            [row["mean_test_negative_log_likelihood"] for row in converged_records],
            dtype=float,
        )
        summary_rows.append(
            {
                "C_lambda": float(constant),
                "completed_folds": int(len(converged_records)),
                "mean_test_negative_log_likelihood": float(fold_means.mean()),
                "fold_standard_error": float(fold_means.std(ddof=1) / np.sqrt(len(fold_means)))
                if len(fold_means) >= 2
                else np.nan,
                "mean_training_negative_log_likelihood": float(
                    np.mean(
                        [row["training_mean_negative_log_likelihood"] for row in converged_records]
                    )
                ),
                "mean_training_lambda_value": float(
                    np.mean([row["training_lambda_value"] for row in converged_records])
                ),
                "selected": False,
            }
        )

    if not summary_rows:
        raise RuntimeError("No C_lambda candidate converged on all CV folds.")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_test_negative_log_likelihood", "C_lambda"],
        kind="mergesort",
    )
    selected_c = float(summary_df.iloc[0]["C_lambda"])
    summary_df["selected"] = np.isclose(summary_df["C_lambda"], selected_c)
    summary_df = summary_df.sort_values("C_lambda", kind="mergesort").reset_index(drop=True)
    return summary_df, selected_c


def cross_validate_proposed(
    module,
    y_blocks: list[np.ndarray],
    constants: list[float],
    fit_kwargs: dict,
    *,
    n_folds: int,
    seed: int,
    cv_workers: int,
) -> tuple[list[dict], pd.DataFrame, float]:
    full_mask = prepare_observation_masks(module, y_blocks, None)
    folds = module.make_entrywise_cv_folds(
        full_mask,
        n_folds=n_folds,
        seed=seed + 100_003,
    )

    assignment_dtype = np.uint8 if n_folds <= 255 else np.uint16
    fold_assignments = [np.zeros_like(y_block, dtype=assignment_dtype) for y_block in y_blocks]
    fold_initializations = [None] * n_folds

    for fold_index, fold in enumerate(folds, start=1):
        for assignment, test_mask in zip(fold_assignments, fold["test_mask"]):
            if np.any(assignment[test_mask] != 0):
                raise RuntimeError("CV test folds overlap.")
            assignment[test_mask] = fold_index
        fold_initializations[fold_index - 1] = module._spectral_initialization(
            [
                np.where(mask_t, y_block, 0.0)
                for y_block, mask_t in zip(y_blocks, fold["training_mask"])
            ],
            fold["training_mask"],
            fit_kwargs["theta_bound"],
            fit_kwargs["a_lower_bound"],
            fit_kwargs["a_bound"],
            fit_kwargs["d_bound"],
        )

    tasks = [
        (fold_index, constant)
        for constant in constants
        for fold_index in range(1, n_folds + 1)
    ]
    process_context = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    uses_spawn = process_context.get_start_method() == "spawn"

    fold_records: list[dict] = []
    global _CV_PROCESS_STATE
    _CV_PROCESS_STATE = (module, y_blocks, fold_assignments, fold_initializations)
    try:
        executor_kwargs = {}
        if uses_spawn:
            executor_kwargs = {
                "initializer": initialize_cv_worker,
                "initargs": (
                    str(Path(module.__file__).resolve()),
                    y_blocks,
                    fold_assignments,
                    fold_initializations,
                ),
            }
        with ProcessPoolExecutor(
            max_workers=min(cv_workers, len(tasks)),
            mp_context=process_context,
            **executor_kwargs,
        ) as executor:
            futures = {
                executor.submit(fit_candidate_task, fold_index, constant, fit_kwargs): (
                    fold_index,
                    constant,
                )
                for fold_index, constant in tasks
            }
            for future in as_completed(futures):
                fold_index, constant = futures[future]
                fold_records.append(future.result())
                print(
                    f"Completed proposed CV fold {fold_index}/{n_folds}, "
                    f"C_lambda={constant:g}."
                )
    finally:
        _CV_PROCESS_STATE = None

    fold_records.sort(key=lambda row: (row["fold"], row["C_lambda"]))
    summary_df, selected_c = summarize_cv_results(fold_records, constants, n_folds)
    return fold_records, summary_df, selected_c


def build_fit_kwargs(args: argparse.Namespace) -> dict:
    return {
        "theta_bound": float(args.theta_bound),
        "a_lower_bound": float(args.a_lower_bound),
        "a_bound": float(args.a_bound),
        "d_bound": float(args.d_bound),
        "learning_rate": float(args.learning_rate),
        "learning_rate_decay_factor": float(args.learning_rate_decay_factor),
        "learning_rate_decay_steps": int(args.learning_rate_decay_steps),
        "max_iter": int(args.max_iter),
        "tolerance": float(args.tolerance),
    }


def main() -> None:
    args = parse_args()
    estimator_script = Path(args.estimator_script)
    if not estimator_script.exists():
        raise FileNotFoundError(
            "Estimator script not found. Pass --estimator-script with the paper's "
            f"implementation path. Missing path: {estimator_script}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_estimator_module(estimator_script)
    blocks, model_ids = load_mmlu_matrix_parts(
        [Path(path) for path in args.matrix_parts],
        Path(args.item_metadata_csv),
        Path(args.item_index_map_csv),
    )
    blocks = drop_miscellaneous_blocks(blocks)
    blocks, constant_filter_manifest = filter_constant_items(blocks)

    subjects = [block["subject"] for block in blocks]
    y_blocks = [block["Y"] for block in blocks]
    fit_kwargs = build_fit_kwargs(args)
    constants = resolved_c_lambda_grid(args.c_lambda_grid)

    atomic_to_csv(
        pd.DataFrame(
            {
                "subject": subjects,
                "items_after_constant_filter": [int(block["Y"].shape[1]) for block in blocks],
            }
        ),
        output_dir / "domain_manifest.csv",
    )
    atomic_to_csv(constant_filter_manifest, output_dir / "constant_item_filter_manifest.csv")
    atomic_to_csv(pd.DataFrame({"C_lambda": constants}), output_dir / "candidate_lambda_grid.csv")
    atomic_to_csv(
        pd.DataFrame(
            [
                {
                    "matrix_parts": ";".join(args.matrix_parts),
                    "item_metadata_csv": args.item_metadata_csv,
                    "item_index_map_csv": args.item_index_map_csv,
                    "estimator_script": str(estimator_script),
                    "output_dir": str(output_dir),
                    "n_folds": int(args.n_folds),
                    "seed": int(args.seed),
                    "common_models": len(model_ids),
                    "domains": len(subjects),
                    "total_items_after_constant_filter": int(sum(block["Y"].shape[1] for block in blocks)),
                    **fit_kwargs,
                }
            ]
        ),
        output_dir / "run_config.csv",
    )

    cv_start = perf_counter()
    fold_records, cv_summary_df, selected_c = cross_validate_proposed(
        module,
        y_blocks,
        constants,
        fit_kwargs,
        n_folds=int(args.n_folds),
        seed=int(args.seed),
        cv_workers=int(args.cv_workers),
    )
    atomic_to_csv(pd.DataFrame(fold_records), output_dir / "proposed_cv_folds.csv")
    atomic_to_csv(cv_summary_df, output_dir / "proposed_cv_summary.csv")

    full_masks = prepare_observation_masks(module, y_blocks, None)
    initialization = module._spectral_initialization(
        y_blocks,
        full_masks,
        fit_kwargs["theta_bound"],
        fit_kwargs["a_lower_bound"],
        fit_kwargs["a_bound"],
        fit_kwargs["d_bound"],
    )

    refit_start = perf_counter()
    proposed_fit = module.fit_proposed_method(
        y_blocks,
        lambda_constant=float(selected_c),
        precomputed_initialization=initialization,
        **fit_kwargs,
    )
    proposed_runtime = perf_counter() - refit_start

    local_start = perf_counter()
    local_fit = module.fit_local_method(y_blocks, **fit_kwargs)
    local_runtime = perf_counter() - local_start

    global_start = perf_counter()
    global_fit = module.fit_global_method(y_blocks, **fit_kwargs)
    global_runtime = perf_counter() - global_start

    write_result_bundle(proposed_fit, output_dir / "proposed", model_ids, subjects)
    write_result_bundle(local_fit, output_dir / "local", model_ids, subjects)
    write_result_bundle(global_fit, output_dir / "global", model_ids, subjects)
    atomic_to_csv(
        pd.DataFrame(
            [
                {
                    **summarize_full_fit("proposed", proposed_fit, proposed_runtime),
                    "selected_C_lambda": float(selected_c),
                    "cv_runtime_seconds": float(perf_counter() - cv_start),
                },
                summarize_full_fit("local", local_fit, local_runtime),
                summarize_full_fit("global", global_fit, global_runtime),
            ]
        ),
        output_dir / "fit_summary.csv",
    )
    atomic_to_csv(
        pd.DataFrame([{"completed_utc": pd.Timestamp.utcnow().isoformat()}]),
        output_dir / "run_complete.csv",
    )
    print(f"Finished. Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
