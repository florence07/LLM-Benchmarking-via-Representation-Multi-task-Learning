#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

from settings_unequalJ_T18_R100 import (
    BASE_SEED,
    C_LAMBDA_VALUES,
    CV_FOLDS,
    CV_JOBS,
    DATA_KWARGS,
    GLOBAL_FIT_KWARGS,
    H_VALUES,
    J_VECTOR_VALUES,
    LOCAL_FIT_KWARGS,
    N_VALUES,
    PROPOSED_CV_FIT_KWARGS,
    PROPOSED_THETA_PERTURBATION,
    R,
    TABLE_FILENAMES,
    TOTAL_TASKS,
    T_VALUES,
)
from simulation_cv_lambda_scaled_T18_R100 import run_three_estimator_comparison


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results_unequalJ_T18_R100"


def decode_task_id(task_id: int) -> tuple[int, int, float, int]:
    if not 0 <= task_id < TOTAL_TASKS:
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS - 1}].")
    h_index = task_id // R
    repeat_index = task_id % R
    return h_index, repeat_index, H_VALUES[h_index], repeat_index + 1


def task_is_complete(task_dir: Path) -> bool:
    required = [task_dir / "DONE"]
    required.extend(task_dir / filename for filename in TABLE_FILENAMES.values())
    return all(path.is_file() for path in required)


def write_frame(frame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def run_task(task_id: int, output_root: Path) -> Path:
    h_index, repeat_index, h_value, repeat_number = decode_task_id(task_id)
    data_seed = BASE_SEED + repeat_index
    task_dir = output_root / f"task_{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "DONE").unlink(missing_ok=True)

    print(
        f"Task {task_id + 1}/{TOTAL_TASKS}: h={h_value}, "
        f"repeat={repeat_number}, seed={data_seed}",
        flush=True,
    )

    results = run_three_estimator_comparison(
        N_values=N_VALUES,
        T_values=T_VALUES,
        J_vector_values=J_VECTOR_VALUES,
        h_values=[h_value],
        C_lambda_values=C_LAMBDA_VALUES,
        R=1,
        cv_folds=CV_FOLDS,
        cv_jobs=CV_JOBS,
        base_seed=data_seed,
        proposed_theta_perturbation=PROPOSED_THETA_PERTURBATION,
        data_kwargs=DATA_KWARGS,
        proposed_cv_fit_kwargs=PROPOSED_CV_FIT_KWARGS,
        local_fit_kwargs=LOCAL_FIT_KWARGS,
        global_fit_kwargs=GLOBAL_FIT_KWARGS,
        progress=True,
    )

    setting_id = h_index + 1
    for frame in results.values():
        if frame.empty:
            continue
        if "setting_id" in frame.columns:
            frame.loc[:, "setting_id"] = setting_id
        if "repeat" in frame.columns:
            frame.loc[:, "repeat"] = repeat_number
        if "seed" in frame.columns:
            frame.loc[:, "seed"] = data_seed

    for table_name, filename in TABLE_FILENAMES.items():
        write_frame(results[table_name], task_dir / filename)

    marker = task_dir / "DONE.tmp"
    marker.write_text(
        f"task_id={task_id}\nh={h_value}\nrepeat={repeat_number}\nseed={data_seed}\n",
        encoding="utf-8",
    )
    marker.replace(task_dir / "DONE")
    print(f"Completed task {task_id}: {task_dir}", flush=True)
    return task_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--end-task", type=int, default=TOTAL_TASKS - 1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-combine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.start_task <= args.end_task < TOTAL_TASKS:
        raise SystemExit(
            f"Task range must satisfy 0 <= start <= end < {TOTAL_TASKS}."
        )

    output_root = args.output_root.resolve()
    selected_count = args.end_task - args.start_task + 1
    print(f"Output directory: {output_root}")
    print(
        f"Selected tasks: {args.start_task}-{args.end_task} "
        f"({selected_count} of {TOTAL_TASKS})"
    )
    print(f"CV jobs per task: {CV_JOBS}")

    if args.dry_run:
        first = decode_task_id(args.start_task)
        last = decode_task_id(args.end_task)
        print(f"First task: h={first[2]}, repeat={first[3]}")
        print(f"Last task: h={last[2]}, repeat={last[3]}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    completed_now = 0
    skipped = 0
    for task_id in range(args.start_task, args.end_task + 1):
        task_dir = output_root / f"task_{task_id:04d}"
        if task_is_complete(task_dir) and not args.force:
            skipped += 1
            print(f"Skipping completed task {task_id}.", flush=True)
            continue
        run_task(task_id, output_root)
        completed_now += 1
        gc.collect()

    total_complete = sum(
        task_is_complete(output_root / f"task_{task_id:04d}")
        for task_id in range(TOTAL_TASKS)
    )
    print(
        f"Run finished: {completed_now} completed now, {skipped} skipped, "
        f"{total_complete}/{TOTAL_TASKS} complete in total."
    )

    if total_complete == TOTAL_TASKS and not args.no_combine:
        from combine_unequalJ_T18_R100 import combine_results

        combine_results(output_root, output_root / "combined", require_complete=True)
    elif not args.no_combine:
        print("Combined tables will be created after all tasks are complete.")


if __name__ == "__main__":
    main()
