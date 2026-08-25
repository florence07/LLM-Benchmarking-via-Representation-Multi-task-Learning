#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from settings_unequalJ_T18_R100 import H_VALUES, R, TABLE_FILENAMES, TOTAL_TASKS
from simulation_cv_lambda_scaled_T18_R100 import _build_comparison_summary


DEFAULT_INPUT_ROOT = Path(__file__).resolve().parent / "results_unequalJ_T18_R100"


def completed_task_dirs(root: Path) -> tuple[list[Path], list[int]]:
    completed = []
    missing = []
    for task_id in range(TOTAL_TASKS):
        task_dir = root / f"task_{task_id:04d}"
        required = [task_dir / "DONE"]
        required.extend(task_dir / filename for filename in TABLE_FILENAMES.values())
        if all(path.is_file() for path in required):
            completed.append(task_dir)
        else:
            missing.append(task_id)
    return completed, missing


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def combine_results(
    input_root: Path,
    output_dir: Path,
    *,
    require_complete: bool,
) -> dict[str, Path]:
    completed, missing = completed_task_dirs(input_root)
    print(f"Found {len(completed)} completed tasks out of {TOTAL_TASKS}.")
    if missing:
        preview = ", ".join(map(str, missing[:30]))
        suffix = " ..." if len(missing) > 30 else ""
        print(f"Missing task IDs ({len(missing)}): {preview}{suffix}")
        if require_complete:
            raise SystemExit("Not all tasks are complete.")
    if not completed:
        raise SystemExit("No completed tasks were found.")

    combined = {}
    for table_name, filename in TABLE_FILENAMES.items():
        frames = [pd.read_csv(task_dir / filename) for task_dir in completed]
        combined[table_name] = pd.concat(frames, ignore_index=True)

    raw = combined["raw"].sort_values(
        ["setting_id", "repeat", "method"], na_position="last"
    ).reset_index(drop=True)
    domains = combined["domains"].sort_values(
        ["setting_id", "repeat", "method", "domain"], na_position="last"
    ).reset_index(drop=True)
    cv_tuning = combined["cv_tuning"].sort_values(
        ["setting_id", "repeat", "C_lambda"]
    ).reset_index(drop=True)
    cv_folds = combined["cv_folds"].sort_values(
        ["setting_id", "repeat", "fold", "C_lambda"]
    ).reset_index(drop=True)
    summary = _build_comparison_summary(raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary": (summary, "simulation_summary.csv"),
        "raw": (raw, "simulation_raw_results.csv"),
        "domains": (domains, "simulation_domain_results.csv"),
        "cv_tuning": (cv_tuning, "simulation_cv_tuning_results.csv"),
        "cv_folds": (cv_folds, "simulation_cv_fold_results.csv"),
    }
    paths = {}
    for name, (frame, filename) in tables.items():
        path = output_dir / filename
        write_frame(frame, path)
        paths[name] = path
        print(f"{name}: {path} ({len(frame)} rows)")

    audit = (
        raw.groupby("h_target")
        .agg(repetitions=("repeat", "nunique"), raw_rows=("method", "size"))
        .reset_index()
    )
    audit["expected_repetitions"] = R
    audit["expected_raw_rows"] = 3 * R
    audit_path = output_dir / "completion_audit.csv"
    write_frame(audit, audit_path)
    paths["audit"] = audit_path
    print(f"audit: {audit_path}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / "combined"
    )
    combine_results(
        input_root,
        output_dir,
        require_complete=not args.allow_incomplete,
    )


if __name__ == "__main__":
    main()
