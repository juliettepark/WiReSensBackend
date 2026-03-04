#!/usr/bin/env python3
"""
Batch runner for heavy wrap internal processing.

Runs `processing_scripts/process_heavy_wrap_internal.py` on each immediate
subfolder of one or more source directories, using the subfolder name as the label.

Example:
  python3 processing_scripts/batch_process_heavy_wrap_internal.py

  # Reset output (v2) then process everything
  python3 processing_scripts/batch_process_heavy_wrap_internal.py --reset-output

  # Dry run
  python3 processing_scripts/batch_process_heavy_wrap_internal.py --dry-run

To change output file, change the RESULT_CSV variable in process_heavy_wrap_internal.py and below
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run process_heavy_wrap_internal.py over labeled subfolders."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["data/hwi/raw_data", "data/hwi/devin_data"],
        help="Source directories whose immediate subfolders are label folders.",
    )
    parser.add_argument(
        "--processor",
        default="processing_scripts/process_heavy_wrap_internal.py",
        help="Path to the processing script to run.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (defaults to current interpreter).",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete the v2 collapsed results CSV before processing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute them.",
    )
    return parser.parse_args()


def iter_label_folders(source_dir: Path) -> list[Path]:
    if not source_dir.exists() or not source_dir.is_dir():
        return []
    return sorted([p for p in source_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    processor = (project_root / args.processor).resolve()
    if not processor.exists():
        print(f"Error: processor script not found: {processor}")
        return 2

    if args.reset_output:
        # Keep this in sync with processing_scripts/process_heavy_wrap_internal.py
        out_csv = (
            project_root
            / "data"
            / "hwi"
            / "labeled_collapsed_results"
            / "heavy_wrap_internal_labeled_collapsed_results_v4.csv"
        )
        if out_csv.exists():
            if args.dry_run:
                print(f"[dry-run] would delete: {out_csv}")
            else:
                out_csv.unlink()
                print(f"Deleted: {out_csv}")

    sources = [(project_root / s).resolve() for s in args.sources]

    total_jobs = 0
    jobs: list[tuple[Path, str]] = []
    for src in sources:
        for label_dir in iter_label_folders(src):
            label = label_dir.name
            csv_count = len(list(label_dir.glob("*.csv")))
            if csv_count == 0:
                continue
            jobs.append((label_dir, label))
    total_jobs = len(jobs)

    if total_jobs == 0:
        print("No label folders with CSV files found.")
        return 0

    print(f"Found {total_jobs} label folder(s) to process.")

    for idx, (label_dir, label) in enumerate(jobs, start=1):
        cmd = [
            args.python,
            str(processor),
            str(label_dir),
            label,
        ]
        print(f"[{idx}/{total_jobs}] {label} <- {label_dir}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue

        result = subprocess.run(cmd, cwd=str(project_root))
        if result.returncode != 0:
            print(f"Error: processing failed for {label_dir} (label={label})")
            return result.returncode

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

