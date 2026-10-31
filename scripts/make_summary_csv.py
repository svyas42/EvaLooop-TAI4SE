"""Builds summary.csv for the multilingual summarization experiment.

Writes one row per (model, configuration) run with the header
``model,configuration,n_tasks,success_rate,asl_score``:

- ``success_rate`` is pass@1 / T and ``asl_score`` is ASL_pow, both computed with
  ``evaloop.analysis.metrics`` for a maximum of 10 cycles.
- Every row comes from a results file written by ``evaloop evaluate``, including the
  standard-loop baseline runs (``--loop_type code_generation_summarization``, the default).

Usage, from the repository root after ``pip install -e .``::

    python scripts/make_summary_csv.py
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from evaloop.analysis.metrics import asl_pow, pass_at_k

MAX_CYCLES = 10

# summary.csv configuration name -> loop type recorded in the results file
LOOP_TYPES = {
    "standard-mbpp-plus": "code_generation_summarization",
    "summarization_hindi": "summarization_hindi",
    "summarization_spanish": "summarization_spanish",
}

# model_id in the data package -> {configuration: results file in --results_dir}
RUNS = {
    "qwen2.5-coder-7b-instruct": {
        "standard-mbpp-plus": "evaloop_qwen2_5_coder_7b_instruct_mbpp_plus_20261031_000111_code_generation_summarization_results.json",
        "summarization_hindi": "evaloop_qwen2_5_coder_7b_instruct_mbpp_plus_20261030_171019_summarization_hindi_results.json",
        "summarization_spanish": "evaloop_qwen2_5_coder_7b_instruct_mbpp_plus_20261030_174348_summarization_spanish_results.json",
    },
    "seed-coder-8b-instruct": {
        "standard-mbpp-plus": "evaloop_seed_coder_8b_instruct_mbpp_plus_20261031_001503_code_generation_summarization_results.json",
        "summarization_hindi": "evaloop_seed_coder_8b_instruct_mbpp_plus_20261030_175817_summarization_hindi_results.json",
        "summarization_spanish": "evaloop_seed_coder_8b_instruct_mbpp_plus_20261030_182542_summarization_spanish_results.json",
    },
}


def load_run(path: Path, loop_type: str) -> List[Dict[str, Any]]:
    """Returns the per-task results of a framework results file.

    Args:
        path: Results file written by ``evaloop evaluate`` (one model, one loop type).
        loop_type: Loop type the file is expected to contain.

    Raises:
        ValueError: If the file does not hold one model run with the expected settings.
    """
    with open(path) as f:
        data = json.load(f)
    if len(data) != 1:
        raise ValueError(f"{path}: expected the results of one model, found {len(data)}")
    run = data[0]
    if run["max_cycles"] != MAX_CYCLES or run["loop_type"] != loop_type:
        raise ValueError(
            f"{path}: expected max_cycles={MAX_CYCLES} and loop_type={loop_type}, "
            f"found {run['max_cycles']} and {run['loop_type']}"
        )
    return run["prompt_results"]


def summary_row(model_id: str, configuration: str, tasks: List[Dict[str, Any]]) -> List[Any]:
    """Computes one summary.csv row from the per-task ``successful_cycles``."""
    cycles = [task["successful_cycles"] for task in tasks]
    success_rate = pass_at_k(cycles, 1) / len(cycles)
    return [model_id, configuration, len(cycles), f"{success_rate:.4f}", f"{asl_pow(cycles, MAX_CYCLES):.4f}"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build summary.csv for the multilingual loops.")
    parser.add_argument("--results_dir", type=Path, default=Path("results_hindi_spanish"),
                        help="Folder with the evaloop results files (default: results_hindi_spanish)")
    parser.add_argument("--output", type=Path, default=Path("summary.csv"),
                        help="Output CSV path (default: summary.csv)")
    args = parser.parse_args()

    rows = []
    for model_id, run_files in RUNS.items():
        runs = {
            configuration: load_run(args.results_dir / file_name, LOOP_TYPES[configuration])
            for configuration, file_name in run_files.items()
        }
        # The baseline and the multilingual loops must be compared on the same tasks
        task_ids = {task["task_id"] for task in runs["standard-mbpp-plus"]}
        if any({task["task_id"] for task in tasks} != task_ids for tasks in runs.values()):
            raise ValueError(f"{model_id}: the runs do not cover the same tasks")
        rows.extend(summary_row(model_id, configuration, tasks) for configuration, tasks in runs.items())

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "configuration", "n_tasks", "success_rate", "asl_score"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
