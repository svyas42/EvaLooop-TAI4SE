"""Prints the numbers reported in the assignment report from the six results files.

Covers every computed number in the report: the results table, the failure counts, the
comparison with the English runs, the example tasks, the run-to-run variation, the summary
checks, the selection of the inspected loops and the run times. The labels of the inspected
loops come from reading them and are not computed here.

The comparison with the assignment data package is printed only when ``--data_dir`` is given.

Usage, from the repository root after ``pip install -e .``::

    python scripts/report_numbers.py
    python scripts/report_numbers.py --data_dir /path/to/assignment4-data
"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from evaloop.analysis.metrics import asl_pow, pass_at_k, pass_drop
from evaloop.tasks.multilingual_prompts import SUMMARY_OPENINGS
from make_summary_csv import LOOP_TYPES, MAX_CYCLES, RUNS, load_run

MODEL_NAMES = {"qwen2.5-coder-7b-instruct": "Qwen2.5-Coder-7B", "seed-coder-8b-instruct": "Seed-Coder-8B"}
LANGUAGES = {"standard-mbpp-plus": "English", "summarization_hindi": "Hindi", "summarization_spanish": "Spanish"}
EXTENSION = ["summarization_hindi", "summarization_spanish"]
ENGLISH = "standard-mbpp-plus"
INSPECTED_PER_RUN = 5
LOG_TIME = "%Y-%m-%d %H:%M:%S"

Tasks = Dict[str, Dict[str, Any]]


def load_runs(results_dir: Path) -> Dict[str, Dict[str, Tasks]]:
    """Returns {model_id: {configuration: {task_id: task}}}, tasks in MBPP+ order."""
    return {
        model_id: {
            configuration: {task["task_id"]: task for task in load_run(results_dir / file_name, LOOP_TYPES[configuration])}
            for configuration, file_name in run_files.items()
        }
        for model_id, run_files in RUNS.items()
    }


def sustained(tasks: Tasks) -> List[int]:
    """Returns the ``successful_cycles`` of every task."""
    return [task["successful_cycles"] for task in tasks.values()]


def failed_after_cycle_1(task: Dict[str, Any]) -> bool:
    """True if the task passed cycle 1 and failed in a later cycle."""
    return 1 <= task["successful_cycles"] < MAX_CYCLES


def summaries(tasks: Tasks) -> List[str]:
    """Returns every raw summary of a run (a failing cycle also has one)."""
    return [cycle["task_2_raw_output"] or "" for task in tasks.values() for cycle in task["cycles"]]


def without_whitespace(code: str) -> str:
    """Removes all whitespace, so that only changes to the code itself count."""
    return re.sub(r"\s+", "", code)


def has_repeated_words(summary: str) -> bool:
    """Rough check: one word 5 times in a row, or one word pair 8 times or more."""
    words = summary.split()
    longest = streak = 1
    for previous, word in zip(words, words[1:]):
        streak = streak + 1 if word == previous else 1
        longest = max(longest, streak)
    pairs = Counter(zip(words, words[1:])).most_common(1)
    return longest >= 5 or bool(pairs and pairs[0][1] >= 8)


def inspected(tasks: Tasks) -> List[str]:
    """Selection rule of the manual inspection: the first tasks that passed cycle 1 and then failed."""
    return [task_id for task_id, task in tasks.items() if failed_after_cycle_1(task)][:INSPECTED_PER_RUN]


def print_results_table(runs: Dict[str, Dict[str, Tasks]]) -> None:
    print("== Results table ==")
    print(f"{'model':18} {'configuration':8} {'success_rate':>12} {'ASL_pow':>8} {'change':>7} {'pass_drop':>9} {'reach_10':>8}")
    for model_id, configs in runs.items():
        english = asl_pow(sustained(configs[ENGLISH]), MAX_CYCLES)
        for configuration, tasks in configs.items():
            cycles = sustained(tasks)
            score = asl_pow(cycles, MAX_CYCLES)
            change = "-" if configuration == ENGLISH else f"{score - english:+.3f}"
            print(f"{MODEL_NAMES[model_id]:18} {LANGUAGES[configuration]:8} {pass_at_k(cycles, 1) / len(cycles):12.4f} "
                  f"{score:8.3f} {change:>7} {pass_drop(cycles, MAX_CYCLES):9.3f} {pass_at_k(cycles, MAX_CYCLES):8d}")
    for model_id, configs in runs.items():
        rates = [pass_at_k(sustained(tasks), 1) / len(tasks) for tasks in configs.values()]
        print(f"{MODEL_NAMES[model_id]}: largest success-rate difference between configurations {max(rates) - min(rates):.4f}")
    qwen, seed = runs.values()
    for configuration in LANGUAGES:
        gap = asl_pow(sustained(seed[configuration]), MAX_CYCLES) - asl_pow(sustained(qwen[configuration]), MAX_CYCLES)
        print(f"ASL_pow Seed-Coder minus Qwen, {LANGUAGES[configuration]}: {gap:+.3f}")


def print_failure_counts(runs: Dict[str, Dict[str, Tasks]]) -> None:
    print("\n== Failed loops (passed cycle 1, failed later) that fail at cycle 2 ==")
    for model_id, configs in runs.items():
        for configuration, tasks in configs.items():
            failed = [task for task in tasks.values() if failed_after_cycle_1(task)]
            at_cycle_2 = sum(task["successful_cycles"] == 1 for task in failed)
            print(f"{MODEL_NAMES[model_id]} {LANGUAGES[configuration]}: {at_cycle_2} of {len(failed)}")

    print("\n== Compared with the English run of the same model ==")
    for model_id, configs in runs.items():
        english = configs[ENGLISH]
        for configuration in EXTENSION:
            tasks = configs[configuration]
            lost = sum(english[t]["successful_cycles"] == MAX_CYCLES and failed_after_cycle_1(tasks[t]) for t in tasks)
            gained = sum(tasks[t]["successful_cycles"] == MAX_CYCLES and failed_after_cycle_1(english[t]) for t in tasks)
            print(f"{MODEL_NAMES[model_id]} {LANGUAGES[configuration]}: English 10 cycles and extension failed after "
                  f"cycle 1: {lost}; extension 10 cycles and English failed after cycle 1: {gained}")
    all_runs = [tasks for configs in runs.values() for tasks in configs.values()]
    first = all_runs[0]
    always = sum(all(tasks[t]["successful_cycles"] == MAX_CYCLES for tasks in all_runs) for t in first)
    print(f"Tasks reaching 10 cycles in all six runs: {always} of {len(first)}")


def print_example_tasks(runs: Dict[str, Dict[str, Tasks]]) -> None:
    print("\n== Example tasks (sustained cycles per run) ==")
    for task_id in ("Mbpp/14", "Mbpp/2", "Mbpp/17"):
        values = ", ".join(
            f"{MODEL_NAMES[model_id]} {LANGUAGES[configuration]} {tasks[task_id]['successful_cycles']}"
            for model_id, configs in runs.items() for configuration, tasks in configs.items()
        )
        print(f"{task_id}: {values}")
    qwen, seed = runs.values()
    loop = qwen["summarization_hindi"]["Mbpp/2"]["cycles"]
    written = {cycle["task_2_raw_output"].strip() for cycle in loop[1:]}
    print(f"Qwen Hindi Mbpp/2: different summaries written in cycles 2-10: {len(written)} {sorted(written)}")
    failing = qwen["summarization_spanish"]["Mbpp/118"]["cycles"][-1]["test_results"]
    plus = str(failing.get("plus_details") or "")
    print(f"Qwen Spanish Mbpp/118, failing cycle: base passed {failing['base_success']}, "
          f"plus failing {plus.count('0')} of {len(plus)}")


def print_run_variation(runs: Dict[str, Dict[str, Tasks]]) -> None:
    print("\n== Run-to-run variation: cycle 1 (same prompt) in the Hindi and Spanish runs ==")
    for model_id, configs in runs.items():
        hindi, spanish = configs["summarization_hindi"], configs["summarization_spanish"]
        first = {t: (hindi[t]["cycles"][0], spanish[t]["cycles"][0]) for t in hindi}
        code = sum(without_whitespace(h["task_1_output"]) != without_whitespace(s["task_1_output"])
                   for h, s in first.values())
        any_change = sum(h["task_1_output"] != s["task_1_output"] for h, s in first.values())
        result = [t for t, (h, s) in first.items() if h["success"] != s["success"]]
        print(f"{MODEL_NAMES[model_id]}: different code on {code} tasks ignoring whitespace ({any_change} counting "
              f"whitespace-only changes), different pass/fail on {len(result)} {result}")


def print_summary_checks(runs: Dict[str, Dict[str, Tasks]]) -> None:
    print("\n== Summary checks (rough heuristics, all summaries of a run) ==")
    for model_id, configs in runs.items():
        for configuration, tasks in configs.items():
            texts = summaries(tasks)
            line = f"{MODEL_NAMES[model_id]} {LANGUAGES[configuration]}: {len(texts)} summaries, " \
                   f"with repeated words {sum(map(has_repeated_words, texts))}"
            if configuration in EXTENSION:
                opening = SUMMARY_OPENINGS[LANGUAGES[configuration].lower()]
                line += f", not starting with the opening phrase {sum(not s.strip().startswith(opening) for s in texts)}"
            print(line)


def print_inspected_loops(runs: Dict[str, Dict[str, Tasks]]) -> None:
    print(f"\n== Inspected loops: first {INSPECTED_PER_RUN} per run that passed cycle 1 and then failed ==")
    no_details = Counter()
    also_in_english = 0
    for model_id, configs in runs.items():
        for configuration, tasks in configs.items():
            selected = inspected(tasks)
            print(f"{MODEL_NAMES[model_id]} {LANGUAGES[configuration]}: {', '.join(selected)}")
            group = "English" if configuration == ENGLISH else "extension"
            for task_id in selected:
                result = tasks[task_id]["cycles"][-1]["test_results"]
                no_details[group] += not result.get("base_details") and not result.get("plus_details")
                if group == "extension":
                    also_in_english += failed_after_cycle_1(configs[ENGLISH][task_id])
    n_extension = INSPECTED_PER_RUN * len(EXTENSION) * len(runs)
    print(f"Inspected extension tasks that also failed after cycle 1 in English: {also_in_english} of {n_extension}")
    print(f"Failing code with no per-test results: extension {no_details['extension']} of {n_extension}, "
          f"English {no_details['English']} of {INSPECTED_PER_RUN * len(runs)}")
    qwen = runs["qwen2.5-coder-7b-instruct"]
    same = qwen[ENGLISH]["Mbpp/6"]["cycles"][1]["task_1_output"] == qwen["summarization_hindi"]["Mbpp/6"]["cycles"][1]["task_1_output"]
    print(f"Qwen Mbpp/6: cycle-2 code identical in English and Hindi: {same}")


def print_run_times(log_path: Path) -> None:
    print(f"\n== Run times (start to results saved, from {log_path}) ==")
    start, total = None, 0.0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if "Starting EvaLoop evaluation experiment" in line:
                start = datetime.strptime(line[:19], LOG_TIME)
            elif "Results saved to" in line and start is not None:
                minutes = (datetime.strptime(line[:19], LOG_TIME) - start).total_seconds() / 60
                total += minutes
                print(f"{minutes:5.1f} min  {Path(line.split('Results saved to')[1].strip()).name}")
    print(f"Total: {total:.1f} min")


def print_package_comparison(runs: Dict[str, Dict[str, Tasks]], data_dir: Path) -> None:
    print(f"\n== Data package comparison ({data_dir}) ==")
    for model_id, configs in runs.items():
        with open(data_dir / "loops" / f"{model_id}.json") as f:
            package = json.load(f)["tasks"]
        package_score = asl_pow([task["successful_cycles"] for task in package], MAX_CYCLES)
        own_score = asl_pow(sustained(configs[ENGLISH]), MAX_CYCLES)
        written = [cycle["summary"] for task in package for cycle in task["cycles"] if cycle.get("summary")]
        first_lines = [s.strip().strip('"').strip().split("\n")[0].strip().lstrip("\"'").lower() for s in written]
        starting = sum(line.startswith("write a python function") for line in first_lines)
        print(f"{MODEL_NAMES[model_id]}: package ASL_pow {package_score:.3f}, own English {own_score:.3f}, "
              f"difference {abs(package_score - own_score):.3f}; package summaries starting with "
              f"'write a python function' {starting} of {len(written)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the numbers reported for the multilingual loops.")
    parser.add_argument("--results_dir", type=Path, default=Path("results_hindi_spanish"),
                        help="Folder with the evaloop results files (default: results_hindi_spanish)")
    parser.add_argument("--log", type=Path, default=Path("results_hindi_spanish/evaloop (1).log"),
                        help="evaloop log covering all six runs (default: results_hindi_spanish/evaloop (1).log)")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Assignment data package folder; if given, also compares with its records")
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    print_results_table(runs)
    print_failure_counts(runs)
    print_example_tasks(runs)
    print_run_variation(runs)
    print_summary_checks(runs)
    print_inspected_loops(runs)
    print_run_times(args.log)
    if args.data_dir is not None:
        print_package_comparison(runs, args.data_dir)


if __name__ == "__main__":
    main()
