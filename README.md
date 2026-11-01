# EvaLooop: LLM Robustness Evaluation Framework

[![arXiv](https://img.shields.io/badge/arXiv-2505.12185-b31b1b.svg)](https://arxiv.org/abs/2505.12185)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Leaderboard](https://img.shields.io/badge/leaderboard-evalooop.github.io-blue)](https://evalooop.github.io/)

EvaLoop is a framework for evaluating the robustness of Large Language Models (LLMs) through iterative dual-task cycles. The framework measures how many cycles an LLM can execute before generating code that fails functional testing, providing a quantitative metric for model robustness. It accompanies the paper [*EvaLoop: Assessing LLM Robustness in Programming from a Self-consistency Perspective*](https://arxiv.org/abs/2505.12185); results across models are published on the [leaderboard](https://evalooop.github.io/).

## 🚀 Quick Start

### Installation

EvaLoop is not yet published on PyPI — install from source:

```bash
git clone https://github.com/evalooop/EvaLooop
cd EvaLooop
pip install -e .
```

Optional extras:

```bash
pip install -e ".[vllm]"   # GPU batch inference via vLLM (Linux + CUDA only)
pip install -e ".[dev]"    # development tools (pytest, ruff, ...)
```

After installing, check your environment:

```bash
evaloop validate_setup                    # full check
evaloop validate_setup --check_gpu=False  # skip GPU checks (e.g. on a laptop)
```

### Basic Usage

```bash
# Evaluate an OpenAI model on the generation<->summarization loop
evaloop evaluate --model "gpt-4"

# Evaluate an open-source model with multi-GPU vLLM
evaloop evaluate --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
                 --gpu_ids "0,1,2,3" \
                 --tensor_parallel_size 4 \
                 --max_cycles 15

# Analyze results (the default ASL metric calls an LLM judge; see "Scoring")
evaloop analyze --results_path "results/experiment_results.json" \
                --judge_api_key_file api.key
```

## 🔄 How the Loop Works

Each evaluation cycle alternates between two tasks:

1. **Initial input**: a natural-language prompt describing a coding task (from MBPP+), formatted as a docstring whose final line is an `assert` statement.
2. **Code generation**: the model generates code from the prompt.
3. **Functional testing**: the generated code is executed against the dataset's test cases ([EvalPlus](https://github.com/evalplus/evalplus) MBPP+).
4. **Code summarization**: the model summarizes the generated code back into a natural-language description.
5. **Iteration**: the summary becomes the next cycle's prompt; the loop repeats until the generated code fails testing or `--max_cycles` is reached.

The number of loops each task survives is aggregated into the **ASL (Average Sustainable Loops)** score, which weights long survival quadratically and discounts semantic drift at the failure boundary; see [Scoring](#-scoring).

### Design note: the retained assert line

The summarization step deliberately re-attaches the final `assert` line from the prompt the cycle *started* with, so each new prompt looks like:

```
"""
<model-generated summary>
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
"""
```

This is intentional, not an artifact: keeping one assert anchors the function name and signature across cycles, so the loop measures **semantic drift** in the model's understanding rather than trivial failures from a renamed function. You will see this line in every generated prompt in the result data. The implementation lives in `CodeSummarizationTask.postprocess` (`evaloop/tasks/code_summarization.py`).

### Supported Task Types

| Task type | Config name | Status |
|---|---|---|
| Code generation | `code_generation` | ✅ Supported |
| Code summarization | `code_summarization` | ✅ Supported |
| Code translation | `code_translation` | 🧪 Experimental — the task class exists, but config/evaluator support is not yet in the public release (`run_code_translation()` raises `NotImplementedError`) |

Want to add your own loop? A new task type is one new file plus a registry entry — no changes to the loop runner. See the [Contributing Guide](CONTRIBUTING.md).

## 🌐 Multilingual Summarization Loops

This fork ([github.com/svyas42/EvaLooop-TAI4SE](https://github.com/svyas42/EvaLooop-TAI4SE), forked from [evalooop/EvaLooop](https://github.com/evalooop/EvaLooop)) adds two loop types in which the summaries are written in Hindi or Spanish instead of English. Everything else stays as in the standard loop, so on the same tasks a difference in ASL between the loops comes from the summary language.

| `--loop_type` | Summary language |
|---|---|
| `code_generation_summarization` (default) | English (standard loop) |
| `summarization_hindi` | Hindi |
| `summarization_spanish` | Spanish |

- **Summarization prompt**: the standard prompt with two changes. It asks for the sentence in the target language, and the fixed opening phrase "write a python function to" is replaced by "एक पायथन फ़ंक्शन लिखें जो" (Hindi) or "escribe una función en Python para" (Spanish). The instruction itself stays in English. The templates are in `evaloop/tasks/multilingual_prompts.py`.
- **Generation prompt**: unchanged, so cycle 1 gets the same input as in the standard loop.
- **Carried between cycles**: the non-English summary wrapped as a docstring together with the original prompt's assert line (the existing `CodeSummarizationTask.postprocess`, unchanged). The assert line keeps the function name and signature stable across cycles.
- **Failure condition**: unchanged. A cycle fails when the generated code fails the MBPP+ base or plus tests; at most `--max_cycles` cycles run.

```bash
evaloop evaluate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --loop_type "summarization_hindi" --seed 42
```

Results are saved as `{experiment_name}_{loop_type}_results.json`, and each results entry records its `loop_type`.

### Changes to the existing framework

| File | Change | Reason |
|---|---|---|
| `evaloop/tasks/multilingual_prompts.py` (new) | Hindi and Spanish summarization templates | Keeps the new prompts in a new file |
| `evaloop/core/config.py` | New `loop_type` setting (validated, included in `to_dict()`); new branch in `get_task_configs()` that reuses the standard generation config and replaces only the summarization template | `get_task_configs()` is where a loop's task sequence is defined (see [CONTRIBUTING.md](CONTRIBUTING.md)) |
| `evaloop/core/evaluator.py` | Builds tasks and names the results file from `config.loop_type` instead of the hard-coded `"code_generation_summarization"`; records `loop_type` in the results | The loop type was hard-coded |
| `evaloop/core/evaluator.py` | Joins `plus_details` only when the test result contains it | A test error result has no `plus_details`, so the old code raised `KeyError` and stopped the run; results are written only at the end, so the whole run was lost |
| `evaloop/cli.py` | New `--loop_type` option | Selects the loop from the command line |
| `tests/test_multilingual.py` (new) | Offline tests for the templates, the config wiring and the `KeyError` fix | The contributing guide asks for unit tests |
| `scripts/make_summary_csv.py` (new) | Builds `summary.csv` from the six results files | Makes every number in `summary.csv` reproducible with one command |
| `scripts/report_numbers.py` (new) | Prints the other numbers in the report from the six results files and the run log | Makes the report's numbers reproducible with one command |

The loop runner (`evaloop/evaluation/cycle.py`), the task classes, the tester and the analysis code are unchanged. With the default `--loop_type`, the prompts and the results filename are the same as before.

### Setup used for the reported runs

- **Machine**: an NC State VCL virtual machine (Ubuntu 24, no root access) with one NVIDIA L40S GPU; `evaloop validate_setup` reported CUDA 12.8.
- **Software**: Python 3.12.3 and vLLM 0.19.1; the models ran in bfloat16 through the repository's vLLM path.
- **Models**:

  | Hugging Face model | Data package id | Revision (from the run log) |
  |---|---|---|
  | `Qwen/Qwen2.5-Coder-7B-Instruct` | `qwen2.5-coder-7b-instruct` | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
  | `ByteDance-Seed/Seed-Coder-8B-Instruct` | `seed-coder-8b-instruct` | `1c5af72afa21c22c9272a0b60a3a404195e88fcd` |

- **Settings**: the CLI defaults (temperature 0.0, top-p 1.0, at most 1024 new tokens, context length 4096, at most 10 cycles) plus `--seed 42`, on all 378 MBPP+ tasks. No API key or LLM judge was used.

Installation on the VM, from the home folder. Ubuntu's Python has no `ensurepip`, so the virtual environment is created without pip and pip is installed with `get-pip.py`:

```bash
git clone https://github.com/svyas42/EvaLooop-TAI4SE
cd EvaLooop-TAI4SE
python3 -m venv --without-pip .venv
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install -e .
pip install -e ".[vllm]"
```

When it starts, vLLM compiles GPU code with `gcc`, which needs `Python.h`. The VM did not have `python3.12-dev`, so the header files were unpacked into the home folder without root:

```bash
mkdir -p ~/pydev && cd ~/pydev
apt download libpython3.12-dev python3.12-dev
dpkg -x libpython3.12-dev_*.deb ~/pydev
dpkg -x python3.12-dev_*.deb ~/pydev
cd ~/EvaLooop-TAI4SE
```

### Running the six runs

In every new terminal, first run:

```bash
source .venv/bin/activate
export CPATH=$HOME/pydev/usr/include/python3.12:$HOME/pydev/usr/include
```

Then, from the repository root, one run at a time (in the order they were run):

```bash
evaloop evaluate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --loop_type "summarization_hindi" --seed 42
evaloop evaluate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --loop_type "summarization_spanish" --seed 42
evaloop evaluate --model "ByteDance-Seed/Seed-Coder-8B-Instruct" --loop_type "summarization_hindi" --seed 42
evaloop evaluate --model "ByteDance-Seed/Seed-Coder-8B-Instruct" --loop_type "summarization_spanish" --seed 42
evaloop evaluate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --seed 42
evaloop evaluate --model "ByteDance-Seed/Seed-Coder-8B-Instruct" --seed 42
```

The last two are the English baseline (default `--loop_type`). Each run writes `results/evaloop_<model>_mbpp_plus_<date>_<time>_<loop_type>_results.json` and adds to `logs/evaloop.log`. The generated code runs only inside EvalPlus on the VM. The six results files and the log were then copied into `results_hindi_spanish/`. There, `evaloop (1).log` covers all six runs, and `evaloop.log` is an earlier copy that covers only the four multilingual runs.

A rerun will not give exactly the same files. The file names get new timestamps, so `RUNS` in `scripts/make_summary_csv.py` has to list the new names. The numbers can also differ a little: at temperature 0, the same cycle-1 prompt gave different code in the Hindi and Spanish runs of the same model (printed by `scripts/report_numbers.py`).

### Reproducing the reported numbers

From the repository root, after `pip install -e .` (no GPU or model needed):

```bash
python scripts/make_summary_csv.py
python scripts/report_numbers.py
python scripts/report_numbers.py --data_dir /path/to/assignment4-data
```

- `make_summary_csv.py` writes `summary.csv`. `success_rate` is pass@1 and `asl_score` is `ASL_pow`, both computed with `evaloop/analysis/metrics.py` (max 10 cycles). Every row comes from a results file in `results_hindi_spanish/`: the `standard-mbpp-plus` rows from the standard-loop runs (default `--loop_type`), and the `summarization_hindi` and `summarization_spanish` rows from the multilingual runs. All six runs cover the same 378 MBPP+ tasks.
- `report_numbers.py` prints the report's other numbers from the same files: the results table (including the change against English, `pass_drop` and the tasks reaching 10 cycles), the failed loops that fail at cycle 2, the tasks lost or gained against English, the example tasks (Mbpp/14, Mbpp/2, Mbpp/17, Mbpp/118), the cycle-1 differences between runs, the summary checks (repeated words, missing opening phrase), the inspected loops chosen by the selection rule with the counts taken from them, and the run times from `results_hindi_spanish/evaloop (1).log`.
- With `--data_dir`, `report_numbers.py` also compares the runs with the assignment data package (its ASL_pow for the two models, and how many of its summaries start with "write a python function"). The data package is not part of this repository.
- The failure-mode labels of the inspected loops come from reading the loops and are not computed.

## 🛠️ CLI Reference

### `evaluate` — Code Generation ↔ Summarization

```bash
evaloop evaluate --model MODEL [OPTIONS]
```

**Key options:**
- `--model`: A single model name or HuggingFace path (required). Examples: `"gpt-4"`, `"Qwen/Qwen2.5-Coder-32B-Instruct"`
- `--dataset`: Dataset to use. Currently only `mbpp_plus` is supported (HumanEval is planned).
- `--max_cycles`: Maximum evaluation cycles (default: 10)
- `--loop_type`: `code_generation_summarization` (default), `summarization_hindi` or `summarization_spanish`; see [Multilingual Summarization Loops](#-multilingual-summarization-loops)
- `--temperature`: Generation temperature (default: 0.0)
- `--output_dir`: Results output directory (default: `results/`)
- `--gpu_ids`: GPU IDs for vLLM models (e.g., `"0,1,2,3"`)
- `--tensor_parallel_size`: GPUs for tensor parallelism (default: 1)

**Examples:**

```bash
# Basic OpenAI evaluation
evaloop evaluate --model "gpt-4"

# Large model with multi-GPU setup
evaloop evaluate --model "meta-llama/Llama-3-70b-instruct-hf" \
                 --gpu_ids "0,1,2,3" \
                 --tensor_parallel_size 4 \
                 --gpu_memory_utilization 0.85

# Custom experiment parameters
evaloop evaluate --model "deepseek-coder-33b" \
                 --max_cycles 20 \
                 --temperature 0.2 \
                 --experiment_name "deepseek_robust_eval"
```

To evaluate several models, run `evaloop evaluate` once per model (see [Batch Processing](#batch-processing)).

### `analyze` — Result Analysis

```bash
evaloop analyze --results_path "results/experiment_results.json" [OPTIONS]
```

**Key options:**
- `--results_path`: Path to results JSON file (required)
- `--metrics`: Metrics to compute (default: `"ASL,ASL_pow,ASL_bias,pass@1,pass_drop"`, see [Scoring](#-scoring))
- `--generate_plots`: Generate visualization plots (default: True)
- `--output_dir`: Analysis output directory (default: `<results dir>/analysis`)
- `--max_cycles`: Loop budget M of the run (default: value recorded in the results file; for older
  files, the value implied by tasks flagged `max_cycles_reached`; else 10 with a warning)

**Similarity judge options** (only used by the `ASL` metric, and only for pairs not cached yet):
- `--judge_model`: Judge model id (default: `gpt-4-turbo-2024-04-09`, the paper's judge)
- `--judge_api_key` / `--judge_api_key_file`: Judge API key (default: `OPENAI_API_KEY`)
- `--judge_base_url`: Any OpenAI-compatible endpoint, e.g. a local `vllm serve` server (default: OpenAI)
- `--judge_max_tokens`: Answer token budget (default: 10; raise it for reasoning models)
- `--judge_temperature`: Judge temperature (default: 0.0)
- `--judge_system_prompt`: Optional system message (default: none)
- `--judge_prompt_file`: Custom prompt template with `{prompt1}`, `{code1}`, `{prompt2}`, `{code2}`
- `--judge_delay`: Seconds between judge calls (default: 0)
- `--similarity_path`: Similarity sidecar to read and extend (default: next to the results file)
- `--rejudge_failed`: Retry pairs whose previous judge call failed (default: True)

**Examples:**

```bash
# Default analysis: all metrics, paper judge (needs an OpenAI key the first time)
evaloop analyze --results_path "results/my_experiment_results.json" --judge_api_key_file api.key

# Use a self-hosted judge through vLLM's OpenAI-compatible server
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000
evaloop analyze --results_path "results/exp.json" \
                 --judge_model "Qwen/Qwen2.5-72B-Instruct" \
                 --judge_base_url "http://localhost:8000/v1"

# Skip the judge: metrics that only need the loop counts
evaloop analyze --results_path "results/exp.json" \
                 --metrics "ASL_pow,ASL_bias,pass@1,pass_drop" \
                 --output_dir "analysis/custom/"
```

**Outputs** (in `--output_dir`): `analysis_results.json` (all metric values per model, the loop
histogram `n_i`, and judge statistics such as the bucket weights `s_i`), `analysis_summary.txt`,
and the plots. Judge scores go to the similarity sidecar next to the results file.

### `list_models` — Available Models

```bash
evaloop list_models
```

### `validate_setup` — System Validation

```bash
evaloop validate_setup [--check_gpu=False] [--check_api_keys=False]
```

## 🔧 Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

### Model Configuration

EvaLoop automatically configures models based on their names/paths:

- **OpenAI models**: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo`
- **HuggingFace paths**: `meta-llama/Llama-3-70b-instruct-hf`
- **Pre-registered names**: use `evaloop list_models` to see available options

Large models automatically use vLLM for efficient inference when it is installed (`pip install -e ".[vllm]"`).

## 📊 Results and Analysis

### Output Structure

Results are saved in JSON format:

```json
{
  "model": "gpt-4",
  "max_cycles": 10,
  "prompt_results": [
    {
      "task_id": "Mbpp/2",
      "initial_prompt": "Write a function to find the shared elements ...",
      "cycles": [...],
      "successful_cycles": 5,
      "max_cycles_reached": false
    }
  ],
  "average_successful_cycles": 4.2
}
```

### Metrics

`evaloop analyze` reports `ASL` (the headline score, used by default), `ASL_pow`, `ASL_bias`,
`pass@1` and `pass_drop`. How each one is computed is described in [Scoring](#-scoring).

### Visualization

`evaloop analyze` writes three plots to the output directory: `asl_comparison.png` (ASL per
model), `cycle_distribution.png` (histogram of sustained loops) and `success_heatmap.png`
(sustained loops per task and model, first 50 tasks).

## 📐 Scoring

The scores below are the ones used in the paper and on the [leaderboard](https://evalooop.github.io/);
`evaloop analyze` reproduces the published values exactly when given the same results and judge scores.

### Notation

- **T**: number of tasks in the run (378 for MBPP Plus).
- **M**: maximum number of loops (`max_cycles`, 10 by default).
- **i**: number of loops a task *sustains* (`successful_cycles` in the results file): its code passed
  the tests in loops 1..i and failed in loop i+1. `i = 0` means it already failed in the first loop,
  `i = M` means it never failed.
- **n_i**: number of tasks that sustain exactly i loops.
- **pass@k**: number of tasks that sustain at least k loops (so pass@1 is the usual one-shot pass count).

### ASL: the headline score (default)

```
ASL = Σ_{i=1..M} n_i · i² · s_i / (T · M)          range [0, M]
```

- **i²** rewards sustained correctness quadratically: surviving 8 loops counts 4× as much as surviving 4.
- **s_i** is the semantic similarity weight of bucket i. It discounts tasks whose last successful
  loop had already drifted away from the original intent.

**How s_i is obtained.** For every task with a failure boundary (`1 ≤ i < M`), an LLM judge
compares the last successful transition with the failing one:

| | Prompt | Code |
|---|---|---|
| Side 1 | prompt that fed loop i (the original task if i = 1, else the summary written in loop i−1) | code generated in loop i (passed) |
| Side 2 | summary written in loop i, i.e. the prompt of loop i+1 | code generated in loop i+1 (failed) |

The judge returns a score `sim ∈ [0, 1]`. A task that sustained i loops went through i prompt
transitions; the first i−1 produced passing code, so they count as similarity 1.0, and only the
boundary transition uses the judge score. With `m_i` the mean judge score of the tasks in bucket i:

```
s_i = (m_i + (i − 1)) / i        for 1 ≤ i < M
s_i = 1                          for i = M (never failed) and for buckets without judge scores
```

Because the mean is linear, this equals giving every boundary task the weight `i · (i − 1 + sim)`
instead of `i²`, so `ASL ≤ ASL_pow` and the gap is `Σ_tasks i · (1 − sim) / (T · M)`.

*Example:* with T = 378 and M = 10, a task that sustains 4 loops and gets a judge score of 0.85
contributes `4 · (3 + 0.85) / 3780 = 0.00407` to ASL, versus `16 / 3780 = 0.00423` without the
similarity weighting.

### All metrics

| Metric | Formula | Range | What it measures | Leaderboard field |
|---|---|---|---|---|
| `ASL` | Σ n_i · i² · s_i / (T · M) | [0, M] | Headline robustness score (default) | `semanticSimilarityScore`, shown as "ASL" |
| `ASL_pow` | Σ n_i · i² / (T · M) | [0, M] | ASL without semantic weighting (all s_i = 1) | `aslScore` |
| `ASL_bias` | Σ n_i · i² / (pass@1 · M) | [0, M] | Robustness on the tasks the model solves at all, independent of one-shot accuracy (0 if pass@1 = 0) | `robustnessScore` |
| `pass@1` | pass@1 / T | [0, 1] | One-shot accuracy | `successRate` |
| `pass_drop` | (pass@1 − pass@M) / T | [0, 1] | Absolute accuracy lost within M loops | "drop" (as a percentage) |

Only `ASL` needs the judge; the other metrics use the loop counts alone. The leaderboard also
averages repeated runs of a model and only lists models with `pass@1 > 0.2` and `ASL_bias > 3`;
those steps happen outside this package.

### Similarity judge

- **Default:** `gpt-4-turbo-2024-04-09` at temperature 0 with a 10-token answer budget, no system
  message and the paper's prompt (`evaloop.analysis.similarity.DEFAULT_JUDGE_PROMPT`). Scores
  from a different judge or prompt are not comparable with the leaderboard, so compare models
  only under the same judge.
- **Changing the judge:** the judge is served through the same API layer as the evaluated models
  (`OpenAILLM`), so any OpenAI-compatible endpoint works via `--judge_model`, `--judge_base_url`
  and `--judge_api_key(_file)`. In Python, pass `judge_config=JudgeConfig(...)` to
  `ResultAnalyzer`, or wrap any `BaseLLM` in `SemanticSimilarityJudge(llm)` and pass it as
  `judge=...`.
- **Caching:** scores are stored in a sidecar next to the results file,
  `<results stem>_<judge>_similarity_scores.json`, as
  `{"<task_id>_cycle_<i>": {"score": ..., "input_sha256": ...}}`. The judge is only called for pairs
  missing from the sidecar, so reruns are free and an interrupted run resumes where it stopped. No
  API key is needed when the sidecar is complete.
- **Cache validation:** `input_sha256` fingerprints the exact judge prompt (the four judged strings
  rendered into the template). If a results file is overwritten (e.g. `evaluate` rerun with the same
  `--experiment_name`) or the prompt template changes, mismatching entries are re-judged with a
  warning instead of being mixed into the new run. Sidecars from the published data archive
  (`{"<task_id>_cycle_<i>": score}`, no hash) are still accepted; their entries cannot be verified
  and are used as is.
- **Failures:** if the judge call fails or its answer cannot be parsed, the pair is stored as
  `null`, left out of its bucket mean (with a warning), and retried on the next run
  (`--rejudge_failed=False` keeps it as is). The original scripts recorded such failures as 0.0.
- **Parsing:** a bare number is used directly; otherwise the first number that looks like a score
  is taken. Results are clamped to [0, 1].

### Programmatic scoring

```python
from evaloop.analysis.analyzer import ResultAnalyzer
from evaloop.analysis.similarity import JudgeConfig

analyzer = ResultAnalyzer(
    "results/exp_results.json",
    "results/analysis",
    judge_config=JudgeConfig(model="gpt-4-turbo-2024-04-09", api_key="sk-..."),
)
results = analyzer.analyze()          # all metrics; results["ASL"]["by_model"][model_name]
```

The pure metric functions (`asl_semantic`, `asl_pow`, `asl_bias`, `pass_at_k`, `pass_drop`,
`bucket_similarity`) live in `evaloop.analysis.metrics`.

## 🧪 Advanced Usage

### Custom Model Registration

```python
from evaloop.models.registry import ModelRegistry

registry = ModelRegistry()
registry.register_model("my-custom-model", {
    "name": "my-custom-model",
    "type": "vllm",
    "path": "path/to/my/model",
    "max_model_len": 4096,
    "tensor_parallel_size": 2
})
```

### Programmatic API

```python
from evaloop.core.config import EvaluationConfig
from evaloop.core.evaluator import EvaLoopEvaluator

config = EvaluationConfig(
    model="gpt-4",
    dataset="mbpp_plus",
    max_cycles=10,
    temperature=0.0,
)

evaluator = EvaLoopEvaluator(config)
results = evaluator.run_code_generation_summarization()
```

### Batch Processing

```bash
# Process multiple experiments
for model in "gpt-4" "gpt-3.5-turbo" "deepseek-coder-33b"; do
    evaloop evaluate --model "$model" \
                     --experiment_name "batch_${model}" \
                     --output_dir "results/batch/"
done

# Analyze all results
for result in results/batch/*_results.json; do
    evaloop analyze --results_path "$result"
done
```

## 🤝 Contributing

We welcome contributions — new loop/task types especially! Please see the [Contributing Guide](CONTRIBUTING.md) for the development setup, the "add a new loop type" walkthrough, and PR expectations.

### Development Setup

```bash
git clone https://github.com/evalooop/EvaLooop
cd EvaLooop
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
ruff check evaloop tests
```

The unit test suite runs without GPU, API keys, or the heavy model dependencies.

## 📖 Citation

If you use EvaLoop in your research, please cite:

```bibtex
@article{fang2024evaloop,
  title={EvaLoop: Assessing LLM Robustness in Programming from a Self-consistency Perspective},
  author={Fang, Sen and Ding, Weiyuan and Xu, Bowen},
  journal={arXiv preprint arXiv:2505.12185},
  year={2024},
  url={https://arxiv.org/abs/2505.12185}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Python Fire](https://github.com/google/python-fire) for CLI interface
- Uses [vLLM](https://github.com/vllm-project/vllm) for efficient model inference
- Evaluation dataset from [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- Code testing powered by [EvalPlus](https://github.com/evalplus/evalplus)

## 📞 Support

- 🏆 [Leaderboard](https://evalooop.github.io/)
- 📄 [Paper](https://arxiv.org/abs/2505.12185)
- 🐛 [Issue Tracker](https://github.com/evalooop/EvaLooop/issues)

---

**EvaLoop** - Robust LLM Evaluation Made Simple 🔄
