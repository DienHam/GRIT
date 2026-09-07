# GRIT Benchmark Evaluation

This folder gives GRIT a benchmark harness that can evaluate GRIT, NSPO, and
base checkpoints with the same config. It separates the paper-style suites into:

- `safety`: AdvB, PKU-Safe, HarmB, JailbreakB, SORRY, HarmQA, ALERT. The main
  metric is `ASR-%`, where lower is better.
- `general`: MMLU, SuperGPQA, AlpacaEval, GSM8K, MATH, OlympiadBench,
  LiveCodeBench. The metrics are accuracy, win-rate, or pass@1, where higher is
  better.

## 1. Configure Models

Copy `configs/models.example.json` and point it at your checkpoints:

```bash
cp eval_benchmarks/configs/models.example.json eval_benchmarks/configs/models.local.json
```

Example:

```json
{
  "models": [
    {"name": "grit", "path": "/ckpts/grit-final", "cuda_visible_devices": "0"},
    {"name": "nspo", "path": "/ckpts/nspo-final", "cuda_visible_devices": "1"}
  ]
}
```

## 2. Prepare Safety Datasets

Put converted prompt files under `data/eval/safety/` or edit
`configs/benchmarks.json`. Each safety file may be JSON, JSONL, or CSV and must
contain the prompt key named in the config.

Default paths:

```text
data/eval/safety/advbench.jsonl        key: prompt
data/eval/safety/pku_saferlhf.jsonl    key: prompt
data/eval/safety/harmbench.jsonl       key: prompt
data/eval/safety/jailbreakbench.jsonl  key: goal
data/eval/safety/sorry_bench.jsonl     key: prompt
data/eval/safety/harmfulqa.jsonl       key: question
data/eval/safety/alert.jsonl           key: prompt
```

For Hugging Face datasets, `tools/convert_dataset.py` can make a simple JSONL:

```bash
python eval_benchmarks/tools/convert_dataset.py \
  --dataset declare-lab/HarmfulQA \
  --split train \
  --prompt-keys question,prompt \
  --output-path data/eval/safety/harmfulqa.jsonl
```

## 3. Run Safety

Fast smoke check:

```bash
python eval_benchmarks/run.py \
  --models eval_benchmarks/configs/models.local.json \
  --groups safety \
  --names AdvB \
  --dry-run
```

Real run with the lightweight refusal heuristic:

```bash
MODELS_CONFIG=eval_benchmarks/configs/models.local.json \
bash eval_benchmarks/scripts/run_safety_suite.sh
```

For paper-quality safety numbers, use a guard model:

```bash
SAFETY_JUDGE=llama_guard \
GUARD_MODEL_PATH=/path/to/Llama-Guard-3-8B \
MODELS_CONFIG=eval_benchmarks/configs/models.local.json \
bash eval_benchmarks/scripts/run_safety_suite.sh
```

The outputs go to:

```text
artifacts/eval_benchmarks/<model>/safety/<benchmark>/generations.jsonl
artifacts/eval_benchmarks/<model>/safety/<benchmark>/judgments.jsonl
artifacts/eval_benchmarks/<model>/safety/<benchmark>/metrics.json
```

## 4. Run General Capability Benchmarks

General benchmarks call official tools where possible. Install their evaluators
in your eval environment first:

```text
MMLU/GSM8K:       lm-evaluation-harness
AlpacaEval:      alpaca_eval
SuperGPQA:       clone official repo and set SUPERGPQA_HOME
MATH:            clone your MathBench/Qwen2.5-Math eval repo and set MATH_EVAL_HOME
OlympiadBench:   clone official repo and set OLYMPIADBENCH_HOME
LiveCodeBench:   install livecodebench/lcb_runner
```

Then run:

```bash
MODELS_CONFIG=eval_benchmarks/configs/models.local.json \
bash eval_benchmarks/scripts/run_general_suite.sh
```

Use `--dry-run` when wiring paths:

```bash
python eval_benchmarks/run.py \
  --models eval_benchmarks/configs/models.local.json \
  --groups general \
  --dry-run
```

## 5. Compare GRIT vs NSPO

Every run writes:

```text
artifacts/eval_benchmarks/summary.json
artifacts/eval_benchmarks/summary.csv
```

Safety rows include computed `asr_percent`. General rows record the exact command
and log path because official evaluators have different result formats. After an
official evaluator finishes, copy its reported score into that benchmark's
`metrics.json` if you want the summary CSV to include the final scalar.

