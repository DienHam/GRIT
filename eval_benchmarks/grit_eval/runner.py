from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import os
import shlex
import subprocess

from .generation import generate_with_vllm
from .io_utils import read_records, write_json, write_jsonl
from .registry import Benchmark
from .safety_judge import load_guard_model, score_safety_records


def render_command(template: list[str], context: dict[str, Any]) -> list[str]:
    return [part.format(**context) for part in template]


def command_to_string(command: list[str], env: dict[str, str] | None = None) -> str:
    prefix = ""
    if env:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "
    return prefix + " ".join(shlex.quote(part) for part in command)


def run_command(
    command: list[str],
    log_path: Path,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = command_to_string(command, env)
    if dry_run:
        log_path.write_text(command_text + "\n", encoding="utf-8")
        print(command_text)
        return 0

    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + command_text + "\n\n")
        log_file.flush()
        proc = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=process_env,
            check=False,
        )
    return proc.returncode


def run_safety_benchmark(
    benchmark: Benchmark,
    model: dict[str, Any],
    output_root: Path,
    batch_size: int,
    tensor_parallel_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    judge: str,
    guard_model: Any | None,
    dry_run: bool,
) -> dict[str, Any]:
    if not benchmark.dataset_path or not benchmark.prompt_key:
        raise ValueError(f"{benchmark.name} must define dataset_path and prompt_key")

    model_name = model["name"]
    benchmark_dir = output_root / model_name / benchmark.group / benchmark.name
    generation_path = benchmark_dir / "generations.jsonl"
    judgment_path = benchmark_dir / "judgments.jsonl"
    metrics_path = benchmark_dir / "metrics.json"

    if dry_run:
        payload = {
            "model": model_name,
            "benchmark": asdict(benchmark),
            "would_read": benchmark.dataset_path,
            "would_write": str(benchmark_dir),
            "judge": judge,
        }
        write_json(metrics_path, payload)
        print(f"[dry-run] {model_name}/{benchmark.name}: {benchmark.dataset_path}")
        return payload

    records = read_records(benchmark.dataset_path)
    prompts = [str(record[benchmark.prompt_key]) for record in records]
    responses = generate_with_vllm(
        model_path=model["path"],
        prompts=prompts,
        batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    generations = [
        {
            "id": index,
            "question": prompt,
            "response": response,
            "source": benchmark.name,
            "model": model_name,
        }
        for index, (prompt, response) in enumerate(zip(prompts, responses))
    ]
    write_jsonl(generation_path, generations)

    judged, metrics = score_safety_records(generations, judge=judge, guard_model=guard_model)
    metrics_payload = {
        "model": model_name,
        "model_path": model["path"],
        "benchmark": benchmark.name,
        "group": benchmark.group,
        "metric": benchmark.metric,
        "higher_is_better": benchmark.higher_is_better,
        **metrics,
    }
    write_jsonl(judgment_path, judged)
    write_json(metrics_path, metrics_payload)
    return metrics_payload


def run_general_benchmark(
    benchmark: Benchmark,
    model: dict[str, Any],
    output_root: Path,
    dry_run: bool,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not benchmark.command:
        raise ValueError(f"{benchmark.name} must define a command")

    model_name = model["name"]
    benchmark_dir = output_root / model_name / benchmark.group / benchmark.name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "model_name": model_name,
        "model_path": model["path"],
        "output_dir": str(benchmark_dir),
    }
    if extra_context:
        context.update(extra_context)
    command = render_command(benchmark.command, context)

    env = {}
    cuda_visible_devices = model.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    log_path = benchmark_dir / "run.log"
    returncode = run_command(command, log_path=log_path, dry_run=dry_run, env=env)
    metrics_payload = {
        "model": model_name,
        "model_path": model["path"],
        "benchmark": benchmark.name,
        "group": benchmark.group,
        "metric": benchmark.metric,
        "higher_is_better": benchmark.higher_is_better,
        "runner": benchmark.runner,
        "returncode": returncode,
        "log_path": str(log_path),
        "command": command_to_string(command, env),
        "status": "dry_run" if dry_run else ("ok" if returncode == 0 else "failed"),
    }
    write_json(benchmark_dir / "metrics.json", metrics_payload)
    return metrics_payload

