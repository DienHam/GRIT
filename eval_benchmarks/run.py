from __future__ import annotations

from pathlib import Path
import argparse

from grit_eval.registry import load_benchmarks, load_model_config, select_benchmarks
from grit_eval.runner import run_general_benchmark, run_safety_benchmark
from grit_eval.safety_judge import load_guard_model
from grit_eval.summarize import write_summary


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRIT/NSPO benchmark suites.")
    parser.add_argument("--benchmarks", default="eval_benchmarks/configs/benchmarks.json")
    parser.add_argument("--models", default="eval_benchmarks/configs/models.example.json")
    parser.add_argument("--output-root", default="artifacts/eval_benchmarks")
    parser.add_argument("--groups", default=None, help="Comma-separated groups: safety,general")
    parser.add_argument("--names", default=None, help="Comma-separated benchmark names")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--safety-judge", choices=["heuristic", "llama_guard"], default="heuristic")
    parser.add_argument("--guard-model-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    benchmarks = select_benchmarks(
        load_benchmarks(args.benchmarks),
        groups=parse_csv_set(args.groups),
        names=parse_csv_set(args.names),
    )
    model_config = load_model_config(args.models)
    output_root = Path(args.output_root)

    guard_model = None
    if args.safety_judge == "llama_guard" and not args.dry_run:
        if not args.guard_model_path:
            raise ValueError("--guard-model-path is required for --safety-judge llama_guard")
        guard_model = load_guard_model(args.guard_model_path, args.tensor_parallel_size)

    all_metrics = []
    for model in model_config["models"]:
        for benchmark in benchmarks:
            print(f"==> {model['name']} / {benchmark.name} ({benchmark.group})")
            if benchmark.group == "safety":
                all_metrics.append(
                    run_safety_benchmark(
                        benchmark=benchmark,
                        model=model,
                        output_root=output_root,
                        batch_size=args.batch_size,
                        tensor_parallel_size=args.tensor_parallel_size,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        judge=args.safety_judge,
                        guard_model=guard_model,
                        dry_run=args.dry_run,
                    )
                )
            else:
                all_metrics.append(
                    run_general_benchmark(
                        benchmark=benchmark,
                        model=model,
                        output_root=output_root,
                        dry_run=args.dry_run,
                    )
                )

    summary_json, summary_csv = write_summary(output_root)
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    print(f"Collected {len(all_metrics)} benchmark rows")


if __name__ == "__main__":
    main()

