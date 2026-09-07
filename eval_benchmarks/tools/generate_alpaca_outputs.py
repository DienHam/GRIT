from __future__ import annotations

from pathlib import Path
import argparse
import json

from eval_benchmarks.grit_eval.generation import generate_with_vllm


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AlpacaEval model outputs with vLLM.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("AlpacaEval generation requires the 'datasets' package") from exc

    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"].to_list()
    prompts = [str(record["instruction"]) for record in eval_set]
    responses = generate_with_vllm(
        model_path=args.model_path,
        prompts=prompts,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    outputs = [
        {
            "dataset": record.get("dataset", "alpaca_eval"),
            "instruction": record["instruction"],
            "output": response,
            "generator": args.model_name,
        }
        for record, response in zip(eval_set, responses)
    ]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

