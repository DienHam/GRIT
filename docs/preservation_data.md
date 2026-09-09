# NSPO-domain preservation data

The GRIT preservation sources are AlpacaFarm, GSM8K and LeetCodeDataset, matching
the references in [NSPO section 5.1](https://arxiv.org/html/2512.11391v1#S5.SS1).
The exact splits and balanced allocation below are our reproducible choices,
not verified reconstruction of NSPO's sample list.

| Source | Split/file | Prompts |
|---|---|---:|
| [AlpacaFarm](https://huggingface.co/datasets/tatsu-lab/alpaca_farm) | `alpaca_instructions/unlabeled.json` | 334 |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | `main/train` | 333 |
| [LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset) | `train` | 333 |

Pinned source revisions and seed 66 are in `config/preservation/nspo_mix.json`.
AlpacaFarm combines `instruction` and optional `input`. GSM8K uses `question`.
LeetCodeDataset uses `query`, which includes the problem and coding instructions;
its `prompt` column is mostly shared code imports. Existing answers/solutions
are not used as preservation targets. Sampling removes normalized duplicate
prompts globally. Official evaluation splits are not sampled; this does not
establish absence of cross-benchmark overlap (especially among coding benchmarks).

## 1. Prepare the prompt pool

Use the repository Python environment. For this stage alone the dependencies are
`pyarrow` and `huggingface_hub`:

```bash
python -m pip install pyarrow huggingface_hub
python scripts/prepare_preservation_data.py sample
```

Outputs under `data/preservation/nspo_mix/`:

- `preserve_prompts.parquet`: 1,000 prompt-only records, including a `text` column.
- `prompts.jsonl`: the same records in a readable format.
- `manifest.json`: source revisions/checksums, counts, seed, and artifact checksum.

The manifest marks this stage `prompts_only`. This is sufficient for prompt-only
projector experiments, but the plan's response-context KL requires step 2.
The command refuses to overwrite an existing output directory. For a repeatability
check, use `--output-dir data/preservation/nspo_mix_repeat`.

## 2. Generate fixed-base response contexts

Run in the GRIT model environment (`requirements.txt`, or the existing Kaggle
environment). Generation uses the frozen base model's chat template and greedy
decoding, one prompt at a time. It records the resolved model revision, tokenizer
fingerprint, exact token IDs, and response boundary. Greedy decoding is repeatable
within a fixed environment; it is not a promise of bitwise equality across hardware.

First verify a short run:

```bash
python scripts/prepare_preservation_data.py generate \
  --prompts data/preservation/nspo_mix/preserve_prompts.parquet \
  --output-dir data/preservation/qwen2_5_0_5b_smoke \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --limit 3 --max-new-tokens 16
```

Then generate all 1,000:

```bash
python scripts/prepare_preservation_data.py generate \
  --prompts data/preservation/nspo_mix/preserve_prompts.parquet \
  --output-dir data/preservation/qwen2_5_0_5b \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --max-prompt-length 2048 --max-new-tokens 256
```

Output `preserve_contexts.parquet` includes `text` (decoded full chat), `input_ids`,
`response_start`, `response_mask`, `base_response`, `base_model`, `base_revision`,
`tokenizer_sha256`, and source metadata. Masks here align to the full token sequence;
the standalone trainer shifts them for next-token loss. This is not the fixed-width
`DataProto` schema consumed by the separate `verl` preservation loader.

`contexts.partial.jsonl` is retained if generation fails. Only a successful run
gets the final parquet and completion manifest. Use a fresh output directory to
retry; partial-run resume is not implemented. Long prompts fail explicitly instead
of silently removing the problem statement. Increase the prompt limit if needed.

## 3. Use the new artifacts

Rebuild projectors when changing preservation data. Point the existing builder at
`preserve_contexts.parquet`, `--text-column text`, and `--max-length 2304` to cover
the default 2048+256 token budget. Use the same base checkpoint/revision as the
generation manifest (a local Hugging Face snapshot path can pin the builder).
The builder retokenizes `text`; the standalone KL loader uses stored token IDs.

For standalone training, pass:

```text
--preserve-file data/preservation/qwen2_5_0_5b/preserve_contexts.parquet
--projectors-path <new projector artifact>
--model-revision <base_revision from the generation manifest>
--max-preserve-length 2304
```

When using `run_kaggle_grit_train.sh`, set `PRESERVE_FILE` and `PROJECTORS_PATH`
environment variables to those files, and append `--model-revision` and
`--max-preserve-length` as CLI overrides. Existing text-only data remains supported.
The new loader checks tokenizer identity and the base revision and computes KL
only over generated response tokens. A 2304-token budget can require substantially
more preservation memory than the old 256-token smoke configuration.

All generated datasets and model/cache files stay out of Git. Only the scripts,
source config, tests and documentation are versioned.

## Validation

```bash
python test_function/check_preservation_data.py
```

Tests cover source formatting, reproducible sampling, duplicate removal, insufficient
unique samples, response boundaries, padding, truncation and tokenizer mismatch.
Model generation and GPU training are separate checks; these unit tests do not
establish end-to-end training correctness.
