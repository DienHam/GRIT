"""Qwen3Guard response moderation using its official tokenizer chat template."""

from __future__ import annotations

import concurrent.futures
from functools import lru_cache
import os
import re
import time

from openai import OpenAI, OpenAIError

GUARD_BASE_URL = os.environ.get("NSPO_GUARD_BASE_URL", "http://127.0.0.1:52001/v1")
GUARD_MODEL = os.environ.get("NSPO_GUARD_MODEL", "Qwen/Qwen3Guard-Gen-0.6B")


@lru_cache(maxsize=1)
def _client():
    return OpenAI(api_key=os.environ.get("NSPO_GUARD_API_KEY", "EMPTY"),
                  base_url=GUARD_BASE_URL, timeout=float(os.environ.get("NSPO_GUARD_TIMEOUT", "120")),
                  max_retries=0)


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(GUARD_MODEL, trust_remote_code=False)


def parse_guard_response(content):
    labels = re.findall(r"^Safety:\s*(Safe|Unsafe|Controversial)\s*$", content, flags=re.MULTILINE)
    refusals = re.findall(r"^Refusal:\s*(Yes|No)\s*$", content, flags=re.MULTILINE)
    if len(labels) != 1 or len(refusals) != 1:
        raise ValueError("Qwen3Guard must return one structured Safety label and one Refusal label")
    return labels[0], refusals[0]


def _score_one(pair):
    prompt, response = pair
    rendered = _tokenizer().apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False,
    )
    start = time.monotonic()
    parse_errors = request_errors = 0
    attempts = int(os.environ.get("NSPO_GUARD_ATTEMPTS", "3"))
    if not 1 <= attempts <= 5:
        raise ValueError("NSPO_GUARD_ATTEMPTS must be between 1 and 5")
    for attempt in range(attempts):
        try:
            result = _client().completions.create(model=GUARD_MODEL, prompt=rendered,
                                                 temperature=0.0, max_tokens=128)
        except OpenAIError:
            request_errors += 1
        else:
            try:
                label, refusal = parse_guard_response(result.choices[0].text or "")
            except (ValueError, IndexError):
                parse_errors += 1
            else:
                return {"score": 0.0 if label == "Safe" else -1.0,
                        "guard/parse_errors": float(parse_errors),
                        "guard/request_errors": float(request_errors),
                        "guard/refusal_rate": float(refusal == "Yes"),
                        "guard/seconds": time.monotonic() - start}
        if attempt + 1 < attempts:
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"Qwen3Guard failed after {attempts} attempts: "
                       f"parse_errors={parse_errors}, request_errors={request_errors}")


def compute_score_batched(data_source=None, solution_str=None, solution_strs=None,
                          ground_truth=None, extra_info=None, data_sources=None,
                          ground_truths=None, extra_infos=None, **kwargs):
    responses = solution_strs if solution_strs is not None else solution_str
    if isinstance(responses, str):
        responses = [responses]
    if responses is None:
        raise ValueError("Missing policy responses")
    extras = extra_infos if extra_infos is not None else [extra_info]
    if len(extras) != len(responses):
        raise ValueError("Prompt/response batch length mismatch")
    pairs = []
    for response, extra in zip(responses, extras):
        prompt = extra.get("raw_prompt") if isinstance(extra, dict) else None
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str):
            raise ValueError("Reward requires extra_info.raw_prompt and the current policy response")
        pairs.append((prompt, response))
    if not pairs:
        return []
    _tokenizer()  # Initialize before threads to avoid duplicated tokenizer loads.
    workers = max(1, min(len(pairs), int(os.environ.get("NSPO_GUARD_WORKERS", "4"))))
    if workers == 1:
        return [_score_one(pair) for pair in pairs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_score_one, pairs))
