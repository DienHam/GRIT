from __future__ import annotations

from typing import Any

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\\n' + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)


def build_chat_prompt(tokenizer: Any, prompt: str, system_prompt: str = "") -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_with_vllm(
    model_path: str,
    prompts: list[str],
    batch_size: int,
    tensor_parallel_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM generation requires 'vllm' plus 'transformers'. "
            "Install the full eval environment before running generation."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = [
            build_chat_prompt(tokenizer, prompt)
            for prompt in prompts[start : start + batch_size]
        ]
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        outputs.extend(output.outputs[0].text for output in batch_outputs)
    return outputs

