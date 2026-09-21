#!/usr/bin/env python3
"""Check source sampling and stored response boundaries without model downloads."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preservation_data import (
    preservation_prompt_ids, prompt_key, sample_source, source_prompt, stored_context_arrays, tokenizer_fingerprint,
)


class ToyTokenizer:
    pad_token_id = 0
    special_tokens_map = {"pad_token": "<pad>"}
    chat_template = "test-template"

    def get_vocab(self):
        return {"<pad>": 0, "a": 1, "b": 2}


class PreservationDataTests(unittest.TestCase):
    def test_chat_encoding_extracts_integer_ids(self):
        from collections import UserDict
        from unittest.mock import Mock
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = UserDict({"input_ids": [1, 2, 3]})
        self.assertEqual(preservation_prompt_ids(tokenizer, "hello"), [1, 2, 3])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            tokenize=True, add_generation_prompt=True, return_dict=True,
        )

    def test_source_formatting_excludes_answers(self):
        self.assertEqual(source_prompt({"instruction": "Explain", "input": "gravity", "output": "secret"}, "general"), "Explain\n\ngravity")
        self.assertEqual(source_prompt({"question": "2+2?", "answer": "4"}, "math"), "2+2?")
        self.assertEqual(source_prompt({"query": "Solve task", "prompt": "import math", "response": "solution"}, "code"), "Solve task")

    def test_sampling_is_unique_reproducible_and_respects_count(self):
        rows = [{"question": " A  B "}, {"question": "a b"}, {"question": ""}, {"question": "C"}]
        source = {"domain": "math", "repo_id": "fixture", "revision": "abc", "split": "train", "count": 2}
        first = sample_source(rows, source, 66, set())
        self.assertEqual(first, sample_source(rows, source, 66, set()))
        self.assertEqual({r["prompt_sha256"] for r in first}, {prompt_key("a b"), prompt_key("c")})
        self.assertTrue(all(r["source_revision"] == "abc" for r in first))
        with self.assertRaises(ValueError):
            sample_source(rows, {**source, "count": 3}, 66, set())
        with self.assertRaises(ValueError):
            sample_source(rows, source, 66, {prompt_key("a b")})

    def test_response_masks_preserve_boundary_and_padding(self):
        tokenizer = ToyTokenizer()
        identity = tokenizer_fingerprint(tokenizer)
        rows = [
            {"input_ids": [1, 2, 3, 4, 5], "response_start": 3, "tokenizer_sha256": identity},
            {"input_ids": [1, 2, 3], "response_start": 1, "tokenizer_sha256": identity},
        ]
        ids, attention, masks = stored_context_arrays(tokenizer, rows, max_length=4)
        self.assertEqual(ids, [[1, 2, 3, 4], [1, 2, 3, 0]])
        self.assertEqual(attention, [[1, 1, 1, 1], [1, 1, 1, 0]])
        self.assertEqual(masks, [[0, 0, 0, 1], [0, 1, 1, 0]])
        # Next-token loss at position start-1 predicts the first response token.
        self.assertEqual([mask[1:] for mask in masks], [[0, 0, 1], [1, 1, 0]])
        with self.assertRaisesRegex(ValueError, "removes all response"):
            stored_context_arrays(tokenizer, rows, max_length=3)
        with self.assertRaisesRegex(ValueError, "tokenizer differs"):
            stored_context_arrays(tokenizer, [{**rows[0], "tokenizer_sha256": "wrong"}], max_length=4)
        with self.assertRaisesRegex(ValueError, "response_start"):
            stored_context_arrays(tokenizer, [{**rows[0], "response_start": 0}], max_length=4)


if __name__ == "__main__":
    unittest.main()
