# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Schema-checked datasets, immutable prompt manifests, and explicit verification."""

import importlib
import logging
import re
import unicodedata
from pathlib import Path

from .config import ConfigurationError
from .storage import atomic_json, digest, file_hash, read_json, write_records

LOG = logging.getLogger(__name__)


def normalized_problem(text):
    """Exact overlap normalization: NFKC, case-folding and whitespace collapse."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def adapt_row(row, dataset_kind, row_index):
    """Validate supported schemas without guessing an answer or task field."""
    if dataset_kind == "dapo":
        if "prompt" not in row or not isinstance(row.get("reward_model"), dict):
            raise ConfigurationError("DAPO schema requires prompt messages and reward_model.ground_truth")
        messages = row["prompt"]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(item, dict)
                or item.get("role") not in ("system", "user", "assistant")
                or not isinstance(item.get("content"), str)
                for item in messages
            )
        ):
            raise ConfigurationError("DAPO prompt must be a nonempty list of role/content chat messages")
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        if not user_messages or messages[-1]["role"] != "user":
            raise ConfigurationError("DAPO prompt must end in a user message")
        problem = user_messages[-1]
        answer = row["reward_model"].get("ground_truth")
        extra = row.get("extra_info") or {}
        row_id = extra.get("index", row_index)
    elif dataset_kind in ("hmmt", "olymmath"):
        # Official HMMT and OlymMATH exports use problem/answer (some HMMT exports capitalize).
        schemas = [("problem", "answer"), ("Problem", "Answer")]
        matches = [(problem, answer) for problem, answer in schemas if problem in row and answer in row]
        if len(matches) != 1:
            raise ConfigurationError(f"{dataset_kind} needs an unambiguous problem/answer schema; got {list(row)}")
        problem_key, answer_key = matches[0]
        problem, answer, messages = row[problem_key], row[answer_key], None
        row_id = row.get("id", row.get("index", row_index))
    else:
        raise ConfigurationError(f"Unknown dataset adapter {dataset_kind}")
    if not isinstance(problem, str) or not problem.strip() or answer is None or str(answer).strip() == "":
        raise ConfigurationError(f"Missing problem/answer in {dataset_kind} row {row_index}")
    return {
        "row_id": str(row_id),
        "dataset_row_index": int(row_index),
        "problem": problem,
        "answer": str(answer),
        "messages": messages,
    }


def validate_wrapper(wrapper):
    """Validate an author-supplied task wrapper; only problem substitution is allowed."""
    import string

    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("user_template"), str):
        raise ConfigurationError("Wrapper JSON must contain user_template with a {problem} placeholder")
    fields = [field for _, field, _, _ in string.Formatter().parse(wrapper["user_template"]) if field is not None]
    if not fields or any(field != "problem" for field in fields):
        raise ConfigurationError("Wrapper must substitute {problem} only; no answers or held-out information")
    if wrapper.get("system") is not None and not isinstance(wrapper["system"], str):
        raise ConfigurationError("Wrapper system must be a string or null")
    if type(wrapper.get("apply_to_existing_user_message", False)) is not bool:
        raise ConfigurationError("apply_to_existing_user_message must be a boolean")
    kwargs = wrapper.get("chat_template_kwargs", {})
    if set(kwargs) - {"enable_thinking"}:
        raise ConfigurationError("Only explicit enable_thinking is supported in chat_template_kwargs")
    return wrapper


def render_prompt(row, tokenizer, wrapper):
    """Apply the approved wrapper and the pinned model tokenizer's chat template."""
    validate_wrapper(wrapper)
    messages = row["messages"]
    if messages is None:
        messages = []
        if wrapper.get("system"):
            messages.append({"role": "system", "content": wrapper["system"]})
        messages.append({"role": "user", "content": wrapper["user_template"].format(problem=row["problem"])})
    elif wrapper.get("apply_to_existing_user_message", False):
        # Retain the original roles and earlier messages; do not mutate dataset rows.
        messages = [dict(message) for message in messages]
        messages[-1]["content"] = wrapper["user_template"].format(problem=messages[-1]["content"])
    if not getattr(tokenizer, "chat_template", None):
        raise ConfigurationError(
            "The pinned base tokenizer has no chat_template; author must resolve prompt convention"
        )
    kwargs = wrapper.get("chat_template_kwargs", {})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
    return {
        **row,
        "messages": messages,
        "prompt_text": text,
        "prompt_token_ids": list(token_ids),
        "prompt_hash": digest({"text": text, "token_ids": list(token_ids)}),
        "prompt_token_count": len(token_ids),
    }


def check_overlap(train_rows, heldout_rows):
    """Stop on exact normalized-text overlap, before observing rewards."""
    heldout = {normalized_problem(row["problem"]): row["row_id"] for row in heldout_rows}
    overlaps = [
        {"train_row_id": row["row_id"], "heldout_row_id": heldout[normalized_problem(row["problem"])]}
        for row in train_rows
        if normalized_problem(row["problem"]) in heldout
    ]
    if overlaps:
        raise ConfigurationError(f"TRAIN_HELDOUT_OVERLAP: {overlaps}; explicit author approval required to proceed")
    return overlaps


def local_snapshot(path):
    """Content fingerprint of a local dataset snapshot."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise ConfigurationError(f"Empty local snapshot {path}")
    return {str(file.relative_to(path) if path.is_dir() else file.name): file_hash(file) for file in files}


def load_dataset_snapshot(spec, *, require_immutable=False):
    """Server-only dataset loading; tests inject synthetic rows instead."""
    local = spec.get("local_path")
    requested = spec.get("revision")
    if not local and require_immutable and not re.fullmatch(r"[0-9a-fA-F]{40}", requested or ""):
        raise ConfigurationError("DAPO revision must be an immutable 40-character commit SHA, or use local_path")

    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi

    if local:
        content = local_snapshot(local)
        if Path(local).is_file():
            dataset = load_dataset("parquet", data_files={spec["split"]: local}, split=spec["split"])
        else:
            dataset = load_from_disk(local)
            if hasattr(dataset, "keys"):
                dataset = dataset[spec["split"]]
        revision = "sha256:" + digest(content)
    else:
        revision = HfApi().dataset_info(spec["id"], revision=requested).sha
        dataset = load_dataset(spec["id"], name=spec.get("config"), split=spec["split"], revision=revision)
        content = None
    schema = dataset.features.to_dict()
    return dataset, {
        "id": spec.get("id"),
        "revision": revision,
        "split": spec["split"],
        "config": spec.get("config"),
        "local_files": content,
        "schema": schema,
        "rows": len(dataset),
    }


def prepare_manifests(config, output_dir, model_lock):
    """Select the immutable D sample and all Q rows before generation/model loading."""
    import numpy as np
    from transformers import AutoTokenizer

    root = Path(output_dir)
    wrapper = validate_wrapper(read_json(config["prompt"]["wrapper_file"]))
    model_path = config["model"].get("local_path") or config["model"]["id"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=model_lock["tokenizer_revision"], trust_remote_code=False
    )
    train, train_info = load_dataset_snapshot(config["train"], require_immutable=True)
    heldout, heldout_info = load_dataset_snapshot(config["heldout"])
    if len(heldout) != config["heldout"]["n_problem"]:
        raise ConfigurationError(
            f"Held-out dataset must contain exactly {config['heldout']['n_problem']} rows, got {len(heldout)}"
        )
    count = config["train"]["n_problem"]
    if len(train) < count:
        raise ConfigurationError("DAPO snapshot has fewer rows than the locked sample size")
    selected = np.random.default_rng(config["seed"]).choice(len(train), count, replace=False).tolist()
    train_rows = [adapt_row(train[index], "dapo", index) for index in selected]
    heldout_kind = "hmmt" if config["round_id"] == "round1" else "olymmath"
    heldout_rows = [adapt_row(heldout[index], heldout_kind, index) for index in range(len(heldout))]
    check_overlap(train_rows, heldout_rows)
    for rows, split in ((train_rows, "train"), (heldout_rows, "heldout")):
        for index, row in enumerate(rows):
            rows[index] = {**render_prompt(row, tokenizer, wrapper), "prompt_index": index, "split": split}
            if (
                rows[index]["prompt_token_count"] + config["sampling"]["max_new_tokens"]
                > config["model"]["max_model_len"]
            ):
                raise ConfigurationError(
                    f"Prompt {split}/{index} plus response cap exceeds max_model_len; no truncation applied"
                )
        if len({row["row_id"] for row in rows}) != len(rows):
            raise ConfigurationError(f"Duplicate row IDs in {split} manifest")
        write_records(root / f"{split}_prompt_manifest.parquet", rows)
    manifest = {
        "round_id": config["round_id"],
        "seed": config["seed"],
        "model": model_lock,
        "train": train_info,
        "heldout": heldout_info,
        "train_row_indices": selected,
        "train_prompts": digest(train_rows),
        "heldout_prompts": digest(heldout_rows),
        "wrapper": wrapper,
        "wrapper_hash": digest(wrapper),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_hash": digest(tokenizer.chat_template),
        "sampling": config["sampling"],
        "overlap_count": 0,
    }
    atomic_json(root / "manifest.json", manifest)
    LOG.info("Prepared fixed prompts and schema snapshots: %s", manifest)
    return [root / "manifest.json", root / "train_prompt_manifest.parquet", root / "heldout_prompt_manifest.parquet"]


def last_boxed_answer(response):
    r"""Isolate the last \boxed{...}, retaining nested LaTeX braces.

    A malformed final box is a parse failure; never fall back to an earlier box
    or a number in the reasoning. Escaped braces do not affect nesting depth.
    """
    start = response.rfind(r"\boxed")
    if start < 0:
        return None
    opening = re.match(r"\\boxed\s*\{", response[start:])
    if opening is None:
        return None
    begin = start + opening.end() - 1
    depth, escaped = 0, False
    for index in range(begin, len(response)):
        character = response[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return r"\boxed{" + response[begin + 1 : index] + "}"
    return None


class MathVerifier:
    """Explicit symbolic backend with separate prediction/gold parse failures."""

    def __init__(self, config):
        self.config = config
        self.backend = config["backend"]
        self.answer_format = config.get("answer_format", "auto")
        if self.answer_format not in ("auto", "boxed"):
            raise ConfigurationError("verifier.answer_format must be auto or boxed")
        if self.backend == "custom":
            module, function = config["callable"].split(":", 1)
            self.custom = getattr(importlib.import_module(module), function)
        elif self.backend != "math_verify":
            raise ConfigurationError("Choose an approved verifier before reward assignment")

    def score(self, response, answer):
        """Return a binary reward, parser status and diagnostic reason."""
        if self.backend == "custom":
            result = self.custom(response=response, answer=answer)
            if not isinstance(result, dict) or result.get("reward") not in (0, 1) or "parse_status" not in result:
                raise ValueError("Custom verifier must return {reward: 0|1, parse_status: str, reason: str}")
            return result
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

        timeout = self.config["timeout_seconds"]
        try:
            gold = parse(
                "\\boxed{" + str(answer) + "}",
                extraction_config=[LatexExtractionConfig()],
                fallback_mode="no_fallback",
                parsing_timeout=timeout,
            )
            if not gold:
                return {"reward": 0, "parse_status": "gold_parse_failure", "reason": "unparseable reference answer"}
            if self.answer_format == "boxed":
                response = last_boxed_answer(response)
                if response is None:
                    return {
                        "reward": 0,
                        "parse_status": "prediction_parse_failure",
                        "reason": "missing or malformed final boxed answer",
                    }
            prediction = parse(
                response,
                extraction_config=(
                    [LatexExtractionConfig(boxed_match_priority=0)]
                    if self.answer_format == "boxed"
                    else [ExprExtractionConfig(), LatexExtractionConfig()]
                ),
                fallback_mode="no_fallback",
                parsing_timeout=timeout,
            )
            if not prediction:
                return {"reward": 0, "parse_status": "prediction_parse_failure", "reason": "no extracted expression"}
            correct = bool(verify(gold, prediction, timeout_seconds=timeout))
            return {"reward": int(correct), "parse_status": "parsed", "reason": "correct" if correct else "incorrect"}
        except Exception as exc:
            return {"reward": 0, "parse_status": "verifier_exception", "reason": f"{type(exc).__name__}: {exc}"}
