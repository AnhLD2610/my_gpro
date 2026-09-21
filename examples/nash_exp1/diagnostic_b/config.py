# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Explicit experimental choices; unresolved author inputs are never guessed."""

import copy
import math
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURE_GEOMETRIES = {
    "delta_proxy": "delta_selected_token_gradient_proxy",
    "exact_tiled_head": "unprojected_output_path_lm_head_proxy",
}
FEATURE_LABELS = {
    "delta_selected_token_gradient_proxy": "DelTA token-gradient proxy",
    "unprojected_output_path_lm_head_proxy": "LM-head proxy",
    "synthetic_exact_gram": "Synthetic feature proxy",
}


def feature_geometry(config):
    backend = config["features"]["backend"]
    # The synthetic backend is injected only by CPU tests; validation rejects it
    # for production runs.
    return FEATURE_GEOMETRIES.get(backend, backend)


ENV_OVERRIDES = {
    "NASH_MAX_NEW_TOKENS": ("sampling.max_new_tokens", int),
    "NASH_RADIUS_C": ("routing.radius_coefficient", float),
    "NASH_SEED": ("seed", int),
    "NASH_DAPO_ID": ("train.id", str),
    "NASH_DAPO_REVISION": ("train.revision", str),
    "NASH_DAPO_PATH": ("train.local_path", str),
    "NASH_PROMPT_WRAPPER": ("prompt.wrapper_file", str),
    "NASH_VERIFIER": ("verifier.backend", str),
    "NASH_FEATURE_BACKEND": ("features.backend", str),
    "NASH_MODEL_REVISION": ("model.revision", str),
    "NASH_MODEL_PATH": ("model.local_path", str),
    "NASH_TENSOR_PARALLEL_SIZE": ("generation.tensor_parallel_size", int),
    "NASH_GPU_MEMORY_GIB": ("memory.gpu_budget_gib", float),
    "NASH_CPU_MEMORY_GIB": ("memory.cpu_budget_gib", float),
    "NASH_DISK_BUDGET_GIB": ("memory.disk_budget_gib", float),
}


class ConfigurationError(ValueError):
    """Required scientific choices or execution prerequisites are unresolved."""


def dotted(config, name):
    """Get a nested config entry."""
    value = config
    for component in name.split("."):
        value = value.get(component) if isinstance(value, dict) else None
    return value


def load_config(path):
    """Load YAML and documented environment overrides without external I/O."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a YAML mapping")
    for variable, (name, cast) in ENV_OVERRIDES.items():
        if os.environ.get(variable):
            target = config
            components = name.split(".")
            for key in components[:-1]:
                target = target.setdefault(key, {})
            target[components[-1]] = cast(os.environ[variable])
    for field in ("prompt.wrapper_file", "train.local_path", "model.local_path"):
        value = dotted(config, field)
        if value:
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            section, key = field.split(".")
            config[section][key] = str(resolved.resolve())
    return config


def validate_config(config, *, enable_round2=False):
    """Reject unresolved choices and departures from the locked diagnostic."""
    errors = []
    round_id = config.get("round_id")
    if round_id not in ("round1", "round2"):
        errors.append("round_id must be round1 or round2")
    if round_id == "round2" and not enable_round2:
        errors.append("Round 2 is disabled; explicit author approval and --enable-round2 are required")
    required = [
        "sampling.max_new_tokens",
        "routing.radius_coefficient",
        "seed",
        "prompt.wrapper_file",
        "verifier.backend",
    ]
    if not dotted(config, "train.local_path"):
        required += ["train.id", "train.revision"]
    for field in required:
        if dotted(config, field) is None or dotted(config, field) == "":
            errors.append(f"unresolved author setting: {field}")
    if (
        not dotted(config, "train.local_path")
        and dotted(config, "train.revision")
        and not re.fullmatch(r"[0-9a-fA-F]{40}", str(dotted(config, "train.revision")))
    ):
        errors.append("train.revision must be an immutable 40-character commit SHA, or supply train.local_path")
    if config.get("seed") is not None and (type(config["seed"]) is not int or config["seed"] < 0):
        errors.append("seed must be a nonnegative integer")
    if dotted(config, "model.id") != "Qwen/Qwen3-1.7B-Base":
        errors.append("model.id must be the locked Qwen/Qwen3-1.7B-Base checkpoint")
    if dotted(config, "model.max_model_len") != 32768:
        errors.append("model.max_model_len must be 32768")
    if dotted(config, "model.dtype") not in ("bfloat16", "float16", "float32"):
        errors.append("model.dtype must be bfloat16, float16, or float32")
    expected = {"round1": (64, 30, "MathArena/hmmt_feb_2025"), "round2": (128, 100, "RUC-AIBOX/OlymMATH")}
    if round_id in expected:
        train_count, heldout_count, heldout_id = expected[round_id]
        if dotted(config, "train.n_problem") != train_count or dotted(config, "heldout.n_problem") != heldout_count:
            errors.append(
                f"{round_id} requires {train_count} training prompts and all {heldout_count} held-out prompts"
            )
        if dotted(config, "heldout.id") != heldout_id:
            errors.append(f"{round_id} requires held-out dataset {heldout_id}")
    if config.get("n_rollout") != 32:
        errors.append("n_rollout must be 32 for both datasets")
    if round_id == "round2" and (dotted(config, "heldout.config"), dotted(config, "heldout.split")) != (
        "en-hard",
        "test",
    ):
        errors.append("Round 2 requires OlymMATH config en-hard, split test")
    sampling = config.get("sampling", {})
    if (sampling.get("top_p"), sampling.get("top_k"), sampling.get("min_p")) != (1.0, -1, 0.0):
        errors.append("Full-support sampling requires top_p=1, top_k=-1, min_p=0")
    if sampling.get("temperature") != 1.0:
        errors.append("This experiment is approved for temperature=1.0; non-unit temperature needs author approval")
    for field in ("sampling.max_new_tokens", "routing.radius_coefficient", "routing.epsilon"):
        value = dotted(config, field)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            errors.append(f"{field} must be finite and positive")
    if sampling.get("max_new_tokens") is not None and type(sampling["max_new_tokens"]) is not int:
        errors.append("sampling.max_new_tokens must be an integer")
    if isinstance(sampling.get("max_new_tokens"), int) and sampling["max_new_tokens"] >= 32768:
        errors.append("max_new_tokens must leave room for the prompt within 32768; prompts are never truncated")
    wrapper = dotted(config, "prompt.wrapper_file")
    if wrapper and not Path(wrapper).is_file():
        errors.append(f"Approved wrapper file is missing: {wrapper}")
    for name in ("train.local_path", "model.local_path"):
        value = dotted(config, name)
        if value and not Path(value).exists():
            errors.append(f"{name} does not exist: {value}")
    if dotted(config, "verifier.backend") not in (None, "math_verify", "custom"):
        errors.append("verifier.backend must be explicitly selected as math_verify or custom")
    if dotted(config, "verifier.backend") == "custom" and not dotted(config, "verifier.callable"):
        errors.append("custom verifier requires verifier.callable='module:function' returning reward and parse_status")
    if config.get("verifier", {}).get("answer_format", "auto") not in ("auto", "boxed"):
        errors.append("verifier.answer_format must be auto or boxed")
    if dotted(config, "features.backend") not in FEATURE_GEOMETRIES:
        errors.append("features.backend must be delta_proxy (DelTA) or exact_tiled_head")
    if errors:
        raise ConfigurationError("Required configuration/preflight decisions:\n- " + "\n- ".join(errors))
    return copy.deepcopy(config)
