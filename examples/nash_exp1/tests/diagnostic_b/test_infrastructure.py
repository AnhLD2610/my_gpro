# Copyright 2026 The Nash Credit Routing Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Infrastructure contracts using local files, fake tokenizers and real CPU math verification."""

import builtins
import copy
import json
import logging
import warnings
from pathlib import Path

import pytest
from diagnostic_b import config, data, logging_utils, storage

EXP_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def valid_config(tmp_path, monkeypatch):
    for variable in config.ENV_OVERRIDES:
        monkeypatch.delenv(variable, raising=False)
    settings = config.load_config(EXP_ROOT / "configs" / "diagnostic_b_round1.yaml")
    wrapper = tmp_path / "approved_wrapper.json"
    wrapper.write_text(json.dumps({"user_template": "Solve: {problem}"}))
    settings["prompt"]["wrapper_file"] = str(wrapper)
    settings["sampling"]["max_new_tokens"] = 256
    settings["routing"]["radius_coefficient"] = 0.2
    settings["seed"] = 42
    settings["verifier"]["backend"] = "math_verify"
    settings["train"]["id"] = "synthetic/DAPO"
    settings["train"]["revision"] = "a" * 40
    return settings


def _deny_external_imports(monkeypatch):
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "datasets", "vllm", "huggingface_hub"}:
            imported.append(name)
            raise AssertionError(f"External/model library imported before validation: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return imported


def test_unresolved_author_gate_runs_before_model_or_dataset_imports(tmp_path, monkeypatch):
    for variable in config.ENV_OVERRIDES:
        monkeypatch.delenv(variable, raising=False)
    imported = _deny_external_imports(monkeypatch)
    settings = config.load_config(EXP_ROOT / "configs" / "diagnostic_b_round1.yaml")
    fields = (
        "sampling.max_new_tokens",
        "routing.radius_coefficient",
        "seed",
        "prompt.wrapper_file",
        "verifier.backend",
        "train.id",
        "train.revision",
    )
    for field in fields:
        target = settings
        components = field.split(".")
        for component in components[:-1]:
            target = target[component]
        target[components[-1]] = None
    with pytest.raises(config.ConfigurationError) as error:
        config.validate_config(settings)
    message = str(error.value)
    for field in fields:
        assert field in message
    assert imported == []


def test_valid_configuration_is_copied_and_round2_requires_explicit_gate(valid_config, monkeypatch):
    imported = _deny_external_imports(monkeypatch)
    original = copy.deepcopy(valid_config)
    result = config.validate_config(valid_config)
    assert result == valid_config == original
    assert result is not valid_config
    valid_config.update(round_id="round2")
    valid_config["train"]["n_problem"] = 128
    valid_config["heldout"].update(id="RUC-AIBOX/OlymMATH", n_problem=100, config="en-hard", split="test")
    with pytest.raises(config.ConfigurationError, match="--enable-round2"):
        config.validate_config(valid_config)
    assert config.validate_config(valid_config, enable_round2=True)["round_id"] == "round2"
    assert imported == []


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("sampling.top_p", 0.9),
        ("sampling.top_k", 50),
        ("sampling.min_p", 0.1),
        ("sampling.temperature", 0.7),
        ("sampling.max_new_tokens", 32768),
        ("sampling.max_new_tokens", 2.5),
        ("sampling.max_new_tokens", True),
        ("routing.radius_coefficient", float("nan")),
        ("routing.radius_coefficient", float("inf")),
        ("routing.epsilon", float("nan")),
        ("seed", -1),
        ("seed", 1.5),
        ("seed", True),
        ("train.revision", "main"),
        ("verifier.answer_format", "unsupported"),
    ],
)
def test_scientific_config_rejects_invalid_values_at_author_gate(valid_config, field, bad_value):
    target = valid_config
    components = field.split(".")
    for component in components[:-1]:
        target = target[component]
    target[components[-1]] = bad_value
    with pytest.raises(config.ConfigurationError):
        config.validate_config(valid_config)


def test_immutable_dataset_revision_is_checked_before_optional_imports(monkeypatch):
    imported = _deny_external_imports(monkeypatch)
    with pytest.raises(config.ConfigurationError, match="immutable|revision"):
        data.load_dataset_snapshot(
            {"id": "synthetic/DAPO", "revision": "main", "split": "train"}, require_immutable=True
        )
    assert imported == []


class FakeTokenizer:
    chat_template = "synthetic tokenizer template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(copy.deepcopy({"messages": messages, "kwargs": kwargs}))
        assert add_generation_prompt is True
        text = "\n".join(message["role"] + ": " + message["content"] for message in messages) + "\nassistant:"
        return list(text.encode()) if tokenize else text


def test_dapo_adapter_preserves_existing_messages_and_excludes_gold_from_prompts():
    messages = [
        {"role": "system", "content": "Original system instruction."},
        {"role": "user", "content": "Find the roots of the polynomial."},
    ]
    raw = {
        "prompt": copy.deepcopy(messages),
        "reward_model": {"ground_truth": "PRIVATE_GOLD_ANSWER"},
        "extra_info": {"index": "dapo-row-7"},
    }
    adapted = data.adapt_row(raw, "dapo", 7)
    tokenizer = FakeTokenizer()
    rendered = data.render_prompt(adapted, tokenizer, {"user_template": "Unused wrapper: {problem}"})
    assert rendered["row_id"] == "dapo-row-7"
    assert rendered["messages"] == messages == raw["prompt"]
    assert rendered["answer"] == "PRIVATE_GOLD_ANSWER"
    assert "PRIVATE_GOLD_ANSWER" not in rendered["prompt_text"]
    assert all(call["messages"] == messages for call in tokenizer.calls)
    assert rendered["prompt_token_count"] == len(rendered["prompt_token_ids"])


def test_boxed_wrapper_applies_same_instruction_to_dapo_and_heldout_without_mutation():
    wrapper = data.validate_wrapper(storage.read_json(EXP_ROOT / "configs" / "boxed_prompt.json"))
    problem = "Find the roots of {x^2 - 1}."
    messages = [
        {"role": "system", "content": "Original system instruction."},
        {"role": "user", "content": problem},
    ]
    raw_dapo = {"prompt": copy.deepcopy(messages), "reward_model": {"ground_truth": "PRIVATE_GOLD"}}
    dapo = data.adapt_row(raw_dapo, "dapo", 0)
    heldout = data.adapt_row({"problem": problem, "answer": "PRIVATE_GOLD"}, "hmmt", 0)
    tokenizer = FakeTokenizer()
    rendered_dapo = data.render_prompt(dapo, tokenizer, wrapper)
    rendered_heldout = data.render_prompt(heldout, tokenizer, wrapper)
    expected = problem + "\n\nLet's think step by step and output the final answer within \\boxed{}."
    assert rendered_dapo["messages"] == [messages[0], {"role": "user", "content": expected}]
    assert rendered_heldout["messages"] == [{"role": "user", "content": expected}]
    assert dapo["messages"] == raw_dapo["prompt"] == messages
    assert "PRIVATE_GOLD" not in rendered_dapo["prompt_text"] + rendered_heldout["prompt_text"]


@pytest.mark.parametrize("kind", ["hmmt", "olymmath"])
@pytest.mark.parametrize("keys", [("problem", "answer"), ("Problem", "Answer")])
def test_heldout_adapters_apply_wrapper_without_answer_leakage(kind, keys):
    row = {keys[0]: "Solve this symbolic problem.", keys[1]: "PRIVATE_SYMBOLIC_GOLD", "id": "heldout-3"}
    adapted = data.adapt_row(row, kind, 3)
    wrapper = {
        "system": "Approved system.",
        "user_template": "Problem: {problem}",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    tokenizer = FakeTokenizer()
    rendered = data.render_prompt(adapted, tokenizer, wrapper)
    assert rendered["row_id"] == "heldout-3"
    assert "PRIVATE_SYMBOLIC_GOLD" not in rendered["prompt_text"]
    assert "PRIVATE_SYMBOLIC_GOLD" not in json.dumps(tokenizer.calls)
    assert rendered["messages"] == [
        {"role": "system", "content": "Approved system."},
        {"role": "user", "content": "Problem: Solve this symbolic problem."},
    ]
    assert tokenizer.calls[0]["kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    "wrapper",
    [
        {"user_template": "Answer: {answer}"},
        {"user_template": "{problem} {answer}"},
        {"user_template": "No problem placeholder"},
        {"user_template": "{problem}", "apply_to_existing_user_message": "yes"},
    ],
)
def test_wrapper_rejects_answer_fields_and_missing_problem(wrapper):
    with pytest.raises(config.ConfigurationError):
        data.validate_wrapper(wrapper)


def test_schema_ambiguity_missing_gold_and_missing_chat_template_are_errors():
    with pytest.raises(config.ConfigurationError, match="unambiguous"):
        data.adapt_row({"problem": "x", "answer": "1", "Problem": "y", "Answer": "2"}, "hmmt", 0)
    with pytest.raises(config.ConfigurationError, match="Missing"):
        data.adapt_row({"problem": "x", "answer": None}, "olymmath", 0)
    tokenizer = FakeTokenizer()
    tokenizer.chat_template = None
    with pytest.raises(config.ConfigurationError, match="no chat_template"):
        data.render_prompt(
            data.adapt_row({"problem": "x", "answer": "1"}, "hmmt", 0), tokenizer, {"user_template": "{problem}"}
        )


def test_overlap_stops_after_exact_normalization():
    train = [{"row_id": "D1", "problem": "  FIND   Ｘ + y\n"}]
    heldout = [{"row_id": "Q1", "problem": "find x + Y"}]
    with pytest.raises(config.ConfigurationError, match="TRAIN_HELDOUT_OVERLAP.*D1.*Q1"):
        data.check_overlap(train, heldout)
    assert data.check_overlap(train, [{"row_id": "Q2", "problem": "find x - y"}]) == []


@pytest.mark.parametrize(
    ("response", "answer", "reward", "status"),
    [
        (r"\boxed{\frac{1}{2}}", "0.5", 1, "parsed"),
        (r"\boxed{2*x+2}", "2(x+1)", 1, "parsed"),
        (r"\boxed{2}", "3", 0, "parsed"),
        ("I cannot solve this.", "3", 0, "prediction_parse_failure"),
        (r"\boxed{1}", r"\notacommand", 0, "gold_parse_failure"),
    ],
)
def test_real_symbolic_verifier_reports_equivalence_and_parse_failures(response, answer, reward, status):
    verifier = data.MathVerifier({"backend": "math_verify", "timeout_seconds": 5})
    result = verifier.score(response, answer)
    assert result["reward"] == reward
    assert result["parse_status"] == status
    assert result["reason"]


@pytest.mark.parametrize(
    ("response", "answer", "reward", "status"),
    [
        (r"\boxed{\frac{1}{2}}", "0.5", 1, "parsed"),
        (r"\boxed{2*x+2}", "2(x+1)", 1, "parsed"),
        ("The final answer is 42.", "42", 0, "prediction_parse_failure"),
        (r"First \boxed{17}; final answer: \boxed{42}.", "42", 1, "parsed"),
        (r"First \boxed{42}; final answer: \boxed{17}.", "42", 0, "parsed"),
        (r"\boxed{17}. The reasoning mentions 42.", "42", 0, "parsed"),
        (r"First \boxed{42}; final answer: \boxed{", "42", 0, "prediction_parse_failure"),
        (r"First \boxed{42}; final answer: \boxed{}", "42", 0, "prediction_parse_failure"),
    ],
)
def test_boxed_verifier_scores_only_the_final_box(response, answer, reward, status):
    verifier = data.MathVerifier({"backend": "math_verify", "answer_format": "boxed", "timeout_seconds": 5})
    result = verifier.score(response, answer)
    assert result["reward"] == reward
    assert result["parse_status"] == status


def test_final_box_extraction_preserves_nested_latex_and_escaped_braces():
    expected = r"\boxed{\left\{\frac{1}{2}, 3\right\}}"
    assert data.last_boxed_answer(r"Earlier \boxed{0}, then " + expected + " trailing text") == expected
    assert data.last_boxed_answer(r"Answer: \boxed   {\frac{1}{2}}") == r"\boxed{\frac{1}{2}}"


def test_resume_after_postgeneration_failure_preserves_one_rollout_cache(tmp_path):
    calls = {"generate": 0, "features": 0}
    cache = tmp_path / "train_rollouts.jsonl.zst"

    def generate():
        calls["generate"] += 1
        storage.write_compressed_records(cache, [{"prompt_index": 0, "rollout_index": 0, "token_ids": [1, 2]}])
        return [cache]

    def fail_features():
        calls["features"] += 1
        raise RuntimeError("synthetic feature failure")

    runner = storage.StageRunner(tmp_path)
    generation_record = runner.run("generate", {"sampling_seed": 7}, generate, immutable=True)
    with pytest.raises(RuntimeError, match="synthetic feature failure"):
        runner.run("features", runner.dependencies(["generate"]), fail_features)
    assert cache.exists()
    state = storage.read_json(tmp_path / "stage_state.json")
    assert state["last_completed_stage"] == "generate"
    assert state["stages"]["features"]["status"] == "failed"
    resumed = storage.StageRunner(tmp_path, resume=True)
    assert resumed.run("generate", {"sampling_seed": 7}, generate, immutable=True) == generation_record
    output = tmp_path / "features.json"

    def success_features():
        calls["features"] += 1
        storage.atomic_json(output, {"ok": True})
        return [output]

    resumed.run("features", resumed.dependencies(["generate"]), success_features)
    assert calls == {"generate": 1, "features": 2}
    assert list(storage.read_compressed_records(cache))[0]["token_ids"] == [1, 2]


@pytest.mark.parametrize("conflict", ["inputs", "tamper", "force"])
def test_completed_generation_rejects_mismatch_tamper_or_force(tmp_path, conflict):
    output = tmp_path / "rollout.json"
    called = []

    def generate():
        called.append(True)
        storage.atomic_json(output, {"tokens": [1, 2, 3]})
        return [output]

    storage.StageRunner(tmp_path).run("generate", {"seed": 1}, generate, immutable=True)
    inputs = {"seed": 2 if conflict == "inputs" else 1}
    if conflict == "tamper":
        output.write_text('{"tokens": [9]}')
    runner = storage.StageRunner(tmp_path, resume=True, force_stage="generate" if conflict == "force" else None)
    with pytest.raises(storage.CacheConflict, match="Immutable"):
        runner.run("generate", inputs, generate, immutable=True)
    assert len(called) == 1


def test_forced_mutable_stage_preserves_unrelated_outputs_and_interruption_state(tmp_path):
    immutable_cache = tmp_path / "cached_rollout.json"
    storage.atomic_json(immutable_cache, {"cached": True})
    original = storage.file_hash(immutable_cache)
    runner = storage.StageRunner(tmp_path)

    def interrupt():
        raise KeyboardInterrupt("synthetic interruption")

    with pytest.raises(KeyboardInterrupt):
        runner.run("statistics", {"version": 1}, interrupt)
    assert storage.read_json(tmp_path / "stage_state.json")["stages"]["statistics"]["status"] == "interrupted"
    assert storage.file_hash(immutable_cache) == original


def test_single_log_contains_command_stage_traceback_resume_warnings_and_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("NASH_LOG_TEE", raising=False)
    secret = "synthetic-secret-value-123"
    monkeypatch.setenv("NASH_TEST_TOKEN", secret)
    log_path = tmp_path / "diagnostic_b_round1.log"
    command = ["python", "-m", "diagnostic_b", "all"]
    resume = command + ["--resume"]

    def failure():
        print(f"synthetic stdout token={secret}")
        logging.getLogger("test").warning("synthetic warning before failure")
        warnings.warn("synthetic captured warning", UserWarning, stacklevel=1)
        raise RuntimeError(f"synthetic failure token={secret}")

    with pytest.raises(RuntimeError):
        with logging_utils.single_log(log_path, command, resume):
            storage.StageRunner(tmp_path).run("heldout_features", {"test": True}, failure)
    text = log_path.read_text()
    for phrase in (
        "COMMAND:",
        "stage=heldout_features",
        "Traceback",
        "RESUME:",
        "--resume",
        "synthetic stdout",
        "synthetic captured warning",
        "Return this one log file:",
    ):
        assert phrase in text
    assert secret not in text
    assert "<redacted>" in text
    assert "UTC=" in text and "LOCAL=" in text


def test_shell_tee_mode_redacts_secrets_before_the_shell_captures_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NASH_LOG_TEE", "1")
    secret = "synthetic-shell-tee-token-987"
    monkeypatch.setenv("HF_TOKEN", secret)
    with logging_utils.single_log(tmp_path / "diagnostic_b_round1.log", ["test"], ["test", "--resume"]):
        print(f"message that shell tee captures: {secret}")
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "<redacted>" in captured.out + captured.err


def test_environment_report_redacts_sensitive_variable_names_without_model_imports(tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "NASH_API_KEY", "NASH_SECRET", "NASH_PASSWORD"):
        monkeypatch.setenv(name, "synthetic-secret")
    monkeypatch.setenv("NASH_PUBLIC_SETTING", "visible")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    monkeypatch.setattr(logging_utils, "command_output", lambda args: {"command": args, "stdout": "synthetic"})
    imported = _deny_external_imports(monkeypatch)
    report = logging_utils.environment_report(tmp_path, gpu=True)
    for name in ("HF_TOKEN", "NASH_API_KEY", "NASH_SECRET", "NASH_PASSWORD"):
        assert report["environment"][name] == "<redacted>"
    assert report["environment"]["NASH_PUBLIC_SETTING"] == "visible"
    assert report["visible_gpu_ids"] == "2,4"
    assert report["nvidia_smi"]["command"] == ["nvidia-smi"]
    assert imported == []
