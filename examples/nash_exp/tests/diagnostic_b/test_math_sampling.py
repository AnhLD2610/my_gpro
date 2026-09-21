# Copyright 2026 Nash Credit Routing contributors
# SPDX-License-Identifier: Apache-2.0
"""MATH gold extraction and reproducible train/test subsampling before generation."""

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from diagnostic_b import data, storage
from diagnostic_b.config import ConfigurationError

PINNED_REVISION = "c" * 40


class PromptOnlyTokenizer:
    chat_template = "synthetic prompt-only template"

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        assert model_path == "synthetic/model"
        assert kwargs == {"revision": PINNED_REVISION, "trust_remote_code": False}
        return cls()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert add_generation_prompt
        assert not kwargs
        text = "\n".join(message["content"] for message in messages) + "\nAssistant:"
        return list(text.encode()) if tokenize else text


@pytest.fixture
def math_preparation(tmp_path, monkeypatch):
    wrapper_path = tmp_path / "wrapper.json"
    storage.atomic_json(wrapper_path, {"user_template": "Solve: {problem}"})
    config = {
        "round_id": "round1",
        "seed": 42,
        "model": {"id": "synthetic/model", "max_model_len": 4096},
        "prompt": {"wrapper_file": str(wrapper_path)},
        "sampling": {"max_new_tokens": 100},
        "train": {
            "id": "ShuoZheLi/MATH-train-MATH500-test",
            "revision": PINNED_REVISION,
            "split": "train",
            "adapter": "math",
            "selection": "sample",
            "expected_rows": 7500,
            "n_problem": 128,
        },
        "heldout": {
            "id": "ShuoZheLi/MATH-train-MATH500-test",
            "revision": PINNED_REVISION,
            "split": "test",
            "adapter": "math",
            "selection": "sample",
            "expected_rows": 500,
            "n_problem": 100,
        },
    }
    datasets = {
        split: [
            {
                "problem": f"Problem from {split} number {index}",
                "solution": rf"SECRET_GOLD_{split}_{index}: reasoning then \boxed{{{index + 10000}}}",
            }
            for index in range(count)
        ]
        for split, count in (("train", 7500), ("test", 500))
    }

    def load_snapshot(spec, *, require_immutable=False):
        assert spec["revision"] == PINNED_REVISION
        if spec["split"] == "train":
            assert require_immutable
        rows = datasets[spec["split"]]
        return rows, {
            "id": spec["id"],
            "revision": spec["revision"],
            "split": spec["split"],
            "rows": len(rows),
        }

    monkeypatch.setattr(data, "load_dataset_snapshot", load_snapshot)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=PromptOnlyTokenizer))

    def prepare(name="prepared", overrides=None):
        current = copy.deepcopy(config)
        for key, value in (overrides or {}).items():
            if isinstance(value, dict):
                current[key].update(value)
            else:
                current[key] = value
        output = tmp_path / name
        artifacts = data.prepare_manifests(current, output, {"tokenizer_revision": PINNED_REVISION})
        assert set(artifacts) == {
            output / "manifest.json",
            output / "train_prompt_manifest.parquet",
            output / "heldout_prompt_manifest.parquet",
        }
        return (
            storage.read_json(output / "manifest.json"),
            storage.read_records(output / "train_prompt_manifest.parquet"),
            storage.read_records(output / "heldout_prompt_manifest.parquet"),
        )

    return prepare, config, datasets


def test_math_gold_uses_final_box_preserving_nested_latex_and_only_problem_in_prompt():
    solution = r"PRIVATE_SOLUTION: first \boxed{wrong}, finally \boxed{\frac{1}{2}}."
    row = data.adapt_row({"problem": "Find the rational value.", "solution": solution}, "math", 17)
    assert row["answer"] == r"\frac{1}{2}"
    assert row["dataset_row_index"] == 17
    assert row["row_id"] == "17"
    rendered = data.render_prompt(row, PromptOnlyTokenizer(), {"user_template": "Solve: {problem}"})
    assert rendered["prompt_text"] == "Solve: Find the rational value.\nAssistant:"
    assert "PRIVATE_SOLUTION" not in rendered["prompt_text"]
    assert row["answer"] not in rendered["prompt_text"]


@pytest.mark.parametrize(
    "row",
    [
        {"problem": "Question"},
        {"problem": "Question", "solution": 42},
        {"problem": "Question", "solution": "answer is 42"},
        {"problem": "Question", "solution": r"\boxed{42} then \boxed{unclosed"},
        {"problem": "Question", "solution": r"\boxed{42} then \boxed wrong"},
        {"problem": "Question", "solution": r"\boxed{}"},
        {"problem": "Question", "solution": r"\boxed{   }"},
        {"problem": "", "solution": r"\boxed{42}"},
        {"problem": 123, "solution": r"\boxed{42}"},
    ],
)
def test_math_adapter_rejects_invalid_gold_or_problem(row):
    with pytest.raises(ConfigurationError):
        data.adapt_row(row, "math", 9)


def test_math_prepare_selects_requested_unique_samples_with_locked_rng(math_preparation):
    prepare, config, _ = math_preparation
    manifest, train, heldout = prepare()
    expected_train = np.random.default_rng(config["seed"]).choice(7500, 128, replace=False).tolist()
    expected_heldout = (
        np.random.default_rng(np.random.SeedSequence([config["seed"], 1])).choice(500, 100, replace=False).tolist()
    )
    assert manifest["train_row_indices"] == expected_train
    assert manifest["heldout_row_indices"] == expected_heldout
    assert manifest["selection"] == {
        "train": {
            "mode": "sample",
            "population_rows": 7500,
            "sample_size": 128,
            "replace": False,
            "seed": 42,
            "rng": "numpy.default_rng(seed)",
        },
        "heldout": {
            "mode": "sample",
            "population_rows": 500,
            "sample_size": 100,
            "replace": False,
            "seed": 42,
            "rng": "numpy.default_rng(SeedSequence([seed, 1]))",
        },
    }
    for name, rows, indices, population in (
        ("train", train, expected_train, 7500),
        ("heldout", heldout, expected_heldout, 500),
    ):
        assert len(rows) == len(indices) == len({row["row_id"] for row in rows})
        assert [row["dataset_row_index"] for row in rows] == indices
        assert [row["prompt_index"] for row in rows] == list(range(len(indices)))
        assert all(row["split"] == name for row in rows)
        assert all(row["answer"] == str(row["dataset_row_index"] + 10000) for row in rows)
        assert all("SECRET_GOLD" not in row["prompt_text"] for row in rows)
        assert all(row["answer"] not in row["prompt_text"] for row in rows)
        assert manifest[name]["rows"] == population
        assert manifest[name]["revision"] == PINNED_REVISION
        assert manifest[name]["adapter"] == "math"
        assert manifest[name]["answer_extraction"] == "last_boxed_solution"
    repeated, repeated_train, repeated_heldout = prepare("repeated")
    assert repeated == manifest
    assert [row["prompt_hash"] for row in repeated_train] == [row["prompt_hash"] for row in train]
    assert [row["prompt_hash"] for row in repeated_heldout] == [row["prompt_hash"] for row in heldout]


def test_math_heldout_sample_is_independent_of_training_sample_size(math_preparation):
    prepare, _, _ = math_preparation
    first, _, _ = prepare("first")
    second, _, _ = prepare("second", {"train": {"n_problem": 64}})
    assert first["heldout_row_indices"] == second["heldout_row_indices"]
    assert first["heldout_prompts"] == second["heldout_prompts"]
    assert len(second["train_row_indices"]) == 64


def test_scaled_math_prepare_samples_500_train_and_preserves_all_500_test_rows(math_preparation):
    prepare, config, _ = math_preparation
    overrides = {
        "train": {"selection": "sample", "n_problem": 500},
        "heldout": {"selection": "all", "n_problem": 500},
    }
    manifest, train, heldout = prepare("scaled", overrides)
    expected_train = np.random.default_rng(config["seed"]).choice(7500, 500, replace=False).tolist()
    assert manifest["train_row_indices"] == expected_train
    assert len(set(expected_train)) == 500
    assert any(index >= 500 for index in expected_train)
    assert manifest["selection"]["train"]["mode"] == "sample"
    assert manifest["selection"]["train"]["replace"] is False
    assert [row["dataset_row_index"] for row in train] == expected_train
    assert manifest["heldout_row_indices"] == list(range(500))
    assert manifest["selection"]["heldout"] == {
        "mode": "all",
        "population_rows": 500,
        "sample_size": 500,
        "replace": False,
        "seed": None,
        "rng": None,
    }
    assert [row["dataset_row_index"] for row in heldout] == list(range(500))
    assert len(train) == len(heldout) == 500
    for rows in (train, heldout):
        assert len({row["row_id"] for row in rows}) == 500
        assert [row["prompt_index"] for row in rows] == list(range(500))
        assert all("SECRET_GOLD" not in row["prompt_text"] for row in rows)
    alternate, _, alternate_heldout = prepare("scaled_different_seed", overrides | {"seed": 43})
    assert alternate["train_row_indices"] != expected_train
    assert alternate["heldout_row_indices"] == manifest["heldout_row_indices"]
    assert [row["prompt_hash"] for row in alternate_heldout] == [row["prompt_hash"] for row in heldout]


@pytest.mark.parametrize("split,source", [("train", "train"), ("heldout", "test")])
def test_math_rejects_unexpected_source_population(math_preparation, split, source):
    prepare, _, datasets = math_preparation
    datasets[source].pop()
    with pytest.raises(ConfigurationError):
        prepare(overrides={split: {"selection": "sample"}})


@pytest.mark.parametrize("split,count", [("train", 7501), ("heldout", 501)])
def test_math_rejects_oversized_sample_instead_of_replacement(math_preparation, split, count):
    prepare, _, _ = math_preparation
    with pytest.raises(ConfigurationError):
        prepare(overrides={split: {"n_problem": count}})


def test_legacy_selection_keeps_training_sample_and_all_heldout(math_preparation):
    prepare, config, _ = math_preparation
    config["train"].pop("selection")
    config["heldout"].pop("selection")
    config["heldout"]["n_problem"] = 500
    manifest, train, heldout = prepare()
    assert (
        manifest["train_row_indices"] == np.random.default_rng(config["seed"]).choice(7500, 128, replace=False).tolist()
    )
    assert manifest["heldout_row_indices"] == list(range(500))
    assert manifest["selection"]["heldout"]["mode"] == "all"
    assert manifest["selection"]["heldout"]["seed"] is None
    assert manifest["selection"]["heldout"]["rng"] is None
    assert len(train) == 128
    assert len(heldout) == 500


def test_all_selection_requires_exact_heldout_count(math_preparation):
    prepare, _, _ = math_preparation
    with pytest.raises(ConfigurationError):
        prepare(overrides={"heldout": {"selection": "all", "n_problem": 100}})


@pytest.mark.parametrize("split,source", [("train", "train"), ("heldout", "test")])
@pytest.mark.parametrize("prompt_length,context", [(1024, 4096), (1025, 8192)])
def test_preparation_enforces_prompt_cap_separately_from_context(
    math_preparation, monkeypatch, split, source, prompt_length, context
):
    prepare, _, _ = math_preparation
    original_template = PromptOnlyTokenizer.apply_chat_template

    def sized_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = original_template(self, messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)
        if tokenize:
            count = prompt_length if f"Problem from {source} number" in text else 64
            return [7] * count
        return text

    monkeypatch.setattr(PromptOnlyTokenizer, "apply_chat_template", sized_template)
    overrides = {
        "prompt": {"max_tokens": 1024},
        "sampling": {"max_new_tokens": 3072},
        "model": {"max_model_len": context},
    }
    if prompt_length > 1024:
        assert prompt_length + 3072 < context
        with pytest.raises(ConfigurationError, match="prompt.max_tokens"):
            prepare(overrides=overrides)
    else:
        manifest, train, heldout = prepare(overrides=overrides)
        assert manifest["prompt_max_tokens"] == 1024
        rows = train if split == "train" else heldout
        assert all(row["prompt_token_count"] == 1024 for row in rows)
        assert all(list(row["prompt_token_ids"]) == [7] * 1024 for row in rows)
