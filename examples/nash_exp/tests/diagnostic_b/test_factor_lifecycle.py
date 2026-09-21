# Copyright 2026 Nash Credit Routing contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded factor storage preserves geometry and durable solver replay caches."""

import shutil
from contextlib import contextmanager

import numpy as np
import pytest
from diagnostic_b import config as configuration
from diagnostic_b import pipeline, storage
from diagnostic_b.features import DeltaProxyExtractor, HeadAtom
from test_features import _toy_extractor
from test_infrastructure import EXP_ROOT
from test_pipeline_contract import _mock_model_boundary, _prepare_synthetic_inputs, _synthetic_config


def _group_config():
    return {"features": {"backend": "delta_proxy", "factor_cache_retention": "group"}}


@pytest.mark.parametrize("fail", [False, True])
def test_factor_workspace_cleans_only_owned_scratch(tmp_path, fail):
    persistent = tmp_path / "features" / "factors" / "existing.npy"
    persistent.parent.mkdir(parents=True)
    persistent.write_bytes(b"previous retained factors")
    unrelated = tmp_path / "features" / "factor_scratch" / "another-run" / "marker"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"not owned by this context")
    observed = []

    def use_workspace():
        with pipeline.factor_workspace(_group_config(), tmp_path) as factor_root:
            observed.append(factor_root)
            assert factor_root != tmp_path
            assert factor_root.is_relative_to(tmp_path / "features" / "factor_scratch")
            (factor_root / "temporary.npy").write_bytes(b"disposable")
            if fail:
                raise RuntimeError("interrupted extraction")

    if fail:
        with pytest.raises(RuntimeError, match="interrupted extraction"):
            use_workspace()
    else:
        use_workspace()
    assert len(observed) == 1 and not observed[0].exists()
    assert persistent.read_bytes() == b"previous retained factors"
    assert unrelated.read_bytes() == b"not owned by this context"


@pytest.mark.parametrize("retention", [None, "all"])
def test_retained_factor_workspace_preserves_legacy_cache(tmp_path, retention):
    config = {"features": {"backend": "delta_proxy"}}
    if retention is not None:
        config["features"]["factor_cache_retention"] = retention
    with pipeline.factor_workspace(config, tmp_path) as factor_root:
        assert factor_root == tmp_path
        marker = factor_root / "retained.npy"
        marker.write_bytes(b"kept")
    assert marker.read_bytes() == b"kept"


def test_streamed_heldout_matches_retained_fp32_accumulation(tmp_path):
    extractor, factors, *_ = _toy_extractor(tmp_path, extractor_class=DeltaProxyExtractor)
    atoms = [HeadAtom(factors)] * 3
    # Independent reference for the original per-response FP32 accumulation;
    # the retained API may delegate to the streaming implementation.
    expected = np.zeros(extractor.feature_shape, dtype=np.float32)
    for atom in (atoms[0], atoms[2]):
        expected += extractor.aggregate_proxy(atom) / np.float32(7)
    retained = extractor.build_heldout_head(atoms, [1, 0, 1], 7, tmp_path / "retained.npy")
    streamed = extractor.build_heldout_head_stream(
        iter([atoms[0], atoms[2]]), 7, tmp_path / "streamed.npy", expected_successes=2
    )
    assert np.load(streamed).dtype == np.float32
    np.testing.assert_array_equal(np.load(streamed), expected)
    np.testing.assert_array_equal(np.load(streamed), np.load(retained))
    np.testing.assert_allclose(np.load(streamed), extractor.aggregate_proxy(atoms[0]) * (2 / 7), atol=1e-7)


@pytest.mark.parametrize("failure", ["aggregate", "weighted", "count"])
def test_stream_failure_closes_scratch_generator_and_preserves_output(tmp_path, monkeypatch, failure):
    extractor, factors, *_ = _toy_extractor(tmp_path, extractor_class=DeltaProxyExtractor)
    destination = tmp_path / "heldout.npy"
    np.save(destination, np.array([8.0, 9.0], dtype=np.float32))
    previous_hash = storage.file_hash(destination)
    workspaces = []

    def successful_atoms():
        with pipeline.factor_workspace(_group_config(), tmp_path) as factor_root:
            workspaces.append(factor_root)
            copied = factor_root / "response"
            shutil.copytree(factors, copied)
            weights = [1 / 3] * 3 if failure == "weighted" else None
            yield HeadAtom(copied, weights)

    if failure == "aggregate":

        def fail_aggregate(atom):
            assert atom.factor_path.exists()
            raise RuntimeError("injected aggregation failure")

        monkeypatch.setattr(extractor, "aggregate_proxy", fail_aggregate)
    expected = 2 if failure == "count" else 1
    with pytest.raises((RuntimeError, ValueError)):
        extractor.build_heldout_head_stream(successful_atoms(), 4, destination, expected_successes=expected)
    assert workspaces and all(not path.exists() for path in workspaces)
    assert factors.exists()
    assert storage.file_hash(destination) == previous_hash


def test_heldout_pipeline_keeps_only_one_successful_response_live(tmp_path, monkeypatch):
    extractor, factors, *_ = _toy_extractor(tmp_path, extractor_class=DeltaProxyExtractor)
    config = _group_config() | {"sampling": {"temperature": 1.0}}
    storage.atomic_json(tmp_path / "model_lock.json", {"revision": "pinned"})
    storage.atomic_json(tmp_path / "verification_summary.json", {"heldout": {"successes": 2, "draws": 4}})
    rows = [
        {"split": "heldout", "prompt_index": 0, "rollout_index": i, "response_token_ids": [1, 3, 0]} for i in range(4)
    ]
    storage.write_compressed_records(tmp_path / "heldout_rollouts.jsonl.zst", rows)
    storage.write_records(
        tmp_path / "verifier_results.parquet", [row | {"reward": int(i % 2 == 0)} for i, row in enumerate(rows)]
    )
    observed = []

    def extract_response(extractor, row, factor_root, identity):
        assert all(not directory.exists() for directory in observed)
        assert factor_root != tmp_path
        destination = factor_root / "response"
        shutil.copytree(factors, destination)
        observed.append(destination)
        return destination

    monkeypatch.setattr(pipeline, "feature_extractor", lambda *args: extractor)
    monkeypatch.setattr(pipeline, "extract_response", extract_response)
    expected = extractor.aggregate_proxy(HeadAtom(factors)) / 2
    pipeline.compute_heldout_features(config, tmp_path)
    assert len(observed) == 2 and all(not path.exists() for path in observed)
    assert factors.exists()
    np.testing.assert_array_equal(np.load(tmp_path / "features" / "heldout_head.npy"), expected)
    manifest = storage.read_json(tmp_path / "heldout_feature_manifest.json")
    assert manifest["denominator"] == 4
    assert manifest["response_aggregation"] == "token_sum"


def test_training_scratch_removed_after_durable_grams_and_solver_replay_needs_no_factors(tmp_path, monkeypatch):
    config = _synthetic_config()
    config["features"].update(_group_config()["features"])
    config["routing"]["solver"]["max_iterations"] = 500
    _prepare_synthetic_inputs(tmp_path, [[1, 1, 0], [1, 0, 1]])
    calls = _mock_model_boundary(monkeypatch, np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.3]]))
    original_atoms = pipeline.make_group_atoms
    original_workspace = pipeline.factor_workspace
    scratch = []
    durable_groups = []

    def make_atoms(rows, rewards, extractor, tokenizer, factor_root, *args):
        assert all(not path.exists() for path in scratch)
        marker = factor_root / "response.npy"
        marker.write_bytes(b"group factors")
        scratch.append(marker)
        return original_atoms(rows, rewards, extractor, tokenizer, factor_root, *args)

    @contextmanager
    def observe_workspace(*args):
        with original_workspace(*args) as factor_root:
            yield factor_root
            groups = sorted((tmp_path / "features").glob("train_group_*/gram.npz"))
            assert len(groups) == len(durable_groups) + 1
            for path in groups:
                assert path.with_name("metadata.json").is_file()
            durable_groups[:] = groups

    monkeypatch.setattr(pipeline, "make_group_atoms", make_atoms)
    monkeypatch.setattr(pipeline, "factor_workspace", observe_workspace)
    pipeline.compute_training_features_and_routes(config, tmp_path)
    assert len(scratch) == 2 and all(not path.exists() for path in scratch)
    before = {path: (storage.file_hash(path), path.stat().st_mtime_ns) for path in durable_groups}
    assert calls == {"build_gram": 2, "tokenizer": 1}

    def forbidden_extraction(*args):
        pytest.fail("solver-only replay extracted factors")

    monkeypatch.setattr(pipeline, "make_group_atoms", forbidden_extraction)
    config["routing"]["solver"]["max_iterations"] = 501
    pipeline.compute_training_features_and_routes(config, tmp_path)
    assert calls == {"build_gram": 2, "tokenizer": 1}
    assert before == {path: (storage.file_hash(path), path.stat().st_mtime_ns) for path in durable_groups}


def test_training_extraction_failure_removes_scratch(tmp_path, monkeypatch):
    config = _synthetic_config()
    config["features"].update(_group_config()["features"])
    _prepare_synthetic_inputs(tmp_path, [[1, 1, 0]])
    _mock_model_boundary(monkeypatch, np.eye(3))
    scratch = []

    def failed_atoms(rows, rewards, extractor, tokenizer, factor_root, *args):
        marker = factor_root / "unfinished.npy"
        marker.write_bytes(b"incomplete factors")
        scratch.append(marker)
        raise RuntimeError("injected extraction failure")

    monkeypatch.setattr(pipeline, "make_group_atoms", failed_atoms)
    with pytest.raises(RuntimeError, match="injected extraction failure"):
        pipeline.compute_training_features_and_routes(config, tmp_path)
    assert scratch and all(not path.exists() for path in scratch)
    assert not (tmp_path / "features" / "train_group_00000" / "metadata.json").exists()


@pytest.mark.parametrize("backend, retention", [("exact_tiled_head", "group"), ("delta_proxy", "unknown")])
def test_configuration_rejects_unsupported_factor_retention(monkeypatch, backend, retention):
    for variable in configuration.ENV_OVERRIDES:
        monkeypatch.delenv(variable, raising=False)
    settings = configuration.load_config(EXP_ROOT / "configs" / "diagnostic_b_round1.yaml")
    settings["features"].update(backend=backend, factor_cache_retention=retention)
    with pytest.raises(configuration.ConfigurationError, match="factor_cache_retention"):
        configuration.validate_config(settings)


def test_environment_can_restore_all_factor_retention(monkeypatch):
    for variable in configuration.ENV_OVERRIDES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NASH_FACTOR_CACHE_RETENTION", "all")
    settings = configuration.load_config(EXP_ROOT / "configs" / "diagnostic_b_round1.yaml")
    assert settings["features"]["factor_cache_retention"] == "all"
