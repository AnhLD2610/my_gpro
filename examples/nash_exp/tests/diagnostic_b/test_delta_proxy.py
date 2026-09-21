# Copyright 2026 Nash Credit Routing contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit DelTA Appendix F against sampled-row autograd and production caches."""

import json

import numpy as np
import pytest
from diagnostic_b import pipeline, storage
from diagnostic_b.features import (
    DeltaProxyExtractor,
    FeatureExtractor,
    HeadAtom,
    delta_proxy_aggregate,
    delta_proxy_tokens,
)
from test_features import _toy_extractor


def test_delta_proxy_equals_each_sampled_row_autograd():
    import torch

    hidden = torch.tensor([[0.2, 1.3], [0.7, -0.1], [-0.2, 0.8]], dtype=torch.float64)
    weight = torch.tensor([[0.4, -0.2], [-0.3, 0.6], [0.9, 0.2]], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([2, 0, 1])
    log_probs = torch.log_softmax(hidden @ weight.T, dim=-1)[torch.arange(3), ids]
    expected = np.stack(
        [torch.autograd.grad(log_probs[t], weight, retain_graph=True)[0][ids[t]].numpy() for t in range(3)]
    )
    actual = delta_proxy_tokens(hidden.numpy(), log_probs.detach().numpy())
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    weights = np.array([0.5, 0, -0.4])
    np.testing.assert_allclose(
        delta_proxy_aggregate(hidden.numpy(), (hidden @ weight.T).detach().numpy(), ids.numpy(), weights),
        weights @ expected,
        atol=1e-12,
    )


def test_different_sampled_rows_share_hidden_coordinates_as_in_delta():
    # Two equally likely but different actions have the same DelTA vector.
    # Full-head vectors for these actions have negative, not positive, inner product.
    h, logits = [[2.0, 3.0]], [[0.0, 0.0]]
    a = delta_proxy_aggregate(h, logits, [0])
    b = delta_proxy_aggregate(h, logits, [1])
    np.testing.assert_array_equal(a, [1.0, 1.5])
    np.testing.assert_array_equal(a, b)
    assert a @ b > 0


@pytest.mark.parametrize("weights", [None, [1 / 3] * 3, [0.5, 0.5, 0.0]])
def test_production_delta_aggregate_gram_and_heldout_match_reference(tmp_path, monkeypatch, weights):
    extractor, path, hidden, weight, ids = _toy_extractor(tmp_path, extractor_class=DeltaProxyExtractor)
    # Once factors exist, DelTA needs neither full-head aggregation nor logits again.
    monkeypatch.setattr(extractor, "aggregate_row_block", lambda *args: pytest.fail("full-head aggregation used"))
    monkeypatch.setattr(
        extractor, "_normalization_factors", lambda *args: pytest.fail("recomputed cached probabilities")
    )
    atom = HeadAtom(path, weights)
    expected = delta_proxy_aggregate(hidden, hidden @ weight.T, ids, weights)
    summed = delta_proxy_aggregate(hidden, hidden @ weight.T, ids)
    np.testing.assert_allclose(extractor.aggregate_proxy(atom), expected, atol=2e-7, rtol=2e-6)
    heldout_path = extractor.build_heldout_head([HeadAtom(path), HeadAtom(path)], [1, 0], 8, tmp_path / "heldout.npy")
    assert np.load(heldout_path).shape == (2,)
    np.testing.assert_allclose(np.load(heldout_path), summed / 8, atol=2e-7, rtol=2e-6)
    result = extractor.build_gram([atom, HeadAtom(path)], heldout_path)
    vectors = np.stack([expected, summed])
    np.testing.assert_allclose(result["gram"], vectors @ vectors.T, atol=5e-7, rtol=2e-6)
    np.testing.assert_allclose(result["heldout_cross"], vectors @ (summed / 8), atol=5e-7, rtol=2e-6)
    assert result["gram"].dtype == np.float32
    assert extractor.feature_shape == (2,)
    assert extractor.head_shape == (4, 2)


def test_delta_rejects_full_head_factor_cache(tmp_path):
    exact, path, *_ = _toy_extractor(tmp_path)
    proxy = DeltaProxyExtractor(
        "toy", "toy-pinned", device="cpu", token_chunk_size=2, vocab_chunk_size=2, logit_vocab_chunk_size=3
    )
    proxy.model, proxy._torch = exact.model, exact._torch
    with pytest.raises(ValueError, match="FACTOR_CACHE_MISMATCH"):
        proxy.aggregate_proxy(HeadAtom(path))


def test_delta_rejects_corrupt_selected_token_probabilities(tmp_path):
    extractor, path, *_ = _toy_extractor(tmp_path, extractor_class=DeltaProxyExtractor)
    values = np.load(path / "token_log_probs.npy")
    values[0] -= 0.1
    np.save(path / "token_log_probs.npy", values)
    with pytest.raises(ValueError, match="FACTOR_CACHE_CORRUPT"):
        extractor.aggregate_proxy(HeadAtom(path))


def test_delta_heldout_guards_normalization_and_geometry(tmp_path):
    extractor, path, *_ = _toy_extractor(tmp_path, extractor_class=DeltaProxyExtractor)
    with pytest.raises(ValueError, match="token sums"):
        extractor.build_heldout_head([HeadAtom(path, [1 / 3] * 3)], [1], 32, tmp_path / "bad.npy")
    with pytest.raises(ValueError, match="INSUFFICIENT_HELDOUT_SUCCESSES"):
        extractor.build_heldout_head([HeadAtom(path)], [0], 32, tmp_path / "bad.npy")
    with pytest.raises(ValueError, match="all held-out"):
        extractor.build_heldout_head([HeadAtom(path)] * 2, [1, 1], 1, tmp_path / "bad.npy")
    np.save(tmp_path / "full_head.npy", np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="geometry"):
        extractor.build_gram([HeadAtom(path)], tmp_path / "full_head.npy")


def test_delta_unit_temperature_and_probability_guards():
    with pytest.raises(ValueError, match="temperature=1.0"):
        DeltaProxyExtractor("toy", "pinned", temperature=0.7)
    for invalid in ([np.nan], [np.inf], [0.1]):
        with pytest.raises(ValueError):
            delta_proxy_tokens([[1.0, 2.0]], invalid)
    np.testing.assert_array_equal(delta_proxy_tokens([[1.0, 2.0]], [0.0]), [[0.0, 0.0]])
    np.testing.assert_array_equal(delta_proxy_tokens([[1.0, 2.0]], [1e-6]), [[0.0, 0.0]])
    np.testing.assert_allclose(delta_proxy_tokens([[1.0, 2.0]], [-1000.0]), [[1.0, 2.0]])


@pytest.mark.parametrize("backend, cls", [("delta_proxy", DeltaProxyExtractor), ("exact_tiled_head", FeatureExtractor)])
def test_pipeline_selects_backend_without_loading_model(tmp_path, backend, cls):
    storage.atomic_json(tmp_path / "model_lock.json", {"revision": "pinned", "tokenizer_revision": "pinned"})
    config = {
        "model": {"id": "toy", "dtype": "float32"},
        "sampling": {"temperature": 1.0},
        "features": {"backend": backend, "token_chunk_size": 2, "vocab_chunk_size": 2},
    }
    extractor = pipeline.feature_extractor(config, tmp_path)
    assert type(extractor) is cls
    assert extractor.model is None
    assert extractor._torch is None


def test_heldout_pipeline_records_delta_vector_shape_and_all_draw_denominator(tmp_path, monkeypatch):
    extractor, factor_dir, hidden, weight, ids = _toy_extractor(tmp_path, extractor_class=DeltaProxyExtractor)
    config = {"features": {"backend": "delta_proxy"}, "sampling": {"temperature": 1.0}}
    storage.atomic_json(tmp_path / "model_lock.json", {"revision": "pinned"})
    storage.atomic_json(tmp_path / "verification_summary.json", {"heldout": {"successes": 1, "draws": 4}})
    rows = [
        {"split": "heldout", "prompt_index": 0, "rollout_index": i, "response_token_ids": ids.tolist()}
        for i in range(4)
    ]
    storage.write_compressed_records(tmp_path / "heldout_rollouts.jsonl.zst", rows)
    storage.write_records(
        tmp_path / "verifier_results.parquet", [row | {"reward": int(i == 0)} for i, row in enumerate(rows)]
    )
    monkeypatch.setattr(pipeline, "feature_extractor", lambda *args: extractor)
    monkeypatch.setattr(pipeline, "extract_response", lambda *args: factor_dir)
    pipeline.compute_heldout_features(config, tmp_path)
    manifest = json.loads((tmp_path / "heldout_feature_manifest.json").read_text())
    assert manifest["geometry"] == "delta_selected_token_gradient_proxy"
    assert manifest["feature_shape"] == [2]
    assert manifest["denominator"] == 4
    expected = delta_proxy_aggregate(hidden, hidden @ weight.T, ids) / 4
    np.testing.assert_allclose(np.load(tmp_path / "features" / "heldout_head.npy"), expected, atol=2e-7)
