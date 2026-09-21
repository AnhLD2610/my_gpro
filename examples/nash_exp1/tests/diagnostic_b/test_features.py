# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""CPU audits of exact output-path score features and held-out normalization."""

import json
import shutil
from contextlib import nullcontext

import numpy as np
import pytest
from diagnostic_b.features import (
    DeltaProxyExtractor,
    FeatureExtractor,
    HeadAtom,
    dense_head_aggregate,
    factorized_inner,
    heldout_estimator,
)
from diagnostic_b.storage import file_hash


@pytest.mark.parametrize("temperature", [1.0, 0.65, 1.7])
def test_head_feature_equals_autograd(temperature):
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[0.2, 1.3], [0.7, -0.1], [-0.2, 0.8]], dtype=torch.float64)
    weight = torch.tensor([[0.4, -0.2], [-0.3, 0.6], [0.9, 0.2]], dtype=torch.float64, requires_grad=True)
    token_ids = torch.tensor([2, 0, 1])
    token_weights = torch.tensor([0.5, 0.2, -0.4], dtype=torch.float64)
    logits = hidden @ weight.T
    loss = (torch.log_softmax(logits / temperature, dim=-1)[torch.arange(3), token_ids] * token_weights).sum()
    loss.backward()
    expected = dense_head_aggregate(
        hidden.numpy(), logits.detach().numpy(), token_ids.numpy(), token_weights.numpy(), temperature
    )
    np.testing.assert_allclose(expected, weight.grad.numpy(), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_factorized_inner_matches_dense(temperature):
    rng = np.random.default_rng(7)
    ha, hb = rng.normal(size=(4, 3)), rng.normal(size=(5, 3))
    w = rng.normal(size=(7, 3))
    ya, yb = np.array([0, 4, 2, 6]), np.array([2, 6, 0, 1, 5])
    wa, wb = np.array([1, 0, -0.4, 0.3]), np.arange(5) / 5
    a = dense_head_aggregate(ha, ha @ w.T, ya, wa, temperature)
    b = dense_head_aggregate(hb, hb @ w.T, yb, wb, temperature)
    actual = factorized_inner(ha, ha @ w.T, ya, hb, hb @ w.T, yb, wa, wb, temperature, token_chunk_size=2)
    np.testing.assert_allclose(actual, np.sum(a * b), atol=1e-12)


def test_output_head_feature_has_all_output_rows():
    result = dense_head_aggregate([[2.0, 3.0]], [[0.0, 0.0, 0.0]], [1])
    np.testing.assert_allclose(result, [[-2 / 3, -1], [4 / 3, 2], [-2 / 3, -1]])
    assert np.count_nonzero(np.linalg.norm(result, axis=1)) == 3


def test_heldout_uses_token_sum_and_all_draws():
    scores = np.array([[[2, 4], [1, 3]], [[100, 100], [100, 100]], [[6, 2], [3, 1]]])
    expected = np.array([[2, 1.5], [1, 1]])
    np.testing.assert_allclose(heldout_estimator(scores, [1, 0, 1], total_draws=4), expected)
    np.testing.assert_allclose(heldout_estimator(scores[[0, 2]], [1, 1], total_draws=4), expected)
    np.testing.assert_array_equal(heldout_estimator(scores, [0, 0, 0], total_draws=3), np.zeros((2, 2)))


def test_heldout_rejects_success_count_as_denominator_for_all_draws():
    with pytest.raises(ValueError, match="every held-out draw"):
        heldout_estimator(np.zeros((3, 2)), [1, 0, 1], total_draws=2)


def test_extractor_constructor_is_lazy_and_requires_revision():
    extractor = FeatureExtractor("not-a-real-model", "pinned-commit", device="cpu")
    assert extractor.model is None
    assert extractor._torch is None
    with pytest.raises(ValueError, match="pinned"):
        FeatureExtractor("not-a-real-model", "main")


def _toy_extractor(tmp_path, temperature=1.0, extractor_class=FeatureExtractor):
    """Inject a tiny local head; never use Transformers or any model loader."""
    torch = pytest.importorskip("torch")
    weight = torch.tensor([[0.3, 0.2], [-0.2, 0.6], [0.5, -0.3], [-0.4, -0.1]])
    head = torch.nn.Linear(2, 4, bias=False)
    head.weight.data.copy_(weight)
    head.requires_grad_(False)

    class ToyModel:
        def get_output_embeddings(self):
            return head

    extractor = extractor_class(
        "toy",
        "toy-pinned",
        device="cpu",
        temperature=temperature,
        token_chunk_size=2,
        vocab_chunk_size=2,
        logit_vocab_chunk_size=3,
    )
    extractor.model = ToyModel()
    extractor._torch = torch
    hidden = np.array([[0.2, 0.7], [-0.3, 1.1], [0.4, -0.2]], dtype=np.float32)
    token_ids = np.array([1, 3, 0])  # The final original ID could be an EOS action.
    log_z = extractor._log_normalizers(hidden)
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    np.save(factor_dir / "hidden.npy", hidden)
    np.save(factor_dir / "token_ids.npy", token_ids)
    np.save(factor_dir / "log_normalizers.npy", log_z)
    with torch.no_grad():
        np.save(factor_dir / "token_log_probs.npy", extractor._selected_log_probs(hidden, token_ids, log_z))
    (factor_dir / "metadata.json").write_text(
        json.dumps(
            extractor._identity()
            | {
                "token_count": 3,
                "head_shape": [4, 2],
                "array_sha256": {
                    name: file_hash(factor_dir / name)
                    for name in ("hidden.npy", "token_ids.npy", "log_normalizers.npy", "token_log_probs.npy")
                },
            }
        )
    )
    return extractor, factor_dir, hidden, weight.numpy(), token_ids


@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_production_tiled_rows_gram_and_cross_match_dense(tmp_path, temperature):
    extractor, factor_dir, hidden, weight, token_ids = _toy_extractor(tmp_path, temperature)
    weights = np.array([0.5, 0, 0.5], dtype=np.float32)
    atom = HeadAtom(factor_dir, weights)
    blocks = [extractor.aggregate_row_block(atom, 0, 2), extractor.aggregate_row_block(atom, 2, 4)]
    dense = dense_head_aggregate(hidden, hidden @ weight.T, token_ids, weights, temperature)
    np.testing.assert_allclose(np.concatenate(blocks), dense, atol=2e-7, rtol=2e-6)
    out_path = extractor.build_heldout_head([HeadAtom(factor_dir)], [1], 4, tmp_path / "heldout.npy")
    heldout = dense_head_aggregate(hidden, hidden @ weight.T, token_ids, temperature=temperature) / 4
    np.testing.assert_allclose(np.load(out_path), heldout, atol=2e-7, rtol=2e-6)
    gram = extractor.build_gram([atom, HeadAtom(factor_dir)], out_path)
    second = heldout * 4
    explicit = np.stack([dense.ravel(), second.ravel()])
    np.testing.assert_allclose(gram["gram"], explicit @ explicit.T, atol=5e-7, rtol=2e-6)
    np.testing.assert_allclose(gram["heldout_cross"], explicit @ heldout.ravel(), atol=5e-7, rtol=2e-6)
    assert gram["gram"].dtype == np.float32
    assert gram["min_eigenvalue"] >= -1e-6


def test_heldout_backend_rejects_response_mean_weights(tmp_path):
    extractor, factor_dir, *_ = _toy_extractor(tmp_path)
    with pytest.raises(ValueError, match="token sums"):
        extractor.build_heldout_head([HeadAtom(factor_dir, np.ones(3) / 3)], [1], 32, tmp_path / "head.npy")
    with pytest.raises(ValueError, match="INSUFFICIENT_HELDOUT_SUCCESSES"):
        extractor.build_heldout_head([HeadAtom(factor_dir)], [0], 32, tmp_path / "head.npy")


def test_factor_identity_mismatch_is_rejected(tmp_path):
    extractor, factor_dir, *_ = _toy_extractor(tmp_path)
    extractor.temperature = 0.5
    with pytest.raises(ValueError, match="FACTOR_CACHE_MISMATCH"):
        extractor.aggregate_row_block(HeadAtom(factor_dir), 0, 2)


@pytest.mark.parametrize("extractor_class", [FeatureExtractor, DeltaProxyExtractor])
def test_extraction_preserves_original_context_and_sampled_last_token(tmp_path, extractor_class):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    extractor, _, _, _, _ = _toy_extractor(tmp_path, extractor_class=extractor_class)
    observed = []

    def backbone(**kwargs):
        observed.append(kwargs["input_ids"].tolist())
        # Deliberately context-dependent toy hidden states.
        sums = kwargs["input_ids"].float().cumsum(dim=-1)
        return SimpleNamespace(last_hidden_state=torch.stack((sums, sums + 1), dim=-1))

    extractor.model.model = backbone
    factors = extractor.extract([2, 1], [3, 0, 1], tmp_path / "extracted", cache_key="rollout-key")
    assert observed == [[[2, 1, 3, 0]]]
    np.testing.assert_array_equal(np.load(factors / "hidden.npy"), [[3, 4], [6, 7], [6, 7]])
    np.testing.assert_array_equal(np.load(factors / "token_ids.npy"), [3, 0, 1])
    assert np.load(factors / "token_log_probs.npy").shape == (3,)
    assert extractor.extract([2, 1], [3, 0, 1], factors, cache_key="rollout-key") == factors
    assert len(observed) == 1
    with pytest.raises(ValueError, match="FACTOR_CACHE_MISMATCH"):
        extractor.extract([2, 1], [3, 0, 2], factors, cache_key="rollout-key")


def test_cuda_attention_context_disables_quadratic_math_backend(monkeypatch):
    torch = pytest.importorskip("torch")
    selected = []
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", lambda **kwargs: (selected.append(kwargs) or nullcontext()))
    extractor = FeatureExtractor("toy", "pinned", device="cuda:0")
    with extractor._attention_context():
        pass
    assert selected[0]["backends"] == [
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    ]
    assert torch.nn.attention.SDPBackend.MATH not in selected[0]["backends"]
    assert extractor.model is None  # No CUDA work/model loading was necessary for this policy test.


def test_factor_cache_limits_open_maps_and_rejects_corrupted_content(tmp_path):
    extractor, factor_dir, *_ = _toy_extractor(tmp_path)
    extractor.max_open_factors = 2
    for index in range(5):
        destination = tmp_path / f"copy{index}"
        shutil.copytree(factor_dir, destination)
        hidden, *_ = extractor._read_factors(destination)
        assert isinstance(hidden, np.memmap)
        assert len(extractor._factors) <= 2
    changed = np.load(destination / "hidden.npy")
    changed[0, 0] += 0.1
    np.save(destination / "hidden.npy", changed)
    with pytest.raises(ValueError, match="FACTOR_CACHE_CORRUPT"):
        extractor._read_factors(destination)


def test_normalization_and_single_aggregate_tiles_are_independent_of_gram_tiles(tmp_path, monkeypatch):
    extractor, factor_dir, hidden, weight, _ = _toy_extractor(tmp_path)
    extractor.vocab_chunk_size = 1
    # Changing a numerical tile setting invalidates factor reuse: reflect this
    # newly declared toy setting in its freshly prepared factor metadata.
    metadata = json.loads((factor_dir / "metadata.json").read_text())
    metadata.update(extractor._identity())
    (factor_dir / "metadata.json").write_text(json.dumps(metadata))
    expected = np.log(np.exp(hidden @ weight.T).sum(axis=1))
    np.testing.assert_allclose(extractor._log_normalizers(hidden), expected, atol=2e-7)
    blocks = []
    original = extractor.aggregate_row_block

    def record_blocks(atom, start, stop):
        blocks.append((start, stop))
        return original(atom, start, stop)

    monkeypatch.setattr(extractor, "aggregate_row_block", record_blocks)
    extractor.build_heldout_head([HeadAtom(factor_dir)], [1], 32, tmp_path / "head.npy")
    assert blocks == [(0, 3), (3, 4)]
    blocks.clear()
    extractor.build_gram([HeadAtom(factor_dir)])
    assert blocks == [(0, 1), (1, 2), (2, 3), (3, 4)]
