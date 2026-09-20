# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""DelTA sampled-row proxy and optional full output-path LM-head features.

Importing this module neither imports Transformers nor loads a model. The small
NumPy functions are independent reference implementations. The production
extractor saves response hidden states and log normalizers, then contracts one
vocabulary block at a time. A token-level ``vocabulary x hidden`` gradient is
never constructed. DelTA aggregates (1 - p(sampled token)) * hidden into a
hidden-size vector. Neither representation is a full-policy score.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np

from .storage import file_hash

LOGGER = logging.getLogger(__name__)


def delta_proxy_tokens(hidden, token_log_probs):
    """DelTA Appendix F: sampled-row gradients in one shared hidden-size space.

    Unit-temperature definition, matching DelTA's dp_actor.py. The row identity
    is deliberately discarded, even when different tokens were sampled. This
    is not an inner-product-preserving projection of the full head gradient.
    """
    hidden = np.asarray(hidden)
    log_probs = np.asarray(token_log_probs)
    if hidden.ndim != 2 or log_probs.shape != (len(hidden),):
        raise ValueError("one sampled-token log probability is required per hidden state")
    if not np.isfinite(hidden).all() or not np.isfinite(log_probs).all():
        raise ValueError("DelTA factors must be finite")
    if np.any(log_probs > 1e-5):
        raise ValueError("sampled-token log probability cannot be positive")
    # expm1 preserves 1-p near p=1. Clip only <=1e-5 positive FP32 roundoff.
    return -np.expm1(np.minimum(log_probs, 0))[:, None] * hidden


def delta_proxy_aggregate(hidden, logits, token_ids, weights=None):
    """Float64 CPU reference for DelTA's weighted token proxy at temperature 1."""
    hidden, probabilities, token_ids, weights = _reference_inputs(hidden, logits, token_ids, weights, 1.0)
    return ((1 - probabilities[np.arange(len(token_ids)), token_ids]) * weights) @ hidden


def _reference_inputs(hidden, logits, token_ids, weights, temperature):
    hidden = np.asarray(hidden, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    token_ids = np.asarray(token_ids, dtype=np.int64)
    if hidden.ndim != 2 or logits.ndim != 2 or len(hidden) != len(logits):
        raise ValueError("hidden and logits must be matrices with equal token counts")
    if token_ids.shape != (len(hidden),):
        raise ValueError("one original token ID is required per hidden state")
    if np.any(token_ids < 0) or np.any(token_ids >= logits.shape[1]):
        raise ValueError("token ID outside output vocabulary")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(hidden).all() or not np.isfinite(logits).all():
        raise ValueError("hidden states and logits must be finite")
    weights = np.ones(len(hidden)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(hidden),) or not np.isfinite(weights).all():
        raise ValueError("token weights must be a finite vector of response length")
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return hidden, probabilities, token_ids, weights


def dense_head_aggregate(hidden, logits, token_ids, weights=None, temperature=1.0):
    """Float64 toy reference: sum weighted exact head score matrices.

    ``logits`` are unscaled logits. None weights give a sequence-token sum;
    response means require explicit ``ones(T) / T`` weights.
    """
    hidden, probabilities, token_ids, weights = _reference_inputs(hidden, logits, token_ids, weights, temperature)
    residual = -probabilities
    residual[np.arange(len(token_ids)), token_ids] += 1.0
    return (residual * (weights / temperature)[:, None]).T @ hidden


def factorized_inner(
    hidden_a,
    logits_a,
    token_ids_a,
    hidden_b,
    logits_b,
    token_ids_b,
    weights_a=None,
    weights_b=None,
    temperature=1.0,
    temperature_b=None,
    token_chunk_size=256,
):
    """Exact inner product from factors, without aggregated head matrices.

    Uses ``<a h^T, b k^T> = <a,b><h,k>`` and bounds the temporary token
    cross-product by ``token_chunk_size``. This CPU reference is practical for
    small audit examples; production uses vocabulary-tiled aggregation below.
    """
    if not isinstance(token_chunk_size, int) or token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be a positive integer")
    temperature_b = temperature if temperature_b is None else temperature_b
    ha, pa, ya, wa = _reference_inputs(hidden_a, logits_a, token_ids_a, weights_a, temperature)
    hb, pb, yb, wb = _reference_inputs(hidden_b, logits_b, token_ids_b, weights_b, temperature_b)
    if ha.shape[1] != hb.shape[1] or pa.shape[1] != pb.shape[1]:
        raise ValueError("feature geometries must match")
    result = 0.0
    for a in range(0, len(ha), token_chunk_size):
        sa = slice(a, a + token_chunk_size)
        for b in range(0, len(hb), token_chunk_size):
            sb = slice(b, b + token_chunk_size)
            residual_cross = pa[sa] @ pb[sb].T
            residual_cross -= pa[sa][:, yb[sb]]
            residual_cross -= pb[sb][:, ya[sa]].T
            residual_cross += ya[sa, None] == yb[None, sb]
            hidden_cross = ha[sa] @ hb[sb].T
            result += np.sum(residual_cross * hidden_cross * wa[sa, None] * wb[None, sb])
    return float(result / (temperature * temperature_b))


def heldout_estimator(sequence_score_sums, rewards, total_draws):
    """Reward times sequence-score SUM, divided by ALL held-out draws.

    Zero-reward draws may have omitted feature matrices to save storage, but
    ``total_draws`` must still count them. No response-length or success-count
    normalization is performed here.
    """
    scores = np.asarray(sequence_score_sums, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if scores.ndim < 2 or rewards.shape != (len(scores),):
        raise ValueError("scores and binary rewards must have the same draw count")
    if not isinstance(total_draws, int | np.integer) or total_draws <= 0 or total_draws < len(scores):
        raise ValueError("total_draws must count every held-out draw")
    if not np.isin(rewards, [0.0, 1.0]).all() or not np.isfinite(scores).all():
        raise ValueError("finite scores and binary rewards are required")
    return np.tensordot(rewards, scores, axes=(0, 0)) / total_draws


@dataclass(frozen=True)
class HeadAtom:
    """One weighted response aggregate in the extractor's declared geometry."""

    factor_path: str | Path
    token_weights: Sequence[float] | np.ndarray | None = None


class FeatureExtractor:
    """A frozen causal model and exact, FP32 vocabulary-tiled contractions.

    Only the approved model checkpoint is loaded, once per process. All
    experiment choices are arguments. Output-head bias is explicitly rejected:
    callers must approve an augmented geometry before using such a model.
    """

    geometry = "unprojected_output_path_lm_head_proxy"

    def __init__(
        self,
        model_path,
        revision,
        *,
        tokenizer_revision=None,
        dtype="bfloat16",
        device="cuda:0",
        temperature=1.0,
        token_chunk_size=128,
        vocab_chunk_size=256,
        logit_vocab_chunk_size=8192,
        local_files_only=False,
        max_open_factors=4,
    ):
        if not revision or str(revision) in {"main", "master", "latest"}:
            raise ValueError("a pinned model revision or local-snapshot identity is required")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if min(token_chunk_size, vocab_chunk_size, logit_vocab_chunk_size) <= 0:
            raise ValueError("chunk sizes must be positive")
        if not isinstance(max_open_factors, int) or max_open_factors <= 0:
            raise ValueError("max_open_factors must be a positive integer")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported model dtype")
        self.model_path = str(model_path)
        self.revision = str(revision)
        self.tokenizer_revision = tokenizer_revision or self.revision
        self.dtype = dtype
        self.device = device
        self.temperature = float(temperature)
        self.token_chunk_size = int(token_chunk_size)
        self.vocab_chunk_size = int(vocab_chunk_size)
        # Single-aggregate stages can use wider tiles than the many-atom Gram.
        self.logit_vocab_chunk_size = int(logit_vocab_chunk_size)
        self.local_files_only = local_files_only
        self.max_open_factors = max_open_factors
        self.model = None
        self._torch = None
        self._factors = OrderedDict()
        self._verified_factors = {}
        self._library_versions = None

    def load(self):
        """Load lazily; no import-time model, tokenizer, dataset, or GPU work."""
        if self.model is not None:
            return self
        import torch
        from transformers import AutoModelForCausalLM

        self._torch = torch
        # TF32 is deliberately disabled for FP32 feature/Gram contractions.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            revision=self.revision,
            torch_dtype=getattr(torch, self.dtype),
            local_files_only=self.local_files_only,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        head = model.get_output_embeddings()
        if head is None or not hasattr(head, "weight") or head.weight.ndim != 2:
            raise ValueError("backend requires a linear output-path LM head")
        if getattr(head, "bias", None) is not None:
            raise ValueError("OUTPUT_BIAS_UNSUPPORTED: approve augmented hidden-state geometry first")
        if not hasattr(model, "model"):
            raise ValueError("backend requires a causal model exposing model.model")
        model.requires_grad_(False)
        model.eval()
        model.to(self.device)
        self.model = model
        LOGGER.info(
            "Feature model loaded: model=%s revision=%s dtype=%s device=%s head=%s temperature=%s",
            self.model_path,
            self.revision,
            self.dtype,
            self.device,
            tuple(head.weight.shape),
            self.temperature,
        )
        return self

    @property
    def head_shape(self):
        self.load()
        return tuple(self.model.get_output_embeddings().weight.shape)

    @property
    def feature_shape(self):
        return self.head_shape

    def _identity(self):
        if self._library_versions is None:
            self._library_versions = {}
            for package in ("torch", "transformers", "numpy"):
                try:
                    self._library_versions[package] = version(package)
                except PackageNotFoundError:
                    self._library_versions[package] = "not-installed"
        return {
            "model_path": self.model_path,
            "model_revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype,
            "temperature": self.temperature,
            "geometry": self.geometry,
            "factor_dtype": "float32",
            "feature_schema_version": 2,
            "token_chunk_size": self.token_chunk_size,
            "vocab_chunk_size": self.vocab_chunk_size,
            "logit_vocab_chunk_size": self.logit_vocab_chunk_size,
            "libraries": self._library_versions,
        }

    def _attention_context(self):
        """Disallow quadratic-memory SDPA math fallback for CUDA extraction."""
        if not str(self.device).startswith("cuda"):
            return nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError as exc:
            raise RuntimeError(
                "MEMORY_SAFE_ATTENTION_UNAVAILABLE: PyTorch must expose sdpa_kernel; "
                "the diagnostic cannot use an unbudgeted quadratic attention fallback"
            ) from exc
        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])

    def extract(self, prompt_token_ids, response_token_ids, factor_dir, *, cache_key):
        """Persist each sampled response token's prediction hidden state.

        The final prompt position predicts response token zero. EOS, if sampled,
        is treated identically to any other action. Text is never retokenized.
        ``cache_key`` must identify the rollout record and generation contract.
        """
        prompt_ids = np.asarray(prompt_token_ids, dtype=np.int64)
        response_ids = np.asarray(response_token_ids, dtype=np.int64)
        if prompt_ids.ndim != 1 or response_ids.ndim != 1 or not len(prompt_ids) or not len(response_ids):
            raise ValueError("nonempty original prompt and response token ID vectors are required")
        factor_dir = Path(factor_dir)
        identity = self._identity() | {
            "cache_key": str(cache_key),
            "prompt_token_ids": prompt_ids.tolist(),
            "response_token_ids": response_ids.tolist(),
        }
        if factor_dir.exists():
            metadata = json.loads((factor_dir / "metadata.json").read_text())
            if any(metadata.get(key) != value for key, value in identity.items()):
                raise ValueError(f"FACTOR_CACHE_MISMATCH: {factor_dir}")
            self._read_factors(factor_dir)
            return factor_dir
        self.load()
        torch = self._torch
        vocabulary_size, hidden_size = self.head_shape
        all_ids = np.concatenate([prompt_ids, response_ids])
        if np.any(all_ids < 0) or np.any(all_ids >= vocabulary_size):
            raise ValueError("original token ID outside model vocabulary")
        input_ids = torch.tensor(all_ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            # Calling the backbone avoids constructing full sequence x vocabulary logits.
            try:
                with self._attention_context():
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
            except RuntimeError as exc:
                message = str(exc).casefold()
                if any(text in message for text in ("no available kernel", "no viable backend", "no suitable kernel")):
                    raise RuntimeError(
                        "MEMORY_SAFE_ATTENTION_UNAVAILABLE: flash/efficient SDPA cannot handle this "
                        "GPU, dtype, or shape. Quadratic math fallback is disabled; inspect the "
                        "server kernel support and revise preflight before changing the backend."
                    ) from exc
                raise
            hidden = outputs.last_hidden_state[0, len(prompt_ids) - 1 :].float().cpu().numpy().copy()
            del outputs, input_ids
            if hidden.shape != (len(response_ids), hidden_size):
                raise RuntimeError("prediction-hidden alignment did not preserve sampled token count")
            log_z = self._log_normalizers(hidden)
            token_log_probs = self._selected_log_probs(hidden, response_ids, log_z)
        if not np.isfinite(hidden).all() or not np.isfinite(log_z).all() or not np.isfinite(token_log_probs).all():
            raise ValueError("nonfinite extracted feature factors")
        factor_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{factor_dir.name}.", dir=factor_dir.parent))
        try:
            np.save(temporary / "hidden.npy", hidden, allow_pickle=False)
            np.save(temporary / "token_ids.npy", response_ids, allow_pickle=False)
            np.save(temporary / "log_normalizers.npy", log_z, allow_pickle=False)
            np.save(temporary / "token_log_probs.npy", token_log_probs, allow_pickle=False)
            metadata = identity | {
                "token_count": len(response_ids),
                "head_shape": list(self.head_shape),
                "array_sha256": {
                    name: file_hash(temporary / name)
                    for name in ("hidden.npy", "token_ids.npy", "log_normalizers.npy", "token_log_probs.npy")
                },
            }
            (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            os.replace(temporary, factor_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        LOGGER.info(
            "Cached feature factors: geometry=%s path=%s tokens=%s", self.geometry, factor_dir, len(response_ids)
        )
        return factor_dir

    def _log_normalizers(self, hidden):
        torch = self._torch
        weight = self.model.get_output_embeddings().weight
        vocabulary_size = weight.shape[0]
        result = np.empty(len(hidden), dtype=np.float32)
        for start in range(0, len(hidden), self.token_chunk_size):
            stop = min(start + self.token_chunk_size, len(hidden))
            h = torch.tensor(hidden[start:stop], dtype=torch.float32, device=self.device)
            log_z = torch.full((len(h),), -torch.inf, dtype=torch.float32, device=self.device)
            for row in range(0, vocabulary_size, self.logit_vocab_chunk_size):
                logits = h @ weight[row : row + self.logit_vocab_chunk_size].float().T / self.temperature
                log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=1))
            result[start:stop] = log_z.cpu().numpy()
        return result

    def _selected_log_probs(self, hidden, response_ids, log_z):
        """FP32 scores for comparison to cached generation-engine logprobs."""
        torch = self._torch
        weight = self.model.get_output_embeddings().weight
        result = np.empty(len(hidden), dtype=np.float32)
        for start in range(0, len(hidden), self.token_chunk_size):
            stop = min(start + self.token_chunk_size, len(hidden))
            h = torch.tensor(hidden[start:stop], dtype=torch.float32, device=self.device)
            ids = torch.tensor(response_ids[start:stop], dtype=torch.long, device=self.device)
            selected = (h * weight[ids].float()).sum(dim=-1) / self.temperature
            result[start:stop] = selected.cpu().numpy() - log_z[start:stop]
        return result

    def _read_factors(self, factor_path):
        key = str(Path(factor_path).resolve())
        path = Path(factor_path)
        names = ("hidden.npy", "token_ids.npy", "log_normalizers.npy", "token_log_probs.npy")
        signature = tuple(
            (name, (path / name).stat().st_size, (path / name).stat().st_mtime_ns) for name in ("metadata.json", *names)
        )
        identity = self._identity()
        previous = self._verified_factors.get(key, {})
        if previous.get("signature") != signature or previous.get("identity") != identity:
            self._factors.pop(key, None)
            metadata = json.loads((path / "metadata.json").read_text())
            if any(metadata.get(name) != value for name, value in identity.items()):
                raise ValueError(f"FACTOR_CACHE_MISMATCH: {path}")
            if set(metadata.get("array_sha256", {})) != set(names):
                raise ValueError(f"FACTOR_CACHE_CORRUPT: missing content checksums: {path}")
            for name in names:
                if file_hash(path / name) != metadata["array_sha256"][name]:
                    raise ValueError(f"FACTOR_CACHE_CORRUPT: checksum differs: {path / name}")
            # Retain only a tiny validation record, never token lists or arrays.
            self._verified_factors[key] = {
                "signature": signature,
                "identity": identity,
                "token_count": metadata["token_count"],
                "head_shape": metadata["head_shape"],
            }
        if key not in self._factors:
            metadata = self._verified_factors[key]
            hidden = np.load(path / "hidden.npy", mmap_mode="r", allow_pickle=False)
            tokens = np.load(path / "token_ids.npy", mmap_mode="r", allow_pickle=False)
            log_z = np.load(path / "log_normalizers.npy", mmap_mode="r", allow_pickle=False)
            token_log_probs = np.load(path / "token_log_probs.npy", mmap_mode="r", allow_pickle=False)
            if (
                hidden.dtype != np.float32
                or log_z.dtype != np.float32
                or tokens.dtype != np.int64
                or token_log_probs.dtype != np.float32
                or hidden.shape != (metadata["token_count"], metadata["head_shape"][1])
                or tokens.shape != (len(hidden),)
                or log_z.shape != (len(hidden),)
                or token_log_probs.shape != (len(hidden),)
            ):
                raise ValueError(f"invalid factor cache array shapes or dtypes: {path}")
            self._factors[key] = hidden, tokens, log_z
        self._factors.move_to_end(key)
        while len(self._factors) > self.max_open_factors:
            self._factors.popitem(last=False)
        return self._factors[key]

    def clear_factor_cache(self):
        """Release memory mappings after a group; persisted factors are unchanged."""
        self._factors.clear()

    def aggregate_row_block(self, atom, row_start, row_stop):
        """Return an exact weighted aggregate block in FP32, shape [rows, d]."""
        self.load()
        torch = self._torch
        vocabulary_size, hidden_size = self.head_shape
        if not 0 <= row_start < row_stop <= vocabulary_size:
            raise ValueError("invalid vocabulary row block")
        hidden, token_ids, log_z = self._read_factors(atom.factor_path)
        weights = np.ones(len(hidden), dtype=np.float32)
        if atom.token_weights is not None:
            weights = np.asarray(atom.token_weights, dtype=np.float32)
        if weights.shape != (len(hidden),) or not np.isfinite(weights).all():
            raise ValueError("atom weights must be a finite vector of response length")
        with torch.inference_mode():
            output_weight = self.model.get_output_embeddings().weight[row_start:row_stop].float()
            aggregate = torch.zeros((row_stop - row_start, hidden_size), dtype=torch.float32, device=self.device)
            for start in range(0, len(hidden), self.token_chunk_size):
                stop = min(start + self.token_chunk_size, len(hidden))
                selected = np.flatnonzero(weights[start:stop]) + start
                if not len(selected):
                    continue
                h = torch.tensor(hidden[selected], dtype=torch.float32, device=self.device)
                normalizers = torch.tensor(log_z[selected], dtype=torch.float32, device=self.device)
                action_ids = torch.tensor(token_ids[selected], dtype=torch.long, device=self.device)
                token_weights = torch.tensor(weights[selected], dtype=torch.float32, device=self.device)
                logits = h @ output_weight.T / self.temperature
                residual = -torch.exp(logits - normalizers[:, None])
                in_block = (action_ids >= row_start) & (action_ids < row_stop)
                positions = torch.nonzero(in_block).flatten()
                residual[positions, action_ids[positions] - row_start] += 1.0
                residual *= (token_weights / self.temperature)[:, None]
                aggregate.add_(residual.T @ h)
            result = aggregate.cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("nonfinite aggregate head block")
        return result

    def build_heldout_head(self, atoms, rewards, total_draws, out_path):
        """Write the shared held-out score-sum estimator as an atomic .npy memmap."""
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.shape != (len(atoms),) or not np.isin(rewards, [0, 1]).all():
            raise ValueError("one binary reward is required for each held-out atom")
        if not isinstance(total_draws, int | np.integer) or total_draws < len(atoms) or total_draws <= 0:
            raise ValueError("total_draws must count all held-out samples")
        if any(atom.token_weights is not None for atom in atoms):
            raise ValueError("held-out atoms must be token sums without response normalization")
        if not rewards.any():
            raise ValueError("INSUFFICIENT_HELDOUT_SUCCESSES")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{out_path.name}.", dir=out_path.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        vocabulary_size, hidden_size = self.head_shape
        try:
            head = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=self.head_shape)
            for start in range(0, vocabulary_size, self.logit_vocab_chunk_size):
                stop = min(start + self.logit_vocab_chunk_size, vocabulary_size)
                block = np.zeros((stop - start, hidden_size), dtype=np.float32)
                for atom, reward in zip(atoms, rewards, strict=True):
                    if reward:
                        block += self.aggregate_row_block(atom, start, stop) / np.float32(total_draws)
                head[start:stop] = block
                LOGGER.info("Held-out head vocabulary rows %s:%s/%s", start, stop, vocabulary_size)
            head.flush()
            del head
            os.replace(temporary, out_path)
        finally:
            if temporary.exists():
                temporary.unlink()
            self.clear_factor_cache()
        return out_path

    def build_gram(self, atoms, heldout_head=None):
        """Small Gram/cross-Gram with vocabulary-tiled GPU FP32 contractions.

        The returned minimum eigenvalue is diagnostic. This function never
        projects the Gram matrix to PSD or changes its geometry.
        """
        if not atoms:
            raise ValueError("at least one training atom is required")
        self.load()
        torch = self._torch
        vocabulary_size, _ = self.head_shape
        heldout = None
        if heldout_head is not None:
            heldout = np.load(heldout_head, mmap_mode="r", allow_pickle=False)
            if heldout.dtype != np.float32 or heldout.shape != self.head_shape:
                raise ValueError("held-out head geometry or accumulation dtype mismatch")
        gram = np.zeros((len(atoms), len(atoms)), dtype=np.float32)
        cross = np.zeros(len(atoms), dtype=np.float32) if heldout is not None else None
        try:
            with torch.inference_mode():
                for start in range(0, vocabulary_size, self.vocab_chunk_size):
                    stop = min(start + self.vocab_chunk_size, vocabulary_size)
                    blocks = np.stack([self.aggregate_row_block(atom, start, stop) for atom in atoms])
                    factors = torch.tensor(blocks.reshape(len(atoms), -1), dtype=torch.float32, device=self.device)
                    gram += (factors @ factors.T).cpu().numpy()
                    if heldout is not None:
                        target = torch.tensor(np.asarray(heldout[start:stop]).reshape(-1), device=self.device)
                        cross += (factors @ target).cpu().numpy()
                    LOGGER.info(
                        "Training Gram vocabulary rows %s:%s/%s atoms=%s", start, stop, vocabulary_size, len(atoms)
                    )
            gram = np.asarray((gram + gram.T) * np.float32(0.5), dtype=np.float32)
            if not np.isfinite(gram).all() or (cross is not None and not np.isfinite(cross).all()):
                raise ValueError("nonfinite Gram or held-out cross-Gram")
            minimum = float(np.linalg.eigvalsh(gram.astype(np.float64))[0])
            LOGGER.info("Symmetrized exact head Gram: shape=%s minimum_eigenvalue=%s", gram.shape, minimum)
            return {"gram": gram, "heldout_cross": cross, "min_eigenvalue": minimum}
        finally:
            self.clear_factor_cache()


class DeltaProxyExtractor(FeatureExtractor):
    """DelTA's (1-p_y)h proxy; no vocabulary-by-hidden aggregates or head Grams.

    Reuses the frozen backbone and chunked selected-token probability extraction.
    Routing and held-out measurements both use the same hidden-size geometry.
    """

    geometry = "delta_selected_token_gradient_proxy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.temperature != 1.0:
            raise ValueError("DelTA proxy matches the paper at temperature=1.0")

    @property
    def feature_shape(self):
        return (self.head_shape[1],)

    def aggregate_proxy(self, atom):
        hidden, _, _ = self._read_factors(atom.factor_path)
        log_probs = np.load(Path(atom.factor_path) / "token_log_probs.npy", mmap_mode="r", allow_pickle=False)
        weights = np.ones(len(hidden), dtype=np.float32)
        if atom.token_weights is not None:
            weights = np.asarray(atom.token_weights, dtype=np.float32)
        if weights.shape != (len(hidden),) or not np.isfinite(weights).all():
            raise ValueError("atom weights must be a finite vector of response length")
        aggregate = np.zeros(hidden.shape[1], dtype=np.float32)
        for start in range(0, len(hidden), self.token_chunk_size):
            stop = min(start + self.token_chunk_size, len(hidden))
            selected = np.flatnonzero(weights[start:stop]) + start
            if len(selected):
                proxy = delta_proxy_tokens(hidden[selected], log_probs[selected])
                aggregate += weights[selected] @ proxy
        if not np.isfinite(aggregate).all():
            raise ValueError("nonfinite DelTA proxy aggregate")
        return aggregate

    def build_heldout_head(self, atoms, rewards, total_draws, out_path):
        """Keep the cache API; the saved held-out estimator is now a d-vector."""
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.shape != (len(atoms),) or not np.isin(rewards, [0, 1]).all():
            raise ValueError("one binary reward is required for each held-out atom")
        if not isinstance(total_draws, int | np.integer) or total_draws < len(atoms) or total_draws <= 0:
            raise ValueError("total_draws must count all held-out samples")
        if any(atom.token_weights is not None for atom in atoms):
            raise ValueError("held-out atoms must be token sums without response normalization")
        if not rewards.any():
            raise ValueError("INSUFFICIENT_HELDOUT_SUCCESSES")
        aggregate = np.zeros(self.feature_shape, dtype=np.float32)
        try:
            for index, (atom, reward) in enumerate(zip(atoms, rewards, strict=True)):
                if reward:
                    aggregate += self.aggregate_proxy(atom) / np.float32(total_draws)
                LOGGER.info("Held-out DelTA proxy response %s/%s", index + 1, len(atoms))
        finally:
            self.clear_factor_cache()
        if not np.isfinite(aggregate).all():
            raise ValueError("nonfinite held-out DelTA proxy estimator")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{out_path.name}.", dir=out_path.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                np.save(handle, aggregate, allow_pickle=False)
            os.replace(temporary, out_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return out_path

    def build_gram(self, atoms, heldout_head=None):
        """Contract hidden-size proxy vectors, without reinstating vocabulary rows."""
        if not atoms:
            raise ValueError("at least one training atom is required")
        self.load()
        torch = self._torch
        try:
            vectors = np.stack([self.aggregate_proxy(atom) for atom in atoms])
            with torch.inference_mode():
                factors = torch.tensor(vectors, dtype=torch.float32, device=self.device)
                gram = (factors @ factors.T).cpu().numpy()
                cross = None
                if heldout_head is not None:
                    heldout = np.load(heldout_head, allow_pickle=False)
                    if heldout.dtype != np.float32 or heldout.shape != self.feature_shape:
                        raise ValueError("held-out DelTA proxy geometry or accumulation dtype mismatch")
                    target = torch.tensor(heldout, dtype=torch.float32, device=self.device)
                    cross = (factors @ target).cpu().numpy()
            gram = (gram + gram.T) * np.float32(0.5)
            if not np.isfinite(gram).all() or (cross is not None and not np.isfinite(cross).all()):
                raise ValueError("nonfinite DelTA Gram or held-out cross-Gram")
            minimum = float(np.linalg.eigvalsh(gram.astype(np.float64))[0])
            LOGGER.info("DelTA proxy Gram: shape=%s minimum_eigenvalue=%s", gram.shape, minimum)
            return {"gram": gram, "heldout_cross": cross, "min_eigenvalue": minimum}
        finally:
            self.clear_factor_cache()
