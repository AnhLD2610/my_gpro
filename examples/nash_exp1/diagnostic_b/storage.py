# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Atomic artifacts and stage-specific cache identities."""

import hashlib
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path

LOG = logging.getLogger(__name__)


class CacheConflict(RuntimeError):
    """An existing immutable artifact belongs to different inputs."""


def json_default(value):
    """Serialize array scalars and paths without pickling executable objects."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(value):
    """Canonical serialization for configuration and manifest fingerprints."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=json_default)


def digest(value):
    """Hash a JSON-compatible value."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path):
    """Hash a file using bounded memory."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value, *, immutable=False):
    """Commit JSON by atomic rename; never replace an immutable artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read_json(path) != json.loads(canonical_json(value)):
            raise CacheConflict(f"Immutable artifact changed: {path}; use a new output directory")
        return
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path):
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_records(path, records):
    """Atomically write structured records to Parquet."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(records).to_parquet(temporary, index=False)
    os.replace(temporary, path)


def read_records(path):
    """Read Parquet without importing datasets or model libraries."""
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def write_compressed_records(path, records):
    """Write a final immutable compressed JSONL cache from an iterable."""
    import zstandard as zstd

    path = Path(path)
    if path.exists():
        raise CacheConflict(f"Refusing to overwrite completed rollout cache {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with zstd.ZstdCompressor(level=6).stream_writer(raw, closefd=False) as stream:
            for record in records:
                stream.write((canonical_json(record) + "\n").encode())
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def read_compressed_records(path):
    """Stream a compressed rollout cache."""
    import zstandard as zstd

    with Path(path).open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8") as text_stream:
                for line in text_stream:
                    yield json.loads(line)


class StageRunner:
    """Persist success, failure, interruption, inputs, and output checksums."""

    def __init__(self, output_dir, *, resume=False, force_stage=None):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "stage_state.json"
        self.state = read_json(self.path) if self.path.exists() else {"stages": {}}
        self.resume = resume
        self.force_stage = force_stage

    def run(self, name, inputs, function, *, immutable=False):
        """Skip only matching completed inputs whose complete outputs still match."""
        fingerprint = digest(inputs)
        previous = self.state["stages"].get(name, {})
        outputs_valid = bool(previous.get("outputs")) and all(
            (self.root / relative).is_file() and file_hash(self.root / relative) == checksum
            for relative, checksum in previous.get("outputs", {}).items()
        )
        same = previous.get("input_hash") == fingerprint
        if previous.get("status") == "complete":
            if immutable and (not same or not outputs_valid or self.force_stage == name):
                raise CacheConflict(f"Immutable {name} cache differs or force requested; use a new output directory")
            if same and outputs_valid and self.resume and self.force_stage != name:
                LOG.info("stage=%s SKIP input_hash=%s output_hashes=%s", name, fingerprint, previous["outputs"])
                return previous
            if immutable and outputs_valid and same:
                LOG.info("stage=%s immutable completed cache reused", name)
                return previous
        started = time.time()
        record = {"status": "running", "input_hash": fingerprint, "started_unix": started}
        self.state["current_stage"] = name
        self.state["stages"][name] = record
        atomic_json(self.path, self.state)
        LOG.info("stage=%s START last_completed=%s inputs=%s", name, self.state.get("last_completed_stage"), inputs)
        try:
            paths = function()
            output_hashes = {}
            for path in paths:
                path = Path(path).resolve()
                output_hashes[str(path.relative_to(self.root.resolve()))] = file_hash(path)
            if not output_hashes:
                raise ValueError(f"Stage {name} returned no artifacts")
            record.update(status="complete", outputs=output_hashes, elapsed_seconds=time.time() - started)
            self.state["last_completed_stage"] = name
            LOG.info("stage=%s COMPLETE elapsed=%.3f outputs=%s", name, record["elapsed_seconds"], output_hashes)
        except BaseException as exc:
            record.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt | SystemExit) else "failed",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.time() - started,
            )
            LOG.exception("stage=%s FAILED; completed caches preserved", name)
            raise
        finally:
            atomic_json(self.path, self.state)
        return record

    def dependencies(self, names):
        """Identity of completed dependency outputs, independent of timestamps."""
        result = {}
        for name in names:
            entry = self.state["stages"].get(name, {})
            if entry.get("status") != "complete":
                raise RuntimeError(f"Required stage {name} is not complete; run all --resume")
            for relative, checksum in entry["outputs"].items():
                if not (self.root / relative).is_file() or file_hash(self.root / relative) != checksum:
                    raise CacheConflict(f"Dependency artifact changed: {relative}; rerun stage {name}")
            result[name] = {"input_hash": entry["input_hash"], "outputs": entry["outputs"]}
        return result
