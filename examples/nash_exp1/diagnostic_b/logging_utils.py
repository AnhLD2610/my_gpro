# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""One returnable failure log for direct Python and shell-launched runs."""

import contextlib
import datetime as dt
import faulthandler
import importlib.metadata
import logging
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

SECRET = re.compile("TOKEN|KEY|SECRET|PASSWORD", re.IGNORECASE)


def command_output(arguments, *, timeout=15):
    """Capture a bounded diagnostic without invoking a shell."""
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=False)
        return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": arguments, "error": str(exc)}


def environment_report(output_dir, *, gpu=False):
    """Record versions without importing a model or initializing CUDA."""
    versions = {}
    for package in (
        "torch",
        "transformers",
        "vllm",
        "datasets",
        "scipy",
        "cvxpy",
        "clarabel",
        "scs",
        "osqp",
        "numpy",
        "math-verify",
        "pyarrow",
        "zstandard",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    relevant = {
        key: ("<redacted>" if SECRET.search(key) else value)
        for key, value in os.environ.items()
        if key.startswith(("NASH_", "HF_", "CUDA_", "VLLM_", "OMP_"))
    }
    report = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local": dt.datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "visible_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "versions": versions,
        "environment": relevant,
        "disk_usage": dict(zip(("total", "used", "free"), shutil.disk_usage(output_dir), strict=True)),
        "memory": command_output(["free", "-b"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": command_output(["git", "status", "--short"]),
    }
    if gpu:
        report["nvidia_smi"] = command_output(["nvidia-smi"])
    return report


class TeeStream:
    """Mirror Python stdout/stderr and redact known secret values."""

    def __init__(self, original, logfile):
        self.original = original
        self.logfile = logfile
        self.secrets = [value for key, value in os.environ.items() if SECRET.search(key) and len(value) >= 8]

    def write(self, text):
        """Write sanitized text to both destinations."""
        for secret in self.secrets:
            text = text.replace(secret, "<redacted>")
        self.original.write(text)
        if self.logfile is not None:
            self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        """Flush both output streams."""
        self.original.flush()
        if self.logfile is not None:
            self.logfile.flush()

    def isatty(self):
        """Disable terminal progress animations in the single text log."""
        return False

    def fileno(self):
        """Support libraries needing an underlying descriptor."""
        return self.original.fileno()


@contextlib.contextmanager
def single_log(path, command, resume_command):
    """Capture commands, warnings, exceptions and resume hint in one file."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    root = logging.getLogger()
    old_handlers, old_level, old_hook = root.handlers[:], root.level, sys.excepthook
    with path.open("a", buffering=1, encoding="utf-8") as handle:
        # The launcher already redirects both descriptors through tee. Avoid duplicate lines.
        destination = None if os.environ.get("NASH_LOG_TEE") == "1" else handle
        sys.stdout = TeeStream(original_stdout, destination)
        sys.stderr = TeeStream(original_stderr, destination)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        logging.captureWarnings(True)
        faulthandler.enable(file=handle)
        sys.excepthook = lambda kind, value, tb: root.critical("Unhandled exception", exc_info=(kind, value, tb))
        root.info(
            "UTC=%s LOCAL=%s", dt.datetime.now(dt.timezone.utc).isoformat(), dt.datetime.now().astimezone().isoformat()
        )
        root.info("COMMAND: %s", shlex.join(command))
        root.info("RESUME: %s", shlex.join(resume_command))
        root.info("SINGLE_LOG: %s", path)
        try:
            yield
        except BaseException:
            root.exception("Pipeline failed. Completed rollout caches are preserved.")
            root.error("RESUME: %s", shlex.join(resume_command))
            root.error("Return this one log file: %s", path)
            raise
        finally:
            faulthandler.disable()
            logging.captureWarnings(False)
            root.handlers, root.level = old_handlers, old_level
            sys.stdout, sys.stderr, sys.excepthook = original_stdout, original_stderr, old_hook
