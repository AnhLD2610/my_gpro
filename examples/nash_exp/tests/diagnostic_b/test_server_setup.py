# Copyright 2026 Nash Credit Routing contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the short Bash installer with offline stand-ins for package tools."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXP_ROOT = Path(__file__).resolve().parents[2]
WHEEL = "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
STUB = r"""
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

name, args = Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ["SETUP_CALLS"], "a") as stream:
    stream.write(json.dumps({"tool": name, "args": args, "cwd": os.getcwd()}) + "\n")
if name == "python3" and args[:2] == ["-m", "uv"]:
    assert args == ["-m", "uv", "venv", "--python", "3.10", "--seed", "--allow-existing", ".venv"]
    target = Path.cwd() / ".venv/bin"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sys.argv[0], target / "pip")
    (target / "activate").write_text("export PATH=" + shlex.quote(str(target)) + ':"$PATH"\n')
elif name == "wget":
    assert args[0] == "-c"
    if os.environ.get("FAIL_DOWNLOAD") == "1":
        sys.exit(41)
    wheel = Path(args[1].rsplit("/", 1)[-1])
    if not wheel.exists():
        wheel.write_bytes(b"offline wheel placeholder")
elif name == "python3":
    assert args == ["-m", "pip", "install", "uv"]
elif name == "pip":
    assert args[0] == "install"
else:
    raise AssertionError("Unexpected network bootstrap: " + name)
"""


@pytest.fixture
def setup(tmp_path):
    repo = tmp_path / "checkout with spaces"
    experiment = repo / "examples/nash_exp"
    (experiment / "scripts").mkdir(parents=True)
    shutil.copy2(EXP_ROOT / "scripts/setup_server.sh", experiment / "scripts/setup_server.sh")
    tools = tmp_path / "fake tools"
    tools.mkdir()
    for name in ("python3", "wget", "curl"):
        path = tools / name
        path.write_text(f"#!{sys.executable}\n" + STUB)
        path.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    environment = os.environ | {
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "SETUP_CALLS": str(log),
    }
    return repo, experiment, log, environment


def run_setup(setup):
    repo, experiment, log, environment = setup
    return subprocess.run(
        ["bash", str(experiment / "scripts/setup_server.sh")],
        cwd=repo.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )


def calls(setup, tool):
    return [row for line in setup[2].read_text().splitlines() if (row := json.loads(line))["tool"] == tool]


def test_simple_setup_install_order_flash_wheel_and_environment_reuse(setup):
    repo, experiment, _, environment = setup
    first = run_setup(setup)
    assert first.returncode == 0, first.stdout + first.stderr
    installs = calls(setup, "pip")
    assert len(installs) == 4
    assert installs[0]["cwd"] == str(repo)
    assert installs[0]["args"][-2:] == ["-e", "."]
    assert installs[1]["args"][-2:] == ["-r", "examples/nash_exp/requirements-server.txt"]
    assert installs[2]["args"][-1] == "vllm==0.11.0"
    assert installs[3]["args"][-1] == "./" + WHEEL
    assert calls(setup, "wget")[0]["args"] == [
        "-c",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/" + WHEEL,
    ]
    activation = repo / ".venv/bin/activate"
    subprocess.run(["bash", "-c", 'source "$1"', "bash", str(activation)], env=environment, check=True)
    second = run_setup(setup)
    assert second.returncode == 0, second.stdout + second.stderr
    assert len(calls(setup, "pip")) == 8


def test_failed_wheel_download_stops_before_install_and_can_be_retried(setup):
    setup[3]["FAIL_DOWNLOAD"] = "1"
    assert run_setup(setup).returncode == 41
    assert len(calls(setup, "pip")) == 3
    setup[3].pop("FAIL_DOWNLOAD")
    retried = run_setup(setup)
    assert retried.returncode == 0, retried.stdout + retried.stderr
    assert calls(setup, "pip")[-1]["args"][-1].endswith(WHEEL)
