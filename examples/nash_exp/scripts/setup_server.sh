#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
python3 -m pip install uv
python3 -m uv venv --python 3.10 --seed --allow-existing .venv
source .venv/bin/activate

pip install -c examples/nash_exp/constraints-server-vllm011.txt -e .
pip install -c examples/nash_exp/constraints-server-vllm011.txt -r examples/nash_exp/requirements-server.txt
pip install -c examples/nash_exp/constraints-server-vllm011.txt vllm==0.11.0

wget -c https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install ./flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
