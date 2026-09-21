# CPU validation record

Date: 2026-09-20 (Asia/Bangkok).

No model weights or benchmark datasets were loaded. No GPU experiment was run.
Round 2 was implemented and its disabled-launcher guard was tested; the round
itself was not executed.

From `verl-0.7.0/examples/nash_exp`, the full test command was:

```bash
/tmp/nash-diagnostic-b-venv/bin/python3 -m pytest -q tests/diagnostic_b
```

Actual result after the author settings and boxed-answer revision: **172 passed, 10 warnings in 12.73 seconds**. The ten CVXPY
warnings report an inaccurate solver status on numerical fixtures. They are
preserved; accepted solutions must separately satisfy the declared feasibility,
certificate, and objective-gap checks. No test was skipped or weakened.

The suite covers population-standard-deviation advantages, exact original-token
segmentation, output-head/autograd identity, temperature differentiation,
held-out denominator and token-sum normalization, collision-power degeneracy and
tie-breaking, matched feasible controls, a strict Nash/Linear witness, support
identities, held-out isolation, paired prompt-cluster uncertainty, schema and
prompt validation, symbolic equivalence and parse failures, one shared generation
engine, request-level resume, cache corruption/mismatch checks, resource
preflight, configuration gates, and single-file failure logging.

The DelTA revision additionally checks the Appendix-F proxy against each
sampled output row's autograd derivative, sharing hidden coordinates across
different token IDs, weighted response/segment aggregation, production proxy
Grams and cross products, original-token prediction alignment, held-out token
sums and all-draw normalization, feature-cache geometry and checksum rejection,
geometry-specific preflight memory estimates, report labels, rejection of mixed
geometries, and reuse of generated responses when switching feature backends.
The original full-head reference and production-backend tests still pass.

The boxed-answer revision checks that the shipped prompt adds the same boxed
instruction to DAPO and held-out questions, preserves earlier DAPO messages and
the original dataset rows, and excludes reference answers. Real Math-Verify CPU
tests check nested fractions, symbolic equivalence, rejection of unboxed numbers,
selection of the final box, malformed final boxes without fallback, and escaped
LaTeX braces. Missing-setting tests now build explicitly unresolved fixtures;
they still check that configuration validation runs before external imports.

These additional checks passed:

```bash
ruff check diagnostic_b tests/diagnostic_b
ruff format --check diagnostic_b tests/diagnostic_b
python3 -m mypy --follow-imports=skip diagnostic_b
python3 -m compileall -q diagnostic_b tests/diagnostic_b
bash -n scripts/run_diagnostic_b_round1_gpu.sh scripts/run_diagnostic_b_round2_gpu.sh
```

Ruff used the parent veRL configuration. Mypy reported no issues in 14 source
files under the repository's existing configuration; that configuration ignores
errors by default outside its selected veRL modules, so this is not a claim of
strict static type coverage. Shell syntax and Python compilation checks passed.

An actual standalone CPU smoke run also completed:

```bash
/tmp/nash-diagnostic-b-venv/bin/python3 -m diagnostic_b \
  --cpu-smoke --output-dir /tmp/nash-diagnostic-b-delta-smoke --resume
```

It produced compressed shared caches, direction records, CSV/JSON/Markdown/LaTeX
tables, PDF/PNG figures, and a report labeled `round_cpu_smoke`. The three-panel
scatter was visually inspected and both axes name the DelTA token-gradient proxy.
These are synthetic fixture outputs, not Qwen
results. The CLI tests additionally run smoke/resume in fresh subprocesses while
blocking imports of all model/dataset loading libraries and verify unchanged
rollout hashes and modification times on resume.

The isolated test environment used Python 3.10.12, CPU PyTorch 2.14.0+cpu,
NumPy 2.2.6, SciPy 1.15.3, CVXPY 1.7.5, Clarabel 0.11.1, pandas 2.3.3,
PyArrow 25.0.1, matplotlib 3.10.9, pytest 9.1.1, Ruff 0.12.2, mypy 1.17.0,
and Math-Verify 0.8.0. This environment was created under `/tmp`; the project
does not depend on that path for server execution.

The author selected seed 42, `sungyub/dapo-math-17k-verl`, the boxed prompt,
Math-Verify boxed-answer scoring, radius coefficient 0.4, and a requested response
cap of 32768. Dataset metadata resolved the training revision to
`3cf5c112137795c1f1d5fdae7c747871b26840a4`; no full dataset was downloaded.

A real CLI dry run with the updated Round-1 YAML stopped before external loading
with only the response/context conflict: a 32768-token response cap leaves no
space for a prompt inside the fixed 32768-token context. The requested value is
preserved pending the author's choice of a smaller fixed response cap or an
explicit remaining-context policy. Its log is
`/tmp/nash-author-settings-dry-run/diagnostic_b_round1.log`.

Real vLLM/Transformers execution, benchmark schema snapshots, memory peaks,
throughput, and measured correlations still require the author's server run.
Preflight records and checks the actual server configuration before costly
generation begins.

## Server setup validation

The simplified `scripts/setup_server.sh` has two offline Bash tests in
`tests/diagnostic_b/test_server_setup.py`: **2 passed in 1.32 seconds**.
They cover the Python 3.10 environment request, ordered editable/requirements/
vLLM/FlashAttention installs, the exact requested wheel URL, paths with spaces,
activation, reuse of an existing environment, and retry after a failed wheel
download. Package installation and wget are mocked; the Bash script runs normally.
Ruff and shell syntax checks passed. No GPU packages were installed locally.

The earlier metadata-only dependency resolution covered editable veRL, root and
experiment requirements and vLLM 0.11.0 with the existing server constraints. It
selected PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, torchaudio 2.8.0+cu128,
Transformers 4.57.6 and NumPy 1.26.4 for Python 3.10. This did not test the newly
requested FlashAttention wheel's binary imports or GPU runtime compatibility.
