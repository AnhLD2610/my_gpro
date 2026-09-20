# Diagnostic B: support versus held-out usefulness

This experiment compares GRPO, Linear Credit Routing, and Nash Credit Routing
(NCR) at one **frozen Qwen/Qwen3-1.7B-Base checkpoint**. It constructs local
directions from shared sampled responses. It does not run training or apply an
optimizer update. A larger support/usefulness correlation is evidence about
local alignment; it does not establish causal accuracy improvement or prove the
paper's transfer theorem.

The implementation follows the [Diagnostic B specification](../../../CODEX_MASTER_PROMPT_DIAGNOSTIC_B_NCR.md),
[method](../../../sections/03_method.tex),
[implementation appendix](../../../appendix/H_algorithm.tex), and
[collision-power appendix](../../../appendix/J_collision_powers.tex).

The author's subsequent request to use DelTA's gradient proxy supersedes the
original specification's full-head-only requirement. Both round configs now use
`features.backend: delta_proxy`, matching [DelTA Appendix F](https://arxiv.org/html/2605.21467v1#A6)
and the local [DelTA implementation](../../../DelTA/verl-DelTA/verl/workers/actor/dp_actor.py).
This adopts their feature representation for the three NCR diagnostic mechanisms;
it does not add DelTA's token-weighting algorithm as another training method.

| Round | Training prompts | Held-out prompts | Draws per prompt | Execution status |
| --- | --- | --- | --- | --- |
| 1 | 64 DAPO-Math-17k prompts, sampled uniformly without replacement | All 30 `MathArena/hmmt_feb_2025` problems | 32 for both sets | Implemented; response/context length conflict must be resolved before execution |
| 2 | 128 DAPO-Math-17k prompts, sampled uniformly without replacement | All 100 `RUC-AIBOX/OlymMATH`, `en-hard`, `test` problems | 32 for both sets | Disabled; requires explicit author approval |

There is one training rollout cache and one held-out rollout cache per round.
All mechanisms share the training samples, verifier, geometry, collision powers,
and routing constraints. Held-out gradients and rewards are excluded from the
routing API. Homogeneous groups and numerical fallbacks remain visible in the
output counts; sampled prompts are never replaced after rewards are observed.

## Experiment settings

Both YAML files contain the author's selected settings:

| Setting | Value |
| --- | --- |
| Seed | `42` |
| Training dataset | [`sungyub/dapo-math-17k-verl`](https://huggingface.co/datasets/sungyub/dapo-math-17k-verl) |
| Training revision | `3cf5c112137795c1f1d5fdae7c747871b26840a4` |
| Prompt wrapper | [`configs/boxed_prompt.json`](configs/boxed_prompt.json) |
| Requested response cap | `sampling.max_new_tokens: 32768` |
| Verifier | `math_verify`, with `answer_format: boxed` |
| Routing radius coefficient | `0.4`, giving `R_route = 0.4 * ||g||` |

The requested response cap conflicts with the experiment's fixed
`model.max_model_len: 32768`: the prompt also occupies context tokens. The
[checkpoint config](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json)
specifies `max_position_embeddings: 32768` with no RoPE scaling. Validation
currently stops before model or dataset loading, pending the author's choice
between a smaller fixed response cap and an explicit policy that caps each
response at the remaining context space. No prompt truncation or context
extension is applied. Round 2 remains disabled.

Sampling is fixed to the declared full-support policy: temperature `1.0`,
`top_p=1.0`, `top_k=-1`, and `min_p=0.0`. Changing to a truncated sampling
distribution requires an explicitly revised experiment. The advantage stabilizer
is `1e-6`, matching the local veRL stabilizer; this diagnostic uses the manuscript's
**population** standard deviation, with divisor `G`.

Edit `configs/diagnostic_b_round1.yaml`, or use the documented environment
overrides below. Environment variables take precedence
over YAML. Use absolute paths for wrapper/local snapshots to avoid ambiguity;
relative paths are resolved relative to the YAML directory.

| Environment variable | Configuration entry or launcher behavior |
| --- | --- |
| `NASH_MAX_NEW_TOKENS` | `sampling.max_new_tokens` |
| `NASH_RADIUS_C` | `routing.radius_coefficient` |
| `NASH_SEED` | `seed` |
| `NASH_DAPO_ID` | `train.id` |
| `NASH_DAPO_REVISION` | `train.revision`, immutable 40-character SHA |
| `NASH_DAPO_PATH` | `train.local_path`, a Parquet file or `datasets.save_to_disk` snapshot |
| `NASH_PROMPT_WRAPPER` | `prompt.wrapper_file` |
| `NASH_VERIFIER` | `verifier.backend`: explicitly select `math_verify` or `custom` |
| `NASH_FEATURE_BACKEND` | `features.backend`: `delta_proxy` (default) or `exact_tiled_head` |
| `NASH_MODEL_REVISION` | Requested `model.revision`; resolved once and pinned before generation |
| `NASH_MODEL_PATH` | Local checkpoint including weights, config and tokenizer files |
| `NASH_TENSOR_PARALLEL_SIZE` | Generation tensor parallelism; default 1 |
| `NASH_GPU_MEMORY_GIB` | Declared GPU budget, capped by observed free GPU memory |
| `NASH_CPU_MEMORY_GIB` | Declared RAM budget, capped by observed available RAM |
| `NASH_DISK_BUDGET_GIB` | Declared artifact-disk budget, capped by observed free disk space |
| `CUDA_VISIBLE_DEVICES` | Visible GPUs and their order; the feature model uses the first visible GPU |
| `NASH_OUTPUT_DIR` | Artifact directory; default `artifacts/diagnostic_b/round1` for Round 1 |
| `NASH_CONFIG` | YAML path used by the shell launcher |
| `NASH_PYTHON` | Python executable used by the shell launcher; default `python3` |

Standard Hugging Face cache settings such as `HF_HOME` remain available. Secrets
must stay in the environment or the server's existing authentication mechanism;
do not put credentials in the experiment YAML or prompt wrapper.

The selected wrapper appends veRL's boxed-answer instruction to each question:

```json
{
  "system": null,
  "user_template": "{problem}\n\nLet's think step by step and output the final answer within \\boxed{{}}.",
  "apply_to_existing_user_message": true,
  "chat_template_kwargs": {}
}
```

`user_template` may substitute only `{problem}`. A system string is optional.
`chat_template_kwargs` accepts only an explicitly chosen `enable_thinking`
setting. The doubled braces in the template become literal `\boxed{}` after
formatting. With `apply_to_existing_user_message: true`, the same instruction
is appended to the final DAPO user message and to held-out questions. Earlier
DAPO messages and roles are preserved; answers never enter the prompts. When
that flag is absent or false, existing DAPO messages are left unchanged.
Both datasets use the tokenizer's own chat template at
the pinned checkpoint revision. If the exact base tokenizer has no template,
preparation stops for the author to resolve that convention; an instruct model
or another template is not substituted.

The selected `math_verify` backend uses the pinned package in `requirements-cpu.txt`.
With `answer_format: boxed`, it extracts the last `\boxed{...}` and checks symbolic
equivalence against the dataset's reference answer. Nested LaTeX braces are
supported. Missing, malformed, or unparseable final boxes receive reward `0` with
a prediction parse-failure status; there is no fallback to earlier boxes or
unboxed numbers. Parsed correct answers receive `1`; parsed incorrect answers
receive `0`. Gold parse failures are recorded separately. The optional `auto`
format retains Math-Verify's general expression extraction. For `custom`, set
`verifier.callable: your_module:your_function` in YAML; the function accepts
`response=` and `answer=` and returns a dictionary with binary `reward`,
`parse_status`, and `reason`. Use `parse_status="parsed"` for parsed answers,
including incorrect ones. The same configured verifier scores both caches.

## Run on the GPU server

Work from `verl-0.7.0/examples/nash_exp` on a server with Python 3, pip and wget:

```bash
bash scripts/setup_server.sh
source ../../.venv/bin/activate
```

The script installs `uv`, creates Python **3.10** in `verl-0.7.0/.venv`, then
installs editable veRL, the requirements, **vllm==0.11.0**, and the requested
**FlashAttention 2.8.3** wheel using `wget -c` and `pip`. The existing constraints
keep PyTorch at 2.8.0. It uses default cache locations and keeps an existing
`.venv` when rerun. The wheel stays in the veRL root directory for reuse.

After resolving the response/context length conflict, validate paths and the wrapper without
loading model weights or datasets:

```bash
python3 -m diagnostic_b --config configs/diagnostic_b_round1.yaml --dry-run
```

The exact Round-1 command is:

```bash
bash scripts/run_diagnostic_b_round1_gpu.sh
```

Use that same command after a failure. It appends to the same log and invokes
the pipeline with `--resume`. Do not remove rollout caches when a later stage
fails. Return this one file for debugging:

```text
artifacts/diagnostic_b/round1/diagnostic_b_round1.log
```

If `NASH_OUTPUT_DIR` is set, the log is inside that directory. The launcher and
Python entry point capture commands, timestamps, configuration, environment
information, stage transitions, exceptions, solver diagnostics, cache identities,
and a resume command. GPU stages run in separate processes so the generation
engine and the feature model do not coexist in GPU memory.

Round 2 has a separate YAML and launcher. The launcher refuses to run without
`--enable-round2` as its first argument. That flag is an execution gate, not a
substitute for the required author approval. Round 2 has not been executed.

## Stages and resuming

The CLI exposes these stages in order:

| Stage | Work |
| --- | --- |
| `preflight` | Validate mandatory settings, pin model/tokenizer metadata, inspect resources, estimate the selected feature cost |
| `prepare` | Validate dataset schemas, sample fixed training row IDs, select all held-out rows, check overlap, render/hash prompts |
| `generate` | Load one vLLM engine and generate both shared caches with stable request-level seeds |
| `verify` | Apply the declared verifier and record rewards and parse failures |
| `heldout_features` | Build the shared held-out score-sum estimator using the denominator of all held-out draws |
| `training_features_and_routes` | Process mixed training groups sequentially, construct small Gram matrices, solve matched routes |
| `statistics` | Compute paired prompt-cluster uncertainty, plots, tables, and the report |
| `all` | Execute the stages above in isolated child processes |

For example, rerun statistics while retaining the completed data and features:

```bash
python3 -m diagnostic_b all --config configs/diagnostic_b_round1.yaml --resume --force-stage statistics
```

`--resume` skips a stage only when its recorded input identity and output
checksums match. `--force-stage` reruns the named mutable stage; dependent stages
are reevaluated through their hashes. Completed prompt manifests and generation
caches are immutable. Changing model revisions, prompts, sampling, response cap,
or rollout seed requires a new output directory. Forcing a completed immutable
stage is rejected.

Factor caches are namespaced by their input identity, so code/configuration
repairs can leave previous files intact. Some changes to feature settings may
recompute factors or Gram matrices even when a narrower cache key could reuse
them; they do not require new generation. Model revisions are locked once and
reused. A local checkpoint is initially content-hashed, and subsequent preflights
check its file snapshot for changes.

## Geometry, numerical checks, and resources

The default token feature is DelTA's sampled-row proxy:

```text
u_t = (1 - p(y_t | prompt, previous generated tokens)) * h_t
```

Here `h_t` is the final hidden state entering the LM head at the position that
predicts the original sampled token `y_t`. At temperature 1, this equals that
token's gradient with respect to its own output-weight row under a fixed hidden
representation. Different token IDs share the same `hidden_size` coordinates,
as in DelTA; output-row identities and other vocabulary-row terms are discarded.
There is no per-token unit normalization or extra advantage/reward factor inside
this feature. FP32 log-probability roundoff up to `1e-5` above zero is clipped to
zero before evaluating `1-p`; larger positive log probabilities are rejected.

Each success uses its response-mean feature; each failure is partitioned at literal blank lines using
original token positions. Delimiters and tokens crossing a text boundary belong
to the preceding nonempty segment. Initial empty segments merge forward.
Every generated action token, including exposed EOS actions, is retained.

The held-out estimator sums the same DelTA token proxies for each successful response and divides
by **all** `M*K` draws. It does not divide by response length or the number of
successes. If there are no held-out successes, the pipeline stops with
`INSUFFICIENT_HELDOUT_SUCCESSES` after preserving the rollout and verifier caches.

The `delta_proxy` backend stores FP32 hidden states and selected-token log
probabilities. Full-vocabulary normalization is computed once per extracted
response using `features.logit_vocab_chunk_size`, without storing all token
logits. Aggregates are hidden-size vectors, and group Grams contract those vectors.
The held-out cache `features/heldout_head.npy` retains its historical filename
but contains a hidden-size vector; manifests record the actual feature shape.

`exact_tiled_head` remains an optional comparison backend for the original
`(one_hot(token) - probability) outer hidden_state` geometry. Its vocabulary
tiles use `features.vocab_chunk_size`. Both backends share response aggregation,
segmentation, routing and statistics; they do not mix geometries within a run.
No random sketch, reference-policy KL, or semantic error oracle is used.

Default figures explicitly label both axes as **DelTA token-gradient proxies**.
The selected-row representation is not a common fixed parameter projection:
the selected row changes with the sampled token. Consequently, the reward-weighted
held-out sum is a proxy, not an exact expected-accuracy gradient. Neither it nor
its inner products establish actual local or finite-step accuracy improvement.
The optional full-head representation also omits the input-embedding path when
weights are tied and is labeled separately.

Switching the feature backend invalidates features, routes and reports while
preserving compatible immutable rollout caches. Run the same launcher with
`--resume` (already enabled) to recompute dependent stages. Existing full-head
factor caches are not accepted as DelTA caches, and the statistics stage rejects
records from mixed geometries.

Solvers receive only training Gram matrices and coefficient vectors. Collision
powers use a QP plus the closest-uniform tie-break on its optimal face; Linear
and NCR share those powers and constraints. Linear's second stage retains its
scalar welfare value and minimizes correction norm. NCR does the same for log
welfare. All objective and feasibility tolerances are recorded. An
`OPTIMAL_INACCURATE` solver status is not accepted on status alone: the returned
solution must pass the applicable certificate, feasibility and objective-gap
checks. Warnings remain in the log. Failed solves fall back to GRPO, with actual
solver failures distinguished from invalid-feature or zero-budget group cases.

Gram matrices are checked for finite entries, symmetry, and positive
semidefiniteness. Only roundoff-sized asymmetry/negative eigenvalues may be
corrected, with the change recorded; material violations trigger fallback.
Accepted routes also undergo feasibility checks in the original measured Gram,
so an eigenvalue repair cannot conceal a failed measured protection or radius.

Preflight downloads remote `config.json` metadata only. It estimates model and
attention workspace, held-out vector/head storage, group atom vectors/tiles and Gram matrices,
factor-cache disk use, and maximum token volume. These are conservative sizing
estimates, not measured peaks or predicted runtimes. Long response caps can
require hundreds of GiB of factor storage. The per-response segment budget is a
resource estimate and never truncates segmentation. An actual group exceeding
that estimate stops for review before allocating its atom tiles. GPU generation
tensor parallelism does not shard the feature model.

An insufficient budget stops the costly pipeline with a saved `preflight.json`.
Review the report and provision resources or explicitly revise the backend/budget;
the code does not silently switch feature backends or change the experiment. Server
execution is still required to establish real throughput and peak memory.

## CPU validation

These commands create an isolated CPU environment. They do not download a model
or dataset:

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python3 -m pip install -r requirements-cpu.txt
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pytest -q tests/diagnostic_b
```

Run the synthetic integration path and its resume path with:

```bash
python3 -m diagnostic_b --cpu-smoke --output-dir /tmp/nash_diagnostic_b_cpu_smoke --resume
```

The smoke run uses a fake generation engine and tiny synthetic DelTA features.
Its artifacts include shared rollout caches, direction records, plots, tables,
`report_round_cpu_smoke.md`, and a single log. Its numerical settings and results
are not production settings or evidence about Qwen or benchmark accuracy.

Formatting and lint checks for the example are:

```bash
ruff check diagnostic_b tests/diagnostic_b
ruff format --check diagnostic_b tests/diagnostic_b
```

No GPU model loading, benchmark generation, or final experiment result is claimed
by these CPU checks.

## Code and outputs

| File | Responsibility |
| --- | --- |
| `diagnostic_b/__main__.py`, `diagnostic_b/__init__.py` | CLI, isolated stage execution and package entry point |
| `diagnostic_b/config.py` | Required author choices and validated overrides |
| `diagnostic_b/preflight.py` | Model metadata lock and resource checks |
| `diagnostic_b/data.py` | Schema adapters, prompt manifests and symbolic verification |
| `diagnostic_b/generation.py` | One generation engine, immutable caches and request seeds |
| `diagnostic_b/features.py` | DelTA token proxy, optional full-head backend, and shared frozen feature extraction |
| `diagnostic_b/segmentation.py` | Original-token span decoding and blank-line partitions |
| `diagnostic_b/routing.py` | Training-only Gram solver API and matched routing mechanisms |
| `diagnostic_b/statistics.py` | Shared observation sets, prompt-cluster bootstrap, tables and plots |
| `diagnostic_b/pipeline.py` | Stage dependencies, measurements and cache reuse |
| `diagnostic_b/storage.py` | Atomic files, hashes and persistent stage state |
| `diagnostic_b/logging_utils.py` | Single-file logging and environment diagnostics |
| `diagnostic_b/smoke.py` | Synthetic CPU integration fixture |
| `configs/diagnostic_b_round1.yaml`, `configs/diagnostic_b_round2.yaml` | Locked round sizes and required author inputs |
| `scripts/run_diagnostic_b_round1_gpu.sh`, `scripts/run_diagnostic_b_round2_gpu.sh` | Preflight, logging and resumable server launchers |
| `tests/diagnostic_b/test_*.py`, `pytest.ini` | CPU mathematical, cache, pipeline and CLI tests |
| `requirements-cpu.txt`, `requirements-gpu.txt` | Test and server dependencies |
| `scripts/setup_server.sh`, `requirements-server.txt`, `constraints-server-vllm011.txt` | Python 3.10 setup, FlashAttention wheel and vLLM 0.11.0 compatibility constraints |
| `CPU_VALIDATION.md` | Actual local validation results and remaining server-only checks |

Round outputs include resolved configuration and environment records; model and
dataset manifests; compressed train/held-out rollouts; verifier results; factor
and Gram caches; route, direction, and solver diagnostic Parquet files;
eligibility counts; runtime records; Spearman/Pearson tables; raw-axis, correlation,
and supplementary rank plots in PDF/PNG; and `report_round1.md`.

Pooled Spearman correlation is the primary metric, Pearson is secondary, and
paired uncertainty resamples training prompts with all their successful
directions together. Reports retain undefined correlations, null/adverse results,
fallbacks, and separately declared selector-capable/active subsets. Held-out
prompt resampling and causal finite-step validation are not implemented.
