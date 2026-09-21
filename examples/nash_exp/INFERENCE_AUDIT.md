# Inference audit and 500/500 configuration

Date: 2026-09-21. This change prepares the larger experiment and improves request
scheduling. No new benchmark responses were generated locally and no GPU speedup
has been measured.

## Experiment size

Round 1 now samples 500 distinct training problems with seed 42 from the pinned
7,500-row MATH train split and uses all 500 rows of its MATH500 test split in
source order. The dataset remains `ShuoZheLi/MATH-train-MATH500-test`, revision
`09ad2162c6f3a07ea71e48d7af02922b093df041`. There are 32 draws per problem:
16,000 train responses and 16,000 test responses. MATH500 is a subset of the
original MATH test set.

The frozen Qwen2.5-Math-7B revision, full-support sampling, boxed verifier,
DelTA proxy, and routing coefficient 0.4 remain the experimental settings.
Each method consumes the same responses. Generation is performed once per
request, not once per GRPO/Linear/Nash mechanism.

The prompt cap is 1,024 tokens including the wrapper and chat template; the
response cap is 3,072, within the native 4,096-token context. Preparation must
check the newly selected prompts. The maximum in the saved smaller run does
not establish that all 1,000 new prompts fit. An overlong prompt stops preparation
before generation; prompts are neither dropped nor truncated to pass this check.

## Evidence from the saved run

Evidence is in
[`round1_qwen25math7b_math96_test32_p1024_g3072`](artifacts/diagnostic_b/round1_qwen25math7b_math96_test32_p1024_g3072):
`stage_state.json`, `runtime_breakdown.json`, `environment.json`,
`generation_execution.json`, the main log, and the immutable rollout caches.
These artifacts were read without modification. This run had 96 train problems,
32 test problems, and 4,096 total responses.

| Recorded stage | Elapsed time |
| --- | ---: |
| Generation | 1,795.43 s / 29.92 min |
| Held-out features | 63.70 s / 1.06 min |
| Training features and routes | 795.55 s / 13.26 min |
| Verification | 7.58 s |
| Statistics | 5.55 s |
| Sum of all recorded stages, including preparation and preflight | 2,673.90 s / 44.57 min |

Generation accounted for 67.1% of the recorded stage time. Statistics time comes
from `stage_state.json`; `runtime_breakdown.json` was written during statistics
and has a null entry for that stage.

The old caller passed only 32 separate `n=1` requests to each blocking
`LLM.generate` call. Those were the 32 responses for one prompt. vLLM could batch
within that call, but the caller supplied no next prompt until the entire call
returned. Short responses left fewer active sequences while the longest response
finished. At least one response hit the 3,072-token limit in 81/96 training
groups and 30/32 held-out groups; median lengths were only 531 and 535 tokens.
Completions were also saved only after the entire call returned.

The saved server had eight RTX PRO 6000 Blackwell GPUs with about 95.6 GiB each.
It ran two TP4 engines: four GPUs for train and four for test. The log reports
the custom all-reduce path disabled for more than two PCIe-only GPUs, with NCCL
used instead. Each full BF16 model is approximately 14.19 GiB, so this hardware
can hold independent copies on single GPUs. Removing TP communication is a
hardware-informed optimization; the throughput improvement still needs measurement.

The test worker finished at 16:51:10 UTC and the train worker at 17:10:25 UTC:
four GPUs were idle for about 19 minutes because the split sizes differed.
The new split sizes are equal, although differences in response length can
still produce worker imbalance.

FlashAttention, prefix caching, CUDA graphs, and chunked prefill were already
active in the saved run. Enabling them again would not remove the caller's
32-request barrier. Initial engine setup took about 42 seconds per engine;
most of the generation stage was spent producing responses.

## Implemented changes

| Before | Current Round 1 |
| --- | --- |
| Two engines, each spread across four GPUs | Eight TP1 engines, four per split |
| One prompt's 32 responses per blocking call | Up to 256 outstanding requests per engine, spanning multiple prompts |
| Next batch waits for the longest answer | Refill available slots as completions arrive |
| Cache commits after the whole call returns | Each finished answer validated and atomically committed immediately |
| Split sizes 96 versus 32 | Split sizes 500 versus 500; 125 prompts per worker |

`generation.py` uses the pinned vLLM 0.11 `LLM.llm_engine` request API:
`add_request`, `has_unfinished_requests`, and `step`. `RequestOutputKind.FINAL_ONLY`
avoids repeatedly constructing cumulative outputs on every decode step. This
uses the actual enum in production. Per-request seeds and `n=1` sampling are
retained. Responses are matched by request ID, so engine completion order does
not change the final canonical cache ordering.

`parallel_generation.py` assigns whole prompts by
`prompt_index % replicas_per_split`. All 32 draws of a prompt stay on the same
engine for prefix-cache reuse. GPU groups are disjoint, all workers start before
the supervisor waits, and failures terminate and reap the owned worker groups.
For multiple replicas, only the supervisor publishes the final split caches,
after all workers have succeeded. Completed response files remain available
for resumption.

The scheduler limits are explicit: `max_num_seqs=256`,
`max_num_batched_tokens=16384`, and `enable_chunked_prefill=true`.
`max_in_flight=256` bounds outstanding requests; vLLM determines how many can be
resident in its allocated KV memory. The old blocking dispatcher remains
available through `NASH_GENERATION_MODE=batch` for a separate comparison run.

`preflight.py` accounts for every concurrent model copy and every assigned GPU.
Using the saved server's hardware metadata with the new configuration gives:

| Estimate | Value |
| --- | ---: |
| Concurrent engines / required GPUs | 8 / 8 |
| Host RAM reserve for concurrent generation | 121.48 GiB |
| Feature extraction GPU peak estimate | 17.20 GiB |
| Artifact disk estimate | 119.77 GiB |
| Maximum generated token count | 98,304,000 |

These are conservative sizing calculations, not measured peaks or a runtime
forecast. Existing safety margins are applied. A smaller server must reduce
replica count or change tensor parallelism explicitly; preflight will reject
insufficient resources.

The downloaded experiment folder also lacked required runtime support modules,
the boxed wrapper, and installation requirements. Those were restored from the
local `nash_exp1` copy. Restored feature tests were adapted to the current paired
normalization API; production feature geometry was not changed.

## Launch and inspect

From `verl-0.7.0`, with the compatible environment active:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash examples/nash_exp/scripts/run_diagnostic_b_round1_gpu.sh
```

The default output is a fresh directory:
`examples/nash_exp/artifacts/diagnostic_b/round1_qwen25math7b_math500_test500_p1024_g3072`.
The [README](README.md#run-on-the-gpu-server) covers environment setup and fewer
GPUs. Clear any old `NASH_TENSOR_PARALLEL_SIZE=4` override to use the new TP1 default.

The main log reports completion counts and engine token throughput every ten
seconds during continuous generation. `generation_metrics/*.json` records
completed worker invocations: new requests, new output tokens, reused responses,
and engine time excluding model startup. After an interrupted invocation, its
finished responses are preserved, while its final metrics file may be absent.
On resume, metrics describe the remaining work completed by the new invocation;
do not interpret them as cumulative whole-experiment timing. Use the main log
and stage state for run timing.

Replica count and serial/parallel topology can change while reusing compatible
completed parts. TP, scheduler limits, dispatch mode, and outstanding-request
window are recorded in the immutable generation contract and need a new output
directory if changed. The new 500/500 prompt manifest also requires a fresh
directory; old 96/32 response caches must not be copied into it.

## Validation and remaining cost

Local validation completed with Python 3.10, PyTorch 2.8.0+cpu, Transformers
4.57.6, and Ruff 0.12.2:

- Full diagnostic suite: **361 passed** in 24.69 seconds. Fifteen CVXPY
  inaccurate-solution warnings were emitted by numerical test cases; no tests failed.
- Ruff lint and formatting, Python compilation, and shell syntax checks passed.
- The production 500/500 configuration passed the CLI dry run without loading
  a model or dataset.
- The synthetic CPU pipeline produced its caches, route records, correlations,
  and figures; a second invocation resumed successfully. These are test
  artifacts, not benchmark results.

CPU tests cover prompt selection, stable request seeds, continuous slot refill,
unordered completion, request-level recovery, malformed outputs, final cache
assembly, resource checks, and eight real worker subprocesses using a fake vLLM
engine. Both fresh execution and interrupted resumption exercise the production
continuous dispatcher and replica supervisor together. Independent tests still
check feature math, routing constraints, and report contracts.

GPU execution is required to measure throughput, CUDA memory behavior, and the
best outstanding-request window. This audit makes no speedup-factor claim.
After generation, feature extraction still uses the first visible GPU and
training groups are processed sequentially; routing remains on CPU. Those two
feature stages together took 14.32 minutes in the saved run. They remain a
material part of the larger experiment's cost.

Pinned upstream API references:

- [vLLM 0.11.0 LLMEngine implementation](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/engine/llm_engine.py)
- [vLLM 0.11.0 output kinds and sampling parameters](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/sampling_params.py)
- [vLLM 0.11.0 optimization guidance](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html)
