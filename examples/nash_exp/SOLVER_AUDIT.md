# NCR solver audit — 2026-09-20

The original Round-1 run fell back to GRPO for 24 of its 31 mixed training
groups. The saved diagnostics contain 22 CVXPY solver errors and two rejected
route objective gaps. The original iteration limit was already **500**.
Both round YAML files now state that limit explicitly.

The manuscript comparison remains pending: the README references
`sections/03_method.tex`, `appendix/H_algorithm.tex`,
`appendix/J_collision_powers.tex`, and
`CODEX_MASTER_PROMPT_DIAGNOSTIC_B_NCR.md` at the repository root, but those
files are absent from this checkout. The checks below cover the implemented
equations, local numerical fixtures, and saved experimental inputs.

## Equations checked

Let the columns of the implicit feature matrix `M` be successful response means
followed by failed-segment means, and let `H = M.T @ M`.

- A successful response uses the mean of its token proxy vectors. A failed
  segment uses its own token mean. The segment coefficient includes
  `segment_length / response_length`, preserving the failed response's mean.
- Advantages use `(reward - group_mean) / (population_std + epsilon)`.
  Positive coefficients are `a = A_success / group_size`; negative magnitudes
  are `beta = -A_failure / group_size * segment_length / response_length`.
- The baseline coefficients are `b = (a, -beta)`. With `gamma = sum(a)` and
  `wbar = a / gamma`, the correction coefficients are
  `delta = (gamma * (w - wbar), beta * xi)`.
- Shares `w` lie on the simplex and refunds satisfy `0 <= xi <= 1`.
  Zero-budget entries stay inactive. The correction obeys
  `||M delta|| <= R`, `b.T H delta >= 0`, and `H[:m] delta >= 0`, where
  `R = radius_coefficient * ||M b||`.
- Utilities are `u_i = 1 + (H[i] delta) / (R * sqrt(H[i,i]))`.
  Protection and Cauchy–Schwarz imply `1 <= u_i <= 2`.
- The collision kernel is `((1 + cosine) / 2)**2`. Collision powers minimize
  `q.T B q` on the simplex with `q_i <= (m-1) q_j`, followed by the
  closest-uniform tie-break. Singleton and two-player powers are uniform.
- Linear routing maximizes `sum(q_i * u_i)`. NCR maximizes
  `sum(q_i * log(u_i))`. Both then minimize the correction norm among their
  welfare maximizers. Routing receives no held-out inputs.

These are DelTA proxy features. The saved vectors are not full-model gradients.

## Numerical changes

NCR's second stage previously imposed a log-welfare superlevel constraint
within `1e-8` of the first optimum. Such a nearly tangent constraint can be
numerically difficult. Positive powers make the log objective strictly concave
in the utility vector. Since the feasible utility set is convex, all exact Nash
maximizers share one utility vector. Fixing that vector in the second stage
therefore expresses the same exact tie-break with affine equalities. Linear
routing retains its scalar welfare constraint because its optimal utilities
need not be unique.

The numerical target utilities are recomputed from the first-stage correction
coefficients. Final utility-face, welfare-gap, simplex, refund, radius, anchor,
and protection residuals are checked. Feasibility is also checked against the
original measured Gram after any permitted roundoff PSD repair. Existing
tolerances were retained. A route's reported `objective_gap` measures the
difference between the two stages; it is not an independent global optimality
certificate. Native primal/dual residuals and gaps are recorded separately.

For one player, maximizing utility has the same maximizers as maximizing its
log, so the unnecessary exponential cone is removed. The reported NCR welfare
still uses the logarithm. The norm objective is evaluated through the
equivalent Gram quadratic form to reduce dense auxiliary equalities.

Clarabel uses one solver thread by default. An early numerical stall gets one
fresh solve with automatic equilibration disabled. If direct Nash log welfare
still stalls, the solver tries an equivalent exponential-cone representation
with explicit bounds `1 <= u <= 2` and `0 <= log(u) <= log(2)`. All attempts
retain the same tolerances and 500-iteration cap. An iteration-limit termination
does not trigger these numerical-stall retries. Failures preserve GRPO and
record the failed stage, formulation, native status, iterations, and residuals.

## Validation and saved results

The full Diagnostic-B CPU suite passes **192 tests**, including a synthetic
Nash example with an independent concavity-based global optimum bound,
single-player equivalence, scale invariance, zero-improvement geometry,
bounded-log recovery, original-Gram feasibility, cache reuse, and held-out
isolation. Ruff checks pass for the changed solver and routing tests.

This server returns `OSError: [Errno 14] Bad address` when PyArrow writes under
`/tmp`. A separate three-row Parquet probe reproduced that failure and succeeded
under the workspace. The full passing run used workspace temporary files:

```bash
cd /root/my_gpro/examples/nash_exp
TMPDIR=/root/my_gpro/.cache/nash_solver_audit \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/root/miniforge3/envs/nash/bin/python -m pytest -q tests/diagnostic_b \
  --basetemp=/root/my_gpro/.cache/nash_solver_audit/pytest_validation
```

The original outputs are preserved in
`artifacts/diagnostic_b/round1_tp8/solver_audit/before_fix/`.
The replay uses saved group Grams and the existing held-out estimator.
Its hash comparison is stored in
`artifacts/diagnostic_b/round1_tp8/solver_audit/comparison.json`.
Updated statistics are in
[`report_round1.md`](artifacts/diagnostic_b/round1_tp8/report_round1.md).

| Saved Round-1 result | Original | Revised solver |
| --- | ---: | ---: |
| Mixed training groups | 31 | 31 |
| NCR fallback groups | 24 | 3 |
| Successful directions affected by NCR fallback | 93 | 18 |
| Linear fallback groups | 0 | 0 |
| Collision-power failures | 0 | 0 |

The final replay recovered 21 previously failing NCR groups and introduced no
new fallback groups relative to the original run. The remaining failures are
training prompt indices **30, 49, and 51**, all in the first welfare stage with
native `InsufficientProgress`. Their attempts stopped after 6–19 iterations;
none reached the 500-iteration limit. Their baseline fallbacks remain included
in the report.

Of the 28 accepted NCR groups, 23 first-stage solves reported `optimal` and
five reported `optimal_inaccurate`; the latter passed the independent
feasibility and between-stage checks described above. The largest accepted
absolute between-stage welfare gap was `1.71e-9`; the largest utility-face
residual was `4.82e-9`. These checks do not eliminate all solver uncertainty.

The routing replay took **389.0 seconds**. It regenerated no responses or model
features. SHA-256 checks confirm unchanged rollout caches, verifier outputs,
held-out estimator, and all 31 group Grams. Updated tables and figures were
generated from the revised routes. Pooled Spearman estimates are GRPO `0.1785`,
Linear `0.1976`, and NCR `0.1893`; NCR minus GRPO is `0.0109`, with a paired
95% interval of `[-0.0242, 0.0622]`.

The held-out estimator still contains only **one successful response out of
960 draws**. Solver repairs do not resolve that limitation or establish a
held-out accuracy improvement.
