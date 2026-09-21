# Copyright 2026 The nash_exp contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Fresh-checkpoint routing in a training-only Gram geometry.

Columns of the implicit atom matrix are successful response means followed by
failed-segment means. No held-out quantities are accepted by this module.
Solver arithmetic is float64; upstream feature contractions must accumulate
at least in FP32. Numerical PSD repairs are limited to roundoff and recorded.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainingGeometry:
    gram: np.ndarray
    positive_coefficients: np.ndarray
    negative_coefficients: np.ndarray


@dataclass(frozen=True)
class SolverConfig:
    solver: str = "CLARABEL"
    feasibility_tolerance: float = 5e-6
    objective_tolerance: float = 1e-8
    power_gap_tolerance: float = 2e-7
    psd_relative_tolerance: float = 1e-8
    rank_relative_tolerance: float = 1e-11
    zero_norm_tolerance: float = 1e-12
    solver_tolerance: float = 1e-9
    max_iterations: int = 500


DEFAULT_SOLVER_CONFIG = SolverConfig()


@dataclass
class RouteSolution:
    coefficients: np.ndarray
    positive_shares: np.ndarray
    refunds: np.ndarray
    support: np.ndarray
    added_support: np.ndarray
    utilities: np.ndarray
    fallback: bool
    reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupSolution:
    powers: np.ndarray
    power_diagnostics: dict[str, Any]
    linear: RouteSolution
    ncr: RouteSolution
    baseline_coefficients: np.ndarray
    baseline_norm: float
    radius: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def standardized_advantages(rewards: np.ndarray, epsilon: float) -> np.ndarray:
    """Binary group advantages with population standard deviation and caller epsilon."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or len(rewards) < 2 or not np.isin(rewards, [0, 1]).all():
        raise ValueError("rewards must be a one-dimensional binary group of size >= 2")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a declared positive finite value")
    return (rewards - rewards.mean()) / (rewards.std(ddof=0) + epsilon)


def _psd(matrix: np.ndarray, config: SolverConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("NONFINITE_GRAM")
    symmetry_residual = float(np.max(np.abs(matrix - matrix.T), initial=0))
    entry_scale = max(float(np.max(np.abs(matrix), initial=0)), np.finfo(float).tiny)
    if symmetry_residual > config.psd_relative_tolerance * entry_scale:
        raise ValueError(
            f"MATERIALLY_ASYMMETRIC_GRAM: symmetry_residual={symmetry_residual}, entry_scale={entry_scale}"
        )
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0)), np.finfo(float).tiny)
    minimum = float(eigenvalues.min()) if len(eigenvalues) else 0.0
    diagnostics = {
        "gram_min_eigenvalue": minimum,
        "gram_symmetry_residual": symmetry_residual,
        "gram_psd_relative_tolerance": config.psd_relative_tolerance,
        "gram_roundoff_repair_norm": float(np.linalg.norm(np.minimum(eigenvalues, 0))),
    }
    if minimum < -config.psd_relative_tolerance * scale:
        raise ValueError(f"MATERIALLY_INDEFINITE_GRAM: min_eigenvalue={minimum}, scale={scale}")
    root = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * vectors.T
    # The root, and all residual calculations, refer to the same logged geometry.
    return root.T @ root, root, diagnostics


def _optimize(problem: Any, config: SolverConfig) -> None:
    import cvxpy as cp

    if config.solver not in cp.installed_solvers():
        raise RuntimeError(f"SOLVER_UNAVAILABLE: {config.solver}; installed={cp.installed_solvers()}")
    options: dict[str, Any] = {}
    if config.solver == "CLARABEL":
        options = {
            "tol_gap_abs": config.solver_tolerance,
            "tol_gap_rel": config.solver_tolerance,
            "tol_feas": config.solver_tolerance,
            "max_iter": config.max_iterations,
        }
    elif config.solver == "SCS":
        options = {"eps": config.solver_tolerance, "max_iters": config.max_iterations * 100}
    problem.solve(solver=config.solver, verbose=False, **options)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SOLVE_FAILED: status={problem.status}")


def _ratio_matrix(m: int) -> np.ndarray:
    constraints = []
    for i in range(m):
        for j in range(m):
            if i != j:
                row = np.zeros(m)
                row[i], row[j] = 1, -(m - 1)
                constraints.append(row)
    return np.asarray(constraints)


def _power_certificate(collision: np.ndarray, q: np.ndarray, ratio: np.ndarray) -> float:
    from scipy.optimize import linprog

    gradient = 2 * collision @ q
    result = linprog(
        gradient,
        A_ub=ratio,
        b_ub=np.zeros(len(ratio)),
        A_eq=np.ones((1, len(q))),
        b_eq=[1],
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"POWER_CERTIFICATE_FAILED: {result.message}")
    return max(0.0, float(gradient @ q - result.fun))


def collision_powers(
    positive_gram: np.ndarray, config: SolverConfig = DEFAULT_SOLVER_CONFIG
) -> tuple[np.ndarray, dict]:
    """Collision QP followed by the closest-uniform optimum on its optimal face."""
    import cvxpy as cp

    positive_gram = np.asarray(positive_gram, dtype=np.float64)
    m = len(positive_gram)
    if positive_gram.shape != (m, m) or m == 0:
        raise ValueError("positive_gram must be a nonempty square matrix")
    positive_gram, _, diagnostics = _psd(positive_gram, config)
    norms = np.sqrt(np.maximum(np.diag(positive_gram), 0))
    if np.any(norms <= config.zero_norm_tolerance):
        raise ValueError("ZERO_REQUIRED_POSITIVE_FEATURE")
    cosine = positive_gram / np.outer(norms, norms)
    excess = float(np.max(np.maximum(np.abs(cosine) - 1, 0)))
    if excess > config.feasibility_tolerance:
        raise ValueError("INVALID_COSINE_GRAM")
    collision = ((1 + np.clip(cosine, -1, 1)) / 2) ** 2
    collision, _, collision_diagnostics = _psd(collision, config)
    diagnostics.update({f"collision_{key}": value for key, value in collision_diagnostics.items()})
    diagnostics["cosine_roundoff_clip"] = excess
    uniform = np.full(m, 1 / m)
    if m <= 2:
        diagnostics.update(
            status="analytic_singleton" if m == 1 else "analytic_ratio_two",
            collision=float(uniform @ collision @ uniform),
            first_stage_gap=0.0,
            final_stage_gap=0.0,
            closest_uniform_objective=0.0,
        )
        return uniform, diagnostics
    ratio = _ratio_matrix(m)
    q = cp.Variable(m)
    constraints = [q >= 0, cp.sum(q) == 1, ratio @ q <= 0]
    tangent = np.eye(m) - np.ones((m, m)) / m
    tangent_collision = tangent @ collision @ tangent
    tangent_collision = (tangent_collision + tangent_collision.T) / 2
    objective_scale = max(float(np.linalg.norm(tangent_collision, ord=2)), 1e-12)
    # Remove the simplex-constant term and scale curvature. This is algebraic
    # conditioning, not a ridge; nearly coincident types otherwise lose power
    # accuracy even when the unscaled collision objective is well certified.
    centered = q - uniform
    first_objective = (
        cp.quad_form(centered, cp.psd_wrap(tangent_collision)) + 2 * (tangent @ collision @ uniform) @ centered
    ) / objective_scale
    first = cp.Problem(cp.Minimize(first_objective), constraints)
    _optimize(first, config)
    q0 = np.asarray(q.value).copy()
    first_residual = max(abs(float(q0.sum() - 1)), -float(q0.min()), float((ratio @ q0).max()))
    if not np.isfinite(q0).all() or first_residual > config.feasibility_tolerance:
        raise RuntimeError(f"INVALID_FIRST_POWER_FEASIBILITY: residual={first_residual}")
    first_gap = _power_certificate(collision, q0, ratio)
    if first_gap > config.power_gap_tolerance:
        raise RuntimeError(f"UNCERTIFIED_POWER_OPTIMUM: LP_gap={first_gap}")
    eigenvalues, vectors = np.linalg.eigh(collision)
    rank_threshold = config.rank_relative_tolerance * max(float(eigenvalues.max()), np.finfo(float).tiny)
    retained = eigenvalues > rank_threshold
    # These independent rows represent B q = B q0; no ridge or temperature.
    face_rows = vectors[:, retained].T
    face = face_rows @ q == face_rows @ q0
    second = cp.Problem(cp.Minimize(cp.sum_squares(q - uniform)), [*constraints, face])
    _optimize(second, config)
    answer = np.asarray(q.value).copy()
    final_gap = _power_certificate(collision, answer, ratio)
    residual = max(abs(float(answer.sum() - 1)), -float(answer.min()), float((ratio @ answer).max()))
    face_residual = float(np.max(np.abs(collision @ (answer - q0))))
    if (
        not np.isfinite(answer).all()
        or residual > config.feasibility_tolerance
        or final_gap > config.power_gap_tolerance
        or face_residual > config.feasibility_tolerance
    ):
        raise RuntimeError(f"INVALID_POWER_SOLUTION: residual={residual}, gap={final_gap}, face={face_residual}")
    diagnostics.update(
        status=second.status,
        first_stage_status=first.status,
        first_stage_collision=float(q0 @ collision @ q0),
        collision=float(answer @ collision @ answer),
        uniform_collision=float(uniform @ collision @ uniform),
        first_stage_gap=first_gap,
        first_stage_feasibility_residual=first_residual,
        final_stage_gap=final_gap,
        simplex_residual=abs(float(answer.sum() - 1)),
        ratio_residual=max(0.0, float((ratio @ answer).max())),
        power_ratio=float(answer.max() / answer.min()),
        minimum_power=float(answer.min()),
        closest_uniform_objective=float(np.sum((answer - uniform) ** 2)),
        optimal_face_residual=face_residual,
        numerical_rank=int(retained.sum()),
        rank_threshold=rank_threshold,
        first_objective_scale=objective_scale,
    )
    return answer, diagnostics


def _fallback(h: np.ndarray, b: np.ndarray, a: np.ndarray, beta: np.ndarray, reason: str, details=None):
    m = len(a)
    norms = np.sqrt(np.maximum(np.diag(h)[:m], 0))
    support = np.full(m, np.nan)
    np.divide(h[:m] @ b, norms, out=support, where=np.isfinite(norms) & (norms > 0))
    w = a / a.sum() if np.isfinite(a).all() and a.sum() > 0 else np.zeros(m)
    return RouteSolution(
        coefficients=np.zeros(len(b)),
        positive_shares=w,
        refunds=np.zeros(len(beta)),
        support=support,
        added_support=np.zeros(m),
        utilities=np.ones(m),
        fallback=True,
        reason=reason,
        diagnostics={"solve_attempted": False, **({} if details is None else details)},
    )


def _route(h, root, b, a, beta, powers, radius, baseline_norm, objective, config):
    import cvxpy as cp

    m, s = len(a), len(beta)
    gamma = float(a.sum())
    wbar = a / gamma
    norms = np.sqrt(np.diag(h)[:m])
    w = cp.Variable(m)
    xi = cp.Variable(s) if s else None
    delta = cp.hstack([gamma * (w - wbar), cp.multiply(beta, xi)]) if s else gamma * (w - wbar)
    normalized_slopes = h[:m] @ delta / (radius * norms)
    utilities = 1 + normalized_slopes
    constraints = [
        w >= 0,
        cp.sum(w) == 1,
        cp.norm(root @ delta / radius, 2) <= 1,
        (b @ h @ delta) / (baseline_norm * radius) >= 0,
        normalized_slopes >= 0,
    ]
    if s:
        constraints.extend([xi >= 0, xi <= 1])
        if np.any(beta == 0):
            constraints.append(xi[np.flatnonzero(beta == 0)] == 0)
    if np.any(a == 0):
        constraints.append(w[np.flatnonzero(a == 0)] == 0)
    welfare = powers @ utilities if objective == "linear" else powers @ cp.log(utilities)
    first = cp.Problem(cp.Maximize(welfare), constraints)
    _optimize(first, config)
    first_value = float(welfare.value)
    # Retain only scalar welfare for the linear control, never its arbitrary utility vector.
    second = cp.Problem(
        cp.Minimize(cp.sum_squares(root @ delta / radius) / 2),
        [*constraints, welfare >= first_value - config.objective_tolerance],
    )
    _optimize(second, config)
    coefficients = np.asarray(delta.value, dtype=np.float64).reshape(-1)
    shares = np.asarray(w.value, dtype=np.float64).reshape(-1)
    refunds = np.asarray(xi.value, dtype=np.float64).reshape(-1) if s else np.empty(0)
    delta_norm = float(np.linalg.norm(root @ coefficients))
    added = h[:m] @ coefficients / norms
    final_utilities = 1 + added / radius
    final_value = float(powers @ (final_utilities if objective == "linear" else np.log(final_utilities)))
    anchor = float(b @ h @ coefficients)
    protections = h[:m] @ coefficients
    diagnostics = {
        "solve_attempted": True,
        "status": second.status,
        "first_stage_status": first.status,
        "objective": objective,
        "first_stage_objective": first_value,
        "objective_value": final_value,
        "objective_gap": first_value - final_value,
        "objective_tolerance": config.objective_tolerance,
        "tie_break_objective": delta_norm**2 / 2,
        "simplex_residual": abs(float(shares.sum() - 1)),
        "minimum_share": float(shares.min()),
        "inactive_share_residual": float(np.max(np.abs(shares[a == 0]), initial=0)),
        "inactive_refund_residual": float(np.max(np.abs(refunds[beta == 0]), initial=0)),
        "refund_lower_residual": max(0.0, -float(refunds.min(initial=0))),
        "refund_upper_residual": max(0.0, float(refunds.max(initial=1)) - 1),
        "radius_residual": max(0.0, delta_norm - radius),
        "relative_radius_residual": max(0.0, delta_norm / radius - 1),
        "correction_norm": delta_norm,
        "anchor_margin": anchor,
        "normalized_anchor_margin": anchor / (baseline_norm * radius),
        "protection_margins": protections.tolist(),
        "normalized_protection_margins": (added / radius).tolist(),
        "minimum_utility": float(final_utilities.min()),
        "maximum_utility": float(final_utilities.max()),
        "feasibility_tolerance": config.feasibility_tolerance,
    }
    badness = max(
        diagnostics["simplex_residual"],
        -diagnostics["minimum_share"],
        diagnostics["inactive_share_residual"],
        diagnostics["inactive_refund_residual"],
        diagnostics["refund_lower_residual"],
        diagnostics["refund_upper_residual"],
        diagnostics["relative_radius_residual"],
        -diagnostics["normalized_anchor_margin"],
        -float((added / radius).min()),
        1 - diagnostics["minimum_utility"],
        diagnostics["maximum_utility"] - 2,
    )
    values = np.concatenate([coefficients, shares, refunds, final_utilities, [final_value]])
    if not np.isfinite(values).all() or badness > config.feasibility_tolerance:
        return _fallback(h, b, a, beta, "INVALID_ROUTE_FEASIBILITY", diagnostics)
    objective_slack = 10 * config.solver_tolerance
    if abs(first_value - final_value) > config.objective_tolerance + objective_slack:
        return _fallback(h, b, a, beta, "INVALID_ROUTE_OBJECTIVE_GAP", diagnostics)
    return RouteSolution(
        coefficients=coefficients,
        positive_shares=shares,
        refunds=refunds,
        support=(h[:m] @ (b + coefficients)) / norms,
        added_support=added,
        utilities=final_utilities,
        fallback=False,
        reason=None,
        diagnostics=diagnostics,
    )


def _audit_measured_route(measured_gram, b, a, beta, route, radius_coefficient, config):
    """Check returned coefficients against the original measured geometry too.

    A small eigenvalue repair can matter after cancellation in the baseline or
    normalization by a small positive norm. Accepted repaired-geometry solutions
    therefore also need the same feasibility tolerance in the measured Gram.
    """
    if route.fallback:
        return route
    m = len(a)
    coefficients = route.coefficients
    positive_squared = np.diag(measured_gram)[:m]
    baseline_squared = float(b @ measured_gram @ b)
    delta_squared = float(coefficients @ measured_gram @ coefficients)
    measured_baseline_norm = float(np.sqrt(max(0.0, baseline_squared)))
    measured_radius = radius_coefficient * measured_baseline_norm
    audit = {
        "measured_geometry_audited": True,
        "measured_baseline_norm_squared": baseline_squared,
        "measured_baseline_norm": measured_baseline_norm,
        "measured_radius": measured_radius,
        "measured_correction_norm_squared": delta_squared,
        "measured_relative_radius_residual": None,
        "measured_normalized_anchor_margin": None,
        "measured_normalized_protection_margins": None,
        "measured_max_support_shift": None,
    }
    if (
        measured_baseline_norm <= config.zero_norm_tolerance
        or measured_radius <= 0
        or np.any(positive_squared <= config.zero_norm_tolerance**2)
        or not np.isfinite([measured_radius, baseline_squared, delta_squared]).all()
    ):
        audit["measured_geometry_failure"] = "undefined measured baseline radius or required positive norm"
        return _fallback(measured_gram, b, a, beta, "INVALID_MEASURED_GEOMETRY_FEASIBILITY", route.diagnostics | audit)
    norms = np.sqrt(positive_squared)
    measured_delta_norm = float(np.sqrt(max(0.0, delta_squared)))
    radius_residual = max(0.0, measured_delta_norm / measured_radius - 1)
    anchor = float(b @ measured_gram @ coefficients)
    normalized_anchor = anchor / (measured_baseline_norm * measured_radius)
    protection = measured_gram[:m] @ coefficients
    normalized_protection = protection / (norms * measured_radius)
    measured_support = measured_gram[:m] @ (b + coefficients) / norms
    support_shift = float(np.max(np.abs(measured_support - route.support)))
    audit.update(
        measured_correction_norm=measured_delta_norm,
        measured_radius_residual=max(0.0, measured_delta_norm - measured_radius),
        measured_relative_radius_residual=radius_residual,
        measured_anchor_margin=anchor,
        measured_normalized_anchor_margin=normalized_anchor,
        measured_protection_margins=protection.tolist(),
        measured_normalized_protection_margins=normalized_protection.tolist(),
        measured_minimum_utility=float(1 + normalized_protection.min()),
        measured_maximum_utility=float(1 + normalized_protection.max()),
        measured_max_support_shift=support_shift,
    )
    violation = max(
        radius_residual, -normalized_anchor, -float(normalized_protection.min()), float(normalized_protection.max()) - 1
    )
    finite = np.isfinite(np.concatenate([measured_support, normalized_protection, [normalized_anchor]])).all()
    if not finite or violation > config.feasibility_tolerance:
        audit["measured_geometry_failure"] = "measured radius, anchor or protection violates the solver tolerance"
        return _fallback(measured_gram, b, a, beta, "INVALID_MEASURED_GEOMETRY_FEASIBILITY", route.diagnostics | audit)
    route.diagnostics.update(audit)
    return route


def solve_group(
    geometry: TrainingGeometry, radius_coefficient: float, config: SolverConfig = DEFAULT_SOLVER_CONFIG
) -> GroupSolution:
    """Solve both matched controls once with shared training-only powers and atoms.

    Numerical failures preserve the baseline for the affected mechanism. A bad
    required positive feature or failed power solve falls back for both methods.
    Structural API errors (dimensions or missing radius) raise immediately.
    """
    h = np.asarray(geometry.gram, dtype=np.float64)
    a = np.asarray(geometry.positive_coefficients, dtype=np.float64)
    beta = np.asarray(geometry.negative_coefficients, dtype=np.float64)
    if a.ndim != 1 or beta.ndim != 1 or h.shape != (len(a) + len(beta),) * 2:
        raise ValueError("geometry requires square Gram with one column per positive or negative atom")
    if not np.isfinite(radius_coefficient) or radius_coefficient <= 0:
        raise ValueError("radius_coefficient must be a declared positive finite value")
    b = np.concatenate([a, -beta])
    measured_gram = (h + h.T) / 2
    baseline_norm, radius = float("nan"), float("nan")
    diagnostics: dict[str, Any] = {"solver_config": asdict(config)}
    powers = np.full(len(a), np.nan)
    power_diagnostics: dict[str, Any] = {"attempted": False}
    phase = "geometry"
    try:
        if len(a) == 0:
            raise ValueError("NO_SUCCESSFUL_PLAYERS")
        if not np.isfinite(b).all() or np.any(a < 0) or np.any(beta < 0):
            raise ValueError("INVALID_ATOM_COEFFICIENTS")
        h, root, gram_diagnostics = _psd(h, config)
        diagnostics.update(gram_diagnostics)
        norms = np.sqrt(np.maximum(np.diag(h)[: len(a)], 0))
        if np.any(norms <= config.zero_norm_tolerance):
            raise ValueError("ZERO_REQUIRED_POSITIVE_FEATURE")
        if a.sum() <= 0:
            raise ValueError("ZERO_POSITIVE_BUDGET")
        baseline_norm = float(np.linalg.norm(root @ b))
        radius = radius_coefficient * baseline_norm
        if baseline_norm <= config.zero_norm_tolerance or not np.isfinite(radius) or radius <= 0:
            raise ValueError("ZERO_OR_INVALID_BASELINE_NORM")
        power_diagnostics["attempted"] = True
        phase = "powers"
        powers, power_details = collision_powers(h[: len(a), : len(a)], config)
        power_diagnostics.update(power_details)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        diagnostics["fallback_stage"] = phase
        power_diagnostics.update(fallback=True, reason=reason)
        return GroupSolution(
            powers,
            power_diagnostics,
            _fallback(h, b, a, beta, reason),
            _fallback(h, b, a, beta, reason),
            b,
            baseline_norm,
            radius,
            diagnostics,
        )
    routes = []
    for objective in ("linear", "ncr"):
        try:
            route = _route(h, root, b, a, beta, powers, radius, baseline_norm, objective, config)
            routes.append(_audit_measured_route(measured_gram, b, a, beta, route, radius_coefficient, config))
        except Exception as exc:
            routes.append(_fallback(h, b, a, beta, f"{type(exc).__name__}: {exc}", {"solve_attempted": True}))
    difference = routes[1].coefficients - routes[0].coefficients
    diagnostics["nash_linear_correction_difference"] = float(np.linalg.norm(root @ difference))
    return GroupSolution(powers, power_diagnostics, *routes, b, baseline_norm, radius, diagnostics)
