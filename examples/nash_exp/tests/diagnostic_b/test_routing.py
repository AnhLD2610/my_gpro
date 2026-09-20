# Copyright 2026 The nash_exp contributors
# SPDX-License-Identifier: Apache-2.0
"""Finite-dimensional fixtures; numerical constants here are not experiment settings."""

import inspect

import numpy as np
import pytest
from diagnostic_b.routing import (
    SolverConfig,
    TrainingGeometry,
    collision_powers,
    solve_group,
    standardized_advantages,
)


def witness():
    """Appendix E fresh G=5 construction, with four segments per failure."""
    vectors = np.array([[1, 0, 0.6], [0, 1, 0], [0, 0, 0.8]])
    q = np.array([25, 38, 25]) / 88
    positives = vectors * (3 * q)
    positive_mean = positives.mean(axis=1)
    # Failed root score -6*pbar and three signed internal segment scores.
    internal = 100 * np.eye(3)
    negatives = np.column_stack([-6 * positive_mean, internal, -6 * positive_mean, -internal])
    atoms = np.column_stack([positives, negatives])
    denominator = np.sqrt(0.24) + 1e-6  # Synthetic epsilon fixture.
    a = np.full(3, 0.08 / denominator)
    beta = np.full(8, 0.03 / denominator)
    return atoms, TrainingGeometry(atoms.T @ atoms, a, beta)


def test_advantages_population_epsilon_and_homogeneous():
    rewards = np.array([1, 0, 0])
    epsilon = 0.3
    expected = (rewards - 1 / 3) / (np.sqrt(2 / 9) + epsilon)
    np.testing.assert_allclose(standardized_advantages(rewards, epsilon), expected)
    np.testing.assert_array_equal(standardized_advantages([1, 1], epsilon), [0, 0])
    np.testing.assert_array_equal(standardized_advantages([0, 0], epsilon), [0, 0])
    with pytest.raises(ValueError):
        standardized_advantages(rewards, 0)


@pytest.mark.parametrize("m", [1, 2])
def test_special_powers(m):
    q, diagnostics = collision_powers(np.eye(m))
    np.testing.assert_allclose(q, np.ones(m) / m)
    assert diagnostics["first_stage_gap"] == 0


def test_identical_directions_lexicographic_uniform():
    q, diagnostics = collision_powers(np.ones((5, 5)))
    np.testing.assert_allclose(q, np.full(5, 0.2), atol=2e-6)
    assert diagnostics["numerical_rank"] == 1
    assert diagnostics["closest_uniform_objective"] < 1e-10


def test_orthogonal_multiplicities_and_ratio_cap():
    vectors = np.array([[1, 1, 1, 1, 0], [0, 0, 0, 0, 1]])
    q, diagnostics = collision_powers(vectors.T @ vectors)
    np.testing.assert_allclose(q, [0.125, 0.125, 0.125, 0.125, 0.5], atol=2e-5)
    assert diagnostics["collision"] == pytest.approx(5 / 8, abs=1e-7)
    assert q.max() <= (len(q) - 1) * q.min() + 1e-7
    assert q.min() >= 1 / (1 + (len(q) - 1) ** 2) - 1e-7
    assert diagnostics["final_stage_gap"] < 2e-7


def test_ratio_cap_preserves_third_direction():
    vectors = np.array([[0.5, 0.5, 1], [np.sqrt(3) / 2, -np.sqrt(3) / 2, 0]])
    q, _ = collision_powers(vectors.T @ vectors)
    np.testing.assert_allclose(q, [0.4, 0.4, 0.2], atol=2e-6)


def test_near_duplicates_separate_power_behavior_from_collision():
    vectors = np.array([[1, 1, np.cos(0.1)], [0, 0, np.sin(0.1)]])
    q, _ = collision_powers(vectors.T @ vectors)
    np.testing.assert_allclose(q, [0.25, 0.25, 0.5], atol=2e-5)
    coincident, _ = collision_powers(np.ones((3, 3)))
    np.testing.assert_allclose(coincident, np.ones(3) / 3, atol=2e-6)


def test_witness_powers_strict_nash_linear_and_constraints():
    atoms, geometry = witness()
    config = SolverConfig()
    result = solve_group(geometry, radius_coefficient=0.5, config=config)
    np.testing.assert_allclose(result.powers, np.array([25, 38, 25]) / 88, atol=2e-6)
    assert not result.linear.fallback, result.linear.reason
    assert not result.ncr.fallback, result.ncr.reason
    difference = np.linalg.norm(atoms @ (result.ncr.coefficients - result.linear.coefficients))
    assert difference > 1e-4 * result.radius
    baseline = atoms @ result.baseline_coefficients
    normp = np.linalg.norm(atoms[:, :3], axis=0)
    for route in [result.linear, result.ncr]:
        delta = atoms @ route.coefficients
        assert abs(route.positive_shares.sum() - 1) < config.feasibility_tolerance
        assert route.positive_shares.min() >= -config.feasibility_tolerance
        assert route.refunds.min() >= -config.feasibility_tolerance
        assert route.refunds.max() <= 1 + config.feasibility_tolerance
        assert np.linalg.norm(delta) <= result.radius * (1 + config.feasibility_tolerance)
        assert baseline @ delta >= -config.feasibility_tolerance
        assert np.min(atoms[:, :3].T @ delta) >= -config.feasibility_tolerance
        np.testing.assert_allclose(route.support, atoms[:, :3].T @ (baseline + delta) / normp, atol=1e-7)
        np.testing.assert_allclose(route.added_support, atoms[:, :3].T @ delta / normp, atol=1e-7)
        assert np.min(route.utilities) >= 1 - config.feasibility_tolerance
        assert route.diagnostics["objective_gap"] <= config.objective_tolerance + 1e-8


def test_scalar_linear_tie_break_does_not_freeze_arbitrary_utilities():
    # Maximal x refund is fixed; y refund is arbitrary for welfare, so min-norm sets y=0.
    atoms = np.array([[1, 1, 0], [0, 0, 1]], dtype=float)
    geometry = TrainingGeometry(atoms.T @ atoms, np.ones(1), np.array([0.2, 0.2]))
    result = solve_group(geometry, radius_coefficient=1)
    assert not result.linear.fallback, result.linear.reason
    delta = atoms @ result.linear.coefficients
    np.testing.assert_allclose(delta, [0.2, 0], atol=1e-4)


def test_baseline_feasible_and_zero_feature_group_fallback():
    _, geometry = witness()
    a = geometry.positive_coefficients
    zero = np.concatenate([a.sum() * (a / a.sum() - a / a.sum()), np.zeros(8)])
    np.testing.assert_array_equal(zero, np.zeros(11))
    gram = geometry.gram.copy()
    gram[0, :] = gram[:, 0] = 0
    result = solve_group(TrainingGeometry(gram, a, geometry.negative_coefficients), 0.5)
    assert result.linear.fallback and result.ncr.fallback
    assert "ZERO_REQUIRED_POSITIVE_FEATURE" in result.ncr.reason
    assert not result.power_diagnostics["attempted"]
    assert not result.ncr.diagnostics["solve_attempted"]
    np.testing.assert_array_equal(result.linear.coefficients, np.zeros(11))
    assert len(result.linear.support) == 3  # Never silently remove the protected player.


def test_materially_indefinite_gram_falls_back_without_psd_projection():
    result = solve_group(TrainingGeometry(np.array([[1, 2], [2, 1]]), np.ones(1), np.ones(1)), 0.5)
    assert result.linear.fallback and result.ncr.fallback
    assert "MATERIALLY_INDEFINITE_GRAM" in result.linear.reason


def test_materially_asymmetric_gram_is_rejected_before_averaging():
    gram = np.array([[1.0, 0.2], [0.8, 1.0]])
    result = solve_group(TrainingGeometry(gram, np.ones(1), np.ones(1)), 0.5)
    assert result.linear.fallback and result.ncr.fallback
    assert "MATERIALLY_ASYMMETRIC_GRAM" in result.linear.reason
    assert result.diagnostics["fallback_stage"] == "geometry"
    assert not result.power_diagnostics["attempted"]


def test_roundoff_asymmetry_is_recorded():
    gram = np.array([[1.0, 0.5 + 1e-12], [0.5, 1.0]])
    powers, diagnostics = collision_powers(gram)
    np.testing.assert_array_equal(powers, [0.5, 0.5])
    assert diagnostics["gram_symmetry_residual"] == pytest.approx(1e-12, abs=1e-15)


def test_admissible_eigenvalue_repair_still_audits_original_geometry():
    gram = np.array([[1.0, 1.0 + 1e-9], [1.0 + 1e-9, 1.0]])
    geometry = TrainingGeometry(gram, np.ones(1), np.array([0.1]))
    result = solve_group(geometry, 0.5)
    assert result.diagnostics["gram_min_eigenvalue"] < 0
    assert result.diagnostics["gram_roundoff_repair_norm"] > 0
    for route in (result.linear, result.ncr):
        assert not route.fallback, route.reason
        assert route.diagnostics["measured_geometry_audited"]
        assert route.diagnostics["measured_relative_radius_residual"] <= SolverConfig().feasibility_tolerance
        assert route.diagnostics["measured_max_support_shift"] < 1e-8
        np.testing.assert_allclose(
            route.diagnostics["measured_protection_margins"], gram[:1] @ route.coefficients, atol=1e-15
        )


@pytest.mark.parametrize("residual_mass", [1e-5, 1e-4])
def test_tiny_psd_repair_cannot_hide_invalid_measured_radius_after_cancellation(residual_mass):
    # The Gram's tiny negative eigenvalue is admissible roundoff. However, its
    # baseline nearly cancels, so repair inflates a small measured radius (or
    # creates one from a nonpositive squared norm). That repaired radius must
    # not authorize reporting a route violating the measured radius.
    gram = np.array([[1.0, 1.0 + 1e-9], [1.0 + 1e-9, 1.0]])
    geometry = TrainingGeometry(gram, np.ones(1), np.array([1 - residual_mass]))
    result = solve_group(geometry, 0.5)
    assert result.diagnostics["gram_min_eigenvalue"] < 0
    assert result.baseline_norm > SolverConfig().zero_norm_tolerance
    assert result.linear.reason == "INVALID_MEASURED_GEOMETRY_FEASIBILITY"
    if residual_mass == 1e-5:
        assert result.linear.diagnostics["measured_baseline_norm_squared"] < 0
    else:
        assert result.linear.diagnostics["measured_baseline_norm_squared"] > 0
        assert result.linear.diagnostics["measured_relative_radius_residual"] > SolverConfig().feasibility_tolerance
    for route in (result.linear, result.ncr):
        assert route.fallback
        assert route.diagnostics["solve_attempted"]
        np.testing.assert_array_equal(route.coefficients, [0, 0])


def test_power_failure_uses_matched_baseline_fallbacks():
    _, geometry = witness()
    result = solve_group(geometry, 0.5, SolverConfig(solver="UNAVAILABLE_TEST_SOLVER"))
    assert result.linear.fallback and result.ncr.fallback
    np.testing.assert_array_equal(result.linear.coefficients, result.ncr.coefficients)
    np.testing.assert_array_equal(result.linear.support, result.ncr.support)
    assert result.power_diagnostics["attempted"] and result.power_diagnostics["fallback"]
    assert not result.ncr.diagnostics["solve_attempted"]


def test_training_api_cannot_accept_heldout_and_no_leakage():
    assert set(inspect.signature(TrainingGeometry).parameters) == {
        "gram",
        "positive_coefficients",
        "negative_coefficients",
    }
    atoms, geometry = witness()
    route = solve_group(geometry, 0.5)
    first_coefficients = route.ncr.coefficients.copy()
    first_powers = route.powers.copy()
    unit_successes = atoms[:, :3] / np.linalg.norm(atoms[:, :3], axis=0)
    hq_a, hq_b = np.array([1, 2, 3]), np.array([-8, 0, 4])
    assert not np.allclose(unit_successes.T @ hq_a, unit_successes.T @ hq_b)
    np.testing.assert_array_equal(route.ncr.coefficients, first_coefficients)
    np.testing.assert_array_equal(route.powers, first_powers)
    with pytest.raises(TypeError):
        solve_group(geometry, 0.5, heldout=hq_a)
