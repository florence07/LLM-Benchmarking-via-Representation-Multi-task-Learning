"""Simulation routines for the unequal-domain 2PL IRT comparison."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import kendalltau, spearmanr


def sine_distance(theta: np.ndarray, eta: np.ndarray) -> float:
    """Return sin(angle(theta, eta)), with the sign-invariant paper definition."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    eta = np.asarray(eta, dtype=float).reshape(-1)
    norm_theta = np.linalg.norm(theta)
    norm_eta = np.linalg.norm(eta)
    if norm_theta < 1e-14 or norm_eta < 1e-14:
        raise ValueError("Both vectors must be nonzero.")
    cosine = float(theta @ eta) / (norm_theta * norm_eta)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))


def normalized_theta_error(theta_hat: np.ndarray, theta_star: np.ndarray) -> float:
    """Return ||theta_hat-theta_star||_2/sqrt(N), as used in the paper."""
    theta_hat = np.asarray(theta_hat, dtype=float).reshape(-1)
    theta_star = np.asarray(theta_star, dtype=float).reshape(-1)
    if theta_hat.shape != theta_star.shape:
        raise ValueError("theta_hat and theta_star must have the same shape.")
    return float(np.linalg.norm(theta_hat - theta_star) / np.sqrt(theta_hat.size))


def ranking_correlations(
    estimated_scores: np.ndarray,
    true_scores: np.ndarray,
) -> tuple[float, float]:
    """Return Spearman's rho and Kendall's tau-b for two ranking vectors."""
    estimated_scores = np.asarray(estimated_scores, dtype=float).reshape(-1)
    true_scores = np.asarray(true_scores, dtype=float).reshape(-1)
    if estimated_scores.shape != true_scores.shape:
        raise ValueError("The two score vectors must have the same shape.")
    rho = float(spearmanr(estimated_scores, true_scores).statistic)
    tau = float(kendalltau(estimated_scores, true_scores, variant="b").statistic)
    return rho, tau


def _project_centered_sphere(vector: np.ndarray) -> np.ndarray:
    """Project onto {theta: 1_N^T theta=0, ||theta||_2=sqrt(N)}."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    centered = vector - vector.mean()
    norm = np.linalg.norm(centered)
    if norm < 1e-14:
        centered = np.linspace(-1.0, 1.0, vector.size)
        centered -= centered.mean()
        norm = np.linalg.norm(centered)
    return np.sqrt(vector.size) * centered / norm


def _centered_clipped_vector(
    vector: np.ndarray,
    scale: float,
    bound: float,
) -> np.ndarray:
    """For fixed scale, shift before clipping so that the result sums to zero."""
    lower = float(vector.min() - bound / scale)
    upper = float(vector.max() + bound / scale)
    for _ in range(70):
        shift = 0.5 * (lower + upper)
        candidate = np.clip(scale * (vector - shift), -bound, bound)
        if candidate.sum() > 0.0:
            lower = shift
        else:
            upper = shift
    return np.clip(scale * (vector - 0.5 * (lower + upper)), -bound, bound)


def _project_centered_sphere_box(
    vector: np.ndarray,
    bound: float | None,
) -> np.ndarray:
    """Enforce centering, norm sqrt(N), and an optional entrywise bound.

    The common fast path is an exact centering-and-normalization.  If that
    result violates the box, nested bisection finds a shifted, clipped vector
    with zero sum and norm sqrt(N).
    """
    vector = np.asarray(vector, dtype=float).reshape(-1)
    projected = _project_centered_sphere(vector)
    if bound is None or np.max(np.abs(projected)) <= bound + 1e-12:
        return projected
    minimum_feasible_bound = (
        1.0
        if vector.size % 2 == 0
        else float(np.sqrt(vector.size / (vector.size - 1.0)))
    )
    if bound < minimum_feasible_bound - 1e-12:
        raise ValueError(
            "theta_bound is too small for a zero-mean vector with norm sqrt(N). "
            f"For N={vector.size}, require theta_bound >= "
            f"{minimum_feasible_bound:.6f}."
        )

    target_norm = np.sqrt(vector.size)
    lower_scale = 0.0
    upper_scale = 1.0
    candidate = _centered_clipped_vector(vector, upper_scale, bound)
    while np.linalg.norm(candidate) < target_norm:
        upper_scale *= 2.0
        if upper_scale > 1e16:
            raise FloatingPointError("Could not enforce the theta constraints.")
        candidate = _centered_clipped_vector(vector, upper_scale, bound)

    for _ in range(70):
        middle_scale = 0.5 * (lower_scale + upper_scale)
        candidate = _centered_clipped_vector(vector, middle_scale, bound)
        if np.linalg.norm(candidate) < target_norm:
            lower_scale = middle_scale
        else:
            upper_scale = middle_scale
    projected = _centered_clipped_vector(vector, upper_scale, bound)

    if abs(projected.sum()) > 1e-8:
        raise FloatingPointError("The theta projection failed to centre the vector.")
    if abs(np.linalg.norm(projected) - target_norm) > 1e-8:
        raise FloatingPointError("The theta projection failed to enforce the norm.")
    if np.max(np.abs(projected)) > bound + 1e-10:
        raise FloatingPointError("The theta projection failed to enforce the box.")
    return projected


def _project_theta_rows(
    theta: np.ndarray,
    theta_bound: float | None,
) -> np.ndarray:
    """Apply the theta projection to every row of a T-by-N array."""
    theta = np.asarray(theta, dtype=float)
    return np.vstack(
        [_project_centered_sphere_box(row, theta_bound) for row in theta]
    )


def _orient_general_trait(theta_G: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Enforce theta_G^T sum_t theta_t >= 0 without changing its subspace."""
    theta_sum = np.asarray(theta, dtype=float).sum(axis=0)
    if float(theta_G @ theta_sum) < 0.0:
        return -theta_G
    return theta_G


def _global_perturbed_theta_initialization(
    global_theta: np.ndarray,
    T: int,
    perturbation: float,
    *,
    theta_bound: float | None,
    seed: int,
) -> np.ndarray:
    """Create T feasible Proposed theta initializers around fitted Global.

    For each domain t,

        theta_t^(0)
          = sqrt(1-delta^2) * theta_Global
            + sqrt(N) * delta * v_t,

    where v_t has unit norm and is orthogonal to both 1_N and theta_Global.
    Hence every initializer is centred, has norm sqrt(N), and satisfies

        sin(theta_t^(0), theta_Global) = delta

    exactly (up to floating-point error).

    The T perturbation directions are orthonormal, so domains do not all start
    from the same perturbed direction.  A fixed seed makes the construction
    reproducible.
    """
    global_theta = np.asarray(global_theta, dtype=float).reshape(-1)
    N = global_theta.size

    if not 0.0 <= perturbation < 1.0:
        raise ValueError("proposed_theta_perturbation must lie in [0, 1).")
    if T > N - 2 and perturbation > 0.0:
        raise ValueError(
            "Need T <= N-2 to construct orthonormal perturbation directions."
        )

    global_theta = _project_centered_sphere_box(global_theta, theta_bound)
    if perturbation == 0.0:
        return np.repeat(global_theta[None, :], T, axis=0)

    rng = np.random.default_rng(seed)

    for _ in range(200):
        directions = _orthonormal_tangent_directions(global_theta, T, rng)
        theta_init = (
            np.sqrt(1.0 - perturbation**2) * global_theta[None, :]
            + np.sqrt(N) * perturbation * directions
        )
        if theta_bound is None or (
            np.max(np.abs(theta_init)) <= theta_bound + 1e-10
        ):
            return theta_init

    raise RuntimeError(
        "Could not construct bounded Proposed perturbation initializers. "
        "Reduce proposed_theta_perturbation or increase theta_bound."
    )


def _validate_J_vector(J: int | Sequence[int], T: int) -> np.ndarray:
    """Return an integer vector (J_1,...,J_T) and validate its dimensions."""
    if isinstance(J, (int, np.integer)):
        J_vec = np.full(T, int(J), dtype=int)
    else:
        raw = np.asarray(J)
        if raw.ndim != 1 or raw.size != T:
            raise ValueError("J must be an integer or a length-T vector.")
        if not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError("Every J_t must be an integer.")
        J_vec = raw.astype(int)
    if np.any(J_vec < 1):
        raise ValueError("Every J_t must be at least 1.")
    return J_vec


def _as_domain_blocks(
    Y: np.ndarray | Sequence[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Validate responses and return true-size N-by-J_t domain blocks."""
    if isinstance(Y, np.ndarray):
        array = np.asarray(Y, dtype=float)
        if array.ndim != 3:
            raise ValueError("A NumPy Y input must have shape (T, N, J).")
        blocks = [array[t] for t in range(array.shape[0])]
    else:
        blocks = [np.asarray(block, dtype=float) for block in Y]
    if not blocks:
        raise ValueError("At least one domain is required.")
    if any(block.ndim != 2 for block in blocks):
        raise ValueError("Each domain response block must have shape (N, J_t).")
    N_values = {block.shape[0] for block in blocks}
    if len(N_values) != 1:
        raise ValueError("All domains must contain the same N subjects.")
    N = N_values.pop()
    if N < 3 or any(block.shape[1] < 1 for block in blocks):
        raise ValueError("Require N >= 3 and every J_t >= 1.")
    if any(not np.isin(block, [0.0, 1.0]).all() for block in blocks):
        raise ValueError("Y must contain only zeros and ones.")

    J_vec = np.asarray([block.shape[1] for block in blocks], dtype=int)
    return blocks, J_vec


def _prepare_observation_masks(
    Y: Sequence[np.ndarray],
    observation_mask: np.ndarray | Sequence[np.ndarray] | None,
) -> list[np.ndarray]:
    """Validate true-size masks without materializing structural padding."""
    if observation_mask is None:
        masks = [np.ones_like(block, dtype=bool) for block in Y]
    elif isinstance(observation_mask, np.ndarray):
        raw = np.asarray(observation_mask)
        if raw.ndim != 3 or raw.shape[:2] != (len(Y), Y[0].shape[0]):
            raise ValueError(
                "A NumPy observation_mask must have shape (T, N, J_max)."
            )
        if raw.shape[2] < max(block.shape[1] for block in Y):
            raise ValueError("observation_mask has too few item columns.")
        masks = [raw[t, :, : block.shape[1]] for t, block in enumerate(Y)]
    else:
        masks = [np.asarray(mask) for mask in observation_mask]
        if len(masks) != len(Y):
            raise ValueError("observation_mask must contain one block per domain.")

    validated: list[np.ndarray] = []
    for t, (block, raw_mask) in enumerate(zip(Y, masks)):
        if raw_mask.shape != block.shape:
            raise ValueError(
                f"observation_mask for domain {t + 1} must have shape "
                f"{block.shape}."
            )
        if not np.isin(raw_mask, [0, 1, False, True]).all():
            raise ValueError("observation_mask must contain only 0/1 values.")
        mask = raw_mask.astype(bool, copy=True)
        if not np.any(mask):
            raise ValueError("Every domain must contain an observed response.")
        validated.append(mask)
    return validated


def _pad_parameter_blocks(
    blocks: Sequence[np.ndarray],
) -> np.ndarray:
    """Pad item parameters only once for backwards-compatible result output."""
    J_max = max(block.size for block in blocks)
    padded = np.zeros((len(blocks), J_max), dtype=float)
    for t, block in enumerate(blocks):
        padded[t, : block.size] = block
    return padded


def _sine_objective_and_gradient(
    theta_G: np.ndarray,
    theta: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return sum_t sin(theta_G, theta_t) and its exact theta_G gradient.

    The sine distance is sqrt(1-cos^2).  At exact alignment the function is
    nondifferentiable; we use the zero subgradient there.  No smoothing is
    applied away from exact alignment.
    """
    norm_G = np.linalg.norm(theta_G)
    norms = np.linalg.norm(theta, axis=1)
    unit_G = theta_G / norm_G
    unit_theta = theta / norms[:, None]
    cosine = np.clip(unit_theta @ unit_G, -1.0, 1.0)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    objective = float(np.sum(sine))

    multiplier = np.zeros_like(cosine)
    nonzero = sine > 1e-12
    multiplier[nonzero] = -cosine[nonzero] / sine[nonzero]
    gradient = np.sum(
        multiplier[:, None]
        * (unit_theta - cosine[:, None] * unit_G[None, :])
        / norm_G,
        axis=0,
    )
    return objective, gradient


@dataclass
class _AdamState:
    first_moment: np.ndarray
    second_moment: np.ndarray


def _new_adam_state(parameter: np.ndarray) -> _AdamState:
    return _AdamState(np.zeros_like(parameter), np.zeros_like(parameter))


def _adam_step(
    parameter: np.ndarray,
    gradient: np.ndarray,
    state: _AdamState,
    iteration: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
) -> np.ndarray:
    state.first_moment *= beta1
    state.first_moment += (1.0 - beta1) * gradient
    state.second_moment *= beta2
    state.second_moment += (1.0 - beta2) * gradient * gradient
    first_unbiased = state.first_moment / (1.0 - beta1**iteration)
    second_unbiased = state.second_moment / (1.0 - beta2**iteration)
    return parameter - learning_rate * first_unbiased / (
        np.sqrt(second_unbiased) + adam_eps
    )


def _decayed_learning_rate(
    initial_learning_rate: float,
    iteration: int,
    decay_factor: float,
    decay_steps: int,
) -> float:
    """Piecewise-exponential learning-rate decay.

    The rate is multiplied by ``decay_factor`` every ``decay_steps`` Adam
    iterations.  With the default fit settings (0.02, 0.5, 200), the learning
    rate is 0.02 for iterations 1--200, 0.01 for 201--400, 0.005 for 401--600,
    0.0025 for 601--800, and 0.00125 afterwards.
    """
    exponent = (iteration - 1) // decay_steps
    return float(initial_learning_rate * (decay_factor ** exponent))


def _initial_general_trait(
    theta: np.ndarray,
    initial_theta_G: np.ndarray | None = None,
) -> np.ndarray:
    """Return a feasible G-factor initializer.

    By default, stack the domain-trait estimates into the T-by-N matrix
    ``theta`` and use its leading right singular vector as the initializer.
    This is the data-driven initialization used throughout the simulation.
    ``initial_theta_G`` is retained only as a generic API override and is not
    supplied by the simulation driver.
    """
    theta = np.asarray(theta, dtype=float)
    if initial_theta_G is None:
        _, _, right_transpose = np.linalg.svd(theta, full_matrices=False)
        raw_initial = right_transpose[0]
    else:
        raw_initial = np.asarray(initial_theta_G, dtype=float).reshape(-1)
        if raw_initial.shape != (theta.shape[1],):
            raise ValueError(
                "initial_theta_G must have shape (N,), matching theta."
            )
        if not np.all(np.isfinite(raw_initial)):
            raise ValueError("initial_theta_G must contain only finite values.")
        if np.linalg.norm(raw_initial - raw_initial.mean()) < 1e-14:
            raise ValueError(
                "initial_theta_G must have a nonzero centred component."
            )
    theta_G = _project_centered_sphere(raw_initial)
    return _orient_general_trait(theta_G, theta)


def estimate_general_trait(
    theta: np.ndarray,
    *,
    initial_theta_G: np.ndarray | None = None,
    learning_rate: float = 0.03,
    learning_rate_decay_factor: float = 0.5,
    learning_rate_decay_steps: int = 200,
    max_iter: int = 1000,
    tolerance: float = 1e-7,
) -> dict:
    """Minimize sum_t sin(theta_G, theta_t) on the centred sphere.

    The unsmoothed sine objective is optimized directly.  A zero subgradient
    is used only at exact alignment, where the sine distance is
    nondifferentiable.  The Adam learning rate is halved every 200 iterations
    by default, and convergence is declared after one relative-objective-change
    check below ``tolerance``.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[0] < 1 or theta.shape[1] < 3:
        raise ValueError("theta must have shape (T, N), with N >= 3.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 < learning_rate_decay_factor <= 1.0:
        raise ValueError("learning_rate_decay_factor must lie in (0, 1].")
    if learning_rate_decay_steps < 1:
        raise ValueError("learning_rate_decay_steps must be positive.")
    if max_iter < 1 or tolerance <= 0.0:
        raise ValueError("max_iter and tolerance must be positive.")

    theta_G = _initial_general_trait(theta, initial_theta_G)
    state = _new_adam_state(theta_G)
    best_objective = np.inf
    best_theta_G = theta_G.copy()
    previous_objective = np.inf
    converged = False

    for iteration in range(1, max_iter + 1):
        objective, gradient = _sine_objective_and_gradient(theta_G, theta)
        if objective < best_objective:
            best_objective = objective
            best_theta_G = theta_G.copy()

        if objective <= tolerance:
            converged = True
            break

        relative_change = (
            np.inf
            if not np.isfinite(previous_objective)
            else abs(objective - previous_objective)
            / max(1.0, abs(previous_objective))
        )
        if relative_change < tolerance:
            converged = True
            break
        previous_objective = objective

        current_learning_rate = _decayed_learning_rate(
            learning_rate,
            iteration,
            learning_rate_decay_factor,
            learning_rate_decay_steps,
        )
        theta_G = _adam_step(
            theta_G,
            gradient / theta.shape[0],
            state,
            iteration,
            current_learning_rate,
            0.9,
            0.999,
            1e-8,
        )
        theta_G = _project_centered_sphere(theta_G)
        theta_G = _orient_general_trait(theta_G, theta)

    exact_objective = float(
        sum(sine_distance(best_theta_G, theta_t) for theta_t in theta)
    )
    return {
        "theta_G_hat": _orient_general_trait(best_theta_G, theta),
        "exact_objective": exact_objective,
        "iterations": int(iteration),
        "converged": bool(converged),
    }


def _orthonormal_tangent_directions(
    theta_G: np.ndarray,
    number: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate orthonormal directions orthogonal to 1_N and theta_G."""
    N = theta_G.size
    if number > N - 2:
        raise ValueError("The number of tangent directions cannot exceed N-2.")
    unit_G = theta_G / np.linalg.norm(theta_G)
    raw = rng.normal(size=(N, number))
    raw -= raw.mean(axis=0, keepdims=True)
    raw -= unit_G[:, None] * (unit_G @ raw)[None, :]
    Q, _ = np.linalg.qr(raw, mode="reduced")
    Q -= Q.mean(axis=0, keepdims=True)
    Q -= unit_G[:, None] * (unit_G @ Q)[None, :]
    Q, _ = np.linalg.qr(Q, mode="reduced")
    return Q.T


def _generate_domain_traits(
    theta_G: np.ndarray,
    T: int,
    h: float,
    structure: str,
    minimum_relative_angle: float,
    majority_proportion: float,
    theta_bound: float,
    shared_h_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate theta_t around theta_G while preserving all constraints."""
    N = theta_G.size
    angle_max = float(np.arcsin(h))

    if structure == "random_alpha":
        relative_alpha = rng.uniform(-1.0, 1.0, size=T)
        relative_alpha[0] = 1.0
        alpha = h * relative_alpha
        directions = _orthonormal_tangent_directions(theta_G, T, rng)
        theta = (
            np.sqrt(np.maximum(0.0, 1.0 - alpha**2))[:, None]
            * theta_G[None, :]
            + np.sqrt(N) * alpha[:, None] * directions
        )
        validation_max_abs = 0.0
        for shared_h in shared_h_values:
            shared_alpha = float(shared_h) * relative_alpha
            shared_theta = (
                np.sqrt(np.maximum(0.0, 1.0 - shared_alpha**2))[:, None]
                * theta_G[None, :]
                + np.sqrt(N) * shared_alpha[:, None] * directions
            )
            validation_max_abs = max(
                validation_max_abs,
                float(np.max(np.abs(shared_theta))),
            )
        domain_angles = np.arcsin(np.abs(alpha)).tolist()
        metadata = {
            "common_trait_unique_by_construction": False,
            "alpha": alpha.copy(),
            "relative_alpha": relative_alpha.copy(),
            "domain_directions": directions.copy(),
            "validation_max_abs": validation_max_abs,
        }

    elif structure == "majority_two_centers":
        if not 0.5 < majority_proportion <= 1.0:
            raise ValueError(
                "majority_proportion must lie in (0.5, 1] for "
                "majority_two_centers."
            )

        number_majority = int(np.ceil(majority_proportion * T - 1e-12))
        if number_majority <= T // 2:
            number_majority = T // 2 + 1
        number_shifted = T - number_majority

        v0 = _orthonormal_tangent_directions(theta_G, 1, rng)[0]

        validation_max_abs = float(np.max(np.abs(theta_G)))
        for shared_h in shared_h_values:
            shared_h = float(shared_h)
            theta_G_prime_shared = (
                np.sqrt(max(0.0, 1.0 - shared_h**2)) * theta_G
                + np.sqrt(N) * shared_h * v0
            )
            validation_max_abs = max(
                validation_max_abs,
                float(np.max(np.abs(theta_G_prime_shared))),
            )

        if validation_max_abs > theta_bound + 1e-10:
            raise RuntimeError(
                "The majority_two_centers design exceeds theta_bound for at "
                "least one shared h value."
            )

        theta_G_prime = (
            np.sqrt(max(0.0, 1.0 - h**2)) * theta_G
            + np.sqrt(N) * h * v0
        )

        majority_indicator = np.array(
            [True] * number_majority + [False] * number_shifted,
            dtype=bool,
        )
        majority_indicator = majority_indicator[rng.permutation(T)]

        theta = np.vstack(
            [
                theta_G.copy() if is_majority else theta_G_prime.copy()
                for is_majority in majority_indicator
            ]
        )
        domain_angles = np.where(
            majority_indicator,
            0.0,
            float(np.arcsin(h)),
        ).tolist()
        alpha = np.where(majority_indicator, 0.0, h).astype(float)

        metadata = {
            "common_trait_unique_by_construction": True,
            "majority_proportion_target": float(majority_proportion),
            "majority_proportion_realized": float(number_majority / T),
            "number_majority_domains": int(number_majority),
            "number_shifted_domains": int(number_shifted),
            "majority_indicator": majority_indicator.copy(),
            "v0": v0.copy(),
            "theta_G_prime": theta_G_prime.copy(),
            "shifted_center_sine_distance": float(
                sine_distance(theta_G, theta_G_prime)
            ),
            "alpha": alpha,
            "validation_max_abs": validation_max_abs,
        }

    elif structure == "paired_directions":
        if T < 4 and h > 0.0:
            raise ValueError(
                "paired_directions requires T >= 4 when h > 0; with only one "
                "pair, the constructed theta_G need not minimize the sine objective."
            )
        number_pairs = T // 2
        number_directions = number_pairs
        directions = _orthonormal_tangent_directions(
            theta_G, number_directions, rng
        )
        if number_pairs:
            relative = np.linspace(minimum_relative_angle, 1.0, number_pairs)
            angles = angle_max * relative
        else:
            angles = np.empty(0)
        rows: list[np.ndarray] = []
        domain_angles: list[float] = []
        for direction, angle in zip(directions, angles):
            rows.append(
                np.cos(angle) * theta_G + np.sqrt(N) * np.sin(angle) * direction
            )
            rows.append(
                np.cos(angle) * theta_G - np.sqrt(N) * np.sin(angle) * direction
            )
            domain_angles.extend([float(angle), float(angle)])
        if T % 2 == 1:
            rows.append(theta_G.copy())
            domain_angles.append(0.0)
        theta = np.vstack(rows)
        metadata = {
            "common_trait_unique_by_construction": False,
            "number_paired_directions": number_pairs,
        }

    elif structure == "strict_majority":
        if T < 3 and h > 0.0:
            raise ValueError(
                "strict_majority requires T >= 3 to realize a positive h."
            )
        number_at_center = T // 2 + 1
        number_away = T - number_at_center
        rows = [theta_G.copy() for _ in range(number_at_center)]
        domain_angles = [0.0] * number_at_center
        if number_away:
            directions = _orthonormal_tangent_directions(
                theta_G, number_away, rng
            )
            relative = np.linspace(minimum_relative_angle, 1.0, number_away)
            for index, (direction, relative_angle) in enumerate(
                zip(directions, relative)
            ):
                signed_direction = direction if index % 2 == 0 else -direction
                angle = angle_max * relative_angle
                rows.append(
                    np.cos(angle) * theta_G
                    + np.sqrt(N) * np.sin(angle) * signed_direction
                )
                domain_angles.append(float(angle))
        theta = np.vstack(rows)
        metadata = {
            "common_trait_unique_by_construction": True,
            "number_domains_at_center": number_at_center,
        }

    elif structure == "single_arc":
        direction = _orthonormal_tangent_directions(theta_G, 1, rng)[0]
        signed_angles = np.linspace(-angle_max, angle_max, T)
        theta = np.vstack(
            [
                np.cos(angle) * theta_G
                + np.sqrt(N) * np.sin(angle) * direction
                for angle in signed_angles
            ]
        )
        domain_angles = signed_angles.tolist()
        metadata = {
            "common_trait_unique_by_construction": False,
            "single_arc_even_T_warning": bool(T % 2 == 0 and h > 0.0),
        }
    else:
        raise ValueError(
            "structure must be 'random_alpha', 'majority_two_centers', "
            "'paired_directions', 'strict_majority', or 'single_arc'."
        )

    permutation = (
        np.arange(T, dtype=int)
        if structure in {"random_alpha", "majority_two_centers"}
        else rng.permutation(T)
    )
    theta = theta[permutation]
    domain_angles_array = np.asarray(domain_angles, dtype=float)[permutation]
    max_abs = float(
        metadata.get("validation_max_abs", np.max(np.abs(theta)))
    )
    if max_abs > theta_bound + 1e-10:
        raise RuntimeError(
            f"Generated max |theta_it|={max_abs:.3f} exceeds theta_bound="
            f"{theta_bound:.3f}. Increase theta_bound or reduce h."
        )
    return theta, domain_angles_array, metadata


def generate_ai_measurement_data(
    N: int = 100,
    T: int = 20,
    J: int | Sequence[int] = 50,
    h: float = 0.20,
    score_power: float = 1.0,
    discrimination_low: float = 0.50,
    discrimination_high: float = 1.50,
    easiness_low: float = -1.00,
    easiness_high: float = 1.00,
    heterogeneity_structure: str = "random_alpha",
    minimum_relative_angle: float = 0.35,
    majority_proportion: float = 0.60,
    theta_bound: float = 4.0,
    shuffle_ai_labels: bool = True,
    population_common_trait_max_iter: int = 3000,
    population_common_trait_tolerance: float = 1e-9,
    shared_h_values: Sequence[float] | None = None,
    seed: int | None = 20260720,
) -> dict:
    """Generate one complete-data multi-domain 2PL IRT dataset.

    A deterministic centred grid is used as the generation centre.
    ``score_power`` changes the spacing of its adjacent ranks without violating
    the required centering and scale normalization.  After the domain-specific
    true traits have been generated, the population sine optimization is
    initialized at that generation centre and numerically minimized.  Its
    minimizer, not the initial centre itself, is stored as the true
    ``theta_G_star`` used in all evaluations.
    """
    if N < 4:
        raise ValueError("N must be at least 4.")
    if T < 2:
        raise ValueError("T must be at least 2.")
    J_vec = _validate_J_vector(J, T)
    if not 0.0 <= h < 1.0:
        raise ValueError("h must lie in [0, 1).")
    if shared_h_values is None:
        shared_h_array = np.asarray([h], dtype=float)
    else:
        shared_h_array = np.asarray(shared_h_values, dtype=float).reshape(-1)
        if shared_h_array.size == 0:
            raise ValueError("shared_h_values cannot be empty.")
        if np.any(~np.isfinite(shared_h_array)) or np.any(
            (shared_h_array < 0.0) | (shared_h_array >= 1.0)
        ):
            raise ValueError("Every shared h value must lie in [0, 1).")
        shared_h_array = np.unique(np.append(shared_h_array, h))
    if score_power <= 0.0:
        raise ValueError("score_power must be positive.")
    if not 0.0 < discrimination_low < discrimination_high:
        raise ValueError("Require 0 < discrimination_low < discrimination_high.")
    if not easiness_low < easiness_high:
        raise ValueError("Require easiness_low < easiness_high.")
    if not 0.0 < minimum_relative_angle <= 1.0:
        raise ValueError("minimum_relative_angle must lie in (0, 1].")
    if not 0.5 < majority_proportion <= 1.0:
        raise ValueError("majority_proportion must lie in (0.5, 1].")
    if population_common_trait_max_iter < 1:
        raise ValueError("population_common_trait_max_iter must be positive.")
    if population_common_trait_tolerance <= 0.0:
        raise ValueError("population_common_trait_tolerance must be positive.")

    seed_sequence = np.random.SeedSequence(seed)
    trait_seed, item_seed, response_seed = seed_sequence.spawn(3)
    trait_rng = np.random.default_rng(trait_seed)
    item_rng = np.random.default_rng(item_seed)
    response_rng = np.random.default_rng(response_seed)
    grid = np.linspace(-1.0, 1.0, N)
    raw_scores = np.sign(grid) * np.abs(grid) ** score_power
    if shuffle_ai_labels:
        raw_scores = raw_scores[trait_rng.permutation(N)]
    theta_G_generation_center = _project_centered_sphere_box(
        raw_scores, theta_bound
    )

    theta_star = None
    domain_angles = None
    design_metadata = None
    for _ in range(100):
        try:
            theta_star, domain_angles, design_metadata = _generate_domain_traits(
                theta_G_generation_center,
                T,
                h,
                heterogeneity_structure,
                minimum_relative_angle,
                majority_proportion,
                theta_bound,
                shared_h_array,
                trait_rng,
            )
            break
        except RuntimeError:
            continue
    if theta_star is None or domain_angles is None or design_metadata is None:
        raise RuntimeError(
            "Could not generate bounded domain traits after 100 attempts. "
            "Increase theta_bound or reduce h."
        )

    if heterogeneity_structure == "majority_two_centers":
        theta_G_star = theta_G_generation_center.copy()
        exact_objective = float(
            sum(sine_distance(theta_G_star, theta_t) for theta_t in theta_star)
        )
        population_common_fit = {
            "theta_G_hat": theta_G_star.copy(),
            "exact_objective": exact_objective,
            "iterations": 0,
            "converged": True,
        }
    else:
        population_common_fit = estimate_general_trait(
            theta_star,
            initial_theta_G=theta_G_generation_center,
            max_iter=population_common_trait_max_iter,
            tolerance=population_common_trait_tolerance,
        )
        theta_G_star = population_common_fit["theta_G_hat"].copy()

    generation_center_sine_error = sine_distance(
        theta_G_star, theta_G_generation_center
    )

    population_constraint_mean = float(abs(theta_G_star.mean()))
    population_constraint_norm_error = float(
        abs(np.linalg.norm(theta_G_star) - np.sqrt(N))
    )
    population_orientation_inner_product = float(
        theta_G_star @ theta_star.sum(axis=0)
    )
    if population_constraint_mean > 1e-8:
        raise FloatingPointError(
            "The population common-trait solution is not centred."
        )
    if population_constraint_norm_error > 1e-8:
        raise FloatingPointError(
            "The population common-trait solution does not have norm sqrt(N)."
        )
    if population_orientation_inner_product < -1e-8:
        raise FloatingPointError(
            "The population common-trait solution violates the sign constraint."
        )

    a_star = [
        item_rng.uniform(
            discrimination_low, discrimination_high, size=int(J_vec[t])
        )
        for t in range(T)
    ]
    d_star = [
        item_rng.uniform(easiness_low, easiness_high, size=int(J_vec[t]))
        for t in range(T)
    ]
    M_star = [
        theta_star[t, :, None] * a_star[t][None, :] + d_star[t][None, :]
        for t in range(T)
    ]
    p_star = [expit(M_star[t]) for t in range(T)]
    response_uniforms = [
        response_rng.random(size=p_star[t].shape) for t in range(T)
    ]
    Y = [
        (response_uniforms[t] < p_star[t]).astype(np.int8)
        for t in range(T)
    ]
    domain_signal = np.asarray(
        [
            np.linalg.norm(M_star[t], ord="fro") / np.sqrt(N * J_vec[t])
            for t in range(T)
        ],
        dtype=float,
    )
    aggregated_M = np.concatenate(M_star, axis=1)
    J_total = int(J_vec.sum())
    aggregated_spectral_signal = float(
        np.linalg.norm(aggregated_M, ord=2) / np.sqrt(N * J_total)
    )
    all_probabilities = np.concatenate([block.ravel() for block in p_star])
    all_discriminations = np.concatenate(a_star)

    domain_distances = np.array(
        [sine_distance(theta_G_star, theta_star[t]) for t in range(T)]
    )
    numerical_common_objective = population_common_fit["exact_objective"]
    return {
        "Y": Y,
        "p_star": p_star,
        "M_star": M_star,
        "theta_star": theta_star,
        "a_star": a_star,
        "d_star": d_star,
        "theta_G_star": theta_G_star,
        "theta_G_generation_center": theta_G_generation_center,
        "domain_angles": domain_angles,
        "alpha": np.asarray(
            design_metadata.get("alpha", np.sin(domain_angles)), dtype=float
        ),
        "N": N,
        "T": T,
        "J": int(J_vec[0]) if np.all(J_vec == J_vec[0]) else np.nan,
        "J_vec": J_vec,
        "J_vec_label": "[" + ",".join(map(str, J_vec.tolist())) + "]",
        "J_total": J_total,
        "J_min": int(J_vec.min()),
        "J_max": int(J_vec.max()),
        "h_target": float(h),
        "h_realized": float(domain_distances.max()),
        "domain_distances": domain_distances,
        "score_power": float(score_power),
        "majority_proportion": float(majority_proportion),
        "probability_min": float(all_probabilities.min()),
        "probability_max": float(all_probabilities.max()),
        "probability_saturation_fraction": float(
            np.mean((all_probabilities < 0.05) | (all_probabilities > 0.95))
        ),
        "theta_constraint_max_mean": float(np.max(np.abs(theta_star.mean(axis=1)))),
        "theta_constraint_max_norm_error": float(
            np.max(np.abs(np.linalg.norm(theta_star, axis=1) - np.sqrt(N)))
        ),
        "minimum_discrimination": float(all_discriminations.min()),
        "minimum_domain_signal": float(domain_signal.min()),
        "maximum_domain_signal": float(domain_signal.max()),
        "aggregated_spectral_signal": aggregated_spectral_signal,
        "generation_center_sine_error": float(generation_center_sine_error),
        "numerical_common_sine_error": float(generation_center_sine_error),
        "numerical_common_objective": float(numerical_common_objective),
        "population_common_iterations": int(
            population_common_fit["iterations"]
        ),
        "population_common_converged": bool(
            population_common_fit["converged"]
        ),
        "population_common_tolerance": float(
            population_common_trait_tolerance
        ),
        "population_common_constraint_mean": population_constraint_mean,
        "population_common_constraint_norm_error": (
            population_constraint_norm_error
        ),
        "population_common_orientation_inner_product": (
            population_orientation_inner_product
        ),
        "heterogeneity_structure": heterogeneity_structure,
        **design_metadata,
    }


def _spectral_initialization(
    Y: Sequence[np.ndarray],
    observation_masks: Sequence[np.ndarray],
    theta_bound: float | None,
    a_bound: float,
    d_bound: float,
    initial_theta_G: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Initialize model parameters without using unobserved responses.

    Each domain-specific theta_t is initialized spectrally from its response
    block.  Unless an explicit generic API override is supplied, theta_G is
    initialized by the leading right singular vector of the stacked T-by-N
    matrix of those theta_t initial estimates.
    """
    N = Y[0].shape[0]
    theta_rows: list[np.ndarray] = []
    a: list[np.ndarray] = []
    d: list[np.ndarray] = []
    for block, mask in zip(Y, observation_masks):
        column_counts = mask.sum(axis=0)
        active_items = column_counts > 0
        if not np.any(active_items):
            raise ValueError(
                "Every domain needs at least one item with a training response."
            )
        column_probability = np.full(block.shape[1], 0.5, dtype=float)
        column_probability[active_items] = np.clip(
            (mask * block).sum(axis=0)[active_items]
            / column_counts[active_items],
            0.02,
            0.98,
        )
        d_t = np.clip(logit(column_probability), -d_bound, d_bound)
        local_slope = np.maximum(
            column_probability * (1.0 - column_probability), 0.05
        )
        column_observation_rate = np.where(
            active_items, column_counts / float(N), 1.0
        )
        working = (
            mask
            * (block - column_probability[None, :])
            / local_slope[None, :]
            / column_observation_rate[None, :]
        )
        left, singular_values, right_transpose = np.linalg.svd(
            working, full_matrices=False
        )
        theta_t = np.sqrt(N) * left[:, 0]
        a_t = singular_values[0] * right_transpose[0] / np.sqrt(N)
        if a_t.sum() < 0.0:
            theta_t *= -1.0
            a_t *= -1.0
        a_t = np.clip(a_t, 1e-4, a_bound)
        a_t[~active_items] = 0.0
        d_t[~active_items] = 0.0
        theta_rows.append(theta_t)
        a.append(a_t)
        d.append(d_t)
    theta = np.vstack(theta_rows)
    theta = _project_theta_rows(theta, theta_bound)
    theta_G = _initial_general_trait(theta, initial_theta_G)
    return theta, a, d, theta_G


def _objective_and_gradients(
    Y: Sequence[np.ndarray],
    observation_masks: Sequence[np.ndarray],
    theta: np.ndarray,
    a: Sequence[np.ndarray],
    d: Sequence[np.ndarray],
    theta_G: np.ndarray,
    lambda_value: float,
) -> tuple[
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
]:
    """Return the paper-weighted objective and analytic gradients.

    The likelihood part is exactly

        sum_t L_t / J_t,

    matching the estimator in the paper.  The returned ``domain_nll`` values
    remain the unweighted domain-specific negative log-likelihoods for
    diagnostics; only the objective and gradients are multiplied by 1/J_t.
    """
    T = len(Y)
    J_vec = np.asarray([block.shape[1] for block in Y], dtype=float)
    domain_weights = 1.0 / J_vec
    domain_nll = np.empty(T, dtype=float)
    gradient_theta = np.empty_like(theta)
    gradient_a: list[np.ndarray] = []
    gradient_d: list[np.ndarray] = []

    for t, (block, mask, a_t, d_t) in enumerate(
        zip(Y, observation_masks, a, d)
    ):
        logits = theta[t, :, None] * a_t[None, :] + d_t[None, :]
        residual = mask * (expit(logits) - block)
        domain_nll[t] = np.sum(
            mask * (np.logaddexp(0.0, logits) - block * logits)
        )
        weighted_residual = domain_weights[t] * residual
        gradient_theta[t] = weighted_residual @ a_t
        gradient_a.append(weighted_residual.T @ theta[t])
        gradient_d.append(weighted_residual.sum(axis=0))

    negative_log_likelihood = float(np.sum(domain_weights * domain_nll))
    gradient_G = np.zeros_like(theta_G)
    penalty = 0.0

    if lambda_value > 0.0:
        norm_G = np.linalg.norm(theta_G)
        norms_theta = np.linalg.norm(theta, axis=1)
        unit_G = theta_G / norm_G
        unit_theta = theta / norms_theta[:, None]
        cosine = np.clip(unit_theta @ unit_G, -1.0, 1.0)
        sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
        penalty = float(lambda_value * np.sum(sine))
        multiplier = np.zeros_like(cosine)
        nonzero = sine > 1e-12
        multiplier[nonzero] = -cosine[nonzero] / sine[nonzero]
        gradient_theta += (
            lambda_value
            * multiplier[:, None]
            * (unit_G[None, :] - cosine[:, None] * unit_theta)
            / norms_theta[:, None]
        )
        gradient_G = (
            lambda_value
            * np.sum(
                multiplier[:, None]
                * (unit_theta - cosine[:, None] * unit_G[None, :]),
                axis=0,
            )
            / norm_G
        )

    objective = negative_log_likelihood + penalty
    return (
        objective,
        negative_log_likelihood,
        penalty,
        domain_nll,
        gradient_theta,
        gradient_a,
        gradient_d,
        gradient_G,
    )

def _pack_optimization_blocks(
    Y: Sequence[np.ndarray],
    observation_masks: Sequence[np.ndarray],
    a: Sequence[np.ndarray],
    d: Sequence[np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Pack true-size domain blocks without adding structural padding.

    Items from domain ``t`` occupy one contiguous slice.  This retains exactly
    ``sum_t J_t`` columns while allowing the likelihood and Adam updates to use
    a few large NumPy operations rather than T repeated small operations.
    """
    J_vec = np.asarray([block.shape[1] for block in Y], dtype=int)
    offsets = np.concatenate(([0], np.cumsum(J_vec)))
    item_domains = np.repeat(np.arange(len(Y), dtype=int), J_vec)
    return (
        np.concatenate(Y, axis=1),
        np.concatenate(observation_masks, axis=1),
        np.concatenate(a),
        np.concatenate(d),
        offsets,
        item_domains,
    )


def _packed_objective_and_gradients(
    Y: np.ndarray,
    observation_mask: np.ndarray,
    offsets: np.ndarray,
    item_domains: np.ndarray,
    item_loss_weights: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    theta_G: np.ndarray,
    lambda_value: float,
) -> tuple[
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Vectorized paper-weighted objective on packed unequal-J data.

    ``item_loss_weights[j]`` is 1/J_t for an item belonging to domain t.
    This also lets the Global (lambda=infinity) estimator keep the original
    domain weights after all domains are concatenated into one shared-theta
    optimization block.
    """
    item_loss_weights = np.asarray(item_loss_weights, dtype=float).reshape(-1)
    if item_loss_weights.shape != (Y.shape[1],):
        raise ValueError("item_loss_weights must have one value per packed item.")
    if np.any(~np.isfinite(item_loss_weights)) or np.any(item_loss_weights <= 0.0):
        raise ValueError("item_loss_weights must be finite and strictly positive.")

    theta_by_item = theta[item_domains].T
    logits = theta_by_item * a[None, :] + d[None, :]
    residual = observation_mask * (expit(logits) - Y)
    item_nll = np.sum(
        observation_mask * (np.logaddexp(0.0, logits) - Y * logits),
        axis=0,
    )
    domain_nll = np.add.reduceat(item_nll, offsets[:-1])

    weighted_residual = residual * item_loss_weights[None, :]
    gradient_theta = np.add.reduceat(
        weighted_residual * a[None, :], offsets[:-1], axis=1
    ).T
    gradient_a = np.sum(weighted_residual * theta_by_item, axis=0)
    gradient_d = weighted_residual.sum(axis=0)
    negative_log_likelihood = float(np.sum(item_loss_weights * item_nll))
    gradient_G = np.zeros_like(theta_G)
    penalty = 0.0

    if lambda_value > 0.0:
        norm_G = np.linalg.norm(theta_G)
        norms_theta = np.linalg.norm(theta, axis=1)
        unit_G = theta_G / norm_G
        unit_theta = theta / norms_theta[:, None]
        cosine = np.clip(unit_theta @ unit_G, -1.0, 1.0)
        sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
        penalty = float(lambda_value * np.sum(sine))
        multiplier = np.zeros_like(cosine)
        nonzero = sine > 1e-12
        multiplier[nonzero] = -cosine[nonzero] / sine[nonzero]
        gradient_theta += (
            lambda_value
            * multiplier[:, None]
            * (unit_G[None, :] - cosine[:, None] * unit_theta)
            / norms_theta[:, None]
        )
        gradient_G = (
            lambda_value
            * np.sum(
                multiplier[:, None]
                * (unit_theta - cosine[:, None] * unit_G[None, :]),
                axis=0,
            )
            / norm_G
        )

    objective = negative_log_likelihood + penalty
    return (
        objective,
        negative_log_likelihood,
        penalty,
        domain_nll,
        gradient_theta,
        gradient_a,
        gradient_d,
        gradient_G,
    )

def _single_adam_fit(
    Y: Sequence[np.ndarray],
    *,
    observation_masks: Sequence[np.ndarray],
    lambda_value: float,
    theta_bound: float | None,
    a_bound: float,
    d_bound: float,
    learning_rate: float,
    learning_rate_decay_factor: float,
    learning_rate_decay_steps: int,
    max_iter: int,
    tolerance: float,
    verbose: bool,
    initialization: tuple[
        np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray
    ],
    item_loss_weights: np.ndarray | None = None,
) -> dict:
    """Run one packed projected-Adam trajectory for the weighted estimator.

    By default, every item in domain t receives weight 1/J_t.  The Adam
    gradients are divided by the effective weighted number of observations,

        sum_t |Omega_t| / J_t,

    rather than by the raw number of entries.  For equal J_t this produces
    exactly the same optimization scale as the earlier unweighted code after
    the common 1/J factor is removed from the objective.
    """
    theta = initialization[0].copy()
    theta_G = initialization[3].copy()
    (
        Y_packed,
        mask_packed,
        a,
        d,
        offsets,
        item_domains,
    ) = _pack_optimization_blocks(
        Y,
        observation_masks,
        initialization[1],
        initialization[2],
    )
    active_items = mask_packed.any(axis=0)

    if item_loss_weights is None:
        J_vec = np.diff(offsets).astype(float)
        domain_loss_weights = 1.0 / J_vec
        item_loss_weights = domain_loss_weights[item_domains]
    else:
        item_loss_weights = np.asarray(item_loss_weights, dtype=float).reshape(-1)
        if item_loss_weights.shape != (Y_packed.shape[1],):
            raise ValueError(
                "item_loss_weights must have length equal to the number of packed items."
            )
        if np.any(~np.isfinite(item_loss_weights)) or np.any(item_loss_weights <= 0.0):
            raise ValueError("item_loss_weights must be finite and strictly positive.")

    state_theta = _new_adam_state(theta)
    state_a = _new_adam_state(a)
    state_d = _new_adam_state(d)
    state_G = _new_adam_state(theta_G)
    optimize_G = lambda_value > 0.0
    total_entries = float(
        sum(np.count_nonzero(mask) for mask in observation_masks)
    )
    weighted_entries = float(
        np.sum(mask_packed * item_loss_weights[None, :])
    )
    if weighted_entries <= 0.0:
        raise ValueError("The weighted observation count must be positive.")

    best_objective = np.inf
    best_parameters = None
    best_iteration = 0
    history: list[dict] = []
    previous_objective = np.inf
    converged = False

    for iteration in range(1, max_iter + 1):
        (
            objective,
            negative_log_likelihood,
            penalty,
            domain_nll,
            gradient_theta,
            gradient_a,
            gradient_d,
            gradient_G,
        ) = _packed_objective_and_gradients(
            Y_packed,
            mask_packed,
            offsets,
            item_domains,
            item_loss_weights,
            theta,
            a,
            d,
            theta_G,
            lambda_value,
        )
        if not np.isfinite(objective):
            break
        if objective < best_objective:
            best_objective = objective
            best_parameters = (
                theta.copy(),
                a.copy(),
                d.copy(),
                theta_G.copy(),
            )
            best_iteration = iteration

        relative_change = (
            np.inf
            if not np.isfinite(previous_objective)
            else abs(objective - previous_objective)
            / max(1.0, abs(previous_objective))
        )
        current_learning_rate = _decayed_learning_rate(
            learning_rate,
            iteration,
            learning_rate_decay_factor,
            learning_rate_decay_steps,
        )
        history.append(
            {
                "iteration": int(iteration),
                "objective": float(objective),
                "negative_log_likelihood": float(negative_log_likelihood),
                "penalty": float(penalty),
                "relative_objective_change": float(relative_change),
                "learning_rate": float(current_learning_rate),
            }
        )
        if verbose:
            print(
                f"iteration={iteration:5d}, objective={objective:.6f}, "
                f"weighted_nll={negative_log_likelihood:.6f}, "
                f"penalty={penalty:.6f}, relative_change={relative_change:.3e}, "
                f"learning_rate={current_learning_rate:.3e}"
            )
        if relative_change < tolerance:
            converged = True
            break
        previous_objective = objective
        if iteration == max_iter:
            break

        scale = 1.0 / weighted_entries
        theta = _adam_step(
            theta,
            gradient_theta * scale,
            state_theta,
            iteration,
            current_learning_rate,
            0.9,
            0.999,
            1e-8,
        )
        a = _adam_step(
            a,
            gradient_a * scale,
            state_a,
            iteration,
            current_learning_rate,
            0.9,
            0.999,
            1e-8,
        )
        d = _adam_step(
            d,
            gradient_d * scale,
            state_d,
            iteration,
            current_learning_rate,
            0.9,
            0.999,
            1e-8,
        )
        if optimize_G:
            theta_G = _adam_step(
                theta_G,
                gradient_G * scale,
                state_G,
                iteration,
                current_learning_rate,
                0.9,
                0.999,
                1e-8,
            )

        theta = _project_theta_rows(theta, theta_bound)
        a = np.clip(a, 0.0, a_bound)
        d = np.clip(d, -d_bound, d_bound)
        a[~active_items] = 0.0
        d[~active_items] = 0.0
        if optimize_G:
            theta_G = _project_centered_sphere(theta_G)
            theta_G = _orient_general_trait(theta_G, theta)

    if best_parameters is None:
        raise RuntimeError("Adam did not produce a finite objective.")
    theta_best, a_best_packed, d_best_packed, theta_G_best = best_parameters
    final = _packed_objective_and_gradients(
        Y_packed,
        mask_packed,
        offsets,
        item_domains,
        item_loss_weights,
        theta_best,
        a_best_packed,
        d_best_packed,
        theta_G_best,
        lambda_value,
    )
    a_best = [
        a_best_packed[offsets[t] : offsets[t + 1]].copy()
        for t in range(len(Y))
    ]
    d_best = [
        d_best_packed[offsets[t] : offsets[t + 1]].copy()
        for t in range(len(Y))
    ]
    raw_negative_log_likelihood = float(np.sum(final[3]))
    return {
        "theta_hat": theta_best,
        "a_hat_blocks": a_best,
        "d_hat_blocks": d_best,
        "theta_G_hat": _orient_general_trait(theta_G_best, theta_best),
        "objective": float(final[0]),
        "negative_log_likelihood": float(final[1]),
        "weighted_negative_log_likelihood": float(final[1]),
        "raw_negative_log_likelihood": raw_negative_log_likelihood,
        "penalty": float(final[2]),
        "domain_negative_log_likelihood": np.asarray(final[3], dtype=float),
        "history": history,
        "iterations": int(iteration),
        "best_iteration": int(best_iteration),
        "returned_iteration": int(best_iteration),
        "returned_best_iteration": True,
        "converged": bool(converged),
        "total_entries": int(total_entries),
        "weighted_entries": float(weighted_entries),
    }

def _validate_fit_options(
    learning_rate: float,
    learning_rate_decay_factor: float,
    learning_rate_decay_steps: int,
    max_iter: int,
    theta_bound: float | None,
    a_bound: float,
    d_bound: float,
) -> None:
    if learning_rate <= 0.0 or max_iter <= 0:
        raise ValueError("learning_rate and max_iter must be positive.")
    if not 0.0 < learning_rate_decay_factor <= 1.0:
        raise ValueError("learning_rate_decay_factor must lie in (0, 1].")
    if learning_rate_decay_steps < 1:
        raise ValueError("learning_rate_decay_steps must be positive.")
    if theta_bound is not None and theta_bound <= 1.0:
        raise ValueError("theta_bound must exceed 1 when supplied.")
    if a_bound <= 0.0 or d_bound <= 0.0:
        raise ValueError("a_bound and d_bound must be positive.")


def fit_proposed_method(
    Y: np.ndarray | Sequence[np.ndarray],
    *,
    observation_mask: np.ndarray | Sequence[np.ndarray] | None = None,
    lambda_value: float | None = None,
    lambda_constant: float = 1.0,
    theta_bound: float | None = 4.0,
    a_bound: float = 4.0,
    d_bound: float = 4.0,
    learning_rate: float = 0.02,
    learning_rate_decay_factor: float = 0.5,
    learning_rate_decay_steps: int = 200,
    max_iter: int = 1000,
    tolerance: float = 1e-3,
    initial_theta_G: np.ndarray | None = None,
    precomputed_initialization: tuple[
        np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray
    ]
    | None = None,
    verbose: bool = False,
) -> dict:
    """Fit the proposed estimator using only entries in ``observation_mask``.

    When ``lambda_value`` is omitted, the paper-compatible scaling is

        C_lambda * max_t N / sqrt(min(N, J_t)).

    This scale depends on the full domain sizes J_t, not on the realized
    number of training entries in a cross-validation fold.  Thus the same
    candidate lambda(c) is used in every fold, exactly as specified in the
    Supplementary Materials.
    """
    Y_blocks, J_vec = _as_domain_blocks(Y)
    masks = _prepare_observation_masks(Y_blocks, observation_mask)
    Y_fit = [np.where(mask, block, 0.0) for block, mask in zip(Y_blocks, masks)]
    T = len(Y_blocks)
    N = Y_blocks[0].shape[0]
    if T < 2:
        raise ValueError("The proposed estimator requires at least two domains.")
    _validate_fit_options(
        learning_rate,
        learning_rate_decay_factor,
        learning_rate_decay_steps,
        max_iter,
        theta_bound,
        a_bound,
        d_bound,
    )
    lambda_scale_by_domain = N / np.sqrt(np.minimum(N, J_vec).astype(float))
    lambda_theory_scale = float(np.max(lambda_scale_by_domain))
    if lambda_value is None:
        lambda_value = float(lambda_constant * lambda_theory_scale)
    if lambda_value < 0.0:
        raise ValueError("lambda_value must be nonnegative.")

    if precomputed_initialization is None:
        initialization = _spectral_initialization(
            Y_fit,
            masks,
            theta_bound,
            a_bound,
            d_bound,
            initial_theta_G=initial_theta_G,
        )
    else:
        initialization = precomputed_initialization
        valid_initialization = (
            len(initialization) == 4
            and np.shape(initialization[0]) == (T, N)
            and len(initialization[1]) == T
            and len(initialization[2]) == T
            and all(
                np.shape(initialization[1][t]) == (int(J_vec[t]),)
                and np.shape(initialization[2][t]) == (int(J_vec[t]),)
                for t in range(T)
            )
            and np.shape(initialization[3]) == (N,)
        )
        if not valid_initialization:
            raise ValueError(
                "precomputed_initialization has incompatible parameter shapes."
            )
    result = _single_adam_fit(
        Y_fit,
        observation_masks=masks,
        lambda_value=lambda_value,
        theta_bound=theta_bound,
        a_bound=a_bound,
        d_bound=d_bound,
        learning_rate=learning_rate,
        learning_rate_decay_factor=learning_rate_decay_factor,
        learning_rate_decay_steps=learning_rate_decay_steps,
        max_iter=max_iter,
        tolerance=tolerance,
        verbose=verbose,
        initialization=initialization,
    )

    result["a_hat"] = _pad_parameter_blocks(result.pop("a_hat_blocks"))
    result["d_hat"] = _pad_parameter_blocks(result.pop("d_hat_blocks"))
    result["lambda_value"] = float(lambda_value)
    result["lambda_constant"] = float(lambda_constant)
    result["lambda_theory_scale"] = float(lambda_theory_scale)
    result["lambda_scale_by_domain"] = lambda_scale_by_domain.copy()
    result["J_vec"] = J_vec.copy()
    result["method"] = "Proposed"
    return result


def fit_local_method(
    Y: np.ndarray | Sequence[np.ndarray],
    *,
    theta_bound: float | None = 4.0,
    a_bound: float = 4.0,
    d_bound: float = 4.0,
    learning_rate: float = 0.02,
    learning_rate_decay_factor: float = 0.5,
    learning_rate_decay_steps: int = 200,
    max_iter: int = 1000,
    tolerance: float = 1e-3,
    initial_theta_G: np.ndarray | None = None,
    precomputed_initialization: tuple[
        np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray
    ] | None = None,
    verbose: bool = False,
) -> dict:
    """Fit each domain independently and then compute theta_G^0.

    ``precomputed_initialization`` may be supplied in simulations to initialize
    every domain at known parameter values.  When omitted, the original
    spectral initialization is used.
    """
    Y_blocks, J_vec = _as_domain_blocks(Y)
    full_masks = _prepare_observation_masks(Y_blocks, None)
    T = len(Y_blocks)
    N = Y_blocks[0].shape[0]
    J_max = int(J_vec.max())
    _validate_fit_options(
        learning_rate,
        learning_rate_decay_factor,
        learning_rate_decay_steps,
        max_iter,
        theta_bound,
        a_bound,
        d_bound,
    )
    theta_hat = np.empty((T, N))
    a_hat = np.zeros((T, J_max))
    d_hat = np.zeros((T, J_max))
    domain_nll = np.empty(T)
    weighted_domain_nll = np.empty(T)
    iterations = np.empty(T, dtype=int)
    best_iterations = np.empty(T, dtype=int)
    converged = np.empty(T, dtype=bool)

    for t in range(T):
        J_t = int(J_vec[t])
        Y_t = [Y_blocks[t]]
        mask_t = [full_masks[t]]
        if precomputed_initialization is None:
            initialization = _spectral_initialization(
                Y_t, mask_t, theta_bound, a_bound, d_bound
            )
        else:
            theta_init, a_init, d_init, theta_G_init = precomputed_initialization
            valid_truth_init = (
                np.shape(theta_init) == (T, N)
                and len(a_init) == T
                and len(d_init) == T
                and np.shape(theta_G_init) == (N,)
                and all(
                    np.shape(a_init[k]) == (int(J_vec[k]),)
                    and np.shape(d_init[k]) == (int(J_vec[k]),)
                    for k in range(T)
                )
            )
            if not valid_truth_init:
                raise ValueError(
                    "precomputed_initialization has incompatible Local shapes."
                )
            initialization = (
                np.asarray(theta_init[t : t + 1], dtype=float).copy(),
                [np.asarray(a_init[t], dtype=float).copy()],
                [np.asarray(d_init[t], dtype=float).copy()],
                np.asarray(theta_G_init, dtype=float).copy(),
            )
        best = _single_adam_fit(
            Y_t,
            observation_masks=mask_t,
            lambda_value=0.0,
            theta_bound=theta_bound,
            a_bound=a_bound,
            d_bound=d_bound,
            learning_rate=learning_rate,
            learning_rate_decay_factor=learning_rate_decay_factor,
            learning_rate_decay_steps=learning_rate_decay_steps,
            max_iter=max_iter,
            tolerance=tolerance,
            verbose=verbose,
            initialization=initialization,
        )
        theta_hat[t] = best["theta_hat"][0]
        a_hat[t, :J_t] = best["a_hat_blocks"][0]
        d_hat[t, :J_t] = best["d_hat_blocks"][0]
        domain_nll[t] = best["raw_negative_log_likelihood"]
        weighted_domain_nll[t] = best["negative_log_likelihood"]
        iterations[t] = best["iterations"]
        best_iterations[t] = best["best_iteration"]
        converged[t] = best["converged"]

    common = estimate_general_trait(
        theta_hat,
        initial_theta_G=initial_theta_G,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    return {
        "method": "Local",
        "theta_hat": theta_hat,
        "a_hat": a_hat,
        "d_hat": d_hat,
        "theta_G_hat": common["theta_G_hat"],
        "objective": float(weighted_domain_nll.sum()),
        "negative_log_likelihood": float(weighted_domain_nll.sum()),
        "weighted_negative_log_likelihood": float(weighted_domain_nll.sum()),
        "raw_negative_log_likelihood": float(domain_nll.sum()),
        "penalty": 0.0,
        "domain_negative_log_likelihood": domain_nll,
        "domain_weighted_negative_log_likelihood": weighted_domain_nll,
        "iterations": iterations,
        "best_iteration": best_iterations,
        "converged": converged,
        "common_trait_iterations": int(common["iterations"]),
        "common_trait_converged": bool(common["converged"]),
        "total_entries": int(
            sum(np.count_nonzero(mask) for mask in full_masks)
        ),
        "weighted_entries": float(
            sum(
                np.count_nonzero(full_masks[t]) / float(J_vec[t])
                for t in range(T)
            )
        ),
        "J_vec": J_vec.copy(),
    }


def fit_global_method(
    Y: np.ndarray | Sequence[np.ndarray],
    *,
    theta_bound: float | None = 4.0,
    a_bound: float = 4.0,
    d_bound: float = 4.0,
    learning_rate: float = 0.02,
    learning_rate_decay_factor: float = 0.5,
    learning_rate_decay_steps: int = 200,
    max_iter: int = 1000,
    tolerance: float = 1e-3,
    precomputed_initialization: tuple[
        np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray
    ] | None = None,
    verbose: bool = False,
) -> dict:
    """Fit the lambda=infinity baseline with one shared capability vector.

    ``precomputed_initialization`` is a one-domain packed initialization.
    In the truth-initialized simulation it uses theta_G^* as the shared
    capability initializer and the true item parameters across all domains.
    """
    Y_blocks, J_vec = _as_domain_blocks(Y)
    T = len(Y_blocks)
    N = Y_blocks[0].shape[0]
    J_max = int(J_vec.max())
    concatenated = [np.concatenate(Y_blocks, axis=1)]
    concatenated_mask = [np.ones_like(concatenated[0], dtype=bool)]
    global_item_loss_weights = np.concatenate(
        [
            np.full(int(J_vec[t]), 1.0 / float(J_vec[t]), dtype=float)
            for t in range(T)
        ]
    )
    _validate_fit_options(
        learning_rate,
        learning_rate_decay_factor,
        learning_rate_decay_steps,
        max_iter,
        theta_bound,
        a_bound,
        d_bound,
    )
    if precomputed_initialization is None:
        initialization = _spectral_initialization(
            concatenated,
            concatenated_mask,
            theta_bound,
            a_bound,
            d_bound,
        )
    else:
        initialization = precomputed_initialization
        valid_global_init = (
            len(initialization) == 4
            and np.shape(initialization[0]) == (1, N)
            and len(initialization[1]) == 1
            and len(initialization[2]) == 1
            and np.shape(initialization[1][0]) == (int(J_vec.sum()),)
            and np.shape(initialization[2][0]) == (int(J_vec.sum()),)
            and np.shape(initialization[3]) == (N,)
        )
        if not valid_global_init:
            raise ValueError(
                "precomputed_initialization has incompatible Global shapes."
            )
    best = _single_adam_fit(
        concatenated,
        observation_masks=concatenated_mask,
        lambda_value=0.0,
        theta_bound=theta_bound,
        a_bound=a_bound,
        d_bound=d_bound,
        learning_rate=learning_rate,
        learning_rate_decay_factor=learning_rate_decay_factor,
        learning_rate_decay_steps=learning_rate_decay_steps,
        max_iter=max_iter,
        tolerance=tolerance,
        verbose=verbose,
        initialization=initialization,
        item_loss_weights=global_item_loss_weights,
    )
    theta_common = best["theta_hat"][0]
    a_hat = np.zeros((T, J_max))
    d_hat = np.zeros((T, J_max))
    domain_nll = np.empty(T)
    start = 0
    for t in range(T):
        J_t = int(J_vec[t])
        stop = start + J_t
        a_hat[t, :J_t] = best["a_hat_blocks"][0][start:stop]
        d_hat[t, :J_t] = best["d_hat_blocks"][0][start:stop]
        logits_t = (
            theta_common[:, None] * a_hat[t, None, :J_t]
            + d_hat[t, None, :J_t]
        )
        domain_nll[t] = np.sum(
            np.logaddexp(0.0, logits_t)
            - Y_blocks[t] * logits_t
        )
        start = stop
    return {
        "method": "Global",
        "theta_hat": np.repeat(theta_common[None, :], T, axis=0),
        "a_hat": a_hat,
        "d_hat": d_hat,
        "theta_G_hat": theta_common.copy(),
        "objective": float(best["objective"]),
        "negative_log_likelihood": float(best["negative_log_likelihood"]),
        "weighted_negative_log_likelihood": float(best["negative_log_likelihood"]),
        "raw_negative_log_likelihood": float(domain_nll.sum()),
        "penalty": 0.0,
        "domain_negative_log_likelihood": domain_nll,
        "domain_weighted_negative_log_likelihood": domain_nll / J_vec,
        "iterations": int(best["iterations"]),
        "best_iteration": int(best["best_iteration"]),
        "converged": bool(best["converged"]),
        "total_entries": int(N * J_vec.sum()),
        "weighted_entries": float(N * T),
        "J_vec": J_vec.copy(),
    }


def evaluate_estimator(result: dict, data: dict) -> dict:
    """Attach theta, ranking, item-parameter, and natural-matrix errors."""
    theta_G_error = normalized_theta_error(
        result["theta_G_hat"], data["theta_G_star"]
    )
    theta_errors = np.array(
        [
            normalized_theta_error(result["theta_hat"][t], data["theta_star"][t])
            for t in range(data["T"])
        ]
    )
    theta_G_sine_error = sine_distance(
        result["theta_G_hat"], data["theta_G_star"]
    )
    theta_sine_errors = np.array(
        [
            sine_distance(result["theta_hat"][t], data["theta_star"][t])
            for t in range(data["T"])
        ]
    )
    central_rho, central_tau = ranking_correlations(
        result["theta_G_hat"], data["theta_G_star"]
    )
    domain_rank = np.array(
        [
            ranking_correlations(
                result["theta_hat"][t], data["theta_star"][t]
            )
            for t in range(data["T"])
        ]
    )
    J_vec = np.asarray(data["J_vec"], dtype=int)
    evaluation_domain_nll = np.empty(data["T"])
    domain_M_rmse = np.empty(data["T"])
    domain_a_rmse = np.empty(data["T"])
    domain_d_rmse = np.empty(data["T"])
    estimated_discriminations: list[np.ndarray] = []
    for t in range(data["T"]):
        J_t = int(J_vec[t])
        a_hat_t = result["a_hat"][t, :J_t]
        d_hat_t = result["d_hat"][t, :J_t]
        M_hat_t = (
            result["theta_hat"][t, :, None] * a_hat_t[None, :]
            + d_hat_t[None, :]
        )
        evaluation_domain_nll[t] = np.sum(
            np.logaddexp(0.0, M_hat_t) - data["Y"][t] * M_hat_t
        )
        domain_M_rmse[t] = np.sqrt(
            np.mean((M_hat_t - data["M_star"][t]) ** 2)
        )
        domain_a_rmse[t] = np.sqrt(
            np.mean((a_hat_t - data["a_star"][t]) ** 2)
        )
        domain_d_rmse[t] = np.sqrt(
            np.mean((d_hat_t - data["d_star"][t]) ** 2)
        )
        estimated_discriminations.append(a_hat_t)

    result.update(
        {
            "theta_G_error": float(theta_G_error),
            "theta_t_errors": theta_errors,
            "mean_theta_t_error": float(theta_errors.mean()),
            "max_theta_t_error": float(theta_errors.max()),
            "theta_G_sine_error": float(theta_G_sine_error),
            "theta_t_sine_errors": theta_sine_errors,
            "mean_theta_t_sine_error": float(theta_sine_errors.mean()),
            "max_theta_t_sine_error": float(theta_sine_errors.max()),
            "theta_G_spearman_rho": central_rho,
            "theta_G_kendall_tau": central_tau,
            "theta_t_spearman_rhos": domain_rank[:, 0],
            "theta_t_kendall_taus": domain_rank[:, 1],
            "mean_theta_t_spearman_rho": float(domain_rank[:, 0].mean()),
            "min_theta_t_spearman_rho": float(domain_rank[:, 0].min()),
            "mean_theta_t_kendall_tau": float(domain_rank[:, 1].mean()),
            "min_theta_t_kendall_tau": float(domain_rank[:, 1].min()),
            "domain_M_rmse": domain_M_rmse,
            "domain_a_rmse": domain_a_rmse,
            "domain_d_rmse": domain_d_rmse,
            "evaluation_domain_nll": evaluation_domain_nll,
            "mean_M_rmse": float(domain_M_rmse.mean()),
            "mean_a_rmse": float(domain_a_rmse.mean()),
            "mean_d_rmse": float(domain_d_rmse.mean()),
            "estimated_theta_max_mean": float(
                np.max(np.abs(result["theta_hat"].mean(axis=1)))
            ),
            "estimated_theta_max_norm_error": float(
                np.max(
                    np.abs(
                        np.linalg.norm(result["theta_hat"], axis=1)
                        - np.sqrt(data["N"])
                    )
                )
            ),
            "estimated_min_a": float(
                np.concatenate(estimated_discriminations).min()
            ),
        }
    )
    return result


def fit_and_evaluate_proposed_method(data: dict, **fit_kwargs) -> dict:
    """Fit Proposed and evaluate it against the simulation truth."""
    fit_kwargs = dict(fit_kwargs)
    return evaluate_estimator(
        fit_proposed_method(data["Y"], **fit_kwargs), data
    )


def _reoptimize_proposed_general_trait(
    fitted_result: dict,
    *,
    initial_theta_G: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> dict:
    """Re-estimate theta_G from the fitted Proposed domain traits.

    After the Proposed optimizer returns its best-iteration theta_t estimates,
    theta_G is updated by solving

        min_{theta_G} sum_t sin(theta_G, theta_t_hat)

    subject to the centred-sphere/orientation constraints.  The domain traits
    and item parameters are held fixed.
    """
    joint_theta_G = np.asarray(
        fitted_result["theta_G_hat"], dtype=float
    ).copy()

    common = estimate_general_trait(
        np.asarray(fitted_result["theta_hat"], dtype=float),
        initial_theta_G=np.asarray(initial_theta_G, dtype=float),
        max_iter=max_iter,
        tolerance=tolerance,
    )

    fitted_result["theta_G_joint_hat"] = joint_theta_G
    fitted_result["theta_G_hat"] = np.asarray(
        common["theta_G_hat"], dtype=float
    ).copy()
    fitted_result["posthoc_general_trait_iterations"] = int(
        common["iterations"]
    )
    fitted_result["posthoc_general_trait_converged"] = bool(
        common["converged"]
    )
    fitted_result["posthoc_general_trait_exact_objective"] = float(
        common["exact_objective"]
    )

    lambda_value = float(fitted_result.get("lambda_value", 0.0))
    posthoc_penalty = lambda_value * float(
        sum(
            sine_distance(fitted_result["theta_G_hat"], theta_t)
            for theta_t in fitted_result["theta_hat"]
        )
    )
    fitted_result["penalty"] = posthoc_penalty
    fitted_result["objective"] = (
        float(fitted_result["negative_log_likelihood"]) + posthoc_penalty
    )
    return fitted_result


def fit_and_evaluate_local_method(data: dict, **fit_kwargs) -> dict:
    """Fit Local and evaluate it against the simulation truth."""
    fit_kwargs = dict(fit_kwargs)
    return evaluate_estimator(fit_local_method(data["Y"], **fit_kwargs), data)


def fit_and_evaluate_global_method(data: dict, **fit_kwargs) -> dict:
    return evaluate_estimator(fit_global_method(data["Y"], **fit_kwargs), data)


def _validate_lambda_grid(lambda_constants: Sequence[float]) -> np.ndarray:
    """Return a finite, nonnegative, duplicate-free C_lambda grid."""
    constants = np.asarray(lambda_constants, dtype=float)
    if constants.ndim != 1 or constants.size == 0:
        raise ValueError("C_lambda_values must be a nonempty one-dimensional grid.")
    if np.any(~np.isfinite(constants)) or np.any(constants < 0.0):
        raise ValueError("Every C_lambda candidate must be finite and nonnegative.")
    if np.unique(constants).size != constants.size:
        raise ValueError("C_lambda_values must not contain duplicates.")
    return constants


def make_ai_stratified_cv_folds(
    observation_mask: np.ndarray | Sequence[np.ndarray],
    *,
    n_folds: int,
    seed: int,
) -> list[dict[str, list[np.ndarray]]]:
    """Create AI-stratified entry-wise K-fold train/test masks.

    For every domain-AI pair, that AI's observed item responses are shuffled
    and split across the K held-out folds.  Hence each originally observed
    entry appears in exactly one test fold, and every AI is represented in
    every training and test fold within each domain.

    This is the transpose of the previous item-stratified scheme: the old
    scheme fixed an item and split its AI responses across folds; this scheme
    fixes an AI and splits its item responses across folds.
    """
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least 2.")
    if isinstance(observation_mask, np.ndarray):
        mask_array = np.asarray(observation_mask)
        if mask_array.ndim != 3:
            raise ValueError("observation_mask must have shape (T, N, J).")
        masks = [
            mask_array[t].astype(bool, copy=True)
            for t in range(mask_array.shape[0])
        ]
    else:
        masks = [np.asarray(mask, dtype=bool) for mask in observation_mask]
    if not masks or any(mask.ndim != 2 for mask in masks):
        raise ValueError("observation_mask must contain N-by-J_t domain blocks.")
    if len({mask.shape[0] for mask in masks}) != 1:
        raise ValueError("All observation masks must contain the same N subjects.")

    for t, mask in enumerate(masks):
        counts = mask.sum(axis=1)
        insufficient = (counts > 0) & (counts < n_folds)
        if np.any(insufficient):
            i = int(np.flatnonzero(insufficient)[0])
            raise ValueError(
                "Each domain-AI row needs at least n_folds observed items; "
                f"domain {t + 1}, AI {i + 1} has {counts[i]}."
            )

    rng = np.random.default_rng(seed)
    test_masks = [
        [np.zeros_like(mask, dtype=bool) for mask in masks]
        for _ in range(n_folds)
    ]

    for t, mask in enumerate(masks):
        for i in range(mask.shape[0]):
            if not np.any(mask[i, :]):
                continue
            observed_items = np.flatnonzero(mask[i, :])
            shuffled_items = rng.permutation(observed_items)
            for fold_index, test_items in enumerate(
                np.array_split(shuffled_items, n_folds)
            ):
                test_masks[fold_index][t][i, test_items] = True

    folds: list[dict[str, list[np.ndarray]]] = []
    for test_mask in test_masks:
        training_mask = [mask & ~test for mask, test in zip(masks, test_mask)]
        for t, (full, training, test) in enumerate(
            zip(masks, training_mask, test_mask)
        ):
            active_ais = full.sum(axis=1) > 0
            if np.any(active_ais & (training.sum(axis=1) == 0)):
                raise ValueError(
                    "A CV training fold leaves an AI without responses."
                )
            if np.any(active_ais & (test.sum(axis=1) == 0)):
                raise ValueError(
                    "A CV test fold leaves an AI without responses."
                )

            active_items = full.sum(axis=0) > 0
            if np.any(active_items & (training.sum(axis=0) == 0)):
                j = int(
                    np.flatnonzero(
                        active_items & (training.sum(axis=0) == 0)
                    )[0]
                )
                raise ValueError(
                    "AI-stratified CV left an item without training responses; "
                    f"domain {t + 1}, item {j + 1}. Try a different seed, "
                    "fewer folds, or a denser observation pattern."
                )
        folds.append(
            {
                "training_mask": training_mask,
                "test_mask": test_mask,
            }
        )

    for t, mask in enumerate(masks):
        coverage = np.sum(
            np.stack([fold_mask[t] for fold_mask in test_masks], axis=0),
            axis=0,
        )
        if not np.array_equal(coverage, mask.astype(int)):
            raise RuntimeError(
                "CV folds do not partition the observed entries exactly."
            )
    return folds


def make_entrywise_cv_folds(
    observation_mask: np.ndarray | Sequence[np.ndarray],
    *,
    n_folds: int,
    seed: int,
) -> list[dict[str, list[np.ndarray]]]:
    """Backward-compatible alias for the new AI-stratified CV splitter."""
    return make_ai_stratified_cv_folds(
        observation_mask,
        n_folds=n_folds,
        seed=seed,
    )

def held_out_negative_log_likelihood(
    Y: np.ndarray | Sequence[np.ndarray],
    test_mask: np.ndarray | Sequence[np.ndarray],
    fitted_result: dict,
) -> dict:
    """Evaluate the domain-balanced held-out Bernoulli NLL used in the paper.

    In one CV fold, domain t contributes its own mean held-out NLL,

        L_{t,test} / |V_{tk}|,

    and the fold score is the simple average of these T domain means.  This
    gives every domain equal weight even when the J_t differ.
    """
    Y_blocks, J_vec = _as_domain_blocks(Y)
    masks = _prepare_observation_masks(Y_blocks, test_mask)
    domain_nll = np.empty(len(Y_blocks), dtype=float)
    domain_entries = np.empty(len(Y_blocks), dtype=int)
    for t, (block, mask) in enumerate(zip(Y_blocks, masks)):
        J_t = int(J_vec[t])
        logits = (
            fitted_result["theta_hat"][t, :, None]
            * fitted_result["a_hat"][t, None, :J_t]
            + fitted_result["d_hat"][t, None, :J_t]
        )
        domain_nll[t] = np.sum(
            mask * (np.logaddexp(0.0, logits) - block * logits)
        )
        domain_entries[t] = np.count_nonzero(mask)

    if np.any(domain_entries <= 0):
        raise ValueError("Every domain must contribute held-out entries in each CV fold.")
    domain_mean_nll = domain_nll / domain_entries
    total_nll = float(domain_nll.sum())
    total_entries = int(domain_entries.sum())
    return {
        "test_negative_log_likelihood": total_nll,
        "mean_test_negative_log_likelihood": float(domain_mean_nll.mean()),
        "test_entries": total_entries,
        "domain_test_negative_log_likelihood": domain_nll,
        "domain_test_entries": domain_entries,
        "domain_mean_test_negative_log_likelihood": domain_mean_nll,
    }

def cross_validate_proposed_method(
    Y: np.ndarray | Sequence[np.ndarray],
    lambda_constants: Sequence[float],
    *,
    n_folds: int = 5,
    observation_mask: np.ndarray | Sequence[np.ndarray] | None = None,
    seed: int = 20260720,
    n_jobs: int = 1,
    progress: bool = False,
    precomputed_initialization: tuple[
        np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray
    ] | None = None,
    **fit_kwargs,
) -> dict:
    """Select C_lambda by AI-stratified K-fold test NLL and refit on all data.

    If ``precomputed_initialization`` is supplied, exactly the same supplied
    initialization is used for every C_lambda candidate in every fold and for
    the final full-data refit.  This option is intended for simulation
    diagnostics such as initialization at the generating truth.
    """
    constants = _validate_lambda_grid(lambda_constants)
    controlled = {
        "lambda_value",
        "lambda_constant",
        "observation_mask",
        "precomputed_initialization",
        "seed",
    }.intersection(fit_kwargs)
    if controlled:
        raise ValueError(
            "Cross-validation controls these fit arguments: "
            f"{sorted(controlled)}."
        )
    if not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")

    Y_blocks, J_vec = _as_domain_blocks(Y)
    N = Y_blocks[0].shape[0]
    full_lambda_scale_by_domain = (
        N / np.sqrt(np.minimum(N, J_vec).astype(float))
    )
    full_lambda_theory_scale = float(np.max(full_lambda_scale_by_domain))
    full_mask = _prepare_observation_masks(Y_blocks, observation_mask)
    folds = make_ai_stratified_cv_folds(
        full_mask,
        n_folds=n_folds,
        seed=seed + 100_003,
    )
    cv_start = perf_counter()
    fold_contexts: list[dict] = []
    for fold_index, fold in enumerate(folds, start=1):
        training_entries_by_domain = np.asarray(
            [np.count_nonzero(mask) for mask in fold["training_mask"]]
        )
        test_entries_by_domain = np.asarray(
            [np.count_nonzero(mask) for mask in fold["test_mask"]]
        )

        if precomputed_initialization is None:
            fold_initialization = _spectral_initialization(
                [
                    np.where(mask, block, 0.0)
                    for block, mask in zip(Y_blocks, fold["training_mask"])
                ],
                fold["training_mask"],
                fit_kwargs.get("theta_bound", 4.0),
                fit_kwargs.get("a_bound", 4.0),
                fit_kwargs.get("d_bound", 4.0),
                initial_theta_G=fit_kwargs.get("initial_theta_G"),
            )
        else:
            fold_initialization = precomputed_initialization
        full_weighted_entries = float(
            sum(
                np.count_nonzero(full_mask[t]) / float(J_vec[t])
                for t in range(len(Y_blocks))
            )
        )
        training_weighted_entries = float(
            sum(
                np.count_nonzero(fold["training_mask"][t]) / float(J_vec[t])
                for t in range(len(Y_blocks))
            )
        )
        training_fraction = training_weighted_entries / full_weighted_entries

        fold_contexts.append(
            {
                "fold_index": fold_index,
                "fold": fold,
                "initialization": fold_initialization,
                "training_entries_by_domain": training_entries_by_domain,
                "test_entries_by_domain": test_entries_by_domain,
                "training_fraction": float(training_fraction),
            }
        )

    tasks = [
        (context, float(constant))
        for context in fold_contexts
        for constant in constants
    ]

    def fit_candidate(task: tuple[dict, float]) -> dict:
        context, constant = task
        fold = context["fold"]
        fit_start = perf_counter()
        full_candidate_lambda_value = float(
            constant * full_lambda_theory_scale
        )
        candidate_lambda_value = float(
            context["training_fraction"] * full_candidate_lambda_value
        )
        fitted = fit_proposed_method(
            Y_blocks,
            observation_mask=fold["training_mask"],
            lambda_value=candidate_lambda_value,
            lambda_constant=constant,
            precomputed_initialization=context["initialization"],
            **fit_kwargs,
        )
        score = held_out_negative_log_likelihood(
            Y_blocks,
            fold["test_mask"],
            fitted,
        )
        training_entries_by_domain = context["training_entries_by_domain"]
        test_entries_by_domain = context["test_entries_by_domain"]
        return {
            "fold": int(context["fold_index"]),
            "C_lambda": constant,
            "training_lambda_value": float(fitted["lambda_value"]),
            "training_fraction": float(context["training_fraction"]),
            "training_entries": int(training_entries_by_domain.sum()),
            "minimum_domain_training_entries": int(
                training_entries_by_domain.min()
            ),
            "maximum_domain_training_entries": int(
                training_entries_by_domain.max()
            ),
            "test_entries": int(score["test_entries"]),
            "minimum_domain_test_entries": int(test_entries_by_domain.min()),
            "maximum_domain_test_entries": int(test_entries_by_domain.max()),
            "test_negative_log_likelihood": float(
                score["test_negative_log_likelihood"]
            ),
            "mean_test_negative_log_likelihood": float(
                score["mean_test_negative_log_likelihood"]
            ),
            "training_mean_negative_log_likelihood": float(
                fitted["negative_log_likelihood"] / fitted["weighted_entries"]
            ),
            "converged": bool(fitted["converged"]),
            "iterations": int(fitted["iterations"]),
            "runtime_seconds": float(perf_counter() - fit_start),
        }

    workers = min(int(n_jobs), len(tasks))
    if progress:
        print(f"  CV: {len(tasks)} candidate fits using {workers} job(s)")
    if workers == 1:
        fold_records = [fit_candidate(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fold_records = list(executor.map(fit_candidate, tasks))
    if progress:
        for record in fold_records:
            print(
                f"    fold={record['fold']}/{n_folds}, "
                f"C_lambda={record['C_lambda']:.8g}, "
                f"iterations={record['iterations']}, "
                f"fit_time={record['runtime_seconds']:.3f}s"
            )

    path_records: list[dict] = []
    for constant in constants:
        candidate = [
            record
            for record in fold_records
            if np.isclose(record["C_lambda"], constant)
        ]
        fold_means = np.array(
            [record["mean_test_negative_log_likelihood"] for record in candidate]
        )
        path_records.append(
            {
                "C_lambda": float(constant),
                "mean_test_negative_log_likelihood": float(fold_means.mean()),
                "fold_standard_error": float(
                    fold_means.std(ddof=1) / np.sqrt(n_folds)
                ),
                "mean_training_negative_log_likelihood": float(
                    np.mean(
                        [
                            record["training_mean_negative_log_likelihood"]
                            for record in candidate
                        ]
                    )
                ),
                "mean_training_lambda_value": float(
                    np.mean(
                        [record["training_lambda_value"] for record in candidate]
                    )
                ),
                "converged_rate": float(
                    np.mean([record["converged"] for record in candidate])
                ),
                "runtime_seconds": float(
                    sum(record["runtime_seconds"] for record in candidate)
                ),
            }
        )

    best_index = min(
        range(len(path_records)),
        key=lambda index: (
            path_records[index]["mean_test_negative_log_likelihood"],
            path_records[index]["C_lambda"],
        ),
    )
    selected_constant = float(path_records[best_index]["C_lambda"])
    for index, record in enumerate(path_records):
        record["selected"] = bool(index == best_index)

    cv_runtime = perf_counter() - cv_start
    refit_start = perf_counter()
    final_fit = fit_proposed_method(
        Y_blocks,
        observation_mask=full_mask,
        lambda_value=float(selected_constant * full_lambda_theory_scale),
        lambda_constant=selected_constant,
        precomputed_initialization=precomputed_initialization,
        **fit_kwargs,
    )
    final_fit.update(
        {
            "selected_C_lambda": selected_constant,
            "cv_mean_test_negative_log_likelihood": float(
                path_records[best_index]["mean_test_negative_log_likelihood"]
            ),
            "cv_fold_standard_error": float(
                path_records[best_index]["fold_standard_error"]
            ),
            "cv_n_folds": int(n_folds),
            "cv_runtime_seconds": float(cv_runtime),
            "refit_runtime_seconds": float(perf_counter() - refit_start),
            "cv_path_records": path_records,
            "cv_fold_records": fold_records,
        }
    )
    return final_fit


def oracle_tune_proposed_method(
    data: dict,
    lambda_constants: Sequence[float],
    *,
    n_jobs: int = 1,
    progress: bool = False,
    **fit_kwargs,
) -> dict:
    """Select C_lambda by the mean domain-specific sine error.

    This tuning rule is available only in simulations because it uses the
    generated truth theta_t^*.  Every candidate is fitted on the complete
    simulated dataset, and the fitted candidate minimizing

        T^{-1} sum_t sin(theta_hat_t, theta_t^*)

    is returned directly.  Exact ties are resolved toward the smaller
    regularization constant.
    """
    constants = _validate_lambda_grid(lambda_constants)
    controlled = {
        "lambda_value",
        "lambda_constant",
        "observation_mask",
        "precomputed_initialization",
        "initial_theta_G",
    }.intersection(fit_kwargs)
    if controlled:
        raise ValueError(
            "Oracle tuning controls these fit arguments: "
            f"{sorted(controlled)}."
        )
    if not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")

    Y_blocks, _ = _as_domain_blocks(data["Y"])
    full_masks = _prepare_observation_masks(Y_blocks, None)
    initialization = _spectral_initialization(
        Y_blocks,
        full_masks,
        fit_kwargs.get("theta_bound", 4.0),
        fit_kwargs.get("a_bound", 4.0),
        fit_kwargs.get("d_bound", 4.0),
    )
    tuning_start = perf_counter()

    def fit_candidate(constant: float) -> tuple[dict, dict]:
        fit_start = perf_counter()
        candidate = fit_and_evaluate_proposed_method(
            data,
            lambda_constant=float(constant),
            precomputed_initialization=initialization,
            **fit_kwargs,
        )
        record = {
            "C_lambda": float(constant),
            "lambda_value": float(candidate["lambda_value"]),
            "oracle_mean_theta_t_sine_error": float(
                candidate["mean_theta_t_sine_error"]
            ),
            "theta_G_sine_error": float(candidate["theta_G_sine_error"]),
            "max_theta_t_sine_error": float(
                candidate["max_theta_t_sine_error"]
            ),
            "mean_training_negative_log_likelihood": float(
                candidate["negative_log_likelihood"]
                / candidate["weighted_entries"]
            ),
            "converged": bool(candidate["converged"]),
            "iterations": int(candidate["iterations"]),
            "runtime_seconds": float(perf_counter() - fit_start),
        }
        return candidate, record

    workers = min(int(n_jobs), len(constants))
    if progress:
        print(
            f"  Oracle tuning: {len(constants)} candidate fits "
            f"using {workers} job(s)"
        )
    if workers == 1:
        fitted_candidates = [fit_candidate(constant) for constant in constants]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fitted_candidates = list(executor.map(fit_candidate, constants))

    candidate_results = [candidate for candidate, _ in fitted_candidates]
    path_records = [record for _, record in fitted_candidates]
    if progress:
        for record in path_records:
            print(
                f"    C_lambda={record['C_lambda']:.8g}, "
                "mean domain sine error="
                f"{record['oracle_mean_theta_t_sine_error']:.8g}, "
                f"iterations={record['iterations']}, "
                f"fit_time={record['runtime_seconds']:.3f}s"
            )

    best_index = min(
        range(len(path_records)),
        key=lambda index: (
            path_records[index]["oracle_mean_theta_t_sine_error"],
            path_records[index]["C_lambda"],
        ),
    )
    selected_constant = float(path_records[best_index]["C_lambda"])
    for index, record in enumerate(path_records):
        record["selected"] = bool(index == best_index)

    selected = candidate_results[best_index]
    selected.update(
        {
            "selected_C_lambda": selected_constant,
            "oracle_mean_theta_t_sine_error": float(
                path_records[best_index]["oracle_mean_theta_t_sine_error"]
            ),
            "oracle_tuning_runtime_seconds": float(
                perf_counter() - tuning_start
            ),
            "oracle_path_records": path_records,
        }
    )
    return selected


def _result_record(
    result: dict,
    data: dict,
    setting_id: int,
    repeat: int,
    seed: int,
    runtime: float,
    C_lambda: float | None,
) -> dict:
    iterations = np.asarray(result["iterations"])
    best_iteration = np.asarray(result["best_iteration"])
    converged = np.asarray(result["converged"])
    return {
        "setting_id": setting_id,
        "repeat": repeat,
        "seed": seed,
        "N": data["N"],
        "T": data["T"],
        "J_vec": data["J_vec_label"],
        "J_total": data["J_total"],
        "J_min": data["J_min"],
        "J_max": data["J_max"],
        "h_target": data["h_target"],
        "h_realized": data["h_realized"],
        "score_power": data["score_power"],
        "heterogeneity_structure": data["heterogeneity_structure"],
        "majority_proportion_target": float(
            data.get("majority_proportion_target", np.nan)
        ),
        "majority_proportion_realized": float(
            data.get("majority_proportion_realized", np.nan)
        ),
        "number_majority_domains": float(
            data.get("number_majority_domains", np.nan)
        ),
        "number_shifted_domains": float(
            data.get("number_shifted_domains", np.nan)
        ),
        "shifted_center_sine_distance": float(
            data.get("shifted_center_sine_distance", np.nan)
        ),
        "method": result["method"],
        "C_lambda": np.nan if C_lambda is None else float(C_lambda),
        "lambda_value": float(result.get("lambda_value", np.nan)),
        "theta_G_error": result["theta_G_error"],
        "mean_theta_t_error": result["mean_theta_t_error"],
        "max_theta_t_error": result["max_theta_t_error"],
        "theta_G_sine_error": result["theta_G_sine_error"],
        "mean_theta_t_sine_error": result["mean_theta_t_sine_error"],
        "max_theta_t_sine_error": result["max_theta_t_sine_error"],
        "theta_G_spearman_rho": result["theta_G_spearman_rho"],
        "theta_G_kendall_tau": result["theta_G_kendall_tau"],
        "mean_theta_t_spearman_rho": result["mean_theta_t_spearman_rho"],
        "min_theta_t_spearman_rho": result["min_theta_t_spearman_rho"],
        "mean_theta_t_kendall_tau": result["mean_theta_t_kendall_tau"],
        "min_theta_t_kendall_tau": result["min_theta_t_kendall_tau"],
        "mean_M_rmse": result["mean_M_rmse"],
        "mean_a_rmse": result["mean_a_rmse"],
        "mean_d_rmse": result["mean_d_rmse"],
        "probability_min": data["probability_min"],
        "probability_max": data["probability_max"],
        "probability_saturation_fraction": data["probability_saturation_fraction"],
        "minimum_domain_signal": data["minimum_domain_signal"],
        "aggregated_spectral_signal": data["aggregated_spectral_signal"],
        "generation_center_sine_error": data["generation_center_sine_error"],
        "numerical_common_sine_error": data["numerical_common_sine_error"],
        "population_common_objective": data["numerical_common_objective"],
        "population_common_iterations": data["population_common_iterations"],
        "population_common_converged": data["population_common_converged"],
        "population_common_tolerance": data["population_common_tolerance"],
        "population_common_constraint_mean": data[
            "population_common_constraint_mean"
        ],
        "population_common_constraint_norm_error": data[
            "population_common_constraint_norm_error"
        ],
        "population_common_orientation_inner_product": data[
            "population_common_orientation_inner_product"
        ],
        "objective": float(result["objective"]),
        "mean_log_likelihood": -float(
            result.get("raw_negative_log_likelihood", result["negative_log_likelihood"])
        ) / result["total_entries"],
        "weighted_mean_log_likelihood": -float(result["negative_log_likelihood"])
        / float(result.get("weighted_entries", result["total_entries"])),
        "oracle_mean_theta_t_sine_error": float(
            result.get("oracle_mean_theta_t_sine_error", np.nan)
        ),
        "oracle_tuning_runtime_seconds": float(
            result.get("oracle_tuning_runtime_seconds", np.nan)
        ),
        "cv_mean_test_nll": float(
            result.get("cv_mean_test_negative_log_likelihood", np.nan)
        ),
        "cv_fold_standard_error": float(
            result.get("cv_fold_standard_error", np.nan)
        ),
        "cv_n_folds": float(result.get("cv_n_folds", np.nan)),
        "cv_runtime_seconds": float(result.get("cv_runtime_seconds", np.nan)),
        "refit_runtime_seconds": float(
            result.get("refit_runtime_seconds", np.nan)
        ),
        "iterations_mean": float(iterations.mean()),
        "best_iteration_mean": float(best_iteration.mean()),
        "converged": bool(converged.all()),
        "runtime_seconds": float(runtime),
        "constraint_max_mean": result["estimated_theta_max_mean"],
        "constraint_max_norm_error": result["estimated_theta_max_norm_error"],
        "estimated_min_a": result["estimated_min_a"],
    }


def _domain_result_records(
    result: dict,
    data: dict,
    setting_id: int,
    repeat: int,
    seed: int,
    C_lambda: float | None,
) -> list[dict]:
    """Return one row per domain without averaging domain-level metrics."""
    domain_nll = np.asarray(result["evaluation_domain_nll"], dtype=float)
    return [
        {
            "setting_id": setting_id,
            "repeat": repeat,
            "seed": seed,
            "N": data["N"],
            "T": data["T"],
            "J_vec": data["J_vec_label"],
            "J_total": data["J_total"],
            "J_min": data["J_min"],
            "J_max": data["J_max"],
            "h_target": data["h_target"],
            "h_realized": data["h_realized"],
            "method": result["method"],
            "C_lambda": np.nan if C_lambda is None else float(C_lambda),
            "domain": t + 1,
            "J_t": int(data["J_vec"][t]),
            "alpha": float(data["alpha"][t]),
            "domain_angle": float(data["domain_angles"][t]),
            "majority_domain": bool(
                np.asarray(
                    data.get(
                        "majority_indicator",
                        np.zeros(data["T"], dtype=bool),
                    )
                )[t]
            ),
            "truth_center_group": (
                "majority_theta_G"
                if bool(
                    np.asarray(
                        data.get(
                            "majority_indicator",
                            np.zeros(data["T"], dtype=bool),
                        )
                    )[t]
                )
                else "shifted_theta_G_prime"
            )
            if data["heterogeneity_structure"] == "majority_two_centers"
            else "not_applicable",
            "theta_t_error": float(result["theta_t_errors"][t]),
            "theta_t_sine_error": float(result["theta_t_sine_errors"][t]),
            "theta_t_spearman_rho": float(result["theta_t_spearman_rhos"][t]),
            "theta_t_kendall_tau": float(result["theta_t_kendall_taus"][t]),
            "M_t_rmse": float(result["domain_M_rmse"][t]),
            "a_t_rmse": float(result["domain_a_rmse"][t]),
            "d_t_rmse": float(result["domain_d_rmse"][t]),
            "domain_mean_log_likelihood": float(
                -domain_nll[t] / (data["N"] * data["J_vec"][t])
            ),
            "theta_G_error": float(result["theta_G_error"]),
            "theta_G_sine_error": float(result["theta_G_sine_error"]),
            "theta_G_spearman_rho": float(result["theta_G_spearman_rho"]),
            "theta_G_kendall_tau": float(result["theta_G_kendall_tau"]),
        }
        for t in range(data["T"])
    ]


def run_simulation_grid(
    N_values: Sequence[int],
    T_values: Sequence[int],
    J_vector_values: Sequence[Sequence[int]],
    h_values: Sequence[float],
    *,
    C_lambda_values: Sequence[float] = (0.1, 0.2, 0.4, 0.8, 1.2),
    R: int = 5,
    tune_C_lambda: bool = True,
    tuning_jobs: int = 1,
    base_seed: int = 20260721,
    data_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the unequal-J simulation with per-dataset oracle tuning.

    Every simulated dataset selects its own C_lambda from the candidate grid
    by minimizing the mean domain-specific sine error.  This criterion uses
    the generated theta_t^* and is therefore intended only for simulation
    diagnostics.  Each element of
    ``J_vector_values`` is one length-T vector (J_1,...,J_T).  For example,
    with T=3, ``J_vector_values=[[20, 40, 80]]`` specifies one setting.
    """
    if R < 1:
        raise ValueError("R must be positive.")
    constants = _validate_lambda_grid(C_lambda_values)
    if not tune_C_lambda:
        constants = constants[:1]
    if not isinstance(tuning_jobs, (int, np.integer)) or tuning_jobs < 1:
        raise ValueError("tuning_jobs must be a positive integer.")
    h_grid = tuple(float(value) for value in h_values)
    if not h_grid:
        raise ValueError("h_values cannot be empty.")
    data_kwargs = {} if data_kwargs is None else dict(data_kwargs)
    supplied_shared_h = data_kwargs.get("shared_h_values")
    supplied_shared_h = () if supplied_shared_h is None else supplied_shared_h
    data_kwargs["shared_h_values"] = tuple(
        sorted(
            set(h_grid).union(float(value) for value in supplied_shared_h)
        )
    )
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    forbidden_fit = {
        "seed",
        "observation_mask",
        "lambda_constant",
        "lambda_value",
        "initial_theta_G",
        "precomputed_initialization",
    }.intersection(fit_kwargs)
    if forbidden_fit:
        raise ValueError(
            "The simulation driver controls these fit arguments: "
            f"{sorted(forbidden_fit)}."
        )
    settings: list[tuple[int, int, int, np.ndarray, float]] = []
    seed_group_id = 0
    for N in N_values:
        for T in T_values:
            for J_vector in J_vector_values:
                J_vec = _validate_J_vector(J_vector, int(T))
                seed_group_id += 1
                for h in h_grid:
                    settings.append(
                        (
                            seed_group_id,
                            int(N),
                            int(T),
                            J_vec.copy(),
                            float(h),
                        )
                    )
    records: list[dict] = []
    domain_records: list[dict] = []
    oracle_path_records: list[dict] = []
    dataset_total = len(settings) * R
    dataset_index = 0

    for setting_id, (seed_group_id, N, T, J_vec, h) in enumerate(
        settings, start=1
    ):
        for repeat in range(1, R + 1):
            dataset_index += 1
            data_seed = base_seed + (seed_group_id - 1) * R + repeat - 1
            data = generate_ai_measurement_data(
                N=N, T=T, J=J_vec, h=h, seed=data_seed, **data_kwargs
            )
            if progress:
                print(
                    f"Dataset {dataset_index}/{dataset_total}: N={N}, T={T}, "
                    f"J_vec={data['J_vec_label']}, h_target={h:.3f}, "
                    f"h_realized={data['h_realized']:.3f}, "
                    f"repeat={repeat}, seed={data_seed}"
                )

            if tune_C_lambda:
                start = perf_counter()
                proposed = oracle_tune_proposed_method(
                    data,
                    constants,
                    n_jobs=tuning_jobs,
                    progress=progress,
                    **fit_kwargs,
                )
                proposed_runtime = perf_counter() - start
                records.append(
                    _result_record(
                        proposed,
                        data,
                        setting_id,
                        repeat,
                        data_seed,
                        proposed_runtime,
                        float(proposed["selected_C_lambda"]),
                    )
                )
                domain_records.extend(
                    _domain_result_records(
                        proposed,
                        data,
                        setting_id,
                        repeat,
                        data_seed,
                        float(proposed["selected_C_lambda"]),
                    )
                )
                common_oracle_fields = {
                    "setting_id": setting_id,
                    "repeat": repeat,
                    "seed": data_seed,
                    "N": N,
                    "T": T,
                    "J_vec": data["J_vec_label"],
                    "J_total": data["J_total"],
                    "J_min": data["J_min"],
                    "J_max": data["J_max"],
                    "h_target": h,
                    "h_realized": float(data["h_realized"]),
                }
                oracle_path_records.extend(
                    [
                        {**common_oracle_fields, **record}
                        for record in proposed["oracle_path_records"]
                    ]
                )
                if progress:
                    print(
                        "  Proposed complete: "
                        f"C_lambda={proposed['selected_C_lambda']:.8g}, "
                        "oracle error="
                        f"{proposed['oracle_mean_theta_t_sine_error']:.8g}, "
                        "tuning="
                        f"{proposed['oracle_tuning_runtime_seconds']:.3f}s, "
                        f"total={proposed_runtime:.3f}s"
                    )
            else:
                constant = float(constants[0])
                start = perf_counter()
                proposed = fit_and_evaluate_proposed_method(
                    data,
                    lambda_constant=constant,
                    **fit_kwargs,
                )
                records.append(
                    _result_record(
                        proposed,
                        data,
                        setting_id,
                        repeat,
                        data_seed,
                        perf_counter() - start,
                        constant,
                    )
                )
                domain_records.extend(
                    _domain_result_records(
                        proposed,
                        data,
                        setting_id,
                        repeat,
                        data_seed,
                        constant,
                    )
                )

            start = perf_counter()
            local = fit_and_evaluate_local_method(data, **fit_kwargs)
            local_runtime = perf_counter() - start
            records.append(
                _result_record(
                    local,
                    data,
                    setting_id,
                    repeat,
                    data_seed,
                    local_runtime,
                    None,
                )
            )
            domain_records.extend(
                _domain_result_records(
                    local,
                    data,
                    setting_id,
                    repeat,
                    data_seed,
                    None,
                )
            )
            if progress:
                print(f"  Local complete: total={local_runtime:.3f}s")

            start = perf_counter()
            global_result = fit_and_evaluate_global_method(data, **fit_kwargs)
            global_runtime = perf_counter() - start
            records.append(
                _result_record(
                    global_result,
                    data,
                    setting_id,
                    repeat,
                    data_seed,
                    global_runtime,
                    None,
                )
            )
            domain_records.extend(
                _domain_result_records(
                    global_result,
                    data,
                    setting_id,
                    repeat,
                    data_seed,
                    None,
                )
            )
            if progress:
                print(f"  Global complete: total={global_runtime:.3f}s")

    raw = pd.DataFrame.from_records(records)
    raw["selected_C_lambda"] = raw["method"].eq("Proposed")
    domain_results = pd.DataFrame.from_records(domain_records)

    if oracle_path_records:
        oracle_path_raw = pd.DataFrame.from_records(oracle_path_records)
        lambda_path = (
            oracle_path_raw.groupby(
                [
                    "setting_id",
                    "N",
                    "T",
                    "J_vec",
                    "J_total",
                    "J_min",
                    "J_max",
                    "h_target",
                    "C_lambda",
                ],
                as_index=False,
            )
            .agg(
                R=("repeat", "size"),
                h_realized=("h_realized", "mean"),
                mean_oracle_theta_t_sine_error=(
                    "oracle_mean_theta_t_sine_error",
                    "mean",
                ),
                sd_oracle_theta_t_sine_error=(
                    "oracle_mean_theta_t_sine_error",
                    "std",
                ),
                mean_theta_G_sine_error=("theta_G_sine_error", "mean"),
                mean_max_theta_t_sine_error=(
                    "max_theta_t_sine_error",
                    "mean",
                ),
                mean_training_nll=(
                    "mean_training_negative_log_likelihood",
                    "mean",
                ),
                selection_rate=("selected", "mean"),
                mean_lambda_value=("lambda_value", "mean"),
                converged_rate=("converged", "mean"),
                runtime_seconds=("runtime_seconds", "mean"),
            )
            .sort_values(["setting_id", "C_lambda"])
            .reset_index(drop=True)
        )
    else:
        lambda_path = pd.DataFrame(
            columns=[
                "setting_id",
                "N",
                "T",
                "J_vec",
                "J_total",
                "J_min",
                "J_max",
                "h_target",
                "C_lambda",
                "R",
                "mean_oracle_theta_t_sine_error",
                "selection_rate",
            ]
        )
    tuning_details = (
        pd.DataFrame.from_records(oracle_path_records)
        .sort_values(["setting_id", "repeat", "C_lambda"])
        .reset_index(drop=True)
        if oracle_path_records
        else pd.DataFrame()
    )

    group_columns = [
        "setting_id",
        "N",
        "T",
        "J_vec",
        "J_total",
        "J_min",
        "J_max",
        "h_target",
        "method",
    ]
    summary = (
        raw.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            R=("repeat", "size"),
            C_lambda=("C_lambda", "mean"),
            C_lambda_median=("C_lambda", "median"),
            C_lambda_min=("C_lambda", "min"),
            C_lambda_max=("C_lambda", "max"),
            lambda_value=("lambda_value", "mean"),
            h_realized=("h_realized", "mean"),
            theta_G_error=("theta_G_error", "mean"),
            theta_G_error_sd=("theta_G_error", "std"),
            mean_theta_t_error=("mean_theta_t_error", "mean"),
            mean_theta_t_error_sd=("mean_theta_t_error", "std"),
            max_theta_t_error=("max_theta_t_error", "mean"),
            max_theta_t_error_sd=("max_theta_t_error", "std"),
            theta_G_sine_error=("theta_G_sine_error", "mean"),
            theta_G_sine_error_sd=("theta_G_sine_error", "std"),
            mean_theta_t_sine_error=("mean_theta_t_sine_error", "mean"),
            mean_theta_t_sine_error_sd=("mean_theta_t_sine_error", "std"),
            max_theta_t_sine_error=("max_theta_t_sine_error", "mean"),
            max_theta_t_sine_error_sd=("max_theta_t_sine_error", "std"),
            theta_G_spearman_rho=("theta_G_spearman_rho", "mean"),
            theta_G_spearman_rho_sd=("theta_G_spearman_rho", "std"),
            theta_G_kendall_tau=("theta_G_kendall_tau", "mean"),
            theta_G_kendall_tau_sd=("theta_G_kendall_tau", "std"),
            mean_theta_t_spearman_rho=("mean_theta_t_spearman_rho", "mean"),
            mean_theta_t_spearman_rho_sd=(
                "mean_theta_t_spearman_rho",
                "std",
            ),
            min_theta_t_spearman_rho=("min_theta_t_spearman_rho", "mean"),
            min_theta_t_spearman_rho_sd=("min_theta_t_spearman_rho", "std"),
            mean_theta_t_kendall_tau=("mean_theta_t_kendall_tau", "mean"),
            mean_theta_t_kendall_tau_sd=(
                "mean_theta_t_kendall_tau",
                "std",
            ),
            min_theta_t_kendall_tau=("min_theta_t_kendall_tau", "mean"),
            min_theta_t_kendall_tau_sd=("min_theta_t_kendall_tau", "std"),
            mean_M_rmse=("mean_M_rmse", "mean"),
            mean_M_rmse_sd=("mean_M_rmse", "std"),
            mean_a_rmse=("mean_a_rmse", "mean"),
            mean_a_rmse_sd=("mean_a_rmse", "std"),
            mean_d_rmse=("mean_d_rmse", "mean"),
            mean_d_rmse_sd=("mean_d_rmse", "std"),
            oracle_mean_theta_t_sine_error=(
                "oracle_mean_theta_t_sine_error",
                "mean",
            ),
            oracle_tuning_runtime_seconds=(
                "oracle_tuning_runtime_seconds",
                "mean",
            ),
            cv_mean_test_nll=("cv_mean_test_nll", "mean"),
            cv_fold_standard_error=("cv_fold_standard_error", "mean"),
            cv_n_folds=("cv_n_folds", "mean"),
            cv_runtime_seconds=("cv_runtime_seconds", "mean"),
            refit_runtime_seconds=("refit_runtime_seconds", "mean"),
            converged_rate=("converged", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        )
        .reset_index(drop=True)
    )
    mcse_metrics = [
        "theta_G_error",
        "mean_theta_t_error",
        "max_theta_t_error",
        "theta_G_sine_error",
        "mean_theta_t_sine_error",
        "max_theta_t_sine_error",
        "theta_G_spearman_rho",
        "theta_G_kendall_tau",
        "mean_theta_t_spearman_rho",
        "min_theta_t_spearman_rho",
        "mean_theta_t_kendall_tau",
        "min_theta_t_kendall_tau",
        "mean_M_rmse",
        "mean_a_rmse",
        "mean_d_rmse",
    ]
    for metric in mcse_metrics:
        summary[f"{metric}_mcse"] = summary[f"{metric}_sd"] / np.sqrt(
            summary["R"]
        )
    method_order = {"Proposed": 0, "Local": 1, "Global": 2}
    summary["_order"] = summary["method"].map(method_order)
    summary = summary.sort_values(["setting_id", "_order"]).drop(columns="_order")
    return {
        "summary": summary.reset_index(drop=True),
        "raw": raw.sort_values(
            ["setting_id", "repeat", "method", "C_lambda"], na_position="last"
        ).reset_index(drop=True),
        "domains": domain_results.sort_values(
            ["setting_id", "repeat", "method", "domain", "C_lambda"],
            na_position="last",
        ).reset_index(drop=True),
        "lambda_path": lambda_path,
        "oracle_tuning": tuning_details,
    }


def save_simulation_results(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save all simulation tables in a newly created output directory.

    Refusing an existing directory prevents an accidental overwrite if a
    caller supplies a non-unique name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "summary": output_dir / "simulation_summary.csv",
        "raw": output_dir / "simulation_raw_results.csv",
        "domains": output_dir / "simulation_domain_results.csv",
        "lambda_path": output_dir / "simulation_lambda_path.csv",
        "oracle_tuning": output_dir / "simulation_oracle_tuning_results.csv",
    }
    for name, path in paths.items():
        results[name].to_csv(path, index=False)
    return paths


def _set_method_label(result: dict, label: str) -> dict:
    """Set a plotting/output label without changing fitted parameters."""
    result["method"] = label
    return result


def _append_result_and_domains(
    records: list[dict],
    domain_records: list[dict],
    result: dict,
    data: dict,
    setting_id: int,
    repeat: int,
    seed: int,
    runtime: float,
    C_lambda: float | None,
) -> None:
    record = _result_record(
        result,
        data,
        setting_id,
        repeat,
        seed,
        runtime,
        C_lambda,
    )
    record["theta_G_frobenius_error"] = record["theta_G_error"]
    record["mean_theta_t_frobenius_error"] = record["mean_theta_t_error"]
    records.append(record)

    rows = _domain_result_records(
        result,
        data,
        setting_id,
        repeat,
        seed,
        C_lambda,
    )
    for row in rows:
        row["theta_t_frobenius_error"] = row["theta_t_error"]
    domain_records.extend(rows)


def _build_comparison_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Monte Carlo summary for Proposed-CV, Local, and Global."""
    group_columns = [
        "setting_id",
        "N",
        "T",
        "J_vec",
        "J_total",
        "J_min",
        "J_max",
        "h_target",
        "method",
    ]
    metric_columns = [
        "theta_G_sine_error",
        "theta_G_frobenius_error",
        "theta_G_spearman_rho",
        "theta_G_kendall_tau",
        "mean_theta_t_sine_error",
        "mean_theta_t_frobenius_error",
        "mean_theta_t_spearman_rho",
        "mean_theta_t_kendall_tau",
    ]

    named_agg: dict[str, tuple[str, str]] = {
        "R": ("repeat", "size"),
        "C_lambda": ("C_lambda", "mean"),
        "C_lambda_sd": ("C_lambda", "std"),
        "C_lambda_median": ("C_lambda", "median"),
        "h_realized": ("h_realized", "mean"),
        "population_common_iterations": ("population_common_iterations", "mean"),
        "population_common_converged_rate": ("population_common_converged", "mean"),
        "converged_rate": ("converged", "mean"),
        "runtime_seconds": ("runtime_seconds", "mean"),
        "cv_mean_test_nll": ("cv_mean_test_nll", "mean"),
    }
    for metric in metric_columns:
        named_agg[metric] = (metric, "mean")
        named_agg[f"{metric}_sd"] = (metric, "std")

    summary = (
        raw.groupby(group_columns, as_index=False, dropna=False)
        .agg(**named_agg)
        .reset_index(drop=True)
    )
    for metric in metric_columns:
        summary[f"{metric}_mcse"] = (
            summary[f"{metric}_sd"] / np.sqrt(summary["R"])
        )

    method_order = {
        "Proposed-CV": 0,
        "Local": 1,
        "Global": 2,
    }
    summary["_order"] = summary["method"].map(method_order)
    return (
        summary.sort_values(["setting_id", "_order"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _validate_method_fit_kwargs(
    method_name: str,
    fit_kwargs: dict,
) -> None:
    """Reject arguments controlled internally by the comparison/CV driver."""
    forbidden = {
        "seed",
        "observation_mask",
        "lambda_constant",
        "lambda_value",
        "initial_theta_G",
        "precomputed_initialization",
    }.intersection(fit_kwargs)
    if forbidden:
        raise ValueError(
            f"{method_name} fit kwargs contain driver-controlled arguments: "
            f"{sorted(forbidden)}."
        )


def run_three_estimator_comparison(
    N_values: Sequence[int],
    T_values: Sequence[int],
    J_vector_values: Sequence[Sequence[int]],
    h_values: Sequence[float],
    *,
    C_lambda_values: Sequence[float],
    R: int = 100,
    cv_folds: int = 5,
    cv_jobs: int = 1,
    base_seed: int = 20260810,
    proposed_theta_perturbation: float = 0.10,
    data_kwargs: dict | None = None,
    proposed_cv_fit_kwargs: dict | None = None,
    local_fit_kwargs: dict | None = None,
    global_fit_kwargs: dict | None = None,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compare Proposed-CV, Local, and Global on exactly the same paired data.

    This diagnostic version fits Global first and uses the fitted Global
    parameters to initialize both Proposed-CV and Local.  For Proposed-CV,
    theta_G starts at the fitted Global shared trait, while each domain trait
    starts at an exact sine-distance ``proposed_theta_perturbation`` from that
    fitted Global trait along a reproducible orthogonal tangent direction.
    The item parameters start at fitted Global (a_t, d_t).  Every CV
    fold/candidate and the final full-data refit use exactly this same
    perturbed initialization.  Local uses the unperturbed fitted Global
    parameters as the
    starting point for every domain-specific fit, and its post-hoc common-trait
    optimization is initialized at the fitted Global shared trait.  Global
    itself retains the original truth initialization.

    Each estimator has its own optimization dictionary:

    ``proposed_cv_fit_kwargs``
        Used for every Proposed-CV training fit and the final full-data refit.

    ``local_fit_kwargs``
        Used for every domain-specific Local fit.  Its ``max_iter`` and
        ``tolerance`` are also passed to the post-hoc common-trait optimization
        that computes theta_G^0.

    ``global_fit_kwargs``
        Used only for the lambda=infinity Global fit.

    The population target theta_G^* is computed during data generation and is
    controlled separately by ``population_common_trait_max_iter`` and
    ``population_common_trait_tolerance`` inside ``data_kwargs``.
    """
    if R < 1:
        raise ValueError("R must be positive.")
    constants = _validate_lambda_grid(C_lambda_values)
    if np.any(constants <= 0.0):
        raise ValueError(
            "The Proposed-CV grid should contain positive C_lambda values; "
            "Local already represents lambda=0."
        )
    if not isinstance(cv_jobs, (int, np.integer)) or cv_jobs < 1:
        raise ValueError("cv_jobs must be a positive integer.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if not 0.0 <= proposed_theta_perturbation < 1.0:
        raise ValueError(
            "proposed_theta_perturbation must lie in [0, 1)."
        )

    h_grid = tuple(float(value) for value in h_values)
    if not h_grid:
        raise ValueError("h_values cannot be empty.")

    data_kwargs = {} if data_kwargs is None else dict(data_kwargs)
    supplied_shared_h = data_kwargs.get("shared_h_values")
    supplied_shared_h = () if supplied_shared_h is None else supplied_shared_h
    data_kwargs["shared_h_values"] = tuple(
        sorted(set(h_grid).union(float(x) for x in supplied_shared_h))
    )

    proposed_cv_fit_kwargs = (
        {} if proposed_cv_fit_kwargs is None else dict(proposed_cv_fit_kwargs)
    )
    local_fit_kwargs = (
        {} if local_fit_kwargs is None else dict(local_fit_kwargs)
    )
    global_fit_kwargs = (
        {} if global_fit_kwargs is None else dict(global_fit_kwargs)
    )
    _validate_method_fit_kwargs("Proposed-CV", proposed_cv_fit_kwargs)
    _validate_method_fit_kwargs("Local", local_fit_kwargs)
    _validate_method_fit_kwargs("Global", global_fit_kwargs)

    settings: list[tuple[int, int, int, np.ndarray, float]] = []
    seed_group_id = 0
    for N in N_values:
        for T in T_values:
            for J_vector in J_vector_values:
                J_vec = _validate_J_vector(J_vector, int(T))
                seed_group_id += 1
                for h in h_grid:
                    settings.append(
                        (seed_group_id, int(N), int(T), J_vec.copy(), float(h))
                    )

    records: list[dict] = []
    domain_records: list[dict] = []
    cv_path_records: list[dict] = []
    cv_fold_records: list[dict] = []

    dataset_total = len(settings) * R
    dataset_index = 0

    for setting_id, (seed_group_id, N, T, J_vec, h) in enumerate(
        settings, start=1
    ):
        for repeat in range(1, R + 1):
            dataset_index += 1
            data_seed = base_seed + (seed_group_id - 1) * R + repeat - 1
            data = generate_ai_measurement_data(
                N=N,
                T=T,
                J=J_vec,
                h=h,
                seed=data_seed,
                **data_kwargs,
            )

            truth_global_initialization = (
                np.asarray(data["theta_G_star"], dtype=float)[None, :].copy(),
                [
                    np.concatenate(
                        [np.asarray(x, dtype=float) for x in data["a_star"]]
                    )
                ],
                [
                    np.concatenate(
                        [np.asarray(x, dtype=float) for x in data["d_star"]]
                    )
                ],
                np.asarray(data["theta_G_star"], dtype=float).copy(),
            )
            if progress:
                print(
                    f"Dataset {dataset_index}/{dataset_total}: "
                    f"N={N}, T={T}, J_vec={data['J_vec_label']}, "
                    f"h={h:.3f}, repeat={repeat}, seed={data_seed}"
                )
                print(
                    "  Population common trait: "
                    f"iterations={data['population_common_iterations']}, "
                    f"converged={data['population_common_converged']}"
                )

            common_fields = {
                "setting_id": setting_id,
                "repeat": repeat,
                "seed": data_seed,
                "N": N,
                "T": T,
                "J_vec": data["J_vec_label"],
                "J_total": data["J_total"],
                "J_min": data["J_min"],
                "J_max": data["J_max"],
                "h_target": h,
                "h_realized": float(data["h_realized"]),
            }

            start = perf_counter()
            global_result = fit_and_evaluate_global_method(
                data,
                precomputed_initialization=truth_global_initialization,
                **global_fit_kwargs,
            )
            global_runtime = perf_counter() - start
            _append_result_and_domains(
                records,
                domain_records,
                global_result,
                data,
                setting_id,
                repeat,
                data_seed,
                global_runtime,
                None,
            )
            if progress:
                print(
                    "  Global: "
                    f"iterations={global_result['iterations']}, "
                    f"converged={global_result['converged']}, "
                    f"time={global_runtime:.2f}s"
                )

            global_a_initialization = [
                np.asarray(
                    global_result["a_hat"][t, : int(J_vec[t])],
                    dtype=float,
                ).copy()
                for t in range(T)
            ]
            global_d_initialization = [
                np.asarray(
                    global_result["d_hat"][t, : int(J_vec[t])],
                    dtype=float,
                ).copy()
                for t in range(T)
            ]
            fitted_global_theta = np.asarray(
                global_result["theta_G_hat"], dtype=float
            ).copy()

            global_domain_initialization = (
                np.repeat(
                    fitted_global_theta[None, :],
                    T,
                    axis=0,
                ),
                [x.copy() for x in global_a_initialization],
                [x.copy() for x in global_d_initialization],
                fitted_global_theta.copy(),
            )

            proposed_theta_initialization = (
                _global_perturbed_theta_initialization(
                    fitted_global_theta,
                    T,
                    proposed_theta_perturbation,
                    theta_bound=proposed_cv_fit_kwargs.get(
                        "theta_bound", 4.0
                    ),
                    seed=data_seed + 700_001,
                )
            )
            proposed_global_perturbed_initialization = (
                proposed_theta_initialization,
                [x.copy() for x in global_a_initialization],
                [x.copy() for x in global_d_initialization],
                fitted_global_theta.copy(),
            )

            start = perf_counter()
            cv_fit = cross_validate_proposed_method(
                data["Y"],
                constants,
                n_folds=cv_folds,
                seed=data_seed,
                n_jobs=cv_jobs,
                progress=False,
                precomputed_initialization=proposed_global_perturbed_initialization,
                **proposed_cv_fit_kwargs,
            )

            cv_fit = _reoptimize_proposed_general_trait(
                cv_fit,
                initial_theta_G=fitted_global_theta,
                max_iter=int(
                    proposed_cv_fit_kwargs.get("max_iter", 1000)
                ),
                tolerance=float(
                    proposed_cv_fit_kwargs.get("tolerance", 1e-3)
                ),
            )
            cv_result = evaluate_estimator(cv_fit, data)
            cv_runtime_total = perf_counter() - start
            _set_method_label(cv_result, "Proposed-CV")
            _append_result_and_domains(
                records,
                domain_records,
                cv_result,
                data,
                setting_id,
                repeat,
                data_seed,
                cv_runtime_total,
                float(cv_result["selected_C_lambda"]),
            )
            cv_path_records.extend(
                [
                    {
                        **common_fields,
                        "method": "Proposed-CV",
                        **path_record,
                    }
                    for path_record in cv_result["cv_path_records"]
                ]
            )
            cv_fold_records.extend(
                [
                    {
                        **common_fields,
                        "method": "Proposed-CV",
                        **fold_record,
                    }
                    for fold_record in cv_result["cv_fold_records"]
                ]
            )
            if progress:
                print(
                    "  Proposed-CV (Global+perturb init; post-hoc G): "
                    f"perturb={proposed_theta_perturbation:.4g}, "
                    f"C={cv_result['selected_C_lambda']:.4g}, "
                    f"CV NLL={cv_result['cv_mean_test_negative_log_likelihood']:.6g}, "
                    f"iterations={cv_result['iterations']}, "
                    f"converged={cv_result['converged']}, "
                    f"time={cv_runtime_total:.2f}s"
                )

            start = perf_counter()
            local = fit_and_evaluate_local_method(
                data,
                initial_theta_G=np.asarray(
                    global_result["theta_G_hat"], dtype=float
                ),
                precomputed_initialization=global_domain_initialization,
                **local_fit_kwargs,
            )
            local_runtime = perf_counter() - start
            _append_result_and_domains(
                records,
                domain_records,
                local,
                data,
                setting_id,
                repeat,
                data_seed,
                local_runtime,
                None,
            )
            if progress:
                print(
                    "  Local: "
                    f"mean iterations={np.mean(local['iterations']):.1f}, "
                    f"all converged={bool(np.all(local['converged']))}, "
                    f"common iterations={local['common_trait_iterations']}, "
                    f"common converged={local['common_trait_converged']}, "
                    f"time={local_runtime:.2f}s"
                )

    raw = pd.DataFrame.from_records(records)
    domains = pd.DataFrame.from_records(domain_records)
    summary = _build_comparison_summary(raw)

    cv_tuning = (
        pd.DataFrame.from_records(cv_path_records)
        .sort_values(["setting_id", "repeat", "C_lambda"])
        .reset_index(drop=True)
        if cv_path_records
        else pd.DataFrame()
    )
    cv_folds_table = (
        pd.DataFrame.from_records(cv_fold_records)
        .sort_values(["setting_id", "repeat", "fold", "C_lambda"])
        .reset_index(drop=True)
        if cv_fold_records
        else pd.DataFrame()
    )

    return {
        "summary": summary,
        "raw": raw.sort_values(
            ["setting_id", "repeat", "method"],
            na_position="last",
        ).reset_index(drop=True),
        "domains": domains.sort_values(
            ["setting_id", "repeat", "method", "domain"],
            na_position="last",
        ).reset_index(drop=True),
        "cv_tuning": cv_tuning,
        "cv_folds": cv_folds_table,
    }


def save_three_estimator_comparison(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save all tables for the three-estimator comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "summary": output_dir / "simulation_summary.csv",
        "raw": output_dir / "simulation_raw_results.csv",
        "domains": output_dir / "simulation_domain_results.csv",
        "cv_tuning": output_dir / "simulation_cv_tuning_results.csv",
        "cv_folds": output_dir / "simulation_cv_fold_results.csv",
    }
    for name, path in paths.items():
        results[name].to_csv(path, index=False)
    return paths

