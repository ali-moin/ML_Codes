"""Accelerated proximal algorithms for penalised linear models.

This module started as a faithful Python translation of the MATLAB
group-lasso routine (``proximal.m`` together with ``soft_threshodg.m``).
It now exposes a unified accelerated proximal solver capable of handling
Ridge, Lasso, Elastic Net, and Group Lasso penalties while operating on the
pre-computed sufficient statistics ``XX`` and ``XY``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _f_grad(xx: np.ndarray, xy: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return the gradient of the quadratic loss at ``w``.

    The objective is ``0.5 * w' * XX * w - XY' * w``; taking derivatives with
    respect to ``w`` yields ``XX @ w - XY``. This mirrors the MATLAB helper
    ``f_grad.m``.
    """

    return xx @ w - xy


def _soft_threshold_group(
    groups: np.ndarray,
    w: np.ndarray,
    mu: float,
    n_groups: Optional[int] = None,
) -> np.ndarray:
    """Apply the group soft-thresholding operator.

    Args:
        groups: Integer vector assigning every coefficient to a group. The
            contents must align with MATLAB's ``groups`` input where each entry
            is an integer in ``{1, ..., nc}``.
        w: Current iterate to be thresholded.
        mu: Shrinkage level (``lambda * step_size``).
        n_groups: Optional number of distinct groups. When omitted, the value
            is inferred from ``groups``.

    Returns:
        Thresholded coefficients with each group shrunk toward zero according
        to its Euclidean norm.
    """

    w_new = w.copy()
    if n_groups is None:
        unique_groups = np.unique(groups)
    else:
        unique_groups = range(1, n_groups + 1)

    for g in unique_groups:
        mask = groups == g
        wg = w_new[mask]
        norm = np.linalg.norm(wg)
        if norm <= mu:
            w_new[mask] = 0.0
        else:
            w_new[mask] = wg - mu * wg / norm
    return w_new


def _soft_threshold_scalar(w: np.ndarray, mu: float) -> np.ndarray:
    """Component-wise soft-thresholding used for the lasso penalty."""

    return np.sign(w) * np.maximum(np.abs(w) - mu, 0.0)


def _apply_prox(
    w: np.ndarray,
    step: float,
    penalty: float,
    penalty_type: str,
    *,
    groups: Optional[np.ndarray] = None,
    n_groups: Optional[int] = None,
    l1_ratio: Optional[float] = None,
) -> np.ndarray:
    """Dispatch to the appropriate proximal operator for the penalty."""

    penalty_type = penalty_type.lower()

    if penalty_type == "ridge":
        if penalty < 0:
            raise ValueError("Ridge penalty must be non-negative.")
        return w / (1.0 + penalty * step)

    if penalty_type == "lasso":
        if penalty < 0:
            raise ValueError("Lasso penalty must be non-negative.")
        return _soft_threshold_scalar(w, penalty * step)

    if penalty_type == "elastic_net":
        if l1_ratio is None:
            raise ValueError("Elastic Net requires an l1_ratio in [0, 1].")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must lie between 0 and 1 inclusive.")
        if penalty < 0:
            raise ValueError("Elastic Net penalty must be non-negative.")
        l1 = penalty * l1_ratio
        l2 = penalty * (1.0 - l1_ratio)
        w = _soft_threshold_scalar(w, l1 * step)
        return w / (1.0 + l2 * step)

    if penalty_type == "group_lasso":
        if groups is None:
            raise ValueError("Group Lasso requires a groups array.")
        if penalty < 0:
            raise ValueError("Group Lasso penalty must be non-negative.")
        return _soft_threshold_group(groups, w, penalty * step, n_groups=n_groups)

    raise ValueError(
        "Unsupported penalty_type. Expected one of {'ridge', 'lasso', "
        "'elastic_net', 'group_lasso'}."
    )


def accelerated_proximal_solver(
    xx: np.ndarray,
    xy: np.ndarray,
    lipschitz: float,
    penalty: float,
    tol: float,
    *,
    penalty_type: str = "group_lasso",
    groups: Optional[np.ndarray] = None,
    n_groups: Optional[int] = None,
    l1_ratio: Optional[float] = None,
    max_iter: int = 30_000,
) -> np.ndarray:
    """Solve penalised quadratic objectives via an accelerated proximal method.

    The optimisation is performed on the quadratic surrogate defined by
    ``XX`` and ``XY`` while applying the requested penalty through the
    corresponding proximal operator.

    Args:
        xx: Symmetric Gram matrix ``X'X``.
        xy: Cross-product ``X'y``.
        lipschitz: Lipschitz constant ``L`` used to set the step size.
        penalty: Regularisation strength (interpretation depends on
            ``penalty_type``).
        tol: Relative tolerance for the stopping criterion.
        penalty_type: One of ``{"ridge", "lasso", "elastic_net", "group_lasso"}``.
        groups: Integer array assigning each coefficient to a group (only for
            ``penalty_type="group_lasso"``).
        n_groups: Optional explicit number of groups; inferred when ``None``.
        l1_ratio: Fraction of the Elastic Net penalty applied to the L1 term.
        max_iter: Maximum number of iterations (default 30,000 as in MATLAB).

    Returns:
        The coefficient vector that minimises the penalised objective.
    """

    dim = xx.shape[0]
    step = 1.0 / lipschitz

    w = np.zeros(dim)
    v = np.zeros(dim)

    for t in range(max_iter):
        v_old = v.copy()
        w_prev = w.copy()

        grad = _f_grad(xx, xy, v)
        w = v - step * grad
        w = _apply_prox(
            w,
            step,
            penalty,
            penalty_type,
            groups=groups,
            n_groups=n_groups,
            l1_ratio=l1_ratio,
        )

        if t >= 0:
            v = w + (t / (t + 3.0)) * (w - w_prev)
        else:
            v = w

        diff = v - v_old
        if np.sum(diff**2) < np.sum(v_old**2) * tol or np.allclose(diff, 0.0):
            break

    return v


def accelerated_proximal_group_lasso(
    groups: np.ndarray,
    xx: np.ndarray,
    xy: np.ndarray,
    lipschitz: float,
    penalty: float,
    tol: float,
    n_groups: Optional[int] = None,
    max_iter: int = 30_000,
) -> np.ndarray:
    """Backward-compatible wrapper that executes the group-lasso solver."""

    return accelerated_proximal_solver(
        xx,
        xy,
        lipschitz,
        penalty,
        tol,
        penalty_type="group_lasso",
        groups=groups,
        n_groups=n_groups,
        max_iter=max_iter,
    )


__all__ = [
    "accelerated_proximal_solver",
    "accelerated_proximal_group_lasso",
]
