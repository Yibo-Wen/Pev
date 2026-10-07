"""The metrics the discovery screen reports.

Six views of one ranking, chosen so no single number carries the result: the
yield in ``k`` wells, that yield over the panel's own base rate, how much of the
achievable gain was recovered, the same with a position discount, budget-free
ranking quality, and the wells spent before the first hit. Intervals come from a
bootstrap over whole families, because the panels of one family are not
independent draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_at_k(labels: np.ndarray, ranking: np.ndarray, k: int) -> float:
    """Beneficial fraction among the ``k`` edits a lab would actually assay."""
    k = min(int(k), int(np.asarray(labels).size))
    if k <= 0:
        return float("nan")
    return float(np.asarray(labels, dtype=float)[ranking[:k]].mean())


def enrichment_at_k(labels: np.ndarray, ranking: np.ndarray, k: int) -> float:
    """``Precision@k`` over the pool's own beneficial prevalence.

    The denominator is what makes panels of different difficulty comparable -- a
    raw hit rate would rank a method by which panels it drew. A uniform ranking
    scores exactly 1 whatever the panel. NaN when nothing in the pool is
    beneficial, since scoring 0 would punish methods for an impossible panel.
    """
    base = float(np.asarray(labels, dtype=float).mean()) if np.size(labels) else 0.0
    if base <= 0:
        return float("nan")
    return precision_at_k(labels, ranking, k) / base


def improvement_ratio(utilities: np.ndarray, selected: np.ndarray) -> float:
    """The fraction of the achievable gain the selection recovers.

    The denominator is always the **full** panel's best, so every method and
    every budget is scored against the same target. Zero when no edit in the full
    panel improves.
    """
    if utilities.size == 0:
        return 0.0
    best_available = max(float(utilities.max()), 0.0)
    if best_available <= 0:
        return 0.0
    best_selected = max(float(utilities[selected].max()) if selected.size else 0.0, 0.0)
    return best_selected / best_available


def ndcg_at_k(utilities: np.ndarray, ranking: np.ndarray, k: int) -> float:
    """Discounted gain over the ideal ordering, with ``relevance = max(u, 0)``.

    Graded rather than binary relevance, because a screen that surfaces the one
    large win has done something a screen that surfaces ten marginal ones has
    not. Harmful edits get relevance 0 -- they are not negatively relevant, they
    are simply not worth a well.
    """
    rel = np.clip(np.asarray(utilities, dtype=float), 0.0, None)
    k = min(int(k), rel.size)
    if k <= 0:
        return float("nan")
    discount = 1.0 / np.log2(np.arange(k) + 2.0)
    ideal = float((np.sort(rel)[::-1][:k] * discount).sum())
    if ideal <= 0:
        return float("nan")
    return float((rel[ranking[:k]] * discount).sum() / ideal)


def panel_auroc(labels: np.ndarray, ranking: np.ndarray) -> float:
    """P(a beneficial edit is ranked above a non-beneficial one), within one panel.

    The metric to reach for when ``Enrichment@K`` saturates: 0.5 for any ranking
    that ignores the labels and 1.0 for a perfect one, whatever the panel size and
    prevalence, where enrichment's ceiling is ``min(1/p, n/K)``. NaN when the panel
    is all hits or no hits -- there is no pair to order.
    """
    y = np.asarray(labels, dtype=bool)
    if y.size == 0 or y.all() or not y.any():
        return float("nan")
    position = np.empty(y.size, dtype=float)
    position[np.asarray(ranking, dtype=int)] = np.arange(y.size, dtype=float)
    # Mann-Whitney: rank sum of the positives, with 1 = best.
    rank = pd.Series(-position).rank().to_numpy()
    hits, misses = float(y.sum()), float((~y).sum())
    return float((rank[y].sum() - hits * (hits + 1) / 2) / (hits * misses))


def assays_to_first_hit(labels: np.ndarray, ranking: np.ndarray) -> float:
    """How many wells a lab spends before the first beneficial edit. Lower is better.

    A perfect ranker spends 1. NaN on a panel with no beneficial edit -- there is
    no hit to find.
    """
    y = np.asarray(labels, dtype=bool)
    if not y.any():
        return float("nan")
    return float(np.flatnonzero(y[np.asarray(ranking, dtype=int)])[0] + 1)


def family_cluster_bootstrap(
    values: np.ndarray,
    families: np.ndarray,
    *,
    n_resamples: int = 1000,
    statistic=np.mean,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Resample whole families, not rows.

    The panels of one family share a reference protein and are not independent
    trials, so resampling rows would understate the interval. Returns
    ``(point, lower, upper)``.
    """
    values = np.asarray(values, dtype=float)
    families = np.asarray(families)
    point = float(statistic(values)) if values.size else float("nan")
    unique = np.unique(families)
    if values.size == 0 or unique.size < 2:
        return point, float("nan"), float("nan")

    index: dict = {f: np.flatnonzero(families == f) for f in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        picked = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index[f] for f in picked])
        draws[i] = statistic(values[rows])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    families: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap the paired difference ``a - b`` over shared families.

    Pairing matters: both methods see identical panels, so the difference has far
    less variance than the two marginals, and comparing independent intervals
    would be needlessly conservative.
    """
    return family_cluster_bootstrap(
        np.asarray(a, dtype=float) - np.asarray(b, dtype=float),
        families,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
    )
