"""
gnina_vs.enrichment
===================
Retrospective validation: does this pipeline actually enrich known actives
over decoys on a target where the answer is already known?

Why this module is not optional
-------------------------------
Without it, the pipeline emits CNNaffinity deltas, rank bands and ligand
efficiencies with nothing establishing that any of it separates actives from
decoys. Those numbers are then not *wrong* so much as **uninterpretable** --
there is no evidence about what a good score means on this target. This is
also the first thing an examiner or reviewer asks about.

What this module provides, and what it does not
-----------------------------------------------
It provides the metrics and their uncertainty: ROC AUC, semi-log AUC,
enrichment factor at a fraction, BEDROC, each with a bootstrap confidence
interval, plus a label-permutation null. It also provides property-matched
decoy selection from a pool you supply.

It does **not** provide a decoy library, and it does not run docking. You
supply actives and decoys, dock them through the normal pipeline, and feed
the scored table back in. LIT-PCBA or property-matched decoys on a single
target is enough to make the pipeline's numbers interpretable.

NOT YET EXERCISED AGAINST REAL DOCKING OUTPUT. Every function here is unit
tested against synthetic rankings with known answers (perfect, random,
inverted), but no run of this code against actual Gnina scores has happened.
Treat the first real run as part of the validation, not as a formality.

Two things the literature is emphatic about
-------------------------------------------
**Report the interval, not the point.** Bootstrap confidence intervals on
enrichment curves are wide -- wide enough that most published differences
between methods are not distinguishable from each other -- and the practice
of declaring one method better from point estimates alone is exactly what
the enrichment-metric critiques target. Every function here returns an
interval, and `evaluate` refuses to report a point estimate without one.

**Decoy construction decides the answer.** The best-documented case: an
apparent optimisation gain of +1.11 logAUC on DUD-E decoys became **-0.34**
on properly charge-matched decoys. A pipeline that looks good against
decoys which differ from the actives in net charge or molecular weight has
measured the property difference, not the binding. `match_decoys` matches on
properties and records what it matched on; `charge_balance_report` is there
to catch the specific failure above before it flatters a result.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EF_FRACTIONS = (0.005, 0.01, 0.05)
DEFAULT_BEDROC_ALPHA = 20.0
DEFAULT_LOGAUC_LAMBDA = 0.001
DEFAULT_BOOTSTRAP = 2000
RNG_SEED = 0


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _as_ranking(scores, labels, higher_is_better: bool) -> tuple[np.ndarray, np.ndarray]:
    """Validate, drop unscored entries, and orient so that larger is better.

    The sign convention is the trap here: CNNaffinity is pKd-like (higher is
    better) while Vina affinity is kcal/mol (more negative is better). Getting
    it backwards turns a good result into a mirror-image bad one, which is why
    `higher_is_better` has no default anywhere in this module.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels differ in length: {scores.shape} vs {labels.shape}")
    if scores.size == 0:
        raise ValueError("no scores given")

    labels = labels.astype(bool)
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("every score is NaN or infinite -- nothing to evaluate")
    if not finite.all():
        # A failed docking is not a weak binder. Dropping is the honest choice
        # and the count is logged, because treating it as "worst" would invent
        # a ranking for a compound that was never scored.
        logger.warning(
            "  [enrichment] %d of %d compounds have no score and are excluded "
            "(%d actives, %d decoys).",
            int((~finite).sum()), finite.size,
            int(labels[~finite].sum()), int((~labels[~finite]).sum()),
        )
        scores, labels = scores[finite], labels[finite]

    n_act, n_dec = int(labels.sum()), int((~labels).sum())
    if n_act == 0 or n_dec == 0:
        raise ValueError(f"need both actives and decoys, got {n_act} actives and {n_dec} decoys")

    return (scores if higher_is_better else -scores), labels


def _tpr_fpr(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC points, with ties handled by sorting them together."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)

    # Collapse tied scores to a single operating point -- otherwise the curve
    # depends on the arbitrary order of equal scores.
    keep = np.append(np.diff(sorted_scores) != 0, True)
    tp, fp = tp[keep], fp[keep]

    tpr = np.concatenate([[0.0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0.0], fp / max(fp[-1], 1)])
    return tpr, fpr


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def roc_auc(scores, labels, higher_is_better: bool) -> float:
    """Area under the ROC curve. 0.5 is random, 1.0 perfect."""
    s, y = _as_ranking(scores, labels, higher_is_better)
    tpr, fpr = _tpr_fpr(s, y)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def log_auc(scores, labels, higher_is_better: bool,
            lam: float = DEFAULT_LOGAUC_LAMBDA, adjusted: bool = True) -> float:
    """Semi-log ROC AUC from ``lam`` to 1, which weights early recognition.

    With ``adjusted=True`` the random-classifier baseline is subtracted, so 0
    is random and positive is better than random. The baseline is computed
    numerically for this ``lam`` rather than hard-coded, because it depends on
    it.
    """
    if not 0 < lam < 1:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    s, y = _as_ranking(scores, labels, higher_is_better)
    tpr, fpr = _tpr_fpr(s, y)

    def _area(x: np.ndarray, t: np.ndarray) -> float:
        x = np.clip(x, lam, 1.0)
        logx = np.log10(x)
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(trap(t, logx) / np.log10(1.0 / lam))

    value = _area(fpr, tpr)
    if not adjusted:
        return value
    grid = np.logspace(np.log10(lam), 0.0, 512)
    return value - _area(grid, grid)


def enrichment_factor(scores, labels, higher_is_better: bool, fraction: float = 0.01) -> float:
    """Actives in the top ``fraction`` relative to what chance would give.

    1.0 is random. The maximum is 1/fraction, and is lower when there are
    more actives than the top slice can hold -- `max_enrichment_factor`
    reports that ceiling, because an EF quoted without it is unreadable.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    s, y = _as_ranking(scores, labels, higher_is_better)
    n = s.size
    k = max(int(np.ceil(fraction * n)), 1)
    order = np.argsort(-s, kind="mergesort")
    hits = int(y[order][:k].sum())
    expected = k * (y.sum() / n)
    return float(hits / expected) if expected > 0 else float("nan")


def max_enrichment_factor(labels, fraction: float = 0.01) -> float:
    """The ceiling EF can reach for this active/decoy ratio and fraction."""
    y = np.asarray(labels).astype(bool)
    n, n_act = y.size, int(y.sum())
    if n == 0 or n_act == 0:
        return float("nan")
    k = max(int(np.ceil(fraction * n)), 1)
    return float(min(k, n_act) / (k * (n_act / n)))


def bedroc(scores, labels, higher_is_better: bool, alpha: float = DEFAULT_BEDROC_ALPHA) -> float:
    """Boltzmann-enhanced discrimination of ROC, in [0, 1].

    Weights early ranks exponentially; ``alpha=20`` concentrates ~80% of the
    weight in the top ~8%. Note that BEDROC's own critics point out it can
    give a misleading sense of early improvement, and that it depends on the
    active fraction -- so report ``alpha`` and the active fraction alongside
    it, as `evaluate` does.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    s, y = _as_ranking(scores, labels, higher_is_better)
    n = s.size
    n_act = int(y.sum())
    ratio = n_act / n

    order = np.argsort(-s, kind="mergesort")
    ranks = np.nonzero(y[order])[0] + 1                    # 1-based ranks of actives
    rie_num = np.sum(np.exp(-alpha * ranks / n))
    denom = (ratio * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1))
    rie = rie_num / denom if denom > 0 else float("nan")

    factor = ratio * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ratio))
    offset = 1.0 / (1.0 - np.exp(alpha * (1 - ratio)))
    return float(rie * factor + offset)


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_ci(metric, scores, labels, higher_is_better: bool,
                 n_boot: int = DEFAULT_BOOTSTRAP, ci_level: float = 0.95,
                 seed: int = RNG_SEED, **metric_kwargs) -> tuple[float, float, float]:
    """``(point, low, high)`` by resampling compounds with replacement.

    Stratified by class, so every resample keeps the active/decoy ratio --
    otherwise the interval mixes sampling noise with a varying active
    fraction, which several of these metrics depend on.

    These intervals are wide. That is the finding, not a defect of the
    implementation: published enrichment differences between methods are
    frequently smaller than the interval around either one.

    The confidence level is ``ci_level``, deliberately not named ``alpha``:
    `bedroc` takes its own ``alpha`` and the two collided through
    ``**metric_kwargs``, silently passing BEDROC's 20.0 in as the CI level.
    """
    s, y = _as_ranking(scores, labels, higher_is_better)
    point = metric(s, y, higher_is_better=True, **metric_kwargs)

    rng = np.random.default_rng(seed)
    act_idx = np.nonzero(y)[0]
    dec_idx = np.nonzero(~y)[0]
    values = []
    for _ in range(n_boot):
        pick = np.concatenate([
            rng.choice(act_idx, act_idx.size, replace=True),
            rng.choice(dec_idx, dec_idx.size, replace=True),
        ])
        try:
            values.append(metric(s[pick], y[pick], higher_is_better=True, **metric_kwargs))
        except ValueError:
            continue
    if not values:
        return float(point), float("nan"), float("nan")
    tail = (1.0 - ci_level) / 2.0
    low, high = np.percentile(values, [100 * tail, 100 * (1 - tail)])
    return float(point), float(low), float(high)


def permutation_p_value(metric, scores, labels, higher_is_better: bool,
                        n_perm: int = 1000, seed: int = RNG_SEED, **metric_kwargs) -> float:
    """One-sided p-value against shuffled labels.

    This is the cheapest honest check available: if shuffling which compounds
    are active does not degrade the metric, the pipeline has not separated
    anything, whatever the point estimate looks like.
    """
    s, y = _as_ranking(scores, labels, higher_is_better)
    observed = metric(s, y, higher_is_better=True, **metric_kwargs)
    rng = np.random.default_rng(seed)
    at_least = 1          # +1 for the observed value, the standard correction
    for _ in range(n_perm):
        shuffled = rng.permutation(y)
        try:
            if metric(s, shuffled, higher_is_better=True, **metric_kwargs) >= observed:
                at_least += 1
        except ValueError:
            continue
    return float(at_least / (n_perm + 1))


# --------------------------------------------------------------------------- #
# Decoys
# --------------------------------------------------------------------------- #
def charge_balance_report(active_props, decoy_props, key: str = "formal_charge") -> dict:
    """Compare a property's distribution between actives and decoys.

    Named for the documented failure it exists to catch: an apparent
    optimisation gain of +1.11 logAUC on DUD-E decoys became **-0.34** once
    the decoys were properly charge-matched. If actives and decoys differ in
    net charge, an enrichment result may be measuring charge.
    """
    a = np.asarray([v for v in active_props if v is not None and np.isfinite(v)], dtype=float)
    d = np.asarray([v for v in decoy_props if v is not None and np.isfinite(v)], dtype=float)
    out = {
        "property": key,
        "n_actives": int(a.size), "n_decoys": int(d.size),
        "active_mean": float(a.mean()) if a.size else float("nan"),
        "decoy_mean": float(d.mean()) if d.size else float("nan"),
    }
    out["difference"] = out["active_mean"] - out["decoy_mean"]
    pooled = np.sqrt((a.var() + d.var()) / 2) if a.size and d.size else float("nan")
    out["standardised_difference"] = (
        float(out["difference"] / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else float("nan")
    )
    # 0.2 is the conventional "small effect" landmark; anything past it means
    # the two sets are distinguishable on this property alone.
    out["balanced"] = bool(abs(out["standardised_difference"]) < 0.2) if np.isfinite(
        out["standardised_difference"]) else False
    return out


def match_decoys(active_props: np.ndarray, pool_props: np.ndarray,
                 n_per_active: int = 50, seed: int = RNG_SEED) -> np.ndarray:
    """Pick property-matched decoys from a pool, nearest first.

    ``active_props`` and ``pool_props`` are 2-D arrays of the same properties
    in the same order (molecular weight, logP, formal charge, HBD, HBA,
    rotatable bonds is the usual set). Each column is standardised so no
    single property dominates the distance, and each pool member is used at
    most once.

    This matches on the properties you give it and nothing else. Matching
    physicochemistry does not make a decoy inactive -- a pool drawn from
    compounds untested against this target may contain real binders, and that
    is a limitation of the construction, not something this function can fix.
    """
    active_props = np.atleast_2d(np.asarray(active_props, dtype=float))
    pool_props = np.atleast_2d(np.asarray(pool_props, dtype=float))
    if active_props.shape[1] != pool_props.shape[1]:
        raise ValueError(
            f"actives have {active_props.shape[1]} properties, pool has {pool_props.shape[1]}"
        )
    if pool_props.shape[0] < active_props.shape[0] * n_per_active:
        logger.warning(
            "  [enrichment] pool has %d candidates for %d requested decoys -- "
            "matching will be looser than asked for.",
            pool_props.shape[0], active_props.shape[0] * n_per_active,
        )

    scale = pool_props.std(axis=0)
    scale[scale == 0] = 1.0
    a_norm = active_props / scale
    p_norm = pool_props / scale

    rng = np.random.default_rng(seed)
    taken: set[int] = set()
    chosen: list[int] = []
    for i in rng.permutation(a_norm.shape[0]):
        distances = np.linalg.norm(p_norm - a_norm[i], axis=1)
        for j in np.argsort(distances, kind="mergesort"):
            if len(chosen) >= (len(taken) + n_per_active):
                break
            if int(j) not in taken:
                taken.add(int(j))
                chosen.append(int(j))
                if len([c for c in chosen]) % n_per_active == 0:
                    break
    return np.asarray(sorted(set(chosen)), dtype=int)


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
def evaluate(scores, labels, higher_is_better: bool,
             ef_fractions=DEFAULT_EF_FRACTIONS,
             bedroc_alpha: float = DEFAULT_BEDROC_ALPHA,
             logauc_lambda: float = DEFAULT_LOGAUC_LAMBDA,
             n_boot: int = DEFAULT_BOOTSTRAP, n_perm: int = 1000,
             seed: int = RNG_SEED) -> dict:
    """Every metric with its interval, plus a permutation p-value.

    No metric is reported as a bare point estimate: each carries a 95%
    bootstrap interval, because the width of that interval is usually the
    thing that decides whether a difference means anything.
    """
    s, y = _as_ranking(scores, labels, higher_is_better)
    n, n_act = s.size, int(y.sum())

    def _ci(metric, **kw):
        point, low, high = bootstrap_ci(metric, s, y, True, n_boot=n_boot, seed=seed, **kw)
        return {"value": point, "ci95_low": low, "ci95_high": high}

    out = {
        "n_compounds": n,
        "n_actives": n_act,
        "n_decoys": n - n_act,
        "active_fraction": n_act / n,
        "higher_is_better": higher_is_better,
        "roc_auc": _ci(roc_auc),
        "log_auc_adjusted": _ci(log_auc, lam=logauc_lambda),
        "bedroc": _ci(bedroc, alpha=bedroc_alpha),
        "parameters": {
            "logauc_lambda": logauc_lambda,
            "bedroc_alpha": bedroc_alpha,
            "n_bootstrap": n_boot,
            "n_permutations": n_perm,
            "seed": seed,
        },
        "enrichment_factor": {},
        "permutation_p_roc_auc": permutation_p_value(
            roc_auc, s, y, True, n_perm=n_perm, seed=seed),
    }
    for fraction in ef_fractions:
        key = f"ef_{fraction:g}".replace("0.", "")
        entry = _ci(enrichment_factor, fraction=fraction)
        entry["max_possible"] = max_enrichment_factor(y, fraction)
        entry["fraction"] = fraction
        out["enrichment_factor"][key] = entry

    out["interpretation_note"] = (
        "Report the intervals, not the point estimates. A difference between two "
        "configurations is only meaningful if their intervals are well separated; "
        "published enrichment differences are frequently smaller than the interval "
        "around either value. Confirm the actives and decoys are property-matched "
        "(see charge_balance_report) before attributing enrichment to binding."
    )
    return out
