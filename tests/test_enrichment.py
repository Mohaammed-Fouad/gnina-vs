"""
Tests for gnina_vs.enrichment -- retrospective validation metrics.

Everything here runs against synthetic rankings with known answers
(perfect, good, random, inverted), because that is the only way to check a
metric implementation without real docking output. The module has NOT been
run against real Gnina scores; see its docstring.
"""
import numpy as np
import pytest

from gnina_vs import enrichment as E

N_ACT, N_DEC = 40, 760


def _labels():
    return np.array([True] * N_ACT + [False] * N_DEC)


def _scores(kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "perfect":
        return np.concatenate([np.linspace(10, 9, N_ACT), np.linspace(5, 0, N_DEC)])
    if kind == "inverted":
        return np.concatenate([np.linspace(0, 1, N_ACT), np.linspace(5, 10, N_DEC)])
    if kind == "good":
        return np.concatenate([rng.normal(1.5, 1, N_ACT), rng.normal(0, 1, N_DEC)])
    return rng.normal(size=N_ACT + N_DEC)


# --------------------------------------------------------------------------- #
# Metrics against known answers
# --------------------------------------------------------------------------- #
def test_roc_auc_spans_perfect_random_inverted():
    y = _labels()
    assert E.roc_auc(_scores("perfect"), y, True) == pytest.approx(1.0)
    assert E.roc_auc(_scores("inverted"), y, True) == pytest.approx(0.0)
    assert E.roc_auc(_scores("random"), y, True) == pytest.approx(0.5, abs=0.08)


def test_log_auc_is_zero_for_random_and_positive_for_good():
    y = _labels()
    assert E.log_auc(_scores("random"), y, True) == pytest.approx(0.0, abs=0.06)
    assert E.log_auc(_scores("good"), y, True) > 0.05
    assert E.log_auc(_scores("perfect"), y, True) > E.log_auc(_scores("good"), y, True)


def test_log_auc_adjustment_shifts_the_baseline_to_zero():
    y = _labels()
    raw = E.log_auc(_scores("random"), y, True, adjusted=False)
    adj = E.log_auc(_scores("random"), y, True, adjusted=True)
    assert raw > adj                      # the random baseline was subtracted
    assert abs(adj) < abs(raw)


def test_log_auc_rejects_a_lambda_outside_the_unit_interval():
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="lam"):
            E.log_auc(_scores("good"), _labels(), True, lam=bad)


def test_enrichment_factor_hits_its_ceiling_on_a_perfect_ranking():
    y = _labels()
    ef = E.enrichment_factor(_scores("perfect"), y, True, fraction=0.01)
    assert ef == pytest.approx(E.max_enrichment_factor(y, 0.01))


def test_enrichment_factor_is_about_one_for_random():
    vals = [E.enrichment_factor(_scores("random", s), _labels(), True, 0.1) for s in range(12)]
    assert np.mean(vals) == pytest.approx(1.0, abs=0.4)


def test_max_enrichment_factor_accounts_for_the_active_fraction():
    # 40 actives in 800 at the top 1% (8 slots): ceiling is 8/(8*0.05) = 20
    assert E.max_enrichment_factor(_labels(), 0.01) == pytest.approx(20.0)
    # at 50% the top half can hold every active, so the ceiling is just 2
    assert E.max_enrichment_factor(_labels(), 0.5) == pytest.approx(2.0)


def test_enrichment_factor_rejects_a_bad_fraction():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            E.enrichment_factor(_scores("good"), _labels(), True, fraction=bad)


def test_bedroc_spans_zero_to_one():
    y = _labels()
    assert E.bedroc(_scores("perfect"), y, True) == pytest.approx(1.0, abs=1e-6)
    assert E.bedroc(_scores("inverted"), y, True) == pytest.approx(0.0, abs=1e-3)
    assert 0.0 < E.bedroc(_scores("good"), y, True) < 1.0


def test_bedroc_rejects_a_non_positive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        E.bedroc(_scores("good"), _labels(), True, alpha=0)


# --------------------------------------------------------------------------- #
# The sign-convention trap
# --------------------------------------------------------------------------- #
def test_higher_is_better_flips_the_result():
    """CNNaffinity is pKd-like (higher better); Vina affinity is kcal/mol
    (more negative better). Getting this backwards mirrors the answer."""
    y = _labels()
    s = _scores("perfect")
    assert E.roc_auc(s, y, higher_is_better=True) == pytest.approx(1.0)
    assert E.roc_auc(s, y, higher_is_better=False) == pytest.approx(0.0)


def test_vina_style_scores_evaluate_correctly_when_declared():
    """A Vina-like set where actives are more negative."""
    y = _labels()
    vina = np.concatenate([np.linspace(-11, -10, N_ACT), np.linspace(-6, -2, N_DEC)])
    assert E.roc_auc(vina, y, higher_is_better=False) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def test_unscored_compounds_are_dropped_not_ranked_last():
    """A failed docking is not a weak binder -- treating it as 'worst' would
    invent a ranking for a compound that was never scored."""
    y = np.array([True, True, False, False, False])
    s = np.array([5.0, np.nan, 1.0, 0.5, np.inf])
    ranked, labels = E._as_ranking(s, y, True)
    assert ranked.size == 3
    assert labels.tolist() == [True, False, False]


def test_all_nan_scores_raise():
    with pytest.raises(ValueError, match="NaN or infinite"):
        E.roc_auc(np.array([np.nan, np.nan]), np.array([True, False]), True)


def test_one_class_only_raises():
    with pytest.raises(ValueError, match="both actives and decoys"):
        E.roc_auc(np.array([1.0, 2.0]), np.array([True, True]), True)
    with pytest.raises(ValueError, match="both actives and decoys"):
        E.roc_auc(np.array([1.0, 2.0]), np.array([False, False]), True)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="differ in length"):
        E.roc_auc(np.array([1.0, 2.0]), np.array([True]), True)


def test_tied_scores_do_not_depend_on_input_order():
    """All-tied scores must give 0.5, not an artefact of row order."""
    y = np.array([True] * 10 + [False] * 10)
    tied = np.ones(20)
    assert E.roc_auc(tied, y, True) == pytest.approx(0.5)
    assert E.roc_auc(tied, y[::-1], True) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def test_bootstrap_interval_brackets_the_point_estimate():
    point, low, high = E.bootstrap_ci(E.roc_auc, _scores("good"), _labels(), True, n_boot=300)
    assert low <= point <= high
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_ci_level_is_not_confused_with_bedroc_alpha():
    """Regression: the CI level used to be called `alpha`, so BEDROC's own
    alpha=20.0 flowed in through **metric_kwargs and was read as the
    confidence level, which raised deep inside numpy."""
    point, low, high = E.bootstrap_ci(
        E.bedroc, _scores("good"), _labels(), True, n_boot=200, alpha=20.0,
    )
    assert low <= point <= high


def test_a_wider_confidence_level_gives_a_wider_interval():
    kw = dict(scores=_scores("good"), labels=_labels(), higher_is_better=True, n_boot=400)
    _, lo95, hi95 = E.bootstrap_ci(E.roc_auc, **kw, ci_level=0.95)
    _, lo50, hi50 = E.bootstrap_ci(E.roc_auc, **kw, ci_level=0.50)
    assert (hi95 - lo95) > (hi50 - lo50)


def test_permutation_p_separates_signal_from_noise():
    y = _labels()
    assert E.permutation_p_value(E.roc_auc, _scores("good"), y, True, n_perm=200) < 0.05
    assert E.permutation_p_value(E.roc_auc, _scores("random"), y, True, n_perm=200) > 0.05


def test_results_are_reproducible_for_a_fixed_seed():
    kw = dict(scores=_scores("good"), labels=_labels(), higher_is_better=True, n_boot=200)
    a = E.bootstrap_ci(E.roc_auc, **kw, seed=7)
    b = E.bootstrap_ci(E.roc_auc, **kw, seed=7)
    assert a == b


# --------------------------------------------------------------------------- #
# Decoys
# --------------------------------------------------------------------------- #
def test_charge_balance_flags_an_unbalanced_decoy_set():
    """The documented failure: an apparent +1.11 logAUC gain on DUD-E decoys
    became -0.34 once the decoys were charge-matched."""
    unbalanced = E.charge_balance_report([-1] * 50, [0] * 50)
    assert unbalanced["balanced"] is False
    assert unbalanced["difference"] == pytest.approx(-1.0)

    rng = np.random.default_rng(0)
    balanced = E.charge_balance_report(rng.normal(0, 1, 200), rng.normal(0, 1, 200))
    assert balanced["balanced"] is True


def test_charge_balance_ignores_missing_values():
    out = E.charge_balance_report([0, 1, None, np.nan], [0, 1])
    assert out["n_actives"] == 2


def test_match_decoys_picks_property_neighbours():
    rng = np.random.default_rng(0)
    actives = np.array([[300.0, 2.0, 0.0], [500.0, 4.0, -1.0]])
    # a pool split into near-matches and far-away compounds
    near = np.concatenate([
        actives[0] + rng.normal(0, 1, (40, 3)),
        actives[1] + rng.normal(0, 1, (40, 3)),
    ])
    far = np.array([[1200.0, 12.0, 5.0]] * 40)
    pool = np.vstack([near, far])
    chosen = E.match_decoys(actives, pool, n_per_active=10)
    assert len(chosen) > 0
    # nothing from the far block should be picked while near ones remain
    assert all(i < len(near) for i in chosen)


def test_match_decoys_never_reuses_a_pool_member():
    rng = np.random.default_rng(1)
    actives = rng.normal(0, 1, (5, 3))
    pool = rng.normal(0, 1, (200, 3))
    chosen = E.match_decoys(actives, pool, n_per_active=10)
    assert len(chosen) == len(set(chosen.tolist()))


def test_match_decoys_rejects_mismatched_property_counts():
    with pytest.raises(ValueError, match="properties"):
        E.match_decoys(np.zeros((2, 3)), np.zeros((10, 4)))


# --------------------------------------------------------------------------- #
# evaluate()
# --------------------------------------------------------------------------- #
def test_evaluate_never_reports_a_point_without_an_interval():
    r = E.evaluate(_scores("good"), _labels(), True, n_boot=200, n_perm=100)
    for key in ("roc_auc", "log_auc_adjusted", "bedroc"):
        assert set(r[key]) >= {"value", "ci95_low", "ci95_high"}
    for entry in r["enrichment_factor"].values():
        assert set(entry) >= {"value", "ci95_low", "ci95_high", "max_possible", "fraction"}


def test_evaluate_records_the_parameters_that_change_the_answer():
    r = E.evaluate(_scores("good"), _labels(), True, n_boot=100, n_perm=50, bedroc_alpha=10.0)
    p = r["parameters"]
    assert p["bedroc_alpha"] == 10.0
    assert p["n_bootstrap"] == 100 and p["seed"] == E.RNG_SEED
    # BEDROC depends on the active fraction, so it has to travel with it
    assert r["active_fraction"] == pytest.approx(N_ACT / (N_ACT + N_DEC))
    assert r["higher_is_better"] is True


def test_evaluate_carries_the_interpretation_warning():
    r = E.evaluate(_scores("good"), _labels(), True, n_boot=100, n_perm=50)
    assert "intervals" in r["interpretation_note"]
    assert "property-matched" in r["interpretation_note"]
