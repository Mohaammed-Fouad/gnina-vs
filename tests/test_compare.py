"""
Phase 5 tests: gnina_vs.compare.build_comparison_table -- deltas vs parent
and reference, ligand efficiency (correct sign convention for each score
type), ranking within a drug's group, and safe handling of failed/NaN rows.
"""
import numpy as np
import pandas as pd
import pytest

from gnina_vs import compare


def _results(rows):
    cols = ["compound_id", "ID", "smiles", "status", "pose_quality_flag",
            "vina_affinity", "cnn_score", "cnn_affinity"]
    return pd.DataFrame(rows, columns=cols)


def _prepared(rows):
    cols = ["compound_id", "base_compound_id", "parent_id", "stereo_id",
            "protonation_state", "depth", "phase", "reaction", "is_parent"]
    return pd.DataFrame(rows, columns=cols)


def test_delta_vs_parent_and_ligand_efficiency_sign_conventions():
    results = _results([
        {"compound_id": "drugA", "ID": "drugA", "smiles": "c1ccccc1O", "status": "ok",
         "pose_quality_flag": "ok", "vina_affinity": -6.0, "cnn_score": 0.9, "cnn_affinity": 6.0},
        {"compound_id": "drugA_m1", "ID": "drugA_m1", "smiles": "Oc1ccccc1O", "status": "ok",
         "pose_quality_flag": "ok", "vina_affinity": -8.0, "cnn_score": 0.9, "cnn_affinity": 7.0},
    ])
    prepared = _prepared([
        {"compound_id": "drugA", "base_compound_id": "drugA", "parent_id": "drugA", "stereo_id": 1,
         "protonation_state": "x", "depth": 0, "phase": None, "reaction": None, "is_parent": True},
        {"compound_id": "drugA_m1", "base_compound_id": "drugA_m1", "parent_id": "drugA", "stereo_id": 1,
         "protonation_state": "x", "depth": 1, "phase": "I", "reaction": "hydroxylation", "is_parent": False},
    ])
    df = compare.build_comparison_table(results, prepared, target_id="hERG")

    parent = df[df["compound_id"] == "drugA"].iloc[0]
    metabolite = df[df["compound_id"] == "drugA_m1"].iloc[0]

    # metabolite binds more tightly on both scores
    assert metabolite["delta_vina_affinity_vs_parent"] == pytest.approx(-2.0)  # more negative = better
    assert metabolite["delta_cnn_affinity_vs_parent"] == pytest.approx(1.0)    # higher = better

    # ligand efficiency sign conventions: both positive when binding is favorable
    assert parent["ligand_efficiency_vina"] > 0
    assert parent["ligand_efficiency_cnn"] > 0
    assert metabolite["heavy_atoms"] == 8  # catechol: 6 ring C + 2 O
    assert parent["heavy_atoms"] == 7      # phenol: 6 ring C + 1 O


def test_delta_vs_reference():
    results = _results([
        {"compound_id": "drugA", "ID": "drugA", "smiles": "CCO", "status": "ok",
         "pose_quality_flag": "ok", "vina_affinity": -7.0, "cnn_score": 0.9, "cnn_affinity": 6.5},
    ])
    prepared = _prepared([
        {"compound_id": "drugA", "base_compound_id": "drugA", "parent_id": "drugA", "stereo_id": 1,
         "protonation_state": "x", "depth": 0, "phase": None, "reaction": None, "is_parent": True},
    ])
    df = compare.build_comparison_table(results, prepared, "MOR",
                                        reference_scores={"cnn_affinity": 7.5, "vina_affinity": -9.0})
    row = df.iloc[0]
    assert row["delta_cnn_affinity_vs_reference"] == pytest.approx(6.5 - 7.5)
    assert row["delta_vina_affinity_vs_reference"] == pytest.approx(-7.0 - (-9.0))


def test_no_reference_scores_leaves_deltas_nan_not_zero():
    results = _results([
        {"compound_id": "drugA", "ID": "drugA", "smiles": "CCO", "status": "ok",
         "pose_quality_flag": "ok", "vina_affinity": -7.0, "cnn_score": 0.9, "cnn_affinity": 6.5},
    ])
    prepared = _prepared([
        {"compound_id": "drugA", "base_compound_id": "drugA", "parent_id": "drugA", "stereo_id": 1,
         "protonation_state": "x", "depth": 0, "phase": None, "reaction": None, "is_parent": True},
    ])
    df = compare.build_comparison_table(results, prepared, "MOR", reference_scores=None)
    assert pd.isna(df.iloc[0]["delta_cnn_affinity_vs_reference"])
    assert pd.isna(df.iloc[0]["delta_vina_affinity_vs_reference"])


def test_ranking_within_parent_group_and_rank_change():
    results = _results([
        {"compound_id": "d", "ID": "d", "smiles": "CCO", "status": "ok", "pose_quality_flag": "ok",
         "vina_affinity": -6.0, "cnn_score": 0.9, "cnn_affinity": 6.0},
        {"compound_id": "d_m1", "ID": "d_m1", "smiles": "CCN", "status": "ok", "pose_quality_flag": "ok",
         "vina_affinity": -9.0, "cnn_score": 0.9, "cnn_affinity": 9.0},   # best
        {"compound_id": "d_m2", "ID": "d_m2", "smiles": "CCC", "status": "ok", "pose_quality_flag": "ok",
         "vina_affinity": -5.0, "cnn_score": 0.9, "cnn_affinity": 5.0},   # worst
    ])
    prepared = _prepared([
        {"compound_id": "d", "base_compound_id": "d", "parent_id": "d", "stereo_id": 1,
         "protonation_state": "x", "depth": 0, "phase": None, "reaction": None, "is_parent": True},
        {"compound_id": "d_m1", "base_compound_id": "d_m1", "parent_id": "d", "stereo_id": 1,
         "protonation_state": "x", "depth": 1, "phase": "I", "reaction": "r1", "is_parent": False},
        {"compound_id": "d_m2", "base_compound_id": "d_m2", "parent_id": "d", "stereo_id": 1,
         "protonation_state": "x", "depth": 1, "phase": "I", "reaction": "r2", "is_parent": False},
    ])
    df = compare.build_comparison_table(results, prepared, "hERG")
    by_id = df.set_index("compound_id")
    assert by_id.loc["d_m1", "rank_in_group"] == 1   # highest cnn_affinity -> best rank
    assert by_id.loc["d", "rank_in_group"] == 2
    assert by_id.loc["d_m2", "rank_in_group"] == 3
    assert by_id.loc["d", "rank_vs_parent_change"] == 0
    assert by_id.loc["d_m1", "rank_vs_parent_change"] == 1 - 2  # moved up (better)
    assert by_id.loc["d_m2", "rank_vs_parent_change"] == 3 - 2  # moved down (worse)


def test_failed_docking_row_kept_and_no_division_by_zero():
    results = _results([
        {"compound_id": "d", "ID": "d", "smiles": "CCO", "status": "ok", "pose_quality_flag": "ok",
         "vina_affinity": -6.0, "cnn_score": 0.9, "cnn_affinity": 6.0},
        {"compound_id": "d_m1", "ID": "d_m1", "smiles": None, "status": "timeout",
         "pose_quality_flag": "not_scored", "vina_affinity": None, "cnn_score": None, "cnn_affinity": None},
    ])
    prepared = _prepared([
        {"compound_id": "d", "base_compound_id": "d", "parent_id": "d", "stereo_id": 1,
         "protonation_state": "x", "depth": 0, "phase": None, "reaction": None, "is_parent": True},
        {"compound_id": "d_m1", "base_compound_id": "d_m1", "parent_id": "d", "stereo_id": 1,
         "protonation_state": "x", "depth": 1, "phase": "I", "reaction": "r1", "is_parent": False},
    ])
    df = compare.build_comparison_table(results, prepared, "hERG")
    assert len(df) == 2  # both rows kept
    failed = df[df["compound_id"] == "d_m1"].iloc[0]
    assert failed["status"] == "timeout"
    assert pd.isna(failed["ligand_efficiency_vina"]) and pd.isna(failed["ligand_efficiency_cnn"])
    assert pd.isna(failed["heavy_atoms"])  # no SMILES -> no heavy-atom count, no crash
    assert pd.isna(failed["delta_vina_affinity_vs_parent"]) or np.isnan(failed["delta_vina_affinity_vs_parent"])


def test_missing_parent_row_warns_and_leaves_delta_nan(caplog):
    import logging
    results = _results([
        {"compound_id": "orphan_m1", "ID": "orphan_m1", "smiles": "CCO", "status": "ok",
         "pose_quality_flag": "ok", "vina_affinity": -6.0, "cnn_score": 0.9, "cnn_affinity": 6.0},
    ])
    prepared = _prepared([
        {"compound_id": "orphan_m1", "base_compound_id": "orphan_m1", "parent_id": "orphan",
         "stereo_id": 1, "protonation_state": "x", "depth": 1, "phase": "I", "reaction": "r",
         "is_parent": False},
    ])
    with caplog.at_level(logging.WARNING, logger="gnina_vs.compare"):
        df = compare.build_comparison_table(results, prepared, "hERG")
    assert pd.isna(df.iloc[0]["delta_cnn_affinity_vs_parent"])
    assert "no parent row" in caplog.text


def test_all_comparison_columns_present():
    results = _results([{"compound_id": "d", "ID": "d", "smiles": "CCO", "status": "ok",
                        "pose_quality_flag": "ok", "vina_affinity": -6.0, "cnn_score": 0.9, "cnn_affinity": 6.0}])
    prepared = _prepared([{"compound_id": "d", "base_compound_id": "d", "parent_id": "d", "stereo_id": 1,
                          "protonation_state": "x", "depth": 0, "phase": None, "reaction": None, "is_parent": True}])
    df = compare.build_comparison_table(results, prepared, "hERG")
    assert list(df.columns) == compare.COMPARISON_COLUMNS
