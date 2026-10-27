"""
Tests for gnina_vs.tables -- the three-grain output split.

The point of the split is that molecule-level facts stop repeating across
target rows, that "rank" stops being ambiguous, and that a missing value
stays distinguishable from 0 / False / "". Each of those is asserted here.
"""
import json

import numpy as np
import pandas as pd
import pytest

from gnina_vs import tables

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
PARA_GLUC = "CC(=O)Nc1ccc(OC2OC(C(=O)O)C(O)C(O)C2O)cc1"
PARA_SULF = "CC(=O)Nc1ccc(OS(=O)(=O)O)cc1"


def _flat(targets=("T1", "T2")):
    """A flat comparison frame shaped like compare.build_comparison_table's
    output: several targets, a parent plus metabolites, one failed compound."""
    spec = [
        ("p", PARACETAMOL, True, 1.0, np.nan, "ok"),
        ("m1", PARA_GLUC, False, 3.0, -1.2, "ok"),
        ("m2", PARA_SULF, False, 2.0, 0.1, "ok"),
        ("m3", "not-a-smiles", False, np.nan, np.nan, "no_output"),
    ]
    rows = []
    for target in targets:
        for cid, smi, is_parent, rank, dcnn, status in spec:
            # annotate.add_annotations leaves every annotation column None
            # for an unparseable SMILES, so the fixture must too.
            parses = smi != "not-a-smiles"
            rows.append({
                "target_id": target, "compound_id": cid, "base_compound_id": "p",
                "parent_id": "p", "ID": "DRUG1", "smiles": smi, "status": status,
                "pose_quality_flag": "ok" if status == "ok" else "not_scored",
                "cnn_affinity": 6.0, "vina_affinity": -8.3, "cnn_score": 0.75,
                "heavy_atoms": 11, "mol_weight": 151.16,
                "ligand_efficiency_cnn": 0.55, "ligand_efficiency_vina": 0.75,
                "delta_cnn_affinity_vs_parent": dcnn,
                "delta_vina_affinity_vs_parent": dcnn,
                "delta_cnn_affinity_vs_reference": np.nan,
                "delta_vina_affinity_vs_reference": np.nan,
                "rank_in_group": rank, "rank_vs_parent_change": np.nan,
                "is_parent": is_parent, "depth": 0 if is_parent else 1,
                "phase": "II", "reaction": "glucuronidation", "error": "",
                # a few annotation columns, to prove they land in compounds
                "mol_weight_annot": 151.16 if parses else None,
                "clogp": 0.46 if parses else None,
                "tpsa": 49.3 if parses else None,
                "hbd": 2 if parses else None,
                "alert_brenk_n": (1 if is_parent else 0) if parses else None,
                "conjugate_flag": (not is_parent) if parses else None,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Grains
# --------------------------------------------------------------------------- #
def test_each_table_has_its_own_grain():
    t = tables.split_tables(_flat())
    # 4 compounds regardless of how many targets they were docked against
    assert len(t["compounds"]) == 4
    # one row per (target, compound)
    assert len(t["poses"]) == 8
    assert len(t["screen"]) == 8


def test_molecule_facts_stop_repeating_across_targets():
    """The duplication the split exists to remove."""
    flat = _flat()
    t = tables.split_tables(flat)
    # the flat frame carried tpsa once per target row...
    assert len(flat) == 8
    # ...the split carries it once per molecule
    assert len(t["compounds"]) == 4
    assert "mol_tpsa_a2" in t["compounds"].columns
    assert "mol_tpsa_a2" not in t["poses"].columns
    assert "mol_tpsa_a2" not in t["screen"].columns


def test_scores_live_only_in_poses_and_derived_values_only_in_screen():
    t = tables.split_tables(_flat())
    assert "dock_cnn_affinity" in t["poses"].columns
    assert "dock_cnn_affinity" not in t["screen"].columns
    assert "le_cnn_per_heavy_atom" in t["screen"].columns
    assert "le_cnn_per_heavy_atom" not in t["poses"].columns


def test_duplicate_mol_weight_columns_are_coalesced():
    """The flat table carried molecular weight twice (compare + annotate)."""
    t = tables.split_tables(_flat())
    weight_cols = [c for c in t["compounds"].columns if "weight" in c]
    assert weight_cols == ["mol_weight_g_mol"]
    assert t["compounds"]["mol_weight_g_mol"].notna().sum() == 4


# --------------------------------------------------------------------------- #
# The no-drop rule, made architectural
# --------------------------------------------------------------------------- #
def test_a_failed_compound_keeps_its_rows_with_a_reason():
    t = tables.split_tables(_flat())
    bad_poses = t["poses"][t["poses"]["compound_id"] == "m3"]
    assert len(bad_poses) == 2                       # one per target, not dropped
    assert set(bad_poses["qc_status"]) == {"no_pose"}
    assert bad_poses["dock_cnn_affinity"].isna().all()
    # no pose means no rank: a non-null pose_rank would imply a pose exists
    assert bad_poses["pose_rank"].isna().all()
    assert bad_poses["pose_key"].isna().all()


def test_unparseable_smiles_is_one_compound_row_flagged_not_dropped():
    t = tables.split_tables(_flat())
    row = t["compounds"][t["compounds"]["compound_id"] == "m3"]
    assert len(row) == 1
    assert row.iloc[0]["qc_status"] == "smiles_invalid"
    assert pd.isna(row.iloc[0]["inchikey"])


def test_status_is_recorded_per_target_not_globally():
    """A compound can succeed on one target and fail on another."""
    flat = _flat()
    flat.loc[(flat["target_id"] == "T2") & (flat["compound_id"] == "m1"), "status"] = "timeout"
    t = tables.split_tables(flat)
    m1 = t["poses"][t["poses"]["compound_id"] == "m1"].set_index("target_id")
    assert m1.loc["T1", "qc_status"] == "ok"
    assert m1.loc["T2", "qc_status"] == "timeout"
    # and still exactly one compounds row
    assert (t["compounds"]["compound_id"] == "m1").sum() == 1


# --------------------------------------------------------------------------- #
# Bands rather than positions
# --------------------------------------------------------------------------- #
def test_integer_ranks_become_bands():
    t = tables.split_tables(_flat())
    assert "rank_band_in_group" in t["screen"].columns
    assert "rank_in_group" not in t["screen"].columns
    bands = set(t["screen"]["rank_band_in_group"].dropna())
    assert bands.issubset(set(tables.RANK_BANDS))
    parent = t["screen"][t["screen"]["compound_id"] == "p"]
    assert set(parent["rank_band_in_group"]) == {"best"}


def test_a_delta_below_resolution_is_not_ranked():
    """+0.1 pKd is below the configured resolution, so it must collapse into
    one band rather than being reported as an improvement."""
    t = tables.split_tables(_flat())
    screen = t["screen"].set_index(["target_id", "compound_id"])
    assert screen.loc[("T1", "m2"), "dcmp_cnn_band_vs_parent"] == "no_meaningful_change"
    assert screen.loc[("T1", "m1"), "dcmp_cnn_band_vs_parent"] == "worse"


def test_delta_band_respects_the_sign_convention():
    # CNNaffinity is pKd-like: higher is better
    assert tables._delta_band(+2.0, 0.5, higher_is_better=True) == "better"
    assert tables._delta_band(-2.0, 0.5, higher_is_better=True) == "worse"
    # Vina affinity: more negative is better
    assert tables._delta_band(-2.0, 1.0, higher_is_better=False) == "better"
    assert tables._delta_band(+2.0, 1.0, higher_is_better=False) == "worse"
    # below resolution either way
    assert tables._delta_band(0.2, 0.5, higher_is_better=True) == "no_meaningful_change"
    assert tables._delta_band(None, 0.5, higher_is_better=True) is None


def test_resolution_is_configurable_per_call():
    """The resolution decides what counts as a finding, so it must not be
    hard-wired at the call site."""
    flat = _flat(targets=("T1",))
    loose = tables.split_tables(flat, cnn_resolution=5.0)["screen"].set_index("compound_id")
    assert loose.loc["m1", "dcmp_cnn_band_vs_parent"] == "no_meaningful_change"
    tight = tables.split_tables(flat, cnn_resolution=0.05)["screen"].set_index("compound_id")
    assert tight.loc["m1", "dcmp_cnn_band_vs_parent"] == "worse"


def test_rank_change_compares_bands_not_positions():
    t = tables.split_tables(_flat())
    changes = set(t["screen"]["rank_band_change_vs_parent"].dropna())
    assert changes.issubset(set(tables.RANK_CHANGE_BANDS))
    parent = t["screen"][t["screen"]["compound_id"] == "p"]
    assert set(parent["rank_band_change_vs_parent"]) == {"unchanged"}


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
def test_counts_are_nullable_ints_not_floats():
    t = tables.split_tables(_flat())
    assert str(t["compounds"]["alert_brenk_n"].dtype) == "Int64"
    assert str(t["compounds"]["mol_hbd"].dtype) == "Int64"


def test_flags_are_nullable_booleans_not_objects():
    t = tables.split_tables(_flat())
    for col in ("flag_conjugate", "is_parent"):
        assert str(t["compounds"][col].dtype) == "boolean"


def test_missing_never_collapses_into_zero_or_false():
    t = tables.split_tables(_flat())
    bad = t["compounds"][t["compounds"]["compound_id"] == "m3"].iloc[0]
    assert pd.isna(bad["alert_brenk_n"])
    assert pd.isna(bad["flag_conjugate"])


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #
def test_integrity_passes_on_a_well_formed_split():
    tables.assert_integrity(tables.split_tables(_flat()))


def test_integrity_rejects_duplicate_compound_ids():
    t = tables.split_tables(_flat())
    t["compounds"] = pd.concat([t["compounds"], t["compounds"].head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="compound_id is not unique"):
        tables.assert_integrity(t)


def test_integrity_rejects_duplicate_pose_keys():
    t = tables.split_tables(_flat())
    posed = t["poses"][t["poses"]["pose_rank"].notna()].head(1)
    t["poses"] = pd.concat([t["poses"], posed], ignore_index=True)
    with pytest.raises(ValueError, match="poses is not unique"):
        tables.assert_integrity(t)


def test_integrity_rejects_an_orphan_compound_reference():
    t = tables.split_tables(_flat())
    t["compounds"] = t["compounds"][t["compounds"]["compound_id"] != "m2"]
    with pytest.raises(ValueError, match="absent from compounds"):
        tables.assert_integrity(t)


def test_integrity_rejects_a_status_outside_the_vocabulary():
    t = tables.split_tables(_flat())
    t["poses"] = t["poses"].copy()
    t["poses"]["qc_status"] = t["poses"]["qc_status"].astype(object)
    t["poses"].loc[0, "qc_status"] = "whatever"
    with pytest.raises(ValueError, match="closed vocabulary"):
        tables.assert_integrity(t)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def test_write_emits_parquet_csv_and_a_data_dictionary(tmp_path):
    t = tables.split_tables(_flat())
    written = tables.write_tables(t, tmp_path, manifest={"run_id": "abc123", "seed": 42})
    for name in ("compounds", "poses", "screen"):
        assert (tmp_path / f"{name}.parquet").exists()
        assert (tmp_path / f"{name}.csv").exists()
    assert "data_dictionary" in written
    doc = json.loads((tmp_path / "data_dictionary.json").read_text())
    assert doc["schema_version"] == tables.SCHEMA_VERSION
    assert doc["grains"]["screen"] == ["target_id", "compound_id"]
    # the convention must announce itself as local, not standard
    assert "no community schema" in doc["note"].lower()
    # and the unvalidated resolution must carry its warning
    assert "not validated" in doc["score_resolution"]["warning"].lower()


def test_manifest_round_trips_through_parquet_metadata(tmp_path):
    t = tables.split_tables(_flat())
    tables.write_tables(t, tmp_path, manifest={"run_id": "abc123", "seed": 42, "gnina": "1.3.3"})
    back = tables.read_manifest(tmp_path / "screen.parquet")
    assert back["run_id"] == "abc123"
    assert back["seed"] == 42
    assert back["schema_version"] == tables.SCHEMA_VERSION


def test_parquet_preserves_types_that_csv_destroys(tmp_path):
    """The reason Parquet is authoritative, asserted rather than claimed."""
    t = tables.split_tables(_flat())
    tables.write_tables(t, tmp_path, manifest={})

    parquet = pd.read_parquet(tmp_path / "compounds.parquet")
    assert str(parquet["flag_conjugate"].dtype) == "boolean"
    assert str(parquet["alert_brenk_n"].dtype) == "Int64"
    assert pd.isna(parquet.loc[parquet["compound_id"] == "m3", "flag_conjugate"].iloc[0])

    # The CSV cannot carry that, which is why it is generated and not the source.
    csv = pd.read_csv(tmp_path / "compounds.csv")
    assert str(csv["flag_conjugate"].dtype) != "boolean"


def test_write_refuses_a_broken_split(tmp_path):
    """Integrity is checked at write time, so a bad schema cannot reach disk."""
    t = tables.split_tables(_flat())
    t["compounds"] = pd.concat([t["compounds"], t["compounds"].head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        tables.write_tables(t, tmp_path, manifest={})
    assert not (tmp_path / "compounds.parquet").exists()


# --------------------------------------------------------------------------- #
# Keys and compatibility
# --------------------------------------------------------------------------- #
def test_pose_key_is_stable_and_distinct():
    a = tables.pose_key("T1", "m1", 1)
    assert a == tables.pose_key("T1", "m1", 1)
    assert a != tables.pose_key("T1", "m1", 2)
    assert a != tables.pose_key("T2", "m1", 1)


def test_legacy_view_rebuilds_a_flat_frame():
    t = tables.split_tables(_flat())
    legacy = tables.legacy_comparison_view(t)
    assert len(legacy) == len(t["screen"])
    # it carries both target-level and molecule-level columns again
    assert "rank_band_in_group" in legacy.columns
    assert "mol_tpsa_a2" in legacy.columns


def test_poses_grain_is_currently_degenerate():
    """Documents a real limitation rather than implying multi-pose support:
    reporter._parse_gnina_log stops at mode 1, so every pose_rank is 1."""
    t = tables.split_tables(_flat())
    ranks = set(t["poses"]["pose_rank"].dropna().astype(int))
    assert ranks == {1}
