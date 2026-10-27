"""
Phase 6 tests: gnina_vs.annotate -- descriptor columns, Lipinski violation
count, Brenk/PAINS structural alerts, heuristic conjugate flag, and the
hard rule that annotation never removes or reorders a row.
"""
import pandas as pd
import pytest

from gnina_vs import annotate, applicability


def test_annotate_smiles_basic_descriptors():
    a = annotate.annotate_smiles("CCO")  # ethanol
    assert a["mol_weight_annot"] == pytest.approx(46.07, abs=0.05)
    assert a["hbd"] == 1 and a["hba"] == 1
    assert a["formal_charge"] == 0
    assert a["lipinski_violations"] == 0


def test_annotate_smiles_invalid_returns_all_none():
    a = annotate.annotate_smiles("not-a-smiles")
    assert all(v is None for v in a.values())
    assert set(a.keys()) == set(annotate.ANNOTATION_COLUMNS)


def test_annotate_smiles_none_input_returns_all_none():
    a = annotate.annotate_smiles(None)
    assert all(v is None for v in a.values())


def test_lipinski_violations_counts_all_four():
    # A large, greasy, H-bond-heavy molecule that violates all four Ro5 criteria
    huge = "C" * 40 + "O" * 12  # crude but MW>500, many HBA, still parses
    a = annotate.annotate_smiles(huge)
    assert a["lipinski_violations"] >= 1  # at least MW violated; exact count depends on RDKit valence rules


def test_brenk_alert_detected_and_counted_by_family():
    # An aldehyde is a classic Brenk alert
    a = annotate.annotate_smiles("O=Cc1ccccc1C=O")
    assert a["brenk_alerts"] is not None and "aldehyde" in a["brenk_alerts"].lower()
    assert a["alert_brenk_n"] >= 1
    assert a["alert_total_n"] >= a["alert_brenk_n"]


def test_no_reactive_alert_boolean_is_reported():
    """The boolean was a category error on the filters' own terms; counts
    per family replaced it. Guard against it coming back."""
    assert "reactive_alert" not in annotate.ANNOTATION_COLUMNS
    assert "reactive_alert" not in annotate.annotate_smiles("O=Cc1ccccc1C=O")


def test_pains_families_are_counted_separately():
    """The three PAINS families differ by more than an order of magnitude in
    how much recognised-compound mass they carry, so they must not be merged."""
    for fam in ("alert_pains_a_n", "alert_pains_b_n", "alert_pains_c_n"):
        assert fam in annotate.ANNOTATION_COLUMNS
    a = annotate.annotate_smiles("CCO")
    assert a["alert_pains_a_n"] == 0


def test_no_alert_for_benign_molecule():
    a = annotate.annotate_smiles("CCO")
    assert a["brenk_alerts"] is None and a["alert_brenk_n"] == 0


def test_glucuronide_flag():
    # phenol glucuronide: pyranose ring bearing a carboxylic acid, O-linked to phenol
    glucuronide = "OC1OC(C(=O)O)C(O)C(O)C1Oc1ccccc1"
    a = annotate.annotate_smiles(glucuronide)
    assert a["glucuronide_flag"] is True
    assert a["conjugate_flag"] is True


def test_sulfate_flag():
    sulfate_ester = "c1ccccc1OS(=O)(=O)O"
    a = annotate.annotate_smiles(sulfate_ester)
    assert a["sulfate_flag"] is True
    assert a["conjugate_flag"] is True


def test_plain_compound_not_flagged_as_conjugate():
    a = annotate.annotate_smiles("c1ccccc1O")  # phenol, not a conjugate
    assert a["glucuronide_flag"] is False and a["sulfate_flag"] is False
    assert a["conjugate_flag"] is False


def test_annotate_smiles_leaves_row_context_columns_empty():
    a = annotate.annotate_smiles("CCO")
    assert a["parent_tanimoto"] is None
    assert a["admet_status"] is None
    for col in applicability.AD_COLUMNS:
        assert a[col] is None


def test_physchem_out_of_range_flags_small_molecule():
    assert "MW=" in annotate.annotate_smiles("CCO")["physchem_out_of_range"]  # MW 46 < 100
    assert annotate.annotate_smiles("CC(=O)Nc1ccc(O)cc1")["physchem_out_of_range"] is None


def test_admet_status_without_dependency(monkeypatch):
    monkeypatch.setattr(annotate, "_admet_ai_available", lambda: False)
    out = annotate.add_annotations(pd.DataFrame({"smiles": ["CC(=O)Nc1ccc(O)cc1", "bad"]}), run_ad=False)
    assert out.iloc[0]["admet_status"] == "skipped: admet_ai not installed"
    assert out.iloc[1]["admet_status"] == "skipped: invalid_smiles"
    assert out.iloc[0]["ad_status"] == "skipped: disabled"


def test_admet_disabled():
    out = annotate.add_annotations(pd.DataFrame({"smiles": ["CCO"]}), run_admet=False, run_ad=False)
    assert out.iloc[0]["admet_status"] == "skipped: disabled"


# --------------------------------------------------------------------------- #
# add_annotations: DataFrame-level, never drops/reorders rows
# --------------------------------------------------------------------------- #
def test_add_annotations_never_drops_rows_including_invalid_smiles():
    df = pd.DataFrame({"compound_id": ["a", "b", "c"], "smiles": ["CCO", "not-a-smiles", None]})
    out = annotate.add_annotations(df, run_ad=False)
    assert len(out) == 3
    assert list(out["compound_id"]) == ["a", "b", "c"]  # order preserved
    for col in annotate.ANNOTATION_COLUMNS:
        assert col in out.columns
    assert out.iloc[0]["alert_brenk_n"] == 0
    assert pd.isna(out.iloc[1]["mol_weight_annot"])
    assert pd.isna(out.iloc[2]["mol_weight_annot"])


def test_add_annotations_requires_smiles_column():
    df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match="smiles"):
        annotate.add_annotations(df)


def test_add_annotations_does_not_mutate_input():
    df = pd.DataFrame({"smiles": ["CCO"]})
    original_cols = list(df.columns)
    annotate.add_annotations(df, run_ad=False)
    assert list(df.columns) == original_cols  # input untouched


# --------------------------------------------------------------------------- #
# Parent similarity & applicability-domain proxy
# --------------------------------------------------------------------------- #
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
PARA_GLUC = "CC(=O)Nc1ccc(OC2OC(C(=O)O)C(O)C(O)C2O)cc1"
PARA_SULF = "CC(=O)Nc1ccc(OS(=O)(=O)O)cc1"


def _comparison_like_df():
    # Mirrors compare.build_comparison_table: parent_id + is_parent, no parent_smiles.
    return pd.DataFrame({
        "compound_id": ["p", "m1", "m2", "m3"],
        "parent_id": ["p", "p", "p", "p"],
        "is_parent": [True, False, False, False],
        "smiles": [PARACETAMOL, PARA_GLUC, PARA_SULF, "not-a-smiles"],
    })


def test_parent_tanimoto_derived_from_parent_id():
    out = annotate.add_annotations(_comparison_like_df(), run_admet=False, run_ad=False)
    assert pd.isna(out.iloc[0]["parent_tanimoto"])          # parent not compared with itself
    assert 0 < out.iloc[1]["parent_tanimoto"] < 1
    assert 0 < out.iloc[2]["parent_tanimoto"] < 1
    assert pd.isna(out.iloc[3]["parent_tanimoto"])          # invalid SMILES


def test_parent_tanimoto_from_explicit_column_with_missing_parent():
    # NaN parent_smiles used to crash the AD check (bool(nan) is True).
    df = pd.DataFrame({"smiles": ["CCO", PARA_GLUC], "parent_smiles": [float("nan"), PARACETAMOL]})
    out = annotate.add_annotations(df, run_admet=False, run_ad=False)
    assert pd.isna(out.iloc[0]["parent_tanimoto"])
    assert out.iloc[1]["parent_tanimoto"] > 0


def test_parent_tanimoto_absent_without_parent_info():
    out = annotate.add_annotations(pd.DataFrame({"smiles": [PARACETAMOL]}), run_admet=False, run_ad=False)
    assert pd.isna(out.iloc[0]["parent_tanimoto"])


def test_admet_predictions_merge_onto_valid_rows_only(monkeypatch):
    monkeypatch.setattr(annotate, "_admet_ai_available", lambda: True)
    monkeypatch.setattr(
        annotate, "try_admet_ai",
        lambda smi: pd.DataFrame({"hERG": [0.1 * (i + 1) for i in range(len(smi))]}, index=smi),
    )
    out = annotate.add_annotations(_comparison_like_df(), run_ad=False)
    assert list(out["admet_status"][:3]) == ["computed"] * 3
    assert out.iloc[3]["admet_status"] == "skipped: invalid_smiles"
    # predictions merged with prefix, one value per valid row, NaN for the invalid row
    assert "admet_hERG" in out.columns
    assert out["admet_hERG"].iloc[:3].notna().all() and pd.isna(out.iloc[3]["admet_hERG"])
    assert len(out) == 4 and list(out["compound_id"]) == ["p", "m1", "m2", "m3"]


def test_admet_failure_is_reported(monkeypatch):
    monkeypatch.setattr(annotate, "_admet_ai_available", lambda: True)
    monkeypatch.setattr(annotate, "try_admet_ai", lambda smi: None)
    out = annotate.add_annotations(pd.DataFrame({"smiles": [PARACETAMOL]}), run_ad=False)
    assert out.iloc[0]["admet_status"] == "failed: see log"


def test_no_deprecation_warnings(recwarn):
    annotate.add_annotations(_comparison_like_df(), run_admet=False, run_ad=False)
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


# --------------------------------------------------------------------------- #
# Exposure plausibility (a heuristic, and labelled as one)
# --------------------------------------------------------------------------- #
def test_exposure_tier_counts_liabilities():
    # no liabilities -> likely
    tier, basis = annotate.exposure_plausibility(mw=200, logp=2.0, tpsa=50, charge=0, conjugate=False)
    assert tier == "likely" and basis is None
    # one liability -> reduced
    tier, basis = annotate.exposure_plausibility(mw=200, logp=2.0, tpsa=50, charge=-1, conjugate=False)
    assert tier == "reduced" and "formal_charge=-1" in basis
    # two or more -> unlikely, and the basis names each one
    tier, basis = annotate.exposure_plausibility(mw=600, logp=-1.0, tpsa=150, charge=-1, conjugate=True)
    assert tier == "unlikely"
    for expected in ("phase_II_conjugate", "formal_charge=-1", "TPSA=150", "MW=600", "logP=-1.0"):
        assert expected in basis


def test_conjugates_are_less_plausible_than_their_parent():
    parent = annotate.annotate_smiles(PARACETAMOL)
    gluc = annotate.annotate_smiles(PARA_GLUC)
    assert parent["exposure_plausibility"] == "likely"
    assert gluc["exposure_plausibility"] == "unlikely"
    assert "phase_II_conjugate" in gluc["exposure_basis"]


def test_exposure_tier_is_none_for_invalid_smiles():
    a = annotate.annotate_smiles("not-a-smiles")
    assert a["exposure_plausibility"] is None and a["exposure_basis"] is None


# --------------------------------------------------------------------------- #
# Applicability domain is wired in, and is not parent similarity
# --------------------------------------------------------------------------- #
def test_parent_similarity_is_no_longer_an_applicability_flag():
    """parent_tanimoto is still reported, but it must not be the AD answer."""
    assert "admet_applicability_flag" not in annotate.ANNOTATION_COLUMNS
    assert "parent_tanimoto" in annotate.ANNOTATION_COLUMNS
    assert set(applicability.AD_COLUMNS).issubset(annotate.ANNOTATION_COLUMNS)


def test_ad_disabled_records_the_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("GNINA_VS_CACHE", str(tmp_path))
    out = annotate.add_annotations(_comparison_like_df(), run_admet=False, run_ad=False)
    assert out["ad_status"].unique().tolist() == ["skipped: disabled"]
    assert out["ad_distance"].isna().all()


def test_ad_runs_independently_of_admet_ai(monkeypatch):
    """Knowing a compound sits outside the training chemistry is useful even
    when ADMET-AI itself is not installed, so the two must not be coupled."""
    monkeypatch.setattr(annotate, "_admet_ai_available", lambda: False)
    monkeypatch.setattr(
        applicability, "evaluate",
        lambda smiles, **kw: {
            "ad_reference": ["stub"] * len(smiles),
            "ad_distance": [0.2] * len(smiles),
            "ad_threshold": [0.8] * len(smiles),
            "ad_in_domain": [True] * len(smiles),
            "ad_status": ["computed"] * len(smiles),
        },
    )
    out = annotate.add_annotations(_comparison_like_df(), run_admet=True, run_ad=True)
    assert out["admet_status"].iloc[0] == "skipped: admet_ai not installed"
    assert out["ad_status"].iloc[0] == "computed"
    assert bool(out["ad_in_domain"].iloc[0]) is True
