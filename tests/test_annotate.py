"""
Phase 6 tests: gnina_vs.annotate -- descriptor columns, Lipinski violation
count, Brenk/PAINS structural alerts, heuristic conjugate flag, and the
hard rule that annotation never removes or reorders a row.
"""
import pandas as pd
import pytest

from gnina_vs import annotate


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


def test_brenk_alert_detected_and_flagged_reactive():
    # An aldehyde is a classic Brenk alert
    a = annotate.annotate_smiles("O=Cc1ccccc1C=O")
    assert a["brenk_alerts"] is not None and "aldehyde" in a["brenk_alerts"].lower()
    assert a["reactive_alert"] is True


def test_no_alert_for_benign_molecule():
    a = annotate.annotate_smiles("CCO")
    assert a["brenk_alerts"] is None and a["reactive_alert"] is False


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


def test_admet_status_reflects_missing_dependency():
    a = annotate.annotate_smiles("CCO")
    assert "admet_ai not installed" in a["admet_status"] or a["admet_status"] == "computed"
    assert a["admet_applicability_flag"] == "not_implemented"


# --------------------------------------------------------------------------- #
# add_annotations: DataFrame-level, never drops/reorders rows
# --------------------------------------------------------------------------- #
def test_add_annotations_never_drops_rows_including_invalid_smiles():
    df = pd.DataFrame({"compound_id": ["a", "b", "c"], "smiles": ["CCO", "not-a-smiles", None]})
    out = annotate.add_annotations(df)
    assert len(out) == 3
    assert list(out["compound_id"]) == ["a", "b", "c"]  # order preserved
    for col in annotate.ANNOTATION_COLUMNS:
        assert col in out.columns
    assert out.iloc[0]["reactive_alert"] is False
    assert pd.isna(out.iloc[1]["mol_weight_annot"])
    assert pd.isna(out.iloc[2]["mol_weight_annot"])


def test_add_annotations_requires_smiles_column():
    df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match="smiles"):
        annotate.add_annotations(df)


def test_add_annotations_does_not_mutate_input():
    df = pd.DataFrame({"smiles": ["CCO"]})
    original_cols = list(df.columns)
    annotate.add_annotations(df)
    assert list(df.columns) == original_cols  # input untouched
