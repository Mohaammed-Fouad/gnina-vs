"""
Phase 2 tests: metabolite-aware ligand preparation -- standardization
(salts/charge/tautomer), stereocentre enumeration (undefined only, capped),
protonation (graceful fallback without Dimorphite-DL), and failure tracking
(embedding/minimization failures are reported, not silently dropped).
"""
import pandas as pd
import pytest
from rdkit import Chem

from gnina_vs import ligand, metabolism


def _df(rows):
    return pd.DataFrame(rows, columns=metabolism.STANDARD_COLUMNS)


# --------------------------------------------------------------------------- #
# standardize_mol
# --------------------------------------------------------------------------- #
def test_standardize_mol_strips_salt_and_logs_note():
    mol = Chem.MolFromSmiles("CC(=O)[O-].[Na+]")
    out, notes = ligand.standardize_mol(mol)
    assert Chem.GetMolFrags(out).__len__() == 1
    assert any("removed fragment" in n for n in notes)


def test_standardize_mol_neutralizes_charge():
    mol = Chem.MolFromSmiles("CC(=O)[O-]")  # acetate anion, no counter-ion
    out, notes = ligand.standardize_mol(mol)
    assert Chem.GetFormalCharge(out) == 0
    assert any("neutraliz" in n for n in notes)


def test_standardize_mol_no_changes_needed():
    mol = Chem.MolFromSmiles("CCO")
    out, notes = ligand.standardize_mol(mol)
    assert Chem.MolToSmiles(out) == Chem.MolToSmiles(mol)
    assert notes == []


# --------------------------------------------------------------------------- #
# enumerate_stereoisomers
# --------------------------------------------------------------------------- #
def test_enumerate_stereoisomers_only_for_undefined():
    defined = Chem.MolFromSmiles("C[C@H](N)C(=O)O")  # L-alanine, fully defined
    assert ligand.enumerate_stereoisomers(defined) == [defined] or \
        len(ligand.enumerate_stereoisomers(defined)) == 1

    undefined = Chem.MolFromSmiles("CC(N)C(=O)O")  # racemic alanine, undefined
    isomers = ligand.enumerate_stereoisomers(undefined)
    assert len(isomers) == 2
    smis = {Chem.MolToSmiles(m) for m in isomers}
    assert len(smis) == 2  # genuinely distinct stereoisomers


def test_enumerate_stereoisomers_respects_cap():
    # A molecule with several undefined stereocentres
    poly_stereo = Chem.MolFromSmiles("CC(O)C(O)C(O)C(O)C=O")  # several undefined centres
    isomers = ligand.enumerate_stereoisomers(poly_stereo, max_isomers=3)
    assert 1 <= len(isomers) <= 3


def test_enumerate_stereoisomers_keeps_defined_stereochemistry():
    # One centre defined, one undefined -- the defined one must never flip
    mol = Chem.MolFromSmiles("C[C@H](N)C(O)C(=O)O")
    isomers = ligand.enumerate_stereoisomers(mol)
    for iso in isomers:
        centres = dict(Chem.FindMolChiralCenters(iso, includeUnassigned=True, useLegacyImplementation=False))
        # the originally-defined centre (index 1, the [C@H]) must stay assigned
        assert centres.get(1) in ("R", "S")


# --------------------------------------------------------------------------- #
# protonate
# --------------------------------------------------------------------------- #
def test_protonate_without_dimorphite_returns_unmodified_and_labels_it():
    mol = Chem.MolFromSmiles("CC(=O)O")
    out, state = ligand.protonate(mol, ph=7.4)
    assert Chem.MolToSmiles(out) == Chem.MolToSmiles(mol)
    assert "unavailable" in state or "Dimorphite" in state


def test_protonate_import_error_is_handled_gracefully(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "dimorphite_dl":
            raise ImportError("no dimorphite_dl in this test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    mol = Chem.MolFromSmiles("CCO")
    out, state = ligand.protonate(mol)
    assert out is not None and "unavailable" in state


# --------------------------------------------------------------------------- #
# prepare_ligands_extended
# --------------------------------------------------------------------------- #
def test_prepare_ligands_extended_basic(tmp_path):
    df = _df([
        {"compound_id": "drugA", "parent_id": "drugA", "smiles": "CCO", "inchikey": "K1",
         "reaction": None, "enzyme": None, "depth": 0, "phase": None, "is_parent": True},
        {"compound_id": "drugA_m1", "parent_id": "drugA", "smiles": "CCN", "inchikey": "K2",
         "reaction": "hydroxylation", "enzyme": "CYP3A4", "depth": 1, "phase": "I", "is_parent": False},
    ])
    sdf = str(tmp_path / "out.sdf")
    prepared_df, failures_df = ligand.prepare_ligands_extended(df, sdf)

    assert failures_df.empty
    assert set(prepared_df["compound_id"]) == {"drugA", "drugA_m1"}
    assert set(prepared_df.columns) >= {"compound_id", "base_compound_id", "parent_id",
                                        "stereo_id", "protonation_state", "smiles",
                                        "depth", "phase", "reaction", "is_parent"}
    row = prepared_df[prepared_df["compound_id"] == "drugA_m1"].iloc[0]
    assert row["parent_id"] == "drugA" and row["reaction"] == "hydroxylation" and row["phase"] == "I"

    mols = [m for m in Chem.SDMolSupplier(sdf, removeHs=False)]
    assert len(mols) == 2
    assert {m.GetProp("CompoundID") for m in mols} == {"drugA", "drugA_m1"}


def test_prepare_ligands_extended_enumerates_stereoisomers_as_separate_entries(tmp_path):
    df = _df([
        {"compound_id": "cmpd1", "parent_id": "cmpd1", "smiles": "CC(N)C(=O)O", "inchikey": "K1",
         "reaction": None, "enzyme": None, "depth": 0, "phase": None, "is_parent": True},
    ])
    sdf = str(tmp_path / "out.sdf")
    prepared_df, failures_df = ligand.prepare_ligands_extended(df, sdf)

    assert failures_df.empty
    assert set(prepared_df["compound_id"]) == {"cmpd1_s1", "cmpd1_s2"}
    assert (prepared_df["base_compound_id"] == "cmpd1").all()
    assert (prepared_df["stereo_id"].isin([1, 2])).all()

    mols = [m for m in Chem.SDMolSupplier(sdf, removeHs=False)]
    assert {m.GetProp("CompoundID") for m in mols} == {"cmpd1_s1", "cmpd1_s2"}


def test_prepare_ligands_extended_invalid_smiles_recorded_not_dropped(tmp_path):
    df = _df([
        {"compound_id": "bad", "parent_id": "bad", "smiles": "not-a-smiles", "inchikey": None,
         "reaction": None, "enzyme": None, "depth": 0, "phase": None, "is_parent": True},
        {"compound_id": "good", "parent_id": "good", "smiles": "CCO", "inchikey": "K1",
         "reaction": None, "enzyme": None, "depth": 0, "phase": None, "is_parent": True},
    ])
    sdf = str(tmp_path / "out.sdf")
    prepared_df, failures_df = ligand.prepare_ligands_extended(df, sdf)

    assert list(prepared_df["compound_id"]) == ["good"]
    assert len(failures_df) == 1
    assert failures_df.iloc[0]["compound_id"] == "bad"
    assert failures_df.iloc[0]["stage"] == "parse"


def test_prepare_ligands_extended_carries_depth_phase_reaction_for_metabolites(tmp_path):
    df = _df([
        {"compound_id": "d", "parent_id": "d", "smiles": "c1ccccc1O", "inchikey": "K1",
         "reaction": None, "enzyme": None, "depth": 0, "phase": None, "is_parent": True},
        {"compound_id": "d_m1", "parent_id": "d", "smiles": "c1ccccc1OS(=O)(=O)O", "inchikey": "K2",
         "reaction": "sulfation", "enzyme": "SULT1A1", "depth": 1, "phase": "II", "is_parent": False},
    ])
    sdf = str(tmp_path / "out.sdf")
    prepared_df, _ = ligand.prepare_ligands_extended(df, sdf)
    m1 = prepared_df[prepared_df["base_compound_id"] == "d_m1"].iloc[0]
    assert m1["depth"] == 1 and m1["phase"] == "II" and m1["reaction"] == "sulfation"
    parent = prepared_df[prepared_df["base_compound_id"] == "d"].iloc[0]
    assert bool(parent["is_parent"]) is True


def test_prepare_ligands_extended_all_fail_raises(tmp_path):
    df = _df([{"compound_id": "bad", "parent_id": "bad", "smiles": "nope", "inchikey": None,
              "reaction": None, "enzyme": None, "depth": 0, "phase": None, "is_parent": True}])
    with pytest.raises(ValueError, match="No ligands"):
        ligand.prepare_ligands_extended(df, str(tmp_path / "out.sdf"))
