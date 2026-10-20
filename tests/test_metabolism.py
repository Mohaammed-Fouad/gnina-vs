"""
Phase 1 tests: BioTransformer wrapper, fixture-based parsing (no real jar
required -- see gnina_vs.metabolism module docstring for what's unverified),
--metabolites-csv bypass, caching, and dedup.
"""
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from gnina_vs import metabolism

DATA = Path(__file__).parent / "data"
TERFENADINE = "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1"


# --------------------------------------------------------------------------- #
# Fixture parsing (the "saved BioTransformer output" the prompt asks for)
# --------------------------------------------------------------------------- #
def test_parse_biotransformer_csv_fixture():
    df = metabolism.parse_biotransformer_csv(str(DATA / "biotransformer_fixture.csv"), "terf", TERFENADINE)
    assert set(df.columns) == set(metabolism.STANDARD_COLUMNS)
    assert len(df) == 3  # 3 valid metabolite rows parse; BTM0004 (empty SMILES) is skipped
    assert (df["parent_id"] == "terf").all()
    assert (df["is_parent"] == False).all()  # noqa: E712
    phases = dict(zip(df["compound_id"], df["phase"]))
    reactions = dict(zip(df["compound_id"], df["reaction"]))
    assert "I" in phases.values() and "II" in phases.values()
    assert any("Glucuronidation" in str(r) for r in reactions.values())


def test_parse_biotransformer_csv_empty(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("Precursor SMILES,Metabolite SMILES,Metabolite ID,Reaction,Enzyme(s),Biosystem\n")
    df = metabolism.parse_biotransformer_csv(str(empty), "x", "CCO")
    assert df.empty and list(df.columns) == metabolism.STANDARD_COLUMNS


def test_parse_biotransformer_csv_missing_smiles_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(metabolism.BioTransformerError, match="SMILES column"):
        metabolism.parse_biotransformer_csv(str(bad), "x", "CCO")


# --------------------------------------------------------------------------- #
# Java/jar checks
# --------------------------------------------------------------------------- #
def test_check_java_and_jar_missing_java():
    with pytest.raises(metabolism.BioTransformerError, match="Java"):
        metabolism.check_java_and_jar("definitely-not-a-real-java-binary", "whatever.jar")


def test_check_java_and_jar_missing_jar(tmp_path):
    with pytest.raises(metabolism.BioTransformerError, match="jar not found"):
        metabolism.check_java_and_jar("java", str(tmp_path / "nope.jar"))


# --------------------------------------------------------------------------- #
# run_biotransformer with a mocked "java" (no real BioTransformer needed)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_java(tmp_path, monkeypatch):
    jar = tmp_path / "biotransformer.jar"
    jar.write_text("not a real jar")
    calls = []

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kw):
        calls.append(cmd)
        if "--help" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "usage: biotransformer -k pred -b <type> ...", "")
        out_csv = cmd[cmd.index("-ocsv") + 1]
        import shutil as _sh
        _sh.copy(DATA / "biotransformer_fixture.csv", out_csv)
        return subprocess.CompletedProcess(cmd, 0, "done", "")

    monkeypatch.setattr(metabolism.shutil, "which", lambda n, *a, **k: "/usr/bin/" + str(n))
    monkeypatch.setattr(metabolism.subprocess, "run", fake_run)
    return str(jar), calls


def test_run_biotransformer_writes_and_caches(tmp_path, fake_java):
    jar, calls = fake_java
    cache_dir = tmp_path / "cache"
    p1 = metabolism.run_biotransformer(TERFENADINE, jar, cache_dir=cache_dir)
    assert Path(p1).exists()
    n_calls_after_first = len(calls)

    p2 = metabolism.run_biotransformer(TERFENADINE, jar, cache_dir=cache_dir)
    assert p1 == p2
    assert len(calls) == n_calls_after_first, "second call should hit the cache, not re-run java"


def test_run_biotransformer_rejects_bad_steps(fake_java):
    jar, _ = fake_java
    with pytest.raises(ValueError, match="steps"):
        metabolism.run_biotransformer(TERFENADINE, jar, steps=3)


def test_check_biotransformer_runs_help(fake_java):
    jar, calls = fake_java
    text = metabolism.check_biotransformer(jar)
    assert "usage" in text
    assert "--help" in calls[-1]


# --------------------------------------------------------------------------- #
# --metabolites-csv bypass
# --------------------------------------------------------------------------- #
def test_load_metabolites_csv(tmp_path):
    csv = tmp_path / "curated.csv"
    csv.write_text("smiles,id,reaction\nCC(=O)O,acetic_acid,manual\n")
    df = metabolism.load_metabolites_csv(str(csv), "drugX")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["compound_id"] == "acetic_acid" and row["parent_id"] == "drugX"
    assert row["reaction"] == "manual" and row["is_parent"] == False  # noqa: E712


def test_load_metabolites_csv_requires_smiles_column(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("name\nfoo\n")
    with pytest.raises(ValueError, match="smiles"):
        metabolism.load_metabolites_csv(str(csv), "drugX")


def test_load_metabolites_csv_without_parent_column_applies_to_every_drug_legacy(tmp_path):
    # Documents the legacy/single-drug behavior: no parent-scoping column
    # at all means every row is used for every drug that calls this.
    csv = tmp_path / "curated.csv"
    csv.write_text("smiles,id\nCC(=O)O,metabolite_a\n")
    df_for_x = metabolism.load_metabolites_csv(str(csv), "drugX")
    df_for_y = metabolism.load_metabolites_csv(str(csv), "drugY")
    assert len(df_for_x) == 1 and len(df_for_y) == 1  # same row, attached to both


def test_metabolites_csv_has_parent_column_detects_presence(tmp_path):
    with_col = tmp_path / "with.csv"
    with_col.write_text("smiles,parent_id\nCCO,drugA\n")
    without_col = tmp_path / "without.csv"
    without_col.write_text("smiles,id\nCCO,m1\n")
    assert metabolism.metabolites_csv_has_parent_column(str(with_col)) is True
    assert metabolism.metabolites_csv_has_parent_column(str(without_col)) is False


def test_metabolites_csv_has_parent_column_missing_file():
    with pytest.raises(FileNotFoundError):
        metabolism.metabolites_csv_has_parent_column("/no/such/file.csv")


def test_load_metabolites_csv_with_parent_column_scopes_correctly(tmp_path):
    """Regression for the real bug found while documenting this feature:
    a curated metabolite meant for one drug was silently attached to every
    drug sharing the same metabolites_csv file."""
    csv = tmp_path / "curated.csv"
    csv.write_text(
        "parent_id,smiles,id\n"
        "drugA,CC(=O)O,metabolite_for_a\n"
        "drugB,CCN,metabolite_for_b\n"
    )
    df_a = metabolism.load_metabolites_csv(str(csv), "drugA")
    df_b = metabolism.load_metabolites_csv(str(csv), "drugB")
    df_c = metabolism.load_metabolites_csv(str(csv), "drugC")  # not in the file at all

    assert list(df_a["compound_id"]) == ["metabolite_for_a"]
    assert list(df_b["compound_id"]) == ["metabolite_for_b"]
    assert df_c.empty  # drugC has no curated metabolites -- not every row, zero rows


def test_load_metabolites_csv_parent_column_alias_variants(tmp_path):
    for colname in ("parent", "drug_id", "drug"):
        csv = tmp_path / f"curated_{colname}.csv"
        csv.write_text(f"{colname},smiles,id\ndrugA,CCO,m1\ndrugB,CCN,m2\n")
        df = metabolism.load_metabolites_csv(str(csv), "drugA")
        assert list(df["compound_id"]) == ["m1"], colname


def test_generate_metabolites_multi_drug_run_scopes_curated_metabolites(tmp_path):
    """End-to-end version of the bug: two drugs, one shared metabolites_csv
    with a parent_id column -- each drug must only get its own metabolite."""
    csv = tmp_path / "curated.csv"
    csv.write_text(
        "parent_id,smiles,id\n"
        "terfenadine,OC(=O)c1ccc(cc1)C(C)(C)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1,fexofenadine\n"
    )
    terf_df = metabolism.generate_metabolites(
        "terfenadine", TERFENADINE, metabolites_csv=str(csv),
    )
    ibuprofen_df = metabolism.generate_metabolites(
        "ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", metabolites_csv=str(csv),
    )
    assert "fexofenadine" in set(terf_df["compound_id"])
    assert "fexofenadine" not in set(ibuprofen_df["compound_id"])
    assert len(ibuprofen_df) == 1 and ibuprofen_df.iloc[0]["is_parent"]  # parent only, correctly


def test_load_metabolites_csv_skips_bad_smiles(tmp_path):
    csv = tmp_path / "curated.csv"
    csv.write_text("smiles,id\nnotasmiles,bad\nCCO,good\n")
    df = metabolism.load_metabolites_csv(str(csv), "drugX")
    assert list(df["compound_id"]) == ["good"]


# --------------------------------------------------------------------------- #
# standardize_and_dedupe
# --------------------------------------------------------------------------- #
def test_standardize_and_dedupe_keeps_parent_and_shortest_pathway():
    df = pd.DataFrame([
        {"compound_id": "d", "parent_id": "d", "smiles": "CCO", "inchikey": "K1",
         "reaction": None, "enzyme": None, "depth": 0, "phase": None, "is_parent": True},
        {"compound_id": "d_m1", "parent_id": "d", "smiles": "CCN", "inchikey": "K2",
         "reaction": "r", "enzyme": "CYP", "depth": 2, "phase": "I", "is_parent": False},
        {"compound_id": "d_m1_dup", "parent_id": "d", "smiles": "CCN", "inchikey": "K2",
         "reaction": "r2", "enzyme": "CYP", "depth": 1, "phase": "I", "is_parent": False},
    ], columns=metabolism.STANDARD_COLUMNS)
    out = metabolism.standardize_and_dedupe(df)
    assert len(out) == 2
    assert out[out["inchikey"] == "K1"]["is_parent"].iloc[0]
    kept = out[out["inchikey"] == "K2"].iloc[0]
    assert kept["depth"] == 1 and kept["compound_id"] == "d_m1_dup"  # shortest pathway wins


def test_standardize_and_dedupe_empty():
    df = pd.DataFrame(columns=metabolism.STANDARD_COLUMNS)
    assert metabolism.standardize_and_dedupe(df).empty


# --------------------------------------------------------------------------- #
# generate_metabolites (full orchestration)
# --------------------------------------------------------------------------- #
def test_generate_metabolites_parent_only_with_no_jar_and_no_csv(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="gnina_vs.metabolism"):
        df = metabolism.generate_metabolites("aspirin", "CC(=O)Oc1ccccc1C(=O)O")
    assert len(df) == 1 and df.iloc[0]["is_parent"]
    assert "only the parent" in caplog.text


def test_generate_metabolites_with_biotransformer(tmp_path, fake_java):
    jar, _ = fake_java
    df = metabolism.generate_metabolites(
        "terf", TERFENADINE, jar_path=jar, cache_dir=tmp_path / "cache",
    )
    assert df["is_parent"].sum() == 1
    assert len(df) >= 2  # parent + at least one real metabolite from the fixture
    assert set(df["parent_id"]) == {"terf"}


def test_generate_metabolites_with_curated_csv_only(tmp_path):
    csv = tmp_path / "curated.csv"
    csv.write_text("smiles,id\nCC(=O)O,metabolite_a\n")
    df = metabolism.generate_metabolites("drugY", "CCO", metabolites_csv=str(csv))
    assert len(df) == 2
    assert set(df["compound_id"]) == {"drugY", "metabolite_a"}


def test_generate_metabolites_for_table_basic(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,Name,SMILES\nC1,phenol,c1ccccc1O\nC2,ethanol,CCO\n")
    df = metabolism.generate_metabolites_for_table(str(csv))
    assert set(df["parent_id"]) == {"C1", "C2"}
    assert df["is_parent"].all()  # no jar/csv -> parents only


def test_generate_metabolites_for_table_with_curated_csv_scoped_per_row(tmp_path):
    compounds = tmp_path / "compounds.csv"
    compounds.write_text("ID,SMILES\nterf,CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1\nibu,CC(C)Cc1ccc(cc1)C(C)C(=O)O\n")
    curated = tmp_path / "curated.csv"
    curated.write_text(
        "parent_id,smiles,id\n"
        "terf,OC(=O)c1ccc(cc1)C(C)(C)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1,fexofenadine\n"
    )
    df = metabolism.generate_metabolites_for_table(str(compounds), metabolites_csv=str(curated))
    terf_rows = df[df["parent_id"] == "terf"]
    ibu_rows = df[df["parent_id"] == "ibu"]
    assert "fexofenadine" in set(terf_rows["compound_id"])
    assert "fexofenadine" not in set(ibu_rows["compound_id"])
    assert len(ibu_rows) == 1  # parent only, correctly not contaminated


def test_generate_metabolites_for_table_id_falls_back_to_name_then_row(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("Name,SMILES\naspirin,CC(=O)Oc1ccccc1C(=O)O\n,CCO\n")
    df = metabolism.generate_metabolites_for_table(str(csv))
    ids = set(df["parent_id"])
    assert "aspirin" in ids
    assert any(i.startswith("compound_") for i in ids)


def test_generate_metabolites_for_table_dedupes_colliding_ids(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,SMILES\ndup,CCO\ndup,CCN\n")
    df = metabolism.generate_metabolites_for_table(str(csv))
    ids = set(df["parent_id"])
    assert "dup" in ids and "dup_2" in ids
    assert len(ids) == 2


def test_generate_metabolites_for_table_missing_smiles_column(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,Structure\nC1,CCO\n")
    with pytest.raises(ValueError, match="SMILES"):
        metabolism.generate_metabolites_for_table(str(csv))


def test_generate_metabolites_for_table_skips_invalid_smiles_rows(tmp_path, caplog):
    import logging
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,SMILES\ngood,CCO\nbad,not-a-smiles\n")
    with caplog.at_level(logging.WARNING, logger="gnina_vs.metabolism"):
        df = metabolism.generate_metabolites_for_table(str(csv))
    assert set(df["parent_id"]) == {"good"}
    assert "bad" in caplog.text


def test_generate_metabolites_for_table_all_invalid_raises(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,SMILES\nbad,not-a-smiles\n")
    with pytest.raises(ValueError, match="No valid parent compounds"):
        metabolism.generate_metabolites_for_table(str(csv))


def test_generate_metabolites_for_table_with_biotransformer(tmp_path, fake_java):
    jar, _ = fake_java
    csv = tmp_path / "compounds.csv"
    csv.write_text(f"ID,SMILES\nterf,{TERFENADINE}\n")
    df = metabolism.generate_metabolites_for_table(str(csv), jar_path=jar, cache_dir=tmp_path / "cache")
    assert (df["parent_id"] == "terf").all()
    assert len(df) >= 2  # parent + at least one real fixture metabolite


def test_generate_metabolites_invalid_parent_smiles():
    with pytest.raises(ValueError, match="Invalid parent SMILES"):
        metabolism.generate_metabolites("bad", "not-a-smiles")
