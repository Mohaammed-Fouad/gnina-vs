"""Phase A items 2, 4 (recover), 5, 15, 18 (reporter)."""
import logging

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from conftest import TABLE_CNN, TABLE_VINA, write_csv
from gnina_vs import docking, ligand, reporter


def _entry(name, idx, log, out="", status=None):
    e = {"mol_name": name, "smiles": "C", "index": idx, "ligand_id": name,
         "output_path": out, "raw_log": log}
    if status:
        e["status"] = status
    return e


def _sdf_with_props(path, **props):
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(m, randomSeed=1)
    for k, v in props.items():
        m.SetProp(k, str(v))
    w = Chem.SDWriter(str(path)); w.write(m); w.close()
    return str(path)


# ---- parsing (regression + item 2) -------------------------------------------
def test_parse_full_cnn_table():
    r = reporter._parse_gnina_log(TABLE_CNN.format(aff=-8.32, cs=0.7574, ca=6.438))
    assert r == {"vina_affinity": -8.32, "cnn_score": 0.7574, "cnn_affinity": 6.438}


def test_parse_vina_only_table_gives_affinity():
    r = reporter._parse_gnina_log(TABLE_VINA.format(aff=-6.1))
    assert r["vina_affinity"] == -6.1 and r["cnn_score"] is None and r["cnn_affinity"] is None


# ---- item 2 -------------------------------------------------------------------
def test_cnn_none_run_exports_rows_ranked_by_vina(tmp_path):
    res = [_entry("weak", 0, TABLE_VINA.format(aff=-5.0)),
           _entry("strong", 1, TABLE_VINA.format(aff=-9.0)),
           _entry("mid", 2, TABLE_VINA.format(aff=-7.0))]
    df = reporter.extract_results(res, None, str(tmp_path / "o.csv"), str(tmp_path / "o.xlsx"), top_n="all")
    xl = pd.read_excel(tmp_path / "o.xlsx")
    assert len(xl) == 3
    assert list(xl["mol_name"]) == ["strong", "mid", "weak"]          # most negative first


def test_top_n_limits_rows_in_vina_mode(tmp_path):
    res = [_entry(f"m{i}", i, TABLE_VINA.format(aff=-5.0 - i)) for i in range(5)]
    reporter.extract_results(res, None, str(tmp_path / "o.csv"), str(tmp_path / "o.xlsx"), top_n=2)
    assert len(pd.read_excel(tmp_path / "o.xlsx")) == 2


def test_no_scores_at_all_still_writes_csv_and_does_not_crash(tmp_path):
    res = [_entry("x", 0, "garbage")]
    df = reporter.extract_results(res, None, str(tmp_path / "o.csv"), str(tmp_path / "o.xlsx"))
    assert len(pd.read_csv(tmp_path / "o.csv")) == 1


# ---- item 5 -------------------------------------------------------------------
def test_failed_entry_never_reads_stale_sdf(tmp_path):
    stale = _sdf_with_props(tmp_path / "old.sdf", minimizedAffinity=-9.9, CNNscore=0.99, CNNaffinity=9.9)
    res = [_entry("x", 0, "TIMEOUT", out=stale, status="timeout"),
           _entry("y", 1, "ERROR", out=stale, status="failed")]
    df = reporter.extract_results(res, None, str(tmp_path / "o.csv"), str(tmp_path / "o.xlsx"))
    assert list(df["status"]) == ["timeout", "failed"]
    assert df["cnn_affinity"].isna().all() and df["vina_affinity"].isna().all()


def test_ok_entry_can_still_fall_back_to_sdf_properties(tmp_path):
    sdf = _sdf_with_props(tmp_path / "d.sdf", minimizedAffinity=-7.7, CNNscore=0.8, CNNaffinity=6.6)
    df = reporter.extract_results([_entry("x", 0, "no table here", out=sdf, status="ok")], None,
                                  str(tmp_path / "o.csv"), str(tmp_path / "o.xlsx"))
    assert df.iloc[0]["cnn_affinity"] == pytest.approx(6.6)
    assert df.iloc[0]["status"] == "ok"


# ---- item 15 ------------------------------------------------------------------
def test_pose_quality_flag_and_cnn_score_always_present(tmp_path):
    res = [_entry("good", 0, TABLE_CNN.format(aff=-8, cs=0.9, ca=7.0)),
           _entry("bad_pose", 1, TABLE_CNN.format(aff=-9, cs=0.2, ca=8.0)),   # best affinity, bad pose
           _entry("noscore", 2, "TIMEOUT", status="timeout")]
    df = reporter.extract_results(res, None, str(tmp_path / "o.csv"), str(tmp_path / "o.xlsx"), top_n="all",
                                  merged_sdf=None)
    csv = pd.read_csv(tmp_path / "o.csv")
    xl = pd.read_excel(tmp_path / "o.xlsx")
    assert {"cnn_score", "cnn_affinity", "pose_quality_flag"} <= set(csv.columns)
    assert {"cnn_score", "cnn_affinity", "pose_quality_flag"} <= set(xl.columns)
    flags = dict(zip(csv["mol_name"], csv["pose_quality_flag"]))
    assert flags["good"] == "ok" and flags["bad_pose"] == "low_cnn_score"
    assert flags["noscore"] == "not_scored"    # timed out: never scored at all
    assert set(xl["mol_name"]) == {"good", "bad_pose"}                 # flagged row NOT removed


def test_pose_quality_flag_no_cnn_score_for_successful_vina_only_job(tmp_path):
    res = [_entry("vina_only", 0, TABLE_VINA.format(aff=-6.5), status="ok")]
    df = reporter.extract_results(res, None, str(tmp_path / "o.csv"), str(tmp_path / "o.xlsx"), top_n="all")
    assert df.iloc[0]["pose_quality_flag"] == "no_cnn_score"


# ---- item 4 (recover) -----------------------------------------------------------
def test_recover_finds_each_duplicate_name_output(tmp_path, fake_gnina):
    csv = write_csv(tmp_path / "l.csv", [{"ID": "A", "Name": "dup", "SMILES": "CCO"},
                                         {"ID": "B", "Name": "dup", "SMILES": "CCN"}])
    lig_sdf = str(tmp_path / "prepared.sdf")
    ligand.prepare_ligands(csv, lig_sdf)
    docking.run_gnina_docking("rec.pdb", lig_sdf, (0, 0, 0), out_dir=tmp_path / "dock", num_workers=2)
    df = reporter.recover_results(lig_sdf, tmp_path / "dock", csv, str(tmp_path / "o.csv"),
                                  str(tmp_path / "o.xlsx"), top_n="all")
    assert df["cnn_affinity"].nunique() == 2 and len(df) == 2
    assert set(df["ID"]) == {"A", "B"}


# ---- item 18 ------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [("all", 0), ("ALL", 0), ("0", 0), (0, 0), ("25", 25), (7, 7)])
def test_parse_top_n_valid(value, expected):
    assert reporter.parse_top_n(value) == expected


@pytest.mark.parametrize("value", ["abc", "-3", "1.5", ""])
def test_parse_top_n_invalid(value):
    with pytest.raises(ValueError, match="top-n"):
        reporter.parse_top_n(value)
