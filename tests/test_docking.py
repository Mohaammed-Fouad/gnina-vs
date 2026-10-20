"""Phase A items 1, 4, 5, 13, 14 (docking module)."""
import logging
import os
from pathlib import Path

import pytest
from rdkit import Chem

from conftest import write_csv
from gnina_vs import docking, ligand


def _ligands(tmp_path, rows):
    csv = write_csv(tmp_path / "lig.csv", rows)
    sdf = str(tmp_path / "lig.sdf")
    ligand.prepare_ligands(csv, sdf)
    return sdf


ROWS_DUP = [{"ID": "A", "Name": "dup", "SMILES": "CCO"},
            {"ID": "B", "Name": "dup", "SMILES": "CCN"},
            {"ID": "C", "Name": "a/b c:d", "SMILES": "CCC"}]


def _dock(tmp_path, sdf, out_dir=None, **kw):
    return docking.run_gnina_docking(
        receptor_pdb="rec.pdb", ligands_sdf=sdf, center=(1.0, 2.0, 3.0),
        out_dir=out_dir or tmp_path / "dock", num_workers=2, **kw)


# ---- item 1 -----------------------------------------------------------------
@pytest.mark.parametrize("given,expected", [("refine", "refinement"), ("full", "all"),
                                            ("rescore", "rescore"), ("none", "none"),
                                            ("metrorescore", "metrorescore"), ("metrorefine", "metrorefine")])
def test_cnn_scoring_aliases_and_valid_values(tmp_path, fake_gnina, caplog, given, expected):
    sdf = _ligands(tmp_path, ROWS_DUP[:1])
    with caplog.at_level(logging.WARNING, logger="gnina_vs.docking"):
        _dock(tmp_path, sdf, cnn_scoring=given)
    cmd = fake_gnina.commands[0]
    assert cmd[cmd.index("--cnn_scoring") + 1] == expected
    assert ("deprecated" in caplog.text.lower()) == (given in ("refine", "full"))


def test_invalid_cnn_scoring_rejected(tmp_path, fake_gnina):
    sdf = _ligands(tmp_path, ROWS_DUP[:1])
    with pytest.raises(ValueError, match="cnn"):
        _dock(tmp_path, sdf, cnn_scoring="bogus")


# ---- item 4 -----------------------------------------------------------------
def test_duplicate_and_unsafe_names_do_not_collide(tmp_path, fake_gnina):
    sdf = _ligands(tmp_path, ROWS_DUP)
    out_dir = tmp_path / "dock"
    res = _dock(tmp_path, sdf, out_dir=out_dir)
    paths = [r["output_path"] for r in res]
    assert len(res) == 3 and len(set(paths)) == 3
    assert all(Path(p).parent == out_dir and Path(p).exists() for p in paths)
    for r in res:                                        # each output holds *its own* ligand
        m = next(iter(Chem.SDMolSupplier(r["output_path"], removeHs=False)))
        assert m.GetProp("SMILES") == r["smiles"]


# ---- item 5 -----------------------------------------------------------------
def test_failed_job_is_marked_and_stale_output_removed(tmp_path, fake_gnina):
    sdf = _ligands(tmp_path, ROWS_DUP[:2])
    out_dir = tmp_path / "dock"
    first = _dock(tmp_path, sdf, out_dir=out_dir)
    assert all(r["status"] == "ok" for r in first)
    fake_gnina.behavior = lambda mol, cmd: "fail"
    second = _dock(tmp_path, sdf, out_dir=out_dir)
    assert all(r["status"] == "failed" for r in second)
    assert not any(Path(r["output_path"]).exists() for r in second), "stale SDF left behind"


def test_timeout_is_marked(tmp_path, fake_gnina):
    sdf = _ligands(tmp_path, ROWS_DUP[:2])
    fake_gnina.behavior = lambda mol, cmd: "timeout" if mol.GetProp("MolIndex") == "0" else "ok"
    res = {r["ligand_id"]: r for r in _dock(tmp_path, sdf)}
    assert res["A"]["status"] == "timeout" and res["B"]["status"] == "ok"


# ---- item 13 ----------------------------------------------------------------
def test_seed_passed_to_gnina(tmp_path, fake_gnina):
    sdf = _ligands(tmp_path, ROWS_DUP[:1])
    _dock(tmp_path, sdf, seed=7)
    cmd = fake_gnina.commands[-1]
    assert cmd[cmd.index("--seed") + 1] == "7"
    _dock(tmp_path, sdf, out_dir=tmp_path / "d2")
    assert "--seed" not in fake_gnina.commands[-1]


# ---- item 14 ----------------------------------------------------------------
def test_autobox_ligand_replaces_center_and_size(tmp_path, fake_gnina):
    sdf = _ligands(tmp_path, ROWS_DUP[:1])
    ref = tmp_path / "ref.sdf"
    ref.write_text("dummy")
    docking.run_gnina_docking("rec.pdb", sdf, center=None, out_dir=tmp_path / "d",
                              num_workers=1, autobox_ligand=str(ref), autobox_add=6.0)
    cmd = fake_gnina.commands[-1]
    assert cmd[cmd.index("--autobox_ligand") + 1] == str(ref)
    assert cmd[cmd.index("--autobox_add") + 1] == "6.0"
    assert "--center_x" not in cmd and "--size_x" not in cmd


def test_oversized_box_warns(tmp_path, fake_gnina, caplog):
    sdf = _ligands(tmp_path, ROWS_DUP[:2])              # ethanol/ethylamine: tiny
    with caplog.at_level(logging.WARNING, logger="gnina_vs.docking"):
        _dock(tmp_path, sdf, box_size=60.0)
    assert any("box" in r.getMessage().lower() and "larger" in r.getMessage().lower()
               for r in caplog.records)


def test_reasonable_box_does_not_warn(tmp_path, fake_gnina, caplog):
    big = "CCCCCCCCCCCCCCCCCC(=O)NCCCCCCCCCCCCCCCC"      # ~ 25+ A extent
    sdf = _ligands(tmp_path, [{"ID": "A", "Name": "long", "SMILES": big}])
    with caplog.at_level(logging.WARNING, logger="gnina_vs.docking"):
        _dock(tmp_path, sdf, box_size=25.0)
    assert not any("larger" in r.getMessage().lower() for r in caplog.records)
