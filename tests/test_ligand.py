"""Phase A items 3, 4 (ids), 9, 13 (RDKit seed)."""
import logging
import re

import pytest
from rdkit import Chem

from conftest import write_csv
from gnina_vs import ligand


def _prep(tmp_path, rows, **kw):
    csv = write_csv(tmp_path / "lig.csv", rows)
    sdf = str(tmp_path / "lig.sdf")
    res = ligand.prepare_ligands(csv, sdf, **kw)
    return res, sdf


def _mols(sdf):
    return [m for m in Chem.SDMolSupplier(sdf, removeHs=False) if m is not None]


# ---- item 3 ---------------------------------------------------------------
def test_salts_stripped_and_not_lost(tmp_path, caplog):
    rows = [{"ID": "A", "Name": "acetate", "SMILES": "CC(=O)[O-].[Na+]"},
            {"ID": "B", "Name": "alaHCl", "SMILES": "CC(N)C(=O)O.Cl"},   # used to fail MMFF94 -> lost
            {"ID": "C", "Name": "plain", "SMILES": "CCO"}]
    with caplog.at_level(logging.INFO, logger="gnina_vs.ligand"):
        res, sdf = _prep(tmp_path, rows)
    mols = _mols(sdf)
    assert len(mols) == 3 and len(res) == 3
    for m in mols:
        assert len(Chem.GetMolFrags(m)) == 1, m.GetProp("_Name")
        assert "." not in m.GetProp("SMILES")
    heavy = {m.GetProp("_Name"): m.GetNumHeavyAtoms() for m in mols}
    assert heavy["acetate"] == 4 and heavy["alaHCl"] == 6
    assert "Na" in caplog.text and "Cl" in caplog.text            # removed fragments logged


# ---- item 4 (ids) -----------------------------------------------------------
def test_compound_ids_unique_and_filesystem_safe(tmp_path):
    rows = [{"ID": "A", "Name": "dup", "SMILES": "CCO"},
            {"ID": "B", "Name": "dup", "SMILES": "CCN"},
            {"ID": "C", "Name": "a/b c:d*?", "SMILES": "CCC"},
            {"ID": "D", "Name": "", "SMILES": "CCCC"}]
    res, sdf = _prep(tmp_path, rows)
    ids = [r["compound_id"] for r in res]
    assert len(ids) == len(set(ids)) == 4
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+", i) for i in ids), ids
    assert [m.GetProp("CompoundID") for m in _mols(sdf)] == ids


# ---- item 9 -----------------------------------------------------------------
def test_undefined_stereocentres_warn(tmp_path, caplog):
    rows = [{"ID": "A", "Name": "racemic_ala", "SMILES": "CC(N)C(=O)O"},
            {"ID": "B", "Name": "L_ala", "SMILES": "C[C@H](N)C(=O)O"}]
    with caplog.at_level(logging.WARNING, logger="gnina_vs.ligand"):
        _prep(tmp_path, rows)
    warned = [r.getMessage() for r in caplog.records if "stereo" in r.getMessage().lower()]
    assert any("racemic_ala" in m for m in warned)
    assert not any("L_ala" in m for m in warned)


# ---- item 13 (RDKit side) -----------------------------------------------------
def test_embedding_is_deterministic_for_a_seed(tmp_path):
    csv = write_csv(tmp_path / "l.csv", [{"ID": "A", "Name": "flex", "SMILES": "CCCCCCCC(=O)NCCc1ccccc1"}])
    s1, s2, s3 = (str(tmp_path / n) for n in ("a.sdf", "b.sdf", "c.sdf"))
    ligand.prepare_ligands(csv, s1, seed=11)
    ligand.prepare_ligands(csv, s2, seed=11)
    ligand.prepare_ligands(csv, s3, seed=12)
    pos = lambda f: _mols(f)[0].GetConformer().GetPositions().tolist()
    assert pos(s1) == pos(s2)
    assert pos(s1) != pos(s3)
