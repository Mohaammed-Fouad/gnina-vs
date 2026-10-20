"""
Phase 4 tests: gnina_vs.validation -- co-crystal ligand extraction, RMSD
between two poses, full redocking flow with the fake-gnina fixture, and
reference-ligand docking. Uses the same pdbutil helpers as test_pocket.py.
"""
from pathlib import Path

import pytest
from rdkit import Chem

from gnina_vs import config as config_mod
from gnina_vs import validation
from pdbutil import atom_line, hetero, peptide, write_pdb


def _receptor_with_ligand(tmp_path, resname="LIG", n_atoms=6, center=(10.0, 10.0, 10.0)):
    protein_lines, s = peptide("A", [("ALA", 1, "ATOM"), ("ALA", 2, "ATOM")])
    lig_lines, s = hetero(resname, "A", 401, n_atoms, center, serial_start=s, element="C")
    return write_pdb(tmp_path / "receptor.pdb", protein_lines + lig_lines)


def _target_config(**overrides):
    base = dict(id="T1", role="therapeutic", pdb="whatever.pdb", center_from_ligand=True,
               ligand_resname="LIG", chain=None, keep_hetatm=[], reference_ligands=[], center=None)
    base.update(overrides)
    return config_mod.TargetConfig(**base)


# --------------------------------------------------------------------------- #
# extract_cocrystal_ligand / heavy_atom_rmsd
# --------------------------------------------------------------------------- #
def test_extract_cocrystal_ligand_found(tmp_path):
    pdb = _receptor_with_ligand(tmp_path)
    mol = validation.extract_cocrystal_ligand(pdb, "LIG")
    assert mol is not None and mol.GetNumAtoms() == 6


def test_extract_cocrystal_ligand_missing_returns_none(tmp_path):
    pdb = _receptor_with_ligand(tmp_path)
    assert validation.extract_cocrystal_ligand(pdb, "NOPE") is None


def test_extract_cocrystal_ligand_chain_filter(tmp_path):
    lines_a, s = peptide("A", [("ALA", 1, "ATOM")])
    lig_a, s = hetero("LIG", "A", 401, 4, (0, 0, 0), serial_start=s)
    lig_b, s = hetero("LIG", "B", 401, 4, (50, 50, 50), serial_start=s)
    pdb = write_pdb(tmp_path / "dimer.pdb", lines_a + lig_a + lig_b)
    mol_a = validation.extract_cocrystal_ligand(pdb, "LIG", chain="A")
    mol_b = validation.extract_cocrystal_ligand(pdb, "LIG", chain="B")
    assert mol_a.GetNumAtoms() == 4 and mol_b.GetNumAtoms() == 4
    pos_a = mol_a.GetConformer().GetPositions()
    assert pos_a[0][0] < 10  # near (0,0,0), not (50,50,50)


def test_heavy_atom_rmsd_identical_pose_is_zero(tmp_path):
    pdb = _receptor_with_ligand(tmp_path)
    mol = validation.extract_cocrystal_ligand(pdb, "LIG")
    rmsd = validation.heavy_atom_rmsd(mol, mol)
    assert rmsd == pytest.approx(0.0, abs=1e-6)


def test_heavy_atom_rmsd_known_shift():
    from rdkit.Geometry import Point3D
    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=1)
    shifted = Chem.Mol(mol)
    conf = shifted.GetConformer()
    for i in range(shifted.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + 1.0, p.y, p.z))  # shift by exactly 1 A
    rmsd = validation.heavy_atom_rmsd(mol, shifted)
    assert rmsd == pytest.approx(1.0, abs=1e-6)


def test_heavy_atom_rmsd_mismatched_atom_count_returns_none(tmp_path):
    pdb6 = _receptor_with_ligand(tmp_path, n_atoms=6)
    pdb4_path = tmp_path / "other.pdb"
    lines, s = peptide("A", [("ALA", 1, "ATOM")])
    lig, s = hetero("LIG", "A", 401, 4, (10, 10, 10), serial_start=s)
    write_pdb(pdb4_path, lines + lig)
    mol6 = validation.extract_cocrystal_ligand(pdb6, "LIG")
    mol4 = validation.extract_cocrystal_ligand(str(pdb4_path), "LIG")
    assert validation.heavy_atom_rmsd(mol6, mol4) is None


# --------------------------------------------------------------------------- #
# validate_target: no co-crystal ligand configured
# --------------------------------------------------------------------------- #
def test_validate_target_without_ligand_resname_is_not_validated(tmp_path):
    target = _target_config(center_from_ligand=False, ligand_resname=None, center=(1.0, 2.0, 3.0))
    result = validation.validate_target(target, "raw.pdb", "prep.pdb", (1.0, 2.0, 3.0),
                                        docking_config=None, target_dir=tmp_path, seed=1)
    assert result == {"target_id": "T1", "status": validation.NOT_VALIDATED_NO_REFERENCE, "rmsd": None}


def test_validate_target_ligand_not_found_in_raw_pdb(tmp_path):
    pdb = _receptor_with_ligand(tmp_path, resname="LIG")
    target = _target_config(ligand_resname="MISSING")
    result = validation.validate_target(target, pdb, pdb, (10.0, 10.0, 10.0),
                                        docking_config=None, target_dir=tmp_path, seed=1)
    assert result["status"].startswith("not validated: could not find")
    assert result["rmsd"] is None


# --------------------------------------------------------------------------- #
# Full redocking flow with the fake gnina fixture
# --------------------------------------------------------------------------- #
def test_redock_cocrystal_ligand_pass(tmp_path, fake_gnina, monkeypatch):
    from gnina_vs.config import DockingConfig
    pdb = _receptor_with_ligand(tmp_path)
    dcfg = DockingConfig()

    # Make the fake gnina write back the *exact same* coordinates it was given,
    # so RMSD should come out at (near) zero -> "pass".
    def identity_run(cmd, capture_output=True, text=True, timeout=None, **kw):
        import subprocess
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "fake\n", "")
        if "--help" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        lig = cmd[cmd.index("--ligand") + 1]
        out = cmd[cmd.index("--out") + 1]
        mol = next(iter(Chem.SDMolSupplier(lig, removeHs=False)))
        mol.SetProp("minimizedAffinity", "-7.0")
        w = Chem.SDWriter(out); w.write(mol); w.close()
        return subprocess.CompletedProcess(cmd, 0, "mode 1 -7.0 0.0 0.0\n", "")

    from gnina_vs import docking as docking_mod
    monkeypatch.setattr(docking_mod.subprocess, "run", identity_run)
    monkeypatch.setattr(docking_mod.shutil, "which", lambda n, *a, **k: "/usr/bin/" + str(n))

    result = validation.redock_cocrystal_ligand(
        raw_pdb=pdb, resname="LIG", chain=None, prepared_receptor=pdb,
        center=(10.0, 10.0, 10.0), docking_config=dcfg, target_dir=tmp_path, seed=1,
    )
    assert result["status"] == "pass"
    assert result["rmsd"] == pytest.approx(0.0, abs=1e-3)


def test_redock_cocrystal_ligand_missing_residue_reports_clearly(tmp_path):
    from gnina_vs.config import DockingConfig
    pdb = _receptor_with_ligand(tmp_path)
    result = validation.redock_cocrystal_ligand(
        raw_pdb=pdb, resname="ABSENT", chain=None, prepared_receptor=pdb,
        center=(10.0, 10.0, 10.0), docking_config=DockingConfig(), target_dir=tmp_path, seed=1,
    )
    assert result["status"].startswith("not validated:")
    assert result["rmsd"] is None


# --------------------------------------------------------------------------- #
# Reference ligand docking
# --------------------------------------------------------------------------- #
def test_dock_reference_ligands_empty_when_none_configured(tmp_path):
    target = _target_config(reference_ligands=[])
    df = validation.dock_reference_ligands(target, None, "prep.pdb", (1, 2, 3), tmp_path, seed=1)
    assert df.empty


def test_dock_reference_ligands_runs_and_scores(tmp_path, fake_gnina):
    from gnina_vs.config import DockingConfig, ReferenceLigand
    target = _target_config(reference_ligands=[ReferenceLigand(name="morphine_like", smiles="CCO")])
    df = validation.dock_reference_ligands(
        target, DockingConfig(), "receptor.pdb", (1.0, 2.0, 3.0), tmp_path, seed=3,
    )
    assert len(df) == 1
    assert df.iloc[0]["compound_id"] == "ref__morphine_like"
    assert df.iloc[0]["status"] == "ok"
