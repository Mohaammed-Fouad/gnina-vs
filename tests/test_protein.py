"""Phase A items 6, 7, 16, 17 (protein preparation)."""
import logging
import subprocess

from gnina_vs import protein
from pdbutil import atom_line, hetero, parse_atoms, peptide, write_pdb


def _pdb_with_modified_residues(tmp_path):
    lines, s = peptide("A", [("ALA", 1, "ATOM"), ("MSE", 2, "HETATM"), ("ALA", 3, "ATOM"),
                             ("SEP", 4, "HETATM"), ("ALA", 5, "ATOM")])
    lines.append("TER\n")
    lig, s = hetero("LIG", "A", 101, 12, (30, 30, 30), serial_start=s)
    lines += lig
    lines.append(atom_line(s, "O", "HOH", "A", 201, 40, 40, 40, record="HETATM", element="O"))
    return write_pdb(tmp_path / "mod.pdb", lines)


# ---- item 6 ---------------------------------------------------------------
def test_mse_converted_to_met_and_sep_kept(tmp_path, caplog):
    src = _pdb_with_modified_residues(tmp_path)
    out = str(tmp_path / "prep.pdb")
    with caplog.at_level(logging.INFO, logger="gnina_vs.protein"):
        protein.prepare_protein(src, out, add_hydrogens=False)
    atoms = parse_atoms(out)
    resnames = {(a["resseq"], a["resname"]) for a in atoms}
    assert (2, "MET") in resnames and not any(r == "MSE" for _, r in resnames)
    assert (4, "SEP") in resnames                              # in-chain modified residue kept
    assert {a["resseq"] for a in atoms} == {1, 2, 3, 4, 5}     # no gaps
    met = [a for a in atoms if a["resseq"] == 2]
    assert all(a["record"] == "ATOM" for a in met)
    assert "SD" in {a["name"] for a in met} and "SE" not in {a["name"] for a in met}
    assert {a["element"] for a in met if a["name"] == "SD"} == {"S"}
    assert not any(a["resname"] in ("LIG", "HOH") for a in atoms)  # ligand + water still stripped
    assert "MSE" in caplog.text and "MET" in caplog.text        # conversion is logged
    assert "SEP" in caplog.text


def test_real_structure_1a8o_keeps_all_residues(tmp_path, raw_1a8o):
    out = str(tmp_path / "prep.pdb")
    protein.prepare_protein(raw_1a8o, out, add_hydrogens=False)
    raw_res = {(a["chain"], a["resseq"]) for a in parse_atoms(raw_1a8o) if a["resname"] != "HOH"}
    out_res = {(a["chain"], a["resseq"]) for a in parse_atoms(out)}
    assert raw_res == out_res                                   # 4 MSE residues used to vanish
    atom_lines = [l for l in open(out) if l.startswith("ATOM") or l.startswith("HETATM")]
    assert not any(l[17:20].strip() == "MSE" for l in atom_lines)  # SEQRES may still list it


# ---- item 7 ---------------------------------------------------------------
def test_altloc_keeps_single_highest_occupancy_atom(tmp_path):
    lines, s = peptide("A", [("ALA", 1, "ATOM")])
    lines += [
        atom_line(s, "CG", "ALA", "A", 1, 5.0, 5.0, 5.0, altloc="A", occ=0.4),
        atom_line(s + 1, "CG", "ALA", "A", 1, 6.0, 6.0, 6.0, altloc="B", occ=0.6),
        atom_line(s + 2, "CD", "ALA", "A", 1, 7.0, 7.0, 7.0, altloc="A", occ=0.5),   # tie -> first
        atom_line(s + 3, "CD", "ALA", "A", 1, 8.0, 8.0, 8.0, altloc="B", occ=0.5),
    ]
    src = write_pdb(tmp_path / "alt.pdb", lines)
    out = str(tmp_path / "prep.pdb")
    protein.prepare_protein(src, out, add_hydrogens=False)
    atoms = parse_atoms(out)
    keys = [(a["chain"], a["resseq"], a["name"]) for a in atoms]
    assert len(keys) == len(set(keys)), "duplicate atoms remain"
    cg = next(a for a in atoms if a["name"] == "CG")
    cd = next(a for a in atoms if a["name"] == "CD")
    assert cg["xyz"] == (6.0, 6.0, 6.0)          # highest occupancy
    assert cd["xyz"] == (7.0, 7.0, 7.0)          # tie -> first in file
    assert all(a["altloc"] == " " for a in atoms)


# ---- item 16 --------------------------------------------------------------
def test_obabel_call_has_no_partialcharge(tmp_path, monkeypatch):
    src = _pdb_with_modified_residues(tmp_path)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        out = cmd[cmd.index("-O") + 1]
        open(out, "w").write(open(cmd[1]).read())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(protein.shutil, "which", lambda n, *a, **k: "/usr/bin/obabel")
    monkeypatch.setattr(protein.subprocess, "run", fake_run)
    protein.prepare_protein(src, str(tmp_path / "prep.pdb"), add_hydrogens=True, ph=7.4)
    assert calls and calls[0][0] == "obabel" and "-p" in calls[0]
    assert "--partialcharge" not in calls[0] and "gasteiger" not in calls[0]


# ---- item 17 --------------------------------------------------------------
def _cofactor_pdb(tmp_path):
    lines, s = peptide("A", [("ALA", 1, "ATOM"), ("ALA", 2, "ATOM")])
    nad, s = hetero("NAD", "A", 301, 44, (20, 20, 20), serial_start=s)
    zn, s = hetero("ZN", "A", 302, 1, (25, 25, 25), serial_start=s, element="ZN")
    return write_pdb(tmp_path / "cof.pdb", lines + nad + zn)


def test_stripped_cofactors_and_metals_are_reported(tmp_path, caplog):
    src = _cofactor_pdb(tmp_path)
    with caplog.at_level(logging.WARNING, logger="gnina_vs.protein"):
        protein.prepare_protein(src, str(tmp_path / "p.pdb"), add_hydrogens=False)
    assert "NAD" in caplog.text and "ZN" in caplog.text and "--keep-hetatm" in caplog.text


def test_keep_hetatm_retains_requested_residue(tmp_path):
    src = _cofactor_pdb(tmp_path)
    out = str(tmp_path / "p.pdb")
    protein.prepare_protein(src, out, keep_hetatm_residues={"NAD"}, add_hydrogens=False)
    names = {a["resname"] for a in parse_atoms(out)}
    assert "NAD" in names and "ZN" not in names
