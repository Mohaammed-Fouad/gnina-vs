"""Phase A items 8, 10, 11, 12 (pocket detection)."""
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from gnina_vs import pocket, protein
from pdbutil import atom_line, hetero, peptide, write_pdb

DATA = Path(__file__).parent / "data"


def _centroid(pts):
    n = len(pts)
    return tuple(sum(p[i] for p in pts) / n for i in range(3))


# ---- item 8: chain-aware residue specs ------------------------------------
def _dimer(tmp_path):
    a, s = peptide("A", [("ALA", 119, "ATOM"), ("TYR", 120, "ATOM"), ("ALA", 121, "ATOM")], serial_start=1)
    b, s = peptide("B", [("ALA", 119, "ATOM"), ("TYR", 120, "ATOM"), ("ALA", 121, "ATOM")],
                   x_start=0.0, y=0.0, z=50.0, serial_start=s)
    return write_pdb(tmp_path / "dimer.pdb", a + ["TER\n"] + b)


def test_residue_spec_without_chain_is_ambiguous_in_multichain(tmp_path):
    pdb = _dimer(tmp_path)
    with pytest.raises(ValueError, match="(?i)chain"):
        pocket.centroid_from_residues(pdb, pocket.parse_residue_spec("TYR120"))


def test_residue_spec_with_chain(tmp_path):
    pdb = _dimer(tmp_path)
    ca = pocket.centroid_from_residues(pdb, pocket.parse_residue_spec("A:TYR120"))
    cb = pocket.centroid_from_residues(pdb, pocket.parse_residue_spec("B:TYR120"))
    assert ca[2] == pytest.approx(0.0, abs=0.6) and cb[2] == pytest.approx(50.0, abs=0.6)
    both = pocket.centroid_from_residues(pdb, pocket.parse_residue_spec("A:TYR120,B:TYR120"))
    assert both[2] == pytest.approx(25.0, abs=0.6)


def test_residue_spec_without_chain_still_works_for_single_chain(tmp_path):
    a, _ = peptide("A", [("ALA", 1, "ATOM"), ("TYR", 2, "ATOM")])
    pdb = write_pdb(tmp_path / "mono.pdb", a)
    assert pocket.centroid_from_residues(pdb, pocket.parse_residue_spec("TYR2")) is not None


# ---- item 10: Tier 3 must not silently return the protein centre ----------
def test_no_ligand_no_fpocket_fails_loudly_not_protein_centre(tmp_path, monkeypatch, raw_1a8o):
    prepared = str(tmp_path / "prep.pdb")
    protein.prepare_protein(raw_1a8o, prepared, add_hydrogens=False)
    monkeypatch.setattr(pocket.shutil, "which", lambda *a, **k: None)   # no fpocket
    with pytest.raises(RuntimeError) as exc:
        pocket.detect_active_site(prepared, raw_pdb=prepared)           # apo, no HETATM
    msg = str(exc.value).lower()
    assert "coordinates" in msg or "center-x" in msg
    assert "ligand" in msg


# ---- item 11: fpocket parsing ---------------------------------------------
def _pqr_centroid(path):
    pts = [(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in open(path) if l.startswith("ATOM")]
    return _centroid(pts)


def test_fpocket_centroid_parsed_from_pockets_dir(tmp_path, monkeypatch):
    fixture = DATA / "fpocket_real" / "receptor_out"          # genuine fpocket 4 output
    src = tmp_path / "in.pdb"
    src.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\nEND\n")

    def fake_run(cmd, cwd=None, **kw):
        shutil.copytree(fixture, Path(cwd) / "receptor_out")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pocket.shutil, "which", lambda n, *a, **k: "/usr/bin/fpocket")
    monkeypatch.setattr(pocket.subprocess, "run", fake_run)
    got = pocket._tier2_fpocket(str(src))
    assert got is not None
    expected = _pqr_centroid(fixture / "pockets" / "pocket1_vert.pqr")
    assert got == pytest.approx(expected, abs=0.01)


@pytest.mark.skipif(shutil.which("fpocket") is None, reason="fpocket not installed")
def test_fpocket_real_run(tmp_path, raw_1a8o):
    prepared = str(tmp_path / "prep.pdb")
    protein.prepare_protein(raw_1a8o, prepared, add_hydrogens=False)
    got = pocket._tier2_fpocket(prepared)
    assert got is not None
    xs = [(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in open(prepared) if l.startswith("ATOM")]
    for i in range(3):
        assert min(p[i] for p in xs) - 5 < got[i] < max(p[i] for p in xs) + 5


# ---- item 12: explicit ligand selection + candidate logging ---------------
def _multi_ligand_pdb(tmp_path):
    lines, s = peptide("A", [("ALA", 1, "ATOM"), ("ALA", 2, "ATOM")])
    atp, s = hetero("ATP", "A", 401, 31, (10, 10, 10), serial_start=s)
    lig_a, s = hetero("LIG", "A", 402, 20, (30, 0, 0), serial_start=s)
    lig_b, s = hetero("LIG", "B", 402, 20, (60, 0, 0), serial_start=s)
    nag, s = hetero("NAG", "A", 403, 14, (0, 40, 0), serial_start=s)
    pge, s = hetero("PGE", "A", 404, 5, (0, 0, 40), serial_start=s)
    return write_pdb(tmp_path / "multi.pdb", lines + atp + lig_a + lig_b + nag + pge)


def test_explicit_ligand_resname_allows_cofactor_site(tmp_path):
    pdb = _multi_ligand_pdb(tmp_path)
    c = pocket.detect_active_site(pdb, raw_pdb=pdb, ligand_resname="ATP")
    assert c == pytest.approx((10, 10, 10), abs=0.01)


def test_explicit_ligand_resname_and_chain(tmp_path):
    pdb = _multi_ligand_pdb(tmp_path)
    b = pocket.detect_active_site(pdb, raw_pdb=pdb, ligand_resname="LIG", ligand_chain="B")
    assert b == pytest.approx((60, 0, 0), abs=0.01)


def test_missing_ligand_resname_fails_clearly(tmp_path):
    lines, _ = peptide("A", [("ALA", 1, "ATOM")])
    pdb = write_pdb(tmp_path / "apo.pdb", lines)
    with pytest.raises(ValueError) as exc:
        pocket.detect_active_site(pdb, raw_pdb=pdb, ligand_resname="XYZ")
    assert "XYZ" in str(exc.value)


def test_auto_selection_logs_all_candidates_and_warns_if_ambiguous(tmp_path, caplog):
    # BLA is a made-up, non-excluded placeholder (not a real glycan/cofactor/
    # solvent code) standing in for "some other small-molecule candidate".
    lines, s = peptide("A", [("ALA", 1, "ATOM")])
    bla, s = hetero("BLA", "A", 403, 14, (0, 40, 0), serial_start=s)
    lig, s = hetero("LIG", "A", 402, 20, (30, 0, 0), serial_start=s)
    pge, s = hetero("PGE", "A", 404, 5, (0, 0, 40), serial_start=s)
    pdb = write_pdb(tmp_path / "amb.pdb", lines + bla + lig + pge)
    with caplog.at_level(logging.INFO, logger="gnina_vs.pocket"):
        pocket.detect_active_site(pdb, raw_pdb=pdb)
    for name in ("BLA", "LIG", "PGE"):
        assert name in caplog.text
    assert any(r.levelno >= logging.WARNING and "ambiguous" in r.getMessage().lower()
               for r in caplog.records)


def test_in_chain_modified_residues_are_not_ligand_candidates(raw_1a8o):
    # 1A8O contains only water + 4 selenomethionines; there is no ligand.
    assert pocket._tier1_cocrystal(raw_1a8o) is None


# ---- regression: glycosylation sugars (e.g. 1CX2's NAG) must never win ----
def test_glycosylation_sugar_never_outranks_a_smaller_real_ligand(tmp_path):
    """Reproduces the 1CX2 (COX-2) failure mode: a glycoprotein carries
    N-linked NAG sugars at its surface, and a real but modest-sized ligand
    sits in the actual pocket. Before NAG/glycans were added to the
    exclusion list, a single ~15-heavy-atom NAG could beat a smaller real
    ligand (or a ligand diluted across multiple glycan copies) under the
    "most heavy atoms wins" heuristic."""
    lines, s = peptide("A", [("ALA", 1, "ATOM"), ("ALA", 2, "ATOM")])
    # three NAG glycosylation sites, 14 heavy atoms each (close to NAG's
    # real ~15), scattered on the "surface" far from the real ligand
    for i, center in enumerate([(0, 40, 0), (40, 0, 0), (0, 0, 40)]):
        nag, s = hetero("NAG", "A", 500 + i, 14, center, serial_start=s)
        lines += nag
    # the real ligand: a small fragment-sized HETATM blob (10 heavy atoms,
    # smaller than any single NAG) at the actual binding site
    real_ligand, s = hetero("LIG", "A", 600, 10, (15.0, 15.0, 15.0), serial_start=s)
    lines += real_ligand
    pdb = write_pdb(tmp_path / "glyco.pdb", lines)

    center = pocket._tier1_cocrystal(pdb)
    assert center == pytest.approx((15.0, 15.0, 15.0), abs=0.01), (
        "Tier 1 picked a glycosylation sugar instead of the real ligand"
    )


def test_tier1_counts_heavy_atoms_only_not_hydrogens(tmp_path):
    """Regression for the precise mechanism behind the 1CX2 (COX-2) bug:
    fetching the real file showed it carries explicit hydrogens throughout
    (added by later reprocessing; the structure is only 3.0 A resolution,
    so H was never experimentally observed) and the real ligand (S58) and
    a glycosylation sugar (NAG) are close enough in total atom count,
    once hydrogens are counted in, to tie or flip Tier 1's "most atoms"
    comparison. Using made-up residue names here (not NAG/S58 specifically,
    since NAG is now excluded outright by the glycan fix) isolates this as
    its own, independently-verified bug: counting hydrogens at all, for
    ANY pair of HETATM groups, is wrong regardless of what they're named.
    """
    lines, s = peptide("A", [("ALA", 1, "ATOM")])
    # "BIG": 10 heavy atoms, no hydrogens modeled -- the real, larger ligand
    for k in range(10):
        lines.append(atom_line(s, f"C{k+1}", "BIG", "A", 500, 5.0 + k * 0.9, 5.0, 5.0,
                               record="HETATM", element="C"))
        s += 1
    # "SML": only 6 heavy atoms, but 6 explicit hydrogens modeled too --
    # under hydrogen-inclusive counting (12) this beats BIG's heavy-only
    # count (10) despite genuinely being the smaller molecule.
    for k in range(6):
        lines.append(atom_line(s, f"C{k+1}", "SML", "A", 600, 30.0 + k * 0.9, 30.0, 30.0,
                               record="HETATM", element="C"))
        s += 1
    for k in range(6):
        lines.append(atom_line(s, f"H{k+1}", "SML", "A", 600, 30.0 + k * 0.9, 30.5, 30.5,
                               record="HETATM", element="H"))
        s += 1
    pdb = write_pdb(tmp_path / "hydrogens.pdb", lines)

    center = pocket._tier1_cocrystal(pdb)
    assert center == pytest.approx((5.0 + 4.5 * 0.9, 5.0, 5.0), abs=0.01), (
        "Tier 1 picked the molecule with fewer heavy atoms because it counted hydrogens"
    )


def test_tier1_logged_atom_count_excludes_hydrogens(tmp_path, caplog):
    lines, s = peptide("A", [("ALA", 1, "ATOM")])
    for k in range(4):
        lines.append(atom_line(s, f"C{k+1}", "LIG", "A", 500, 5.0 + k * 0.9, 5.0, 5.0,
                               record="HETATM", element="C"))
        s += 1
    for k in range(8):  # more H's than heavy atoms -- would dominate a naive count
        lines.append(atom_line(s, f"H{k+1}", "LIG", "A", 500, 5.0 + k * 0.5, 5.5, 5.5,
                               record="HETATM", element="H"))
        s += 1
    pdb = write_pdb(tmp_path / "hcount.pdb", lines)
    with caplog.at_level(logging.INFO, logger="gnina_vs.pocket"):
        pocket._tier1_cocrystal(pdb)
    assert "4 heavy atoms" in caplog.text  # not 12


def test_glycans_excluded_even_when_no_other_candidate_present(tmp_path):
    # An apo-looking structure whose only HETATM residues are glycans
    # must be treated as having no ligand candidate at all (None), not
    # silently pick a sugar.
    lines, s = peptide("A", [("ALA", 1, "ATOM")])
    nag, s = hetero("NAG", "A", 500, 14, (10, 10, 10), serial_start=s)
    bma, s = hetero("BMA", "A", 501, 14, (20, 20, 20), serial_start=s)
    pdb = write_pdb(tmp_path / "glyco_only.pdb", lines + nag + bma)
    assert pocket._tier1_cocrystal(pdb) is None
