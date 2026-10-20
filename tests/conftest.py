"""Shared fixtures: a fake `gnina` executable (no real Gnina needed) and helpers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent))

DATA = Path(__file__).parent / "data"
TABLE_CNN = (
    "mode |  affinity  |  intramol  |    CNN     |   CNN\n"
    "     | (kcal/mol) | (kcal/mol) | pose score | affinity\n"
    "-----+------------+------------+------------+----------\n"
    "    1       {aff:.2f}       -0.34       {cs:.4f}      {ca:.3f}\n"
    "    2       -7.10       -0.30       0.2000      5.000\n"
)
TABLE_VINA = (
    "mode |   affinity | dist from best mode\n"
    "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
    "-----+------------+----------+----------\n"
    "    1       {aff:.1f}      0.000      0.000\n"
    "    2       -5.1      1.200      2.300\n"
)


HELP_TEXT = (
    "Convolutional neural net (CNN) scoring:\n"
    "  --cnn_scoring arg (=1)             Amount of CNN scoring: none, rescore \n"
    "                                     (default), refinement, metrorescore \n"
    "                                     (metropolis+rescore), metrorefine \n"
    "                                     (metropolis+refine), all\n"
)


class FakeGnina:
    """Stands in for the gnina binary: reads --ligand, writes --out with score
    properties, prints a gnina-style table. `behavior(mol, cmd)` returns
    'ok' | 'fail' | 'timeout' per molecule."""

    def __init__(self):
        self.commands = []
        self.behavior = lambda mol, cmd: "ok"

    def run(self, cmd, capture_output=True, text=True, timeout=None, **kw):
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "gnina v1.3.2 (fake)\n", "")
        if "--help" in cmd:
            return subprocess.CompletedProcess(cmd, 0, HELP_TEXT, "")
        if "--ligand" not in cmd:                      # some other tool (obabel -V, ...)
            return subprocess.CompletedProcess(cmd, 0, "fake-tool 1.0\n", "")
        self.commands.append(list(cmd))
        lig = cmd[cmd.index("--ligand") + 1]
        out = cmd[cmd.index("--out") + 1]
        mol = next(iter(Chem.SDMolSupplier(lig, removeHs=False)))
        mode = self.behavior(mol, cmd)
        if mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, timeout)
        if mode == "fail":
            return subprocess.CompletedProcess(cmd, 1, "", "ERROR: fake gnina failure")
        idx = int(mol.GetProp("MolIndex")) if mol.HasProp("MolIndex") and mol.GetProp("MolIndex").isdigit() else 0
        aff = -6.0 - 0.5 * idx
        cs = 0.9 - 0.1 * idx
        ca = 6.0 + 0.4 * idx
        none_mode = cmd[cmd.index("--cnn_scoring") + 1] == "none" if "--cnn_scoring" in cmd else False
        mol.SetProp("minimizedAffinity", f"{aff:.2f}")
        if not none_mode:
            mol.SetProp("CNNscore", f"{cs:.4f}")
            mol.SetProp("CNNaffinity", f"{ca:.3f}")
        w = Chem.SDWriter(out)
        w.write(mol)
        w.close()
        table = (TABLE_VINA.format(aff=aff) if none_mode
                 else TABLE_CNN.format(aff=aff, cs=cs, ca=ca))
        return subprocess.CompletedProcess(cmd, 0, "gnina fake\n" + table, "")


@pytest.fixture
def fake_gnina(monkeypatch):
    from gnina_vs import docking
    fake = FakeGnina()
    monkeypatch.setattr(docking.subprocess, "run", fake.run)
    monkeypatch.setattr(docking.shutil, "which", lambda name, *a, **k: "/usr/bin/" + str(name))
    return fake


@pytest.fixture
def raw_1a8o():
    return str(DATA / "1A8O.pdb")


def write_csv(path, rows):
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)
