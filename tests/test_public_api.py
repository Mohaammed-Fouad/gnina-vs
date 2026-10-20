"""
Phase 0: gnina-vs must be usable as a plain Python library, with cli.py as
a thin wrapper. These tests exercise the pipeline entirely through the
public API (no argparse, no CLI Namespace) and check the core step
functions never depend on one.
"""

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

import gnina_vs
from conftest import write_csv
from gnina_vs import (
    detect_active_site,
    docking,
    dock_ligands,
    extract_results,
    extract_scores,
    ligand,
    pocket,
    prepare_ligands,
    prepare_protein,
    protein,
    reporter,
    resolve_center,
)


# --------------------------------------------------------------------------- #
# Top-level exports
# --------------------------------------------------------------------------- #
def test_core_steps_importable_directly_from_package():
    # Every name the library API promises is importable straight from
    # `gnina_vs`, not only from its submodules.
    for name in ("prepare_protein", "prepare_ligands", "resolve_center",
                "dock_ligands", "extract_results", "extract_scores", "recover_results"):
        assert hasattr(gnina_vs, name), f"gnina_vs.{name} is not exported"
        assert callable(getattr(gnina_vs, name))


def test_resolve_center_is_detect_active_site():
    assert resolve_center is detect_active_site is pocket.detect_active_site


def test_dock_ligands_is_run_gnina_docking():
    assert dock_ligands is docking.run_gnina_docking


# --------------------------------------------------------------------------- #
# extract_scores: the score-parsing building block, usable on its own
# --------------------------------------------------------------------------- #
def test_extract_scores_from_log_only():
    log = (
        "mode |  affinity  |  intramol  |    CNN     |   CNN\n"
        "     | (kcal/mol) | (kcal/mol) | pose score | affinity\n"
        "-----+------------+------------+------------+----------\n"
        "    1       -8.32       -0.34       0.7574      6.438\n"
    )
    assert extract_scores(log) == {"vina_affinity": -8.32, "cnn_score": 0.7574, "cnn_affinity": 6.438}


def test_extract_scores_falls_back_to_sdf_when_log_is_empty(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=1)
    mol.SetProp("minimizedAffinity", "-7.1")
    mol.SetProp("CNNscore", "0.65")
    mol.SetProp("CNNaffinity", "5.9")
    sdf = str(tmp_path / "d.sdf")
    w = Chem.SDWriter(sdf); w.write(mol); w.close()

    assert extract_scores("", sdf) == {"vina_affinity": -7.1, "cnn_score": 0.65, "cnn_affinity": 5.9}


def test_extract_scores_missing_everything_returns_all_none():
    assert extract_scores("TIMEOUT", "") == {"vina_affinity": None, "cnn_score": None, "cnn_affinity": None}


def test_extract_scores_takes_plain_strings_not_a_result_dict():
    # Regression guard: extract_scores must not require a docking-result
    # dict (mol_name/status/...); a bare log string is enough.
    sig = inspect.signature(extract_scores)
    params = list(sig.parameters)
    assert params[:1] == ["raw_log"]
    for p in sig.parameters.values():
        assert p.annotation in (inspect._empty, str) or "dict" not in str(p.annotation).lower()


# --------------------------------------------------------------------------- #
# No argparse/CLI coupling in the core step functions
# --------------------------------------------------------------------------- #
CORE_FUNCTIONS = [
    prepare_protein,
    prepare_ligands,
    detect_active_site,
    dock_ligands,
    extract_results,
    extract_scores,
    reporter.recover_results,
]


@pytest.mark.parametrize("fn", CORE_FUNCTIONS, ids=lambda f: f.__name__)
def test_core_function_has_no_namespace_or_argparse_parameter(fn):
    sig = inspect.signature(fn)
    for name, param in sig.parameters.items():
        assert name not in ("args", "namespace", "opts", "options"), (
            f"{fn.__name__} takes a parameter named '{name}' -- looks like it "
            f"expects a CLI Namespace instead of explicit arguments."
        )
        ann = str(param.annotation)
        assert "Namespace" not in ann, f"{fn.__name__}.{name} is typed as an argparse Namespace"


def test_cli_module_does_not_pass_namespace_into_core_functions():
    # cli.py's job is to unpack `args.*` into explicit keyword arguments;
    # it must never hand a whole Namespace to a pipeline function.
    import gnina_vs.cli as cli_mod
    src = inspect.getsource(cli_mod)
    for call in ("prepare_protein(", "prepare_ligands(", "detect_active_site(",
                "run_gnina_docking(", "extract_results(", "recover_results("):
        start = src.index(call)
        # the argument list up to the matching close paren shouldn't be
        # just a single bare "args" (i.e. no dotted/keyword access at all)
        depth = 0
        arglist = ""
        for ch in src[start + len(call):]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            arglist += ch
        assert arglist.strip() != "args", f"{call.rstrip('(')} is called with a bare Namespace"


# --------------------------------------------------------------------------- #
# End-to-end: drive the whole pipeline through the library API only,
# with no argparse.Namespace / cli.main() involved anywhere.
# --------------------------------------------------------------------------- #
def test_full_pipeline_via_library_api_only(tmp_path, fake_gnina, raw_1a8o):
    csv = write_csv(tmp_path / "c.csv", [
        {"ID": "A", "Name": "dup", "SMILES": "CCO"},
        {"ID": "B", "Name": "dup", "SMILES": "CC(N)C(=O)O.Cl"},
    ])

    prepared_receptor = prepare_protein(
        input_pdb=raw_1a8o,
        output_pdb=str(tmp_path / "receptor_prepared.pdb"),
        add_hydrogens=False,
    )
    prepared_ligands_sdf = str(tmp_path / "ligands.sdf")
    prepare_ligands(input_path=csv, output_sdf=prepared_ligands_sdf, seed=11)

    center = resolve_center(
        pdb_path=prepared_receptor, raw_pdb=raw_1a8o,
        center=(18.7, 35.7, 16.0),
    )
    assert center == (18.7, 35.7, 16.0)

    results = dock_ligands(
        receptor_pdb=prepared_receptor, ligands_sdf=prepared_ligands_sdf,
        center=center, out_dir=tmp_path / "docking_outputs", num_workers=2, seed=7,
    )
    assert len(results) == 2 and all(r["status"] == docking.STATUS_OK for r in results)

    df = extract_results(
        docking_results=results, input_path=csv,
        output_csv=str(tmp_path / "results.csv"), output_excel=str(tmp_path / "top.xlsx"),
        top_n="all",
    )
    assert len(df) == 2
    xl = pd.read_excel(tmp_path / "top.xlsx")
    assert len(xl) == 2 and set(xl["ID"]) == {"A", "B"}

    manifest = {"seed": 7, "center": list(center)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert json.loads((tmp_path / "manifest.json").read_text())["seed"] == 7
