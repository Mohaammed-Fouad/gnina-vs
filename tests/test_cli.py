"""Phase A items 1, 2, 12, 13, 18 at CLI level + an end-to-end run with a fake Gnina."""
import json
import logging

import pandas as pd
import pytest

from conftest import write_csv
from gnina_vs import cli, docking, ligand, protein, reporter

BASE = ["run", "--protein", "x.pdb", "--ligands", "y.csv"]


def test_cnn_scoring_alias_accepted_at_parse_time(caplog):
    with caplog.at_level(logging.WARNING):
        args = cli.build_arg_parser().parse_args(BASE + ["--cnn-scoring", "refine"])
    assert args.cnn_scoring == "refinement" and "deprecated" in caplog.text.lower()
    assert cli.build_arg_parser().parse_args(BASE + ["--cnn-scoring", "full"]).cnn_scoring == "all"


def test_cnn_scoring_invalid_rejected_at_parse_time(capsys):
    with pytest.raises(SystemExit) as e:
        cli.build_arg_parser().parse_args(BASE + ["--cnn-scoring", "bogus"])
    assert e.value.code == 2
    assert "refinement" in capsys.readouterr().err          # error message lists valid values


def test_cnn_scoring_valid_values_match_gnina_docs():
    assert set(docking.CNN_SCORING_MODES) == {"none", "rescore", "refinement", "metrorescore",
                                             "metrorefine", "all"}


def test_seed_default_and_override():
    p = cli.build_arg_parser()
    assert p.parse_args(BASE).seed == 42
    assert p.parse_args(BASE + ["--seed", "5"]).seed == 5


def test_ligand_selection_flags():
    a = cli.build_arg_parser().parse_args(BASE + ["--ligand-resname", "ATP", "--ligand-chain", "B"])
    assert (a.ligand_resname, a.ligand_chain) == ("ATP", "B")


def test_top_n_invalid_rejected_before_anything_runs(capsys):
    with pytest.raises(SystemExit):
        cli.build_arg_parser().parse_args(BASE + ["--top-n", "abc"])


def test_run_command_validates_top_n_before_docking(tmp_path, monkeypatch):
    args = cli.build_arg_parser().parse_args(BASE)
    args.top_n = "abc"                                     # e.g. a Namespace built by other code
    args.outdir = str(tmp_path)
    monkeypatch.setattr(protein, "resolve_protein_source",
                        lambda *a, **k: pytest.fail("pipeline started before --top-n validation"))
    with pytest.raises(ValueError, match="top-n"):
        cli.run_command(args)


# ---- end-to-end with fake Gnina -----------------------------------------------
def _run(tmp_path, raw_1a8o, extra, monkeypatch=None):
    csv = write_csv(tmp_path / "c.csv", [{"ID": "A", "Name": "dup", "SMILES": "CCO"},
                                         {"ID": "B", "Name": "dup", "SMILES": "CC(N)C(=O)O.Cl"},
                                         {"ID": "C", "Name": "c3", "SMILES": "c1ccccc1O"}])
    out = tmp_path / "out"
    rc = cli.main(["run", "--protein", raw_1a8o, "--ligands", csv, "--outdir", str(out),
                   "--no-add-hydrogens", "--center-x", "18.7", "--center-y", "35.7", "--center-z", "16.0",
                   "--workers", "2", "--top-n", "all"] + extra)
    return rc, out


def test_end_to_end_default_run(tmp_path, fake_gnina, raw_1a8o):
    rc, out = _run(tmp_path, raw_1a8o, ["--cnn-scoring", "refine", "--seed", "123"])
    assert rc == 0
    cmd = fake_gnina.commands[0]
    assert cmd[cmd.index("--cnn_scoring") + 1] == "refinement"
    assert cmd[cmd.index("--seed") + 1] == "123"
    xl = pd.read_excel(out / "top_results.xlsx")
    assert len(xl) == 3 and {"cnn_score", "pose_quality_flag", "status"} <= set(xl.columns)
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["seed"] == 123 and man["box"]["center"] == [18.7, 35.7, 16.0]
    assert man["command"] == "run" and "versions" in man


def test_end_to_end_cnn_none_is_not_empty(tmp_path, fake_gnina, raw_1a8o):
    rc, out = _run(tmp_path, raw_1a8o, ["--cnn-scoring", "none"])
    assert rc == 0
    assert len(pd.read_excel(out / "top_results.xlsx")) == 3


def test_end_to_end_all_jobs_fail_exit_code_1(tmp_path, fake_gnina, raw_1a8o):
    fake_gnina.behavior = lambda mol, cmd: "fail"
    rc, out = _run(tmp_path, raw_1a8o, [])
    assert rc == 1
    df = pd.read_csv(out / "results_summary.csv")
    assert set(df["status"]) == {"failed"}                 # failures reported, not hidden


# --------------------------------------------------------------------------- #
# `run --metabolite`: generate metabolites for every row in --ligands, dock
# parents + metabolites together through the normal `run` pipeline.
# --------------------------------------------------------------------------- #
def test_metabolite_flag_parses_with_defaults():
    args = cli.build_arg_parser().parse_args(BASE)
    assert args.metabolite is False
    assert args.biotransformer_jar is None
    assert args.metabolite_type == "allHuman"
    assert args.metabolite_steps == 1
    assert args.metabolite_include_gut is False
    assert args.metabolites_csv is None


def test_metabolite_flag_invalid_type_rejected():
    with pytest.raises(SystemExit):
        cli.build_arg_parser().parse_args(BASE + ["--metabolite", "--metabolite-type", "bogus"])


def test_metabolite_flag_invalid_steps_rejected():
    with pytest.raises(SystemExit):
        cli.build_arg_parser().parse_args(BASE + ["--metabolite", "--metabolite-steps", "3"])


def _run_metabolite(tmp_path, raw_1a8o, rows, extra=None, monkeypatch=None):
    csv = write_csv(tmp_path / "c.csv", rows)
    out = tmp_path / "out"
    rc = cli.main(["run", "--protein", raw_1a8o, "--ligands", csv, "--outdir", str(out),
                   "--no-add-hydrogens", "--center-x", "18.7", "--center-y", "35.7", "--center-z", "16.0",
                   "--workers", "2", "--top-n", "all", "--metabolite"] + (extra or []))
    return rc, out


def test_metabolite_flag_without_jar_or_csv_docks_parents_only(tmp_path, fake_gnina, raw_1a8o, caplog):
    rc, out = _run_metabolite(tmp_path, raw_1a8o, [
        {"ID": "A", "Name": "phenol", "SMILES": "c1ccccc1O"},
        {"ID": "B", "Name": "ethanol", "SMILES": "CCO"},
    ])
    assert rc == 0
    metabolites_df = pd.read_csv(out / "metabolites.csv")
    assert set(metabolites_df["parent_id"]) == {"A", "B"}
    assert metabolites_df["is_parent"].all()  # no jar/csv -> parents only

    comparison_df = pd.read_csv(out / "comparison_table.csv")
    assert set(comparison_df["parent_id"]) == {"A", "B"}
    assert "conjugate_flag" in comparison_df.columns  # Phase 6 annotations present
    assert "delta_cnn_affinity_vs_parent" in comparison_df.columns  # Phase 5 comparison present

    man = json.loads((out / "run_manifest.json").read_text())
    assert man["metabolite"] is True


def test_metabolite_flag_with_curated_csv_scopes_per_compound(tmp_path, fake_gnina, raw_1a8o):
    curated = tmp_path / "curated.csv"
    curated.write_text(
        "parent_id,smiles,id\n"
        "A,Oc1ccccc1O,A_catechol\n"   # only for compound A
    )
    rc, out = _run_metabolite(tmp_path, raw_1a8o, [
        {"ID": "A", "Name": "phenol", "SMILES": "c1ccccc1O"},
        {"ID": "B", "Name": "ethanol", "SMILES": "CCO"},
    ], extra=["--metabolites-csv", str(curated)])
    assert rc == 0

    metabolites_df = pd.read_csv(out / "metabolites.csv")
    a_rows = metabolites_df[metabolites_df["parent_id"] == "A"]
    b_rows = metabolites_df[metabolites_df["parent_id"] == "B"]
    assert "A_catechol" in set(a_rows["compound_id"])
    assert "A_catechol" not in set(b_rows["compound_id"])
    assert len(b_rows) == 1  # B has no curated metabolite -- parent only


def test_metabolite_flag_off_by_default_no_metabolite_outputs(tmp_path, fake_gnina, raw_1a8o):
    rc, out = _run(tmp_path, raw_1a8o, [])  # the ordinary (non-metabolite) end-to-end test helper
    assert rc == 0
    assert not (out / "metabolites.csv").exists()
    assert not (out / "comparison_table.csv").exists()
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["metabolite"] is False
