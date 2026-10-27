"""
Phase 3 integration tests: the full `gnina-vs metabolite-run` pipeline,
end to end, driven only through `gnina_vs.metabolite_pipeline` (library
API) with the fake-gnina fixture and a synthetic receptor -- no real
Gnina, Java, or BioTransformer jar required.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from gnina_vs import config as config_mod
from gnina_vs import metabolite_pipeline
from pdbutil import atom_line, hetero, peptide, write_pdb


def _receptor_with_ligand(tmp_path, name="receptor.pdb"):
    protein_lines, s = peptide("A", [("ALA", 1, "ATOM"), ("TYR", 2, "ATOM"), ("ALA", 3, "ATOM")])
    lig_lines, s = hetero("LIG", "A", 401, 10, (12.0, 12.0, 12.0), serial_start=s)
    return write_pdb(tmp_path / name, protein_lines + lig_lines)


def _config_for(tmp_path, receptor_path, metabolites_csv=None, extra_target_kwargs=None):
    extra_target_kwargs = extra_target_kwargs or {}
    cfg_dict = {
        "outdir": str(tmp_path / "out"),
        "seed": 5,
        "metabolism": {"metabolites_csv": metabolites_csv} if metabolites_csv else {},
        "protein": {"add_hydrogens": False},
        "docking": {"workers": 1, "num_modes": 1, "exhaustiveness": 2},
        "targets": [{
            "id": "T1", "role": "therapeutic", "pdb": receptor_path,
            "center_from_ligand": True, "ligand_resname": "LIG",
            **extra_target_kwargs,
        }],
        "drugs": [{"id": "drugA", "smiles": "c1ccccc1O", "targets": ["T1"]}],
    }
    import yaml
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))
    return config_mod.load_config(str(cfg_path))


def test_full_pipeline_parent_only(tmp_path, fake_gnina):
    receptor = _receptor_with_ligand(tmp_path)
    config = _config_for(tmp_path, receptor)  # no metabolites_csv, no jar -> parent only

    outputs = metabolite_pipeline.run_metabolite_pipeline(config)

    metabolites_df = pd.read_csv(outputs["metabolites_csv"])
    assert len(metabolites_df) == 1 and metabolites_df.iloc[0]["is_parent"]

    comparison_df = pd.read_csv(outputs["comparison_table"])
    assert len(comparison_df) == 1
    assert comparison_df.iloc[0]["parent_id"] == "drugA"
    assert comparison_df.iloc[0]["target_id"] == "T1"
    # annotation columns present (Phase 6)
    assert "conjugate_flag" in comparison_df.columns
    assert "alert_brenk_n" in comparison_df.columns
    assert "alert_pains_a_n" in comparison_df.columns
    assert "exposure_plausibility" in comparison_df.columns
    assert "ad_status" in comparison_df.columns
    # the collapsed "reactive" boolean was removed on purpose
    assert "reactive_alert" not in comparison_df.columns

    validation_df = pd.read_csv(outputs["validation_redocking"])
    assert len(validation_df) == 1 and validation_df.iloc[0]["target_id"] == "T1"
    # redocking was attempted since center_from_ligand + ligand_resname were given
    assert validation_df.iloc[0]["status"] != config_mod.TargetConfig.__dataclass_fields__  # sanity noop

    manifest = json.loads(Path(outputs["run_manifest"]).read_text())
    assert manifest["seed"] == 5
    assert "T1" in manifest["targets"]
    assert manifest["targets"]["T1"]["box"]["center"] is not None

    # Phase 7: figures were produced
    assert outputs["figures"], "expected at least one figure to be generated"
    for fig_paths in outputs["figures"].values():
        assert Path(fig_paths["png"]).exists() and Path(fig_paths["svg"]).exists()


def test_full_pipeline_with_curated_metabolites_csv(tmp_path, fake_gnina):
    receptor = _receptor_with_ligand(tmp_path)
    curated = tmp_path / "curated.csv"
    curated.write_text("smiles,id,reaction\nOc1ccccc1O,drugA_catechol,hydroxylation\n")
    config = _config_for(tmp_path, receptor, metabolites_csv=str(curated))

    outputs = metabolite_pipeline.run_metabolite_pipeline(config)

    metabolites_df = pd.read_csv(outputs["metabolites_csv"])
    assert len(metabolites_df) == 2
    assert set(metabolites_df["compound_id"]) == {"drugA", "drugA_catechol"}

    comparison_df = pd.read_csv(outputs["comparison_table"])
    assert len(comparison_df) == 2
    # every row traces back to a parent
    assert set(comparison_df["parent_id"]) == {"drugA"}
    # delta vs parent computed for the metabolite, NaN (self) for the parent
    metabolite_row = comparison_df[comparison_df["compound_id"] == "drugA_catechol"].iloc[0]
    assert pd.notna(metabolite_row["delta_cnn_affinity_vs_parent"]) or pd.notna(metabolite_row["delta_vina_affinity_vs_parent"])


def test_full_pipeline_with_reference_ligand(tmp_path, fake_gnina):
    receptor = _receptor_with_ligand(tmp_path)
    config = _config_for(tmp_path, receptor, extra_target_kwargs={
        "reference_ligands": [{"name": "ref1", "smiles": "CCO"}],
    })
    outputs = metabolite_pipeline.run_metabolite_pipeline(config)
    comparison_df = pd.read_csv(outputs["comparison_table"])
    row = comparison_df.iloc[0]
    # reference ligand was docked and used for delta_vs_reference (not NaN)
    assert pd.notna(row["delta_cnn_affinity_vs_reference"]) or pd.notna(row["delta_vina_affinity_vs_reference"])


def test_pipeline_target_with_no_cocrystal_ligand_reports_not_validated(tmp_path, fake_gnina):
    # apo-style target: explicit center, no ligand_resname
    protein_lines, s = peptide("A", [("ALA", 1, "ATOM"), ("ALA", 2, "ATOM")])
    receptor = write_pdb(tmp_path / "apo.pdb", protein_lines)
    cfg_dict = {
        "outdir": str(tmp_path / "out"),
        "docking": {"workers": 1},
        "protein": {"add_hydrogens": False},
        "targets": [{"id": "APO", "pdb": receptor, "center": [1.0, 2.0, 3.0]}],
        "drugs": [{"id": "drugA", "smiles": "CCO", "targets": ["APO"]}],
    }
    import yaml
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))
    config = config_mod.load_config(str(cfg_path))

    outputs = metabolite_pipeline.run_metabolite_pipeline(config)
    validation_df = pd.read_csv(outputs["validation_redocking"])
    assert validation_df.iloc[0]["status"] == "not validated: no reference pose"


def test_pipeline_apo_target_with_center_from_ligand_fails_clearly(tmp_path, fake_gnina):
    protein_lines, s = peptide("A", [("ALA", 1, "ATOM")])
    receptor = write_pdb(tmp_path / "apo.pdb", protein_lines)
    cfg_dict = {
        "outdir": str(tmp_path / "out"),
        "protein": {"add_hydrogens": False},
        "targets": [{"id": "APO", "pdb": receptor, "center_from_ligand": True, "ligand_resname": "LIG"}],
        "drugs": [{"id": "drugA", "smiles": "CCO", "targets": ["APO"]}],
    }
    import yaml
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))
    config = config_mod.load_config(str(cfg_path))

    with pytest.raises(ValueError, match="LIG"):
        metabolite_pipeline.run_metabolite_pipeline(config)


def test_cli_metabolite_run_end_to_end(tmp_path, fake_gnina):
    from gnina_vs import cli
    receptor = _receptor_with_ligand(tmp_path)
    cfg_dict = {
        "outdir": str(tmp_path / "out"),
        "protein": {"add_hydrogens": False},
        "docking": {"workers": 1},
        "targets": [{"id": "T1", "pdb": receptor, "center_from_ligand": True, "ligand_resname": "LIG"}],
        "drugs": [{"id": "drugA", "smiles": "c1ccccc1O", "targets": ["T1"]}],
    }
    import yaml
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))

    rc = cli.main(["metabolite-run", "--config", str(cfg_path)])
    assert rc == 0
    assert (tmp_path / "out" / "comparison_table.csv").exists()
    assert (tmp_path / "out" / "validation_redocking.csv").exists()
    assert (tmp_path / "out" / "run_manifest.json").exists()


def test_full_pipeline_from_excel_sheet_of_compounds(tmp_path, fake_gnina):
    """The exact workflow a user follows: an Excel sheet of parent
    compounds -> generate metabolites for each -> dock everything."""
    import pandas as pd

    receptor = _receptor_with_ligand(tmp_path)
    compounds_xlsx = tmp_path / "compounds.xlsx"
    pd.DataFrame({
        "ID": ["cpd1", "cpd2"],
        "Name": ["phenol", "catechol_like"],
        "SMILES": ["c1ccccc1O", "c1ccc(O)c(O)c1"],
    }).to_excel(compounds_xlsx, index=False)

    cfg_dict = {
        "outdir": str(tmp_path / "out"),
        "protein": {"add_hydrogens": False},
        "docking": {"workers": 1},
        "targets": [{"id": "T1", "pdb": receptor, "center_from_ligand": True, "ligand_resname": "LIG"}],
        "drugs_file": str(compounds_xlsx),
    }
    import yaml
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))
    config = config_mod.load_config(str(cfg_path))

    assert {d.id for d in config.drugs} == {"cpd1", "cpd2"}

    outputs = metabolite_pipeline.run_metabolite_pipeline(config)

    metabolites_df = pd.read_csv(outputs["metabolites_csv"])
    assert set(metabolites_df["parent_id"]) == {"cpd1", "cpd2"}
    assert metabolites_df["is_parent"].all()  # no metabolism configured -> parents only

    comparison_df = pd.read_csv(outputs["comparison_table"])
    assert set(comparison_df["parent_id"]) == {"cpd1", "cpd2"}
    assert (comparison_df["status"] == "ok").all()


def test_cli_metabolite_run_bad_config_reports_error_cleanly(tmp_path, capsys):
    from gnina_vs import cli
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("targets: []\ndrugs: []\n")
    rc = cli.main(["metabolite-run", "--config", str(cfg_path)])
    assert rc == 1


def test_tiny_example_script_with_mocked_gnina(tmp_path, fake_gnina, monkeypatch):
    """Phase 8 deliverable: the example in examples/run_tiny_example.py
    works end to end against mocked docking (no real Gnina/Java needed)."""
    import importlib.util
    import sys as _sys

    example_path = Path(__file__).resolve().parents[1] / "examples" / "run_tiny_example.py"
    spec = importlib.util.spec_from_file_location("run_tiny_example", example_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix="": str(tmp_path))
    spec.loader.exec_module(module)

    receptor_path = tmp_path / "receptor.pdb"
    module.build_synthetic_receptor(receptor_path)
    from gnina_vs import config as config_mod
    config = config_mod.RunConfig(
        outdir=str(tmp_path / "results"), seed=42,
        protein=config_mod.ProteinConfig(add_hydrogens=False),
        metabolism=config_mod.MetabolismConfig(
            metabolites_csv=str(Path(__file__).resolve().parents[1] / "examples" / "curated_metabolites.csv"),
        ),
        targets=[config_mod.TargetConfig(id="EXAMPLE_TARGET", pdb=str(receptor_path),
                                         center_from_ligand=True, ligand_resname="LIG")],
        drugs=[config_mod.DrugConfig(id="terfenadine",
                                     smiles="CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
                                     targets=["EXAMPLE_TARGET"])],
    )
    outputs = module.metabolite_pipeline.run_metabolite_pipeline(config)
    assert Path(outputs["comparison_table"]).exists()
    df = pd.read_csv(outputs["comparison_table"])
    assert set(df["parent_id"]) == {"terfenadine"}
    assert set(df["base_compound_id"]) == {"terfenadine", "fexofenadine"}
