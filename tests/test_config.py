"""
Phase 3 tests: gnina_vs.config.load_config -- YAML schema validation with
per-field ConfigError messages (no pydantic dependency).
"""
import textwrap

import pandas as pd
import pytest
from rdkit import Chem

from gnina_vs.config import ConfigError, load_config


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


MINIMAL_VALID = """\
outdir: results
targets:
  - id: T1
    pdb: 1ABC
    center: [1.0, 2.0, 3.0]
drugs:
  - id: drugA
    smiles: "CCO"
"""


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/no/such/config.yaml")


def test_minimal_valid_config(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL_VALID))
    assert cfg.outdir == "results" and cfg.seed == 42
    assert len(cfg.targets) == 1 and cfg.targets[0].center == (1.0, 2.0, 3.0)
    assert len(cfg.drugs) == 1 and cfg.drugs[0].targets == ["T1"]  # defaults to all targets


def test_top_level_must_be_mapping(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(str(p))


def test_seed_must_be_int(tmp_path):
    cfg = _write(tmp_path, MINIMAL_VALID.replace("outdir: results", "outdir: results\nseed: not-a-number"))
    with pytest.raises(ConfigError, match="seed"):
        load_config(cfg)


def test_missing_targets(tmp_path):
    cfg = _write(tmp_path, "outdir: results\ndrugs:\n  - id: d\n    smiles: CCO\n")
    with pytest.raises(ConfigError, match="targets"):
        load_config(cfg)


def test_missing_drugs(tmp_path):
    cfg = _write(tmp_path, "outdir: results\ntargets:\n  - id: T1\n    pdb: x\n    center: [1,2,3]\n")
    with pytest.raises(ConfigError, match="drugs"):
        load_config(cfg)


def test_target_missing_pdb_names_field_and_index(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            center: [1, 2, 3]
        drugs:
          - id: d
            smiles: CCO
    """)
    with pytest.raises(ConfigError, match=r"targets\[0\].*'pdb'"):
        load_config(cfg)


def test_duplicate_target_id(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
          - id: T1
            pdb: y
            center: [4, 5, 6]
        drugs:
          - id: d
            smiles: CCO
    """)
    with pytest.raises(ConfigError, match="duplicate target id"):
        load_config(cfg)


def test_invalid_role(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
            role: villain
            center: [1, 2, 3]
        drugs:
          - id: d
            smiles: CCO
    """)
    with pytest.raises(ConfigError, match="role"):
        load_config(cfg)


def test_center_from_ligand_requires_ligand_resname(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
            center_from_ligand: true
        drugs:
          - id: d
            smiles: CCO
    """)
    with pytest.raises(ConfigError, match="ligand_resname"):
        load_config(cfg)


def test_target_needs_center_or_center_from_ligand(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
        drugs:
          - id: d
            smiles: CCO
    """)
    with pytest.raises(ConfigError, match="center"):
        load_config(cfg)


def test_center_must_be_3_element_list(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
            center: [1, 2]
        drugs:
          - id: d
            smiles: CCO
    """)
    with pytest.raises(ConfigError, match="3-element"):
        load_config(cfg)


def test_reference_ligand_invalid_smiles(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
            reference_ligands:
              - name: bad
                smiles: not-a-smiles
        drugs:
          - id: d
            smiles: CCO
    """)
    with pytest.raises(ConfigError, match="invalid SMILES"):
        load_config(cfg)


def test_drug_invalid_smiles(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs:
          - id: d
            smiles: not-a-smiles
    """)
    with pytest.raises(ConfigError, match="invalid SMILES"):
        load_config(cfg)


def test_duplicate_drug_id(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs:
          - id: d
            smiles: CCO
          - id: d
            smiles: CCN
    """)
    with pytest.raises(ConfigError, match="(?i)duplicate drug id"):
        load_config(cfg)


def test_drug_references_undefined_target(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs:
          - id: d
            smiles: CCO
            targets: [T1, GHOST]
    """)
    with pytest.raises(ConfigError, match="GHOST"):
        load_config(cfg)


def test_invalid_cnn_scoring_in_docking_section(tmp_path):
    cfg = _write(tmp_path, MINIMAL_VALID + "docking:\n  cnn_scoring: bogus\n")
    with pytest.raises(ConfigError, match="cnn_scoring"):
        load_config(cfg)


def test_deprecated_cnn_scoring_alias_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL_VALID + "docking:\n  cnn_scoring: refine\n"))
    assert cfg.docking.cnn_scoring == "refinement"


def test_metabolism_steps_must_be_1_or_2(tmp_path):
    cfg = _write(tmp_path, MINIMAL_VALID + "metabolism:\n  steps: 5\n")
    with pytest.raises(ConfigError, match="steps"):
        load_config(cfg)


def test_drugs_file_csv_default_columns(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,Name,SMILES\nC1,aspirin,CC(=O)Oc1ccccc1C(=O)O\nC2,ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O\n")
    cfg = load_config(_write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs_file: {csv}
    """))
    assert {d.id for d in cfg.drugs} == {"C1", "C2"}
    assert all(d.targets == ["T1"] for d in cfg.drugs)


def test_drugs_file_xlsx_with_custom_columns(tmp_path):
    xlsx = tmp_path / "compounds.xlsx"
    pd.DataFrame({"CompoundID": ["X1"], "Structure": ["CCO"]}).to_excel(xlsx, index=False)
    cfg = load_config(_write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs_file:
          path: {xlsx}
          id_col: CompoundID
          smiles_col: Structure
    """))
    assert len(cfg.drugs) == 1 and cfg.drugs[0].id == "X1" and cfg.drugs[0].smiles == Chem.CanonSmiles("CCO")


def test_drugs_file_falls_back_to_name_then_row_number_for_id(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("Name,SMILES\nparacetamol,CC(=O)Nc1ccc(O)cc1\n,CCO\n")
    cfg = load_config(_write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs_file: {csv}
    """))
    ids = {d.id for d in cfg.drugs}
    assert "paracetamol" in ids
    assert any(i.startswith("compound_") for i in ids)  # second row has no Name either


def test_drugs_file_missing_path(tmp_path):
    cfg = _write(tmp_path, """\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs_file: /no/such/compounds.csv
    """)
    with pytest.raises(ConfigError, match="not found"):
        load_config(cfg)


def test_drugs_file_missing_smiles_column(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,Structure\nC1,CCO\n")
    cfg = _write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs_file: {csv}
    """)
    with pytest.raises(ConfigError, match="SMILES"):
        load_config(cfg)


def test_drugs_file_invalid_smiles_names_the_row(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,SMILES\nC1,CCO\nC2,not-a-smiles\n")
    cfg = _write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs_file: {csv}
    """)
    with pytest.raises(ConfigError, match=r"row 3.*invalid SMILES"):
        load_config(cfg)


def test_drugs_file_empty_smiles_cell(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,SMILES\nC1,\n")
    cfg = _write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs_file: {csv}
    """)
    with pytest.raises(ConfigError, match="empty"):
        load_config(cfg)


def test_drugs_file_custom_targets_subset(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,SMILES\nC1,CCO\n")
    cfg = load_config(_write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
          - id: T2
            pdb: y
            center: [4, 5, 6]
        drugs_file:
          path: {csv}
          targets: [T2]
    """))
    assert cfg.drugs[0].targets == ["T2"]


def test_drugs_file_unknown_target_rejected(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,SMILES\nC1,CCO\n")
    cfg = _write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs_file:
          path: {csv}
          targets: [GHOST]
    """)
    with pytest.raises(ConfigError, match="GHOST"):
        load_config(cfg)


def test_drugs_and_drugs_file_combine(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,SMILES\nC1,CCO\n")
    cfg = load_config(_write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs:
          - id: inline_drug
            smiles: CCN
        drugs_file: {csv}
    """))
    assert {d.id for d in cfg.drugs} == {"inline_drug", "C1"}


def test_drugs_and_drugs_file_duplicate_id_rejected(tmp_path):
    csv = tmp_path / "compounds.csv"
    csv.write_text("ID,SMILES\nsame_id,CCO\n")
    cfg = _write(tmp_path, f"""\
        targets:
          - id: T1
            pdb: x
            center: [1, 2, 3]
        drugs:
          - id: same_id
            smiles: CCN
        drugs_file: {csv}
    """)
    with pytest.raises(ConfigError, match="(?i)duplicate drug id"):
        load_config(cfg)


def test_full_config_with_all_sections(tmp_path):
    cfg = load_config(_write(tmp_path, """\
        outdir: my_results
        seed: 7
        metabolism:
          type: cyp450
          steps: 2
          include_gut: true
        docking:
          cnn_scoring: rescore
          exhaustiveness: 16
          workers: 8
        targets:
          - id: hERG
            role: antitarget
            pdb: 5VA2
            center_from_ligand: true
            ligand_resname: LIG
            chain: A
            keep_hetatm: [ZN]
            reference_ligands:
              - name: terfenadine
                smiles: "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1"
        drugs:
          - id: codeine
            smiles: "CC(=O)Oc1ccccc1C(=O)O"
            targets: [hERG]
    """))
    assert cfg.outdir == "my_results" and cfg.seed == 7
    assert cfg.metabolism.type == "cyp450" and cfg.metabolism.steps == 2 and cfg.metabolism.include_gut
    assert cfg.docking.exhaustiveness == 16 and cfg.docking.workers == 8
    t = cfg.targets[0]
    assert t.role == "antitarget" and t.chain == "A" and t.keep_hetatm == ["ZN"]
    assert t.reference_ligands[0].name == "terfenadine"
