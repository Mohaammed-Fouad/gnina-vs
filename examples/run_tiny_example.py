#!/usr/bin/env python
"""
A tiny, fully self-contained example run of the metabolite-aware pipeline
for one drug against one target, using the curated metabolites bypass (no
BioTransformer jar needed) and a synthetic receptor with a co-crystal
ligand (no network access / no real PDB fetch needed).

Gnina itself is NOT mocked here: if `gnina` isn't on PATH, this will fail
at the docking step with the same clear error `gnina-vs run` gives. To
run this against mocked docking too (no Gnina needed at all), see how
tests/test_metabolite_pipeline.py drives the same pipeline with the
`fake_gnina` pytest fixture instead.

Usage:
    python examples/run_tiny_example.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))  # for pdbutil
from pdbutil import hetero, peptide, write_pdb  # noqa: E402

from gnina_vs import config as config_mod  # noqa: E402
from gnina_vs import metabolite_pipeline  # noqa: E402


def build_synthetic_receptor(path: Path) -> str:
    """A minimal synthetic receptor with one co-crystal ligand (LIG), so
    the example needs no network access or real PDB file."""
    protein_lines, s = peptide("A", [("ALA", 1, "ATOM"), ("TYR", 2, "ATOM"), ("ALA", 3, "ATOM")])
    ligand_lines, s = hetero("LIG", "A", 401, 10, center=(12.0, 12.0, 12.0), serial_start=s)
    return write_pdb(path, protein_lines + ligand_lines)


def main():
    workdir = Path(tempfile.mkdtemp(prefix="gnina_vs_example_"))
    receptor_path = build_synthetic_receptor(workdir / "receptor.pdb")
    curated_csv = Path(__file__).parent / "curated_metabolites.csv"

    config = config_mod.RunConfig(
        outdir=str(workdir / "results"),
        seed=42,
        metabolism=config_mod.MetabolismConfig(metabolites_csv=str(curated_csv)),
        targets=[config_mod.TargetConfig(
            id="EXAMPLE_TARGET", pdb=receptor_path,
            center_from_ligand=True, ligand_resname="LIG",
        )],
        drugs=[config_mod.DrugConfig(
            id="terfenadine",
            smiles="CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
            targets=["EXAMPLE_TARGET"],
        )],
    )

    print(f"Running the tiny example in {workdir} ...")
    outputs = metabolite_pipeline.run_metabolite_pipeline(config)
    print("Done. Key outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
