"""
gnina_vs
===========
A modular virtual-screening pipeline: protein preparation (with RCSB
auto-fetch), ligand preparation, active-site detection (explicit
residue/ligand selection plus two automatic tiers), parallel Gnina
docking, and results reporting/export.

See ``gnina_vs.cli`` for the command-line entry point, which is a thin
wrapper around the library functions below -- it just turns parsed CLI
flags into keyword arguments and calls them.

Library API
-----------
Every pipeline step is a plain function taking explicit arguments (paths,
numbers, strings, tuples) and returning plain data (paths, dicts,
DataFrames, tuples) -- none of them take or depend on an
``argparse.Namespace`` or any other CLI/global state, so they can be
imported and called directly, e.g.::

    from gnina_vs import prepare_protein, prepare_ligands, resolve_center, dock_ligands, extract_results

    prepared = prepare_protein("receptor.pdb", "receptor_prepared.pdb")
    prepare_ligands("compounds.csv", "ligands.sdf")
    center = resolve_center("receptor_prepared.pdb", raw_pdb="receptor.pdb")
    results = dock_ligands("receptor_prepared.pdb", "ligands.sdf", center)
    extract_results(results, output_csv="results.csv", output_excel="top.xlsx")

For finer-grained access (e.g. the individual active-site detection tiers,
or SDF-writing helpers), import the step modules directly: ``protein``,
``ligand``, ``pocket``, ``docking``, ``reporter``.
"""

__version__ = "1.1.0"

from gnina_vs.protein import prepare_protein, resolve_protein_source
from gnina_vs.ligand import prepare_ligands
from gnina_vs.pocket import detect_active_site, resolve_center
from gnina_vs.docking import dock_ligands, run_gnina_docking
from gnina_vs.reporter import extract_results, extract_scores, recover_results

__all__ = [
    "__version__",
    "prepare_protein",
    "resolve_protein_source",
    "prepare_ligands",
    "resolve_center",
    "detect_active_site",
    "dock_ligands",
    "run_gnina_docking",
    "extract_results",
    "extract_scores",
    "recover_results",
]
