"""
gnina_vs.cli
==================
Command-line entry point. Exposes two subcommands:

    gnina-vs run [OPTIONS]        Run the full pipeline end-to-end.
    gnina-vs recover [OPTIONS]    Re-score/re-export from existing
                                   docked SDF files, without re-docking.

Equivalently: ``python -m gnina_vs run [OPTIONS]``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from gnina_vs import __version__, docking, ligand, pocket, protein, reporter

logger = logging.getLogger("gnina_vs")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _add_common_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--outdir", default="gnina_vs_output",
                    help="Directory for all pipeline outputs. Default: gnina_vs_output")
    p.add_argument("--top-n", default="100",
                    help="Number of top-ranked compounds to export. Use 0 or "
                         "'all' to export every scored compound. Default: 100")
    p.add_argument("--id-col", default="ID",
                    help="Identifier column name in the ligand input file. Default: ID")
    p.add_argument("--no-merged-sdf", action="store_true",
                    help="Skip writing a single ranked multi-molecule SDF of "
                         "the exported compounds' best poses.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gnina-vs",
        description="Virtual screening pipeline: protein prep, ligand prep, "
                    "active-site detection, Gnina docking, and results export.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- run ---- #
    run_p = subparsers.add_parser("run", help="Run the full pipeline end-to-end.")

    io_group = run_p.add_argument_group("Input files")
    io_group.add_argument("--protein", required=True,
                           help="Path to a receptor PDB file, OR a bare 4-character "
                                "PDB ID (e.g. '1ABC') to auto-fetch from RCSB.")
    io_group.add_argument("--ligands", required=True,
                           help="Path to a .xlsx/.xls/.csv file with a SMILES column.")
    io_group.add_argument("--smiles-col", default="SMILES", help="SMILES column name. Default: SMILES")
    io_group.add_argument("--name-col", default="Name", help="Compound name column name. Default: Name")

    prot_group = run_p.add_argument_group("Protein preparation")
    prot_group.add_argument("--keep-hetatm", default=None,
                             help="Comma-separated HETATM residue names to retain as "
                                  "cofactors (e.g. 'ZN' or 'ZN,MG'). Default: strip all HETATM.")
    prot_group.add_argument("--no-add-hydrogens", action="store_true",
                             help="Skip adding polar hydrogens via OpenBabel.")
    prot_group.add_argument("--ph", type=float, default=7.4,
                             help="Protonation pH passed to OpenBabel. Default: 7.4")

    pocket_group = run_p.add_argument_group("Active site detection")
    pocket_group.add_argument("--center-x", type=float, default=None)
    pocket_group.add_argument("--center-y", type=float, default=None)
    pocket_group.add_argument("--center-z", type=float, default=None)
    pocket_group.add_argument("--active-site-residues", default=None,
                               help='Comma-separated catalytic residues to center on '
                                    'instead of auto-detection, e.g. "TYR120,GLU150".')

    dock_group = run_p.add_argument_group("Gnina docking parameters")
    dock_group.add_argument("--gnina-exec", default="gnina", help="Path to the gnina executable. Default: gnina")
    dock_group.add_argument("--box-size", type=float, default=25.0, help="Docking box edge length (A). Default: 25.0")
    dock_group.add_argument("--cnn-scoring", choices=docking.CNN_SCORING_MODES, default="rescore",
                             help="Gnina CNN scoring mode. Default: rescore")
    dock_group.add_argument("--exhaustiveness", type=int, default=8, help="Search exhaustiveness. Default: 8")
    dock_group.add_argument("--num-modes", type=int, default=9, help="Binding modes per ligand. Default: 9")
    dock_group.add_argument("--workers", type=int, default=4, help="Parallel Gnina processes. Default: 4")
    dock_group.add_argument("--cpu-per-worker", type=int, default=2, help="--cpu passed to each Gnina process. Default: 2")
    dock_group.add_argument("--timeout", type=int, default=300, help="Per-ligand docking timeout (s). Default: 300")

    _add_common_output_args(run_p)
    run_p.set_defaults(func=run_command)

    # ---- recover ---- #
    rec_p = subparsers.add_parser(
        "recover", help="Re-score/re-export from existing docked SDF files, without re-docking.",
    )
    rec_p.add_argument("--ligands-sdf", required=True,
                        help="Path to the prepared ligands SDF written during a previous 'run' "
                             "(e.g. gnina_vs_output/prepared_ligands.sdf).")
    rec_p.add_argument("--docking-outdir", required=True,
                        help="Directory containing the '<mol_name>_docked.sdf' files from a "
                             "previous 'run' (e.g. gnina_vs_output/docking_outputs).")
    rec_p.add_argument("--input", default=None,
                        help="Original ligand Excel/CSV file, to recover the ID column. Optional.")
    _add_common_output_args(rec_p)
    rec_p.set_defaults(func=recover_command)

    return parser


# --------------------------------------------------------------------------- #
# Subcommand implementations
# --------------------------------------------------------------------------- #
def run_command(args: argparse.Namespace) -> None:
    logger.info("+" + "=" * 58 + "+")
    logger.info("|        VIRTUAL SCREENING PIPELINE  --  START            |")
    logger.info("+" + "=" * 58 + "+")

    os.makedirs(args.outdir, exist_ok=True)

    center_parts = (args.center_x, args.center_y, args.center_z)
    n_given = sum(p is not None for p in center_parts)
    if n_given not in (0, 3):
        raise ValueError(
            "--center-x, --center-y, and --center-z must all be given "
            "together, or not at all."
        )
    manual_center = center_parts if n_given == 3 else None

    keep_hetatm = None
    if args.keep_hetatm:
        keep_hetatm = {r.strip().upper() for r in args.keep_hetatm.split(",") if r.strip()}

    # -- Step 0: resolve + prepare protein --
    raw_pdb = protein.resolve_protein_source(args.protein, download_dir=args.outdir)
    prepared_receptor = protein.prepare_protein(
        input_pdb=raw_pdb,
        output_pdb=os.path.join(args.outdir, "receptor_prepared.pdb"),
        keep_hetatm_residues=keep_hetatm,
        add_hydrogens=not args.no_add_hydrogens,
        ph=args.ph,
    )

    # -- Step 1: prepare ligands --
    prepared_ligands_sdf = os.path.join(args.outdir, "prepared_ligands.sdf")
    ligand.prepare_ligands(
        input_path=args.ligands,
        output_sdf=prepared_ligands_sdf,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        name_col=args.name_col,
    )

    # -- Step 2: detect (or accept) active site --
    # CRITICAL: pass the ORIGINAL (raw) PDB as raw_pdb so that Tier 1 (and
    # residue-based manual overrides) can find the co-crystallised ligand /
    # full side chains, which were stripped from prepared_receptor.
    center = pocket.detect_active_site(
        pdb_path=prepared_receptor,
        raw_pdb=raw_pdb,
        center=manual_center,
        active_site_residues=args.active_site_residues,
        visualization_path=os.path.join(args.outdir, "docking_box.pml"),
        box_size=args.box_size,
    )

    # -- Step 3: run Gnina docking --
    docking_results = docking.run_gnina_docking(
        receptor_pdb=prepared_receptor,
        ligands_sdf=prepared_ligands_sdf,
        center=center,
        box_size=args.box_size,
        out_dir=os.path.join(args.outdir, "docking_outputs"),
        gnina_exec=args.gnina_exec,
        cnn_scoring=args.cnn_scoring,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        num_workers=args.workers,
        cpu_per_worker=args.cpu_per_worker,
        timeout=args.timeout,
    )

    # -- Step 4: extract & export results --
    merged_sdf = None if args.no_merged_sdf else os.path.join(args.outdir, "ranked_docked.sdf")
    reporter.extract_results(
        docking_results=docking_results,
        input_path=args.ligands,
        output_csv=os.path.join(args.outdir, "results_summary.csv"),
        output_excel=os.path.join(args.outdir, "top_results.xlsx"),
        top_n=args.top_n,
        merged_sdf=merged_sdf,
        id_col=args.id_col,
    )

    logger.info("+" + "=" * 58 + "+")
    logger.info("|        VIRTUAL SCREENING PIPELINE  --  DONE             |")
    logger.info("+" + "=" * 58 + "+")


def recover_command(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    merged_sdf = None if args.no_merged_sdf else os.path.join(args.outdir, "ranked_docked.sdf")
    reporter.recover_results(
        ligands_sdf=args.ligands_sdf,
        out_dir=args.docking_outdir,
        input_path=args.input,
        output_csv=os.path.join(args.outdir, "results_summary.csv"),
        output_excel=os.path.join(args.outdir, "top_results.xlsx"),
        top_n=args.top_n,
        merged_sdf=merged_sdf,
        id_col=args.id_col,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        args.func(args)
    except (ValueError, ConnectionError, ImportError, FileNotFoundError, RuntimeError) as e:
        # Expected, user-actionable errors raised deliberately throughout
        # the pipeline (bad input, missing executable, network failure,
        # detection failure, etc.): show the message only, not a full
        # traceback. Anything else (a genuine bug) still surfaces the full
        # traceback for debugging.
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
