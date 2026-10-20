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
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone

from gnina_vs import __version__, docking, ligand, pocket, protein, reporter

logger = logging.getLogger("gnina_vs")


# --------------------------------------------------------------------------- #
# argparse type helpers
# --------------------------------------------------------------------------- #
def _cnn_scoring_type(value: str) -> str:
    """Validate --cnn-scoring at parse time and map deprecated aliases
    (refine -> refinement, full -> all) with a warning."""
    try:
        return docking.normalize_cnn_scoring(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def _top_n_type(value) -> int:
    """Validate --top-n at parse time, before anything else runs."""
    try:
        return reporter.parse_top_n(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _add_common_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--outdir", default="gnina_vs_output",
                    help="Directory for all pipeline outputs. Default: gnina_vs_output")
    p.add_argument("--top-n", default="100", type=_top_n_type,
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
                                  "cofactors (e.g. 'ZN' or 'ZN,MG'). Default: strip all HETATM "
                                  "(selenomethionine and phosphorylated Ser/Thr/Tyr are always "
                                  "kept -- see the README).")
    prot_group.add_argument("--no-add-hydrogens", action="store_true",
                             help="Skip adding hydrogens via OpenBabel.")
    prot_group.add_argument("--ph", type=float, default=7.4,
                             help="Protonation pH passed to OpenBabel. Default: 7.4")

    pocket_group = run_p.add_argument_group("Active site detection")
    pocket_group.add_argument("--center-x", type=float, default=None)
    pocket_group.add_argument("--center-y", type=float, default=None)
    pocket_group.add_argument("--center-z", type=float, default=None)
    pocket_group.add_argument("--active-site-residues", default=None,
                               help='Comma-separated catalytic residues to center on '
                                    'instead of auto-detection, e.g. "TYR120,GLU150" or, '
                                    'for a multi-chain structure, "A:TYR120,B:GLU150".')
    pocket_group.add_argument("--ligand-resname", default=None,
                               help="Exact HETATM residue name to center on (e.g. a "
                                    "cofactor like 'ATP' that automatic detection would "
                                    "otherwise skip), instead of auto-detection.")
    pocket_group.add_argument("--ligand-chain", default=None,
                               help="Chain to disambiguate --ligand-resname when the "
                                    "same residue name appears more than once.")

    dock_group = run_p.add_argument_group("Gnina docking parameters")
    dock_group.add_argument("--gnina-exec", default="gnina", help="Path to the gnina executable. Default: gnina")
    dock_group.add_argument("--box-size", type=float, default=25.0, help="Docking box edge length (A). Default: 25.0")
    dock_group.add_argument(
        "--cnn-scoring", type=_cnn_scoring_type, default="rescore",
        help=f"Gnina CNN scoring mode. One of: {', '.join(docking.CNN_SCORING_MODES)} "
             f"(deprecated aliases 'refine' -> 'refinement', 'full' -> 'all' still "
             f"accepted, with a warning). Default: rescore",
    )
    dock_group.add_argument("--exhaustiveness", type=int, default=8, help="Search exhaustiveness. Default: 8")
    dock_group.add_argument("--num-modes", type=int, default=9, help="Binding modes per ligand. Default: 9")
    dock_group.add_argument("--workers", type=int, default=4, help="Parallel Gnina processes. Default: 4")
    dock_group.add_argument("--cpu-per-worker", type=int, default=2, help="--cpu passed to each Gnina process. Default: 2")
    dock_group.add_argument("--timeout", type=int, default=300, help="Per-ligand docking timeout (s). Default: 300")
    dock_group.add_argument("--seed", type=int, default=42,
                             help="Random seed passed to Gnina (--seed) and used for RDKit "
                                  "conformer embedding, for reproducible runs. Default: 42")
    dock_group.add_argument("--autobox-ligand", default=None,
                             help="Reference ligand file: build the box around it "
                                  "(Gnina --autobox_ligand) instead of --center-x/y/z "
                                  "+ --box-size.")
    dock_group.add_argument("--autobox-add", type=float, default=4.0,
                             help="Buffer (A) added around --autobox-ligand. Default: 4.0")

    meta_group = run_p.add_argument_group("Metabolite generation (optional)")
    meta_group.add_argument(
        "--metabolite", action="store_true",
        help="Generate metabolites for every compound in --ligands first, then dock "
             "the parents AND their metabolites together. Writes metabolites.csv and "
             "a comparison_table.csv (deltas vs parent, ligand efficiency, RDKit/Brenk/"
             "PAINS annotations) alongside the usual outputs. Needs neither a "
             "BioTransformer jar nor --metabolites-csv to be set -- without either, "
             "it still runs the more careful metabolite-aware ligand preparation "
             "(salt stripping, stereoisomer enumeration, pH 7.4 protonation) on just "
             "the parents. For multi-target runs, validation/redocking, and figures, "
             "use `gnina-vs metabolite-run --config config.yaml` instead.")
    meta_group.add_argument("--biotransformer-jar", default=None,
                             help="Path to a BioTransformer jar (needs Java). Omit to skip "
                                  "automatic metabolite prediction (use --metabolites-csv "
                                  "instead, or dock only the parents).")
    meta_group.add_argument("--metabolite-type", default="allHuman",
                             choices=["allHuman", "cyp450", "phaseII", "hgut", "envmicro"],
                             help="BioTransformer predictor to use. Default: allHuman")
    meta_group.add_argument("--metabolite-steps", type=int, default=1, choices=[1, 2],
                             help="Rounds of further metabolism to predict. Default: 1")
    meta_group.add_argument("--metabolite-include-gut", action="store_true",
                             help="Include gut-microbial metabolism in the BioTransformer prediction.")
    meta_group.add_argument("--metabolites-csv", default=None,
                             help="Curated metabolites CSV (needs a 'smiles' column; add a "
                                  "'parent_id' column, matching each compound's --id-col value, "
                                  "when --ligands has more than one row, or every curated "
                                  "metabolite is attached to every compound).")

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
                        help="Directory containing the '<compound_id>_docked.sdf' files from a "
                             "previous 'run' (e.g. gnina_vs_output/docking_outputs).")
    rec_p.add_argument("--input", default=None,
                        help="Original ligand Excel/CSV file, to recover the ID column. Optional.")
    _add_common_output_args(rec_p)
    rec_p.set_defaults(func=recover_command)

    # ---- metabolite-run ---- #
    meta_p = subparsers.add_parser(
        "metabolite-run",
        help="Generate metabolites for one or more drugs and dock parent + "
             "metabolites against one or more targets (see Phases 1-8).",
    )
    meta_p.add_argument("--config", required=True,
                         help="Path to the metabolite-run YAML config file.")
    meta_p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    meta_p.set_defaults(func=metabolite_run_command)

    return parser


# --------------------------------------------------------------------------- #
# Run manifest
# --------------------------------------------------------------------------- #
def _package_versions(gnina_exec: str | None = None) -> dict:
    versions = {
        "gnina_vs": __version__,
        "python": platform.python_version(),
    }
    try:
        import rdkit
        versions["rdkit"] = rdkit.__version__
    except Exception:
        versions["rdkit"] = None
    if gnina_exec:
        versions["gnina"] = docking.get_gnina_version(gnina_exec)
    return versions


def _write_manifest(outdir: str, manifest: dict) -> str:
    path = os.path.join(outdir, "run_manifest.json")
    manifest.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    logger.info("  Run manifest written to '%s'.", path)
    return path


# --------------------------------------------------------------------------- #
# Subcommand implementations
# --------------------------------------------------------------------------- #
def run_command(args: argparse.Namespace) -> None:
    # Validate --top-n before anything else runs (even if this Namespace
    # wasn't built by argparse, e.g. constructed programmatically): a typo
    # here should not be discovered only after docking has finished.
    args.top_n = reporter.parse_top_n(args.top_n)

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
    prepared_df = None  # only set when --metabolite is used; drives Step 4's comparison table
    if args.metabolite:
        from gnina_vs import metabolism  # local import: keeps `run` free of any extra
        logger.info("=" * 60)
        logger.info("STEP 1a -- Metabolite Generation (--metabolite)")
        logger.info("=" * 60)
        metabolites_df = metabolism.generate_metabolites_for_table(
            input_path=args.ligands, smiles_col=args.smiles_col,
            id_col=args.id_col, name_col=args.name_col,
            jar_path=args.biotransformer_jar, btype=args.metabolite_type,
            steps=args.metabolite_steps, include_gut=args.metabolite_include_gut,
            metabolites_csv=args.metabolites_csv,
            cache_dir=os.path.join(args.outdir, ".biotransformer_cache"),
        )
        metabolites_csv_path = os.path.join(args.outdir, "metabolites.csv")
        metabolites_df.to_csv(metabolites_csv_path, index=False)
        logger.info("  %d compound(s) (parents + metabolites) -> '%s'.",
                   len(metabolites_df), metabolites_csv_path)

        prepared_df, failures_df = ligand.prepare_ligands_extended(
            metabolites_df, prepared_ligands_sdf, seed=args.seed,
        )
        if not failures_df.empty:
            failures_path = os.path.join(args.outdir, "ligand_prep_failures.csv")
            failures_df.to_csv(failures_path, index=False)
            logger.warning("  %d compound(s) failed ligand preparation -- see '%s'.",
                           len(failures_df), failures_path)
    else:
        ligand.prepare_ligands(
            input_path=args.ligands,
            output_sdf=prepared_ligands_sdf,
            smiles_col=args.smiles_col,
            id_col=args.id_col,
            name_col=args.name_col,
            seed=args.seed,
        )

    # -- Step 2: detect (or accept) active site --
    # CRITICAL: pass the ORIGINAL (raw) PDB as raw_pdb so that Tier 1 (and
    # residue-/ligand-based manual overrides) can find the co-crystallised
    # ligand / full side chains, which were stripped from prepared_receptor.
    center = pocket.detect_active_site(
        pdb_path=prepared_receptor,
        raw_pdb=raw_pdb,
        center=manual_center,
        active_site_residues=args.active_site_residues,
        ligand_resname=args.ligand_resname,
        ligand_chain=args.ligand_chain,
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
        seed=args.seed,
        autobox_ligand=args.autobox_ligand,
        autobox_add=args.autobox_add,
    )

    # -- Step 4: extract & export results --
    merged_sdf = None if args.no_merged_sdf else os.path.join(args.outdir, "ranked_docked.sdf")
    # Without --metabolite, 'ID' is recovered from args.ligands by row index
    # (unchanged behavior). With it, compound_id IS the id (already carried
    # via each SDF entry's LigandID property from prepare_ligands_extended),
    # since compounds no longer correspond 1:1 to original spreadsheet rows.
    results_df = reporter.extract_results(
        docking_results=docking_results,
        input_path=None if args.metabolite else args.ligands,
        output_csv=os.path.join(args.outdir, "results_summary.csv"),
        output_excel=os.path.join(args.outdir, "top_results.xlsx"),
        top_n=args.top_n,
        merged_sdf=merged_sdf,
        id_col=args.id_col,
    )

    if args.metabolite:
        from gnina_vs import annotate, compare
        logger.info("=" * 60)
        logger.info("STEP 4a -- Metabolite Comparison (--metabolite)")
        logger.info("=" * 60)
        target_label = os.path.splitext(os.path.basename(args.protein))[0]
        comparison_df = compare.build_comparison_table(results_df, prepared_df, target_id=target_label)
        comparison_df = annotate.add_annotations(comparison_df, smiles_col="smiles")
        comparison_path = os.path.join(args.outdir, "comparison_table.csv")
        comparison_df.to_csv(comparison_path, index=False)
        logger.info("  Comparison table (deltas vs parent, ligand efficiency, annotations) -> '%s'.",
                   comparison_path)

    _write_manifest(args.outdir, {
        "command": "run",
        "seed": args.seed,
        "protein": args.protein,
        "ligands": args.ligands,
        "box": {"center": list(center), "size": args.box_size},
        "cnn_scoring": args.cnn_scoring,
        "exhaustiveness": args.exhaustiveness,
        "num_modes": args.num_modes,
        "metabolite": args.metabolite,
        **({"metabolite_type": args.metabolite_type, "metabolite_steps": args.metabolite_steps,
            "metabolite_include_gut": args.metabolite_include_gut,
            "biotransformer_jar": args.biotransformer_jar,
            "metabolites_csv": args.metabolites_csv} if args.metabolite else {}),
        "versions": _package_versions(args.gnina_exec),
    })

    n_ok = sum(r.get("status", docking.STATUS_OK) == docking.STATUS_OK for r in docking_results)
    if n_ok == 0:
        raise RuntimeError("All docking jobs failed. See results_summary.csv and gnina_debug.txt.")

    logger.info("+" + "=" * 58 + "+")
    logger.info("|        VIRTUAL SCREENING PIPELINE  --  DONE             |")
    logger.info("+" + "=" * 58 + "+")


def recover_command(args: argparse.Namespace) -> None:
    args.top_n = reporter.parse_top_n(args.top_n)
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
    _write_manifest(args.outdir, {
        "command": "recover",
        "ligands_sdf": args.ligands_sdf,
        "docking_outdir": args.docking_outdir,
        "versions": _package_versions(),
    })


def metabolite_run_command(args: argparse.Namespace) -> None:
    """Thin wrapper: load + validate the YAML config, then call the
    library-level orchestration in `gnina_vs.metabolite_pipeline`.

    Imports are local to this function so that `run`/`recover` (and the
    base package) never require PyYAML, matplotlib, or Dimorphite-DL --
    those are only needed once `metabolite-run` is actually invoked (see
    the ``[metabolism]``/``[figures]`` extras in pyproject.toml).
    """
    try:
        from gnina_vs import config as config_mod
        from gnina_vs import metabolite_pipeline
    except ImportError as e:
        raise ImportError(
            f"'gnina-vs metabolite-run' needs its extra dependencies: {e}. "
            f"Install them with `pip install gnina-vs[metabolism,figures]`."
        )
    config = config_mod.load_config(args.config)
    logger.info("+" + "=" * 58 + "+")
    logger.info("|        METABOLITE-AWARE DOCKING PIPELINE  --  START      |")
    logger.info("+" + "=" * 58 + "+")
    outputs = metabolite_pipeline.run_metabolite_pipeline(config)
    logger.info("+" + "=" * 58 + "+")
    logger.info("|        METABOLITE-AWARE DOCKING PIPELINE  --  DONE       |")
    logger.info("+" + "=" * 58 + "+")
    for name, path in outputs.items():
        if name == "figures":
            for fig_name, fig_paths in path.items():
                logger.info("  figure %s: %s", fig_name, fig_paths)
        else:
            logger.info("  %s: %s", name, path)


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
