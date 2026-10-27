"""
gnina_vs.metabolite_pipeline
=================================
Phase 3 orchestration for ``gnina-vs metabolite-run --config config.yaml``:

    metabolites (Phase 1) -> metabolite-aware ligand prep (Phase 2)
    -> per-target receptor prep + box resolution -> docking
    -> validation (Phase 4) -> comparison (Phase 5) -> annotation (Phase 6)

``cli.py`` stays a thin wrapper: it just calls `run_metabolite_pipeline`
with a `gnina_vs.config.RunConfig` already loaded and validated.
"""

from __future__ import annotations

import json
import logging
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem

from gnina_vs import (__version__, annotate, compare, docking, figures, ligand, metabolism,
                      pocket, protein, tables, validation)
from gnina_vs.config import RunConfig, TargetConfig
from gnina_vs.reporter import extract_results

logger = logging.getLogger(__name__)


def _filter_sdf(input_sdf: str, compound_ids, output_sdf: str) -> int:
    """Write only the entries whose CompoundID is in ``compound_ids`` from
    ``input_sdf`` to ``output_sdf``. Returns how many were written."""
    wanted = set(compound_ids)
    writer = Chem.SDWriter(output_sdf)
    n = 0
    for mol in Chem.SDMolSupplier(input_sdf, removeHs=False):
        if mol is None:
            continue
        cid = mol.GetProp("CompoundID") if mol.HasProp("CompoundID") else None
        if cid in wanted:
            writer.write(mol)
            n += 1
    writer.close()
    return n


def _prepare_target_receptor(target: TargetConfig, outdir: Path, protein_config):
    """Fetch/prepare the receptor and resolve the docking box for one
    target, exactly once. The box is identical for every compound docked
    against this target (stored in the manifest)."""
    target_dir = outdir / target.id
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_pdb = protein.resolve_protein_source(target.pdb, download_dir=str(target_dir))
    prepared = protein.prepare_protein(
        input_pdb=raw_pdb, output_pdb=str(target_dir / "receptor_prepared.pdb"),
        keep_hetatm_residues=set(target.keep_hetatm) if target.keep_hetatm else None,
        add_hydrogens=protein_config.add_hydrogens, ph=protein_config.ph,
    )

    if target.center_from_ligand:
        try:
            center = pocket.detect_active_site(
                pdb_path=prepared, raw_pdb=raw_pdb,
                ligand_resname=target.ligand_resname, ligand_chain=target.chain,
                visualization_path=str(target_dir / "docking_box.pml"),
            )
        except ValueError as e:
            raise ValueError(
                f"Target '{target.id}': center_from_ligand requires ligand "
                f"'{target.ligand_resname}' in '{target.pdb}', but it wasn't found "
                f"(apo structure?). {e}"
            )
    else:
        center = pocket.detect_active_site(
            pdb_path=prepared, raw_pdb=raw_pdb, center=target.center,
            visualization_path=str(target_dir / "docking_box.pml"),
        )

    return raw_pdb, prepared, center, target_dir


def _check_compound_fits_box(smiles: str, box_size: float, compound_id: str, target_id: str) -> None:
    """Warn (don't fail) when a compound's approximate extent may not fit
    the box -- e.g. a bulky Phase-II glucuronide/sulfate conjugate."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() < 2:
        return
    mol_h = Chem.AddHs(mol)
    from rdkit.Chem import AllChem
    if AllChem.EmbedMolecule(mol_h, randomSeed=1) == -1:
        return
    import numpy as np
    pos = mol_h.GetConformer().GetPositions()
    extent = float(np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1).max())
    if extent > box_size - 4.0:
        logger.warning(
            "  [metabolite-run] Compound '%s' (target '%s') has an estimated extent of "
            "%.1f A, close to or larger than the %.1f A box -- it may not dock properly "
            "(this often happens with Phase II conjugates like glucuronides/sulfates).",
            compound_id, target_id, extent, box_size,
        )


def _package_versions(gnina_exec: str | None = None) -> dict:
    versions = {"gnina_vs": __version__, "python": platform.python_version()}
    try:
        import rdkit
        versions["rdkit"] = rdkit.__version__
    except Exception:
        versions["rdkit"] = None
    if gnina_exec:
        versions["gnina"] = docking.get_gnina_version(gnina_exec)
    return versions


def run_metabolite_pipeline(config: RunConfig) -> dict:
    """Run the full metabolite-aware pipeline for a validated `RunConfig`.

    Returns a dict of the key output paths (``metabolites_csv``,
    ``prepared_ligands_sdf``, ``comparison_table``, ``validation_redocking``,
    ``run_manifest``), matching the files the CLI reports to the user.
    """
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -- Phase 1: metabolites, one shared table across all drugs --
    if (config.metabolism.metabolites_csv and len(config.drugs) > 1
            and not metabolism.metabolites_csv_has_parent_column(config.metabolism.metabolites_csv)):
        logger.warning(
            "metabolism.metabolites_csv ('%s') has no parent_id/parent/drug_id/drug column "
            "and %d drugs are configured -- EVERY curated metabolite in this file will be "
            "attached to EVERY drug. Add a column named 'parent_id' (matching each drug's "
            "'id' in this config) to scope metabolites to the correct parent.",
            config.metabolism.metabolites_csv, len(config.drugs),
        )

    metabolite_frames = []
    for drug in config.drugs:
        df = metabolism.generate_metabolites(
            drug.id, drug.smiles,
            jar_path=config.metabolism.biotransformer_jar,
            btype=config.metabolism.type, steps=config.metabolism.steps,
            include_gut=config.metabolism.include_gut,
            metabolites_csv=config.metabolism.metabolites_csv,
            cache_dir=str(outdir / ".biotransformer_cache"),
        )
        metabolite_frames.append(df)
    metabolites_df = (pd.concat(metabolite_frames, ignore_index=True)
                       if metabolite_frames else pd.DataFrame(columns=metabolism.STANDARD_COLUMNS))
    metabolites_csv_path = outdir / "metabolites.csv"
    metabolites_df.to_csv(metabolites_csv_path, index=False)
    logger.info("Wrote %d compound(s) (parents + metabolites) to '%s'.",
               len(metabolites_df), metabolites_csv_path)

    # -- Phase 2: metabolite-aware ligand prep, one shared SDF --
    prepared_sdf = str(outdir / "prepared_ligands.sdf")
    prepared_df, failures_df = ligand.prepare_ligands_extended(
        metabolites_df, prepared_sdf, seed=config.seed,
    )
    prepared_df.to_csv(outdir / "prepared_ligands.csv", index=False)
    if not failures_df.empty:
        failures_df.to_csv(outdir / "ligand_prep_failures.csv", index=False)
        logger.warning("%d compound(s) failed ligand preparation -- see ligand_prep_failures.csv",
                       len(failures_df))

    compound_to_parent = dict(zip(prepared_df["compound_id"], prepared_df["parent_id"]))

    drugs_by_target: dict[str, list[str]] = {}
    for drug in config.drugs:
        for tid in drug.targets:
            drugs_by_target.setdefault(tid, []).append(drug.id)

    manifest = {
        # run_id ties every emitted row back to this run, and is the one
        # manifest field that also lives as a column (in poses.run_id).
        "run_id": uuid.uuid4().hex[:12],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": tables.SCHEMA_VERSION,
        "seed": config.seed, "outdir": str(outdir), "docking": vars(config.docking),
        "targets": {}, "versions": _package_versions(config.docking.gnina_exec),
    }
    comparison_frames = []
    validation_rows = []

    for target in config.targets:
        if target.id not in drugs_by_target:
            logger.info("Target '%s' has no assigned drugs -- skipping.", target.id)
            continue

        raw_pdb, prepared_receptor, center, target_dir = _prepare_target_receptor(target, outdir, config.protein)

        parent_ids_here = set(drugs_by_target[target.id])
        target_compound_ids = [cid for cid, pid in compound_to_parent.items() if pid in parent_ids_here]
        if not target_compound_ids:
            logger.warning("Target '%s': no prepared ligands matched -- skipping docking.", target.id)
            continue

        for cid in target_compound_ids:
            smi = prepared_df.loc[prepared_df["compound_id"] == cid, "smiles"].iloc[0]
            _check_compound_fits_box(smi, config.docking.box_size, cid, target.id)

        target_sdf = str(target_dir / "ligands.sdf")
        n = _filter_sdf(prepared_sdf, target_compound_ids, target_sdf)
        if n == 0:
            logger.warning("Target '%s': ligand filtering produced no entries -- skipping.", target.id)
            continue

        docking_results = docking.run_gnina_docking(
            receptor_pdb=prepared_receptor, ligands_sdf=target_sdf, center=center,
            box_size=config.docking.box_size, out_dir=str(target_dir / "docking_outputs"),
            gnina_exec=config.docking.gnina_exec, cnn_scoring=config.docking.cnn_scoring,
            exhaustiveness=config.docking.exhaustiveness, num_modes=config.docking.num_modes,
            num_workers=config.docking.workers, cpu_per_worker=config.docking.cpu_per_worker,
            timeout=config.docking.timeout, seed=config.seed,
        )
        results_df = extract_results(
            docking_results=docking_results, input_path=None,
            output_csv=str(target_dir / "results_summary.csv"),
            output_excel=str(target_dir / "top_results.xlsx"), top_n="all",
            merged_sdf=str(target_dir / "ranked_docked.sdf"),
        )

        # -- Phase 4: validation --
        val_row = validation.validate_target(
            target, raw_pdb, prepared_receptor, center, config.docking, target_dir, config.seed,
        )
        validation_rows.append(val_row)

        ref_results_df = validation.dock_reference_ligands(
            target, config.docking, prepared_receptor, center, target_dir, config.seed,
        )
        reference_scores = None
        if not ref_results_df.empty:
            best_ref = ref_results_df.iloc[0]
            reference_scores = {"cnn_affinity": best_ref.get("cnn_affinity"),
                                "vina_affinity": best_ref.get("vina_affinity")}

        # -- Phase 5: comparison table for this target --
        comp_df = compare.build_comparison_table(
            results_df, prepared_df, target_id=target.id, reference_scores=reference_scores,
        )
        comp_df.to_csv(target_dir / "comparison_table.csv", index=False)
        comparison_frames.append(comp_df)

        manifest["targets"][target.id] = {
            "pdb": target.pdb, "box": {"center": list(center), "size": config.docking.box_size},
            "n_compounds": n, "validation_status": val_row["status"],
        }

    comparison_all = (pd.concat(comparison_frames, ignore_index=True)
                       if comparison_frames else pd.DataFrame(columns=compare.COMPARISON_COLUMNS))

    # -- Phase 6: annotations (columns only) --
    if not comparison_all.empty:
        comparison_all = annotate.add_annotations(comparison_all, smiles_col="smiles")

    # -- Phase 6b: split into three grains, Parquet authoritative --
    # compounds / poses / screen, written with nullable dtypes and the run
    # manifest embedded in the Parquet metadata. comparison_table.csv is kept
    # as a GENERATED legacy view so existing scripts and figures keep working;
    # new code should read screen joined to compounds.
    table_paths: dict[str, str] = {}
    comparison_path = outdir / "comparison_table.csv"
    if not comparison_all.empty:
        try:
            split = tables.split_tables(comparison_all, run_id=manifest.get("run_id"))
            table_paths = tables.write_tables(split, outdir / "tables", manifest=manifest)
            logger.info(
                "  [tables] compounds=%d  poses=%d  screen=%d -> '%s'.",
                len(split["compounds"]), len(split["poses"]), len(split["screen"]),
                outdir / "tables",
            )
            comparison_all.to_csv(comparison_path, index=False, na_rep="NA")
        except Exception as exc:
            # The split must never cost the run its results.
            logger.warning(
                "  [tables] splitting the output failed (%s) -- only the flat "
                "comparison_table.csv was written.", exc,
            )
            comparison_all.to_csv(comparison_path, index=False, na_rep="NA")
    else:
        comparison_all.to_csv(comparison_path, index=False, na_rep="NA")

    validation_df = pd.DataFrame(
        validation_rows, columns=["target_id", "status", "rmsd"],
    ) if validation_rows else pd.DataFrame(columns=["target_id", "status", "rmsd"])
    validation_path = outdir / "validation_redocking.csv"
    validation_df.to_csv(validation_path, index=False)

    manifest_path = outdir / "run_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # -- Phase 7: figures (best-effort; a figure failing must never fail the run) --
    figures_dir = outdir / "figures"
    figures_dir.mkdir(exist_ok=True)
    figure_paths: dict = {}
    if not comparison_all.empty:
        for drug in config.drugs:
            try:
                figure_paths[f"metabolic_tree_{drug.id}"] = figures.metabolic_tree(
                    comparison_all, drug.id, str(figures_dir / f"metabolic_tree_{drug.id}"),
                )
            except ValueError as exc:
                logger.info("  [figures] Skipped metabolic tree for '%s': %s", drug.id, exc)
        for target in config.targets:
            try:
                figure_paths[f"bar_{target.id}"] = figures.bar_chart_parent_vs_metabolites(
                    comparison_all, target.id, str(figures_dir / f"bar_{target.id}"),
                )
            except ValueError as exc:
                logger.info("  [figures] Skipped bar chart for '%s': %s", target.id, exc)
        try:
            figure_paths["heatmap"] = figures.heatmap_compounds_targets(
                comparison_all, str(figures_dir / "heatmap"),
            )
        except ValueError as exc:
            logger.info("  [figures] Skipped heatmap: %s", exc)
        try:
            figure_paths["scatter_mw"] = figures.scatter_score_vs_mw(
                comparison_all, str(figures_dir / "scatter_score_vs_mw"),
            )
        except ValueError as exc:
            logger.info("  [figures] Skipped MW scatter: %s", exc)

    return {
        "metabolites_csv": str(metabolites_csv_path),
        "prepared_ligands_sdf": prepared_sdf,
        "comparison_table": str(comparison_path),
        "validation_redocking": str(validation_path),
        "run_manifest": str(manifest_path),
        "figures": figure_paths,
        **table_paths,
    }
