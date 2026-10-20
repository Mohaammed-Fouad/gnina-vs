"""
gnina_vs.docking
====================
Step 3 of the pipeline: run Gnina docking for every prepared ligand in
parallel, using a `ThreadPoolExecutor` (Gnina itself is a subprocess, so
threads are sufficient -- no GIL contention on the actual docking work).
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import SDWriter
from tqdm import tqdm

from gnina_vs.ligand import make_compound_id

logger = logging.getLogger(__name__)

# Gnina's supported values for --cnn_scoring (see the Gnina README /
# `gnina --help`).
CNN_SCORING_MODES = ("none", "rescore", "refinement", "metrorescore", "metrorefine", "all")

# Names used by earlier gnina-vs versions; they are NOT valid Gnina values.
_DEPRECATED_CNN_ALIASES = {"refine": "refinement", "full": "all"}

# Job outcomes recorded in every result entry ("status").
STATUS_OK = "ok"
STATUS_FAILED = "failed"        # gnina exited non-zero
STATUS_TIMEOUT = "timeout"
STATUS_NO_OUTPUT = "no_output"  # exit 0 but no docked SDF was written
STATUS_ERROR = "error"          # exception in the worker


def normalize_cnn_scoring(value: str) -> str:
    """Validate a ``--cnn-scoring`` value and map deprecated names.

    ``refine`` -> ``refinement`` and ``full`` -> ``all`` (with a warning);
    anything not in `CNN_SCORING_MODES` raises `ValueError`.
    """
    v = str(value).strip().lower()
    if v in _DEPRECATED_CNN_ALIASES:
        new = _DEPRECATED_CNN_ALIASES[v]
        logger.warning(
            "--cnn-scoring '%s' is deprecated (not a valid Gnina value); using '%s' instead.",
            v, new,
        )
        v = new
    if v not in CNN_SCORING_MODES:
        raise ValueError(
            f"Invalid --cnn-scoring '{value}'. Valid values: {', '.join(CNN_SCORING_MODES)} "
            f"(deprecated aliases: refine -> refinement, full -> all)."
        )
    return v


def get_gnina_version(gnina_exec: str = "gnina") -> str | None:
    """First line of ``gnina --version`` (or None if it can't be determined)."""
    try:
        test = subprocess.run([gnina_exec, "--version"], capture_output=True,
                              text=True, timeout=15)
    except (subprocess.TimeoutExpired, OSError):
        return None
    lines = (test.stdout + test.stderr).strip().splitlines()
    return lines[0].strip() if lines else None


def _warn_if_binary_lacks_cnn_mode(gnina_exec: str, mode: str) -> None:
    """Cross-check the requested mode against the installed binary's
    ``--help`` text (the Gnina CLI has changed between releases)."""
    try:
        proc = subprocess.run([gnina_exec, "--help"], capture_output=True,
                              text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return
    text = proc.stdout + proc.stderr
    pos = text.find("--cnn_scoring")
    if pos < 0:
        return
    section = text[pos:pos + 500]
    if not re.search(rf"\b{re.escape(mode)}\b", section):
        logger.warning(
            "  The installed gnina's --help does not list cnn_scoring mode '%s'. "
            "Check `%s --help` if docking fails.", mode, gnina_exec,
        )


def _max_extent(mol) -> float:
    """Largest interatomic distance (A) in the first conformer."""
    if mol.GetNumConformers() == 0:
        return 0.0
    pos = mol.GetConformer().GetPositions()
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1).max())


def check_box_size(molecules, box_size: float, margin: float = 4.0) -> None:
    """Warn when a cubic box is much larger than the ligand set needs.

    A ligand of extent E fits comfortably in a box of ~ E + 2*margin (the
    same buffer Gnina's ``--autobox_add`` uses). Oversized boxes waste search
    effort and lower sampling quality at a given exhaustiveness.
    """
    extents = [_max_extent(m) for m in molecules]
    if not extents:
        return
    largest = max(extents)
    needed = largest + 2 * margin
    if box_size > 1.5 * needed:
        logger.warning(
            "  Docking box (%.1f A) is much larger than this ligand set needs (largest "
            "ligand extent %.1f A -> ~%.1f A suggested). Consider a smaller --box-size, "
            "a higher --exhaustiveness, or --autobox-ligand.", box_size, largest, needed,
        )


def run_gnina_docking(
    receptor_pdb: str,
    ligands_sdf: str,
    center: tuple[float, float, float] | None,
    box_size: float = 25.0,
    out_dir: Path | str = "docking_outputs",
    gnina_exec: str = "gnina",
    cnn_scoring: str = "rescore",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    num_workers: int = 4,
    cpu_per_worker: int = 2,
    timeout: int = 300,
    seed: int | None = None,
    autobox_ligand: str | None = None,
    autobox_add: float = 4.0,
) -> list[dict]:
    """Iterate over molecules in a multi-molecule SDF, run Gnina in
    parallel, and collect raw output log strings.

    Parameters
    ----------
    receptor_pdb : str
        Path to the prepared receptor PDB.
    ligands_sdf : str
        Multi-molecule SDF of prepared ligands (see `gnina_vs.ligand`).
    center : tuple[float, float, float] | None
        Docking box center (see `gnina_vs.pocket`). May be None only when
        ``autobox_ligand`` is given.
    box_size : float
        Docking box edge length in Angstroms (cubic box).
    out_dir : Path | str
        Directory for per-ligand docked SDFs and the debug log.
    gnina_exec : str
        Path to (or name of) the gnina executable.
    cnn_scoring : str
        One of `CNN_SCORING_MODES`. The deprecated names ``refine`` and
        ``full`` are mapped to ``refinement`` / ``all`` with a warning.
    exhaustiveness : int
        Gnina/AutoDock Vina search exhaustiveness.
    num_modes : int
        Number of binding modes to generate per ligand.
    num_workers : int
        Number of ligands docked concurrently.
    cpu_per_worker : int
        ``--cpu`` passed to each Gnina process. Total CPU threads used ~=
        ``num_workers * cpu_per_worker`` -- keep this <= your logical core
        count.
    timeout : int
        Per-ligand docking timeout in seconds.
    seed : int | None
        Passed to Gnina as ``--seed`` for reproducible docking (omitted if None).
    autobox_ligand : str | None
        Reference ligand file: Gnina builds the box around it
        (``--autobox_ligand``/``--autobox_add``) instead of ``center``/``box_size``.
    autobox_add : float
        Buffer (A) added around ``autobox_ligand`` (``--autobox_add``).

    Returns
    -------
    list[dict]
        One entry per ligand: mol_name, compound_id, smiles, index,
        ligand_id, output_path, raw_log, ``status`` (``ok`` | ``failed`` |
        ``timeout`` | ``no_output`` | ``error``) and ``error``. Failed jobs
        are reported, never dropped, and never leave a stale output file.
    """
    logger.info("=" * 60)
    logger.info("STEP 3 -- Gnina Docking")
    logger.info("=" * 60)

    cnn_scoring = normalize_cnn_scoring(cnn_scoring)

    if autobox_ligand is None and center is None:
        raise ValueError("Either a box center or an autobox reference ligand is required.")

    if shutil.which(gnina_exec) is None:
        raise FileNotFoundError(
            f"Gnina executable not found: '{gnina_exec}'\n"
            "\n"
            "  Option A -- download the pre-built Linux binary from GitHub:\n"
            "    wget https://github.com/gnina/gnina/releases/latest/download/gnina\n"
            "    chmod +x gnina && sudo mv gnina /usr/local/bin/\n"
            "\n"
            "  Option B -- pass --gnina-exec with the full path to your "
            "gnina binary."
        )

    try:
        test = subprocess.run(
            [gnina_exec, "--version"],
            capture_output=True, text=True, timeout=15,
        )
        if test.returncode in (0, 1):
            ver_lines = (test.stdout + test.stderr).strip().splitlines()
            ver_str = ver_lines[0].strip() if ver_lines else "unknown version"
            logger.info("  Gnina found: %s", ver_str)
        else:
            logger.warning("  Gnina smoke-test returned code %d.", test.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("  Gnina smoke-test timed out -- continuing anyway.")

    _warn_if_binary_lacks_cnn_mode(gnina_exec, cnn_scoring)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    supplier = Chem.SDMolSupplier(ligands_sdf, removeHs=False)
    molecules = [mol for mol in supplier if mol is not None]

    if not molecules:
        raise ValueError(f"No valid molecules found in '{ligands_sdf}'.")

    if autobox_ligand is None:
        check_box_size(molecules, box_size)

    logger.info(
        "%d ligands queued for docking (%d workers x %d CPUs each, "
        "cnn_scoring=%s).",
        len(molecules), num_workers, cpu_per_worker, cnn_scoring,
    )

    _debug_saved = threading.Event()

    def _box_args() -> list[str]:
        if autobox_ligand is not None:
            return ["--autobox_ligand", str(autobox_ligand),
                    "--autobox_add", str(autobox_add)]
        cx, cy, cz = center
        return [
            "--center_x", f"{cx:.4f}", "--center_y", f"{cy:.4f}", "--center_z", f"{cz:.4f}",
            "--size_x", str(box_size), "--size_y", str(box_size), "--size_z", str(box_size),
        ]

    def _dock_one(mol) -> dict:
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown"
        smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else ""
        mol_index = mol.GetProp("MolIndex") if mol.HasProp("MolIndex") else "-1"
        lig_id = mol.GetProp("LigandID") if mol.HasProp("LigandID") else mol_index
        compound_id = (mol.GetProp("CompoundID") if mol.HasProp("CompoundID")
                       else make_compound_id(mol_name, mol_index))

        # compound_id is unique and filesystem-safe: no collisions between
        # workers, no path separators from odd compound names.
        tmp_sdf = out_dir / f"{compound_id}_input.sdf"
        out_sdf_path = out_dir / f"{compound_id}_docked.sdf"

        # Never let a result from a previous run be mistaken for this run's.
        out_sdf_path.unlink(missing_ok=True)

        with SDWriter(str(tmp_sdf)) as w:
            w.write(mol)

        cmd = [
            gnina_exec,
            "--receptor", receptor_pdb,
            "--ligand", str(tmp_sdf),
            "--out", str(out_sdf_path),
            *_box_args(),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--cnn_scoring", cnn_scoring,
            "--cpu", str(cpu_per_worker),
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]

        raw_log = "TIMEOUT"
        proc = None
        status = STATUS_OK
        error = ""
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            raw_log = proc.stdout + proc.stderr
            if proc.returncode != 0:
                status = STATUS_FAILED
                error = f"gnina exit code {proc.returncode}"
                logger.warning("  Gnina non-zero exit for '%s': %d", mol_name, proc.returncode)
            elif not out_sdf_path.exists() or out_sdf_path.stat().st_size == 0:
                status = STATUS_NO_OUTPUT
                error = "gnina exited 0 but wrote no docked SDF"
                logger.warning("  No docked SDF produced for '%s'.", mol_name)
        except FileNotFoundError:
            raise FileNotFoundError(f"Gnina executable not found at '{gnina_exec}'.")
        except subprocess.TimeoutExpired:
            status = STATUS_TIMEOUT
            error = f"timed out after {timeout} s"
            logger.warning("  Docking timed out for '%s' -- skipping.", mol_name)

        if status != STATUS_OK:
            out_sdf_path.unlink(missing_ok=True)   # drop partial output from a failed job

        # Save debug log for the very first completed job only
        if proc is not None and not _debug_saved.is_set():
            _debug_saved.set()
            debug_path = out_dir / "gnina_debug.txt"
            try:
                with open(debug_path, "w") as dbg:
                    dbg.write(f"=== Command ===\n{' '.join(cmd)}\n\n")
                    dbg.write(f"=== Return code: {proc.returncode} ===\n\n")
                    dbg.write(f"=== STDOUT ===\n{proc.stdout}\n\n")
                    dbg.write(f"=== STDERR ===\n{proc.stderr}\n")
                logger.info("  Raw gnina output saved to '%s'.", debug_path)
            except Exception:
                pass

        tmp_sdf.unlink(missing_ok=True)

        return {
            "mol_name": mol_name,
            "compound_id": compound_id,
            "smiles": smiles,
            "index": mol_index,
            "ligand_id": lig_id,
            "output_path": str(out_sdf_path),
            "raw_log": raw_log,
            "status": status,
            "error": error,
        }

    docking_results: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_dock_one, mol): mol for mol in molecules}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(molecules),
            desc="Docking ligands",
            unit="mol",
            colour="green",
        ):
            try:
                docking_results.append(future.result())
            except Exception as exc:
                mol = futures[future]
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown"
                idx = mol.GetProp("MolIndex") if mol.HasProp("MolIndex") else "-1"
                logger.warning("  Worker exception for '%s': %s", name, exc)
                cid = (mol.GetProp("CompoundID") if mol.HasProp("CompoundID")
                       else make_compound_id(name, idx))
                docking_results.append({
                    "mol_name": name, "compound_id": cid,
                    "smiles": mol.GetProp("SMILES") if mol.HasProp("SMILES") else "",
                    "index": idx,
                    "ligand_id": mol.GetProp("LigandID") if mol.HasProp("LigandID") else idx,
                    "output_path": str(out_dir / f"{cid}_docked.sdf"),
                    "raw_log": f"ERROR: {exc}", "status": STATUS_ERROR, "error": str(exc),
                })

    n_ok = sum(r["status"] == STATUS_OK for r in docking_results)
    logger.info("Docking complete -- %d/%d jobs succeeded.", n_ok, len(docking_results))
    if n_ok == 0:
        logger.error(
            "No docking job succeeded. See '%s' for the raw gnina output of the first job.",
            out_dir / "gnina_debug.txt",
        )
    elif n_ok < len(docking_results):
        logger.warning("%d job(s) failed or timed out; they are listed with their status "
                       "in the results table.", len(docking_results) - n_ok)
    return docking_results


# Library-API alias (see gnina_vs.__init__): same function, name matching
# the "dock_ligands" step described for programmatic use.
dock_ligands = run_gnina_docking
