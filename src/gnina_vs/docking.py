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
import os
import shutil
import subprocess
import threading
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import SDWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Gnina's supported CNN scoring modes.
CNN_SCORING_MODES = ("none", "rescore", "refine", "full")


def run_gnina_docking(
    receptor_pdb: str,
    ligands_sdf: str,
    center: tuple[float, float, float],
    box_size: float = 25.0,
    out_dir: Path | str = "docking_outputs",
    gnina_exec: str = "gnina",
    cnn_scoring: str = "rescore",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    num_workers: int = 4,
    cpu_per_worker: int = 2,
    timeout: int = 300,
) -> list[dict]:
    """Iterate over molecules in a multi-molecule SDF, run Gnina in
    parallel, and collect raw output log strings.

    Parameters
    ----------
    receptor_pdb : str
        Path to the prepared receptor PDB.
    ligands_sdf : str
        Multi-molecule SDF of prepared ligands (see `gnina_vs.ligand`).
    center : tuple[float, float, float]
        Docking box center (see `gnina_vs.pocket`).
    box_size : float
        Docking box edge length in Angstroms (cubic box).
    out_dir : Path | str
        Directory for per-ligand docked SDFs and the debug log.
    gnina_exec : str
        Path to (or name of) the gnina executable.
    cnn_scoring : str
        One of ``"none"``, ``"rescore"``, ``"refine"``, ``"full"``. Passed
        straight through to Gnina's ``--cnn_scoring`` flag.
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

    Returns
    -------
    list[dict]
        One entry per docked ligand: mol_name, smiles, index, ligand_id,
        output_path, raw_log.
    """
    logger.info("=" * 60)
    logger.info("STEP 3 -- Gnina Docking")
    logger.info("=" * 60)

    if cnn_scoring not in CNN_SCORING_MODES:
        raise ValueError(
            f"--cnn-scoring must be one of {CNN_SCORING_MODES}, got '{cnn_scoring}'."
        )

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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cx, cy, cz = center

    supplier = Chem.SDMolSupplier(ligands_sdf, removeHs=False)
    molecules = [mol for mol in supplier if mol is not None]

    if not molecules:
        raise ValueError(f"No valid molecules found in '{ligands_sdf}'.")

    logger.info(
        "%d ligands queued for docking (%d workers x %d CPUs each, "
        "cnn_scoring=%s).",
        len(molecules), num_workers, cpu_per_worker, cnn_scoring,
    )

    _debug_saved = threading.Event()

    def _dock_one(mol) -> dict:
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown"
        smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else ""
        mol_index = mol.GetProp("MolIndex") if mol.HasProp("MolIndex") else "-1"
        lig_id = mol.GetProp("LigandID") if mol.HasProp("LigandID") else mol_index

        # Include mol_index in filename to avoid collisions between workers
        tmp_sdf = out_dir / f"{mol_name}_{mol_index}_input.sdf"
        out_sdf_path = out_dir / f"{mol_name}_docked.sdf"

        with SDWriter(str(tmp_sdf)) as w:
            w.write(mol)

        cmd = [
            gnina_exec,
            "--receptor", receptor_pdb,
            "--ligand", str(tmp_sdf),
            "--out", str(out_sdf_path),
            "--center_x", f"{cx:.4f}",
            "--center_y", f"{cy:.4f}",
            "--center_z", f"{cz:.4f}",
            "--size_x", str(box_size),
            "--size_y", str(box_size),
            "--size_z", str(box_size),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--cnn_scoring", cnn_scoring,
            "--cpu", str(cpu_per_worker),
        ]

        raw_log = "TIMEOUT"
        proc = None
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            raw_log = proc.stdout + proc.stderr
            if proc.returncode != 0:
                logger.warning("  Gnina non-zero exit for '%s': %d", mol_name, proc.returncode)
        except FileNotFoundError:
            raise FileNotFoundError(f"Gnina executable not found at '{gnina_exec}'.")
        except subprocess.TimeoutExpired:
            logger.warning("  Docking timed out for '%s' -- skipping.", mol_name)

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
            "smiles": smiles,
            "index": mol_index,
            "ligand_id": lig_id,
            "output_path": str(out_sdf_path),
            "raw_log": raw_log,
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
                logger.warning("  Worker exception for '%s': %s", name, exc)

    logger.info("Docking complete -- %d jobs finished.", len(docking_results))
    if not docking_results:
        raise RuntimeError("No docking jobs completed successfully.")
    return docking_results
