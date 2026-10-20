"""
gnina_vs.validation
========================
Phase 4: validate a target before trusting its docking results.

Redocking
---------
For a target with a known co-crystallised ligand (``center_from_ligand`` +
``ligand_resname``), the crystal ligand is extracted from the raw PDB and
docked back into the same box; heavy-atom RMSD to the crystal pose is the
standard sanity check (pass: < 2.0 A). For a target with no co-crystal
ligand (an apo structure, or a manually-specified box), there is nothing to
redock against, and this is reported explicitly as
``"not validated: no reference pose"`` rather than silently skipped.

Caveat (documented, not hidden): atom-index correspondence between the
extracted crystal ligand and the docked pose is assumed to be preserved by
Gnina (same atoms in, same atoms out) since both come from the exact same
input file. This is a simplification -- a fully general implementation
would use graph-based atom matching (e.g. RDKit's `rdMolAlign.GetBestRMS`)
to handle any reordering or molecular symmetry. If the docked pose's heavy
atom count doesn't match the crystal ligand's, RMSD is not computed and the
mismatch is reported rather than silently producing a wrong number.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from gnina_vs import docking, ligand
from gnina_vs.reporter import extract_results

logger = logging.getLogger(__name__)

RMSD_PASS_THRESHOLD = 2.0

NOT_VALIDATED_NO_REFERENCE = "not validated: no reference pose"


def extract_cocrystal_ligand(raw_pdb: str, resname: str, chain: str | None = None) -> Chem.Mol | None:
    """Pull one HETATM residue's atoms straight out of the raw PDB and
    return them as an RDKit Mol with the crystal's own 3-D coordinates
    (bond perception is RDKit's distance-based guess for an arbitrary
    HETATM block, sufficient for RMSD comparison, not for chemistry).
    Returns None if the residue isn't found or can't be parsed at all.
    """
    lines = []
    with open(raw_pdb) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip().upper() != resname.upper():
                continue
            if chain is not None and line[21] != chain:
                continue
            lines.append(line)
    if not lines:
        return None

    block = "".join(lines) + "END\n"
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    except Exception as exc:
        logger.debug("  [validation] Sanitization of extracted co-crystal ligand relaxed: %s", exc)
    return mol


def heavy_atom_rmsd(ref_mol: Chem.Mol, docked_mol: Chem.Mol) -> float | None:
    """RMSD between two heavy-atom conformers assumed to be in the same
    atom order (see the module docstring's caveat). Returns None (rather
    than a misleading number) if the heavy-atom counts don't match."""
    ref = Chem.RemoveHs(ref_mol)
    docked = Chem.RemoveHs(docked_mol)
    if ref.GetNumAtoms() != docked.GetNumAtoms():
        return None
    ref_pos = ref.GetConformer().GetPositions()
    docked_pos = docked.GetConformer().GetPositions()
    return float(np.sqrt(((ref_pos - docked_pos) ** 2).sum(axis=1).mean()))


def redock_cocrystal_ligand(
    raw_pdb: str,
    resname: str,
    chain: str | None,
    prepared_receptor: str,
    center: tuple[float, float, float],
    docking_config,
    target_dir: Path,
    seed: int,
) -> dict:
    """Redock one target's co-crystal ligand and report pass/fail. Returns
    a row for ``validation_redocking.csv``: ``{target_id (filled by
    caller), status, rmsd}``."""
    target_dir = Path(target_dir)
    ref_mol = extract_cocrystal_ligand(raw_pdb, resname, chain)
    if ref_mol is None:
        return {"status": f"not validated: could not find/parse co-crystal ligand '{resname}'", "rmsd": None}

    ref_sdf = str(target_dir / "redocking_reference.sdf")
    w = Chem.SDWriter(ref_sdf)
    w.write(ref_mol)
    w.close()
    input_sdf = str(target_dir / "redocking_input.sdf")
    shutil.copy(ref_sdf, input_sdf)

    try:
        results = docking.run_gnina_docking(
            receptor_pdb=prepared_receptor, ligands_sdf=input_sdf, center=center,
            box_size=docking_config.box_size, out_dir=str(target_dir / "redocking_outputs"),
            gnina_exec=docking_config.gnina_exec, cnn_scoring=docking_config.cnn_scoring,
            exhaustiveness=docking_config.exhaustiveness, num_modes=docking_config.num_modes,
            num_workers=1, cpu_per_worker=docking_config.cpu_per_worker,
            timeout=docking_config.timeout, seed=seed,
        )
    except Exception as exc:
        return {"status": f"not validated: redocking failed ({exc})", "rmsd": None}

    if not results or results[0].get("status") != docking.STATUS_OK:
        job_status = results[0].get("status") if results else "no result"
        return {"status": f"not validated: redocking job did not complete ({job_status})", "rmsd": None}

    docked_mol = next(iter(Chem.SDMolSupplier(results[0]["output_path"], removeHs=False)), None)
    if docked_mol is None:
        return {"status": "not validated: could not read docked pose", "rmsd": None}

    rmsd = heavy_atom_rmsd(ref_mol, docked_mol)
    if rmsd is None:
        return {"status": "not validated: docked pose atom count did not match the crystal ligand", "rmsd": None}

    return {"status": "pass" if rmsd < RMSD_PASS_THRESHOLD else "fail", "rmsd": rmsd}


def validate_target(target, raw_pdb: str, prepared_receptor: str,
                    center: tuple[float, float, float], docking_config,
                    target_dir: Path, seed: int) -> dict:
    """Redocking validation for one target config (see
    `gnina_vs.config.TargetConfig`). Returns
    ``{"target_id", "status", "rmsd"}``; ``status`` is
    `NOT_VALIDATED_NO_REFERENCE` verbatim for a target with no
    co-crystal ligand, so it can be matched exactly downstream."""
    if not (target.center_from_ligand and target.ligand_resname):
        return {"target_id": target.id, "status": NOT_VALIDATED_NO_REFERENCE, "rmsd": None}

    result = redock_cocrystal_ligand(
        raw_pdb=raw_pdb, resname=target.ligand_resname, chain=target.chain,
        prepared_receptor=prepared_receptor, center=center, docking_config=docking_config,
        target_dir=target_dir, seed=seed,
    )
    return {"target_id": target.id, **result}


def dock_reference_ligands(
    target, docking_config, prepared_receptor: str,
    center: tuple[float, float, float], target_dir: Path, seed: int,
) -> pd.DataFrame:
    """Dock a target's configured reference ligand(s) with the exact same
    settings as the main run, so their scores are directly comparable to
    the parent/metabolites. Returns an empty DataFrame if the target has
    no reference ligands configured."""
    if not target.reference_ligands:
        return pd.DataFrame()

    target_dir = Path(target_dir)
    rows = [{
        "compound_id": f"ref__{r.name}", "parent_id": f"ref__{r.name}", "smiles": r.smiles,
        "inchikey": None, "reaction": None, "enzyme": None, "depth": 0, "phase": None, "is_parent": True,
    } for r in target.reference_ligands]
    ref_df = pd.DataFrame(rows, columns=["compound_id", "parent_id", "smiles", "inchikey",
                                         "reaction", "enzyme", "depth", "phase", "is_parent"])

    ref_sdf = str(target_dir / "reference_ligands.sdf")
    prepared_ref_df, _ = ligand.prepare_ligands_extended(ref_df, ref_sdf, seed=seed)

    results = docking.run_gnina_docking(
        receptor_pdb=prepared_receptor, ligands_sdf=ref_sdf, center=center,
        box_size=docking_config.box_size, out_dir=str(target_dir / "reference_docking_outputs"),
        gnina_exec=docking_config.gnina_exec, cnn_scoring=docking_config.cnn_scoring,
        exhaustiveness=docking_config.exhaustiveness, num_modes=docking_config.num_modes,
        num_workers=1, cpu_per_worker=docking_config.cpu_per_worker,
        timeout=docking_config.timeout, seed=seed,
    )
    return extract_results(
        results, None, str(target_dir / "reference_results.csv"),
        str(target_dir / "reference_top.xlsx"), top_n="all",
    )
