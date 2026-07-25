"""
gnina_vs.protein
====================
Step 0 of the pipeline: obtain a receptor structure (auto-fetching it from
the RCSB PDB if a bare PDB ID is given instead of a file path) and prepare
it for docking (strip waters/crystallization artefacts, keep ATOM/ANISOU
records, optionally add polar hydrogens via OpenBabel).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Residue names that are *never* part of the protein and should be removed
# during preparation (waters, cryo-protectants, buffer salts, common ions).
_REMOVE_RESIDUES = {
    # water
    "HOH", "WAT", "DOD", "H2O", "TIP", "TIP3",
    # common cryo-protectants / buffer salts
    "SO4", "PO4", "GOL", "EDO", "ACT", "ACE", "DMS", "MPD",
    "PEG", "EOH", "ETH", "IMD", "TRS", "MES", "BME", "DTT",
    "NHE", "NH4", "FMT", "AZI", "IOD", "BR", "CL",
    # common ions -- remove unless you need them as cofactors
    "NA", "K", "CA", "MG", "ZN", "MN", "CU", "FE",
    "CO", "NI", "CD", "HG", "PB",
}

_WATER_RESIDUES = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3"}

_PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


# --------------------------------------------------------------------------- #
# RCSB auto-fetch
# --------------------------------------------------------------------------- #
def fetch_pdb_from_rcsb(pdb_id: str, output_dir: str) -> str:
    """Download ``<pdb_id>.pdb`` from the RCSB PDB REST API.

    Fetches ``https://files.rcsb.org/download/<PDB_ID>.pdb`` and saves it to
    ``output_dir``. Raises `ValueError` for a malformed ID and
    `ConnectionError` for any network/HTTP failure, so the caller can
    present a clean, actionable message instead of a raw traceback.
    """
    pdb_id = pdb_id.strip().upper()
    if not _PDB_ID_RE.match(pdb_id):
        raise ValueError(
            f"'{pdb_id}' does not look like a valid 4-character PDB ID "
            f"(expected e.g. '1ABC'). Pass either a valid PDB ID or the "
            f"path to a local PDB file."
        )

    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, f"{pdb_id}.pdb")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    logger.info("Fetching %s from RCSB PDB (%s)...", pdb_id, url)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except urllib.error.HTTPError as e:
        raise ConnectionError(
            f"RCSB PDB returned HTTP {e.code} for ID '{pdb_id}'. Double-check "
            f"the entry exists at https://www.rcsb.org/structure/{pdb_id}."
        ) from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not reach RCSB PDB ({e.reason}). Check your internet "
            f"connection and that files.rcsb.org isn't blocked by a "
            f"firewall/proxy/VPN, or supply a local PDB file path instead "
            f"of a PDB ID."
        ) from e

    logger.info("Saved receptor structure to '%s'.", dest)
    return dest


def resolve_protein_source(protein: str, download_dir: str = ".") -> str:
    """Resolve the ``--protein`` CLI argument to a local PDB file path.

    If ``protein`` is an existing file, it's returned as-is. Otherwise, if
    it looks like a bare 4-character PDB ID (e.g. ``"1ABC"``), it is
    auto-fetched from RCSB into ``download_dir``. Anything else raises
    `FileNotFoundError`.
    """
    if os.path.exists(protein):
        return protein

    candidate = protein.strip()
    if _PDB_ID_RE.match(candidate):
        return fetch_pdb_from_rcsb(candidate, download_dir)

    raise FileNotFoundError(
        f"Protein input '{protein}' is neither an existing file nor a "
        f"valid 4-character PDB ID."
    )


# --------------------------------------------------------------------------- #
# Protein preparation
# --------------------------------------------------------------------------- #
def prepare_protein(
    input_pdb: str,
    output_pdb: str,
    keep_hetatm_residues: set[str] | None = None,
    add_hydrogens: bool = True,
    ph: float = 7.4,
) -> str:
    """Prepare a raw PDB structure for molecular docking.

    Steps performed
    ---------------
    1. Remove all water molecules.
    2. Remove common crystallographic artefacts (salts, cryo-protectants).
    3. Keep only ATOM / ANISOU records plus any explicitly listed HETATM
       residues (e.g. essential metal cofactors).
    4. Retain only the first model (NMR / multi-model structures).
    5. Add polar hydrogens via OpenBabel (``obabel``) when available;
       otherwise write the cleaned PDB as-is and log a warning.

    Parameters
    ----------
    input_pdb : str
        Path to the raw receptor PDB file.
    output_pdb : str
        Destination path for the prepared PDB.
    keep_hetatm_residues : set[str] | None
        Residue names of HETATM records to *keep* (e.g. metal cofactors).
        ``None`` -> keep nothing (safest default for most docking workflows).
    add_hydrogens : bool
        Whether to attempt hydrogen addition via OpenBabel.
    ph : float
        Protonation pH passed to OpenBabel (``-p``).

    Returns
    -------
    str
        Path to the prepared PDB (``output_pdb``).
    """
    logger.info("=" * 60)
    logger.info("STEP 0 -- Protein Preparation")
    logger.info("=" * 60)

    if not os.path.exists(input_pdb):
        raise FileNotFoundError(f"Protein PDB not found: {input_pdb}")

    keep_hetatm_residues = keep_hetatm_residues or set()

    cleaned_lines: list[str] = []
    model_done = False
    n_water_removed = 0
    n_hetatm_removed = 0
    n_kept = 0

    with open(input_pdb) as fh:
        for raw in fh:
            rec = raw[:6].strip()

            # -- Handle multi-model PDB (keep Model 1 only) --
            if rec == "MODEL":
                if model_done:
                    break  # second MODEL encountered -- stop reading
                continue  # skip the MODEL record itself
            if rec == "ENDMDL":
                model_done = True
                continue

            # -- Always keep ATOM and ANISOU records --
            if rec in ("ATOM", "ANISOU"):
                cleaned_lines.append(raw)
                n_kept += 1
                continue

            # -- Filter HETATM records --
            if rec == "HETATM":
                res_name = raw[17:20].strip().upper()
                if res_name in _WATER_RESIDUES:
                    n_water_removed += 1
                    continue
                if res_name in keep_hetatm_residues:
                    cleaned_lines.append(raw)
                    n_kept += 1
                    continue
                if res_name in _REMOVE_RESIDUES:
                    n_hetatm_removed += 1
                else:
                    n_hetatm_removed += 1
                continue

            # -- Keep structural / bookkeeping records --
            if rec in ("SEQRES", "SSBOND", "LINK", "CRYST1", "REMARK",
                       "TER", "CONECT", "END"):
                cleaned_lines.append(raw)

    if not cleaned_lines:
        raise ValueError(
            f"No ATOM records survived preparation of '{input_pdb}'. Is "
            f"this a valid protein PDB file?"
        )

    # Write the cleaned PDB to a temporary file first
    tmp_clean = output_pdb + ".tmp_clean.pdb"
    with open(tmp_clean, "w") as fh:
        fh.writelines(cleaned_lines)
        if not cleaned_lines[-1].startswith("END"):
            fh.write("END\n")

    logger.info(
        "  Cleaned: kept %d ATOM records; removed %d water atoms, "
        "%d other HETATM atoms.",
        n_kept, n_water_removed, n_hetatm_removed,
    )

    # -- Add Hydrogens --
    if add_hydrogens and shutil.which("obabel") is not None:
        logger.info("  Adding polar hydrogens with OpenBabel...")
        try:
            proc = subprocess.run(
                [
                    "obabel", tmp_clean,
                    "-O", output_pdb,
                    "-p", str(ph),
                    "--partialcharge", "gasteiger",
                ],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0:
                logger.info("  Hydrogen addition successful (OpenBabel).")
            else:
                logger.warning(
                    "  OpenBabel returned code %d. Using cleaned PDB "
                    "without added hydrogens.\n  stderr: %s",
                    proc.returncode, proc.stderr.strip()[:300],
                )
                shutil.copy(tmp_clean, output_pdb)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("  OpenBabel timed out or failed. Using cleaned PDB.")
            shutil.copy(tmp_clean, output_pdb)
    else:
        shutil.copy(tmp_clean, output_pdb)
        if add_hydrogens:
            logger.warning(
                "  OpenBabel not found on PATH -- hydrogens NOT added.\n"
                "  Install with:  conda install -c conda-forge openbabel\n"
                "  Docking will still work; Gnina adds H internally."
            )

    os.remove(tmp_clean)
    logger.info("  Prepared protein written to '%s'.", output_pdb)
    return output_pdb
