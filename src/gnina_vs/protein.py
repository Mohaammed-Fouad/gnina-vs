"""
gnina_vs.protein
====================
Step 0 of the pipeline: obtain a receptor structure (auto-fetching it from
the RCSB PDB if a bare PDB ID is given instead of a file path) and prepare
it for docking (strip waters/crystallization artefacts, keep ATOM/ANISOU
records, resolve alternate conformations, recognise selenomethionine and
other in-chain modified residues, optionally add hydrogens via OpenBabel).
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

# Modified selenomethionine: HETATM in the PDB, but chemically MET and part
# of the polypeptide chain -- converted rather than stripped.
_SELENOMETHIONINE = "MSE"

# Other modified residues that are covalently part of the polymer chain
# (phosphorylated Ser/Thr/Tyr): retained as-is (still HETATM) rather than
# treated as a ligand/cofactor to be stripped.
_INCHAIN_MODIFIED_KEEP = {"SEP", "TPO", "PTR"}

# HETATM residue names commonly acting as catalytic cofactors or structural
# metals: if these get stripped (because they weren't passed to
# --keep-hetatm), warn explicitly, since removing them can damage the
# binding site (e.g. an NAD(P)H- or heme-dependent enzyme, or a catalytic
# metal in the active site).
_COFACTOR_HINTS = {
    "NAD", "NAI", "NDP", "NAP", "FAD", "FMN", "HEM", "HEC", "PLP",
    "ATP", "ADP", "AMP", "GTP", "GDP", "SAM", "SAH", "COA",
    "ZN", "MG", "CA", "MN", "CU", "FE", "CO", "NI", "K",
}

_PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")

# Fixed-column offsets (0-based, half-open), per the PDB format spec.
_COL_RECORD = slice(0, 6)
_COL_NAME = slice(12, 16)
_COL_ALTLOC = 16
_COL_RESNAME = slice(17, 20)
_COL_CHAIN = 21
_COL_RESSEQ = slice(22, 26)
_COL_ICODE = 26
_COL_OCC = slice(54, 60)


def _set_field(line: str, sl: slice | int, value: str) -> str:
    chars = list(line)
    if isinstance(sl, slice):
        chars[sl] = list(value)
    else:
        chars[sl] = value
    return "".join(chars)


def _atom_key(line: str) -> tuple[str, str, str, str]:
    """(chain, resseq, icode, atom name) -- identifies "the same atom" across
    alternate-location copies."""
    return (line[_COL_CHAIN], line[_COL_RESSEQ], line[_COL_ICODE], line[_COL_NAME])


def _occupancy(line: str) -> float:
    try:
        return float(line[_COL_OCC])
    except ValueError:
        return 1.0


def _convert_mse_atom_line(line: str) -> str:
    """Rewrite one MSE HETATM line as an ATOM/MET line, renaming the
    selenium atom (SE) to the sulfur it replaces (SD)."""
    line = _set_field(line, _COL_RECORD, f"{'ATOM':<6}")
    line = _set_field(line, _COL_RESNAME, f"{'MET':>3}")
    if line[_COL_NAME].strip().upper() == "SE":
        line = _set_field(line, _COL_NAME, f" {'SD':<3}")
        if len(line) >= 78:
            line = _set_field(line, slice(76, 78), f"{'S':>2}")
    return line


def _dedup_altlocs(lines: list[str], atom_idx: set[int]) -> list[str]:
    """Keep exactly one copy of each atom (highest occupancy; ties go to
    the first copy in the file) and blank out its altLoc column, leaving
    every other line untouched."""
    best: dict[tuple, int] = {}
    for i in atom_idx:
        key = _atom_key(lines[i])
        cur = best.get(key)
        if cur is None or _occupancy(lines[i]) > _occupancy(lines[cur]):
            best[key] = i
    winners = set(best.values())
    out = []
    for i, line in enumerate(lines):
        if i in atom_idx and i not in winners:
            continue
        if i in winners:
            line = _set_field(line, _COL_ALTLOC, " ")
        out.append(line)
    return out


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
    2. Remove common crystallographic artefacts (salts, cryo-protectants)
       and any other HETATM not explicitly kept; warn if a stripped residue
       looks like a catalytic cofactor or metal (e.g. NAD, heme, a bound
       ion), since removing it can change the binding site.
    3. Convert selenomethionine (MSE, stored as HETATM) to MET/ATOM, and
       keep other modified residues that are covalently part of the
       polymer chain (SEP, TPO, PTR) rather than stripping them as ligands.
    4. Keep only ATOM / ANISOU records plus any explicitly listed HETATM
       residues (e.g. essential metal cofactors).
    5. Resolve alternate conformations: keep one copy of each atom (highest
       occupancy, ties broken by file order) so no duplicate atoms remain.
    6. Retain only the first model (NMR / multi-model structures).
    7. Add hydrogens via OpenBabel (``obabel``) when available; otherwise
       write the cleaned PDB as-is and log a warning. This adds hydrogens
       for the given pH generally, not only to polar atoms -- OpenBabel
       does not do per-residue pKa or histidine-tautomer assignment.

    Parameters
    ----------
    input_pdb : str
        Path to the raw receptor PDB file.
    output_pdb : str
        Destination path for the prepared PDB.
    keep_hetatm_residues : set[str] | None
        Residue names of HETATM records to *keep* (e.g. metal cofactors).
        ``None`` -> keep nothing beyond the always-kept modified residues
        (safest default for most docking workflows).
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

    keep_hetatm_residues = {r.upper() for r in (keep_hetatm_residues or set())}

    kept_lines: list[str] = []
    atom_idx: set[int] = set()          # indices into kept_lines eligible for altLoc dedup
    model_done = False
    n_water_removed = 0
    removed_counts: dict[str, int] = {}
    mse_conversions: list[tuple[str, str]] = []      # (chain, resseq)
    modified_kept: list[tuple[str, str, str]] = []   # (resname, chain, resseq)

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

            # -- Always keep ATOM records (subject to altLoc dedup) --
            if rec == "ATOM":
                kept_lines.append(raw)
                atom_idx.add(len(kept_lines) - 1)
                continue

            # -- ANISOU: keep alongside its ATOM record, no altLoc dedup logic here --
            if rec == "ANISOU":
                kept_lines.append(raw)
                continue

            # -- Filter HETATM records --
            if rec == "HETATM":
                res_name = raw[17:20].strip().upper()
                chain = raw[21]
                resseq = raw[22:26].strip()

                if res_name in _WATER_RESIDUES:
                    n_water_removed += 1
                    continue

                if res_name == _SELENOMETHIONINE:
                    converted = _convert_mse_atom_line(raw)
                    kept_lines.append(converted)
                    atom_idx.add(len(kept_lines) - 1)
                    mse_conversions.append((chain, resseq))
                    continue

                if res_name in _INCHAIN_MODIFIED_KEEP:
                    kept_lines.append(raw)
                    atom_idx.add(len(kept_lines) - 1)
                    modified_kept.append((res_name, chain, resseq))
                    continue

                if res_name in keep_hetatm_residues:
                    kept_lines.append(raw)
                    atom_idx.add(len(kept_lines) - 1)
                    continue

                removed_counts[res_name] = removed_counts.get(res_name, 0) + 1
                continue

            # -- Keep structural / bookkeeping records --
            if rec in ("SEQRES", "SSBOND", "LINK", "CRYST1", "REMARK",
                       "TER", "CONECT", "END"):
                kept_lines.append(raw)

    # -- Resolve alternate conformations before anything else touches atoms --
    kept_lines = _dedup_altlocs(kept_lines, atom_idx)
    n_kept = sum(1 for l in kept_lines if l[:6].strip() == "ATOM")

    if not any(l[:6].strip() == "ATOM" for l in kept_lines):
        raise ValueError(
            f"No ATOM records survived preparation of '{input_pdb}'. Is "
            f"this a valid protein PDB file?"
        )

    if mse_conversions:
        residues = ", ".join(f"{c}{r}" for c, r in mse_conversions)
        logger.info(
            "  Converted %d selenomethionine residue(s) (MSE, HETATM) to "
            "MET/ATOM (SE -> SD): %s.", len(mse_conversions), residues,
        )
    if modified_kept:
        by_name: dict[str, list[str]] = {}
        for name, chain, resseq in modified_kept:
            by_name.setdefault(name, []).append(f"{chain}{resseq}")
        for name, residues in by_name.items():
            logger.info(
                "  Kept in-chain modified residue %s (HETATM retained, part "
                "of the polymer chain): %s.", name, ", ".join(residues),
            )

    if removed_counts:
        cofactor_like = {n: c for n, c in removed_counts.items() if n in _COFACTOR_HINTS}
        summary = ", ".join(f"{n} x{c}" for n, c in sorted(removed_counts.items()))
        logger.info("  Removed HETATM residues (not requested via --keep-hetatm): %s.", summary)
        if cofactor_like:
            hint = ", ".join(sorted(cofactor_like))
            logger.warning(
                "  Stripped residue(s) that may be a catalytic cofactor or "
                "structural metal: %s. If any of these matter for docking "
                "(e.g. NAD(P)H, heme, FAD, a catalytic metal), re-run with "
                "--keep-hetatm %s.", hint, ",".join(sorted(cofactor_like)),
            )

    n_hetatm_removed = sum(removed_counts.values())

    # Write the cleaned PDB to a temporary file first
    tmp_clean = output_pdb + ".tmp_clean.pdb"
    with open(tmp_clean, "w") as fh:
        fh.writelines(kept_lines)
        if not kept_lines or not kept_lines[-1].startswith("END"):
            fh.write("END\n")

    logger.info(
        "  Cleaned: kept %d ATOM records; removed %d water atoms, "
        "%d other HETATM atoms.",
        n_kept, n_water_removed, n_hetatm_removed,
    )

    # -- Add Hydrogens --
    if add_hydrogens and shutil.which("obabel") is not None:
        logger.info("  Adding hydrogens with OpenBabel (pH %.1f)...", ph)
        try:
            proc = subprocess.run(
                ["obabel", tmp_clean, "-O", output_pdb, "-p", str(ph)],
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
