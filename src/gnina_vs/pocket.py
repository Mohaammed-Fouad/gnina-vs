"""
gnina_vs.pocket
====================
Step 2 of the pipeline: determine the docking box center.

Priority order
--------------
1. **Manual coordinates** (``--center-x/y/z``) -- highest priority, always
   wins if given.
2. **Manual residue selection** (``--active-site-residues``) -- centroid of
   named catalytic residues (e.g. ``"TYR120,GLU150"`` or, in a multi-chain
   structure, ``"A:TYR120,B:GLU150"``).
3. **Explicit ligand** (``--ligand-resname``, optional ``--ligand-chain``)
   -- centroid of a named co-crystallised ligand/cofactor, selected exactly
   rather than guessed.
4. **Tier 1** -- co-crystallised ligand centroid, auto-selected as the
   largest non-excluded HETATM residue (searches the *raw*, un-stripped
   PDB, since protein preparation removes HETATM records).
5. **Tier 2** -- ``fpocket`` geometric pocket finder (requires fpocket on
   PATH).

There is no automatic geometric fallback beyond Tier 2: a grid-based cavity
detector cannot reliably distinguish a real pocket from the probe-radius
shell around the whole protein surface (it tends to just return the
protein's centre), so if Tier 1 and Tier 2 both fail, detection stops and
raises a clear error asking for explicit coordinates, residues, or a named
ligand -- it never silently guesses.

Whichever method wins, a ``.pml`` PyMOL script is written so the detected
box can be visually inspected before committing CPU hours to docking.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Residues that are never the docking target: water, crystallization
# additives/cryo-protectants, common ions, common enzyme cofactors (which
# Tier 1 would otherwise mistake for a bound drug-like ligand), and
# in-chain modified residues (selenomethionine, phosphorylated residues --
# these are part of the polypeptide, not a ligand candidate).
_SOLVENT_RESIDUES = {
    # water
    "HOH", "WAT", "DOD", "H2O",
    # crystallization buffer salts / additives
    "SO4", "PO4", "ACT", "ACE", "NHE", "NH4", "FMT", "AZI",
    "IOD", "BR", "CL",
    # crystallization cryo-protectants / additives (expanded)
    "GOL", "EDO", "DMS", "MPD", "PEG", "OLB", "BOG",
    "EOH", "ETH", "IMD", "TRS", "MES", "BME", "DTT",
    # single-atom ions -- never a ligand
    "NA", "K", "CA", "MG", "ZN", "MN", "CU",
    "FE", "CO", "NI", "CD", "HG", "PB",
    # common enzyme cofactors (expanded) -- these are frequently
    # co-crystallised but are not the compound you want to dock against
    "ATP", "ADP", "AMP", "GTP", "GDP",
    "NAD", "NADP", "FAD", "FMN", "HEM", "PLP",
    # in-chain modified residues (see gnina_vs.protein): part of the
    # polypeptide, never a ligand candidate
    "MSE", "SEP", "TPO", "PTR",
    # N-/O-linked glycosylation sugars -- extremely common on secreted and
    # membrane glycoproteins (e.g. COX-2/1CX2 carries several NAG residues
    # at its N-glycosylation sites). These are structural carbohydrate
    # decorations, never the ligand of interest, but were previously
    # missing from this list -- a single NAG (~15 heavy atoms) could win
    # Tier 1's "most heavy atoms" heuristic over a smaller real ligand.
    "NAG", "NDG", "BMA", "MAN", "GAL", "GLA", "FUC", "FUL",
    "BGC", "GLC", "SIA", "ANE", "NGA", "XYS", "XYP", "A2G",
}

_RESIDUE_SPEC_RE = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9]):)?(?P<name>[A-Za-z]{3})\s*(?P<num>-?\d+)$"
)


def _is_hydrogen_line(line: str) -> bool:
    """True if an ATOM/HETATM line is a hydrogen, using the element column
    (cols 77-78) when filled, falling back to the atom name for older PDB
    files that leave it blank. Needed because Tier 1's "most heavy atoms"
    heuristic must actually count heavy atoms: some depositions carry
    explicit hydrogens on HETATM groups (sometimes added by later
    reprocessing even for a structure solved at a resolution where
    hydrogens were never experimentally observable), and a residue with
    many exchangeable hydrogens (e.g. a sugar's hydroxyls) can otherwise
    inflate its atom count enough to tie or beat a real ligand's.
    """
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element == "H"
    name = line[12:16].strip().lstrip("0123456789")
    return name.upper().startswith("H")


def _centroid(coords: list[tuple]) -> tuple[float, float, float]:
    n = len(coords)
    return (sum(c[0] for c in coords) / n,
            sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n)


def _pocket_centroid_from_file(path: Path) -> tuple[float, float, float] | None:
    """Centroid of every ATOM/HETATM coordinate in a small PDB/PQR file
    (a single fpocket pocket's vertices or contact atoms)."""
    pts = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            pts.append((x, y, z))
    return _centroid(pts) if pts else None


# --------------------------------------------------------------------------- #
# Manual overrides
# --------------------------------------------------------------------------- #
def parse_residue_spec(spec: str) -> list[tuple[str | None, str, int]]:
    """Parse a comma-separated residue spec into ``[(chain_or_None, name,
    number), ...]``.

    Accepts ``"TYR120,GLU150"`` (chain-agnostic; only safe when the residue
    number is unique across chains) or ``"A:TYR120,B:GLU150"`` (chain
    explicit, required whenever the same residue number appears in more
    than one chain).
    """
    residues = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        m = _RESIDUE_SPEC_RE.match(token)
        if not m:
            raise ValueError(
                f"Could not parse residue spec '{token}' in "
                f"--active-site-residues. Expected format like 'TYR120' or "
                f"'A:TYR120' (optional chain + 3-letter residue name + "
                f"residue number)."
            )
        residues.append((m.group("chain"), m.group("name").upper(), int(m.group("num"))))
    if not residues:
        raise ValueError("--active-site-residues was given but no residues could be parsed.")
    return residues


def centroid_from_residues(
    pdb_path: str, residues: list[tuple[str | None, str, int]],
) -> tuple[float, float, float]:
    """Compute the centroid of a set of named residues (e.g. key catalytic
    residues), using every heavy atom of each matching residue (hydrogens,
    when present, are excluded -- see `_is_hydrogen_line`, since including
    them would skew the centroid toward wherever peripheral/exchangeable
    hydrogens happen to be modeled, inconsistently between files that do
    and don't carry explicit hydrogens).

    A residue given without a chain (``(None, "TYR", 120)``) is resolved
    automatically if that (name, number) combination is unique across the
    structure's chains; if it matches more than one chain, this raises
    `ValueError` naming the candidates rather than silently averaging
    across chains that may belong to different subunits/active sites.
    """
    # First pass: which chain(s) does each (name, number) appear in, and
    # collect coordinates keyed by the fully-resolved (chain, name, number).
    chains_seen: dict[tuple[str, int], set[str]] = defaultdict(set)
    coords_by_key: dict[tuple[str, str, int], list[tuple[float, float, float]]] = defaultdict(list)

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if _is_hydrogen_line(line):
                continue
            res_name = line[17:20].strip().upper()
            try:
                res_seq = int(line[22:26])
            except ValueError:
                continue
            chain = line[21]
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            chains_seen[(res_name, res_seq)].add(chain)
            coords_by_key[(chain, res_name, res_seq)].append((x, y, z))

    resolved: list[tuple[str | None, str, int]] = []
    for chain, name, num in residues:
        if chain is None:
            candidates = chains_seen.get((name, num), set())
            if len(candidates) > 1:
                options = ", ".join(f"{c}:{name}{num}" for c in sorted(candidates))
                raise ValueError(
                    f"Residue {name}{num} is ambiguous: it appears in "
                    f"multiple chains ({', '.join(sorted(candidates))}). "
                    f"Specify the chain explicitly, e.g. '{options.split(', ')[0]}'."
                )
            chain = next(iter(candidates)) if candidates else None
        resolved.append((chain, name, num))

    wanted = {(c, n, s) for c, n, s in resolved if c is not None}
    found = set()
    coords = []
    for chain, name, num in resolved:
        if chain is None:
            continue
        pts = coords_by_key.get((chain, name, num))
        if pts:
            coords.extend(pts)
            found.add((chain, name, num))

    missing = wanted | {(c, n, s) for c, n, s in resolved if c is None}
    missing = missing - found
    if missing:
        logger.warning(
            "  [Manual] Residue(s) not found in PDB and ignored: %s",
            ", ".join(f"{(c + ':') if c else ''}{n}{s}" for c, n, s in sorted(missing, key=str)),
        )
    if not coords:
        raise ValueError(
            "None of the residues in --active-site-residues were found in "
            "the PDB file. Double-check residue names/numbers (and chain, "
            "if given) against the structure."
        )

    cx, cy, cz = _centroid(coords)
    logger.info(
        "  [Manual] Centroid of %d/%d requested residues (%d atoms) -> "
        "centre (%.3f, %.3f, %.3f)",
        len(found), len(resolved), len(coords), cx, cy, cz,
    )
    return cx, cy, cz


# --------------------------------------------------------------------------- #
# Explicit ligand selection
# --------------------------------------------------------------------------- #
def select_ligand_by_name(
    pdb_path: str, resname: str, chain: str | None = None,
) -> tuple[float, float, float]:
    """Centroid of one specific, exactly-named HETATM residue (e.g. a
    catalytic cofactor such as ATP/NAD that Tier 1's auto-selection would
    otherwise skip, or a specific chain's copy of a repeated ligand).

    Every distinct HETATM residue with this name is logged (with its chain,
    residue number and heavy-atom count -- hydrogens, when present, are
    excluded from both the count and the centroid; see `_is_hydrogen_line`)
    so an ambiguous choice is visible. If ``chain`` is not given and more
    than one distinct residue (different chain and/or residue number)
    matches ``resname``, this raises `ValueError` listing the candidates
    instead of guessing or averaging across them.
    """
    resname = resname.strip().upper()
    instances: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip().upper() != resname:
                continue
            if _is_hydrogen_line(line):
                continue
            res_chain = line[21]
            res_seq = line[22:26].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            instances[(res_chain, res_seq)].append((x, y, z))

    if not instances:
        raise ValueError(
            f"No HETATM residue named '{resname}' found in '{pdb_path}'. "
            f"Check the residue name (and that this structure isn't apo)."
        )

    for (c, s), pts in sorted(instances.items()):
        logger.info("  [Explicit ligand] %s %s/%s: %d heavy atoms", resname, c, s, len(pts))

    if chain is not None:
        matches = {k: v for k, v in instances.items() if k[0] == chain}
        if not matches:
            raise ValueError(
                f"No HETATM residue named '{resname}' found in chain '{chain}' "
                f"of '{pdb_path}'. Found it in chain(s): "
                f"{', '.join(sorted({c for c, _ in instances}))}."
            )
        instances = matches

    if len(instances) > 1:
        options = ", ".join(f"{c}:{resname}{s}" for c, s in sorted(instances))
        raise ValueError(
            f"Residue name '{resname}' is ambiguous: {len(instances)} distinct "
            f"copies found ({options}). Specify --ligand-chain to pick one."
        )

    (c, s), pts = next(iter(instances.items()))
    cx, cy, cz = _centroid(pts)
    logger.info(
        "  [Explicit ligand] Selected %s %s/%s (%d heavy atoms) -> centre (%.3f, %.3f, %.3f)",
        resname, c, s, len(pts), cx, cy, cz,
    )
    return cx, cy, cz


# --------------------------------------------------------------------------- #
# Tier 1 -- co-crystallised ligand (auto-selected)
# --------------------------------------------------------------------------- #
def _tier1_cocrystal(pdb_path: str) -> tuple[float, float, float] | None:
    """Return centroid of the largest co-crystallised ligand, or None."""
    residue_coords: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            res_name = line[17:20].strip().upper()
            if res_name in _SOLVENT_RESIDUES:
                continue
            if _is_hydrogen_line(line):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            # Key: residue-name + chain + seq-number (unique residue identity)
            chain = line[21:22].strip()
            seq_num = line[22:26].strip()
            key = f"{res_name}_{chain}_{seq_num}"
            residue_coords[key].append((x, y, z))

    if not residue_coords:
        return None

    # Log every detected HETATM residue so the user can verify
    for key, pts in sorted(residue_coords.items()):
        logger.info("  [Tier 1] HETATM residue found: %s (%d heavy atoms)", key, len(pts))

    if len(residue_coords) > 1:
        logger.warning(
            "  [Tier 1] %d candidate HETATM residues found (%s); automatic "
            "selection is ambiguous -- picking the one with the most heavy "
            "atoms. Verify the choice below, or use --ligand-resname / "
            "--ligand-chain to select explicitly if it's wrong.",
            len(residue_coords), ", ".join(sorted(residue_coords)),
        )

    # Use the residue with the most heavy atoms (= most likely the bound ligand)
    best_key = max(residue_coords, key=lambda k: len(residue_coords[k]))
    best_pts = residue_coords[best_key]
    cx, cy, cz = _centroid(best_pts)
    logger.info(
        "  [Tier 1] Selected ligand: %s (%d heavy atoms) -> centre (%.3f, %.3f, %.3f)",
        best_key, len(best_pts), cx, cy, cz,
    )
    return cx, cy, cz


# --------------------------------------------------------------------------- #
# Tier 2 -- fpocket
# --------------------------------------------------------------------------- #
def _tier2_fpocket(pdb_path: str) -> tuple[float, float, float] | None:
    """Run fpocket and return the centroid of the top-ranked pocket, or None.

    fpocket ranks pockets by druggability score and names them ``pocket1``,
    ``pocket2``, ... in that order, writing (per pocket) a
    ``pockets/pocketN_vert.pqr`` (Voronoi vertices defining the cavity) and
    a ``pockets/pocketN_atm.pdb`` (protein atoms lining it). Neither
    ``<stem>_info.txt`` nor the ``.pml``/``.tcl`` visualisation scripts
    contain a machine-readable centroid, so the centroid is computed
    directly from ``pocket1_vert.pqr`` (falling back to ``pocket1_atm.pdb``
    if the vertex file is missing).
    """
    if shutil.which("fpocket") is None:
        logger.debug("  [Tier 2] fpocket not found on PATH -- skipping.")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pdb = os.path.join(tmpdir, "receptor.pdb")
        shutil.copy(pdb_path, tmp_pdb)

        try:
            proc = subprocess.run(
                ["fpocket", "-f", tmp_pdb],
                capture_output=True, text=True, timeout=180, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            logger.warning("  [Tier 2] fpocket timed out.")
            return None

        if proc.returncode != 0:
            logger.warning("  [Tier 2] fpocket exited with code %d.", proc.returncode)
            return None

        pockets_dir = Path(tmpdir) / "receptor_out" / "pockets"
        if not pockets_dir.exists():
            logger.warning("  [Tier 2] fpocket output directory not found: '%s'.", pockets_dir)
            return None

        vert_file = pockets_dir / "pocket1_vert.pqr"
        atm_file = pockets_dir / "pocket1_atm.pdb"

        centroid = None
        source = None
        if vert_file.exists():
            centroid = _pocket_centroid_from_file(vert_file)
            source = vert_file.name
        if centroid is None and atm_file.exists():
            centroid = _pocket_centroid_from_file(atm_file)
            source = atm_file.name

        if centroid is None:
            logger.warning(
                "  [Tier 2] Could not find or parse pocket 1's coordinates "
                "in '%s' (expected pocket1_vert.pqr or pocket1_atm.pdb).",
                pockets_dir,
            )
            return None

        cx, cy, cz = centroid
        logger.info(
            "  [Tier 2] fpocket top pocket (from %s) -> centre (%.3f, %.3f, %.3f)",
            source, cx, cy, cz,
        )
        return cx, cy, cz


# --------------------------------------------------------------------------- #
# PyMOL visualization
# --------------------------------------------------------------------------- #
def write_pymol_box_script(
    center: tuple[float, float, float],
    box_size: float,
    out_path: str,
    receptor_pdb: str | None = None,
    label: str = "docking_box",
) -> str:
    """Write a ``.pml`` PyMOL script that draws the docking box as a CGO
    wireframe cube plus a pseudo-atom at its center, so the grid can be
    visually inspected before committing CPU hours to docking.

    Open with: ``pymol <receptor.pdb> <out_path>``
    """
    cx, cy, cz = center
    half = box_size / 2.0
    load_line = f"load {receptor_pdb}, receptor" if receptor_pdb else \
        "# NOTE: no receptor path given -- load your receptor PDB manually"

    script = f"""\
# Auto-generated by gnina_vs -- visualize the docking search box in PyMOL.
# Usage:
#   pymol {receptor_pdb or 'receptor.pdb'} {os.path.basename(out_path)}
{load_line}
show cartoon, receptor
hide lines, receptor
color grey80, receptor

python
from pymol import cmd
from pymol.cgo import BEGIN, LINES, VERTEX, END, COLOR

cx, cy, cz = {cx:.4f}, {cy:.4f}, {cz:.4f}
half = {half:.4f}

x0, x1 = cx - half, cx + half
y0, y1 = cy - half, cy + half
z0, z1 = cz - half, cz + half

corners = [
    (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
    (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
]
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

box = [COLOR, 1.0, 0.4, 0.0, BEGIN, LINES]
for a, b in edges:
    box += [VERTEX, *corners[a], VERTEX, *corners[b]]
box += [END]

cmd.load_cgo(box, "{label}")
cmd.pseudoatom("{label}_center", pos=[cx, cy, cz])
cmd.show("spheres", "{label}_center")
cmd.set("sphere_scale", 0.5, "{label}_center")
cmd.color("red", "{label}_center")
cmd.zoom("{label}", buffer=5)
python end
"""
    with open(out_path, "w") as fh:
        fh.write(script)
    logger.info("  PyMOL box-visualization script written to '%s'.", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def detect_active_site(
    pdb_path: str,
    raw_pdb: str | None = None,
    center: tuple[float, float, float] | None = None,
    active_site_residues: str | None = None,
    ligand_resname: str | None = None,
    ligand_chain: str | None = None,
    visualization_path: str | None = None,
    box_size: float = 25.0,
) -> tuple[float, float, float]:
    """Detect (or accept) the docking box center, in priority order:
    manual coordinates > manual residue selection > explicit ligand name >
    Tier 1 (auto-selected co-crystal ligand) > Tier 2 (fpocket).

    Parameters
    ----------
    pdb_path : str
        Path to the **prepared** receptor PDB (waters/HETATM removed). Used
        for Tier 2 (fpocket).
    raw_pdb : str | None
        Path to the **original, unmodified** PDB file that still contains
        the co-crystallised ligand HETATM records. When provided, the
        residue- and ligand-name-based lookups search this file instead of
        ``pdb_path``.
    center : tuple[float, float, float] | None
        Explicit ``(x, y, z)`` override. Highest priority -- always wins.
    active_site_residues : str | None
        Comma-separated residue spec (e.g. ``"TYR120,GLU150"`` or, for a
        multi-chain structure, ``"A:TYR120,B:GLU150"``). Used if ``center``
        is not given.
    ligand_resname : str | None
        Exact HETATM residue name to center on (e.g. a cofactor Tier 1
        would otherwise skip, such as ``"ATP"``). Used if neither
        ``center`` nor ``active_site_residues`` is given.
    ligand_chain : str | None
        Optional chain to disambiguate ``ligand_resname`` when the same
        residue name appears more than once.
    visualization_path : str | None
        If given, a ``.pml`` PyMOL script is written here showing the
        resulting box.
    box_size : float
        Box edge length (A), only used for the visualization script.

    Returns
    -------
    tuple[float, float, float]
        The ``(x, y, z)`` docking box center.

    Raises
    ------
    ValueError
        If an explicit residue or ligand selection is ambiguous or not
        found (e.g. an apo structure with no matching HETATM).
    RuntimeError
        If no center is given and both automatic tiers (co-crystal ligand,
        fpocket) fail to find anything -- this never falls back to
        guessing the protein's centre; it asks for an explicit choice.
    """
    logger.info("=" * 60)
    logger.info("STEP 2 -- Active Site Detection")
    logger.info("=" * 60)

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Prepared protein PDB not found: {pdb_path}")

    # For residue/ligand lookups we prefer the raw (un-stripped) PDB so
    # HETATM ligands and full side-chain atoms are still present.
    lookup_pdb = raw_pdb if raw_pdb is not None else pdb_path
    if raw_pdb is not None and not os.path.exists(raw_pdb):
        logger.warning(
            "  raw_pdb '%s' not found -- falling back to the prepared PDB "
            "(co-crystal ligand will likely NOT be detected).", raw_pdb,
        )
        lookup_pdb = pdb_path

    result: tuple[float, float, float] | None = None
    method = None

    if center is not None:
        result = center
        method = "manual coordinates"
        logger.info("  Using manual coordinates -> centre (%.3f, %.3f, %.3f)", *center)
    elif active_site_residues:
        residues = parse_residue_spec(active_site_residues)
        result = centroid_from_residues(lookup_pdb, residues)
        method = "manual active-site residues"
    elif ligand_resname:
        result = select_ligand_by_name(lookup_pdb, ligand_resname, ligand_chain)
        method = f"explicit ligand ({ligand_resname})"
    else:
        logger.info("  Tier 1: searching for a co-crystallised ligand in '%s'...", lookup_pdb)
        result = _tier1_cocrystal(lookup_pdb)
        if result is not None:
            method = "Tier 1 (co-crystallised ligand)"
        else:
            logger.info("  No co-crystallised ligand found -- trying Tier 2 (fpocket)...")
            result = _tier2_fpocket(pdb_path)
            if result is not None:
                method = "Tier 2 (fpocket)"

    if result is None:
        raise RuntimeError(
            "Automatic active-site detection failed: no co-crystallised "
            "ligand was found and fpocket is unavailable or found nothing "
            "(this apo/pocket-free case is NOT resolved by guessing the "
            "protein's geometric centre). Provide one of:\n"
            "  - Explicit coordinates: --center-x/--center-y/--center-z, OR\n"
            "  - --active-site-residues (named catalytic residues), OR\n"
            "  - --ligand-resname (and --ligand-chain) to select a specific "
            "co-crystallised ligand/cofactor explicitly, OR\n"
            "  - Install fpocket (https://fpocket.sourceforge.net/)."
        )

    logger.info("  Detection method: %s", method)

    if visualization_path:
        write_pymol_box_script(
            result, box_size, visualization_path, receptor_pdb=pdb_path,
        )

    return result


# Library-API alias (see gnina_vs.__init__): same function, name matching
# the "resolve_center" step described for programmatic use.
resolve_center = detect_active_site
