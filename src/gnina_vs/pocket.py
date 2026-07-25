"""
gnina_vs.pocket
====================
Step 2 of the pipeline: determine the docking box center.

Priority order
--------------
1. **Manual coordinates** (``--center-x/y/z``) -- highest priority, always
   wins if given.
2. **Manual residue selection** (``--active-site-residues``) -- centroid of
   named catalytic residues (e.g. ``"TYR120,GLU150"``).
3. **Tier 1** -- co-crystallised ligand centroid (searches the *raw*,
   un-stripped PDB, since protein preparation removes HETATM records).
4. **Tier 2** -- ``fpocket`` geometric pocket finder (requires fpocket on
   PATH).
5. **Tier 3** -- built-in grid-based cavity detector (pure Python + NumPy,
   with an optional scipy-accelerated dilation step).

Whichever method wins, a ``.pml`` PyMOL script is written so the detected
box can be visually inspected before committing CPU hours to docking.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

# Residues that are never the docking target: water, crystallization
# additives/cryo-protectants, common ions, AND (new) common enzyme
# cofactors, which Tier 1 would otherwise mistake for a bound drug-like
# ligand.
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
}

_VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "ZN": 1.22, "FE": 1.26, "MG": 1.73, "CA": 1.97,
}
_VDW_DEFAULT = 1.70

# Minimum cavity-cluster size (voxels) below which a cluster is discarded.
# At 1 A resolution, 50 voxels ~= 50 A^3 -- too small to dock a drug-like molecule.
MIN_CLUSTER_VOXELS = 50

_RESIDUE_SPEC_RE = re.compile(r"^([A-Za-z]{3})\s*(-?\d+)$")


def _parse_protein_atoms(pdb_path: str) -> list[tuple[float, float, float, float]]:
    """Return [(x, y, z, vdw_radius), ...] for all ATOM records."""
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            element = line[76:78].strip().upper() if len(line) > 76 else ""
            if not element:
                element = line[12:16].strip().lstrip("0123456789").upper()[:2]
            r = _VDW.get(element, _VDW_DEFAULT)
            atoms.append((x, y, z, r))
    return atoms


def _centroid(coords: list[tuple]) -> tuple[float, float, float]:
    n = len(coords)
    return (sum(c[0] for c in coords) / n,
            sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n)


# --------------------------------------------------------------------------- #
# Manual overrides
# --------------------------------------------------------------------------- #
def parse_residue_spec(spec: str) -> list[tuple[str, int]]:
    """Parse a comma-separated residue spec like ``"TYR120,GLU150"`` into
    ``[("TYR", 120), ("GLU", 150)]``."""
    residues = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        m = _RESIDUE_SPEC_RE.match(token)
        if not m:
            raise ValueError(
                f"Could not parse residue spec '{token}' in "
                f"--active-site-residues. Expected format like 'TYR120' "
                f"(3-letter residue name + residue number)."
            )
        residues.append((m.group(1).upper(), int(m.group(2))))
    if not residues:
        raise ValueError("--active-site-residues was given but no residues could be parsed.")
    return residues


def centroid_from_residues(
    pdb_path: str, residues: list[tuple[str, int]],
) -> tuple[float, float, float]:
    """Compute the centroid of a set of named residues (e.g. key catalytic
    residues), using every atom of each matching residue."""
    wanted = set(residues)
    found = set()
    coords = []

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            res_name = line[17:20].strip().upper()
            try:
                res_seq = int(line[22:26])
            except ValueError:
                continue
            key = (res_name, res_seq)
            if key not in wanted:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
            found.add(key)

    missing = wanted - found
    if missing:
        logger.warning(
            "  [Manual] Residue(s) not found in PDB and ignored: %s",
            ", ".join(f"{name}{num}" for name, num in sorted(missing)),
        )
    if not coords:
        raise ValueError(
            "None of the residues in --active-site-residues were found in "
            "the PDB file. Double-check residue names/numbers against the "
            "structure."
        )

    cx, cy, cz = _centroid(coords)
    logger.info(
        "  [Manual] Centroid of %d/%d requested residues (%d atoms) -> "
        "centre (%.3f, %.3f, %.3f)",
        len(found), len(wanted), len(coords), cx, cy, cz,
    )
    return cx, cy, cz


# --------------------------------------------------------------------------- #
# Tier 1 -- co-crystallised ligand
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
    """Run fpocket and return the centroid of the top-ranked pocket, or None."""
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

        stem = Path(tmp_pdb).stem
        info_file = Path(tmpdir) / f"{stem}_out" / f"{stem}_info.txt"

        if not info_file.exists():
            logger.warning("  [Tier 2] fpocket info file not found.")
            return None

        cx = cy = cz = None
        in_pocket1 = False
        with open(info_file) as fh:
            for line in fh:
                if re.match(r"Pocket\s+1\s*:", line):
                    in_pocket1 = True
                elif re.match(r"Pocket\s+[2-9]", line):
                    break
                if in_pocket1:
                    m = re.search(r"x_centroid\s*:\s*([-\d.]+)", line)
                    if m:
                        cx = float(m.group(1))
                    m = re.search(r"y_centroid\s*:\s*([-\d.]+)", line)
                    if m:
                        cy = float(m.group(1))
                    m = re.search(r"z_centroid\s*:\s*([-\d.]+)", line)
                    if m:
                        cz = float(m.group(1))

        if None in (cx, cy, cz):
            logger.warning("  [Tier 2] Could not parse fpocket centroid.")
            return None

        logger.info("  [Tier 2] fpocket top pocket -> centre (%.3f, %.3f, %.3f)", cx, cy, cz)
        return cx, cy, cz


# --------------------------------------------------------------------------- #
# Tier 3 -- grid-based cavity detector
# --------------------------------------------------------------------------- #
def _manual_dilate_3d(grid: "np.ndarray", radius_voxels: int) -> "np.ndarray":
    """Simple 3-D spherical dilation using nested loops over a (2r+1)^3
    kernel. Used as a scipy-free fallback for probe-radius expansion."""
    result = grid.copy()
    r = radius_voxels
    offsets = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx * dx + dy * dy + dz * dz <= r * r:
                    offsets.append((dx, dy, dz))

    shape = grid.shape
    xs, ys, zs = np.where(grid)
    for x, y, z in zip(xs, ys, zs):
        for dx, dy, dz in offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                result[nx, ny, nz] = True
    return result


def _tier3_grid(
    pdb_path: str,
    resolution: float = 1.0,
    probe_radius: float = 1.4,
) -> tuple[float, float, float] | None:
    """Geometric cavity detection via a 3-D occupancy grid and exterior flood-fill."""
    if not _NUMPY_AVAILABLE:
        logger.warning("  [Tier 3] NumPy not available -- cannot run grid detector.")
        return None

    atoms = _parse_protein_atoms(pdb_path)
    if not atoms:
        logger.warning("  [Tier 3] No ATOM records found.")
        return None

    coords = np.array([(a[0], a[1], a[2]) for a in atoms])
    radii = np.array([a[3] for a in atoms])

    logger.info(
        "  [Tier 3] Building occupancy grid for %d atoms at %s A resolution...",
        len(atoms), resolution,
    )

    padding = probe_radius + 3.0
    mins = coords.min(axis=0) - padding
    maxs = coords.max(axis=0) + padding
    shape = tuple(int(math.ceil((maxs[i] - mins[i]) / resolution)) + 1
                  for i in range(3))

    occupied = np.zeros(shape, dtype=bool)

    for (x, y, z), r in zip(coords, radii):
        idx = ((np.array([x, y, z]) - mins) / resolution).astype(int)
        span = int(math.ceil(r / resolution)) + 1
        xi, yi, zi = idx
        x_lo, x_hi = max(0, xi - span), min(shape[0], xi + span + 1)
        y_lo, y_hi = max(0, yi - span), min(shape[1], yi + span + 1)
        z_lo, z_hi = max(0, zi - span), min(shape[2], zi + span + 1)

        gx = np.arange(x_lo, x_hi)
        gy = np.arange(y_lo, y_hi)
        gz = np.arange(z_lo, z_hi)
        gxx, gyy, gzz = np.meshgrid(gx, gy, gz, indexing="ij")
        rx = mins[0] + gxx * resolution - x
        ry = mins[1] + gyy * resolution - y
        rz = mins[2] + gzz * resolution - z
        mask = (rx**2 + ry**2 + rz**2) <= r * r
        occupied[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi] |= mask

    # -- Probe-expanded occupancy for solvent-accessibility test --
    probe_voxels = max(1, int(math.ceil(probe_radius / resolution)))
    try:
        from scipy.ndimage import binary_dilation, generate_binary_structure
        struct = generate_binary_structure(3, 1)
        probe_expanded = occupied.copy()
        for _ in range(probe_voxels):
            probe_expanded = binary_dilation(probe_expanded, structure=struct)
        logger.debug("  [Tier 3] Probe dilation via scipy.")
    except ImportError:
        logger.debug(
            "  [Tier 3] scipy not available -- using manual 3-D dilation "
            "(slower but correct)."
        )
        probe_expanded = _manual_dilate_3d(occupied, probe_voxels)

    # -- Flood-fill exterior from all border voxels --
    exterior = np.zeros(shape, dtype=bool)
    queue: deque = deque()

    def _seed(ix, iy, iz):
        if not probe_expanded[ix, iy, iz] and not exterior[ix, iy, iz]:
            exterior[ix, iy, iz] = True
            queue.append((ix, iy, iz))

    for ix in range(shape[0]):
        for iy in range(shape[1]):
            _seed(ix, iy, 0)
            _seed(ix, iy, shape[2] - 1)
    for ix in range(shape[0]):
        for iz in range(shape[2]):
            _seed(ix, 0, iz)
            _seed(ix, shape[1] - 1, iz)
    for iy in range(shape[1]):
        for iz in range(shape[2]):
            _seed(0, iy, iz)
            _seed(shape[0] - 1, iy, iz)

    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while queue:
        ix, iy, iz = queue.popleft()
        for dx, dy, dz in directions:
            nx, ny, nz = ix + dx, iy + dy, iz + dz
            if (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                _seed(nx, ny, nz)

    # -- Cavity voxels --
    cavity_mask = (~occupied) & (~exterior)
    n_cavity = int(cavity_mask.sum())

    logger.info(
        "  [Tier 3] Grid shape: %s | occupied: %d | exterior: %d | cavity voxels: %d",
        shape, int(occupied.sum()), int(exterior.sum()), n_cavity,
    )

    if n_cavity == 0:
        logger.warning(
            "  [Tier 3] No cavity voxels found. Try increasing resolution "
            "or installing scipy."
        )
        return None

    cavity_pts = list(zip(*np.where(cavity_mask)))

    # -- Cluster contiguous cavity voxels (6-connectivity BFS) --
    visited = np.zeros(shape, dtype=bool)
    clusters = []

    for seed in cavity_pts:
        seed = tuple(seed)
        if visited[seed]:
            continue
        cluster = []
        bfs = deque([seed])
        visited[seed] = True
        while bfs:
            vx = bfs.popleft()
            cluster.append(vx)
            for dx, dy, dz in directions:
                nb = (vx[0] + dx, vx[1] + dy, vx[2] + dz)
                if (0 <= nb[0] < shape[0] and 0 <= nb[1] < shape[1] and
                        0 <= nb[2] < shape[2] and cavity_mask[nb] and not visited[nb]):
                    visited[nb] = True
                    bfs.append(nb)
        clusters.append(cluster)

    # -- Discard noise clusters below minimum size --
    clusters = [c for c in clusters if len(c) >= MIN_CLUSTER_VOXELS]
    if not clusters:
        logger.warning("  [Tier 3] No cavity cluster >= %d voxels found.", MIN_CLUSTER_VOXELS)
        return None

    def _score(cluster):
        volume = len(cluster)
        burial_sum = 0.0
        for vx in cluster:
            buried = sum(
                1 for dx, dy, dz in directions
                if (0 <= vx[0] + dx < shape[0] and 0 <= vx[1] + dy < shape[1] and
                    0 <= vx[2] + dz < shape[2] and
                    not exterior[(vx[0] + dx, vx[1] + dy, vx[2] + dz)])
            )
            burial_sum += buried / 6.0
        mean_burial = burial_sum / volume

        vx_set = set(map(tuple, cluster))
        surface = sum(
            1 for vx in cluster
            if any((vx[0] + dx, vx[1] + dy, vx[2] + dz) not in vx_set
                   for dx, dy, dz in directions)
        )
        ideal_surface = (36.0 * math.pi * volume * volume) ** (1.0 / 3.0)
        sphericity = ideal_surface / max(surface, 1e-6)

        return (volume ** 1.5) * (mean_burial ** 0.5) * sphericity

    best_cluster = max(clusters, key=_score)
    best_score = _score(best_cluster)

    pts = np.array(best_cluster, dtype=float)
    centre_grid = pts.mean(axis=0)
    cx = float(mins[0] + centre_grid[0] * resolution)
    cy = float(mins[1] + centre_grid[1] * resolution)
    cz = float(mins[2] + centre_grid[2] * resolution)

    logger.info(
        "  [Tier 3] Best pocket: %d voxels (score=%.1f) | %d valid clusters -> "
        "centre (%.3f, %.3f, %.3f)",
        len(best_cluster), best_score, len(clusters), cx, cy, cz,
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
    visualization_path: str | None = None,
    box_size: float = 25.0,
) -> tuple[float, float, float]:
    """Detect (or accept) the docking box center, in priority order:
    manual coordinates > manual residue selection > Tier 1 > Tier 2 > Tier 3.

    Parameters
    ----------
    pdb_path : str
        Path to the **prepared** receptor PDB (waters/HETATM removed). Used
        for Tier 2 (fpocket) and Tier 3 (grid cavity).
    raw_pdb : str | None
        Path to the **original, unmodified** PDB file that still contains
        the co-crystallised ligand HETATM records. When provided, Tier 1
        and the residue-based manual override search this file instead of
        ``pdb_path``.
    center : tuple[float, float, float] | None
        Explicit ``(x, y, z)`` override. Highest priority -- always wins.
    active_site_residues : str | None
        Comma-separated residue spec (e.g. ``"TYR120,GLU150"``). Used if
        ``center`` is not given.
    visualization_path : str | None
        If given, a ``.pml`` PyMOL script is written here showing the
        resulting box.
    box_size : float
        Box edge length (A), only used for the visualization script.

    Returns
    -------
    tuple[float, float, float]
        The ``(x, y, z)`` docking box center.
    """
    logger.info("=" * 60)
    logger.info("STEP 2 -- Active Site Detection")
    logger.info("=" * 60)

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Prepared protein PDB not found: {pdb_path}")

    # For Tier 1 / residue lookups we prefer the raw (un-stripped) PDB so
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
            else:
                logger.info("  fpocket unavailable or failed -- trying Tier 3 (grid cavity)...")
                result = _tier3_grid(pdb_path)
                if result is not None:
                    method = "Tier 3 (geometric grid cavity)"

    if result is None:
        raise RuntimeError(
            "All active-site detection methods failed.\n"
            "  - Provide a PDB with a co-crystallised ligand (pass the "
            "original, pre-preparation file as raw_pdb), OR\n"
            "  - Install fpocket (https://fpocket.sourceforge.net/), OR\n"
            "  - Ensure NumPy (+ scipy recommended) is installed, OR\n"
            "  - Supply --center-x/--center-y/--center-z or "
            "--active-site-residues manually."
        )

    logger.info("  Detection method: %s", method)

    if visualization_path:
        write_pymol_box_script(
            result, box_size, visualization_path, receptor_pdb=pdb_path,
        )

    return result
