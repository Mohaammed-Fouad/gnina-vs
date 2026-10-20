"""
gnina_vs.metabolism
========================
Phase 1: metabolite generation. Wraps the BioTransformer jar
(https://bitbucket.org/wishartlab/biotransformer3.0/) to predict metabolites
for a parent drug, with a ``--metabolites-csv`` bypass for curated
literature metabolites BioTransformer misses.

IMPORTANT -- verification status
---------------------------------
No BioTransformer jar was available in this environment (only a JRE), so
the command-line flags in `BIOTRANSFORMER_CMD_TEMPLATE` and the
column-name matching in `parse_biotransformer_csv` are written from
BioTransformer 3.0's public documentation, **not** verified against a real
``java -jar biotransformer3.0.jar --help`` run or a real prediction.
Before relying on this for a real run:

1. Call `check_biotransformer(jar_path)` and read its output -- it runs
   ``java -jar <jar> --help`` and returns the raw text.
2. Run one real prediction and inspect the CSV it writes.
3. If the flags or column headers differ from what's below, adjust
   `BIOTRANSFORMER_CMD_TEMPLATE` and `_COLUMN_ALIASES` accordingly.

The parser is intentionally lenient (case-insensitive substring matching
across several likely header names) specifically because of this
uncertainty, and `run_biotransformer` raises `BioTransformerError` with the
full stdout/stderr on any failure so a real mismatch is loud, not silent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)

STANDARD_COLUMNS = [
    "compound_id", "parent_id", "smiles", "inchikey",
    "reaction", "enzyme", "depth", "phase", "is_parent",
]

# Per BioTransformer 3.0's public docs; NOT verified in this environment
# (see module docstring). "allHuman" combines liver + gut metabolism
# predictors; "cyp450" and "phaseII" are the individual predictors.
BIOTRANSFORMER_TYPES = ("allHuman", "cyp450", "phaseII", "hgut", "envmicro")

DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "gnina_vs_biotransformer_cache"

_COLUMN_ALIASES = {
    "smiles": ["metabolite smiles", "smiles", "product smiles"],
    "inchikey": ["inchikey", "metabolite inchikey"],
    "reaction": ["reaction", "biotransformation", "reaction type"],
    "enzyme": ["enzyme(s)", "enzyme", "enzymes", "biosystem"],
    "depth": ["step", "cumulative step", "generation"],
}

# Column names recognised as "which drug this curated metabolite belongs
# to" in a --metabolites-csv file (see load_metabolites_csv).
_PARENT_ID_ALIASES = ["parent_id", "parent", "drug_id", "drug"]


class BioTransformerError(RuntimeError):
    """A BioTransformer invocation, or parsing of its output, failed."""


# --------------------------------------------------------------------------- #
# Java / jar checks
# --------------------------------------------------------------------------- #
def find_java(java_exec: str = "java") -> str | None:
    return shutil.which(java_exec)


def check_java_and_jar(java_exec: str, jar_path: str) -> str:
    """Raise `BioTransformerError` with an actionable message if Java or
    the jar is missing; otherwise return the resolved Java path."""
    java = find_java(java_exec)
    if java is None:
        raise BioTransformerError(
            f"Java executable '{java_exec}' not found on PATH. BioTransformer "
            f"needs a JRE/JDK (Java 8+). Install one, or pass a different "
            f"--java-exec."
        )
    if not os.path.exists(jar_path):
        raise BioTransformerError(f"BioTransformer jar not found: '{jar_path}'.")
    return java


def check_biotransformer(jar_path: str, java_exec: str = "java") -> str | None:
    """Best-effort ``java -jar <jar> --help`` smoke test. Returns the raw
    combined stdout+stderr (or None if it couldn't be run). Use this to
    confirm the CLI flags this module assumes actually match your
    installed BioTransformer version -- see the module docstring."""
    java = check_java_and_jar(java_exec, jar_path)
    try:
        proc = subprocess.run([java, "-jar", jar_path, "--help"],
                              capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("  [metabolism] Could not run BioTransformer --help: %s", e)
        return None
    return proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
# Running BioTransformer (cached)
# --------------------------------------------------------------------------- #
def _cache_key(smiles: str, btype: str, steps: int, include_gut: bool) -> str:
    payload = json.dumps(
        {"smiles": smiles, "type": btype, "steps": steps, "gut": include_gut},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def run_biotransformer(
    smiles: str,
    jar_path: str,
    btype: str = "allHuman",
    steps: int = 1,
    include_gut: bool = False,
    java_exec: str = "java",
    cache_dir: str | Path | None = None,
    timeout: int = 600,
) -> str:
    """Run BioTransformer once for one parent SMILES and return the path
    to the (possibly cached) output CSV.

    Results are cached by ``(smiles, btype, steps, include_gut)`` under
    ``cache_dir`` (default: a shared temp directory), so repeated runs for
    the same drug/settings are instant after the first.
    """
    if btype not in BIOTRANSFORMER_TYPES:
        logger.warning(
            "  [metabolism] BioTransformer type '%s' is not one of the "
            "documented types %s -- passing it through to the jar anyway.",
            btype, BIOTRANSFORMER_TYPES,
        )
    if steps not in (1, 2):
        raise ValueError(f"BioTransformer 'steps' must be 1 or 2 (capped for this project), got {steps}.")

    java = check_java_and_jar(java_exec, jar_path)

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{_cache_key(smiles, btype, steps, include_gut)}.csv"
    if cached.exists():
        logger.info("  [metabolism] Using cached BioTransformer result for '%s' (%s).", smiles, cached.name)
        return str(cached)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "biotransformer_out.csv")
        # NOT VERIFIED against a real installed version -- see module docstring.
        cmd = [java, "-jar", jar_path, "-k", "pred", "-b", btype, "-s", str(steps),
               "-ismi", smiles, "-ocsv", out_csv]
        if include_gut:
            cmd += ["-g"]
        logger.info("  [metabolism] Running BioTransformer (%s, steps=%d) for '%s'...", btype, steps, smiles)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise BioTransformerError(f"BioTransformer timed out after {timeout}s for '{smiles}'.")
        if proc.returncode != 0 or not os.path.exists(out_csv):
            raise BioTransformerError(
                f"BioTransformer failed for '{smiles}' (exit {proc.returncode}). If this "
                f"is your first run, check the flags in BIOTRANSFORMER assumptions "
                f"(gnina_vs.metabolism module docstring) against `java -jar {jar_path} "
                f"--help`.\nstdout (tail): {proc.stdout[-2000:]}\nstderr (tail): {proc.stderr[-2000:]}"
            )
        shutil.copy(out_csv, cached)
        return str(cached)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _find_col(columns, aliases) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for a in aliases:
        if a in lower:
            return lower[a]
    for c in columns:
        cl = c.lower()
        if any(a in cl for a in aliases):
            return c
    return None


def _infer_phase(enzyme) -> str:
    if enzyme is None or (isinstance(enzyme, float) and pd.isna(enzyme)):
        return "unknown"
    e = str(enzyme).upper()
    if any(tag in e for tag in ("UGT", "SULT", "GST", "NAT")):
        return "II"
    if "CYP" in e:
        return "I"
    return "unknown"


def parse_biotransformer_csv(csv_path: str, parent_id: str, parent_smiles: str,
                              default_depth: int = 1) -> pd.DataFrame:
    """Parse one BioTransformer output CSV into the standardized metabolite
    table (see `STANDARD_COLUMNS`). Column names are matched leniently
    (see `_COLUMN_ALIASES`) since the exact header names weren't verified
    against a real installed version -- see the module docstring."""
    raw = pd.read_csv(csv_path)
    if raw.empty:
        logger.warning("  [metabolism] BioTransformer produced no metabolites for '%s' (%s).",
                       parent_id, parent_smiles)
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    smiles_col = _find_col(raw.columns, _COLUMN_ALIASES["smiles"])
    if smiles_col is None:
        raise BioTransformerError(
            f"Could not find a SMILES column in BioTransformer output '{csv_path}' "
            f"(columns found: {list(raw.columns)}). Update _COLUMN_ALIASES in "
            f"gnina_vs.metabolism to match your installed version's output header."
        )
    inchikey_col = _find_col(raw.columns, _COLUMN_ALIASES["inchikey"])
    reaction_col = _find_col(raw.columns, _COLUMN_ALIASES["reaction"])
    enzyme_col = _find_col(raw.columns, _COLUMN_ALIASES["enzyme"])
    depth_col = _find_col(raw.columns, _COLUMN_ALIASES["depth"])

    rows = []
    for i, row in raw.iterrows():
        smi = str(row[smiles_col]).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("  [metabolism] Skipping unparsable metabolite SMILES: %s", smi)
            continue
        canon = Chem.MolToSmiles(mol)
        enzyme_val = row[enzyme_col] if enzyme_col else None
        if isinstance(enzyme_val, float) and pd.isna(enzyme_val):
            enzyme_val = None
        depth = default_depth
        if depth_col and pd.notna(row.get(depth_col)):
            try:
                depth = int(row[depth_col])
            except (TypeError, ValueError):
                pass
        rows.append({
            "compound_id": f"{parent_id}_m{i + 1}",
            "parent_id": parent_id,
            "smiles": canon,
            "inchikey": row[inchikey_col] if inchikey_col and pd.notna(row.get(inchikey_col)) else Chem.MolToInchiKey(mol),
            "reaction": row[reaction_col] if reaction_col and pd.notna(row.get(reaction_col)) else None,
            "enzyme": enzyme_val,
            "depth": depth,
            "phase": _infer_phase(enzyme_val),
            "is_parent": False,
        })
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def metabolites_csv_has_parent_column(path: str) -> bool:
    """Whether a curated metabolites CSV scopes its rows to a specific
    parent drug via a ``parent_id``/``parent``/``drug_id``/``drug`` column.

    If False, *every* row in the file is attached to *every* drug that
    uses it (see `load_metabolites_csv`) -- harmless for a single-drug
    run, a silent correctness trap for a multi-drug one (a metabolite
    meant for drug A would also get attached to drug B). Used by
    `gnina_vs.metabolite_pipeline` to warn when this looks unintentional.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"--metabolites-csv file not found: {path}")
    columns = pd.read_csv(path, nrows=0).columns
    return _find_col(columns, _PARENT_ID_ALIASES) is not None


def load_metabolites_csv(path: str, parent_id: str) -> pd.DataFrame:
    """Load curated literature metabolites from a user-supplied CSV
    (bypassing or supplementing BioTransformer). Requires a ``smiles``
    column; ``id``/``reaction``/``enzyme``/``phase``/``depth`` are optional.

    If the file has a ``parent_id`` (or ``parent``/``drug_id``/``drug``)
    column, only rows whose value matches ``parent_id`` are used -- this
    is how one shared CSV can hold curated metabolites for several
    different drugs in the same `metabolite-run` without cross-
    contaminating each other. Without that column, every row in the file
    is used for every drug that calls this function (fine for a
    single-drug config; see `metabolites_csv_has_parent_column`).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"--metabolites-csv file not found: {path}")
    raw = pd.read_csv(path)

    parent_col = _find_col(raw.columns, _PARENT_ID_ALIASES)
    if parent_col is not None:
        n_total = len(raw)
        raw = raw[raw[parent_col].astype(str).str.strip() == str(parent_id).strip()]
        logger.info(
            "  [metabolism] curated metabolites_csv: %d/%d row(s) scoped to parent '%s' "
            "(matched on '%s' column).", len(raw), n_total, parent_id, parent_col,
        )

    smiles_col = _find_col(raw.columns, ["smiles"])
    if smiles_col is None:
        raise ValueError(f"--metabolites-csv '{path}' must have a 'smiles' column. Found: {list(raw.columns)}")
    id_col = _find_col(raw.columns, ["compound_id", "id", "name"])
    reaction_col = _find_col(raw.columns, _COLUMN_ALIASES["reaction"])
    enzyme_col = _find_col(raw.columns, _COLUMN_ALIASES["enzyme"])
    phase_col = _find_col(raw.columns, ["phase"])
    depth_col = _find_col(raw.columns, _COLUMN_ALIASES["depth"])

    rows = []
    for i, row in raw.iterrows():
        smi = str(row[smiles_col]).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("  [metabolism] Skipping unparsable curated metabolite SMILES: %s", smi)
            continue
        cid = str(row[id_col]) if id_col and pd.notna(row.get(id_col)) else f"{parent_id}_curated{i + 1}"
        rows.append({
            "compound_id": cid,
            "parent_id": parent_id,
            "smiles": Chem.MolToSmiles(mol),
            "inchikey": Chem.MolToInchiKey(mol),
            "reaction": row[reaction_col] if reaction_col and pd.notna(row.get(reaction_col)) else "curated",
            "enzyme": row[enzyme_col] if enzyme_col and pd.notna(row.get(enzyme_col)) else None,
            "depth": int(row[depth_col]) if depth_col and pd.notna(row.get(depth_col)) else 1,
            "phase": row[phase_col] if phase_col and pd.notna(row.get(phase_col)) else "unknown",
            "is_parent": False,
        })
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def standardize_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by InChIKey, keeping the shortest pathway (lowest
    ``depth``) to each distinct structure, and guarantee the parent row
    (``is_parent=True``) survives."""
    if df.empty:
        return df
    df = df.copy()
    df["is_parent"] = df["is_parent"].fillna(False).astype(bool)
    df["depth"] = df["depth"].fillna(1)
    df = df.sort_values(by=["inchikey", "is_parent", "depth"], ascending=[True, False, True])
    df = df.drop_duplicates(subset="inchikey", keep="first")
    if not df["is_parent"].any():
        raise BioTransformerError(
            "standardize_and_dedupe: no parent row (is_parent=True) survived "
            "deduplication -- this should never happen and indicates a bug."
        )
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_metabolites(
    drug_id: str,
    parent_smiles: str,
    *,
    jar_path: str | None = None,
    btype: str = "allHuman",
    steps: int = 1,
    include_gut: bool = False,
    metabolites_csv: str | None = None,
    cache_dir: str | Path | None = None,
    java_exec: str = "java",
) -> pd.DataFrame:
    """Generate the metabolite table for one drug: the parent (always
    included, ``is_parent=True``) plus BioTransformer predictions and/or
    curated ``metabolites_csv`` metabolites, deduplicated by InChIKey.
    """
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        raise ValueError(f"Invalid parent SMILES for drug '{drug_id}': {parent_smiles}")
    parent_canon = Chem.MolToSmiles(parent_mol)

    parent_row = {
        "compound_id": drug_id, "parent_id": drug_id, "smiles": parent_canon,
        "inchikey": Chem.MolToInchiKey(parent_mol), "reaction": None, "enzyme": None,
        "depth": 0, "phase": None, "is_parent": True,
    }
    frames = [pd.DataFrame([parent_row], columns=STANDARD_COLUMNS)]

    if jar_path:
        csv_path = run_biotransformer(
            parent_canon, jar_path, btype=btype, steps=steps, include_gut=include_gut,
            java_exec=java_exec, cache_dir=cache_dir,
        )
        frames.append(parse_biotransformer_csv(csv_path, drug_id, parent_canon))
    elif metabolites_csv is None:
        logger.warning(
            "  [metabolism] No BioTransformer jar and no --metabolites-csv given for "
            "drug '%s' -- only the parent will be docked.", drug_id,
        )

    if metabolites_csv:
        frames.append(load_metabolites_csv(metabolites_csv, drug_id))

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return standardize_and_dedupe(combined)


def generate_metabolites_for_table(
    input_path: str,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    name_col: str = "Name",
    *,
    jar_path: str | None = None,
    btype: str = "allHuman",
    steps: int = 1,
    include_gut: bool = False,
    metabolites_csv: str | None = None,
    cache_dir: str | Path | None = None,
    java_exec: str = "java",
) -> pd.DataFrame:
    """Generate metabolites for *every* compound in a SMILES spreadsheet --
    the same Excel/CSV format `gnina_vs.ligand.prepare_ligands` reads --
    and return one combined table across all of them (each compound's own
    `generate_metabolites` call, concatenated; InChIKey dedup only happens
    *within* each compound's own lineage, not across different parents,
    so two unrelated parents that happen to share a metabolite structure
    both keep their own row/lineage).

    This is what powers ``gnina-vs run --metabolite``: one Excel sheet of
    parent compounds in, metabolites for all of them out, ready for
    `gnina_vs.ligand.prepare_ligands_extended`.

    Row identifiers follow the same fallback as ``id_col``/``name_col``
    elsewhere (``id_col`` value -> ``name_col`` value -> ``compound_<row>``),
    made filesystem-safe and de-duplicated if two rows would otherwise
    collide.
    """
    from gnina_vs.ligand import load_compound_table, safe_filename

    table = load_compound_table(input_path)
    if smiles_col not in table.columns:
        raise ValueError(
            f"'{input_path}' must contain a '{smiles_col}' column. Found columns: {list(table.columns)}"
        )
    has_id = id_col in table.columns
    has_name = name_col in table.columns

    frames = []
    seen_ids: set[str] = set()
    for row_idx, row in table.iterrows():
        smiles = str(row[smiles_col]).strip()

        if has_id and pd.notna(row.get(id_col)) and str(row[id_col]).strip():
            drug_id = safe_filename(str(row[id_col]).strip())
        elif has_name and pd.notna(row.get(name_col)) and str(row[name_col]).strip():
            drug_id = safe_filename(str(row[name_col]).strip())
        else:
            drug_id = f"compound_{row_idx}"
        if drug_id in seen_ids:
            base, n = drug_id, 2
            while drug_id in seen_ids:
                drug_id = f"{base}_{n}"
                n += 1
        seen_ids.add(drug_id)

        try:
            frame = generate_metabolites(
                drug_id, smiles, jar_path=jar_path, btype=btype, steps=steps,
                include_gut=include_gut, metabolites_csv=metabolites_csv,
                cache_dir=cache_dir, java_exec=java_exec,
            )
        except ValueError as exc:
            logger.warning("  [metabolism] Skipping row %d ('%s'): %s", row_idx, smiles, exc)
            continue
        frames.append(frame)

    if not frames:
        raise ValueError(
            f"No valid parent compounds found in '{input_path}' for metabolite generation "
            f"(check the '{smiles_col}' column)."
        )
    return pd.concat(frames, ignore_index=True)
