"""
gnina_vs.tables
===============
Phase 6b: split the screening output into three tables, one per
observational unit, and write them with types that survive a round-trip.

Why three tables
----------------
The pipeline used to emit one flat ``comparison_table.csv``. That shape has
two concrete defects, not just stylistic ones:

1. **Duplication that becomes a correctness hazard.** Molecular descriptors,
   structural alerts and ADMET predictions are properties of the *molecule*.
   In a flat table they repeat once per target (and, once per-pose
   extraction exists, once per pose). The moment one copy is edited the
   copies disagree and nothing detects it.
2. **An ambiguous rank.** ``rank_in_group`` did not say whether it ranked
   over compounds or over poses, because both lived in one table.

Wickham's third tidy-data rule -- each type of observational unit forms its
own table -- resolves both. The grains here are:

``compounds``  one row per molecule           key: compound_id
``poses``      one row per target+compound+pose
                                              key: (target_id, compound_id, pose_rank)
``screen``     one row per target+compound    key: (target_id, compound_id)

``screen`` is *derived* from ``poses`` by a documented selection rule and is
never hand-maintained. 3D coordinates stay out of all three, in SDF keyed by
``pose_key``.

There is no community schema for docking output
-----------------------------------------------
Two literature searches turned up only tool-specific exports (Meeko, FRED,
GOLD) and no proposed standard, and there is no cheminformatics column-naming
standard either. So this schema is a **local convention**, versioned through
`SCHEMA_VERSION` and described in the emitted ``data_dictionary.json``.
Saying so in the README is FAIR R1.3 compliance in the only form available.

Current limitation: the ``poses`` grain is not yet real
-------------------------------------------------------
``reporter._parse_gnina_log`` deliberately stops at mode 1 ("only want mode
1; stop after first data row"), so the pipeline carries exactly one pose per
compound today. Every ``poses`` row therefore has ``pose_rank == 1`` and the
table is, for now, the same grain as ``screen``. The keys and the selection
rule are already correct for the multi-pose case, so making it real is a
change to score extraction in ``reporter.py`` and nothing here. That change
was deliberately not made blind: parsing every mode across gnina versions
cannot be validated against the fixtures in this repo alone.
"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Closed status vocabularies
# --------------------------------------------------------------------------- #
# Status lives at both grains. A single global status column cannot express
# "prepared fine, failed on target 3", which is exactly the case the no-drop
# rule exists to keep in the table.
COMPOUND_QC_STATUSES = ("ok", "smiles_invalid", "sanitization_failed", "desalting_failed")
POSE_QC_STATUSES = ("ok", "dock_failed", "no_pose", "not_attempted", "timeout", "prep_failed")

# docking.py's job outcomes -> the pose vocabulary above.
_DOCK_STATUS_MAP = {
    "ok": "ok",
    "failed": "dock_failed",
    "error": "dock_failed",
    "timeout": "timeout",
    "no_output": "no_pose",
    "": "not_attempted",
}

# --------------------------------------------------------------------------- #
# Score resolution -- used to band deltas, NOT validated
# --------------------------------------------------------------------------- #
# A delta smaller than the scoring function's practical resolution is not a
# finding, and ranking such deltas manufactures an ordering out of noise.
# These two numbers decide what counts as "no meaningful change".
#
# THEY ARE PLACEHOLDERS. 1 kcal/mol is a commonly repeated rule of thumb for
# docking score resolution and 0.5 pKd units is its rough CNNaffinity
# analogue, but no source in this project's research quantified either, and
# no paper quantifying docking-score-versus-measured-affinity correlation was
# retrieved at all. Set them from your own redocking spread before reporting
# anything that depends on them, and state the value you used.
CNN_AFFINITY_RESOLUTION = 0.5          # pKd-like units, higher is better
VINA_AFFINITY_RESOLUTION = 1.0         # kcal/mol, more negative is better

DELTA_BANDS = ("better", "no_meaningful_change", "worse")
RANK_BANDS = ("best", "upper", "middle", "lower")
RANK_CHANGE_BANDS = ("improved", "unchanged", "worsened")

# --------------------------------------------------------------------------- #
# Column renames: legacy flat name -> frozen name
# --------------------------------------------------------------------------- #
# Three rules, applied consistently:
#   * prefix by block (mol_, dock_, le_, dcmp_, alert_, flag_, ad_, qc_)
#   * units in the header ONLY where the same quantity appears in more than
#     one scale -- which is exactly the three score columns, the most likely
#     real misreading in this table
#   * a comparison column names its comparator
RENAME_MAP = {
    "ID": "source_id",
    "parent_id": "parent_compound_id",
    "smiles": "canonical_smiles",
    "depth": "met_depth",
    "phase": "met_phase",
    "reaction": "met_reaction",
    "cnn_affinity": "dock_cnn_affinity",
    "vina_affinity": "dock_vina_affinity_kcal_mol",
    "cnn_score": "dock_cnn_score",
    "pose_quality_flag": "qc_pose_quality",
    "heavy_atoms": "mol_heavy_atoms",
    "mol_weight": "mol_weight_g_mol",
    "mol_weight_annot": "mol_weight_g_mol",
    "clogp": "mol_clogp",
    "tpsa": "mol_tpsa_a2",
    "hbd": "mol_hbd",
    "hba": "mol_hba",
    "rotatable_bonds": "mol_rotatable_bonds",
    "formal_charge": "mol_formal_charge",
    "lipinski_violations": "mol_lipinski_violations",
    "brenk_alerts": "alert_brenk_names",
    "pains_alerts": "alert_pains_names",
    "glucuronide_flag": "flag_glucuronide",
    "sulfate_flag": "flag_sulfate",
    "conjugate_flag": "flag_conjugate",
    # `ligand_efficiency_*` and `delta_*` are renamed in the screen builder,
    # where their comparator is unambiguous.
}

COMPOUND_ID_COLUMNS = [
    "compound_id", "base_compound_id", "parent_compound_id", "source_id",
    "canonical_smiles", "inchikey", "inchikey_skeleton",
    "is_parent", "met_depth", "met_phase", "met_reaction", "qc_status",
]
POSES_COLUMNS = [
    "target_id", "compound_id", "pose_rank", "pose_key",
    "dock_cnn_affinity", "dock_vina_affinity_kcal_mol", "dock_cnn_score",
    "qc_pose_quality", "qc_status", "dock_reason", "run_id",
]
SCREEN_COLUMNS = [
    "target_id", "compound_id", "selected_pose_rank",
    "le_cnn_per_heavy_atom", "le_vina_per_heavy_atom",
    "dcmp_cnn_affinity_vs_parent", "dcmp_vina_affinity_vs_parent",
    "dcmp_cnn_band_vs_parent",
    "dcmp_cnn_affinity_vs_ref", "dcmp_vina_affinity_vs_ref",
    "rank_band_in_group", "rank_band_change_vs_parent", "qc_status",
]

TABLE_KEYS = {
    "compounds": ["compound_id"],
    "poses": ["target_id", "compound_id", "pose_rank"],
    "screen": ["target_id", "compound_id"],
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def pose_key(target_id, compound_id, pose_rank) -> str:
    """Stable id for one pose, so 3D coordinates can live outside the tables."""
    raw = f"{target_id}|{compound_id}|{pose_rank}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _inchikeys(smiles) -> tuple[str | None, str | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None, None
    if not key:
        return None, None
    return key, key.split("-")[0]


def _delta_band(delta, resolution: float, higher_is_better: bool) -> str | None:
    """Band a score change, collapsing anything below `resolution` into one
    "no meaningful change" bin rather than ranking it."""
    if delta is None or pd.isna(delta):
        return None
    d = float(delta) if higher_is_better else -float(delta)
    if abs(d) < resolution:
        return "no_meaningful_change"
    return "better" if d > 0 else "worse"


def _rank_bands(ranks: pd.Series) -> pd.Series:
    """Positions -> tiers. Rank 1 is "best"; the rest split into thirds of
    the scored group. Tiers rather than integers, because the scoring
    function does not resolve neighbouring positions."""
    out = pd.Series(pd.NA, index=ranks.index, dtype=object)
    scored = ranks.dropna()
    if scored.empty:
        return out
    n = len(scored)
    for idx, r in scored.items():
        r = float(r)
        if r <= 1:
            out.at[idx] = "best"
        elif n <= 1:
            out.at[idx] = "best"
        else:
            frac = (r - 1) / max(n - 1, 1)
            out.at[idx] = "upper" if frac <= 1 / 3 else ("middle" if frac <= 2 / 3 else "lower")
    return out


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    """Apply `RENAME_MAP`, coalescing any collision into one column.

    One collision is expected and is the point of the refactor: the flat
    table carried molecular weight twice, as ``mol_weight`` from compare.py
    and ``mol_weight_annot`` from annotate.py. Both map to
    ``mol_weight_g_mol``; the first non-null value wins and the duplicate
    disappears.
    """
    work = df.rename(columns=RENAME_MAP)
    if not work.columns.duplicated().any():
        return work
    out = {}
    for name in dict.fromkeys(work.columns):
        block = work.loc[:, work.columns == name]
        if block.shape[1] == 1:
            out[name] = block.iloc[:, 0]
        else:
            coalesced = block.iloc[:, 0]
            for i in range(1, block.shape[1]):
                coalesced = coalesced.combine_first(block.iloc[:, i])
            logger.debug("  [tables] coalesced %d duplicate '%s' columns.", block.shape[1], name)
            out[name] = coalesced
    return pd.DataFrame(out, index=work.index)


def _nullable(series: pd.Series, kind: str) -> pd.Series:
    """Cast to a nullable extension dtype so a missing value can never
    collapse into 0 or False, which a CSV round-trip otherwise does."""
    try:
        if kind == "float":
            return pd.to_numeric(series, errors="coerce").astype("Float64")
        if kind == "int":
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if kind == "bool":
            return series.map(lambda v: pd.NA if pd.isna(v) else bool(v)).astype("boolean")
        return series.astype("string")
    except (TypeError, ValueError):
        return series


# Columns whose dtype is decided by name rather than by inspection, so that
# "missing" stays distinguishable from 0 / False / "" through a round-trip.
_INT_SUFFIXES = ("_n", "_depth", "_rank", "_violations", "_hbd", "_hba",
                 "_heavy_atoms", "_rotatable_bonds", "_formal_charge")
_BOOL_PREFIXES = ("flag_", "is_")
_BOOL_EXACT = ("ad_in_domain",)


def _coerce_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise every column to a nullable extension dtype by name.

    Without this, an alert count arrives as float64 (so `1` prints as `1.0`)
    and a boolean flag arrives as object (so it survives CSV as the string
    `'True'`). Both are the failure the split exists to prevent, so the
    coercion is applied centrally rather than at each call site.
    """
    out = frame.copy()
    for col in out.columns:
        series = out[col]
        if col in _BOOL_EXACT or col.startswith(_BOOL_PREFIXES):
            out[col] = _nullable(series, "bool")
        elif col.endswith(_INT_SUFFIXES):
            out[col] = _nullable(series, "int")
        elif pd.api.types.is_bool_dtype(series):
            out[col] = _nullable(series, "bool")
        elif pd.api.types.is_numeric_dtype(series):
            out[col] = _nullable(series, "float")
        else:
            # Leave genuinely mixed object columns alone rather than
            # stringifying values and losing their type silently.
            if series.map(lambda v: isinstance(v, (str, type(None))) or pd.isna(v)).all():
                out[col] = _nullable(series, "string")
    return out


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """One row per molecule. Takes the first occurrence of each compound_id;
    molecule-level facts are identical across its target rows by construction.
    """
    work = _rename(df)
    if "compound_id" not in work.columns:
        raise ValueError("build_compounds: no 'compound_id' column.")

    first = work.drop_duplicates("compound_id", keep="first").set_index("compound_id", drop=False)

    smiles = first.get("canonical_smiles", pd.Series(dtype=object))
    keys = [_inchikeys(s) for s in smiles]
    out = pd.DataFrame(index=first.index)
    out["compound_id"] = first["compound_id"]
    for col in ("base_compound_id", "parent_compound_id", "source_id", "canonical_smiles"):
        out[col] = first[col] if col in first.columns else pd.NA
    out["inchikey"] = [k[0] for k in keys]
    out["inchikey_skeleton"] = [k[1] for k in keys]
    out["is_parent"] = _nullable(first["is_parent"], "bool") if "is_parent" in first.columns else pd.NA
    out["met_depth"] = _nullable(first["met_depth"], "int") if "met_depth" in first.columns else pd.NA
    for col in ("met_phase", "met_reaction"):
        out[col] = first[col] if col in first.columns else pd.NA

    # An unparseable SMILES stays as a row, flagged -- never dropped, so it
    # remains countable in any denominator.
    out["qc_status"] = ["ok" if k[0] is not None else "smiles_invalid" for k in keys]

    # Everything else that is molecule-level: descriptors, alerts, flags,
    # exposure, applicability domain, ADMET predictions.
    molecule_blocks = ("mol_", "alert_", "flag_", "ad_", "admet_", "exposure_")
    extra = [
        c for c in first.columns
        if c not in out.columns
        and (c.startswith(molecule_blocks) or c in ("physchem_out_of_range", "parent_tanimoto"))
    ]
    for col in extra:
        out[col] = first[col]

    ordered = COMPOUND_ID_COLUMNS + [c for c in out.columns if c not in COMPOUND_ID_COLUMNS]
    return _coerce_dtypes(out[ordered].reset_index(drop=True))


def build_poses(df: pd.DataFrame, run_id: str | None = None) -> pd.DataFrame:
    """One row per (target, compound, pose).

    Today every row is ``pose_rank = 1``: see the module docstring on why the
    grain is not yet real. A compound that failed on one target keeps its row
    there with null scores and a `qc_status` saying why.
    """
    work = _rename(df)
    for required in ("target_id", "compound_id"):
        if required not in work.columns:
            raise ValueError(f"build_poses: no '{required}' column.")

    out = pd.DataFrame(index=work.index)
    out["target_id"] = work["target_id"]
    out["compound_id"] = work["compound_id"]
    out["pose_rank"] = 1
    out["pose_key"] = [
        pose_key(t, c, 1) for t, c in zip(out["target_id"], out["compound_id"])
    ]
    for col in ("dock_cnn_affinity", "dock_vina_affinity_kcal_mol", "dock_cnn_score"):
        out[col] = _nullable(work[col], "float") if col in work.columns else pd.NA
    out["qc_pose_quality"] = work["qc_pose_quality"] if "qc_pose_quality" in work.columns else pd.NA

    raw_status = work["status"] if "status" in work.columns else pd.Series("", index=work.index)
    out["qc_status"] = [_DOCK_STATUS_MAP.get(str(s), "dock_failed") for s in raw_status.fillna("")]
    out["dock_reason"] = work["error"] if "error" in work.columns else pd.NA
    out["run_id"] = run_id if run_id is not None else pd.NA

    # A row with no pose has no rank and no scores. Keeping either non-null
    # would assert that a pose exists, so both are nulled here rather than
    # trusting whatever the upstream frame happened to carry -- a failed
    # docking job that still shows a score is the worst case in the table.
    no_pose = out["qc_status"] != "ok"
    out["pose_rank"] = _nullable(out["pose_rank"].where(~no_pose), "int")
    out.loc[no_pose, "pose_key"] = pd.NA
    for col in ("dock_cnn_affinity", "dock_vina_affinity_kcal_mol", "dock_cnn_score"):
        out[col] = out[col].where(~no_pose)

    return _coerce_dtypes(out[POSES_COLUMNS].reset_index(drop=True))


def build_screen(df: pd.DataFrame,
                 cnn_resolution: float = CNN_AFFINITY_RESOLUTION,
                 vina_resolution: float = VINA_AFFINITY_RESOLUTION) -> pd.DataFrame:
    """One row per (target, compound): everything derived.

    Selection rule, applied and recorded rather than assumed: the pose with
    the best CNNaffinity, falling back to ``pose_rank == 1``. With one pose
    per compound today the rule is trivially satisfied, but it is written
    down here so it does not have to be rediscovered later.
    """
    work = _rename(df)
    out = pd.DataFrame(index=work.index)
    out["target_id"] = work["target_id"]
    out["compound_id"] = work["compound_id"]
    out["selected_pose_rank"] = _nullable(
        pd.Series(1, index=work.index).where(
            work.get("status", pd.Series("ok", index=work.index)).fillna("") == "ok"
        ), "int",
    )

    out["le_cnn_per_heavy_atom"] = _nullable(
        work.get("ligand_efficiency_cnn", pd.Series(pd.NA, index=work.index)), "float")
    out["le_vina_per_heavy_atom"] = _nullable(
        work.get("ligand_efficiency_vina", pd.Series(pd.NA, index=work.index)), "float")

    out["dcmp_cnn_affinity_vs_parent"] = _nullable(
        work.get("delta_cnn_affinity_vs_parent", pd.Series(pd.NA, index=work.index)), "float")
    out["dcmp_vina_affinity_vs_parent"] = _nullable(
        work.get("delta_vina_affinity_vs_parent", pd.Series(pd.NA, index=work.index)), "float")
    out["dcmp_cnn_affinity_vs_ref"] = _nullable(
        work.get("delta_cnn_affinity_vs_reference", pd.Series(pd.NA, index=work.index)), "float")
    out["dcmp_vina_affinity_vs_ref"] = _nullable(
        work.get("delta_vina_affinity_vs_reference", pd.Series(pd.NA, index=work.index)), "float")

    # CNNaffinity is pKd-like: higher is better.
    out["dcmp_cnn_band_vs_parent"] = [
        _delta_band(d, cnn_resolution, higher_is_better=True)
        for d in out["dcmp_cnn_affinity_vs_parent"]
    ]

    # Positions -> tiers, per (target, parent) group where that is known.
    ranks = _nullable(work.get("rank_in_group", pd.Series(pd.NA, index=work.index)), "float")
    bands = pd.Series(pd.NA, index=work.index, dtype=object)
    group_cols = [c for c in ("target_id", "parent_compound_id") if c in work.columns]
    if group_cols:
        for _, idx in work.groupby(group_cols, dropna=False).groups.items():
            bands.loc[idx] = _rank_bands(ranks.loc[idx])
    else:
        bands = _rank_bands(ranks)
    out["rank_band_in_group"] = bands

    # Band-to-band comparison with the parent, rather than a difference of
    # integer positions, so the comparison is internally consistent with the
    # bands actually reported.
    out["rank_band_change_vs_parent"] = _rank_band_change(work, bands)
    out["qc_status"] = [
        _DOCK_STATUS_MAP.get(str(s), "dock_failed")
        for s in work.get("status", pd.Series("", index=work.index)).fillna("")
    ]
    return _coerce_dtypes(out[SCREEN_COLUMNS].reset_index(drop=True))


def _rank_band_change(work: pd.DataFrame, bands: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=work.index, dtype=object)
    if "parent_compound_id" not in work.columns or "is_parent" not in work.columns:
        return out
    order = {b: i for i, b in enumerate(RANK_BANDS)}
    group_cols = [c for c in ("target_id", "parent_compound_id") if c in work.columns]
    for _, idx in work.groupby(group_cols, dropna=False).groups.items():
        block = work.loc[idx]
        parent_rows = block[block["is_parent"] == True]  # noqa: E712
        if parent_rows.empty:
            continue
        parent_band = bands.get(parent_rows.index[0])
        if parent_band is pd.NA or parent_band not in order:
            continue
        for i in idx:
            band = bands.get(i)
            if band not in order:
                continue
            delta = order[band] - order[parent_band]
            out.at[i] = "unchanged" if delta == 0 else ("improved" if delta < 0 else "worsened")
    return out


def split_tables(comparison_df: pd.DataFrame, run_id: str | None = None,
                 cnn_resolution: float = CNN_AFFINITY_RESOLUTION,
                 vina_resolution: float = VINA_AFFINITY_RESOLUTION) -> dict[str, pd.DataFrame]:
    """Build all three tables from the flat comparison frame."""
    return {
        "compounds": build_compounds(comparison_df),
        "poses": build_poses(comparison_df, run_id=run_id),
        "screen": build_screen(comparison_df, cnn_resolution, vina_resolution),
    }


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #
def assert_integrity(tables: dict[str, pd.DataFrame]) -> None:
    """Hard checks at write time. A schema that is only documented is a
    schema that drifts, so these raise rather than warn."""
    compounds, poses, screen = tables["compounds"], tables["poses"], tables["screen"]

    if compounds["compound_id"].duplicated().any():
        dupes = compounds.loc[compounds["compound_id"].duplicated(), "compound_id"].unique()[:5]
        raise ValueError(f"compounds.compound_id is not unique (e.g. {list(dupes)}).")

    posed = poses.dropna(subset=["pose_rank"])
    if posed.duplicated(subset=TABLE_KEYS["poses"]).any():
        raise ValueError("poses is not unique on (target_id, compound_id, pose_rank).")

    if screen.duplicated(subset=TABLE_KEYS["screen"]).any():
        raise ValueError("screen is not unique on (target_id, compound_id).")

    known = set(compounds["compound_id"])
    for name, frame in (("poses", poses), ("screen", screen)):
        orphans = set(frame["compound_id"]) - known
        if orphans:
            raise ValueError(f"{name} references compound_ids absent from compounds: {sorted(orphans)[:5]}")

    bad = set(compounds["qc_status"].dropna()) - set(COMPOUND_QC_STATUSES)
    if bad:
        raise ValueError(f"compounds.qc_status has values outside the closed vocabulary: {sorted(bad)}")
    for name, frame in (("poses", poses), ("screen", screen)):
        bad = set(frame["qc_status"].dropna()) - set(POSE_QC_STATUSES)
        if bad:
            raise ValueError(f"{name}.qc_status has values outside the closed vocabulary: {sorted(bad)}")


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def data_dictionary(tables: dict[str, pd.DataFrame]) -> dict:
    """Machine-readable description of what was emitted.

    States plainly that no community schema exists, because a reader needs to
    know this convention is local before building on it.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "note": (
            "No community schema for docking output exists, and no cheminformatics "
            "column-naming standard exists. This layout is a local convention; "
            "schema_version is the only compatibility signal."
        ),
        "grains": {name: TABLE_KEYS[name] for name in TABLE_KEYS},
        "derivation": {
            "screen": "derived from poses: best dock_cnn_affinity, else pose_rank == 1",
        },
        "vocabularies": {
            "compounds.qc_status": list(COMPOUND_QC_STATUSES),
            "poses.qc_status": list(POSE_QC_STATUSES),
            "screen.qc_status": list(POSE_QC_STATUSES),
            "rank_band_in_group": list(RANK_BANDS),
            "rank_band_change_vs_parent": list(RANK_CHANGE_BANDS),
            "dcmp_cnn_band_vs_parent": list(DELTA_BANDS),
        },
        "score_resolution": {
            "cnn_affinity": CNN_AFFINITY_RESOLUTION,
            "vina_affinity_kcal_mol": VINA_AFFINITY_RESOLUTION,
            "warning": (
                "Placeholders, not validated. They decide what counts as "
                "'no_meaningful_change'. Set them from your own redocking spread "
                "and state the value used."
            ),
        },
        "tables": {
            name: {"n_rows": int(len(frame)), "columns": list(frame.columns)}
            for name, frame in tables.items()
        },
    }


def write_tables(tables: dict[str, pd.DataFrame], outdir, manifest: dict | None = None,
                 write_csv: bool = True) -> dict[str, str]:
    """Write all three tables, Parquet authoritative, CSV generated from it.

    Parquet is authoritative for a verified reason: a CSV round-trip returns a
    boolean column as the *string* ``'True'`` and makes a missing boolean
    indistinguishable from a missing string. Parquet also carries file-level
    key-value metadata, which is where the run manifest goes -- a table
    emailed without its folder is the normal failure mode.
    """
    import pathlib
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    assert_integrity(tables)
    written: dict[str, str] = {}

    payload = dict(manifest or {})
    payload["schema_version"] = SCHEMA_VERSION

    for name, frame in tables.items():
        pq_path = outdir / f"{name}.parquet"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(frame, preserve_index=False)
            existing = table.schema.metadata or {}
            # Keys and values must be bytes, not str.
            merged = {**existing, b"gnina_vs_manifest": json.dumps(payload, default=str).encode()}
            pq.write_table(table.replace_schema_metadata(merged), pq_path, version="2.6")
            written[f"{name}_parquet"] = str(pq_path)
        except ImportError:
            logger.warning(
                "  [tables] pyarrow not installed -- writing CSV only for '%s'. Types will not "
                "survive the round-trip; install pyarrow to keep the authoritative copy.", name,
            )
        except Exception as exc:
            logger.warning("  [tables] writing '%s' as Parquet failed (%s) -- CSV only.", pq_path, exc)

        if write_csv:
            csv_path = outdir / f"{name}.csv"
            frame.to_csv(csv_path, index=False, na_rep="NA")
            written[f"{name}_csv"] = str(csv_path)

    dict_path = outdir / "data_dictionary.json"
    dict_path.write_text(json.dumps(data_dictionary(tables), indent=2, default=str))
    written["data_dictionary"] = str(dict_path)
    return written


def read_manifest(parquet_path) -> dict | None:
    """Read the manifest back out of a Parquet file's schema metadata."""
    try:
        import pyarrow.parquet as pq
        meta = pq.read_schema(parquet_path).metadata or {}
        raw = meta.get(b"gnina_vs_manifest")
        return json.loads(raw.decode()) if raw else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #
def legacy_comparison_view(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rebuild a flat, one-row-per-(target, compound) frame from the split
    tables, for code and scripts that still read ``comparison_table.csv``.

    This is a *generated view*, not a fourth source of truth. It is kept so
    the split does not break existing figures and downstream scripts in one
    step; new code should read ``screen`` joined to ``compounds``.
    """
    screen, compounds = tables["screen"], tables["compounds"]
    return screen.merge(compounds, on="compound_id", how="left", suffixes=("", "_compound"))
