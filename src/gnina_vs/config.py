"""
gnina_vs.config
====================
YAML config schema and validation for ``gnina-vs metabolite-run``.

A plain custom validator is used instead of pydantic (not a project
dependency) -- `load_config` raises `ConfigError` with a message that
names the exact field and location at fault, e.g.
``"targets[1]: missing required field 'pdb'."``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
import yaml
from rdkit import Chem

from gnina_vs.docking import normalize_cnn_scoring


class ConfigError(ValueError):
    """The config file is missing a required field or has an invalid value."""


@dataclass
class ReferenceLigand:
    name: str
    smiles: str


@dataclass
class TargetConfig:
    id: str
    role: str = "therapeutic"
    pdb: str = ""
    center_from_ligand: bool = False
    ligand_resname: str | None = None
    chain: str | None = None
    keep_hetatm: list[str] = field(default_factory=list)
    reference_ligands: list[ReferenceLigand] = field(default_factory=list)
    center: tuple[float, float, float] | None = None


@dataclass
class DrugConfig:
    id: str
    smiles: str
    targets: list[str] = field(default_factory=list)


@dataclass
class ProteinConfig:
    add_hydrogens: bool = True
    ph: float = 7.4


@dataclass
class MetabolismConfig:
    biotransformer_jar: str | None = None
    type: str = "allHuman"
    steps: int = 1
    include_gut: bool = False
    metabolites_csv: str | None = None


@dataclass
class DockingConfig:
    cnn_scoring: str = "rescore"
    exhaustiveness: int = 8
    num_modes: int = 9
    workers: int = 4
    cpu_per_worker: int = 2
    box_size: float = 25.0
    gnina_exec: str = "gnina"
    timeout: int = 300


@dataclass
class RunConfig:
    outdir: str = "results"
    seed: int = 42
    protein: ProteinConfig = field(default_factory=ProteinConfig)
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    targets: list[TargetConfig] = field(default_factory=list)
    drugs: list[DrugConfig] = field(default_factory=list)


def _require(d: dict, key: str, ctx: str):
    if not isinstance(d, dict):
        raise ConfigError(f"{ctx}: expected a mapping, got {type(d).__name__}.")
    if key not in d or d[key] in (None, ""):
        raise ConfigError(f"{ctx}: missing required field '{key}'.")
    return d[key]


def _parse_protein(raw: dict) -> ProteinConfig:
    p = raw.get("protein") or {}
    try:
        return ProteinConfig(add_hydrogens=bool(p.get("add_hydrogens", True)), ph=float(p.get("ph", 7.4)))
    except (TypeError, ValueError) as e:
        raise ConfigError(f"protein: invalid value ({e}).")


def _parse_metabolism(raw: dict) -> MetabolismConfig:
    m = raw.get("metabolism") or {}
    steps = int(m.get("steps", 1))
    if steps not in (1, 2):
        raise ConfigError(f"metabolism.steps must be 1 or 2, got {steps}.")
    return MetabolismConfig(
        biotransformer_jar=m.get("biotransformer_jar"),
        type=m.get("type", "allHuman"),
        steps=steps,
        include_gut=bool(m.get("include_gut", False)),
        metabolites_csv=m.get("metabolites_csv"),
    )


def _parse_docking(raw: dict) -> DockingConfig:
    dk = raw.get("docking") or {}
    try:
        cnn_scoring = normalize_cnn_scoring(dk.get("cnn_scoring", "rescore"))
    except ValueError as e:
        raise ConfigError(f"docking.cnn_scoring: {e}")
    try:
        return DockingConfig(
            cnn_scoring=cnn_scoring,
            exhaustiveness=int(dk.get("exhaustiveness", 8)),
            num_modes=int(dk.get("num_modes", 9)),
            workers=int(dk.get("workers", 4)),
            cpu_per_worker=int(dk.get("cpu_per_worker", 2)),
            box_size=float(dk.get("box_size", 25.0)),
            gnina_exec=dk.get("gnina_exec", "gnina"),
            timeout=int(dk.get("timeout", 300)),
        )
    except (TypeError, ValueError) as e:
        raise ConfigError(f"docking: invalid numeric value ({e}).")


def _parse_targets(raw: dict) -> list[TargetConfig]:
    targets_raw = raw.get("targets")
    if not targets_raw:
        raise ConfigError("Config must define at least one target under 'targets'.")
    if not isinstance(targets_raw, list):
        raise ConfigError("'targets' must be a list.")

    targets = []
    seen_ids = set()
    for i, t in enumerate(targets_raw):
        ctx = f"targets[{i}]"
        tid = _require(t, "id", ctx)
        if tid in seen_ids:
            raise ConfigError(f"{ctx}: duplicate target id '{tid}'.")
        seen_ids.add(tid)

        role = t.get("role", "therapeutic")
        if role not in ("therapeutic", "antitarget"):
            raise ConfigError(f"{ctx}: 'role' must be 'therapeutic' or 'antitarget', got '{role}'.")

        pdb = _require(t, "pdb", ctx)
        center_from_ligand = bool(t.get("center_from_ligand", False))
        ligand_resname = t.get("ligand_resname")
        if center_from_ligand and not ligand_resname:
            raise ConfigError(f"{ctx}: center_from_ligand=true requires 'ligand_resname'.")

        center = t.get("center")
        if center is not None:
            if not (isinstance(center, (list, tuple)) and len(center) == 3):
                raise ConfigError(f"{ctx}: 'center' must be a 3-element [x, y, z] list.")
            try:
                center = tuple(float(c) for c in center)
            except (TypeError, ValueError):
                raise ConfigError(f"{ctx}: 'center' values must be numbers.")

        if not center_from_ligand and center is None:
            raise ConfigError(
                f"{ctx}: must give either center_from_ligand: true (+ ligand_resname) "
                f"or an explicit 'center: [x, y, z]'."
            )

        refs = []
        for j, r in enumerate(t.get("reference_ligands", []) or []):
            rctx = f"{ctx}.reference_ligands[{j}]"
            name = _require(r, "name", rctx)
            smiles = _require(r, "smiles", rctx)
            if Chem.MolFromSmiles(smiles) is None:
                raise ConfigError(f"{rctx}: invalid SMILES '{smiles}'.")
            refs.append(ReferenceLigand(name=name, smiles=smiles))

        targets.append(TargetConfig(
            id=tid, role=role, pdb=pdb, center_from_ligand=center_from_ligand,
            ligand_resname=ligand_resname, chain=t.get("chain"),
            keep_hetatm=list(t.get("keep_hetatm", []) or []),
            reference_ligands=refs, center=center,
        ))
    return targets


def _check_targets(tgs, target_ids: set[str], ctx: str) -> list[str]:
    for tg in tgs:
        if tg not in target_ids:
            raise ConfigError(f"{ctx}: target '{tg}' is not defined under 'targets'.")
    return list(tgs)


def _parse_inline_drugs(drugs_raw, target_ids: set[str]) -> list[DrugConfig]:
    """Parse the inline ``drugs:`` list (drugs written directly in the YAML)."""
    if not isinstance(drugs_raw, list):
        raise ConfigError("'drugs' must be a list.")

    drugs = []
    for i, d in enumerate(drugs_raw):
        ctx = f"drugs[{i}]"
        did = _require(d, "id", ctx)
        smiles = _require(d, "smiles", ctx)
        if Chem.MolFromSmiles(smiles) is None:
            raise ConfigError(f"{ctx}: invalid SMILES '{smiles}'.")
        tgs = _check_targets(d.get("targets") or sorted(target_ids), target_ids, ctx)
        drugs.append(DrugConfig(id=did, smiles=smiles, targets=tgs))
    return drugs


def _parse_drugs_file(spec, target_ids: set[str]) -> list[DrugConfig]:
    """Parse the ``drugs_file:`` section: load parent compounds from an
    Excel/CSV spreadsheet instead of (or alongside) writing them inline.
    ``spec`` is either a bare path string (defaults for every column/target)
    or a mapping with ``path`` plus optional ``smiles_col``/``id_col``/
    ``name_col``/``targets``. This mirrors ``gnina-vs run --ligands`` /
    ``--smiles-col`` / ``--id-col`` so the same spreadsheet format works
    for both commands.
    """
    ctx = "drugs_file"
    if isinstance(spec, str):
        spec = {"path": spec}
    path = _require(spec, "path", ctx)
    if not os.path.exists(path):
        raise ConfigError(f"{ctx}: file not found: '{path}'.")

    smiles_col = spec.get("smiles_col", "SMILES")
    id_col = spec.get("id_col", "ID")
    name_col = spec.get("name_col", "Name")
    tgs = _check_targets(spec.get("targets") or sorted(target_ids), target_ids, ctx)

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".xlsx", ".xls"):
            table = pd.read_excel(path)
        elif ext == ".csv":
            table = pd.read_csv(path)
        else:
            raise ConfigError(f"{ctx}: unsupported file format '{ext}' for '{path}' "
                              f"(use .xlsx, .xls, or .csv).")
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"{ctx}: could not read '{path}': {e}")

    if smiles_col not in table.columns:
        raise ConfigError(
            f"{ctx}: '{path}' has no '{smiles_col}' column. Found columns: {list(table.columns)}. "
            f"Set drugs_file.smiles_col if your SMILES column is named differently."
        )
    has_id = id_col in table.columns
    has_name = name_col in table.columns

    drugs = []
    for row_idx, row in table.iterrows():
        row_ctx = f"{ctx} row {row_idx + 2}"  # +2: 1-based + header row, matches Excel row numbers
        raw_smiles = row[smiles_col]
        if pd.isna(raw_smiles) or not str(raw_smiles).strip():
            raise ConfigError(f"{row_ctx}: empty '{smiles_col}' value.")
        smiles = str(raw_smiles).strip()
        if Chem.MolFromSmiles(smiles) is None:
            raise ConfigError(f"{row_ctx}: invalid SMILES '{smiles}'.")

        if has_id and pd.notna(row[id_col]) and str(row[id_col]).strip():
            did = str(row[id_col]).strip()
        elif has_name and pd.notna(row[name_col]) and str(row[name_col]).strip():
            did = str(row[name_col]).strip()
        else:
            did = f"compound_{row_idx}"

        drugs.append(DrugConfig(id=did, smiles=smiles, targets=list(tgs)))

    if not drugs:
        raise ConfigError(f"{ctx}: '{path}' contains no rows.")
    return drugs


def load_config(path: str) -> RunConfig:
    """Load and validate a ``metabolite-run`` YAML config. Raises
    `ConfigError` (a `ValueError` subclass) naming the exact field at
    fault on any problem."""
    if not os.path.exists(path):
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as fh:
        try:
            raw = yaml.safe_load(fh) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Could not parse YAML: {e}")
    if not isinstance(raw, dict):
        raise ConfigError("Config file must be a YAML mapping at the top level.")

    outdir = raw.get("outdir", "results")
    seed = raw.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"'seed' must be an integer, got {seed!r}.")

    metabolism_cfg = _parse_metabolism(raw)
    docking_cfg = _parse_docking(raw)
    protein_cfg = _parse_protein(raw)
    targets = _parse_targets(raw)
    target_ids = {t.id for t in targets}

    drugs: list[DrugConfig] = []
    if raw.get("drugs"):
        drugs += _parse_inline_drugs(raw["drugs"], target_ids)
    if raw.get("drugs_file"):
        drugs += _parse_drugs_file(raw["drugs_file"], target_ids)
    if not drugs:
        raise ConfigError(
            "Config must define at least one drug, either inline under 'drugs' "
            "or loaded from a spreadsheet under 'drugs_file'."
        )
    seen_ids: dict[str, str] = {}
    for d in drugs:
        if d.id in seen_ids:
            raise ConfigError(
                f"Duplicate drug id '{d.id}' (appears in both 'drugs' and 'drugs_file', "
                f"or more than once within one of them). Compound ids must be unique; "
                f"set a distinct 'id' column value or inline id."
            )
        seen_ids[d.id] = d.smiles

    return RunConfig(outdir=outdir, seed=seed, protein=protein_cfg, metabolism=metabolism_cfg,
                     docking=docking_cfg, targets=targets, drugs=drugs)
