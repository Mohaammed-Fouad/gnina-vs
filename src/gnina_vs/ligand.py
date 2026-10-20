"""
gnina_vs.ligand
====================
Step 1 of the pipeline: load a compound table (Excel or CSV), strip salts /
counter-ions, generate 3-D conformers with ETKDGv3, minimize with MMFF94, and
write everything to a single multi-molecule SDF ready for docking.
"""

from __future__ import annotations

import logging
import os
import re

import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(text: str, max_len: int = 60) -> str:
    """Reduce arbitrary text to a filesystem-safe token (letters, digits,
    ``.``, ``_``, ``-``). Never returns an empty string."""
    cleaned = _UNSAFE_CHARS.sub("_", str(text)).strip("._-")[:max_len].strip("._-")
    return cleaned or "cmpd"


def make_compound_id(mol_name: str, index) -> str:
    """Unique, filesystem-safe identifier used in every output filename.

    The (sanitised) input row index is always appended, so two compounds that
    share a display name can never collide: ``"<safe-name>_<index>"``.
    """
    return f"{safe_filename(mol_name)}_{safe_filename(str(index), max_len=20)}"


def strip_to_largest_fragment(mol: Chem.Mol) -> tuple[Chem.Mol, list[str]]:
    """Keep the largest (organic-preferred) fragment of a multi-component
    molecule (salts, counter-ions, solvents). Returns ``(mol, removed_smiles)``."""
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) <= 1:
        return mol, []
    chosen = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    chosen_smi = Chem.MolToSmiles(chosen)
    removed: list[str] = []
    skipped_once = False
    for frag in frags:
        smi = Chem.MolToSmiles(frag)
        if not skipped_once and smi == chosen_smi:
            skipped_once = True
            continue
        removed.append(smi)
    return chosen, removed


def _undefined_stereocentres(mol: Chem.Mol) -> list[int]:
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False)
    return [idx for idx, label in centers if label == "?"]


def load_compound_table(input_path: str) -> pd.DataFrame:
    """Load a compound table from ``.xlsx``/``.xls`` or ``.csv``. Shared by
    `prepare_ligands` and `gnina_vs.metabolism.generate_metabolites_for_table`
    (``gnina-vs run --metabolite``) so both read the same spreadsheet the
    same way."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(input_path)
    if ext == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(
        f"Unsupported ligand input format '{ext}' for '{input_path}'. "
        f"Use a .xlsx, .xls, or .csv file."
    )


# Backward-compatible alias (this was private; kept in case anything imports it).
_load_table = load_compound_table


def prepare_ligands(
    input_path: str,
    output_sdf: str,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    name_col: str = "Name",
    seed: int = 42,
) -> list[dict]:
    """Read SMILES from an Excel/CSV file, strip salts (largest fragment is
    kept), generate 3-D conformers, minimize with MMFF94, and write all
    successful structures to a multi-molecule SDF.

    Parameters
    ----------
    input_path : str
        Path to a ``.xlsx``/``.xls``/``.csv`` file with at least a SMILES
        column.
    output_sdf : str
        Destination multi-molecule SDF path.
    smiles_col, id_col, name_col : str
        Column names for the SMILES string, compound identifier, and
        (optional) display name. ``id_col`` and ``name_col`` are optional --
        if missing, the row index is used instead.
    seed : int
        Random seed for ETKDGv3 conformer embedding (reproducibility).

    Returns
    -------
    list[dict]
        Each entry: 'index', 'smiles' (the structure actually docked),
        'input_smiles', 'removed_fragments', 'mol_name', 'ligand_id' (value
        of ``id_col``) and 'compound_id' (unique, filesystem-safe; used in
        all output filenames).
    """
    logger.info("=" * 60)
    logger.info("STEP 1 -- Ligand Preparation")
    logger.info("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ligand input file not found: {input_path}")

    df = load_compound_table(input_path)

    if smiles_col not in df.columns:
        raise ValueError(
            f"Ligand input file must contain a '{smiles_col}' column. "
            f"Found columns: {list(df.columns)}"
        )

    has_id_col = id_col in df.columns
    if not has_id_col:
        logger.warning(
            "  '%s' column not found in '%s' -- results will use row "
            "index as identifier.", id_col, input_path,
        )
    has_name_col = name_col in df.columns

    logger.info("Loaded %d compounds from '%s'.", len(df), input_path)

    prepared_mols: list[dict] = []

    with SDWriter(output_sdf) as writer:
        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Preparing ligands",
            unit="mol",
            colour="cyan",
        ):
            input_smiles = str(row[smiles_col]).strip()
            raw_name = row[name_col] if has_name_col else None
            mol_name = (f"compound_{idx}" if raw_name is None or pd.isna(raw_name)
                        or not str(raw_name).strip() else str(raw_name).strip())
            lig_id = str(row[id_col]) if has_id_col else str(idx)
            compound_id = make_compound_id(mol_name, idx)

            mol = Chem.MolFromSmiles(input_smiles)
            if mol is None:
                logger.warning("  [idx=%s] Invalid SMILES -- skipping: %s", idx, input_smiles)
                continue

            # -- strip salts / counter-ions / solvents: dock ONE molecule --
            mol, removed = strip_to_largest_fragment(mol)
            if removed:
                logger.info(
                    "  [idx=%s] '%s': removed %d fragment(s) (%s); kept the largest.",
                    idx, mol_name, len(removed), ", ".join(removed),
                )
            smiles = Chem.MolToSmiles(mol)

            # -- undefined stereochemistry: RDKit would pick ONE isomer arbitrarily --
            undefined = _undefined_stereocentres(mol)
            if undefined:
                logger.warning(
                    "  [idx=%s] '%s' has %d undefined stereocentre(s) (atom index %s): "
                    "only one arbitrary stereoisomer will be embedded and docked. "
                    "Specify stereochemistry in the SMILES, or supply each isomer as "
                    "its own row.", idx, mol_name, len(undefined),
                    ",".join(map(str, undefined)),
                )

            mol.SetProp("_Name", mol_name)

            mol_h = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            result = AllChem.EmbedMolecule(mol_h, params)

            if result == -1:
                logger.warning("  [idx=%s] Embedding failed -- skipping: %s", idx, mol_name)
                continue

            ff_result = AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94")
            if ff_result == -1:
                logger.warning(
                    "  [idx=%s] MMFF94 minimisation failed -- skipping: %s",
                    idx, mol_name,
                )
                continue

            mol_h.SetProp("SMILES", smiles)            # structure actually docked
            mol_h.SetProp("InputSMILES", input_smiles)
            mol_h.SetProp("MolIndex", str(idx))
            mol_h.SetProp("LigandID", lig_id)          # carry ID into SDF
            mol_h.SetProp("CompoundID", compound_id)   # unique, filesystem-safe

            writer.write(mol_h)
            prepared_mols.append(
                {"index": idx, "smiles": smiles, "input_smiles": input_smiles,
                 "removed_fragments": removed, "mol_name": mol_name,
                 "ligand_id": lig_id, "compound_id": compound_id}
            )

    logger.info(
        "Ligand preparation complete -- %d/%d molecules saved to '%s'.",
        len(prepared_mols), len(df), output_sdf,
    )
    if not prepared_mols:
        raise ValueError(
            "No ligands were prepared successfully -- check the input "
            "file's SMILES column for valid structures."
        )
    return prepared_mols


# =========================================================================== #
# Phase 2: metabolite-aware ligand preparation
# =========================================================================== #
# Used by gnina_vs.metabolite_pipeline (the `gnina-vs metabolite-run`
# command) rather than the original `run`/`recover` commands, so the
# original `prepare_ligands` above is untouched and stays backward
# compatible.

DEFAULT_MAX_STEREOISOMERS = 8


def standardize_mol(mol: Chem.Mol) -> tuple[Chem.Mol, list[str]]:
    """Strip salts (largest fragment kept), neutralize where possible, and
    canonicalize the tautomer conservatively (RDKit's default scoring
    rules). Returns ``(mol, notes)`` where ``notes`` lists what changed,
    for logging -- nothing here is applied silently.
    """
    notes: list[str] = []
    mol, removed = strip_to_largest_fragment(mol)
    if removed:
        notes.append(f"removed fragment(s): {', '.join(removed)}")

    before = Chem.MolToSmiles(mol)
    try:
        neutral = rdMolStandardize.Uncharger().uncharge(mol)
        if Chem.MolToSmiles(neutral) != before:
            notes.append("neutralized")
            mol = neutral
    except Exception as exc:
        logger.debug("  Neutralization skipped: %s", exc)

    before = Chem.MolToSmiles(mol)
    try:
        canon_taut = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        if Chem.MolToSmiles(canon_taut) != before:
            notes.append("tautomer canonicalized")
            mol = canon_taut
    except Exception as exc:
        logger.debug("  Tautomer canonicalization skipped: %s", exc)

    return mol, notes


def enumerate_stereoisomers(mol: Chem.Mol, max_isomers: int = DEFAULT_MAX_STEREOISOMERS) -> list[Chem.Mol]:
    """Enumerate stereoisomers for undefined stereocentres only -- defined
    stereochemistry in the input is always kept as-is, never overwritten.
    Returns ``[mol]`` unchanged if there is nothing undefined to enumerate.
    """
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

    if not _undefined_stereocentres(mol):
        return [mol]
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=max_isomers)
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    return isomers if isomers else [mol]


def protonate(mol: Chem.Mol, ph: float = 7.4) -> tuple[Chem.Mol, str]:
    """Protonate ``mol`` for ``ph`` using Dimorphite-DL if it's importable.

    Returns ``(mol, protonation_state)`` where ``protonation_state``
    documents what was actually done -- callers must not assume a specific
    protonation state was applied just because this function ran; check
    the returned label (e.g. it will read ``"unmodified (Dimorphite-DL
    unavailable)"`` when the optional dependency isn't installed, per the
    project's "optional dependencies fail gracefully" rule).
    """
    try:
        from dimorphite_dl import DimorphiteDL
    except ImportError:
        return mol, "unmodified (Dimorphite-DL unavailable)"

    try:
        dl = DimorphiteDL(min_ph=ph, max_ph=ph, pka_precision=0.0)
        protonated_smiles = dl.protonate(Chem.MolToSmiles(mol))
        if protonated_smiles:
            new_mol = Chem.MolFromSmiles(protonated_smiles[0])
            if new_mol is not None:
                return new_mol, f"dimorphite-dl@pH{ph}"
        return mol, f"dimorphite-dl@pH{ph} (no change)"
    except Exception as exc:
        logger.warning("  Dimorphite-DL protonation failed (%s); using the unmodified structure.", exc)
        return mol, "unmodified (Dimorphite-DL error)"


def prepare_ligands_extended(
    metabolite_df: pd.DataFrame,
    output_sdf: str,
    seed: int = 42,
    max_stereoisomers: int = DEFAULT_MAX_STEREOISOMERS,
    ph: float = 7.4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metabolite-aware ligand preparation: standardize (salts, charge,
    tautomer), enumerate undefined stereocentres (each isomer docked
    separately, never picked arbitrarily), protonate at ``ph``, embed with
    ETKDGv3 + MMFF94, and write everything to one multi-molecule SDF.

    Unlike `prepare_ligands`, every row of a stereoisomer set gets its own
    entry (``<compound_id>_s1``, ``_s2``, ...), and embedding/minimization
    failures are returned as a table rather than only logged and dropped.

    Parameters
    ----------
    metabolite_df : pd.DataFrame
        A table with (at least) the columns in
        `gnina_vs.metabolism.STANDARD_COLUMNS`: ``compound_id``,
        ``parent_id``, ``smiles``, and optionally ``depth``, ``phase``,
        ``reaction``, ``is_parent`` (carried through to the returned table
        for downstream comparison/annotation).
    output_sdf : str
        Destination multi-molecule SDF path.
    seed : int
        ETKDGv3 random seed.
    max_stereoisomers : int
        Cap on how many stereoisomers are enumerated (and docked) per
        compound with undefined stereocentres.
    ph : float
        Target pH for protonation (see `protonate`).

    Returns
    -------
    (prepared_df, failures_df) : tuple[pd.DataFrame, pd.DataFrame]
        ``prepared_df`` has one row per SDF entry actually written:
        ``compound_id`` (stereo-suffixed if enumerated), ``base_compound_id``,
        ``parent_id``, ``stereo_id``, ``protonation_state``, ``smiles``
        (the exact structure docked), plus ``depth``/``phase``/``reaction``/
        ``is_parent`` carried from ``metabolite_df``.
        ``failures_df`` has one row per compound that could not be
        prepared: ``compound_id``, ``parent_id``, ``stage``
        (``parse``/``embed``/``minimize``), ``reason``.
    """
    logger.info("=" * 60)
    logger.info("STEP 1 (metabolite-aware) -- Ligand Preparation")
    logger.info("=" * 60)

    prepared: list[dict] = []
    failures: list[dict] = []

    with SDWriter(output_sdf) as writer:
        for _, row in tqdm(metabolite_df.iterrows(), total=len(metabolite_df),
                           desc="Preparing ligands (metabolite-aware)", unit="cmpd", colour="cyan"):
            base_id = str(row["compound_id"])
            parent_id = str(row["parent_id"])
            input_smiles = str(row["smiles"])

            mol = Chem.MolFromSmiles(input_smiles)
            if mol is None:
                logger.warning("  [%s] Invalid SMILES -- skipping: %s", base_id, input_smiles)
                failures.append({"compound_id": base_id, "parent_id": parent_id,
                                 "stage": "parse", "reason": "invalid SMILES"})
                continue

            mol, notes = standardize_mol(mol)
            if notes:
                logger.info("  [%s] standardization: %s", base_id, "; ".join(notes))

            undefined = _undefined_stereocentres(mol)
            isomers = enumerate_stereoisomers(mol, max_stereoisomers)
            if len(isomers) > 1:
                logger.info("  [%s] %d undefined stereocentre(s) -> enumerating %d isomer(s) "
                           "(cap %d).", base_id, len(undefined), len(isomers), max_stereoisomers)

            for s_idx, iso in enumerate(isomers, start=1):
                stereo_id = f"{base_id}_s{s_idx}" if len(isomers) > 1 else base_id

                prot_mol, prot_state = protonate(iso, ph)
                docked_smiles = Chem.MolToSmiles(prot_mol)

                mol_h = Chem.AddHs(prot_mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                if AllChem.EmbedMolecule(mol_h, params) == -1:
                    logger.warning("  [%s] Embedding failed -- recorded as a failure, not silently dropped.", stereo_id)
                    failures.append({"compound_id": stereo_id, "parent_id": parent_id,
                                     "stage": "embed", "reason": "ETKDG embedding failed"})
                    continue
                if AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94") == -1:
                    logger.warning("  [%s] MMFF94 minimization failed -- recorded as a failure.", stereo_id)
                    failures.append({"compound_id": stereo_id, "parent_id": parent_id,
                                     "stage": "minimize", "reason": "MMFF94 minimization failed"})
                    continue

                mol_h.SetProp("_Name", stereo_id)
                mol_h.SetProp("CompoundID", stereo_id)
                mol_h.SetProp("LigandID", stereo_id)
                mol_h.SetProp("ParentID", parent_id)
                mol_h.SetProp("StereoID", str(s_idx))
                mol_h.SetProp("ProtonationState", prot_state)
                mol_h.SetProp("SMILES", docked_smiles)
                mol_h.SetProp("InputSMILES", input_smiles)
                mol_h.SetProp("MolIndex", stereo_id)
                writer.write(mol_h)

                prepared.append({
                    "compound_id": stereo_id,
                    "base_compound_id": base_id,
                    "parent_id": parent_id,
                    "stereo_id": s_idx,
                    "protonation_state": prot_state,
                    "smiles": docked_smiles,
                    "depth": row.get("depth"),
                    "phase": row.get("phase"),
                    "reaction": row.get("reaction"),
                    "is_parent": bool(row.get("is_parent", False)),
                })

    prepared_df = pd.DataFrame(prepared)
    failures_df = pd.DataFrame(failures, columns=["compound_id", "parent_id", "stage", "reason"])

    logger.info(
        "Ligand preparation (metabolite-aware) complete -- %d entries written to '%s' "
        "(%d compound(s) failed).",
        len(prepared_df), output_sdf, len(failures_df),
    )
    if prepared_df.empty:
        raise ValueError(
            "No ligands were prepared successfully in the metabolite-aware pipeline -- "
            "check the input metabolite table's SMILES."
        )
    return prepared_df, failures_df
