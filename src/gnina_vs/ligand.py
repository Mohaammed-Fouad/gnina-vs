"""
gnina_vs.ligand
====================
Step 1 of the pipeline: load a compound table (Excel or CSV), generate 3-D
conformers with ETKDGv3, minimize with MMFF94, and write everything to a
single multi-molecule SDF ready for docking.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter

logger = logging.getLogger(__name__)


def _load_table(input_path: str) -> pd.DataFrame:
    """Load a compound table from ``.xlsx``/``.xls`` or ``.csv``."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(input_path)
    if ext == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(
        f"Unsupported ligand input format '{ext}' for '{input_path}'. "
        f"Use a .xlsx, .xls, or .csv file."
    )


def prepare_ligands(
    input_path: str,
    output_sdf: str,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    name_col: str = "Name",
    seed: int = 42,
) -> list[dict]:
    """Read SMILES from an Excel/CSV file, generate 3-D conformers, minimize
    with MMFF94, and write all successful structures to a multi-molecule SDF.

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
        Each entry: 'index', 'smiles', 'mol_name', and 'ligand_id' (the
        value of ``id_col``, if present).
    """
    logger.info("=" * 60)
    logger.info("STEP 1 -- Ligand Preparation")
    logger.info("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ligand input file not found: {input_path}")

    df = _load_table(input_path)

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
            smiles = str(row[smiles_col]).strip()
            mol_name = str(row[name_col]) if has_name_col else f"compound_{idx}"
            lig_id = str(row[id_col]) if has_id_col else str(idx)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning("  [idx=%s] Invalid SMILES -- skipping: %s", idx, smiles)
                continue

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

            mol_h.SetProp("SMILES", smiles)
            mol_h.SetProp("MolIndex", str(idx))
            mol_h.SetProp("LigandID", lig_id)  # carry ID into SDF

            writer.write(mol_h)
            prepared_mols.append(
                {"index": idx, "smiles": smiles, "mol_name": mol_name,
                 "ligand_id": lig_id}
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
