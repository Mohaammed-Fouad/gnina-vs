"""
gnina_vs.compare
====================
Phase 5: build the per-target comparison table -- the core scientific
output answering "do predicted metabolites change the docking picture
compared with the parent alone?".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

COMPARISON_COLUMNS = [
    "target_id", "compound_id", "base_compound_id", "parent_id", "ID",
    "smiles", "status", "pose_quality_flag",
    "cnn_affinity", "vina_affinity", "cnn_score",
    "heavy_atoms", "mol_weight",
    "ligand_efficiency_cnn", "ligand_efficiency_vina",
    "delta_cnn_affinity_vs_parent", "delta_vina_affinity_vs_parent",
    "delta_cnn_affinity_vs_reference", "delta_vina_affinity_vs_reference",
    "rank_in_group", "rank_vs_parent_change",
    "is_parent", "depth", "phase", "reaction",
]


def _heavy_atoms(smiles) -> float | None:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return float(mol.GetNumHeavyAtoms()) if mol else None


def _mol_weight(smiles) -> float | None:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return float(Descriptors.MolWt(mol)) if mol else None


def _ligand_efficiency_vina(vina_affinity, heavy_atoms) -> float | None:
    """Vina/Gnina affinity is negative (more negative = better binding);
    LE = -affinity / heavy_atoms so a higher LE is still better."""
    if pd.isna(vina_affinity) or not heavy_atoms:
        return None
    return -float(vina_affinity) / heavy_atoms


def _ligand_efficiency_cnn(cnn_affinity, heavy_atoms) -> float | None:
    """CNNaffinity is a predicted pKd-like value (higher = better binding),
    the opposite sign convention from Vina affinity; LE = affinity /
    heavy_atoms, no negation."""
    if pd.isna(cnn_affinity) or not heavy_atoms:
        return None
    return float(cnn_affinity) / heavy_atoms


def build_comparison_table(
    results_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    target_id: str,
    reference_scores: dict | None = None,
) -> pd.DataFrame:
    """Join one target's docking results with the metabolite-aware
    ligand-prep table and compute the comparison metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of `gnina_vs.reporter.extract_results` for this target
        (``compound_id``, ``status``, ``vina_affinity``, ``cnn_affinity``,
        ``cnn_score``, ``pose_quality_flag``, ...).
    prepared_df : pd.DataFrame
        Output of `gnina_vs.ligand.prepare_ligands_extended` (``compound_id``,
        ``base_compound_id``, ``parent_id``, ``depth``, ``phase``,
        ``reaction``, ``is_parent``, ...), filtered or not -- only rows
        matching ``results_df`` are used.
    target_id : str
        This target's id, stamped onto every row.
    reference_scores : dict | None
        ``{"cnn_affinity": ..., "vina_affinity": ...}`` for this target's
        reference ligand (see `gnina_vs.metabolite_pipeline`), or None if
        no reference ligand was configured/docked -- deltas vs reference
        are then left as NaN, not silently zero.

    Returns
    -------
    pd.DataFrame
        One row per docked compound, with every column in
        `COMPARISON_COLUMNS`. NaN/failed docking is kept and flagged
        (``status``, ``pose_quality_flag``), never dropped, and no
        division by a zero/missing heavy-atom count is ever attempted.
    """
    df = results_df.merge(prepared_df, on="compound_id", how="left", suffixes=("", "_prep"))
    df["target_id"] = target_id

    df["heavy_atoms"] = df["smiles"].apply(_heavy_atoms)
    df["mol_weight"] = df["smiles"].apply(_mol_weight)
    df["ligand_efficiency_vina"] = [
        _ligand_efficiency_vina(a, h) for a, h in zip(df["vina_affinity"], df["heavy_atoms"])
    ]
    df["ligand_efficiency_cnn"] = [
        _ligand_efficiency_cnn(a, h) for a, h in zip(df["cnn_affinity"], df["heavy_atoms"])
    ]

    # -- delta vs parent, per parent_id group --
    df["delta_cnn_affinity_vs_parent"] = np.nan
    df["delta_vina_affinity_vs_parent"] = np.nan
    if "parent_id" in df.columns:
        for parent_id, group in df.groupby("parent_id", dropna=True):
            parent_rows = group[group.get("is_parent", False) == True]  # noqa: E712
            if parent_rows.empty:
                logger.warning(
                    "  [compare] target '%s': no parent row found for parent_id '%s' "
                    "-- delta_vs_parent left as NaN for its metabolites.", target_id, parent_id,
                )
                continue
            if len(parent_rows) > 1:
                logger.warning(
                    "  [compare] target '%s': parent_id '%s' has %d parent rows "
                    "(likely enumerated stereoisomers of the parent) -- using the "
                    "best-scoring one for delta_vs_parent.", target_id, parent_id, len(parent_rows),
                )
                parent_rows = parent_rows.sort_values(
                    by="cnn_affinity" if parent_rows["cnn_affinity"].notna().any() else "vina_affinity",
                    ascending=not parent_rows["cnn_affinity"].notna().any(),
                    na_position="last",
                )
            parent_cnn = parent_rows.iloc[0]["cnn_affinity"]
            parent_vina = parent_rows.iloc[0]["vina_affinity"]
            idx = group.index
            df.loc[idx, "delta_cnn_affinity_vs_parent"] = group["cnn_affinity"] - parent_cnn
            df.loc[idx, "delta_vina_affinity_vs_parent"] = group["vina_affinity"] - parent_vina

    # -- delta vs reference (single reference score per target, if any) --
    if reference_scores:
        ref_cnn = reference_scores.get("cnn_affinity")
        ref_vina = reference_scores.get("vina_affinity")
        df["delta_cnn_affinity_vs_reference"] = (df["cnn_affinity"] - ref_cnn) if ref_cnn is not None else np.nan
        df["delta_vina_affinity_vs_reference"] = (df["vina_affinity"] - ref_vina) if ref_vina is not None else np.nan
    else:
        df["delta_cnn_affinity_vs_reference"] = np.nan
        df["delta_vina_affinity_vs_reference"] = np.nan

    # -- rank within each (parent_id, target) group --
    df["rank_in_group"] = np.nan
    df["rank_vs_parent_change"] = np.nan
    have_cnn = df["cnn_affinity"].notna().any()
    rank_col = "cnn_affinity" if have_cnn else "vina_affinity"
    ascending = not have_cnn  # cnn: higher better (rank desc); vina: lower better (rank asc)
    if "parent_id" in df.columns:
        for parent_id, group in df.groupby("parent_id", dropna=True):
            scored = group[group[rank_col].notna()]
            if scored.empty:
                continue
            order = scored.sort_values(by=rank_col, ascending=ascending).index
            ranks = {idx: r for r, idx in enumerate(order, start=1)}
            for idx, r in ranks.items():
                df.loc[idx, "rank_in_group"] = r
            parent_idx = group[group.get("is_parent", False) == True]  # noqa: E712
            if not parent_idx.empty and parent_idx.index[0] in ranks:
                parent_rank = ranks[parent_idx.index[0]]
                for idx, r in ranks.items():
                    df.loc[idx, "rank_vs_parent_change"] = r - parent_rank

    for col in COMPARISON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[COMPARISON_COLUMNS]
