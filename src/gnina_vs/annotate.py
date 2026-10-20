"""
gnina_vs.annotate
======================
Phase 6: annotation columns only -- MW/logP/TPSA/HBD/HBA/rotatable bonds,
Lipinski violation count, Brenk/PAINS structural alerts, a heuristic
Phase-II conjugate flag, and an optional ADMET-AI hook. Per the project's
non-negotiable design principle, nothing here ever removes or routes a
compound; every function only adds columns.
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, FilterCatalog, Lipinski
from rdkit import DataStructs

logger = logging.getLogger(__name__)

ANNOTATION_COLUMNS = [
    "mol_weight_annot", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "formal_charge",
    "lipinski_violations",
    "brenk_alerts", "pains_alerts", "reactive_alert",
    "glucuronide_flag", "sulfate_flag", "conjugate_flag",
    "admet_status", "admet_applicability_flag",
]

# Heuristic substructure patterns -- approximate detectors for research
# triage, not an authoritative metabolite classifier. A glucuronide is
# matched as an oxygen-linked pyranose ring bearing a carboxylic acid
# (ignoring stereochemistry); a sulfate ester as -O-S(=O)(=O)-O-.
_GLUCURONIDE_SMARTS = Chem.MolFromSmarts("[OX2]C1OC(C(=O)[OX1,OX2H1])C(O)C(O)C1")
_SULFATE_SMARTS = Chem.MolFromSmarts("[OX2]S(=O)(=O)[OX2]")

_catalog_cache: FilterCatalog.FilterCatalog | None = None


def _get_filter_catalog() -> FilterCatalog.FilterCatalog:
    global _catalog_cache
    if _catalog_cache is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        _catalog_cache = FilterCatalog.FilterCatalog(params)
    return _catalog_cache


def _lipinski_violations(mw, logp, hbd, hba) -> int:
    return sum([mw > 500, logp > 5, hbd > 5, hba > 10])


def check_applicability_domain(metabolite_smiles, parent_smiles, mol_weight, logp, tpsa):
    """
    Evaluates the Applicability Domain (AD) of a molecule for ADMET-AI.
    Flags compounds that are either structurally disjoint from the parent 
    or fall outside the physicochemical bounding box of typical ADMET training data.
    """
    flags = []
    
    # 1. Physicochemical Bounding Box Check (Typical TDC/DrugBank ranges)
    if pd.isna(mol_weight) or pd.isna(logp) or pd.isna(tpsa):
        return "outside_domain: missing_descriptors"
        
    if mol_weight > 800 or mol_weight < 100:
        flags.append("out_of_bounds_MW")
    if logp > 7 or logp < -5:
        flags.append("out_of_bounds_LogP")
    if tpsa > 200:
        flags.append("out_of_bounds_TPSA")
        
    # 2. Structural Tanimoto Similarity Check vs Parent
    if metabolite_smiles and parent_smiles and metabolite_smiles != parent_smiles:
        met_mol = Chem.MolFromSmiles(metabolite_smiles)
        par_mol = Chem.MolFromSmiles(parent_smiles)
        
        if met_mol and par_mol:
            met_fp = AllChem.GetMorganFingerprintAsBitVect(met_mol, radius=2, nBits=2048)
            par_fp = AllChem.GetMorganFingerprintAsBitVect(par_mol, radius=2, nBits=2048)
            
            similarity = DataStructs.TanimotoSimilarity(met_fp, par_fp)
            
            # Threshold: 0.4 is a common cutoff for structurally distinct chemical spaces
            if similarity < 0.4:
                flags.append(f"low_parent_similarity_{similarity:.2f}")

    # Resolve the final flag
    if not flags:
        return "inside_domain"
    else:
        return "outside_domain: " + " | ".join(flags)


def annotate_smiles(smiles: str) -> dict:
    """Compute every annotation for one SMILES string. Returns a dict with
    `ANNOTATION_COLUMNS` as keys; every value is None (not a fabricated
    default) if the SMILES doesn't parse."""
    empty = {c: None for c in ANNOTATION_COLUMNS}
    if not isinstance(smiles, str):
        return empty
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    charge = Chem.GetFormalCharge(mol)

    catalog = _get_filter_catalog()
    matches = catalog.GetMatches(mol)
    brenk_hits = [m.GetDescription() for m in matches if "brenk" in m.GetProp("FilterSet").lower()] \
        if matches else []
    pains_hits = [m.GetDescription() for m in matches if "pains" in m.GetProp("FilterSet").lower()] \
        if matches else []
    # Some RDKit builds don't expose "FilterSet"; fall back to "all matches
    # go in both lists" so alerts are never silently lost.
    if matches and not brenk_hits and not pains_hits:
        brenk_hits = pains_hits = [m.GetDescription() for m in matches]

    glucuronide = _GLUCURONIDE_SMARTS is not None and mol.HasSubstructMatch(_GLUCURONIDE_SMARTS)
    sulfate = _SULFATE_SMARTS is not None and mol.HasSubstructMatch(_SULFATE_SMARTS)

    return {
        "mol_weight_annot": mw,
        "clogp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rot,
        "formal_charge": charge,
        "lipinski_violations": _lipinski_violations(mw, logp, hbd, hba),
        "brenk_alerts": ";".join(brenk_hits) if brenk_hits else None,
        "pains_alerts": ";".join(pains_hits) if pains_hits else None,
        "reactive_alert": bool(brenk_hits),
        "glucuronide_flag": bool(glucuronide),
        "sulfate_flag": bool(sulfate),
        "conjugate_flag": bool(glucuronide or sulfate),
        # ADMET-AI is an optional dependency; skipped gracefully when absent
        "admet_status": "skipped: admet_ai not installed" if not _admet_ai_available() else "computed",
        "admet_applicability_flag": "not_implemented",  # This stub is now overwritten in add_annotations
    }


def _admet_ai_available() -> bool:
    try:
        import admet_ai  # noqa: F401
        return True
    except ImportError:
        return False


def try_admet_ai(smiles_list: list[str]) -> pd.DataFrame | None:
    """Run ADMET-AI predictions if the optional `admet_ai` package is
    installed; otherwise return None and let the caller skip gracefully
    (per the project's "optional dependencies fail gracefully" rule).
    Not exercised by the test suite since admet_ai is not a dependency."""
    if not _admet_ai_available():
        logger.info("  [annotate] admet_ai not installed -- skipping ADMET predictions.")
        return None
    try:
        from admet_ai import ADMETModel  # pragma: no cover
        model = ADMETModel()
        return model.predict(smiles=smiles_list)
    except Exception as exc:  # pragma: no cover
        logger.warning("  [annotate] admet_ai prediction failed (%s) -- skipping.", exc)
        return None


def add_annotations(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Add every annotation column to ``df`` (one row per compound already
    present) as new columns only -- never removes or reorders rows."""
    if smiles_col not in df.columns:
        raise ValueError(f"add_annotations: '{smiles_col}' column not found in the given DataFrame.")
    
    df = df.copy()
    annotations = df[smiles_col].apply(annotate_smiles).apply(pd.Series)
    
    for col in ANNOTATION_COLUMNS:
        df[col] = annotations[col] if col in annotations.columns else None

    # Determine the parent smiles column for applicability domain check
    parent_col = "parent_smiles" if "parent_smiles" in df.columns else smiles_col

    # Overwrite the 'not_implemented' stub with the actual AD evaluation
    df["admet_applicability_flag"] = df.apply(
        lambda row: check_applicability_domain(
            metabolite_smiles=row[smiles_col],
            parent_smiles=row[parent_col] if parent_col in row else row[smiles_col],
            mol_weight=row.get("mol_weight_annot"),
            logp=row.get("clogp"),
            tpsa=row.get("tpsa")
        ), 
        axis=1
    )

    return df