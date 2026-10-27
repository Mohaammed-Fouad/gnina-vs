"""
gnina_vs.annotate
=================
Phase 6: annotation columns only.

Adds, per compound:

* RDKit descriptors -- MW, Crippen logP, TPSA, HBD, HBA, rotatable bonds,
  formal charge -- and a Lipinski Rule-of-Five violation count.
* Brenk and PAINS structural alerts (``reactive_alert`` is set from Brenk).
* A heuristic Phase-II conjugate flag (glucuronide / sulfate ester).
* Optional ADMET-AI predictions (``admet_*`` columns) when the ``admet_ai``
  package is installed.
* A heuristic applicability-domain *proxy* for those predictions:
  ``physchem_out_of_range`` (descriptor bounding box), ``parent_tanimoto``
  (Morgan similarity to the compound's parent) and the combined
  ``admet_applicability_flag``.

Per the project's non-negotiable design principle, nothing here removes,
reorders or routes a compound; every function only adds columns.

Notes on the applicability-domain proxy
---------------------------------------
ADMET-AI does not expose its training-set domain, so this is a proxy, not
a true AD. Two known biases:

* Phase-II conjugates (glucuronides especially) add many fingerprint bits
  and routinely fall below the similarity threshold to their parent; read
  ``parent_tanimoto`` together with ``conjugate_flag``.
* Similarity to the parent says nothing about coverage of ADMET-AI's
  training data; a compound can be dissimilar to its parent yet
  well-represented there, and vice versa.

All thresholds below are research-triage heuristics, not validated cutoffs.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
ANNOTATION_COLUMNS = [
    "mol_weight_annot", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "formal_charge",
    "lipinski_violations",
    "brenk_alerts", "pains_alerts", "reactive_alert",
    "glucuronide_flag", "sulfate_flag", "conjugate_flag",
    "physchem_out_of_range", "parent_tanimoto",
    "admet_status", "admet_applicability_flag",
]
ADMET_PREFIX = "admet_"

# --------------------------------------------------------------------------- #
# Heuristic thresholds (triage only -- see module docstring)
# --------------------------------------------------------------------------- #
MW_RANGE = (100.0, 800.0)
LOGP_RANGE = (-5.0, 7.0)
TPSA_MAX = 200.0
PARENT_SIMILARITY_MIN = 0.4  # Morgan r=2, 2048 bits, Tanimoto

# Glucuronide: O-linked pyranose ring bearing a carboxylic acid (no stereo).
# Sulfate ester: -O-S(=O)(=O)-O-.
_GLUCURONIDE = Chem.MolFromSmarts("[OX2]C1OC(C(=O)[OX1,OX2H1])C(O)C(O)C1")
_SULFATE = Chem.MolFromSmarts("[OX2]S(=O)(=O)[OX2]")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_catalog(which: FilterCatalogParams.FilterCatalogs) -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(which)
    return FilterCatalog(params)


# Separate catalogs, so PAINS vs Brenk never depends on entry metadata.
_PAINS = _build_catalog(FilterCatalogParams.FilterCatalogs.PAINS)
_BRENK = _build_catalog(FilterCatalogParams.FilterCatalogs.BRENK)


def _alerts(catalog: FilterCatalog, mol: Chem.Mol) -> list[str]:
    return [m.GetDescription() for m in catalog.GetMatches(mol)]


def _mol_from_smiles(smiles) -> Chem.Mol | None:
    """Parse a SMILES, returning None for anything that isn't a parseable
    non-empty string (None, NaN, numbers, garbage)."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=None)
def _fingerprint(smiles: str):
    mol = _mol_from_smiles(smiles)
    return _MORGAN.GetFingerprint(mol) if mol is not None else None


def _lipinski_violations(mw: float, logp: float, hbd: int, hba: int) -> int:
    return int(sum([mw > 500, logp > 5, hbd > 5, hba > 10]))


def _physchem_out_of_range(mw: float, logp: float, tpsa: float) -> str | None:
    flags = []
    if not MW_RANGE[0] <= mw <= MW_RANGE[1]:
        flags.append(f"MW={mw:.0f}")
    if not LOGP_RANGE[0] <= logp <= LOGP_RANGE[1]:
        flags.append(f"logP={logp:.1f}")
    if tpsa > TPSA_MAX:
        flags.append(f"TPSA={tpsa:.0f}")
    return ";".join(flags) if flags else None


def parent_similarity(smiles, parent_smiles) -> float | None:
    """Morgan (r=2, 2048-bit) Tanimoto similarity between a compound and its
    parent. Returns None if either SMILES is missing or unparseable."""
    if not isinstance(smiles, str) or not isinstance(parent_smiles, str):
        return None
    fp, parent_fp = _fingerprint(smiles), _fingerprint(parent_smiles)
    if fp is None or parent_fp is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp, parent_fp))


def _admet_ai_available() -> bool:
    try:
        import admet_ai  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Per-molecule annotation
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _annotate_cached(smiles: str) -> tuple:
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return tuple((c, None) for c in ANNOTATION_COLUMNS)

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    brenk = _alerts(_BRENK, mol)
    pains = _alerts(_PAINS, mol)
    glucuronide = mol.HasSubstructMatch(_GLUCURONIDE)
    sulfate = mol.HasSubstructMatch(_SULFATE)

    row = {
        "mol_weight_annot": mw,
        "clogp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "lipinski_violations": _lipinski_violations(mw, logp, hbd, hba),
        "brenk_alerts": ";".join(brenk) if brenk else None,
        "pains_alerts": ";".join(pains) if pains else None,
        "reactive_alert": bool(brenk),
        "glucuronide_flag": bool(glucuronide),
        "sulfate_flag": bool(sulfate),
        "conjugate_flag": bool(glucuronide or sulfate),
        "physchem_out_of_range": _physchem_out_of_range(mw, logp, tpsa),
        # Row-context columns, filled by add_annotations():
        "parent_tanimoto": None,
        "admet_status": None,
        "admet_applicability_flag": None,
    }
    return tuple(row.items())


def annotate_smiles(smiles) -> dict:
    """Compute every per-molecule annotation for one SMILES.

    Returns a dict keyed by `ANNOTATION_COLUMNS`. Every value is None (never
    a fabricated default) if the SMILES doesn't parse. ``parent_tanimoto``,
    ``admet_status`` and ``admet_applicability_flag`` need table context
    (the parent, the batch ADMET run) and are left None here; they are
    filled by `add_annotations`.
    """
    if not isinstance(smiles, str):
        return {c: None for c in ANNOTATION_COLUMNS}
    return dict(_annotate_cached(smiles))


# --------------------------------------------------------------------------- #
# ADMET-AI (optional)
# --------------------------------------------------------------------------- #
def try_admet_ai(smiles_list: list[str]) -> pd.DataFrame | None:
    """Run ADMET-AI on ``smiles_list`` if the optional ``admet_ai`` package is
    installed. Returns a DataFrame indexed by SMILES, or None if the package
    is missing or prediction fails (the caller then skips gracefully)."""
    if not smiles_list:
        return None
    if not _admet_ai_available():
        logger.info("  [annotate] admet_ai not installed -- skipping ADMET predictions.")
        return None
    try:  # pragma: no cover - admet_ai is not a test dependency
        from admet_ai import ADMETModel
        preds = ADMETModel().predict(smiles=list(smiles_list))
        if not isinstance(preds, pd.DataFrame):
            raise TypeError(f"unexpected return type {type(preds).__name__}")
        if len(preds) != len(smiles_list):
            raise ValueError(f"{len(preds)} predictions for {len(smiles_list)} SMILES")
        preds.index = list(smiles_list)
        return preds
    except Exception as exc:  # pragma: no cover
        logger.warning("  [annotate] admet_ai prediction failed (%s) -- skipping.", exc)
        return None


# --------------------------------------------------------------------------- #
# Table-level annotation
# --------------------------------------------------------------------------- #
def _resolve_parent_smiles(df: pd.DataFrame, smiles_col: str) -> pd.Series:
    """Return each row's parent SMILES (NaN where unknown or the row is a parent).

    Uses an explicit ``parent_smiles`` column when present; otherwise derives
    it from ``parent_id`` + ``is_parent`` as produced by
    `gnina_vs.compare.build_comparison_table`.
    """
    empty = pd.Series(pd.NA, index=df.index, dtype=object)
    is_parent = (df["is_parent"] == True) if "is_parent" in df.columns else None  # noqa: E712

    if "parent_smiles" in df.columns:
        parents = df["parent_smiles"].astype(object)
    elif "parent_id" in df.columns and is_parent is not None:
        parent_rows = df.loc[is_parent & df[smiles_col].map(lambda s: _mol_from_smiles(s) is not None)]
        lookup = parent_rows.drop_duplicates("parent_id").set_index("parent_id")[smiles_col]
        parents = df["parent_id"].map(lookup).astype(object)
    else:
        return empty

    if is_parent is not None:
        parents = parents.where(~is_parent, pd.NA)  # a parent isn't compared with itself
    return parents


def _applicability_flag(admet_status, physchem, tanimoto, is_parent_row, parseable) -> str:
    if not parseable:
        return "not_evaluated: invalid_smiles"
    if admet_status != "computed":
        return "not_evaluated: no ADMET predictions"
    reasons = []
    if physchem:
        reasons.append(f"physchem({physchem})")
    if tanimoto is not None and tanimoto < PARENT_SIMILARITY_MIN:
        reasons.append(f"parent_tanimoto={tanimoto:.2f}")
    if reasons:
        return "outside_proxy_domain: " + " | ".join(reasons)
    if tanimoto is None and not is_parent_row:
        return "inside_proxy_domain: physchem only (no parent similarity)"
    return "inside_proxy_domain"


def add_annotations(df: pd.DataFrame, smiles_col: str = "smiles", run_admet: bool = True) -> pd.DataFrame:
    """Add every annotation column to ``df`` as new columns only.

    Never removes or reorders rows and never mutates the input. When
    ``run_admet`` is True and ``admet_ai`` is installed, prediction columns
    are added with the ``admet_`` prefix.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"add_annotations: '{smiles_col}' column not found in the given DataFrame.")

    df = df.copy()
    smiles = df[smiles_col]
    records = [annotate_smiles(s) for s in smiles]
    ann = pd.DataFrame.from_records(records, index=df.index, columns=ANNOTATION_COLUMNS)
    parseable = ann["mol_weight_annot"].notna()

    # -- parent similarity --
    parents = _resolve_parent_smiles(df, smiles_col)
    ann["parent_tanimoto"] = [parent_similarity(s, p) for s, p in zip(smiles, parents)]

    # -- ADMET-AI --
    unique_valid = list(dict.fromkeys(s for s, ok in zip(smiles, parseable) if ok))
    if not run_admet:
        status, preds = "skipped: disabled", None
    elif not _admet_ai_available():
        status, preds = "skipped: admet_ai not installed", None
    else:
        preds = try_admet_ai(unique_valid)
        status = "computed" if preds is not None else "failed: see log"
    ann["admet_status"] = [status if ok else "skipped: invalid_smiles" for ok in parseable]

    # -- applicability-domain proxy --
    is_parent = (df["is_parent"] == True) if "is_parent" in df.columns else pd.Series(False, index=df.index)  # noqa: E712
    ann["admet_applicability_flag"] = [
        _applicability_flag(st, pc, tn, ip, ok)
        for st, pc, tn, ip, ok in zip(
            ann["admet_status"], ann["physchem_out_of_range"], ann["parent_tanimoto"], is_parent, parseable
        )
    ]

    for col in ANNOTATION_COLUMNS:
        df[col] = ann[col]

    if preds is not None:
        for col in preds.columns:
            out_col = col if str(col).startswith(ADMET_PREFIX) else f"{ADMET_PREFIX}{col}"
            if out_col in ANNOTATION_COLUMNS:
                out_col = f"{out_col}_pred"
            df[out_col] = smiles.map(preds[col])

    return df
