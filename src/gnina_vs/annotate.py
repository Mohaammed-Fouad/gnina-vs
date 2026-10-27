"""
gnina_vs.annotate
=================
Phase 6: annotation columns only.

Adds, per compound:

* RDKit descriptors -- MW, Crippen logP, TPSA, HBD, HBA, rotatable bonds,
  formal charge -- and a Lipinski Rule-of-Five violation count.
* Structural alerts as **per-family counts** across RDKit's filter
  catalogs (Brenk, PAINS A/B/C, NIH, ZINC, the ChEMBL sets), plus the
  matched names for Brenk and PAINS.
* A heuristic Phase-II conjugate flag (glucuronide / sulfate ester).
* A heuristic exposure-plausibility tier (see `exposure_plausibility`).
* Optional ADMET-AI predictions, with a real applicability-domain check
  against the model's training chemistry (see `gnina_vs.applicability`).

Per the project's non-negotiable design principle, nothing here removes,
reorders or routes a compound; every function only adds columns.

Two things this module deliberately no longer does
--------------------------------------------------
**There is no `reactive_alert` boolean.** Collapsing a filter-catalog hit
into one "reactive" flag is a category error on the filters' own terms.
The PAINS authors state that black-box use of these filters "is fraught
with danger", that a match does not imply interference, and that absence
of a match does not imply cleanliness -- epoxides and aziridines were
excluded from the library the filters were derived from. The families are
also wildly asymmetric: a handful of Family A substructures account for
most recognised compounds while hundreds of Family C substructures account
for a small fraction, so one boolean discards the only usable signal.
Roughly 5% of approved drugs match a PAINS filter. Counts per family are
reported instead; interpretation is the reader's.

**There is no similarity-to-parent applicability domain.** That lived here
and was wrong; see `gnina_vs.applicability` for what replaced it and why.
`parent_tanimoto` is still reported, because structural distance from the
parent is genuinely interesting for a metabolite -- it is simply not an
applicability domain.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from gnina_vs import applicability

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Alert catalogs
# --------------------------------------------------------------------------- #
# Each entry becomes one `alert_<key>_n` count column. PAINS is split into
# its three families deliberately: they differ by more than an order of
# magnitude in how much of the recognised-compound mass they carry, so a
# combined count is much less informative than three separate ones.
_CATALOG_SPEC = {
    "brenk": FilterCatalogParams.FilterCatalogs.BRENK,
    "pains_a": FilterCatalogParams.FilterCatalogs.PAINS_A,
    "pains_b": FilterCatalogParams.FilterCatalogs.PAINS_B,
    "pains_c": FilterCatalogParams.FilterCatalogs.PAINS_C,
    "nih": FilterCatalogParams.FilterCatalogs.NIH,
    "zinc": FilterCatalogParams.FilterCatalogs.ZINC,
    "chembl_dundee": FilterCatalogParams.FilterCatalogs.CHEMBL_Dundee,
    "chembl_glaxo": FilterCatalogParams.FilterCatalogs.CHEMBL_Glaxo,
    "chembl_bms": FilterCatalogParams.FilterCatalogs.CHEMBL_BMS,
    "chembl_mlsmr": FilterCatalogParams.FilterCatalogs.CHEMBL_MLSMR,
    "chembl_inpharmatica": FilterCatalogParams.FilterCatalogs.CHEMBL_Inpharmatica,
    "chembl_surechembl": FilterCatalogParams.FilterCatalogs.CHEMBL_SureChEMBL,
    "chembl_lint": FilterCatalogParams.FilterCatalogs.CHEMBL_LINT,
}
ALERT_COUNT_COLUMNS = [f"alert_{key}_n" for key in _CATALOG_SPEC]

DESCRIPTOR_COLUMNS = [
    "mol_weight_annot", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "formal_charge",
    "lipinski_violations",
]

ANNOTATION_COLUMNS = (
    DESCRIPTOR_COLUMNS
    + ALERT_COUNT_COLUMNS
    + [
        "alert_total_n", "brenk_alerts", "pains_alerts",
        "glucuronide_flag", "sulfate_flag", "conjugate_flag",
        "physchem_out_of_range",
        "exposure_plausibility", "exposure_basis",
        "parent_tanimoto",
        "admet_status",
    ]
    + applicability.AD_COLUMNS
)
ADMET_PREFIX = "admet_"

# --------------------------------------------------------------------------- #
# Heuristic thresholds -- research triage, not validated cutoffs
# --------------------------------------------------------------------------- #
MW_RANGE = (100.0, 800.0)
LOGP_RANGE = (-5.0, 7.0)
TPSA_MAX = 200.0

# Exposure heuristic. Each threshold is a conventional drug-likeness
# landmark, not a validated predictor of tissue exposure. See
# `exposure_plausibility` for what this does and does not claim.
EXPOSURE_TPSA_MAX = 120.0
EXPOSURE_MW_MAX = 500.0
EXPOSURE_LOGP_MIN = 0.0
EXPOSURE_TIERS = ("likely", "reduced", "unlikely")

# Glucuronide: O-linked pyranose ring bearing a carboxylic acid (no stereo).
# Sulfate ester: -O-S(=O)(=O)-O-.
_GLUCURONIDE = Chem.MolFromSmarts("[OX2]C1OC(C(=O)[OX1,OX2H1])C(O)C(O)C1")
_SULFATE = Chem.MolFromSmarts("[OX2]S(=O)(=O)[OX2]")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# --------------------------------------------------------------------------- #
# Catalogs
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _catalog(key: str) -> FilterCatalog:
    """One FilterCatalog per family, built on first use and cached.

    Separate catalogs rather than one combined catalog: a combined catalog
    makes family attribution depend on per-entry metadata that not every
    RDKit build exposes, which previously forced a fallback that assigned
    the same hits to both Brenk and PAINS.
    """
    params = FilterCatalogParams()
    params.AddCatalog(_CATALOG_SPEC[key])
    return FilterCatalog(params)


def _alert_names(key: str, mol: Chem.Mol) -> list[str]:
    return [m.GetDescription() for m in _catalog(key).GetMatches(mol)]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mol_from_smiles(smiles) -> Chem.Mol | None:
    """Parse a SMILES, returning None for anything that is not a parseable
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


def exposure_plausibility(mw: float, logp: float, tpsa: float, charge: int,
                          conjugate: bool) -> tuple[str, str | None]:
    """Crude tier for whether a compound is likely to reach an intracellular
    target, from descriptors this module already computes.

    Returns ``(tier, basis)`` where tier is one of `EXPOSURE_TIERS` and
    basis names the criteria that fired, or None if none did.

    WHAT THIS IS
        A count of conventional passive-permeability liabilities: a
        Phase-II conjugate, a non-zero formal charge, TPSA above
        `EXPOSURE_TPSA_MAX`, MW above `EXPOSURE_MW_MAX`, logP below
        `EXPOSURE_LOGP_MIN`. Zero criteria is "likely", one is "reduced",
        two or more is "unlikely".

    WHAT THIS IS NOT
        * Not a prediction of whether the metabolite **forms**. Formation
          happens in liver and gut wall and is independent of the docking
          target; this column is about presence at a site of action, which
          is a distribution question, not a formation question.
        * Not tissue-specific. It has no model of any particular tissue,
          and in particular it is not a blood-brain-barrier model.
        * Not a pharmacokinetic quantity. A real answer here is an unbound
          tissue-to-plasma partition coefficient (Kp,uu) from a PBPK model,
          which needs data this pipeline does not have.
        * Not validated. Every threshold is a drug-likeness landmark
          borrowed from the medicinal-chemistry literature and applied
          here without any evaluation against measured exposure.

        Also note ``charge`` is the formal charge of the SMILES **as
        written**, not a pKa-derived charge at pH 7.4. If the upstream
        protonation step did not run, a carboxylic acid drawn neutral
        reads as uncharged here.

    For a target that is itself an off-target liability screen (hERG, say),
    this column should not be used to narrow anything: there you want the
    widest metabolite set, because a circulating conjugate you filtered out
    is a liability you failed to screen.
    """
    reasons = []
    if conjugate:
        reasons.append("phase_II_conjugate")
    if charge != 0:
        reasons.append(f"formal_charge={charge:+d}")
    if tpsa > EXPOSURE_TPSA_MAX:
        reasons.append(f"TPSA={tpsa:.0f}")
    if mw > EXPOSURE_MW_MAX:
        reasons.append(f"MW={mw:.0f}")
    if logp < EXPOSURE_LOGP_MIN:
        reasons.append(f"logP={logp:.1f}")

    tier = EXPOSURE_TIERS[min(len(reasons), 2)]
    return tier, ";".join(reasons) if reasons else None


def parent_similarity(smiles, parent_smiles) -> float | None:
    """Morgan (r=2, 2048-bit) Tanimoto similarity between a compound and its
    parent. Returns None if either SMILES is missing or unparseable.

    This is reported for interest, not as an applicability domain -- see the
    module docstring.
    """
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
    charge = Chem.GetFormalCharge(mol)

    names = {key: _alert_names(key, mol) for key in _CATALOG_SPEC}
    counts = {f"alert_{key}_n": len(v) for key, v in names.items()}
    pains_names = names["pains_a"] + names["pains_b"] + names["pains_c"]

    glucuronide = mol.HasSubstructMatch(_GLUCURONIDE)
    sulfate = mol.HasSubstructMatch(_SULFATE)
    conjugate = bool(glucuronide or sulfate)
    tier, basis = exposure_plausibility(mw, logp, tpsa, charge, conjugate)

    row = {
        "mol_weight_annot": mw,
        "clogp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "formal_charge": charge,
        "lipinski_violations": _lipinski_violations(mw, logp, hbd, hba),
        **counts,
        "alert_total_n": int(sum(counts.values())),
        "brenk_alerts": ";".join(names["brenk"]) if names["brenk"] else None,
        "pains_alerts": ";".join(pains_names) if pains_names else None,
        "glucuronide_flag": bool(glucuronide),
        "sulfate_flag": bool(sulfate),
        "conjugate_flag": conjugate,
        "physchem_out_of_range": _physchem_out_of_range(mw, logp, tpsa),
        "exposure_plausibility": tier,
        "exposure_basis": basis,
        # Row-context columns, filled by add_annotations():
        "parent_tanimoto": None,
        "admet_status": None,
        **{c: None for c in applicability.AD_COLUMNS},
    }
    return tuple(row.items())


def annotate_smiles(smiles) -> dict:
    """Compute every per-molecule annotation for one SMILES.

    Returns a dict keyed by `ANNOTATION_COLUMNS`. Every value is None (never
    a fabricated default) if the SMILES does not parse. ``parent_tanimoto``,
    ``admet_status`` and the ``ad_*`` columns need table context (the parent,
    the batch ADMET run, the reference set) and are left None here; they are
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
    """Return each row's parent SMILES (NA where unknown or the row is a parent).

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
        parents = parents.where(~is_parent, pd.NA)  # a parent is not compared with itself
    return parents


def add_annotations(df: pd.DataFrame, smiles_col: str = "smiles", run_admet: bool = True,
                    run_ad: bool = True, ad_reference: str = "tdc_admet") -> pd.DataFrame:
    """Add every annotation column to ``df`` as new columns only.

    Never removes or reorders rows and never mutates the input. When
    ``run_admet`` is True and ``admet_ai`` is installed, prediction columns
    are added with the ``admet_`` prefix. When ``run_ad`` is True the
    applicability-domain columns are filled from the cached reference set
    (see `gnina_vs.applicability`); the AD check runs independently of
    whether ADMET-AI itself is available, because knowing that a compound
    sits outside the training chemistry is useful either way.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"add_annotations: '{smiles_col}' column not found in the given DataFrame.")

    df = df.copy()
    smiles = df[smiles_col]
    records = [annotate_smiles(s) for s in smiles]
    ann = pd.DataFrame.from_records(records, index=df.index, columns=ANNOTATION_COLUMNS)
    parseable = ann["mol_weight_annot"].notna()

    # -- parent similarity (reported, not an applicability domain) --
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

    # -- applicability domain, against the model's training chemistry --
    if run_ad:
        ad = applicability.evaluate(list(smiles), reference_name=ad_reference)
    else:
        ad = {c: ["skipped: disabled" if c == "ad_status" else None] * len(df)
              for c in applicability.AD_COLUMNS}
    for col in applicability.AD_COLUMNS:
        ann[col] = ad[col]

    for col in ANNOTATION_COLUMNS:
        df[col] = ann[col]

    if preds is not None:
        for col in preds.columns:
            out_col = col if str(col).startswith(ADMET_PREFIX) else f"{ADMET_PREFIX}{col}"
            if out_col in ANNOTATION_COLUMNS:
                out_col = f"{out_col}_pred"
            df[out_col] = smiles.map(preds[col])

    return df
