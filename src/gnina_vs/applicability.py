"""
gnina_vs.applicability
======================
Applicability-domain (AD) check for the optional ADMET-AI predictions.

Why this module exists
----------------------
An applicability domain is defined relative to *the model's training set*
(OECD validation principle 3). An earlier version of this pipeline used
Morgan/Tanimoto similarity to the compound's **parent** as a proxy, which
is not an applicability domain at all, and fails in a specifically
perverse way: a glucuronide keeps most of its parent's substructure, so
it scores *high* on parent similarity while being exactly the chemistry
ADMET training sets under-represent. The criterion was anti-correlated
with the risk it was meant to detect.

ADMET-AI is trained on Therapeutics Data Commons (TDC) ADMET datasets and
ships no applicability domain of its own, so the reference set has to be
assembled once from TDC and cached. This module does that, then measures
each query compound's mean Tanimoto distance to its k nearest reference
molecules (`skfp.applicability_domain.KNNADChecker`) and compares it with
a threshold taken from a percentile of the reference set's own distance
distribution -- so the cutoff is derived from data rather than asserted.

What this does not do
---------------------
* It does not produce a per-endpoint domain. ADMET-AI predicts dozens of
  endpoints, each with its own training set; a compound can be inside the
  domain for one and outside for another. ``reference_name`` records which
  reference set was used, and a per-endpoint version is the honest next
  step. Treat a single in/out answer as a coarse screen.
* It is not a confidence score. Inside the domain means "the training set
  contains similar chemistry", not "the prediction is correct".

Optional dependencies
---------------------
``scikit-fingerprints`` (for the AD checker) and ``PyTDC`` (to build the
reference set) are both optional. Without them every AD column is None
with a reason in ``ad_status`` -- consistent with the project rule that
optional dependencies fail gracefully and nothing is ever fabricated.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)

AD_COLUMNS = ["ad_reference", "ad_distance", "ad_threshold", "ad_in_domain", "ad_status"]

# k and the aggregation follow skfp's defaults for a k-NN domain; the
# threshold is a percentile of the *reference* set's own aggregated
# distances, so it is derived rather than asserted. These are defensible
# defaults, not validated cutoffs -- record them with the results.
AD_K = 5
AD_AGG = "mean"
AD_METRIC = "tanimoto_binary_distance"
AD_THRESHOLD_PERCENTILE = 95.0

# The TDC ADMET benchmark group. ADMET-AI is trained on these; the exact
# membership is a property of the TDC release, which is why the resolved
# list is written into the cache alongside the SMILES.
TDC_ADMET_GROUP = "admet_group"

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# --------------------------------------------------------------------------- #
# Cache location
# --------------------------------------------------------------------------- #
def cache_dir() -> Path:
    """Directory holding the cached AD reference set.

    ``GNINA_VS_CACHE`` overrides it; otherwise ``XDG_CACHE_HOME`` or
    ``~/.cache``. The reference set is downloaded once and reused, so a
    pipeline run never reaches the network.
    """
    env = os.environ.get("GNINA_VS_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "gnina-vs"


def reference_path(name: str = "tdc_admet") -> Path:
    return cache_dir() / f"ad_reference_{name}.txt"


# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #
def _skfp_available() -> bool:
    try:
        from skfp.applicability_domain import KNNADChecker  # noqa: F401
        return True
    except ImportError:
        return False


def _tdc_available() -> bool:
    try:
        import tdc  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Building the reference set (run once, needs network)
# --------------------------------------------------------------------------- #
def build_tdc_reference(name: str = "tdc_admet", overwrite: bool = False) -> Path | None:
    """Download the TDC ADMET training SMILES and cache them, once.

    Returns the cache path, or None if ``PyTDC`` is missing or the download
    fails. Needs network access and is never called during a pipeline run
    -- invoke it deliberately (see ``gnina-vs fetch-ad-reference``).

    NOT EXERCISED BY THE TEST SUITE: the TDC data hosts are unreachable
    from the environment this was written in, so the download path is
    unverified. Run it once and check the reported molecule count before
    trusting the AD columns.
    """
    path = reference_path(name)
    if path.exists() and not overwrite:
        logger.info("  [applicability] reference set already cached at '%s'.", path)
        return path
    if not _tdc_available():
        logger.warning("  [applicability] PyTDC not installed -- cannot build the reference set.")
        return None

    try:  # pragma: no cover - needs network and PyTDC
        from tdc.benchmark_group import admet_group

        group = admet_group(path=str(cache_dir() / "tdc_raw"))
        smiles: set[str] = set()
        datasets = list(group.dataset_names)
        for dataset in datasets:
            benchmark = group.get(dataset)
            train_val = benchmark["train_val"]
            col = "Drug" if "Drug" in train_val.columns else train_val.columns[1]
            smiles.update(str(s) for s in train_val[col].dropna())
        if not smiles:
            logger.warning("  [applicability] TDC returned no SMILES -- reference set not written.")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# gnina-vs AD reference: TDC {TDC_ADMET_GROUP}, {len(datasets)} datasets\n"
        path.write_text(header + "\n".join(sorted(smiles)) + "\n")
        logger.info(
            "  [applicability] cached %d unique training SMILES from %d TDC datasets -> '%s'.",
            len(smiles), len(datasets), path,
        )
        return path
    except Exception as exc:  # pragma: no cover
        logger.warning("  [applicability] building the TDC reference set failed (%s).", exc)
        return None


def load_reference_smiles(name: str = "tdc_admet", path: str | Path | None = None) -> list[str] | None:
    """Read cached reference SMILES. Returns None if no cache exists."""
    target = Path(path) if path is not None else reference_path(name)
    if not target.exists():
        return None
    lines = [ln.strip() for ln in target.read_text().splitlines()]
    smiles = [ln for ln in lines if ln and not ln.startswith("#")]
    return smiles or None


# --------------------------------------------------------------------------- #
# The checker
# --------------------------------------------------------------------------- #
def _fingerprint_matrix(smiles_list) -> tuple[np.ndarray, list[int]]:
    """Morgan bit vectors for parseable SMILES, plus their positions."""
    rows, keep = [], []
    for i, smi in enumerate(smiles_list):
        if not isinstance(smi, str) or not smi.strip():
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append(np.frombuffer(_MORGAN.GetFingerprintAsNumPy(mol), dtype=np.uint8))
        keep.append(i)
    if not rows:
        return np.empty((0, 2048), dtype=np.uint8), []
    return np.vstack(rows), keep


class DomainChecker:
    """k-NN applicability-domain check against a fixed reference set.

    ``fit`` takes the *reference* (training) SMILES -- never a single
    comparator compound. The threshold is the configured percentile of the
    reference set's own aggregated nearest-neighbour distances.
    """

    def __init__(self, k: int = AD_K, agg: str = AD_AGG, metric: str = AD_METRIC,
                 threshold_percentile: float = AD_THRESHOLD_PERCENTILE,
                 reference_name: str = "tdc_admet"):
        self.k = k
        self.agg = agg
        self.metric = metric
        self.threshold_percentile = threshold_percentile
        self.reference_name = reference_name
        self._checker = None
        self.threshold: float | None = None
        self.n_reference: int = 0

    def fit(self, reference_smiles) -> "DomainChecker":
        from skfp.applicability_domain import KNNADChecker

        X, _ = _fingerprint_matrix(reference_smiles)
        if X.shape[0] <= self.k:
            raise ValueError(
                f"AD reference set has {X.shape[0]} usable molecules, which is not more "
                f"than k={self.k}; refusing to fit a domain on it."
            )
        self._checker = KNNADChecker(
            k=self.k, metric=self.metric, agg=self.agg, threshold=self.threshold_percentile,
        )
        self._checker.fit(X)
        self.threshold = float(getattr(self._checker, "threshold_", np.nan))
        self.n_reference = int(X.shape[0])
        return self

    def score(self, smiles_list) -> tuple[list[float | None], list[bool | None]]:
        """Distances and in-domain booleans, aligned with ``smiles_list``.

        An unparseable SMILES yields (None, None) rather than a number.
        """
        n = len(smiles_list)
        distances: list[float | None] = [None] * n
        in_domain: list[bool | None] = [None] * n
        if self._checker is None:
            return distances, in_domain
        X, keep = _fingerprint_matrix(smiles_list)
        if not keep:
            return distances, in_domain
        scores = np.asarray(self._checker.score_samples(X), dtype=float)
        flags = np.asarray(self._checker.predict(X)).astype(bool)
        for pos, idx in enumerate(keep):
            distances[idx] = float(scores[pos])
            in_domain[idx] = bool(flags[pos])
        return distances, in_domain


# --------------------------------------------------------------------------- #
# Entry point used by annotate.add_annotations
# --------------------------------------------------------------------------- #
def evaluate(smiles_list, reference_name: str = "tdc_admet",
             reference_path_override: str | Path | None = None) -> dict:
    """Score a batch of SMILES against the cached reference set.

    Returns a dict with the `AD_COLUMNS` keys; every value is either a
    per-row list or a scalar repeated by the caller. When the check cannot
    run, ``ad_status`` says why and every other column is None -- the
    reason is recorded rather than guessed at.
    """
    n = len(smiles_list)
    blank = {
        "ad_reference": [None] * n,
        "ad_distance": [None] * n,
        "ad_threshold": [None] * n,
        "ad_in_domain": [None] * n,
        "ad_status": ["skipped: not evaluated"] * n,
    }

    if not _skfp_available():
        blank["ad_status"] = ["skipped: scikit-fingerprints not installed"] * n
        return blank

    reference = load_reference_smiles(reference_name, reference_path_override)
    if reference is None:
        blank["ad_status"] = ["skipped: no cached AD reference (run gnina-vs fetch-ad-reference)"] * n
        return blank

    try:
        checker = DomainChecker(reference_name=reference_name).fit(reference)
    except Exception as exc:
        logger.warning("  [applicability] fitting the domain failed (%s) -- AD columns left empty.", exc)
        blank["ad_status"] = [f"failed: {type(exc).__name__}"] * n
        return blank

    distances, flags = checker.score(smiles_list)
    label = f"{reference_name} (n={checker.n_reference}, k={checker.k}, p{checker.threshold_percentile:g})"
    return {
        "ad_reference": [label if d is not None else None for d in distances],
        "ad_distance": distances,
        "ad_threshold": [checker.threshold if d is not None else None for d in distances],
        "ad_in_domain": flags,
        "ad_status": ["computed" if d is not None else "skipped: invalid_smiles" for d in distances],
    }
