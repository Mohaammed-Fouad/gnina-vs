"""
Tests for gnina_vs.applicability -- the k-NN applicability-domain check
that replaced the similarity-to-parent proxy.

The TDC download path (`build_tdc_reference`) is NOT tested: it needs
network access and the optional PyTDC package. Everything downstream of
"a reference set exists" is tested here against a small stand-in set.
"""
import pytest

from gnina_vs import applicability

# A stand-in reference set. Deliberately small and obviously not TDC --
# these tests check the domain machinery, not ADMET-AI's real training space.
REFERENCE = [
    "CC(=O)Nc1ccc(O)cc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1",
    "CC(N)Cc1ccccc1", "OC(=O)c1ccccc1O", "CN(C)CCCN1c2ccccc2CCc2ccccc21",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1", "Nc1ccc(cc1)S(=O)(=O)N", "CCCCNC(=O)NS(=O)(=O)c1ccc(C)cc1",
    "OCC1OC(O)C(O)C(O)C1O", "c1ccc2[nH]c3ccccc3c2c1", "CC(=O)Nc1ccc(cc1)S(=O)(=O)N",
    "CN1CCN(CC1)c1ccccn1", "Oc1ccc(cc1)C(=O)c1ccccc1", "CCOC(=O)c1ccccc1N",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "NC(=O)c1ccccn1", "CSCCC(N)C(=O)O", "Nc1nc(N)nc(n1)N",
    "COc1ccccc1OCC(O)CNC(C)C", "CC(C)NCC(O)COc1ccccc1", "Clc1cccc(Cl)c1NC1=NCCN1",
]

pytest.importorskip("skfp", reason="scikit-fingerprints is an optional dependency")


@pytest.fixture
def cached_reference(tmp_path, monkeypatch):
    """Point the cache at tmp_path and write a stand-in reference set."""
    monkeypatch.setenv("GNINA_VS_CACHE", str(tmp_path))
    path = applicability.reference_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# stand-in reference set for tests\n" + "\n".join(REFERENCE) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Cache location and reference loading
# --------------------------------------------------------------------------- #
def test_cache_dir_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GNINA_VS_CACHE", str(tmp_path / "somewhere"))
    assert applicability.cache_dir() == tmp_path / "somewhere"


def test_load_reference_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("GNINA_VS_CACHE", str(tmp_path))
    assert applicability.load_reference_smiles() is None


def test_load_reference_skips_comments(cached_reference):
    smiles = applicability.load_reference_smiles()
    assert smiles is not None
    assert len(smiles) == len(REFERENCE)
    assert not any(s.startswith("#") for s in smiles)


# --------------------------------------------------------------------------- #
# The domain itself
# --------------------------------------------------------------------------- #
def test_threshold_is_derived_from_the_reference_set():
    checker = applicability.DomainChecker().fit(REFERENCE)
    assert checker.n_reference == len(REFERENCE)
    # The cutoff comes from a percentile of the reference set's own distance
    # distribution, so it must be a real number in Tanimoto-distance range.
    assert checker.threshold is not None
    assert 0.0 < checker.threshold <= 1.0


def test_fit_refuses_a_reference_set_smaller_than_k():
    with pytest.raises(ValueError, match="not more than k"):
        applicability.DomainChecker(k=5).fit(REFERENCE[:3])


def test_fit_rejects_a_single_comparator_compound():
    """The old proxy compared each molecule with its parent. A one-molecule
    'reference set' must not be fittable as a domain."""
    with pytest.raises(ValueError):
        applicability.DomainChecker().fit(["CC(=O)Nc1ccc(O)cc1"])


def test_a_reference_member_is_closer_than_an_absurd_molecule():
    checker = applicability.DomainChecker().fit(REFERENCE)
    distances, flags = checker.score(["CC(=O)Nc1ccc(O)cc1", "C" * 40])
    assert distances[0] < distances[1]
    assert flags[0] is True
    assert flags[1] is False


def test_unparseable_smiles_scores_none_not_a_number():
    checker = applicability.DomainChecker().fit(REFERENCE)
    distances, flags = checker.score(["not-a-smiles", None, "", "CC(=O)Nc1ccc(O)cc1"])
    assert distances[:3] == [None, None, None]
    assert flags[:3] == [None, None, None]
    assert distances[3] is not None


def test_score_before_fit_returns_all_none():
    distances, flags = applicability.DomainChecker().score(["CCO"])
    assert distances == [None] and flags == [None]


# --------------------------------------------------------------------------- #
# evaluate(): the batch entry point used by annotate
# --------------------------------------------------------------------------- #
def test_evaluate_reports_every_column(cached_reference):
    out = applicability.evaluate(["CC(=O)Nc1ccc(O)cc1", "C" * 40, "not-a-smiles"])
    assert set(out) == set(applicability.AD_COLUMNS)
    assert all(len(v) == 3 for v in out.values())
    assert out["ad_status"][0] == "computed"
    assert out["ad_status"][2] == "skipped: invalid_smiles"
    assert out["ad_in_domain"][2] is None


def test_evaluate_records_reference_provenance(cached_reference):
    out = applicability.evaluate(["CC(=O)Nc1ccc(O)cc1"])
    label = out["ad_reference"][0]
    # The label must state what the domain was measured against, so a
    # distance in the results table is interpretable later.
    assert "tdc_admet" in label
    assert f"n={len(REFERENCE)}" in label
    assert "k=" in label and "p95" in label


def test_evaluate_without_a_cached_reference_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("GNINA_VS_CACHE", str(tmp_path))
    out = applicability.evaluate(["CCO"])
    assert out["ad_distance"] == [None]
    assert out["ad_in_domain"] == [None]
    assert "no cached AD reference" in out["ad_status"][0]


def test_evaluate_without_skfp_says_so(monkeypatch):
    monkeypatch.setattr(applicability, "_skfp_available", lambda: False)
    out = applicability.evaluate(["CCO", "CCC"])
    assert out["ad_status"] == ["skipped: scikit-fingerprints not installed"] * 2
    assert out["ad_distance"] == [None, None]


def test_build_tdc_reference_without_pytdc_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("GNINA_VS_CACHE", str(tmp_path))
    monkeypatch.setattr(applicability, "_tdc_available", lambda: False)
    assert applicability.build_tdc_reference() is None


def test_build_tdc_reference_is_a_noop_when_cached(cached_reference, monkeypatch):
    def fail():
        raise AssertionError("must not consult PyTDC when a cache already exists")
    monkeypatch.setattr(applicability, "_tdc_available", fail)
    assert applicability.build_tdc_reference() == cached_reference
