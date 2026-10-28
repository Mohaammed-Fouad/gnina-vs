"""
Tests for gnina_vs.report -- the self-contained HTML triage file.

Two of these guard bugs that only appeared when the page was actually
rendered in a browser, and both were silent failures:

* itables defaults to ``maxBytes='64KB'`` and downsamples past it, so with
  inline depictions (~6-7 KB each) the table quietly dropped compounds.
* RDKit writes newlines inside SVG path ``d`` attributes; serialised into
  the page they became literal ``\\n`` and every structure rendered blank.

Neither raised anything, so both are asserted here directly.
"""
import numpy as np
import pandas as pd
import pytest

from gnina_vs import report, tables

pytest.importorskip("itables", reason="itables is an optional dependency")

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
PARA_GLUC = "CC(=O)Nc1ccc(OC2OC(C(=O)O)C(O)C(O)C2O)cc1"
PARA_SULF = "CC(=O)Nc1ccc(OS(=O)(=O)O)cc1"


def _flat(n_per_target=4, targets=("T1", "T2")):
    smiles = [PARACETAMOL, PARA_GLUC, PARA_SULF, "not-a-smiles"]
    rows = []
    for target in targets:
        for i in range(n_per_target):
            smi = smiles[i % len(smiles)]
            parses = smi != "not-a-smiles"
            is_parent = i == 0
            rows.append({
                "target_id": target, "compound_id": f"c{i}", "base_compound_id": "c0",
                "parent_id": "c0", "ID": "DRUG1", "smiles": smi,
                "status": "ok" if parses else "no_output",
                "pose_quality_flag": "ok", "cnn_affinity": 6.0 + 0.1 * i,
                "vina_affinity": -8.3, "cnn_score": 0.75, "heavy_atoms": 11,
                "mol_weight": 151.16, "ligand_efficiency_cnn": 0.5512345,
                "ligand_efficiency_vina": 0.75,
                "delta_cnn_affinity_vs_parent": np.nan if is_parent else -1.234,
                "delta_vina_affinity_vs_parent": np.nan,
                "delta_cnn_affinity_vs_reference": np.nan,
                "delta_vina_affinity_vs_reference": np.nan,
                "rank_in_group": float(i + 1), "rank_vs_parent_change": np.nan,
                "is_parent": is_parent, "depth": 0 if is_parent else 1,
                "phase": "II", "reaction": "glucuronidation", "error": "",
                "mol_weight_annot": 151.16 if parses else None,
                "clogp": 0.46 if parses else None,
                "tpsa": 49.3 if parses else None,
                "alert_brenk_n": 1 if parses else None,
                "alert_pains_a_n": 0 if parses else None,
                "conjugate_flag": (not is_parent) if parses else None,
                "exposure_plausibility": ("likely" if is_parent else "unlikely") if parses else None,
                "ad_in_domain": (i % 2 == 0) if parses else None,
            })
    return pd.DataFrame(rows)


def _split(**kw):
    return tables.split_tables(_flat(**kw), run_id="testrun")


# --------------------------------------------------------------------------- #
# Depictions
# --------------------------------------------------------------------------- #
def test_svg_contains_no_newlines():
    """RDKit puts newlines inside path `d` attributes. Serialised into the
    page they become literal backslash-n and the browser rejects the path,
    so every structure renders blank -- with no error anywhere."""
    svg = report.molecule_svg(PARACETAMOL)
    assert svg.startswith("<svg") or "<svg" in svg
    assert "\n" not in svg
    assert "\\n" not in svg
    assert "<path" in svg


def test_svg_is_empty_for_unusable_input():
    for bad in ("not-a-smiles", "", None, float("nan"), 42):
        assert report.molecule_svg(bad) == ""


def test_svg_carries_no_xml_declaration():
    assert "<?xml" not in report.molecule_svg(PARACETAMOL)


# --------------------------------------------------------------------------- #
# The triage frame
# --------------------------------------------------------------------------- #
def test_triage_frame_joins_both_grains():
    frame = report.build_triage_frame(_split())
    assert "Structure" in frame.columns          # from compounds
    assert "Rank band" in frame.columns          # from screen
    assert "Exposure" in frame.columns           # from compounds
    assert len(frame) == 8


def test_parent_sorts_above_its_own_metabolites():
    frame = report.build_triage_frame(_split())
    first_target = frame[frame["Target"] == "T1"]
    assert first_target.iloc[0]["Role"] == "parent"


def test_truncation_is_recorded_not_silent():
    frame = report.build_triage_frame(_split(), max_rows=3)
    assert len(frame) == 3
    assert frame.attrs["truncated"] == 5
    assert frame.attrs["total_rows"] == 8


def test_no_truncation_reports_zero():
    frame = report.build_triage_frame(_split())
    assert frame.attrs["truncated"] == 0


# --------------------------------------------------------------------------- #
# Display formatting
# --------------------------------------------------------------------------- #
def test_scores_are_rounded_to_one_decimal():
    """Three decimals on a docking score is an implied-precision claim."""
    frame = report.build_triage_frame(_split())
    shown = report._format_for_display(frame)
    deltas = [v for v in shown["ΔCNNaff"] if v != "--"]
    assert deltas, "expected at least one delta"
    for value in deltas:
        assert len(value.split(".")[-1]) == 1, value


def test_booleans_render_as_words_and_missing_as_a_dash():
    frame = report.build_triage_frame(_split())
    shown = report._format_for_display(frame)
    assert set(shown["Conjugate"]).issubset({"yes", "no", "--"})
    assert "--" in set(shown["Conjugate"])      # the unparseable row


# --------------------------------------------------------------------------- #
# The file itself
# --------------------------------------------------------------------------- #
def test_report_makes_no_external_requests(tmp_path):
    """The hard requirement: it has to open with no server and no network.

    Measured, not assumed -- mols2grid's interactive grid references four
    CDNs (RDKit.js among them) and bokehmol fetches RDKit.js even with
    `resources=INLINE`, so neither satisfies this.
    """
    path = report.write_report(_split(), tmp_path / "triage.html")
    html = open(path, encoding="utf-8").read()
    assert report.external_references(html) == []


def test_report_embeds_structures_rather_than_fetching_a_renderer(tmp_path):
    path = report.write_report(_split(), tmp_path / "triage.html")
    html = open(path, encoding="utf-8").read()
    assert "<svg" in html
    assert "RDKit_minimal" not in html          # no client-side renderer
    assert "unpkg.com" not in html


def test_report_does_not_silently_drop_rows(tmp_path):
    """itables downsamples past maxBytes='64KB' without raising. Inline
    depictions cross that at about ten molecules, so the report has to
    disable it or it would violate the pipeline's no-drop rule."""
    split = _split(n_per_target=12, targets=("T1", "T2", "T3"))
    n_rows = len(split["screen"])
    assert n_rows > 20
    path = report.write_report(split, tmp_path / "triage.html", max_rows=None)
    html = open(path, encoding="utf-8").read()
    # every compound_id must appear in the serialised table data
    for compound_id in split["screen"]["compound_id"].unique():
        assert f'\\"{compound_id}\\"' in html or f'"{compound_id}"' in html, compound_id
    assert '"maxBytes": 0' in html or "maxBytes" not in html.split("dt_args")[0]


def test_report_states_its_caveats_on_the_page(tmp_path):
    """A supervisor reads the page, not the README, so the caveats travel
    with the table rather than sitting in a footnote elsewhere."""
    path = report.write_report(_split(), tmp_path / "triage.html")
    html = open(path, encoding="utf-8").read()
    assert "not affinities" in html
    assert "no error bars" in html.lower()
    assert "Nothing has been filtered" in html


def test_report_announces_truncation_on_the_page(tmp_path):
    path = report.write_report(_split(), tmp_path / "triage.html", max_rows=2)
    html = open(path, encoding="utf-8").read()
    assert "Truncated" in html
    assert "screen.parquet" in html


def test_report_without_depictions_is_much_smaller(tmp_path):
    with_svg = report.write_report(_split(), tmp_path / "a.html")
    without = report.write_report(_split(), tmp_path / "b.html", with_depictions=False)
    import os
    assert os.path.getsize(without) < os.path.getsize(with_svg)


def test_external_references_detects_a_cdn_link():
    html = '<script src="https://unpkg.com/x.js"></script><a href="/local">x</a>'
    assert report.external_references(html) == ["https://unpkg.com/x.js"]
