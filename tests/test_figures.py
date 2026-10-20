"""
Phase 7 tests: gnina_vs.figures -- each figure function writes a non-empty
PNG and SVG from a synthetic comparison table, headless (Agg backend).
Not testing pixel content, only that the functions run end to end on
realistic input shapes and fail clearly on bad/empty input.
"""
import os

import numpy as np
import pandas as pd
import pytest

from gnina_vs import figures


def _sample_comparison_table():
    rows = [
        {"target_id": "hERG", "compound_id": "drugA", "base_compound_id": "drugA", "parent_id": "drugA",
         "is_parent": True, "cnn_affinity": 6.0, "vina_affinity": -6.5, "mol_weight": 300.0,
         "mol_weight_annot": 300.0, "delta_cnn_affinity_vs_parent": np.nan,
         "delta_vina_affinity_vs_parent": np.nan, "reaction": None, "phase": None},
        {"target_id": "hERG", "compound_id": "drugA_m1", "base_compound_id": "drugA_m1", "parent_id": "drugA",
         "is_parent": False, "cnn_affinity": 7.5, "vina_affinity": -8.0, "mol_weight": 320.0,
         "mol_weight_annot": 320.0, "delta_cnn_affinity_vs_parent": 1.5,
         "delta_vina_affinity_vs_parent": -1.5, "reaction": "hydroxylation", "phase": "I"},
        {"target_id": "hERG", "compound_id": "drugA_m2", "base_compound_id": "drugA_m2", "parent_id": "drugA",
         "is_parent": False, "cnn_affinity": 4.0, "vina_affinity": -4.5, "mol_weight": 480.0,
         "mol_weight_annot": 480.0, "delta_cnn_affinity_vs_parent": -2.0,
         "delta_vina_affinity_vs_parent": 2.0, "reaction": "glucuronidation", "phase": "II"},
        {"target_id": "MOR", "compound_id": "drugA", "base_compound_id": "drugA", "parent_id": "drugA",
         "is_parent": True, "cnn_affinity": 5.0, "vina_affinity": -5.5, "mol_weight": 300.0,
         "mol_weight_annot": 300.0, "delta_cnn_affinity_vs_parent": np.nan,
         "delta_vina_affinity_vs_parent": np.nan, "reaction": None, "phase": None},
        {"target_id": "MOR", "compound_id": "drugA_m1", "base_compound_id": "drugA_m1", "parent_id": "drugA",
         "is_parent": False, "cnn_affinity": 5.2, "vina_affinity": -5.6, "mol_weight": 320.0,
         "mol_weight_annot": 320.0, "delta_cnn_affinity_vs_parent": 0.2,
         "delta_vina_affinity_vs_parent": -0.1, "reaction": "hydroxylation", "phase": "I"},
    ]
    return pd.DataFrame(rows)


def _assert_png_svg(paths, min_size=200):
    assert set(paths) == {"png", "svg"}
    for p in paths.values():
        assert os.path.exists(p)
        assert os.path.getsize(p) > min_size


# --------------------------------------------------------------------------- #
def test_metabolic_tree(tmp_path):
    df = _sample_comparison_table()
    paths = figures.metabolic_tree(df, "drugA", str(tmp_path / "tree"))
    _assert_png_svg(paths)


def test_metabolic_tree_single_target(tmp_path):
    df = _sample_comparison_table()
    paths = figures.metabolic_tree(df, "drugA", str(tmp_path / "tree_herg"), target_id="hERG")
    _assert_png_svg(paths)


def test_metabolic_tree_unknown_parent_raises(tmp_path):
    df = _sample_comparison_table()
    with pytest.raises(ValueError, match="drugX"):
        figures.metabolic_tree(df, "drugX", str(tmp_path / "tree"))


def test_bar_chart_parent_vs_metabolites(tmp_path):
    df = _sample_comparison_table()
    paths = figures.bar_chart_parent_vs_metabolites(df, "hERG", str(tmp_path / "bar"))
    _assert_png_svg(paths)


def test_bar_chart_unknown_target_raises(tmp_path):
    df = _sample_comparison_table()
    with pytest.raises(ValueError, match="NOPE"):
        figures.bar_chart_parent_vs_metabolites(df, "NOPE", str(tmp_path / "bar"))


def test_heatmap_compounds_targets(tmp_path):
    df = _sample_comparison_table()
    paths = figures.heatmap_compounds_targets(df, str(tmp_path / "heatmap"))
    _assert_png_svg(paths)


def test_heatmap_empty_raises(tmp_path):
    empty = pd.DataFrame(columns=["base_compound_id", "target_id", "cnn_affinity", "vina_affinity"])
    with pytest.raises(ValueError):
        figures.heatmap_compounds_targets(empty, str(tmp_path / "heatmap"))


def test_scatter_score_vs_mw(tmp_path):
    df = _sample_comparison_table()
    paths = figures.scatter_score_vs_mw(df, str(tmp_path / "scatter"))
    _assert_png_svg(paths)


def test_scatter_reveals_size_bias_data_present(tmp_path):
    # sanity: the heaviest compound (the glucuronide, drugA_m2) is included
    df = _sample_comparison_table()
    paths = figures.scatter_score_vs_mw(df, str(tmp_path / "scatter"))
    _assert_png_svg(paths)
    assert (df["mol_weight"] == 480.0).any()


def test_interaction_diagram_not_implemented():
    with pytest.raises(NotImplementedError, match="PLIP"):
        figures.interaction_diagram()


def test_output_path_with_extension_is_normalized(tmp_path):
    df = _sample_comparison_table()
    paths = figures.heatmap_compounds_targets(df, str(tmp_path / "heatmap.png"))
    assert paths["png"] == str(tmp_path / "heatmap.png")
    assert paths["svg"] == str(tmp_path / "heatmap.svg")
