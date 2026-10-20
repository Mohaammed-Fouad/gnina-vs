"""
gnina_vs.figures
=====================
Phase 7: figures built from the comparison table, saved as PNG and SVG.
Every function takes a DataFrame and an output path (without extension --
both ``<path>.png`` and ``<path>.svg`` are written) and works headless
(matplotlib's non-interactive "Agg" backend, set at import time).

2D interaction diagrams (PLIP/ProLIF) are out of scope for this
environment (neither package nor a working PyMOL/OpenBabel-based contact
analysis stack was available) -- `interaction_diagram` below is a stub
that raises `NotImplementedError` with a clear message, so a caller finds
out immediately rather than silently getting nothing.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # headless: no display server needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CMAP = plt.get_cmap("coolwarm")


def _save(fig, output_path: str) -> dict:
    base = output_path[:-4] if output_path.lower().endswith((".png", ".svg")) else output_path
    png_path, svg_path = f"{base}.png", f"{base}.svg"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "svg": svg_path}


def _primary_score_col(df: pd.DataFrame) -> str:
    return "cnn_affinity" if df.get("cnn_affinity", pd.Series(dtype=float)).notna().any() else "vina_affinity"


def _delta_col_for(score_col: str) -> str:
    return "delta_cnn_affinity_vs_parent" if score_col == "cnn_affinity" else "delta_vina_affinity_vs_parent"


# --------------------------------------------------------------------------- #
# Metabolic tree: parent -> metabolites, colored by delta vs parent
# --------------------------------------------------------------------------- #
def metabolic_tree(df: pd.DataFrame, parent_id: str, output_path: str, target_id: str | None = None) -> dict:
    """One drug's metabolic tree (parent at top, metabolites below),
    nodes colored by their docking-score delta vs the parent. If
    ``target_id`` is given, only that target's rows are used; otherwise
    each compound's delta is averaged across all targets it was docked
    against.
    """
    sub = df[df["parent_id"] == parent_id].copy()
    if target_id is not None:
        sub = sub[sub["target_id"] == target_id]
    if sub.empty:
        suffix = f", target_id='{target_id}'" if target_id else ""
        raise ValueError(f"metabolic_tree: no rows found for parent_id='{parent_id}'{suffix}.")

    score_col = _primary_score_col(sub)
    delta_col = _delta_col_for(score_col)

    # one row per compound (average delta across targets when target_id is None)
    grouped = sub.groupby("base_compound_id", dropna=False).agg(
        is_parent=("is_parent", "max"), delta=(delta_col, "mean"),
        reaction=("reaction", "first"),
    ).reset_index()

    parent_rows = grouped[grouped["is_parent"] == True]  # noqa: E712
    metabolites = grouped[grouped["is_parent"] != True]  # noqa: E712
    if parent_rows.empty:
        raise ValueError(f"metabolic_tree: no parent row found for parent_id='{parent_id}'.")
    parent_label = parent_rows.iloc[0]["base_compound_id"]

    n = max(len(metabolites), 1)
    fig, ax = plt.subplots(figsize=(max(4, 1.8 * n), 4.5))

    finite_deltas = grouped["delta"].dropna()
    vmax = float(np.abs(finite_deltas).max()) if not finite_deltas.empty else 1.0
    vmax = vmax if vmax > 0 else 1.0
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    ax.scatter([n / 2.0], [1.0], s=1800, color="0.85", edgecolor="black", zorder=3)
    ax.text(n / 2.0, 1.0, parent_label, ha="center", va="center", fontsize=8, zorder=4)

    for i, (_, row) in enumerate(metabolites.iterrows()):
        x = i + 0.5
        color = _CMAP(norm(row["delta"])) if pd.notna(row["delta"]) else "0.9"
        ax.plot([n / 2.0, x], [0.92, 0.12], color="grey", zorder=1, linewidth=1)
        ax.scatter([x], [0.0], s=1400, color=color, edgecolor="black", zorder=3)
        ax.text(x, 0.0, row["base_compound_id"], ha="center", va="center", fontsize=7, zorder=4)
        label = row["reaction"] if pd.notna(row["reaction"]) else ""
        ax.text(x, 0.5, str(label), ha="center", va="center", fontsize=6, color="grey", zorder=4)

    if not finite_deltas.empty:
        sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=f"delta {score_col} vs parent", shrink=0.7)

    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-0.4, 1.4)
    ax.axis("off")
    ax.set_title(f"Metabolic tree: {parent_label}" + (f" @ {target_id}" if target_id else " (avg. across targets)"))
    return _save(fig, output_path)


# --------------------------------------------------------------------------- #
# Bar chart: parent vs metabolites, per target
# --------------------------------------------------------------------------- #
def bar_chart_parent_vs_metabolites(df: pd.DataFrame, target_id: str, output_path: str,
                                    parent_id: str | None = None) -> dict:
    """Bar chart of the primary docking score for every compound of one
    target (optionally restricted to one drug's parent_id group)."""
    sub = df[df["target_id"] == target_id].copy()
    if parent_id is not None:
        sub = sub[sub["parent_id"] == parent_id]
    if sub.empty:
        raise ValueError(f"bar_chart_parent_vs_metabolites: no rows for target_id='{target_id}'.")

    score_col = _primary_score_col(sub)
    sub = sub[sub[score_col].notna()].sort_values(score_col, ascending=(score_col == "vina_affinity"))
    if sub.empty:
        raise ValueError(f"bar_chart_parent_vs_metabolites: no scored rows for target_id='{target_id}'.")

    colors = ["#4C72B0" if p else "#DD8452" for p in sub["is_parent"]]
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(sub)), 4))
    ax.bar(sub["base_compound_id"].astype(str), sub[score_col], color=colors)
    ax.set_ylabel(score_col)
    ax.set_title(f"{target_id}: parent vs metabolites ({score_col})")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="parent"), Patch(color="#DD8452", label="metabolite")])
    return _save(fig, output_path)


# --------------------------------------------------------------------------- #
# Heatmap: compounds x targets
# --------------------------------------------------------------------------- #
def heatmap_compounds_targets(df: pd.DataFrame, output_path: str, score_col: str | None = None) -> dict:
    """Heatmap of one score across every (compound, target) pair docked."""
    score_col = score_col or _primary_score_col(df)
    pivot = df.pivot_table(index="base_compound_id", columns="target_id", values=score_col, aggfunc="mean")
    if pivot.empty:
        raise ValueError("heatmap_compounds_targets: nothing to plot (empty comparison table).")

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * pivot.shape[1] + 2), max(3, 0.4 * pivot.shape[0] + 1)))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label=score_col)
    ax.set_title(f"Compounds x targets ({score_col})")
    return _save(fig, output_path)


# --------------------------------------------------------------------------- #
# Scatter: docking score vs molecular weight (exposes size bias)
# --------------------------------------------------------------------------- #
def scatter_score_vs_mw(df: pd.DataFrame, output_path: str) -> dict:
    """Scatter of docking score vs molecular weight, to visually expose
    any size bias in the ranking (bigger molecules tend to score better on
    raw affinity even when ligand efficiency is unchanged or worse)."""
    score_col = _primary_score_col(df)
    mw_col = "mol_weight_annot" if "mol_weight_annot" in df.columns and df["mol_weight_annot"].notna().any() \
        else "mol_weight"
    sub = df[df[score_col].notna() & df[mw_col].notna()]
    if sub.empty:
        raise ValueError("scatter_score_vs_mw: no rows with both a score and a molecular weight.")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4C72B0" if p else "#DD8452" for p in sub["is_parent"]]
    ax.scatter(sub[mw_col], sub[score_col], c=colors, edgecolor="black", alpha=0.8)
    for _, row in sub.iterrows():
        ax.annotate(str(row["base_compound_id"]), (row[mw_col], row[score_col]), fontsize=6,
                   xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Molecular weight (Da)")
    ax.set_ylabel(score_col)
    ax.set_title("Docking score vs molecular weight")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="parent"), Patch(color="#DD8452", label="metabolite")])
    return _save(fig, output_path)


# --------------------------------------------------------------------------- #
# Interaction diagrams -- out of scope in this environment
# --------------------------------------------------------------------------- #
def interaction_diagram(*args, **kwargs):
    """2D protein-ligand interaction diagrams via PLIP or ProLIF.

    Not implemented: neither PLIP nor ProLIF (nor a working contact-analysis
    stack) was available in this environment to build and test against.
    Raises `NotImplementedError` rather than silently doing nothing, so a
    caller finds out immediately. A real implementation would take a
    docked-pose SDF/PDB pair and call PLIP's or ProLIF's Python API to
    render one PNG/SVG per (compound, target) pose.
    """
    raise NotImplementedError(
        "interaction_diagram is not implemented: PLIP/ProLIF were not available "
        "to build and test this against in the environment this was developed in. "
        "Install one of them and implement this hook against its API."
    )
