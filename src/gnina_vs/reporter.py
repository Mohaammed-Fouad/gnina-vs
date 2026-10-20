"""
gnina_vs.reporter
======================
Step 4 of the pipeline: parse Gnina scores (from stdout or, as a fallback,
from the docked SDF's embedded properties), export a ranked CSV/Excel
report, merge all docked poses into a single ranked SDF, and support
recovery (re-scoring from existing docked SDFs without re-docking).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rdkit import Chem

from gnina_vs.docking import STATUS_ERROR, STATUS_FAILED, STATUS_NO_OUTPUT, STATUS_OK, STATUS_TIMEOUT
from gnina_vs.ligand import make_compound_id

logger = logging.getLogger(__name__)

# Header keyword patterns
_HDR_AFFINITY = re.compile(r"affinity", re.IGNORECASE)
_HDR_CNN_SCORE = re.compile(r"cnn.?score", re.IGNORECASE)
_HDR_CNN_AFFINITY = re.compile(r"cnn.?affinity", re.IGNORECASE)

_KV_AFFINITY = re.compile(r"affinity\s*[=:]\s*([-\d.]+)", re.IGNORECASE)
_KV_CNN_SCORE = re.compile(r"cnn.?score\s*[=:]\s*([\d.]+)", re.IGNORECASE)
_KV_CNN_AFFINITY = re.compile(r"cnn.?affinity\s*[=:]\s*([\d.]+)", re.IGNORECASE)

# Statuses that never get scored, no matter what raw_log or output_path say.
_TERMINAL_FAILURE_STATUSES = {STATUS_FAILED, STATUS_TIMEOUT, STATUS_NO_OUTPUT, STATUS_ERROR}

# Cutoff below which a CNN pose score is flagged as unreliable (pose quality,
# not binding affinity -- see Phase A item 15).
LOW_CNN_SCORE_THRESHOLD = 0.5


def parse_top_n(top_n: int | str) -> int:
    """Validate and normalise ``--top-n``. ``0`` or ``'all'`` (any case)
    means "export everything" and is returned as ``0``. Anything else must
    be a non-negative integer. Raises `ValueError` naming ``--top-n``
    otherwise, so this can be checked before docking starts (item 18)."""
    if isinstance(top_n, str):
        s = top_n.strip()
        if s.lower() == "all":
            return 0
        if not re.fullmatch(r"\d+", s):
            raise ValueError(
                f"--top-n must be a non-negative integer or 'all', got '{top_n}'."
            )
        return int(s)
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise ValueError(f"--top-n must be a non-negative integer or 'all', got '{top_n}'.")
    if top_n < 0:
        raise ValueError(f"--top-n must be a non-negative integer or 'all', got '{top_n}'.")
    return top_n


def _split_pipe_row(line: str) -> list[str]:
    parts = [p.strip() for p in line.split("|")]
    return [p for p in parts if p]


def _parse_gnina_log(raw_log: str) -> dict[str, float | None]:
    """Parse Gnina stdout to extract mode-1 scores.

    Gnina v1.3.2 produces this format::

        mode |  affinity  |  intramol  |    CNN     |   CNN
             | (kcal/mol) | (kcal/mol) | pose score | affinity
        -----+------------+------------+------------+----------
            1       -8.32       -0.34       0.7574      6.438

    With ``--cnn_scoring none``, the CNN columns are absent and only
    ``vina_affinity`` is populated -- callers must not assume ``cnn_affinity``
    is present.

    Key quirks handled here:
    - The column headers span TWO lines (affinity on line 1, CNN scores on
      line 2).
    - Data rows are space-separated, NOT pipe-separated.
    - The separator line starts with "---".
    """
    scores: dict[str, float | None] = {
        "vina_affinity": None,
        "cnn_score": None,
        "cnn_affinity": None,
    }

    if not raw_log or raw_log == "TIMEOUT":
        return scores

    lines = raw_log.splitlines()

    # -- Strategy 1: find the "-----+" separator, then parse the next line --
    # This is the most reliable approach for gnina's space-delimited table.
    for i, line in enumerate(lines):
        if line.strip().startswith("---") and "+" in line:
            for data_line in lines[i + 1:]:
                stripped = data_line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) >= 5 and parts[0] == "1":
                    # Full CNN table: mode affinity intramol cnn_pose_score cnn_affinity
                    try:
                        scores["vina_affinity"] = float(parts[1])
                        scores["cnn_score"] = float(parts[3])
                        scores["cnn_affinity"] = float(parts[4])
                        return scores
                    except (IndexError, ValueError):
                        pass
                elif len(parts) >= 2 and parts[0] == "1":
                    # cnn_scoring=none: mode affinity dist_lb dist_ub (no CNN columns)
                    try:
                        scores["vina_affinity"] = float(parts[1])
                        return scores
                    except (IndexError, ValueError):
                        pass
                break  # only want mode 1; stop after first data row
            break

    # -- Strategy 2: key=value fallback (older gnina versions) --
    m_aff = _KV_AFFINITY.search(raw_log)
    m_cs = _KV_CNN_SCORE.search(raw_log)
    m_ca = _KV_CNN_AFFINITY.search(raw_log)
    if m_aff or m_cs or m_ca:
        if m_aff:
            scores["vina_affinity"] = float(m_aff.group(1))
        if m_cs:
            scores["cnn_score"] = float(m_cs.group(1))
        if m_ca:
            scores["cnn_affinity"] = float(m_ca.group(1))
        return scores

    # -- Strategy 3: pipe-delimited table (some gnina builds) --
    for i, line in enumerate(lines):
        if _HDR_AFFINITY.search(line) and _HDR_CNN_SCORE.search(line):
            for data_line in lines[i + 1:]:
                if not data_line.strip() or data_line.strip().startswith("-"):
                    continue
                parts = _split_pipe_row(data_line)
                if not parts or parts[0] != "1":
                    continue
                try:
                    scores["vina_affinity"] = float(parts[1])
                    scores["cnn_score"] = float(parts[-2])
                    scores["cnn_affinity"] = float(parts[-1])
                    return scores
                except (IndexError, ValueError):
                    break
            break

    return scores


def _score_from_sdf(sdf_path: str) -> dict[str, float | None]:
    """Fallback: read scores from the docked SDF's embedded SD properties
    (``minimizedAffinity``, ``CNNscore``, ``CNNaffinity``), used when the
    stdout log didn't contain them (e.g. captured before writing completed)."""
    scores: dict[str, float | None] = {
        "vina_affinity": None, "cnn_score": None, "cnn_affinity": None,
    }
    if not sdf_path or not os.path.exists(sdf_path):
        return scores
    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        first = next((m for m in suppl if m is not None), None)
        if first is None:
            return scores

        def _prop(mol, *names):
            for n in names:
                if mol.HasProp(n):
                    try:
                        return float(mol.GetProp(n))
                    except ValueError:
                        pass
            return None

        scores["vina_affinity"] = _prop(
            first, "minimizedAffinity", "affinity", "docking_score", "Affinity")
        scores["cnn_score"] = _prop(first, "CNNscore", "cnn_score", "CNN_score")
        scores["cnn_affinity"] = _prop(first, "CNNaffinity", "cnn_affinity", "CNN_affinity")
    except Exception:
        pass
    return scores


def extract_scores(raw_log: str, output_path: str = "") -> dict[str, float | None]:
    """Extract one docking job's scores: parse Gnina's stdout log first,
    then fall back to the docked SDF's embedded properties for whichever
    fields are still missing (e.g. the log was truncated before Gnina
    finished writing it).

    This is the library-level building block behind `extract_results`; it
    takes plain strings, not a docking-result dict or any CLI/argparse
    object, so it can be called directly for a single job.

    Parameters
    ----------
    raw_log : str
        Gnina's captured stdout+stderr for one docking job (or ``""``/
        ``"TIMEOUT"`` if the job never produced output).
    output_path : str
        Path to the docked SDF for this job, used only as a fallback when
        the log didn't contain a score. Safe to omit or point at a
        nonexistent file -- the corresponding fields are simply ``None``.

    Returns
    -------
    dict[str, float | None]
        ``{"vina_affinity": ..., "cnn_score": ..., "cnn_affinity": ...}``.
        Any field Gnina didn't report (e.g. ``cnn_score``/``cnn_affinity``
        with ``--cnn-scoring none``) is ``None``.
    """
    parsed = _parse_gnina_log(raw_log or "")
    if None in parsed.values():
        fallback = _score_from_sdf(output_path)
        for key, val in fallback.items():
            if parsed.get(key) is None:
                parsed[key] = val
    return parsed


def _pose_quality_flag(status: str, cnn_score) -> str:
    if status != STATUS_OK:
        return "not_scored"
    if cnn_score is None:
        return "no_cnn_score"          # e.g. --cnn-scoring none: nothing to flag
    if cnn_score < LOW_CNN_SCORE_THRESHOLD:
        return "low_cnn_score"
    return "ok"


def merge_docked_sdf(df: pd.DataFrame, output_sdf: str) -> str:
    """Merge the best (first) pose of each docked ligand into a single,
    rank-ordered multi-molecule SDF for easy visual inspection.

    ``df`` must already be sorted in the desired (best-first) order and
    contain an ``output_sdf`` column pointing at each ligand's docked SDF,
    plus ``ID`` and rank-relevant score columns.
    """
    writer = Chem.SDWriter(output_sdf)
    n_written = 0
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        sdf_path = getattr(row, "output_sdf", None)
        if not sdf_path or not os.path.exists(sdf_path):
            continue
        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            first = next((m for m in suppl if m is not None), None)
        except Exception:
            first = None
        if first is None:
            continue
        first.SetProp("_Name", f"rank{rank}_{getattr(row, 'ID', rank)}")
        first.SetProp("Rank", str(rank))
        if getattr(row, "vina_affinity", None) is not None:
            first.SetProp("vina_affinity", str(row.vina_affinity))
        if getattr(row, "cnn_score", None) is not None:
            first.SetProp("cnn_score", str(row.cnn_score))
        if getattr(row, "cnn_affinity", None) is not None:
            first.SetProp("cnn_affinity", str(row.cnn_affinity))
        writer.write(first)
        n_written += 1
    writer.close()
    logger.info("Merged %d ranked poses into '%s'.", n_written, output_sdf)
    return output_sdf


def extract_results(
    docking_results: list[dict],
    input_path: str | None = None,
    output_csv: str = "results_summary.csv",
    output_excel: str = "top_results.xlsx",
    top_n: int | str = 100,
    merged_sdf: str | None = None,
    id_col: str = "ID",
) -> pd.DataFrame:
    """Parse Gnina logs, build a results DataFrame, save as CSV, and write
    the top-``top_n`` compounds to an Excel sheet that includes the original
    ID column from ``input_path``.

    Ranking: by ``cnn_affinity`` (descending) then ``vina_affinity``
    (ascending) when CNN scores are available anywhere in this run; when
    none are (e.g. ``--cnn-scoring none``), ranking falls back to
    ``vina_affinity`` alone so the export is never empty just because a
    column doesn't exist for this run (item 2).

    Every row is kept, including failed/timed-out jobs (marked in the
    ``status`` column) and low-confidence poses (marked in
    ``pose_quality_flag``, from ``cnn_score`` -- see item 15); neither is
    used as a reason to drop a compound from the export.

    Parameters
    ----------
    docking_results : list[dict]
        Entries as returned by `gnina_vs.docking.run_gnina_docking` (or
        built by `recover_results`). Each needs at least ``mol_name``,
        ``smiles``, ``index``, ``ligand_id``, ``output_path``, ``raw_log``;
        ``status`` defaults to ``"ok"`` if absent (older-style entries).
    input_path : str | None
        Original ligand Excel/CSV, to recover the ID column.
    output_csv, output_excel : str
        Destination paths.
    top_n : int | str
        Number of top compounds to export to Excel. ``0`` or ``"all"``
        exports every compound (including failed ones).
    merged_sdf : str | None
        If given, write the best pose of every *successfully scored*
        exported compound to this ranked multi-molecule SDF.
    id_col : str
        Name of the identifier column in ``input_path``.

    Returns
    -------
    pd.DataFrame
        Full results table (all compounds, sorted).
    """
    logger.info("=" * 60)
    logger.info("STEP 4 -- Results Extraction & Export")
    logger.info("=" * 60)

    n_export_requested = parse_top_n(top_n)

    # -- Load original input table to recover the ID column --
    id_map: dict[str, str] = {}  # mol_index -> ID
    if input_path and os.path.exists(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        orig_df = pd.read_excel(input_path) if ext in (".xlsx", ".xls") else pd.read_csv(input_path)
        if id_col in orig_df.columns:
            for row_idx, row in orig_df.iterrows():
                id_map[str(row_idx)] = str(row[id_col])
        else:
            logger.warning(
                "  '%s' column not found in '%s'. The ID column in the "
                "output will be populated from the embedded LigandID.",
                id_col, input_path,
            )
    elif input_path:
        logger.warning(
            "  Original input file '%s' not found -- ID column will be "
            "populated from embedded LigandID values.", input_path,
        )

    # -- Build results DataFrame --
    rows = []
    for entry in docking_results:
        status = entry.get("status", STATUS_OK)  # older-style entries: assume ok

        if status in _TERMINAL_FAILURE_STATUSES:
            # Never read scores for a job that failed/timed out/errored, even
            # if output_path happens to point at a leftover file on disk
            # (docking.py already deletes it; this is belt and braces for
            # callers that build entries some other way -- item 5).
            parsed = {"vina_affinity": None, "cnn_score": None, "cnn_affinity": None}
        else:
            parsed = extract_scores(entry.get("raw_log", ""), entry.get("output_path", ""))

        mol_index = str(entry["index"])
        lig_id = id_map.get(mol_index, entry.get("ligand_id", mol_index))
        compound_id = entry.get("compound_id") or make_compound_id(entry["mol_name"], mol_index)

        rows.append({
            "ID": lig_id,
            "compound_id": compound_id,
            "compound_index": entry["index"],
            "mol_name": entry["mol_name"],
            "smiles": entry.get("smiles", ""),
            "status": status,
            "error": entry.get("error", ""),
            "vina_affinity": parsed["vina_affinity"],
            "cnn_score": parsed["cnn_score"],
            "cnn_affinity": parsed["cnn_affinity"],
            "output_sdf": entry.get("output_path", ""),
        })

    df = pd.DataFrame(rows)
    df["pose_quality_flag"] = [
        _pose_quality_flag(s, c) for s, c in zip(df["status"], df["cnn_score"])
    ]

    have_cnn = df["cnn_affinity"].notna().any()
    sort_cols = ["cnn_affinity", "vina_affinity"] if have_cnn else ["vina_affinity"]
    sort_asc = [False, True] if have_cnn else [True]
    df.sort_values(by=sort_cols, ascending=sort_asc, inplace=True, na_position="last")

    # -- Save full results as CSV (every row, including failures) --
    df.to_csv(output_csv, index=False)
    logger.info("Full results saved to '%s' (%d entries).", output_csv, len(df))

    # -- Export top-N (or all) to Excel --
    # A compound only needs a *ranking* score to be exportable: cnn_affinity
    # when CNN scoring produced one anywhere in this run, otherwise
    # vina_affinity. This is what keeps --cnn-scoring none from exporting
    # zero rows (item 2). Failed/timed-out jobs (no score at all) are still
    # visible in the CSV but are not part of a "top N" ranking.
    rank_col = "cnn_affinity" if have_cnn else "vina_affinity"
    successful = df.dropna(subset=[rank_col])
    n_export = len(successful) if n_export_requested == 0 else n_export_requested
    export_df = successful.head(n_export).copy()

    # Ensure 'ID' is the first column
    cols = ["ID"] + [c for c in export_df.columns if c != "ID"]
    export_df = export_df[cols]

    sheet_name = "AllResults" if n_export >= len(successful) else f"Top{n_export}"
    export_df.to_excel(output_excel, index=False, sheet_name=sheet_name)

    # Auto-fit column widths for readability
    try:
        import openpyxl
        from openpyxl.utils import get_column_letter
        wb = openpyxl.load_workbook(output_excel)
        ws = wb.active
        for col_cells in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in col_cells
            )
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 4, 60)
        wb.save(output_excel)
    except Exception as exc:
        logger.debug("  Column auto-fit skipped: %s", exc)

    logger.info(
        "%d compounds saved to '%s' (sheet: '%s').",
        len(export_df), output_excel, sheet_name,
    )

    if merged_sdf:
        merge_docked_sdf(export_df, merged_sdf)

    # -- Console summary --
    n_failed = int((df["status"].isin(_TERMINAL_FAILURE_STATUSES)).sum())
    logger.info("Successfully scored: %d/%d compounds (%d failed/timed out).",
               len(successful), len(df), n_failed)
    if not successful.empty:
        best = successful.iloc[0]
        if have_cnn and pd.notna(best["cnn_affinity"]):
            logger.info(
                "  Top compound  ID=%s  |  CNN Affinity: %.3f  |  CNN Score: %s  |  "
                "Vina: %s kcal/mol",
                best["ID"], best["cnn_affinity"],
                f"{best['cnn_score']:.3f}" if pd.notna(best["cnn_score"]) else "n/a",
                f"{best['vina_affinity']:.3f}" if pd.notna(best["vina_affinity"]) else "n/a",
            )
        else:
            logger.info("  Top compound  ID=%s  |  Vina: %.3f kcal/mol",
                       best["ID"], best["vina_affinity"])

    return df


# --------------------------------------------------------------------------- #
# Recovery -- re-score from existing docked SDF files
# --------------------------------------------------------------------------- #
def recover_results(
    ligands_sdf: str,
    out_dir: Path | str = "docking_outputs",
    input_path: str | None = None,
    output_csv: str = "results_summary.csv",
    output_excel: str = "top_results.xlsx",
    top_n: int | str = 100,
    merged_sdf: str | None = None,
    id_col: str = "ID",
) -> pd.DataFrame:
    """Re-extract results from already-docked SDF files without re-running
    Gnina.

    Use this when docking completed successfully but score extraction
    needs to be re-run (e.g. after fixing a log-parsing issue), or simply
    to regenerate a report with different `--top-n` / export settings.
    Reads the compound list from the prepared ligands SDF, locates each
    ``<compound_id>_docked.sdf`` in ``out_dir`` (the unique, filesystem-safe
    id written by `gnina_vs.ligand.prepare_ligands`, not the display name --
    two compounds can share a display name but never a ``compound_id``), and
    reads scores from the SDF properties Gnina embeds in every output file.
    """
    logger.info("=" * 60)
    logger.info("RECOVERY -- Re-scoring from existing docked SDF files")
    logger.info("=" * 60)

    out_dir = Path(out_dir)

    if not os.path.exists(ligands_sdf):
        raise FileNotFoundError(f"Prepared ligands SDF not found: {ligands_sdf}")

    supplier = Chem.SDMolSupplier(ligands_sdf, removeHs=False)
    molecules = [mol for mol in supplier if mol is not None]
    logger.info("  %d ligands found in '%s'.", len(molecules), ligands_sdf)
    if not molecules:
        raise ValueError(f"No valid molecules found in '{ligands_sdf}'.")

    docking_results = []
    missing = 0
    for mol in tqdm(molecules, desc="Reading docked SDFs", unit="mol", colour="cyan"):
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown"
        smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else ""
        mol_index = mol.GetProp("MolIndex") if mol.HasProp("MolIndex") else "-1"
        lig_id = mol.GetProp("LigandID") if mol.HasProp("LigandID") else mol_index
        compound_id = (mol.GetProp("CompoundID") if mol.HasProp("CompoundID")
                       else make_compound_id(mol_name, mol_index))

        out_sdf = out_dir / f"{compound_id}_docked.sdf"
        status = STATUS_OK
        if not out_sdf.exists():
            missing += 1
            status = STATUS_NO_OUTPUT
            raw_log = "TIMEOUT"
        else:
            raw_log = ""
            try:
                suppl2 = Chem.SDMolSupplier(str(out_sdf), removeHs=False)
                first = next((m for m in suppl2 if m is not None), None)
                if first is not None:
                    aff = first.GetProp("minimizedAffinity") if first.HasProp("minimizedAffinity") else ""
                    cs = first.GetProp("CNNscore") if first.HasProp("CNNscore") else ""
                    ca = first.GetProp("CNNaffinity") if first.HasProp("CNNaffinity") else ""
                    # Reconstruct a minimal gnina-style table so _parse_gnina_log works
                    raw_log = (
                        "mode |  affinity  |  intramol  |    CNN     |   CNN\n"
                        "     | (kcal/mol) | (kcal/mol) | pose score | affinity\n"
                        "-----+------------+------------+------------+----------\n"
                        f"    1    {aff}       0.00    {cs}    {ca}\n"
                    )
                else:
                    status = STATUS_NO_OUTPUT
            except Exception:
                status = STATUS_NO_OUTPUT

        docking_results.append({
            "mol_name": mol_name,
            "compound_id": compound_id,
            "smiles": smiles,
            "index": mol_index,
            "ligand_id": lig_id,
            "output_path": str(out_sdf),
            "raw_log": raw_log,
            "status": status,
        })

    if missing:
        logger.warning("  %d docked SDF files were missing (will appear as unscored).", missing)

    return extract_results(
        docking_results=docking_results,
        input_path=input_path,
        output_csv=output_csv,
        output_excel=output_excel,
        top_n=top_n,
        merged_sdf=merged_sdf,
        id_col=id_col,
    )
