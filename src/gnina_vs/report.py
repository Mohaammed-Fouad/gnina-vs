"""
gnina_vs.report
===============
Phase 7b: one self-contained HTML file for triage and for showing a group.

The requirement that decides the design: the file has to work when someone
double-clicks it with no server and, in the general case, no network. That
rules out more of the ecosystem than its documentation suggests, so the
options were measured rather than taken on trust (numbers from mols2grid
2.2.0, itables 2.9.1, bokehmol 0.2.0):

====================================  ==========  =================================
option                                ext. refs   verdict
====================================  ==========  =================================
mols2grid, interactive (default)              4   NOT offline: pulls RDKit.js from
                                                  unpkg plus jQuery, bootstrap,
                                                  list.js. Opens, but structures
                                                  never render and sorting is dead.
mols2grid, template="static",                 2   images baked in and RDKit.js
  prerender=True                                  gone, but still needs jQuery and
                                                  bootstrap -- and "static" means
                                                  no sorting or filtering.
bokehmol, hover.rdkit / smiles_drawer         2   NOT offline even with
                                                  `resources=INLINE`: fetches
                                                  RDKit.js or smiles-drawer from
                                                  unpkg at runtime.
itables, connected=False, bundle              0   genuinely offline, with sorting,
  inlined                                         filtering and pagination.
====================================  ==========  =================================

So this module uses **itables with its DataTables bundle inlined, and
structures pre-rendered by RDKit into inline SVG**. Rendering the depictions
here rather than in the browser is what removes the RDKit.js dependency, and
it is the only arrangement measured at zero external requests.

Cost of that choice: file size scales with the number of molecules, because
each depiction is real SVG in the document. The DataTables bundle is a fixed
~950 KB; each molecule adds roughly 2-4 KB. `max_rows` exists for that
reason, and `write_report` logs the finished size so it is never a surprise.

What this deliberately does not do
----------------------------------
No 3D viewer and no interaction diagram. 3Dmol.js can be inlined, and a
parent-versus-metabolite ProLIF interaction-difference table is the single
most informative artifact available, but neither is wired up here: ProLIF
is not a dependency of this project and the interaction diff depends on a
deterministically protonated receptor that the pipeline does not yet
guarantee. Adding either without that guarantee would produce a figure whose
differences are partly an artifact of preparation.

No chart of docking score either. The dominant uncertainty in a docking
score is method bias rather than sampling spread, so the page reports banded
values and states the caveat in the header instead of drawing a precise-
looking plot over an imprecise quantity.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)

_XML_DECL = re.compile(r"<\?xml[^>]*\?>")

DEFAULT_MAX_ROWS = 500
SVG_SIZE = (190, 130)

# Columns shown in the triage table, in order, with the label used in the
# header. Anything absent from the frame is skipped rather than faked.
TRIAGE_COLUMNS = [
    ("depiction", "Structure"),
    ("compound_id", "Compound"),
    ("target_id", "Target"),
    ("role", "Role"),
    ("met_reaction", "Transformation"),
    ("dcmp_cnn_band_vs_parent", "vs parent"),
    ("dcmp_cnn_affinity_vs_parent", "ΔCNNaff"),
    ("rank_band_in_group", "Rank band"),
    ("le_cnn_per_heavy_atom", "LE (CNN)"),
    ("exposure_plausibility", "Exposure"),
    ("flag_conjugate", "Conjugate"),
    ("alert_brenk_n", "Brenk"),
    ("alert_pains_a_n", "PAINS-A"),
    ("ad_in_domain", "In AD"),
    ("qc_status", "QC"),
]


# --------------------------------------------------------------------------- #
# Depictions
# --------------------------------------------------------------------------- #
def molecule_svg(smiles, width: int = SVG_SIZE[0], height: int = SVG_SIZE[1]) -> str:
    """Compact inline SVG for one SMILES, or an empty string if it will not
    parse. Rendered here so the page needs no RDKit.js at view time."""
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False          # let the page's theme show through
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    except Exception as exc:
        logger.debug("  [report] could not depict '%s' (%s).", smiles[:40], exc)
        return ""

    # Strip the XML declaration so the fragment can be embedded inline.
    svg = _XML_DECL.sub("", svg)
    # RDKit writes real newlines inside path `d` attributes. The table cell is
    # serialised through JSON on the way into the page, which turns each of
    # those into a literal backslash-n and makes the browser reject the whole
    # path ("Expected path command"). Every molecule then renders blank. So
    # collapse all whitespace before embedding -- SVG treats it as separators.
    return " ".join(svg.split())


# --------------------------------------------------------------------------- #
# The triage frame
# --------------------------------------------------------------------------- #
def build_triage_frame(tables: dict[str, pd.DataFrame], max_rows: int | None = DEFAULT_MAX_ROWS,
                       with_depictions: bool = True) -> pd.DataFrame:
    """Join `screen` to `compounds` and shape it for reading.

    Sorted so a parent sits directly above its own metabolites, which is the
    comparison the table exists to support. Neither itables nor mols2grid has
    row grouping, so the hierarchy is carried by that composite sort plus a
    `role` column rather than by nesting.
    """
    screen, compounds = tables["screen"], tables["compounds"]
    df = screen.merge(compounds, on="compound_id", how="left", suffixes=("", "_cmp"))

    if "is_parent" in df.columns:
        df["role"] = df["is_parent"].map({True: "parent", False: "metabolite"}).astype("string")
    else:
        df["role"] = pd.NA

    sort_cols = [c for c in ("target_id", "parent_compound_id") if c in df.columns]
    if "is_parent" in df.columns:
        df["_parent_first"] = (~df["is_parent"].fillna(False)).astype(int)
        sort_cols.append("_parent_first")
    if "dcmp_cnn_affinity_vs_parent" in df.columns:
        sort_cols.append("dcmp_cnn_affinity_vs_parent")
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True] * len(sort_cols), na_position="last")
    df = df.drop(columns=["_parent_first"], errors="ignore")

    truncated = 0
    if max_rows is not None and len(df) > max_rows:
        truncated = len(df) - max_rows
        df = df.head(max_rows)

    if with_depictions:
        smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else None
        df["depiction"] = (
            [molecule_svg(s) for s in df[smiles_col]] if smiles_col else ""
        )

    present = [(c, label) for c, label in TRIAGE_COLUMNS if c in df.columns]
    out = df[[c for c, _ in present]].copy()
    out.columns = [label for _, label in present]
    out.attrs["truncated"] = truncated
    out.attrs["total_rows"] = len(screen)
    return out


def _format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Round scores to one decimal and render booleans as words.

    One decimal, not three: reporting more precision than the scoring
    function resolves is an implied-precision claim. The bands beside these
    columns are the quantity meant to be read.
    """
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_bool_dtype(series) or str(series.dtype) == "boolean":
            out[col] = series.map({True: "yes", False: "no"}).fillna("--").astype("string")
        elif str(series.dtype) in ("Float64", "float64"):
            out[col] = series.map(lambda v: "--" if pd.isna(v) else f"{float(v):.1f}").astype("string")
        elif str(series.dtype) in ("Int64", "int64"):
            out[col] = series.map(lambda v: "--" if pd.isna(v) else str(int(v))).astype("string")
        elif col != "Structure":
            out[col] = series.astype("string").fillna("--")
    return out


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
_CSS = """
:root{--bg:#fbfcfc;--surface:#fff;--fg:#1a2322;--muted:#5f6d6b;--line:#dfe5e4;
--accent:#0f6e6a;--warn:#8f5513;--warnbg:#faf0e2;--worse:#b4541f;--better:#1f5f96}
@media (prefers-color-scheme:dark){:root{--bg:#101615;--surface:#192120;--fg:#e4eae7;
--muted:#93a29e;--line:#2b3733;--accent:#4fb3ab;--warn:#d79b4e;--warnbg:#2d2415;
--worse:#e08a52;--better:#6aa9db}}
*{box-sizing:border-box}
body{margin:0;padding:24px 20px 64px;background:var(--bg);color:var(--fg);
font:14px/1.55 "IBM Plex Sans",-apple-system,Segoe UI,system-ui,sans-serif}
.wrap{max-width:1500px;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 6px;font-weight:600;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 20px;font-size:.92rem}
.meta{display:flex;flex-wrap:wrap;gap:8px 28px;padding:14px 16px;background:var(--surface);
border:1px solid var(--line);border-radius:6px;margin:0 0 16px;font-size:.86rem}
.meta div{min-width:0}.meta b{display:block;color:var(--muted);font-weight:500;
font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:2px}
.caveat{border-left:3px solid var(--warn);background:var(--warnbg);padding:13px 16px;
margin:0 0 20px;font-size:.87rem;border-radius:0 4px 4px 0}
.caveat p{margin:0 0 7px}.caveat p:last-child{margin:0}
.caveat strong{color:var(--warn)}
table.dataTable{font-size:.84rem!important;background:var(--surface)}
table.dataTable td,table.dataTable th{color:var(--fg)!important;
border-color:var(--line)!important;vertical-align:middle!important}
table.dataTable th{font-size:.73rem!important;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted)!important}
table.dataTable svg{display:block}
.dt-container{color:var(--fg)}
input,select{background:var(--surface);color:var(--fg);border:1px solid var(--line);
border-radius:4px;padding:3px 6px}
footer{margin-top:28px;padding-top:16px;border-top:1px solid var(--line);
color:var(--muted);font-size:.8rem}
"""


def _header_html(title: str, meta: dict, caveats: list[str]) -> str:
    cells = "".join(
        f"<div><b>{html.escape(str(k))}</b>{html.escape(str(v))}</div>"
        for k, v in meta.items()
    )
    notes = "".join(f"<p>{c}</p>" for c in caveats)
    return (
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='sub'>Parents and their predicted metabolites, ranked by docking score &mdash; "
        f"which correlates poorly with measured affinity. Read the bands, not the numbers.</p>"
        f"<div class='meta'>{cells}</div>"
        f"<div class='caveat'>{notes}</div>"
    )


def write_report(tables: dict[str, pd.DataFrame], output_html,
                 title: str = "Metabolite screening triage",
                 max_rows: int | None = DEFAULT_MAX_ROWS,
                 manifest: dict | None = None,
                 with_depictions: bool = True) -> str:
    """Write one self-contained HTML file and return its path.

    Raises ImportError if itables is missing, because without it there is no
    offline-capable interactive table and silently emitting a static one
    would misrepresent what the file does.
    """
    try:
        import itables.options as itables_options
        from itables.javascript import generate_init_offline_itables_html, to_html_datatable
    except ImportError as exc:
        raise ImportError(
            "gnina_vs.report needs itables: pip install gnina-vs[report]. "
            "It is the only option measured at zero external requests, which is "
            "what lets the file be opened without a server or a network connection."
        ) from exc

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    frame = build_triage_frame(tables, max_rows=max_rows, with_depictions=with_depictions)
    display = _format_for_display(frame)

    manifest = manifest or {}
    meta = {
        "Compounds": f"{len(tables['compounds'])}",
        "Rows shown": f"{len(display)} of {frame.attrs.get('total_rows', len(display))}",
        "Targets": f"{tables['screen']['target_id'].nunique()}",
        "Run": str(manifest.get("run_id", "--")),
        "Generated": str(manifest.get("started_utc", "--")),
    }

    caveats = [
        "<strong>Docking scores are not affinities.</strong> &Delta;CNNaff is shown to one "
        "decimal and the &ldquo;vs parent&rdquo; column collapses any change below the scoring "
        "function&rsquo;s resolution into <em>no_meaningful_change</em>. There are deliberately "
        "no error bars: the dominant uncertainty here is method bias, not sampling spread, so an "
        "error bar would measure re-docking variance while reading as distance to the truth.",
        "<strong>Rank bands, not positions.</strong> A tier is reported rather than an integer "
        "rank, because neighbouring positions are not resolved by the score.",
        "<strong>Columns are not commensurable.</strong> A Brenk count is a deterministic "
        "substructure match; &Delta;CNNaff is weakly predictive; an out-of-domain ADMET value is "
        "close to uninformative. They are shown side by side for convenience, not because they "
        "carry equal weight.",
        "<strong>Nothing has been filtered.</strong> Failed and unparseable compounds are kept "
        "as rows with a reason in QC, so counts and denominators stay honest.",
    ]
    if frame.attrs.get("truncated"):
        caveats.append(
            f"<strong>Truncated.</strong> {frame.attrs['truncated']} further rows are not shown "
            f"here; the complete set is in <code>tables/screen.parquet</code>."
        )

    body = to_html_datatable(
        display, connected=False, allow_html=True,
        display_logo_when_loading=False,
        # itables defaults to maxBytes='64KB' and SILENTLY DOWNSAMPLES past
        # it -- rows just disappear from the table. An inline depiction is
        # ~6-7 KB, so the cap is crossed at about ten molecules and the
        # report would drop compounds without saying so. That would break
        # this project's one hard rule, so downsampling is disabled outright
        # and size is controlled by `max_rows` instead, which reports what it
        # truncated.
        maxBytes=0, maxRows=0,
        paging=True, lengthMenu=[25, 50, 100, 250], order=[],
        classes="display nowrap compact",
    )
    head = generate_init_offline_itables_html(itables_options.dt_bundle)

    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title>"
        f"<style>{_CSS}</style>{head}</head><body><div class='wrap'>"
        f"{_header_html(title, meta, caveats)}{body}"
        "<footer>Generated by gnina-vs. Self-contained: no network access required. "
        "Scores are docking scores, not measured affinities.</footer>"
        "</div></body></html>"
    )
    output_html.write_text(page, encoding="utf-8")
    size_kb = output_html.stat().st_size / 1024
    logger.info("  [report] %s (%.0f KB, %d rows).", output_html, size_kb, len(display))
    if size_kb > 20_000:
        logger.warning(
            "  [report] %.0f KB is large for an emailed file -- lower max_rows or pass "
            "with_depictions=False.", size_kb,
        )
    return str(output_html)


def external_references(html_text: str) -> list[str]:
    """Every absolute http(s) URL the page would fetch when opened.

    Used by the test suite to assert the file is genuinely self-contained;
    useful by hand too, before sending the file to anyone.
    """
    return sorted(set(re.findall(r'(?:src|href)\s*=\s*"(https?://[^"]+)"', html_text)))
