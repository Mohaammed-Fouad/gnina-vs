# Gnina-VS: Automated Virtual Screening

A modular virtual-screening pipeline: **protein preparation** (with automatic RCSB PDB fetching), **ligand preparation**, **three-tier active-site detection**, **parallel Gnina docking**, and **results reporting**, packaged as a clean, installable Python CLI (`gnina-vs`).

This is a refactor of a single-file prototype script into a proper package: every step is its own module (`protein`, `ligand`, `pocket`, `docking`, `reporter`), every previously-hardcoded global (input paths, box size, worker counts, Gnina parameters) is now a CLI flag, and the whole thing installs via `pip install -e .` with a real console-script entry point.

---

## Table of Contents

- [Overview](#overview)
- [Workflow](#workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Run the full pipeline](#1-run-the-full-pipeline)
  - [2. Auto-fetch a protein structure from RCSB](#2-auto-fetch-a-protein-structure-from-rcsb)
  - [3. Manual active-site overrides](#3-manual-active-site-overrides)
  - [4. Recovery mode](#4-recovery-mode)
  - [All available options](#all-available-options)
- [Active-Site Detection Priority](#active-site-detection-priority)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Notes and Limitations](#notes-and-limitations)
- [License](#license)

---

## Overview

Given a receptor (a local PDB file or a bare PDB ID) and a spreadsheet of compounds (SMILES), the pipeline:

1. **Fetches** the receptor from RCSB PDB if a 4-character PDB ID is given instead of a file path (`protein.fetch_pdb_from_rcsb`), then **prepares** it: strips waters and crystallization artefacts, keeps ATOM/ANISOU records (plus any HETATM residues you explicitly ask to retain, e.g. a metal cofactor), and adds polar hydrogens via OpenBabel when available (falling back cleanly if it isn't).
2. **Loads** compounds from an Excel or CSV file, generates 3-D conformers with ETKDGv3, minimizes them with MMFF94, and writes everything to a single multi-molecule SDF.
3. **Detects the docking box center**, in priority order: explicit `--center-x/y/z` coordinates, a centroid of named catalytic residues (`--active-site-residues`), a co-crystallised ligand (Tier 1, now excluding a much wider set of common cofactors and crystallization additives so it doesn't mistake them for the real ligand), `fpocket` (Tier 2), or a built-in geometric grid-cavity detector (Tier 3). A `.pml` PyMOL script is always written so you can visually check the box before committing CPU hours.
4. **Docks** every ligand against the receptor in parallel with Gnina (`ThreadPoolExecutor`), with fully configurable CNN scoring mode, exhaustiveness, number of modes, box size, and worker/CPU counts.
5. **Extracts** scores from Gnina's stdout (or, as a fallback, from the properties Gnina embeds in the output SDF), exports a ranked CSV + Excel report (flexible top-N, or all compounds), and merges the best pose of every exported compound into a single ranked SDF for easy visual inspection.

A `recover` mode lets you re-run steps 4-5's scoring/export logic against already-docked SDF files, without re-docking anything.

---

## Workflow

```
                 ┌──────────────────────┐
 --protein  ───► │ 0. protein.py        │  RCSB fetch (if PDB ID) + strip/clean + add H
 (path/PDB ID)   └──────────┬───────────┘
                            │ prepared receptor PDB
                 ┌──────────▼───────────┐
 --ligands  ───► │ 1. ligand.py         │  SMILES → 3-D conformer → MMFF94 minimize
 (.xlsx/.csv)    └──────────┬───────────┘
                            │ prepared_ligands.sdf
                 ┌──────────▼───────────┐
                 │ 2. pocket.py         │  manual override → Tier 1/2/3 → .pml box script
                 └──────────┬───────────┘
                            │ (x, y, z) center
                 ┌──────────▼───────────┐
                 │ 3. docking.py        │  parallel Gnina docking (ThreadPoolExecutor)
                 └──────────┬───────────┘
                            │ per-ligand docked SDF + raw log
                 ┌──────────▼───────────┐
                 │ 4. reporter.py       │  score extraction → CSV/Excel → merged ranked SDF
                 └──────────────────────┘
```

| Stage | Module | Description | Key Library |
|---|---|---|---|
| Protein prep | `protein.py` | RCSB auto-fetch, strip waters/artefacts, add H | `urllib.request`, OpenBabel |
| Ligand prep | `ligand.py` | Load SMILES, embed 3-D, MMFF94 minimize | `rdkit`, `pandas` |
| Pocket detection | `pocket.py` | Manual override / 3-tier auto-detection, PyMOL script | `numpy`, `scipy` (optional), `fpocket` (optional) |
| Docking | `docking.py` | Parallel Gnina execution | `subprocess`, `concurrent.futures` |
| Reporting | `reporter.py` | Score parsing, CSV/Excel export, merged SDF, recovery | `pandas`, `openpyxl`, `rdkit` |

---

## Requirements

- **Python 3.9+**
- **Gnina**, installed and on `PATH` (or pass `--gnina-exec /path/to/gnina`): https://github.com/gnina/gnina
- **OpenBabel** (optional, improves protein-H placement): `conda install -c conda-forge openbabel`
- **fpocket** (optional, enables Tier 2 pocket detection): https://fpocket.sourceforge.net/
- **PyMOL** (optional, only needed to actually open the generated `.pml` box-visualization script)
- An internet connection, only if using a bare PDB ID with `--protein` instead of a local file

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Mohaammed-Fouad/gnina-vs.git
   cd gnina-vs
   ```

2. **(Recommended) Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows PowerShell
   ```

3. **Install the package (editable install, with console-script entry point):**

   ```bash
   pip install -e .
   ```

   This installs the `gnina-vs` command. You can also always run it as `python -m gnina_vs` without installing.

---

## Usage

### 1. Run the full pipeline

```bash
gnina-vs run --protein receptor.pdb --ligands compounds.xlsx
```

Outputs land in `gnina_vs_output/` by default (see [Output Files](#output-files)).

### 2. Auto-fetch a protein structure from RCSB

Pass a bare 4-character PDB ID instead of a file path, and it's downloaded automatically:

```bash
gnina-vs run --protein 1ABC --ligands compounds.xlsx --outdir results
```

### 3. Manual active-site overrides

Skip auto-detection with explicit coordinates:

```bash
gnina-vs run --protein receptor.pdb --ligands compounds.xlsx \
    --center-x 12.5 --center-y 8.3 --center-z -4.1
```

...or with named catalytic residues instead (their centroid is used):

```bash
gnina-vs run --protein receptor.pdb --ligands compounds.xlsx \
    --active-site-residues "TYR120,GLU150,HIS95"
```

Either way, open the generated `<outdir>/docking_box.pml` in PyMOL alongside the prepared receptor to sanity-check the box before docking:

```bash
pymol gnina_vs_output/receptor_prepared.pdb gnina_vs_output/docking_box.pml
```

### 4. Recovery mode

Re-score and re-export from an already-completed docking run, without re-docking:

```bash
gnina-vs recover \
    --ligands-sdf gnina_vs_output/prepared_ligands.sdf \
    --docking-outdir gnina_vs_output/docking_outputs \
    --input compounds.xlsx \
    --outdir gnina_vs_output \
    --top-n all
```

### 5. Configure Gnina and export more/fewer compounds

```bash
gnina-vs run --protein receptor.pdb --ligands compounds.xlsx \
    --cnn-scoring refine --exhaustiveness 16 --num-modes 20 \
    --workers 8 --cpu-per-worker 2 \
    --top-n 250
```

### All available options

```bash
gnina-vs run --help
gnina-vs recover --help
```

**Input files** (`run`)

| Flag | Description | Default |
|---|---|---|
| `--protein` | Receptor PDB path, or a bare 4-character PDB ID to auto-fetch from RCSB | **required** |
| `--ligands` | `.xlsx`/`.xls`/`.csv` file with a SMILES column | **required** |
| `--smiles-col` | SMILES column name | `SMILES` |
| `--name-col` | Compound name column name (optional; falls back to `compound_<row>`) | `Name` |

**Protein preparation**

| Flag | Description | Default |
|---|---|---|
| `--keep-hetatm` | Comma-separated HETATM residue names to retain as cofactors (e.g. `ZN` or `ZN,MG`) | *(strip all)* |
| `--no-add-hydrogens` | Skip adding polar hydrogens via OpenBabel | off |
| `--ph` | Protonation pH passed to OpenBabel | `7.4` |

**Active site detection**

| Flag | Description | Default |
|---|---|---|
| `--center-x`, `--center-y`, `--center-z` | Explicit box center (must all three be given together) | *(none)* |
| `--active-site-residues` | Comma-separated residues to center on, e.g. `"TYR120,GLU150"` | *(none)* |

**Gnina docking parameters**

| Flag | Description | Default |
|---|---|---|
| `--gnina-exec` | Path to the `gnina` executable | `gnina` |
| `--box-size` | Docking box edge length (Å, cubic box) | `25.0` |
| `--cnn-scoring` | `none`, `rescore`, `refine`, or `full` | `rescore` |
| `--exhaustiveness` | Vina/Gnina search exhaustiveness | `8` |
| `--num-modes` | Binding modes generated per ligand | `9` |
| `--workers` | Ligands docked concurrently | `4` |
| `--cpu-per-worker` | `--cpu` passed to each Gnina process (total threads ≈ `workers × cpu-per-worker`) | `2` |
| `--timeout` | Per-ligand docking timeout (seconds) | `300` |

**Output** (`run` and `recover`)

| Flag | Description | Default |
|---|---|---|
| `--outdir` | Directory for all outputs | `gnina_vs_output` |
| `--top-n` | Compounds to export; `0` or `all` exports everything scored | `100` |
| `--id-col` | Identifier column name in the ligand input file | `ID` |
| `--no-merged-sdf` | Skip writing the single ranked multi-molecule SDF | off |
| `-v`, `--verbose` | Enable DEBUG logging | off |

**Recovery-only** (`recover`)

| Flag | Description | Default |
|---|---|---|
| `--ligands-sdf` | Prepared ligands SDF from a previous `run` | **required** |
| `--docking-outdir` | Directory with `<mol_name>_docked.sdf` files from a previous `run` | **required** |
| `--input` | Original ligand Excel/CSV, to recover the ID column | *(none)* |

---

## Active-Site Detection Priority

`pocket.py` resolves the docking box center in this order, stopping at the first that applies:

1. **`--center-x/y/z`** -- explicit coordinates, always wins if given.
2. **`--active-site-residues`** -- centroid of every atom of the named residues (searched against the *original*, un-stripped PDB so full side chains are present).
3. **Tier 1 -- co-crystallised ligand centroid.** Searches the *raw* PDB (not the prepared one, since protein preparation strips HETATM records). The exclusion list now also skips common cofactors (`ATP`, `ADP`, `AMP`, `GTP`, `GDP`, `NAD`, `NADP`, `FAD`, `FMN`, `HEM`, `PLP`) and crystallization additives (`PEG`, `EDO`, `GOL`, `OLB`, `BOG`, `MPD`, `DMS`), so a bound coenzyme or cryo-protectant molecule isn't mistaken for the ligand you actually want to dock against.
4. **Tier 2 -- `fpocket`.** Geometric pocket finder; used if installed and Tier 1 found nothing.
5. **Tier 3 -- built-in grid-cavity detector.** Pure Python + NumPy occupancy-grid flood-fill, with an optional scipy-accelerated dilation step; scores candidate cavities by `volume^1.5 × burial^0.5 × sphericity`.

Whichever method wins, a `.pml` script is written showing the resulting box as a wireframe cube plus a center pseudo-atom, so you can inspect it in PyMOL before docking.

---

## Project Structure

```
gnina-vs/
├── pyproject.toml
├── README.md
└── src/
    └── gnina_vs/
        ├── __init__.py
        ├── __main__.py      # enables `python -m gnina_vs`
        ├── cli.py           # argparse CLI: run / recover subcommands
        ├── protein.py       # RCSB fetcher + protein preparation (Step 0)
        ├── ligand.py        # SMILES loading + 3-D conformer generation (Step 1)
        ├── pocket.py        # 3-tier active-site detector + PyMOL box script (Step 2)
        ├── docking.py       # parallel Gnina docking (Step 3)
        └── reporter.py      # score extraction, CSV/Excel export, recovery (Step 4)
```

---

## Output Files

Running `gnina-vs run --outdir <outdir>` generates:

- `<outdir>/<PDB_ID>.pdb`: the auto-fetched raw receptor, if `--protein` was a PDB ID.
- `<outdir>/receptor_prepared.pdb`: the cleaned, H-added receptor used for docking.
- `<outdir>/prepared_ligands.sdf`: every successfully-embedded, MMFF94-minimized ligand.
- `<outdir>/docking_box.pml`: PyMOL script visualizing the detected/overridden docking box.
- `<outdir>/docking_outputs/`: one `<mol_name>_docked.sdf` per ligand, plus `gnina_debug.txt` (raw stdout/stderr of the first completed job, for troubleshooting).
- `<outdir>/results_summary.csv`: full results table for every docked compound.
- `<outdir>/top_results.xlsx`: the exported top-N (or all) compounds, sorted by CNN Affinity then Vina Affinity, with auto-fit column widths.
- `<outdir>/ranked_docked.sdf`: the best pose of every exported compound, in rank order, tagged with rank/scores (unless `--no-merged-sdf`).

`gnina-vs recover` writes the same `results_summary.csv` / `top_results.xlsx` / `ranked_docked.sdf` trio (no receptor/ligand/docking files, since nothing is re-run).

---

## Notes and Limitations

- Gnina must be installed separately; it is not a Python dependency and can't be `pip install`-ed.
- `--keep-hetatm` is the only way to retain a HETATM cofactor (e.g. a catalytic zinc) through protein preparation; everything else in a HETATM record is stripped, including any co-crystallised ligand, which is why Tier 1 needs to search the *original* PDB rather than the prepared one.
- The Tier 3 grid-cavity detector is pure Python/NumPy and can be slow on large proteins at 1 Å resolution without scipy installed (scipy accelerates the probe-radius dilation step; without it, a manual fallback dilation is used, which is correct but slower).
- Gnina's stdout table format has changed across versions; score parsing tries three strategies (space-delimited table, key=value, pipe-delimited table) before falling back to the properties embedded in the output SDF. If none of these find the docked pose's scores, the compound is exported with blank/NaN scores rather than silently dropped.
- `--top-n` only ever exports compounds that were successfully scored (non-null CNN Affinity); ligands that failed to dock or parse are still present in the full `results_summary.csv`.
- This tool is intended for research and educational purposes; predicted docking scores are not a substitute for experimental validation.

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
