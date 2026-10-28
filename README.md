# Gnina-VS: Metabolite-Aware Virtual Screening with Gnina

A modular virtual-screening pipeline: **protein preparation** (with automatic RCSB PDB fetching), **ligand preparation**, **active-site detection** (explicit residue/ligand selection plus two automatic tiers), **parallel Gnina docking**, and **results reporting**, packaged as a clean, installable Python CLI (`gnina-vs`). A second command, `gnina-vs metabolite-run`, extends this into a **metabolite-aware** pipeline: generate a drug's predicted metabolites, dock parent + metabolites against one or more targets, and compare the results -- see [Metabolite-Aware Docking](#metabolite-aware-docking-metabolite-run).

This is a refactor of a single-file prototype script into a proper package: every step is its own module (`protein`, `ligand`, `pocket`, `docking`, `reporter`), every previously-hardcoded global (input paths, box size, worker counts, Gnina parameters) is now a CLI flag, and the whole thing installs via `pip install -e .` with a real console-script entry point. It's also usable as a plain Python library (`from gnina_vs import prepare_protein, prepare_ligands, resolve_center, dock_ligands, extract_results`) -- `cli.py` is a thin wrapper around these functions.

---

## Table of Contents

- [Overview](#overview)
- [Workflow](#workflow)
- [Detailed Pipeline Steps](#detailed-pipeline-steps)
  - [Protein Preparation](#protein-preparation)
  - [Ligand Preparation](#ligand-preparation)
  - [Metabolite Generation](#metabolite-generation)
  - [Pocket / Active-Site Identification](#pocket--active-site-identification)
  - [Docking](#docking)
  - [Annotation & Toxicity-Relevant Flags](#annotation--toxicity-relevant-flags)
- [Output Tables](#output-tables)
- [The HTML Triage Report](#the-html-triage-report)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Run the full pipeline](#1-run-the-full-pipeline)
  - [2. Auto-fetch a protein structure from RCSB](#2-auto-fetch-a-protein-structure-from-rcsb)
  - [3. Manual active-site overrides](#3-manual-active-site-overrides)
  - [4. Recovery mode](#4-recovery-mode)
  - [5. Configure Gnina and export more/fewer compounds](#5-configure-gnina-and-export-morefewer-compounds)
  - [6. Generate metabolites for every compound, then dock all of them](#6-generate-metabolites-for-every-compound-then-dock-all-of-them---metabolite)
  - [All available options](#all-available-options)
- [Active-Site Detection Priority](#active-site-detection-priority)
- [Metabolite-Aware Docking (`metabolite-run`)](#metabolite-aware-docking-metabolite-run)
- [Quick Start: Dock an Excel Sheet of Compounds With Their Metabolites](#quick-start-dock-an-excel-sheet-of-compounds-with-their-metabolites)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Notes and Limitations](#notes-and-limitations)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Given a receptor (a local PDB file or a bare PDB ID) and a spreadsheet of compounds (SMILES), the pipeline:

1. **Fetches** the receptor from RCSB PDB if a 4-character PDB ID is given instead of a file path (`protein.fetch_pdb_from_rcsb`), then **prepares** it: strips waters and crystallization artefacts, converts selenomethionine (MSE) to MET and keeps other in-chain modified residues (e.g. `SEP`, `TPO`, `PTR`), resolves alternate conformations, keeps ATOM/ANISOU records (plus any HETATM residues you explicitly ask to retain, e.g. a metal cofactor), and adds hydrogens via OpenBabel when available (falling back cleanly if it isn't) -- this uses a pH model, not a polar-only or per-residue pKa/tautomer treatment.
2. **Loads** compounds from an Excel or CSV file, strips salts/counter-ions (keeping the largest fragment), warns on undefined stereocentres, generates 3-D conformers with ETKDGv3 (seeded, for reproducibility), minimizes them with MMFF94, and writes everything to a single multi-molecule SDF with a unique, filesystem-safe `compound_id` per row.
3. **Detects the docking box center**, in priority order: explicit `--center-x/y/z` coordinates, a centroid of named catalytic residues (`--active-site-residues`, chain-aware), an explicitly named co-crystallised ligand/cofactor (`--ligand-resname`/`--ligand-chain`), an automatically-selected co-crystallised ligand (Tier 1, excluding a wide set of common cofactors, crystallization additives, and in-chain modified residues so it doesn't mistake them for the real ligand), or `fpocket` (Tier 2). There is no further geometric fallback: if neither tier finds anything, detection stops with a clear error rather than guessing the protein's centre. A `.pml` PyMOL script is always written so you can visually check the box before committing CPU hours.
4. **Docks** every ligand against the receptor in parallel with Gnina (`ThreadPoolExecutor`), with fully configurable CNN scoring mode, exhaustiveness, number of modes, box size, worker/CPU counts, and a `--seed` for reproducibility. Each job's outcome (`ok`/`failed`/`timeout`/`no_output`/`error`) is recorded explicitly, and a stale docked file from a previous run is never mistaken for this run's result.
5. **Extracts** scores from Gnina's stdout (or, as a fallback, from the properties Gnina embeds in the output SDF), exports a ranked CSV + Excel report (flexible top-N, or all compounds; ranking falls back to Vina affinity when CNN scoring is off), flags low-confidence poses (`pose_quality_flag`, from the CNN pose score) without removing them, and merges the best pose of every exported compound into a single ranked SDF for easy visual inspection. A `run_manifest.json` records the seed, box, scoring settings, and tool/package versions used.

A `recover` mode lets you re-run steps 4-5's scoring/export logic against already-docked SDF files, without re-docking anything.

**Add `--metabolite` to `run`** and step 2 above generates metabolites for every compound in the spreadsheet first (BioTransformer and/or curated literature metabolites), using the more careful metabolite-aware ligand preparation (salt stripping, stereoisomer enumeration, pH 7.4 protonation) on the resulting parent+metabolite set, then docks all of them together and adds a `comparison_table.csv` (scores vs parent, ligand efficiency, drug-likeness/structural-alert annotations) to the usual outputs. See [Quick Start](#quick-start-dock-an-excel-sheet-of-compounds-with-their-metabolites).

---

## Workflow

```
                  ┌──────────────────────┐
 --protein   ───► │ 0. protein.py        │  RCSB fetch (if PDB ID) + strip/clean + add H
 (path/PDB ID)    └──────────┬───────────┘
                             │ prepared receptor PDB
                  ┌──────────▼───────────┐
 --ligands   ───► │ 1. ligand.py         │  SMILES → 3-D conformer → MMFF94 minimize
 (.xlsx/.csv)     └──────────┬───────────┘
                             │ prepared_ligands.sdf
                  ┌──────────▼───────────┐
                  │ 2. pocket.py         │  manual/explicit override → Tier 1/2 → .pml box script
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
| --- | --- | --- | --- |
| Protein prep | `protein.py` | RCSB auto-fetch, strip waters/artefacts, MSE→MET, altLoc dedup, add H | `urllib.request`, OpenBabel |
| Ligand prep | `ligand.py` | Load SMILES, strip salts, embed 3-D, MMFF94 minimize | `rdkit`, `pandas` |
| Pocket detection | `pocket.py` | Manual/explicit override, 2-tier auto-detection, PyMOL script | `fpocket` (optional) |
| Docking | `docking.py` | Parallel Gnina execution | `subprocess`, `concurrent.futures`, `numpy` |
| Reporting | `reporter.py` | Score parsing, CSV/Excel export, merged SDF, recovery | `pandas`, `openpyxl`, `rdkit` |

---

## Detailed Pipeline Steps

This section spells out, step by step, exactly what happens at each stage -- the one-line summaries above link here for the full detail. "Protein Preparation" through "Docking" run in every `gnina-vs run`; "Metabolite Generation" and the fuller "Annotation" step only run under `gnina-vs metabolite-run` (see [Metabolite-Aware Docking](#metabolite-aware-docking-metabolite-run)).

### Protein Preparation

Module: `protein.py`. Input: a local PDB file, or a bare 4-character PDB ID. Output: `receptor_prepared.pdb`.

1. **Resolve the input.** If `--protein` is an existing file path, use it as-is. If it looks like a 4-character PDB ID (e.g. `1CX2`), download `https://files.rcsb.org/download/<ID>.pdb` and save it to `--outdir`.
2. **Keep only the first model** (for NMR/multi-model structures -- everything after the second `MODEL` record is discarded).
3. **Remove water** (`HOH`, `WAT`, `DOD`, `H2O`, ...) unconditionally.
4. **Convert selenomethionine (`MSE`) to methionine**: `HETATM` -> `ATOM`, residue name `MSE` -> `MET`, selenium atom `SE` -> `SD`. (This is a standard crystallographic substitution for phasing, chemically identical to methionine; leaving it as `MSE` would have it stripped as "not protein" in the next step, leaving a gap in the chain.)
5. **Keep other in-chain modified residues** (`SEP`, `TPO`, `PTR` -- phosphorylated Ser/Thr/Tyr) as-is; they're covalently part of the polypeptide, not a ligand or cofactor.
6. **Strip every other `HETATM` residue** (crystallization salts/cryo-protectants, metal ions, cofactors, glycosylation sugars, the bound ligand itself) *unless* its residue name is listed in `--keep-hetatm` (e.g. `--keep-hetatm ZN` to retain a catalytic zinc). If a stripped residue looks like a common cofactor or metal (`NAD`, `FAD`, `HEM`, `ATP`, bound ions, ...), a warning names it and suggests `--keep-hetatm`.
7. **Resolve alternate conformations**: for every atom with more than one modeled location (`altLoc` A/B/...), keep only the highest-occupancy copy (ties broken by whichever comes first in the file) and clear its altLoc flag, so there are no duplicate/overlapping atoms.
8. **Add hydrogens** via OpenBabel (`obabel -p <ph>`) if it's installed and `--no-add-hydrogens` wasn't given. This adds hydrogens for the given pH generally, not a polar-only or per-residue pKa/tautomer treatment. If OpenBabel isn't available, the cleaned (no added-H) structure is used instead -- Gnina adds hydrogens internally regardless, so this step is a refinement, not a requirement.

### Ligand Preparation

Module: `ligand.py`. Input: an Excel/CSV file with a SMILES column. Output: `prepared_ligands.sdf`.

1. **Load the compound table** (`.xlsx`/`.xls`/`.csv`), requiring a SMILES column (`--smiles-col`, default `SMILES`); an ID column (`--id-col`, default `ID`) and a Name column (`--name-col`, default `Name`) are optional -- missing values fall back to the row index.
2. **Parse the SMILES.** Unparsable structures are skipped with a warning (and, under `metabolite-run`, recorded in `ligand_prep_failures.csv` rather than only logged).
3. **Strip salts and counter-ions**: for a multi-fragment structure (e.g. a drug's HCl salt), keep the largest (organic-preferred) fragment and discard the rest, logging what was removed.
4. **Warn on undefined stereochemistry**: if a SMILES has a stereocentre with no defined configuration, a warning names it, since only one arbitrary isomer will be embedded (`gnina-vs run`) -- see [Metabolite Generation](#metabolite-generation) below for how `metabolite-run` instead enumerates every isomer.
5. **Generate a 3-D conformer** with RDKit's ETKDGv3 algorithm (seeded via `--seed`, default 42, for reproducibility).
6. **Minimize with MMFF94** (the Merck Molecular Force Field). Embedding or minimization failures are skipped with a warning.
7. **Write every successful structure** to one multi-molecule SDF, each tagged with a unique, filesystem-safe `compound_id` (so two compounds with the same display name never collide) plus its original SMILES, row index, and ID-column value.

### Metabolite Generation

Module: `metabolism.py`. Runs only under `gnina-vs metabolite-run`, once per configured drug, before ligand preparation. Output: `metabolites.csv` (shared across every drug in the run).

1. **Always include the parent** as its own row (`is_parent: true`), canonicalized and with its InChIKey computed.
2. **If a BioTransformer jar is configured** (`metabolism.biotransformer_jar`), run it (`java -jar <jar> -k pred -b <type> -s <steps> -ismi <parent SMILES> -ocsv <output>`), caching the result by `(SMILES, type, steps, include_gut)` so repeated runs are instant. `type` selects the predictor (`allHuman`, `cyp450`, `phaseII`, `hgut`, `envmicro`); `steps` (1 or 2) controls how many rounds of further metabolism to predict.
   - **Caveat, stated plainly:** the exact CLI flags and output column names above were written from BioTransformer's public documentation, not verified against a real installed jar (none was available while building this). Run `metabolism.check_biotransformer(jar_path)` (does `java -jar <jar> --help`) and inspect one real output file before trusting a production run.
3. **Parse the predicted metabolites**, matching likely column names for SMILES/InChIKey/reaction/enzyme, and infer Phase I vs Phase II from the enzyme name (`CYP*` -> Phase I; `UGT*`/`SULT*`/`GST*`/`NAT*` -> Phase II).
4. **If `metabolites_csv` is given** (a plain CSV with at least a `smiles` column), load those as additional curated metabolites -- useful when BioTransformer misses a metabolite you know from the literature, or to skip BioTransformer entirely. **For a run with more than one drug, add a `parent_id` column** (matching each drug's `id`) so each row is only attached to its own drug; without that column, *every* row in the file is attached to *every* drug in the run, which is almost never what you want once there's more than one. A config with multiple drugs and a `metabolites_csv` lacking this column gets a loud warning for exactly this reason.
5. **Deduplicate by InChIKey**, keeping the shortest metabolic pathway (lowest `depth`) to each distinct structure; the parent row always survives deduplication.

### Pocket / Active-Site Identification

Module: `pocket.py`. Output: the docking box center `(x, y, z)`, plus a `docking_box.pml` PyMOL visualization script. Full detail in [Active-Site Detection Priority](#active-site-detection-priority); summarized here:

1. **Explicit coordinates** (`--center-x/y/z`) always win if given.
2. **Named catalytic residues** (`--active-site-residues "TYR120,GLU150"`, chain-prefixed as `A:TYR120` if the residue number isn't unique across chains) -- centroid of every heavy atom (hydrogens excluded) of the named residues.
3. **An explicitly named ligand/cofactor** (`--ligand-resname ATP`, optionally `--ligand-chain`) -- centroid of that exact HETATM residue's heavy atoms.
4. **Automatic Tier 1**: the non-excluded HETATM residue with the most heavy atoms, searched in the *original* (un-stripped) PDB. Excluded by name: common cofactors, crystallization additives, in-chain modified residues, and glycosylation sugars (`NAG`, `MAN`, `GAL`, ...). Every candidate is logged with its atom count; a warning fires whenever more than one candidate exists, since "most atoms" is a heuristic, not a certainty.
5. **Automatic Tier 2**: `fpocket`'s top-ranked predicted pocket, if `fpocket` is installed and Tier 1 found nothing.
6. **No further fallback.** If neither tier finds anything (an apo structure with no co-crystal ligand and no fpocket), detection stops with an explicit error rather than guessing the protein's geometric centre.

### Docking

Module: `docking.py`. Output: one `<compound_id>_docked.sdf` per ligand, plus `gnina_debug.txt` (the first job's raw Gnina output, for troubleshooting).

1. **Validate the Gnina executable** (`--gnina-exec`, default `gnina`) exists and run a version smoke-test.
2. **For every ligand, build and run a Gnina command**: `--receptor <prepared.pdb> --ligand <this compound> --out <compound_id>_docked.sdf --center_x/y/z <box center> --size_x/y/z <--box-size> --exhaustiveness <N> --num_modes <N> --cnn_scoring <mode> --cpu <cpu-per-worker>` (plus `--seed` if given, or `--autobox_ligand`/`--autobox_add` instead of center/size if configured).
3. **Run up to `--workers` ligands concurrently** (a `ThreadPoolExecutor` -- Gnina itself is a subprocess, so Python threads are sufficient).
4. **Delete any stale output file for this compound before docking it**, so a failed/timed-out job can never be confused with a leftover result from a previous run.
5. **Record every job's outcome explicitly**: `ok`, `failed` (non-zero exit), `timeout`, `no_output` (exit 0 but no file written), or `error` (an exception in the worker) -- never silently dropped.

### Annotation & Toxicity-Relevant Flags

Module: `annotate.py`. Runs only under `gnina-vs metabolite-run`, after docking and scoring, on every compound in `comparison_table.csv`. **These are informational columns, never filters** -- nothing here removes, reorders, or routes a compound; see [Notes and Limitations](#notes-and-limitations) for why none of this is a real toxicity prediction.

1. **Compute standard RDKit descriptors**: molecular weight, calculated logP (Crippen), TPSA, H-bond donor/acceptor counts, rotatable bond count, formal charge.
2. **Count Lipinski Rule-of-Five violations** (MW > 500, logP > 5, HBD > 5, HBA > 10) -- reported as a count, never enforced as a cutoff.
3. **Run RDKit's structural-alert filter catalogs** -- Brenk, PAINS (families A, B and C counted separately), NIH, ZINC and the ChEMBL sets -- reporting an `alert_<family>_n` count per family, an `alert_total_n`, and the matched names for Brenk and PAINS. There is deliberately **no single `reactive_alert` boolean**: the PAINS authors state that black-box use of these filters "is fraught with danger", that a match does not imply interference and that absence of a match does not imply cleanliness, and the three PAINS families are so asymmetric in how much recognised-compound mass they carry that collapsing them discards the usable signal. Counts are reported; interpretation is the reader's.
4. **Flag likely Phase II conjugates**: substructure matches for a glucuronide-like (sugar ring + carboxylic acid, O-linked) or sulfate-ester pattern set `glucuronide_flag`/`sulfate_flag`/`conjugate_flag` -- heuristic pattern matching, not an authoritative metabolite classifier.
5. **Report an exposure-plausibility tier** (`exposure_plausibility`, `exposure_basis`): a count of conventional passive-permeability liabilities -- Phase-II conjugate, non-zero formal charge, TPSA > 120, MW > 500, logP < 0 -- binned into `likely`/`reduced`/`unlikely`, with `exposure_basis` naming whichever criteria fired. This is a crude descriptor heuristic and is labelled as one: it is **not** tissue-specific, **not** a blood-brain-barrier model, **not** a pharmacokinetic quantity (the real answer is a Kp,uu from a PBPK model), and it is not validated against measured exposure. It says nothing about whether a metabolite *forms* -- formation happens in liver and gut wall regardless of where the target sits -- only about whether it plausibly reaches a site of action. For an off-target liability screen such as hERG it should not be used to narrow anything: there you want the widest metabolite set, because a circulating conjugate you filtered out is a liability you failed to screen.
6. **Run ADMET-AI predictions** (hERG, AMES mutagenicity, DILI liver injury, etc.) *if* the optional `admet-ai` package is installed; otherwise skip gracefully and record `admet_status: "skipped: admet_ai not installed"`.
7. **Check the applicability domain properly** (`gnina_vs.applicability`, columns `ad_reference`/`ad_distance`/`ad_threshold`/`ad_in_domain`/`ad_status`). An applicability domain is defined relative to *the model's training set* (OECD validation principle 3), so this measures each compound's mean Tanimoto distance to its k nearest **ADMET-AI training molecules** and compares it with a threshold taken from the 95th percentile of the reference set's own distance distribution -- a cutoff derived from data rather than asserted. `ad_reference` records the reference set, n, k and the percentile so a distance stays interpretable later.

   An earlier version used Morgan/Tanimoto similarity to the compound's **parent** as a proxy. That was not an applicability domain, and it failed in a specifically perverse way: a glucuronide keeps most of its parent's substructure, so it scored *high* on parent similarity while being exactly the chemistry ADMET training sets under-represent -- the criterion was anti-correlated with the risk it was meant to detect. `parent_tanimoto` is still reported, because structural distance from the parent is genuinely interesting for a metabolite; it is simply not an applicability domain.

   The reference set is assembled once from the Therapeutics Data Commons ADMET datasets and cached, so a screening run never reaches the network:

   ```bash
   pip install scikit-fingerprints PyTDC
   gnina-vs fetch-ad-reference            # one-off; prints the molecule count
   ```

   Without a cached reference set every `ad_*` column is empty with the reason in `ad_status`. **The download path has not been exercised against a live TDC endpoint** -- run it once and check the reported count before relying on these columns. The check is also coarse by construction: ADMET-AI predicts dozens of endpoints, each with its own training set, and a per-endpoint domain is the honest next step.

---

## Requirements

- **Python 3.9+**
- **Gnina**, installed and on `PATH` (or pass `--gnina-exec /path/to/gnina`): <https://github.com/gnina/gnina>
- **OpenBabel** (optional, improves protein-H placement): `conda install -c conda-forge openbabel`
- **fpocket** (optional, enables Tier 2 pocket detection): <https://fpocket.sourceforge.net/>
- **PyMOL** (optional, only needed to actually open the generated `.pml` box-visualization script)
- An internet connection, only if using a bare PDB ID with `--protein` instead of a local file
- For `gnina-vs metabolite-run` only: **Java 8+** and the [BioTransformer](https://bitbucket.org/wishartlab/biotransformer3.0/) jar (optional -- omit `metabolism.biotransformer_jar` to dock only parents, or use `metabolites_csv` for curated metabolites instead), and the `metabolism`/`figures` extras below

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

   For `gnina-vs metabolite-run`, also install its extras:

   ```bash
   pip install -e ".[metabolism,figures]"   # + optionally: annotate (ADMET-AI)
   ```

   `run`/`recover` never need these -- `metabolism` covers YAML config parsing (PyYAML) and optional Dimorphite-DL protonation; `figures` covers matplotlib; `annotate` covers the optional ADMET-AI predictions (skipped gracefully without it).

### Installing BioTransformer (Optional)

If you want to run `metabolite-run` with automatic metabolite generation, you must compile the BioTransformer `.jar` file locally:

1. Ensure Java 8+ and Maven are installed (`sudo apt install default-jdk maven git`).
2. Clone the official repository: `git clone https://github.com/Wishartlab-openscience/Biotransformer.git`
3. Enter the directory and build: `cd Biotransformer && chmod +x install-via-maven.sh && ./install-via-maven.sh`
4. Pass the absolute path of the generated `.jar` file to the `--biotransformer-jar` flag.

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

...or with named catalytic residues instead (their centroid is used). For a single-chain structure the chain can be omitted; for a multi-chain structure, prefix each residue with its chain (`A:TYR120`) or the run fails with a clear "ambiguous residue" error rather than averaging across chains:

```bash
gnina-vs run --protein receptor.pdb --ligands compounds.xlsx \
    --active-site-residues "A:TYR120,A:GLU150,A:HIS95"
```

...or by naming a specific co-crystallised ligand or cofactor exactly, instead of relying on automatic selection (useful for a cofactor-dependent active site, e.g. a kinase's ATP pocket):

```bash
gnina-vs run --protein receptor.pdb --ligands compounds.xlsx \
    --ligand-resname ATP --ligand-chain A
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
    --cnn-scoring refinement --exhaustiveness 16 --num-modes 20 \
    --workers 8 --cpu-per-worker 2 \
    --top-n 250
```

### 6. Generate metabolites for every compound, then dock all of them (`--metabolite`)

```bash
gnina-vs run --protein receptor.pdb --ligands compounds.xlsx --metabolite \
    --biotransformer-jar /path/to/biotransformer3.0.jar
```

Every row in `compounds.xlsx` is treated as a parent compound: its metabolites are generated first (written to `metabolites.csv`), the parent and every metabolite are docked together against `receptor.pdb`, and a `comparison_table.csv` is added to the usual outputs (score vs parent, ligand efficiency, drug-likeness/structural-alert annotations). `--biotransformer-jar` is optional -- omit it (and `--metabolites-csv`) to still get the more careful metabolite-aware ligand preparation on just the parents, with no actual metabolites generated. See [Quick Start](#quick-start-dock-an-excel-sheet-of-compounds-with-their-metabolites) for the full walkthrough, and `gnina-vs run --help` for every `--metabolite*` flag.

### All available options

```bash
gnina-vs run --help
gnina-vs recover --help
```

**Input files** (`run`)

| Flag | Description | Default |
| --- | --- | --- |
| `--protein` | Receptor PDB path, or a bare 4-character PDB ID to auto-fetch from RCSB | **required** |
| `--ligands` | `.xlsx`/`.xls`/`.csv` file with a SMILES column | **required** |
| `--smiles-col` | SMILES column name | `SMILES` |
| `--name-col` | Compound name column name (optional; falls back to `compound_<row>`) | `Name` |

**Protein preparation**

| Flag | Description | Default |
| --- | --- | --- |
| `--keep-hetatm` | Comma-separated HETATM residue names to retain as cofactors (e.g. `ZN` or `ZN,MG`) | *(strip all)* |
| `--no-add-hydrogens` | Skip adding hydrogens via OpenBabel | off |
| `--ph` | Protonation pH passed to OpenBabel | `7.4` |

Selenomethionine (`MSE`) is always converted to `MET`/`ATOM` rather than stripped, and other in-chain modified residues (`SEP`, `TPO`, `PTR`) are always kept, regardless of `--keep-hetatm`. If a stripped HETATM residue looks like a catalytic cofactor or metal (e.g. `NAD`, `HEM`, `FAD`, a bound `ZN`/`MG`/... ion), a warning names it and suggests `--keep-hetatm` -- these can matter for the shape and chemistry of the binding site.

**Active site detection**

| Flag | Description | Default |
| --- | --- | --- |
| `--center-x`, `--center-y`, `--center-z` | Explicit box center (all three must be given together) | *(none)* |
| `--active-site-residues` | Comma-separated residues to center on, e.g. `"TYR120,GLU150"` or, for a multi-chain structure, `"A:TYR120,B:GLU150"` | *(none)* |
| `--ligand-resname` | Exact HETATM residue name to center on (e.g. `ATP`), instead of automatic selection | *(none)* |
| `--ligand-chain` | Chain to disambiguate `--ligand-resname` when the name appears more than once | *(none)* |

**Gnina docking parameters**

| Flag | Description | Default |
| --- | --- | --- |
| `--gnina-exec` | Path to the `gnina` executable | `gnina` |
| `--box-size` | Docking box edge length (Å, cubic box) | `25.0` |
| `--cnn-scoring` | `none`, `rescore`, `refinement`, `metrorescore`, `metrorefine`, or `all` (deprecated aliases `refine`→`refinement`, `full`→`all` still accepted, with a warning) | `rescore` |
| `--exhaustiveness` | Vina/Gnina search exhaustiveness | `8` |
| `--num-modes` | Binding modes generated per ligand | `9` |
| `--workers` | Ligands docked concurrently | `4` |
| `--cpu-per-worker` | `--cpu` passed to each Gnina process (total threads ≈ `workers × cpu-per-worker`) | `2` |
| `--timeout` | Per-ligand docking timeout (seconds) | `300` |
| `--seed` | Random seed passed to Gnina (`--seed`) and used for RDKit conformer embedding | `42` |
| `--autobox-ligand` | Reference ligand file: build the box around it (Gnina `--autobox_ligand`) instead of `--center-x/y/z` + `--box-size` | *(none)* |
| `--autobox-add` | Buffer (Å) added around `--autobox-ligand` | `4.0` |

**Metabolite generation** (`run`, optional -- see [Quick Start](#quick-start-dock-an-excel-sheet-of-compounds-with-their-metabolites))

| Flag | Description | Default |
| --- | --- | --- |
| `--metabolite` | Generate metabolites for every compound in `--ligands` first, then dock parents + metabolites together; adds `metabolites.csv` and `comparison_table.csv` to the outputs | off |
| `--biotransformer-jar` | Path to a BioTransformer jar (needs Java). Omit to skip automatic prediction | *(none)* |
| `--metabolite-type` | BioTransformer predictor: `allHuman`, `cyp450`, `phaseII`, `hgut`, `envmicro` | `allHuman` |
| `--metabolite-steps` | Rounds of further metabolism to predict: `1` or `2` | `1` |
| `--metabolite-include-gut` | Include gut-microbial metabolism in the prediction | off |
| `--metabolites-csv` | Curated metabolites CSV (needs a `parent_id` column when `--ligands` has more than one compound -- see the note under [Metabolite Generation](#metabolite-generation)) | *(none)* |

**Output** (`run` and `recover`)

| Flag | Description | Default |
| --- | --- | --- |
| `--outdir` | Directory for all outputs | `gnina_vs_output` |
| `--top-n` | Compounds to export; `0` or `all` exports everything scored | `100` |
| `--id-col` | Identifier column name in the ligand input file | `ID` |
| `--no-merged-sdf` | Skip writing the single ranked multi-molecule SDF | off |
| `-v`, `--verbose` | Enable DEBUG logging | off |

**Recovery-only** (`recover`)

| Flag | Description | Default |
| --- | --- | --- |
| `--ligands-sdf` | Prepared ligands SDF from a previous `run` | **required** |
| `--docking-outdir` | Directory with `<compound_id>_docked.sdf` files from a previous `run` | **required** |
| `--input` | Original ligand Excel/CSV, to recover the ID column | *(none)* |

---

## Active-Site Detection Priority

`pocket.py` resolves the docking box center in this order, stopping at the first that applies:

1. **`--center-x/y/z`** -- explicit coordinates, always wins if given.
2. **`--active-site-residues`** -- centroid of every atom of the named residues (searched against the *original*, un-stripped PDB so full side chains are present). A residue given without a chain (`TYR120`) is resolved automatically only if that residue number is unique across the structure's chains; if it appears in more than one chain, the run fails with a clear error asking for `A:TYR120`-style input instead of silently averaging across chains (which could sit between two unrelated active sites in a dimer).
3. **`--ligand-resname`** (optional `--ligand-chain`) -- centroid of one exactly-named HETATM residue, e.g. `ATP`. Every matching residue is logged with its chain, residue number and atom count; if the name matches more than one residue and no chain is given, the run fails with a clear error listing the candidates rather than guessing. Use this to target a cofactor (`ATP`, `NAD`, ...) that Tier 1's automatic selection below would otherwise skip, or to pick a specific chain's copy of a repeated ligand.
4. **Tier 1 -- co-crystallised ligand centroid (automatic).** Searches the *raw* PDB (not the prepared one, since protein preparation strips HETATM records) and picks the non-excluded HETATM residue with the most heavy atoms -- hydrogens, when a file carries them explicitly, are never counted (PDB files vary on this, and some structures carry computationally-added hydrogens even at resolutions where hydrogen positions were never experimentally observable; counting them would make the "most atoms" comparison file-dependent and could flip a tie toward the wrong group). The exclusion list skips common cofactors (`ATP`, `ADP`, `AMP`, `GTP`, `GDP`, `NAD`, `NADP`, `FAD`, `FMN`, `HEM`, `PLP`), crystallization additives (`PEG`, `EDO`, `GOL`, `OLB`, `BOG`, `MPD`, `DMS`), in-chain modified residues (`MSE`, `SEP`, `TPO`, `PTR`), and N-/O-linked glycosylation sugars (`NAG`, `NDG`, `BMA`, `MAN`, `GAL`, `GLA`, `FUC`, `FUL`, `BGC`, `GLC`, `SIA`, `ANE`, `NGA`, `XYS`, `XYP`, `A2G`), so none of these are mistaken for the ligand you actually want to dock against. This heuristic is still fundamentally "most heavy atoms wins" and the exclusion list can't be exhaustive -- **every candidate residue (name, chain, residue number, heavy-atom count) is logged**, and a warning is printed whenever more than one candidate exists. Always check that log on a new target, or skip the guesswork entirely with `--ligand-resname`/`--ligand-chain`.
5. **Tier 2 -- `fpocket`.** Geometric pocket finder; used if installed and Tier 1 found nothing. The centroid is computed directly from fpocket's own `pockets/pocket1_vert.pqr` (or `pocket1_atm.pdb`) output for its top-ranked pocket.

If neither Tier 1 nor Tier 2 finds anything (e.g. an apo structure with fpocket not installed), detection **stops with a clear error** rather than falling back to a geometric guess -- there is deliberately no third, fully-automatic tier: a purely geometric cavity detector (probe-radius surface dilation + flood-fill) was tried and found to be unreliable in practice, tending to return the protein's own centre rather than a real pocket, which is worse than failing loudly. In that case, supply explicit coordinates, `--active-site-residues`, or `--ligand-resname`.

Whichever method wins, a `.pml` script is written showing the resulting box as a wireframe cube plus a center pseudo-atom, so you can inspect it in PyMOL before docking.

---

## Metabolite-Aware Docking (`metabolite-run`)

A second command extends the base pipeline for a specific research question: **do predicted metabolites change the docking picture compared with the parent alone?** (e.g. terfenadine vs its metabolite fexofenadine on the hERG antitarget; codeine vs morphine on the mu-opioid receptor). It generates *hypotheses and comparisons*, not clinical predictions -- see [Notes and Limitations](#notes-and-limitations).

```bash
gnina-vs metabolite-run --config config.yaml
```

**This is the multi-target, power-user version of `gnina-vs run --metabolite`** ([Quick Start](#quick-start-dock-an-excel-sheet-of-compounds-with-their-metabolites)). Reach for `metabolite-run` + a YAML config specifically when you need any of what it adds over the single-target flag: several targets in one run (sharing one metabolite table), automatic redocking validation of each target's own co-crystal ligand, reference-ligand deltas, and figures. For one target, `run --metabolite` does the same metabolite generation, ligand preparation, docking, comparison, and annotation with no config file at all.

This needs the optional `metabolism` and `figures` dependency groups (`pip install -e ".[metabolism,figures]"`); `run`/`recover` (`--metabolite` included) never require them -- `run --metabolite` only optionally uses Dimorphite-DL (also under `[metabolism]`) for protonation, and needs neither PyYAML nor matplotlib.

### Workflow

```
 config.yaml
      │
      ▼
┌──────────────────┐
│ 1. metabolism.py │  BioTransformer (Java jar) and/or --metabolites-csv,
│    (per drug)    │  standardized + deduped by InChIKey, parent always kept
└────────┬─────────┘
         │ metabolites.csv (shared across all drugs)
         ▼
┌──────────────────────────┐
│ 2. ligand.py             │  salts stripped, neutralized, tautomer-canonicalized,
│ prepare_ligands_extended │  undefined stereocentres enumerated (each isomer
│                          │  docked separately), protonated at pH 7.4
└────────┬─────────────────┘
         │ prepared_ligands.sdf (shared)
         ▼
   for each target (its receptor + box resolved ONCE):

   ┌───────────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────────────┐
   │ protein.py +  │──▶│ docking.py   │──▶│ validation.py │──▶│ compare.py +    │
   │ pocket.py     │   │ (parent +    │   │ (redock       │   │ annotate.py     │
   │ (once/target) │   │ metabolites) │   │ co-crystal    │   │ (deltas, LE,    │
   └───────────────┘   └──────────────┘   │ ligand: RMSD; │   │ ranks, RDKit    │
                                          │ + reference   │   │ properties,     │
                                          │ ligands)      │   │ Brenk/PAINS)    │
                                          └───────────────┘   └────────┬────────┘
                                                                       ▼
                                                              comparison_table.csv
                                                              + figures.py (Phase 7)
```

### Config schema

See `examples/metabolite_run_config.yaml` for a complete, commented example (terfenadine vs hERG/MOR) and `examples/curated_metabolites.csv` for the `metabolites_csv` bypass format. `examples/run_tiny_example.py` is a self-contained runnable example (synthetic receptor, curated metabolites, no network/Java needed -- only a real `gnina` binary is required to actually dock; see the script's docstring for how the test suite runs the same pipeline with Gnina mocked).

```yaml
outdir: results          # everything is written under here
seed: 42                 # RDKit embedding + Gnina --seed, recorded in run_manifest.json

protein:                 # optional; same knobs as `run`'s --no-add-hydrogens/--ph
  add_hydrogens: true
  ph: 7.4

metabolism:              # optional; omit entirely to dock only the parent
  biotransformer_jar: /path/to/biotransformer3.0.jar   # needs Java; see gnina_vs.metabolism
  type: allHuman         # allHuman | cyp450 | phaseII | hgut | envmicro
  steps: 1               # 1 or 2
  include_gut: false
  metabolites_csv: curated.csv   # bypass or supplement BioTransformer

docking:                 # same fields/semantics as `run`'s flags
  cnn_scoring: rescore
  exhaustiveness: 16
  num_modes: 9
  workers: 4
  cpu_per_worker: 2
  box_size: 25.0
  gnina_exec: gnina
  timeout: 300

targets:
  - id: hERG
    role: antitarget                 # or therapeutic; reported, not enforced
    pdb: 5VA2                        # bare PDB ID (auto-fetched) or a local path
    center_from_ligand: true         # + ligand_resname (+ optional chain), OR:
    ligand_resname: 3JW
    # center: [12.3, 45.6, -7.8]     # ...an explicit box centre instead
    keep_hetatm: [ZN]                # per-target cofactors/metals to retain
    reference_ligands:
      - {name: terfenadine, smiles: "..."}

drugs:
  - id: terfenadine
    smiles: "..."
    targets: [hERG]                  # omit to dock against every target
```

Validation is per-field: a missing or invalid value raises `gnina_vs.config.ConfigError` naming the exact location (e.g. `"targets[0]: missing required field 'pdb'."`).

### What each phase adds

| Phase | Module | What it does |
| --- | --- | --- |
| 1 | `metabolism.py` | BioTransformer jar wrapper (cached by SMILES+settings) + `--metabolites-csv` bypass; standardizes and dedupes by InChIKey; the parent is always included (`is_parent=True`). **The BioTransformer CLI flags and output column names are not verified against a real jar in this environment (only a JRE was available, no jar)** -- `metabolism.check_biotransformer(jar_path)` runs `--help` against your real jar so you can confirm before trusting a run; see the module docstring. |
| 2 | `ligand.py` (`prepare_ligands_extended`) | Strips salts (largest fragment), neutralizes, canonicalizes tautomers, enumerates *undefined* stereocentres only (capped, default 8) as separate docked entries, protonates at pH 7.4 with Dimorphite-DL if installed (else clearly labeled as unmodified), and records embedding/minimization failures in a table instead of dropping them. |
| 3 | `config.py` + `metabolite_pipeline.py` | Validates the YAML config; resolves each target's receptor and box exactly once (identical for every compound docked there, recorded in the manifest); warns (never fails) when a compound's estimated extent may not fit the box -- common for glucuronide/sulfate conjugates. |
| 4 | `validation.py` | Redocks a target's co-crystal ligand and reports heavy-atom RMSD (pass: < 2.0 Å) to `validation_redocking.csv`; a target with no co-crystal ligand is reported as exactly `"not validated: no reference pose"`, never silently skipped. Also docks each target's `reference_ligands` under identical settings for `delta_vs_reference`. |
| 5 | `compare.py` | The flat comparison frame: CNN/Vina affinity and pose score, deltas vs parent and vs reference, ligand efficiency (sign-correct for both score types), per-drug rank and rank change vs parent. Every row traces back to a parent via `parent_id`; failed/NaN docking is kept and flagged, never dropped. |
| 6b | `tables.py` | Splits that frame into `compounds`/`poses`/`screen` (one table per observational unit), coerces nullable dtypes, enforces closed `qc_status` vocabularies and key uniqueness at write time, and writes Parquet (authoritative, manifest embedded) plus generated CSV and `data_dictionary.json`. |
| 6 | `annotate.py`, `applicability.py` | Adds (never filters on) RDKit properties, Lipinski violation count, per-family structural-alert counts across RDKit's catalogs (Brenk, PAINS A/B/C, NIH, ZINC, ChEMBL), a heuristic glucuronide/sulfate `conjugate_flag`, a heuristic `exposure_plausibility` tier, an ADMET-AI hook that skips gracefully (with `admet_status` saying so) if the optional package isn't installed, and a k-NN applicability-domain check against ADMET-AI's own TDC training chemistry. |
| 7 | `figures.py` | Metabolic tree (parent → metabolites, colored by delta), parent-vs-metabolite bar chart per target, compound×target heatmap, and a score-vs-molecular-weight scatter (exposes size bias) -- PNG + SVG, headless. 2D interaction diagrams (PLIP/ProLIF) are a documented `NotImplementedError` stub: neither package was available to build and test against here. |

### Output files (under `outdir`)

- `metabolites.csv`, `prepared_ligands.sdf` / `.csv` (shared across all drugs and targets), `ligand_prep_failures.csv` (if any)
- `<target_id>/`: `receptor_prepared.pdb`, `docking_box.pml`, `results_summary.csv`, `top_results.xlsx`, `ranked_docked.sdf`, `comparison_table.csv` (this target only), plus the redocking/reference-ligand working files
- `tables/` -- the authoritative output, split into three grains (see [Output Tables](#output-tables)): `compounds.parquet`, `poses.parquet`, `screen.parquet`, each with a generated `.csv` beside it, plus `data_dictionary.json`
- `comparison_table.csv` (all targets combined, with annotations) -- kept as a **generated legacy view**; new code should read `screen` joined to `compounds`
- `validation_redocking.csv`, `run_manifest.json`
- `figures/`: one PNG+SVG pair per metabolic tree, per-target bar chart, the compound×target heatmap, and the MW scatter

---

## Quick Start: Dock an Excel Sheet of Compounds With Their Metabolites

This walks through the exact workflow: **you have a spreadsheet of compounds, you want gnina-vs to predict each one's metabolites, and dock the parent plus metabolites against your target.**

There are two ways to do this -- pick based on what you need:

- **`gnina-vs run --metabolite`** (below): one target, one command, no config file. This is almost always the right starting point.
- **`gnina-vs metabolite-run --config config.yaml`** ([further up](#metabolite-aware-docking-metabolite-run)): multiple targets in one run, automatic redocking validation of each target's known ligand, and figures (metabolic tree, heatmap, bar charts). Reach for this once you outgrow the single-target case.

### Step 1 -- Prepare your Excel file

A plain spreadsheet with one compound per row. At minimum it needs a SMILES column; an ID (and/or Name) column is recommended so results are easy to trace back:

| ID | Name | SMILES |
| --- | --- | --- |
| CPD001 | terfenadine | `CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1` |
| CPD002 | ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` |

Save it as `compounds.xlsx` (or `.csv`). Different column names are fine -- point `--smiles-col`/`--id-col`/`--name-col` at them in Step 3.

### Step 2 -- (Optional) Get a BioTransformer jar, or write a curated metabolites CSV

To actually predict metabolites you need one of:

- A real [BioTransformer](https://bitbucket.org/wishartlab/biotransformer3.0/) jar (needs Java) -- pass it as `--biotransformer-jar /path/to/biotransformer3.0.jar`.
- Your own curated metabolites, as a CSV with at least a `smiles` column (and a `parent_id` column, matching each compound's `--id-col` value, if your spreadsheet has more than one compound -- see the warning below) -- pass it as `--metabolites-csv curated.csv`.
- Neither: `--metabolite` still runs, using the more careful metabolite-aware ligand preparation (salt stripping, stereoisomer enumeration, pH 7.4 protonation) on just the parent compounds, with no actual metabolites generated.

### Step 3 -- Run it

```bash
gnina-vs run --protein 1CX2 --ligands compounds.xlsx --metabolite \
    --biotransformer-jar /path/to/biotransformer3.0.jar \
    --ligand-resname S58 \
    --outdir results/my_screen
```

`--ligand-resname S58` tells it exactly which co-crystal ligand to center the docking box on (1CX2's real bound inhibitor -- not a glycosylation sugar like `NAG`; see [Active-Site Detection Priority](#active-site-detection-priority) if you're not sure of a target's ligand code). Omit it, with `--center-x/y/z` also omitted, to let automatic pocket detection try instead.

**If your spreadsheet has more than one compound and you're using `--metabolites-csv`**, give it a `parent_id` column matching each compound's ID, or every curated metabolite gets attached to every compound:

```csv
parent_id,smiles,id,reaction
CPD001,OC(=O)c1ccc(cc1)C(C)(C)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1,fexofenadine,oxidation
```

### Step 4 -- Read the results

Everything lands under `--outdir` (`results/my_screen` above), alongside the usual `run` outputs:

- **`metabolites.csv`** -- every parent + predicted/curated metabolite that was generated, before docking.
- **`tables/`** -- the main new output, split into three grains (see [Output Tables](#output-tables)). `screen.parquet` is the triage grain: one row per target+compound with ligand efficiency, `dcmp_*_vs_parent` (does the metabolite bind better or worse than the original compound?) and rank bands. `compounds.parquet` carries one row per molecule with the annotation columns from [Annotation & Toxicity-Relevant Flags](#annotation--toxicity-relevant-flags). `comparison_table.csv` is still written as a flat legacy view.
- **`results_summary.csv`** / **`top_results.xlsx`** / **`ranked_docked.sdf`** -- the normal `run` outputs, now covering parents and metabolites together.
- **`ligand_prep_failures.csv`** -- only written if a compound couldn't be embedded/minimized; it's reported here, never silently dropped.

`examples/compounds.csv` + `examples/curated_metabolites_multi_drug.csv` are a runnable copy of exactly this walkthrough's spreadsheet and curated-metabolites file (the latter already has the `parent_id` column described above). If you want to try the whole thing without a real BioTransformer jar, Gnina, or network access at all, `examples/run_tiny_example.py` runs the same pipeline end to end against a synthetic receptor.

---

## Project Structure

```
gnina-vs/
├── pyproject.toml
├── README.md
├── examples/
│   ├── metabolite_run_config.yaml   # commented example config
│   ├── curated_metabolites.csv      # --metabolites-csv format example
│   └── run_tiny_example.py          # self-contained runnable example
└── src/
    └── gnina_vs/
        ├── __init__.py             # library API: prepare_protein, prepare_ligands, resolve_center, dock_ligands, extract_results, extract_scores
        ├── __main__.py             # enables `python -m gnina_vs`
        ├── cli.py                  # argparse CLI: run / recover / metabolite-run subcommands (thin wrapper)
        ├── protein.py              # RCSB fetcher + protein preparation (Step 0)
        ├── ligand.py               # SMILES loading + 3-D conformer generation (Step 1); prepare_ligands_extended (Phase 2)
        ├── pocket.py               # active-site detector (explicit + 2-tier auto) + PyMOL box script (Step 2)
        ├── docking.py              # parallel Gnina docking (Step 3)
        ├── reporter.py             # score extraction, CSV/Excel export, recovery (Step 4)
        ├── metabolism.py           # Phase 1: BioTransformer wrapper + curated-CSV bypass
        ├── config.py               # Phase 3: metabolite-run YAML schema + validation
        ├── metabolite_pipeline.py  # Phase 3: orchestrates Phases 1/2/4/5/6/7
        ├── validation.py           # Phase 4: redocking RMSD + reference-ligand docking
        ├── compare.py              # Phase 5: flat comparison frame (deltas, LE, ranks)
        ├── tables.py               # Phase 6b: compounds/poses/screen split, Parquet + manifest
        ├── applicability.py        # Phase 6: k-NN applicability domain vs TDC training chemistry
        ├── annotate.py             # Phase 6: RDKit properties, Brenk/PAINS, conjugate flag
        └── figures.py              # Phase 7: metabolic tree, bar chart, heatmap, scatter
```

---

## Output Files

Running `gnina-vs run --outdir <outdir>` generates:

- `<outdir>/<PDB_ID>.pdb`: the auto-fetched raw receptor, if `--protein` was a PDB ID.
- `<outdir>/receptor_prepared.pdb`: the cleaned, H-added receptor used for docking.
- `<outdir>/prepared_ligands.sdf`: every successfully-embedded, MMFF94-minimized ligand, each tagged with a unique `CompoundID`.
- `<outdir>/docking_box.pml`: PyMOL script visualizing the detected/overridden docking box.
- `<outdir>/docking_outputs/`: one `<compound_id>_docked.sdf` per ligand, plus `gnina_debug.txt` (raw stdout/stderr of the first completed job, for troubleshooting). `<compound_id>` is unique per compound (not the display name), so two compounds that share a name never overwrite each other's output.
- `<outdir>/results_summary.csv`: full results table for every docked compound, including a `status` column (`ok`/`failed`/`timeout`/`no_output`/`error`) and a `pose_quality_flag` column (from the CNN pose score) -- failed jobs and low-confidence poses are kept, not dropped.
- `<outdir>/top_results.xlsx`: the exported top-N (or all) *successfully-scored* compounds, sorted by CNN Affinity then Vina Affinity (or by Vina Affinity alone when `--cnn-scoring none` was used), with auto-fit column widths.
- `<outdir>/ranked_docked.sdf`: the best pose of every exported compound, in rank order, tagged with rank/scores (unless `--no-merged-sdf`).
- `<outdir>/run_manifest.json`: the seed, box center/size, CNN scoring settings, and package/tool versions used, for reproducibility.

`gnina-vs recover` writes the same `results_summary.csv` / `top_results.xlsx` / `ranked_docked.sdf` / `run_manifest.json` set (no receptor/ligand/docking files, since nothing is re-run).

---

## Output Tables

The screening output is split into **three tables, one per observational unit**, written to `tables/` with Parquet authoritative and CSV generated from it.

| Table | Grain | Key | Carries |
|---|---|---|---|
| `compounds` | one row per molecule | `compound_id` | identity, InChIKey, metabolite provenance (`met_depth`/`met_phase`/`met_reaction`), all `mol_*` descriptors, `alert_*` counts, `flag_*`, `exposure_*`, `parent_tanimoto`, `admet_*`, `ad_*` |
| `poses` | one row per target x compound x pose | (`target_id`, `compound_id`, `pose_rank`) | raw scores only: `dock_cnn_affinity`, `dock_vina_affinity_kcal_mol`, `dock_cnn_score`, `qc_pose_quality`, `qc_status`, `run_id` |
| `screen` | one row per target x compound | (`target_id`, `compound_id`) | everything derived: `le_*_per_heavy_atom`, `dcmp_*_vs_parent`, `dcmp_*_vs_ref`, `rank_band_in_group`, `rank_band_change_vs_parent` |

Why split at all: the old flat table repeated every molecular descriptor, alert and ADMET prediction once per target, which is duplication that becomes a correctness hazard the moment one copy is edited; and `rank_in_group` did not say whether it ranked over compounds or over poses, because both lived in one table. `screen` is **derived** from `poses` by a documented rule (best CNNaffinity, else `pose_rank == 1`) and is never hand-maintained. 3D coordinates stay outside all three, in SDF keyed by `pose_key`.

**There is no community schema for docking output**, and no cheminformatics column-naming standard. Two literature searches turned up only tool-specific exports (Meeko, FRED, GOLD). This layout is therefore a *local convention*: `schema_version` in `data_dictionary.json` is the only compatibility signal, and that file states as much.

**Parquet is authoritative for a verified reason.** Writing `{'a': [1.0, None], 'b': [True, None], 'c': ['X', None]}` to CSV and reading it back returns `b` as `object` holding the string `'True'`, with the missing boolean and the missing string indistinguishable empty fields; Parquet preserves `double`, nullable `bool` and `large_string` exactly. Parquet also carries file-level key-value metadata, so the run manifest is embedded in each file as well as written to `run_manifest.json` -- a table emailed without its folder is the normal failure mode. Needs `pip install gnina-vs[tables]`; without pyarrow the tables are written as CSV only, with a warning, and the type fidelity is lost.

**Scores are banded, not ranked to the integer.** `rank_band_in_group` reports a tier (`best`/`upper`/`middle`/`lower`) rather than a position, and `dcmp_cnn_band_vs_parent` collapses any change smaller than the scoring function's resolution into a single `no_meaningful_change` bin. Reporting an integer rank or a three-decimal score is an implied-precision claim that nothing in the docking literature supports. The two resolution constants in `tables.py` -- `CNN_AFFINITY_RESOLUTION` (0.5 pKd units) and `VINA_AFFINITY_RESOLUTION` (1.0 kcal/mol) -- **are placeholders and are not validated**: 1 kcal/mol is a widely repeated rule of thumb, but no source quantified either value, and no paper quantifying docking-score-versus-measured-affinity correlation was retrieved at all. Set them from your own redocking spread and state the value you used.

**The no-drop rule is now architectural rather than conventional.** A compound that fails on target A but succeeds on B gets a `poses` row for A with `qc_status='dock_failed'`, null scores, null `pose_rank` and null `pose_key`, plus normal rows for B -- and exactly one `compounds` row regardless. An unparseable SMILES gets one `compounds` row with `qc_status='smiles_invalid'`, so it stays countable in denominators. Scores are nulled wherever `qc_status != 'ok'`, so a failed job can never show a score. Four uniqueness and referential checks run at write time and raise rather than warn, so a broken schema cannot reach disk. Note that no other pipeline documents a convention for retaining failed dockings as rows (DOCKSTRING drops them, discarding 1.8% of molecules) -- this is a defensible local policy, not an inherited standard.

**Current limitation: the `poses` grain is not yet real.** `reporter._parse_gnina_log` stops at mode 1 by design, so the pipeline carries exactly one pose per compound and every `poses` row has `pose_rank == 1`. The keys and the selection rule are already correct for the multi-pose case, so making it real is a change to score extraction in `reporter.py` and nothing in `tables.py`. That change was deliberately not made blind -- parsing every mode across gnina versions cannot be validated against the fixtures in this repo alone -- and it is a prerequisite for PoseBusters gating, interaction-difference tables and showing more than the top pose.

---

## The HTML Triage Report

`metabolite-run` also writes `triage_report.html`: one self-contained file holding parents and their metabolites with structures, bands, flags and QC status, sortable and searchable, meant both for your own triage and for handing to someone who will only open it.

**Self-contained means measured, not assumed.** The requirement is that it opens by double-clicking, with no server and no network. Most of the obvious ecosystem fails that, which is why `report.py` uses the one combination that does not (measured with mols2grid 2.2.0, itables 2.9.1, bokehmol 0.2.0):

| Option | External requests | Verdict |
|---|---|---|
| mols2grid, interactive (default) | 4 | **Not offline.** Pulls RDKit.js from unpkg plus jQuery, bootstrap and list.js. The file opens, but no structure renders and sorting is dead. |
| mols2grid, `template="static"`, `prerender=True` | 2 | Images baked in and RDKit.js gone, but still needs jQuery and bootstrap -- and "static" means no sorting or filtering. |
| bokehmol (`hover.rdkit` or `hover.smiles_drawer`) | 2 | **Not offline even with `resources=INLINE`** -- fetches RDKit.js or smiles-drawer from unpkg at runtime. |
| **itables, `connected=False`, bundle inlined** | **0** | Genuinely offline, with sorting, filtering and pagination. |

So structures are pre-rendered to inline SVG by RDKit here rather than drawn in the browser, which is what removes the RDKit.js dependency. `tests/test_report.py` asserts the finished file references zero external URLs, so a regression fails the suite rather than being discovered by a colleague with no wifi.

Two bugs worth knowing about, because both were silent and neither raised anything:

- **itables defaults to `maxBytes='64KB'` and downsamples past it.** An inline depiction is 6-7 KB, so the cap is crossed at about ten molecules and the table simply drops compounds. That would break this project's one hard rule, so downsampling is disabled (`maxBytes=0`) and size is controlled by `max_rows`, which *reports* what it truncated, on the page.
- **RDKit writes newlines inside SVG path `d` attributes.** Serialised into the page they become literal `\n`, the browser rejects the path with "Expected path command", and every structure renders blank. `molecule_svg` collapses whitespace before embedding.

**The caveats travel with the table.** A supervisor reads the page, not this README, so the header states on the file itself that docking scores are not affinities, that there are deliberately no error bars (the dominant uncertainty is method bias, not sampling spread, so an error bar would measure re-docking variance while reading as distance to the truth), that the columns are not commensurable, and that nothing has been filtered.

**Size.** The DataTables bundle is a fixed ~950 KB; each molecule adds roughly 2-4 KB of SVG. A 60-row report is ~1.1 MB. Pass `with_depictions=False` or lower `max_rows` for something small enough to email, and check the logged size before sending.

**Not included:** no 3D viewer and no interaction diagram. 3Dmol.js can be inlined and a parent-versus-metabolite ProLIF interaction-difference table would be the single most informative artifact available, but the interaction diff depends on a deterministically protonated receptor that the pipeline does not yet guarantee -- phase-II metabolites are typically anionic, so a "lost H-bond, gained salt bridge" result can be an artifact of preparation rather than chemistry. Wiring it up before fixing protonation would produce a confident-looking figure built on an assumption.

---

## Notes and Limitations

- Gnina must be installed separately; it is not a Python dependency and can't be `pip install`-ed.
- `--keep-hetatm` is the only way to retain a HETATM cofactor (e.g. a catalytic zinc) through protein preparation; everything else in a HETATM record is stripped, including any co-crystallised ligand, which is why Tier 1 needs to search the *original* PDB rather than the prepared one. Selenomethionine and other in-chain modified residues are always kept regardless of `--keep-hetatm` (see [Active-Site Detection Priority](#active-site-detection-priority) and the protein-preparation step above).
- There is no fully-automatic third pocket-detection tier: a purely geometric cavity detector was tried and found unreliable (it tended to return the protein's own centre rather than a real pocket), so it was removed from the automatic chain. If Tier 1 and Tier 2 both fail, the run stops with a clear error instead of guessing.
- Ligand preparation strips salts/counter-ions and keeps only the largest fragment (logged), and warns when a compound has an undefined stereocentre -- in that case, only one arbitrary stereoisomer is embedded and docked; specify stereochemistry explicitly, or supply each isomer as its own row, if this matters.
- Gnina's stdout table format has changed across versions; score parsing tries three strategies (space-delimited table, key=value, pipe-delimited table) before falling back to the properties embedded in the output SDF. If none of these find the docked pose's scores, the compound is exported with blank/NaN scores rather than silently dropped.
- `--top-n` only ever exports compounds that have a ranking score (CNN Affinity when any CNN score was produced in the run, otherwise Vina Affinity); ligands that failed to dock, timed out, or errored are still present in the full `results_summary.csv` with their `status` set accordingly, and every row's `pose_quality_flag` is reported without removing any row.
- A high CNN affinity paired with a low CNN pose score (see `pose_quality_flag` in the output) usually means the *pose* is unreliable, not that the compound truly binds well -- inspect such hits before trusting the ranking.
- This tool is intended for research and educational purposes; predicted docking scores are not a substitute for experimental validation.

### Metabolite-aware pipeline (`metabolite-run`) limitations

These apply on top of everything above:

- **Docking scores are not affinities.** CNN affinity is a learned, roughly pKd-like estimate, and Vina/Gnina affinity is a scoring-function value, not a measured binding constant; both are size- and training-set-biased (see the `figures/scatter_score_vs_mw.*` plot). Use `ligand_efficiency_*` and `delta_vs_parent`/`delta_vs_reference` for comparisons, not absolute scores.
- **BioTransformer predictions lack kinetics and exposure information.** A predicted metabolite says nothing about how much of it forms, how fast, or whether it ever reaches a relevant concentration in vivo -- this pipeline compares *if it were present*, not *whether it will be*.
- **Reactive-metabolite toxicity and covalent mechanisms are not captured.** `annotate.py` counts Brenk matches, which include known reactive substructures, but non-covalent docking (what Gnina does) cannot model covalent binding and no mechanistic toxicity prediction is performed. Reactive-metabolite risk is in any case not computable from structure alone: the published formulation is Dose x Fa x CL_RM/CL_total, and two of those three terms are not structural.
- **hERG docking is low-confidence.** hERG's binding cavity is large, flexible, and promiscuous; docking scores against it should be treated as a coarse screen at best, not a reliable predictor of cardiotoxicity risk.
- **Annotations are not gates.** Every column added by `annotate.py` (Lipinski violations, Brenk/PAINS alerts, `conjugate_flag`, ADMET-AI predictions when available) is informational. Nothing in this pipeline removes, reorders, or routes a compound based on these values -- that filtering, if wanted, is a decision for the person reading `comparison_table.csv`, not something this code does automatically.
- **BioTransformer's exact CLI flags and output column names were not verified against a real installed jar** (see `metabolism.py`'s module docstring) -- only a JRE was available in this environment, no jar. Run `metabolism.check_biotransformer(jar_path)` and one real prediction before trusting a production run.
- **Redocking RMSD assumes atom-order correspondence** between the extracted co-crystal ligand and Gnina's docked pose (see `validation.py`'s module docstring); a mismatched heavy-atom count is reported rather than silently producing a wrong RMSD, but this is not a fully general graph-based comparison.
- **2D interaction diagrams (PLIP/ProLIF) are not implemented** -- `figures.interaction_diagram` raises `NotImplementedError` with a clear message; neither package was available to build and test against in this environment.
- **The applicability-domain check is coarse, and its reference set is assembled rather than official.** ADMET-AI ships no applicability domain, so `gnina_vs.applicability` rebuilds one from the TDC ADMET training SMILES. It reports a single in/out answer across all endpoints, where the correct object is one domain per endpoint; and inside the domain means "the training set contains similar chemistry", never "this prediction is correct". The TDC download path is unverified -- check the molecule count after running `gnina-vs fetch-ad-reference`.
- **`exposure_plausibility` is a descriptor heuristic, not a PK model.** It has no model of any tissue, is not a blood-brain-barrier model, and is not validated against measured exposure. It also reads the formal charge of the SMILES *as written*, not a pKa-derived charge at pH 7.4, so a carboxylic acid drawn neutral reads as uncharged unless the upstream protonation step ran.

---

## Acknowledgments

The refactoring of this pipeline into a modular CLI, structural workflow design, and specific implementations (such as the two-tier ADMET-AI applicability domain proxy) were developed with the assistance of large language models, specifically Claude and Gemini.

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).