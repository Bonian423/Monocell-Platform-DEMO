# monocell

Battery test data, from instrument exports to a PyBaMM model.

monocell reads exports from battery cyclers, potentiostats and a stack-pressure
rig into one schema, checks each file as it is written, and keeps every
experiment in an append-only store. From the store it derives a parameter file
for the cell and runs it in PyBaMM. When new data arrives, or an experiment is
withdrawn, anything derived from the old data is reported stale and can be
re-derived.

> **The parameter-extraction module is abstracted in this copy for IP
> reasons.** In the full platform, this step analyses each test type and
> assembles a cell-specific PyBaMM parameter set with per-parameter
> provenance. Here, `monocell/engine/extract.py` keeps the same interface and
> returns published PyBaMM `Chen2020` values, so the rest of the pipeline runs
> end to end.

This repository is shared for review only. See [LICENSE](LICENSE).

> Interviewer guide: [high-level platform overview, ingest flow, validation paths, and demo scope](docs/interview-guide.md).

## Pipeline

[![Compact pipeline from instrument exports to the PyBaMM hand-off, including the extraction boundary in this review copy.](docs/assets/pipeline.svg)](docs/assets/pipeline.svg)

The diagram is sized for the README preview. Open the linked SVG for a scalable
view.

## Platform diagrams

### Original platform loop

[![Cell evidence flows through ingest, analysis, model assembly, held-out validation, and a next-test feedback loop.](docs/assets/original-platform.svg)](docs/assets/original-platform.svg)

### Detailed ingest layer

[![Detailed ingest flow showing inspection, confirmation, mapping, schema refusal, quality flags, and append-only storage.](docs/assets/ingest-layer.svg)](docs/assets/ingest-layer.svg)

### Evidence validation

[![The three separate checks: synthetic truth recovery, file round trip, and held-out model prediction.](docs/assets/evidence-validation.svg)](docs/assets/evidence-validation.svg)

## What is here

| Part | Where | What it does |
|---|---|---|
| Readers | `monocell/schema/ingest.py` | Cycler CSVs by header synonyms, Palmsense4 CSV (UTF-16, several blocks) and EIS XLSX, GITT step logs, stack-pressure CSVs, and `misc` for a file of no known shape |
| Instrument profiles | `monocell/schema/instruments.py` | Landt, Neware, Maccor and Arbin export formats: column names, units, clock restarts per step, step labels, detection by header signature |
| Type by shape | `monocell/schema/shapes.py` | Proposes an experiment type from the current and voltage, with the evidence as a sentence. A proposal is never applied automatically |
| Schema | `monocell/schema/tables.py` | One spec per experiment type: columns, required instrument metadata, checks, ingest rules |
| Checks at ingest | `monocell/schema/quality.py` | Three-electrode sum check and reference drift, linear Kramers-Kronig residual on EIS, capacity SOH for check-ups and cycling, timestamp order, profile parse report |
| Store | `monocell/schema/write_experiment.py`, `manifest.py`, `artifacts.py` | Append-only experiments, retraction without deletion, a DuckDB index with search, a provenance envelope on every derived file |
| Re-derivation | `monocell/rederive.py`, `monocell/autofit.py` | Staleness by content hash, run order from the dependency graph, a fixed-point loop, and derive-on-ingest scoped to what the new file feeds |
| PyBaMM bridge | `monocell/simulate.py` | Loads the named base parameter set, applies the file's overrides (unknown keys are refused) and runs a constant-current discharge |
| CLI | `monocell/cli.py` | `cell`, `inspect`, `ingest`, `find`, `promote`, `retract`, `manifest`, `rederive`, `simulate` |

| Abstracted | Where |
|---|---|
| Parameter extraction | `monocell/engine/extract.py`: same interface, published `Chen2020` values |

## Quickstart

Python 3.11 or later.

```bash
python -m venv .venv
```

Activate the environment, then install the package with its test and notebook
extras:

```bash
pip install -e ".[test,notebook]"
```

The sample campaign in `examples/lab_exports/` is synthetic data for a 5 Ah
NMC811/graphite cell. `examples/lab_exports/INGEST.md` loads all of it from the
command line; the short version, run from `examples/`:

```bash
monocell cell register --cell demo --capacity-Ah 5.0 --chemistry "NMC811/graphite" --data-root demo_store
monocell ingest --cell demo --dir lab_exports/cycler --data-root demo_store --confirm
monocell rederive --cell demo --data-root demo_store
monocell simulate --cell demo --c-rate 1 --data-root demo_store
```

`examples/walkthrough.ipynb` goes through the same steps in Python, with the
stored records, the checks and the PyBaMM result shown along the way.

## Design notes

- **One schema, many instruments.** Readers and profiles map each vendor's
  column names and units onto one set of columns per experiment type. A
  profile is data (column rules and a header signature), so supporting a new
  rig means writing a profile.
- **Detection proposes and the user decides.** A matching profile or a type
  proposal is printed with its evidence. The user names the profile and the
  type, or confirms a folder plan before anything is written.
- **Checks run at ingest and are stored with the data.** A failed check or a
  missing protocol fact becomes a flag on the experiment. The file is still
  stored, and the flag travels with it.
- **Append-only.** Nothing is edited after it is written. A correction is a new
  experiment that records `derived_from`, or a retraction with a reason; in
  both cases the original files stay.
- **Staleness by content hash.** Every derived file records the id and hash of
  each input. New, edited or retracted data makes it stale, and `rederive`
  re-runs what is stale, in dependency order, until nothing is.
- **The index can be rebuilt.** The DuckDB manifest is rebuilt from the files
  with `monocell manifest rebuild`.
- **PyBaMM stays at the edge.** Ingest, queries and re-derivation never import
  it, and neither does `import monocell`.

## Layout

```
monocell/
  schema/        readers, instrument profiles, type proposal, specs, checks,
                 experiment store, manifest, artifacts
  engine/        parameter extraction (abstracted)
  cells.py       build records
  batches.py     lots and their member cells
  rederive.py    staleness and re-derivation
  autofit.py     derive-on-ingest
  simulate.py    PyBaMM bridge
  cli.py         command line
  journal.py     activity log
examples/
  lab_exports/   sample campaign: cycler/, eis/, pressure/, INGEST.md
  walkthrough.ipynb
tests/
```

## Tests

```bash
pytest -n 4
```

The suite runs on the sample campaign and on small frames built inside the
tests. `pytest -m "not slow"` skips the PyBaMM solves.
