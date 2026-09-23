# monocell: Interview Guide

monocell organizes battery-test evidence into traceable experiments, analyses,
and model outputs. This guide explains the original platform at a high level and
shows where this review copy stops.

> **Scope and disclosure:** These diagrams describe system boundaries and data
> flow only. They intentionally omit proprietary extraction logic, exact model
> values, tuned thresholds, instrument-identification signatures, and
> non-public datasets. This repository is an interview demo copy, not the full
> original platform.

## Platform Loop

![High-level flow from test evidence through ingest, analysis, model evaluation, and the next experiment.](assets/original-platform.svg)

In the original platform, a cell record connects evidence from different test
types. Each experiment enters through a shared ingest contract; analysis modules
produce versioned results; model assembly uses those results with explicit
provenance; and model predictions are checked against measurements not used to
build them. When new evidence arrives or an experiment is withdrawn, dependent
results can be refreshed instead of silently reused.

## Ingest Layer

![Detailed, implementation-neutral ingest flow with preview, human confirmation, schema refusal, quality flags, and durable storage.](assets/ingest-layer.svg)

The ingest path is designed to make a file's interpretation reviewable before
it becomes part of a cell record:

- Inspection proposes a format and experiment type and shows what the reader
  can and cannot map. It does not commit a folder plan.
- A person confirms the interpretation. Ambiguous types and unknown explicit
  profiles are surfaced rather than silently guessed.
- Format adapters map source fields into a canonical experiment shape and
  report missing, empty, or unplaced data.
- A structural contract failure stops the write. A valid experiment with a
  quality concern is retained with visible flags for review.
- Stored records include the parsed frame, canonical series, metadata, and
  quality results. Corrections are added as new records rather than rewriting
  prior evidence.

The illustration stays at the contract level; it does not expose vendor header
rules, parser internals, or numeric quality gates.

## Evidence And Validation

![Three distinct validation paths: synthetic truth recovery, export-and-ingest round trips, and held-out prediction.](assets/evidence-validation.svg)

These checks answer different questions. Controlled synthetic data asks whether
an analysis recovers a known answer under declared conditions. Export-and-ingest
round trips ask whether file handling preserves the experiment for downstream
consumers. Held-out measurements ask whether a model predicts evidence it did
not use. Passing one is not a substitute for the others.

## This Review Copy

The copy retains the shared experiment schema, file readers and instrument
profiles, ingest checks, append-only store, query index, re-derivation path, and
PyBaMM hand-off. Its sample exports are synthetic. Parameter extraction is
intentionally abstracted: [the extraction module](../monocell/engine/extract.py)
returns a published parameter set independent of the sample data. Therefore,
the simulation demonstrates the software hand-off, not a data-fitted cell
model.

| Area | What this copy demonstrates | What remains outside this copy |
|---|---|---|
| Ingest | File interpretation, normalization, schema checks, quality flags, and persistence | Validation against a broad collection of production exports |
| Analysis | Ingested records and quality summaries | The original experiment-analysis and model-building implementations |
| Validation | Focused ingest, profile, and quality tests | The original scientific round-trip harnesses and held-out model results |
| Simulation | Parameter-file hand-off to PyBaMM | Parameters inferred from the bundled experiments |

For a runnable example, use the [sample ingest instructions](../examples/lab_exports/INGEST.md)
and [walkthrough notebook](../examples/walkthrough.ipynb). The root
[README](../README.md) documents installation, commands, and the exact scope of
the extraction stand-in.