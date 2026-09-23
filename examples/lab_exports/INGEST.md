# Ingesting the sample campaign

Synthetic exports for one 5 Ah NMC811/graphite cell, each with a sidecar JSON
beside it. Positive current is discharge throughout. Run the commands from
`examples/`; they write a store in `examples/demo_store/`.

## 1. Register the cell

```bash
monocell cell register --cell demo --capacity-Ah 5.0 --chemistry "NMC811/graphite" --data-root demo_store
```

## 2. The cycler exports, as a folder

`lab_exports/cycler/` holds the files with the cycler-table shape (time,
current, voltage): six Landt-format reference performance tests from fresh to
1000 equivalent full cycles, an HPPC sweep, a GITT and three three-electrode
charges.

Look first. This names the instrument profile each file matches and the
experiment type its shape proposes, and writes nothing:

```bash
monocell inspect --dir lab_exports/cycler --cell demo --data-root demo_store
```

Print the ingest plan, then write it:

```bash
monocell ingest --cell demo --dir lab_exports/cycler --data-root demo_store
monocell ingest --cell demo --dir lab_exports/cycler --data-root demo_store --confirm
```

## 3. The EIS and pressure exports, one at a time

The EIS file is a Palmsense4 export (UTF-16, several blocks); the pressure file
comes from a stack-pressure rig. Each has its own reader, selected by `--type`:

```bash
monocell ingest --cell demo --type eis --file lab_exports/eis/eis_fresh.csv --sidecar lab_exports/eis/eis_fresh.sidecar.json --data-root demo_store
monocell ingest --cell demo --type pressure --file lab_exports/pressure/pressure_cycle.csv --sidecar lab_exports/pressure/pressure_cycle.sidecar.json --data-root demo_store
```

## 4. Query, derive, simulate

```bash
monocell find --cell demo --data-root demo_store
monocell rederive --cell demo --data-root demo_store
monocell simulate --cell demo --c-rate 1 --data-root demo_store
```

`rederive` writes the cell's parameter file, citing every experiment above.
Ingest another file and `rederive` reports the parameter file as stale.
