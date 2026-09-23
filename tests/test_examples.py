"""The bundled sample campaign in `examples/lab_exports/`.

Every file there is ingested by `conftest.build_example_store`, the same way
`INGEST.md` tells a reader to do it by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import BUILD, EXAMPLE_FILES, EXAMPLES
from monocell.schema.manifest import list_experiments


def test_every_sample_file_ingests_without_breaking_a_schema_rule(example_store):
    """A sample file that trips a rule would teach a reader that the flags
    can be ignored."""
    stored = list_experiments("demo", None, example_store)
    assert sorted(r["exp_type"] for r in stored) == sorted(t for _, t, _ in EXAMPLE_FILES)

    broken = {}
    for rec in stored:
        meta = json.loads((Path(rec["path"]) / "meta.json").read_text(encoding="utf-8"))
        rules = [f for f in meta["quality"]["flags"] if f.startswith("rule:")]
        if rules:
            broken[rec["experiment_id"]] = rules
    assert not broken, broken


def test_a_soc_column_follows_the_charge_that_was_counted(example_store):
    """Synthetic or not, a state of charge that moves differently from the
    current would be a sample nobody should model from. Positive current is
    discharge across the whole campaign, so SOC falls by the counted charge
    over the capacity, wherever a current flows."""
    capacity = BUILD["capacity_Ah"]
    wrong = []
    for rec in list_experiments("demo", None, example_store):
        series = pd.read_parquet(Path(rec["path"]) / "series.parquet")
        if not {"t_s", "I_A", "soc"} <= set(series.columns):
            continue
        t, i, soc = (series[c].to_numpy(dtype=float) for c in ("t_s", "I_A", "soc"))
        driven = (np.abs(i[1:]) > 0) & (np.abs(i[:-1]) > 0)
        counted = -0.5 * (i[1:] + i[:-1]) * np.diff(t) / 3600.0 / capacity
        if not (0.0 <= soc.min() and soc.max() <= 1.0) or not np.allclose(
                np.diff(soc)[driven], counted[driven], atol=1e-6):
            wrong.append(rec["exp_type"])
    assert not wrong, wrong


def test_the_file_list_the_sidecars_and_the_instructions_agree():
    """Three places name the sample files: the directory, the test list and
    `INGEST.md`. A file added to one and not the others is caught here."""
    on_disk = {p.relative_to(EXAMPLES).with_suffix("").as_posix() for p in EXAMPLES.rglob("*.csv")}
    listed = {stem for stem, _, _ in EXAMPLE_FILES}
    assert on_disk == listed

    missing_sidecars = [s for s in listed if not (EXAMPLES / f"{s}.sidecar.json").exists()]
    assert not missing_sidecars, missing_sidecars

    instructions = (EXAMPLES / "INGEST.md").read_text(encoding="utf-8")
    folders = {Path(s).parts[0] for s in listed}
    unmentioned = [f for f in sorted(folders) if f"lab_exports/{f}" not in instructions]
    assert not unmentioned, unmentioned
