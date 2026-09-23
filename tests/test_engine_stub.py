"""The abstracted extraction step, and the contract around it.

The values are a stand-in. What is tested is everything the rest of the
pipeline relies on: the values are PyBaMM's own, the parameter file cites every
experiment it was derived from, and new data makes it stale.
"""

from __future__ import annotations

import pytest

from conftest import BUILD, register_cell, write_eis, write_hppc
from monocell.engine.extract import (
    BASE_PARAMETER_SET,
    STANDARD_PARAMETERS,
    extract_parameters,
    run,
)
from monocell.rederive import list_stale, rederive
from monocell.schema.artifacts import load_artifact, newest_artifact
from monocell.schema.manifest import list_experiments
from monocell.schema.write_experiment import experiment_hash


def _newest(cell_id, root):
    return newest_artifact(cell_id, "parameters", "parameter_file_*.json", root)


def test_every_standard_value_is_pybamms_own():
    """The literals in `STANDARD_PARAMETERS` are checked against PyBaMM's copy
    of the base set, key by key, so a typo cannot pass as a published value."""
    import pybamm

    published = pybamm.ParameterValues(BASE_PARAMETER_SET)
    for key, value in STANDARD_PARAMETERS.items():
        assert key in published.keys(), f"{key!r} is not a {BASE_PARAMETER_SET} parameter"
        assert float(published[key]) == pytest.approx(value, rel=1e-12), key


def test_the_stand_in_does_not_read_its_inputs():
    """Stated in its docstring and in the record it returns, and checked here."""
    a = extract_parameters([], {})
    b = extract_parameters([{"meta": {}, "series": None}], {"capacity_Ah": 1.0})
    assert a == b
    assert a["source"] == "abstracted" and a["note"]
    assert a["base_parameter_set"] == BASE_PARAMETER_SET


def test_the_parameter_file_cites_every_ingested_sample_file(example_store):
    """The whole sample campaign in, one parameter file out, and every
    experiment in it cited at its current content hash."""
    rederive("demo", example_store)
    env = load_artifact(_newest("demo", example_store))

    stored = {r["experiment_id"]: r["exp_type"] for r in list_experiments("demo", None, example_store)}
    cited = {r["id"]: r["hash"] for r in env["inputs"]}
    assert set(cited) == set(stored)
    for eid, h in cited.items():
        assert h == experiment_hash("demo", eid, example_store), eid

    by_type: dict[str, list[str]] = {}
    for eid, exp_type in stored.items():
        by_type.setdefault(exp_type, []).append(eid)
    assert {t: sorted(ids) for t, ids in env["data"]["consumed"].items()} == \
        {t: sorted(ids) for t, ids in by_type.items()}


def test_new_data_makes_it_stale_and_adds_a_version(root):
    """Each distinct input set is its own file, so the earlier version stays on
    disk and the newest one is the current answer."""
    register_cell("c1", BUILD, root)
    first = write_eis("c1", root)
    rederive("c1", root)
    v1 = _newest("c1", root)
    assert list_stale("c1", root) == []

    second = write_hppc("c1", root)
    (row,) = list_stale("c1", root)
    assert row["module"] == "parameters" and row["experiment_ids"] == [second]

    rederive("c1", root)
    v2 = _newest("c1", root)
    assert v2 != v1 and v1.exists()
    assert {r["id"] for r in load_artifact(v2)["inputs"]} == {first, second}
    assert list_stale("c1", root) == []


def test_one_parameter_file_describes_one_cell(root):
    register_cell("c1", BUILD, root)
    register_cell("c2", BUILD, root)
    ids = [write_eis("c1", root), write_eis("c2", root)]
    with pytest.raises(ValueError, match="one cell"):
        run(ids, root / "artifacts" / "c1" / "parameters", root=root)


def test_no_experiments_is_refused(root):
    with pytest.raises(ValueError, match="no experiments"):
        run([], root / "artifacts" / "c1" / "parameters", root=root)
