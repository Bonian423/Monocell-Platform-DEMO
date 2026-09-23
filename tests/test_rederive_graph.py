"""The re-derive machinery, driven by a toy graph.

The shipped graph holds one module (the abstracted parameter extraction), and
one module cannot show the parts of `rederive` that are general: a run order
over artifact edges, a module going stale because an upstream ARTIFACT moved,
the fixed-point loop and its guards. `conftest.TOY_GRAPH` has three modules
whose runners only record what they consumed, which is all the checker reads.
"""

from __future__ import annotations

import json

import pytest

from conftest import BUILD, register_cell, write_eis, write_hppc
from monocell.rederive import list_stale, rederive
from monocell.schema.artifacts import load_artifact, newest_artifact
from monocell.schema.manifest import content_hash

CELL = "graph_cell"


@pytest.fixture
def cell(root, toy_graph):
    register_cell(CELL, BUILD, root)
    return CELL


def _modules(rows) -> list[str]:
    return [r["module"] for r in rows]


# ---------------------------------------------------------------------------
# order
# ---------------------------------------------------------------------------

def test_a_producer_runs_before_its_consumer_whatever_their_names(cell, root):
    """`cell_model` reads the newest `spectra` artifact and sorts before it.
    Run alphabetically, it would read the PREVIOUS spectra artifact (here: none)
    and the pass would end with it stale."""
    write_eis(cell, root)
    write_hppc(cell, root)

    order = _modules(rederive(cell, root))
    assert order.index("spectra") < order.index("cell_model"), order
    assert list_stale(cell, root) == []


def test_a_cycle_in_the_graph_is_refused(cell, toy_graph, monkeypatch):
    """Running a cycle in some arbitrary order would bring back the bug the
    ordering exists to prevent, so it raises."""
    from monocell.rederive import _topo_order

    toy_graph["spectra"]["input_artifacts"] = ("cell_model",)
    with pytest.raises(ValueError, match="cycle in MODULE_GRAPH"):
        _topo_order(["cell_model", "spectra"])


def test_without_an_edge_the_order_is_alphabetical(cell):
    """Deterministic, and identical to `sorted()` wherever no edge inverts it."""
    from monocell.rederive import _topo_order

    assert _topo_order(["spectra", "pulses"]) == ["pulses", "spectra"]
    assert _topo_order(["cell_model", "spectra", "pulses"]) == ["pulses", "spectra", "cell_model"]


# ---------------------------------------------------------------------------
# what makes a module stale
# ---------------------------------------------------------------------------

def test_a_module_that_needs_two_types_waits_for_both(cell, root):
    """`cell_model` consumes EIS and HPPC. With only one of them it has nothing
    to derive, and reporting it stale would ask for a run that cannot happen."""
    write_eis(cell, root)
    assert _modules(list_stale(cell, root)) == ["spectra"]

    write_hppc(cell, root)
    assert set(_modules(list_stale(cell, root))) == {"spectra", "pulses", "cell_model"}


def test_per_experiment_and_cell_level_rows_have_different_shapes(cell, root):
    """A per-experiment module reports one row per experiment; a cell-level
    module reports one row carrying every id it will be run with."""
    eis_a, eis_b = write_eis(cell, root), write_eis(cell, root)
    hppc = write_hppc(cell, root)

    rows = list_stale(cell, root)
    spectra = [r for r in rows if r["module"] == "spectra"]
    assert sorted(r["experiment_id"] for r in spectra) == sorted([eis_a, eis_b])
    (model,) = [r for r in rows if r["module"] == "cell_model"]
    assert model["experiment_id"] is None
    assert sorted(model["experiment_ids"]) == sorted([eis_a, eis_b, hppc])


def test_a_changed_upstream_artifact_makes_its_consumer_stale(cell, root):
    """No experiment moved; the artifact `cell_model` read did. The row names
    the artifact and carries no experiment ids, because none are new."""
    write_eis(cell, root)
    write_hppc(cell, root)
    rederive(cell, root)

    upstream = newest_artifact(cell, "spectra", "spectrum_*.json", root)
    env = load_artifact(upstream)
    env["params"] = {"note": "re-run with different settings"}
    upstream.write_text(json.dumps(env, indent=2), encoding="utf-8")

    (row,) = list_stale(cell, root)
    assert row["module"] == "cell_model"
    assert row["experiment_ids"] == []
    assert "artifact:spectra" in row["reason"]

    assert _modules(rederive(cell, root)) == ["cell_model"]
    model = newest_artifact(cell, "cell_model", "model_*.json", root)
    recorded = {r["id"]: r["hash"] for r in load_artifact(model)["inputs"]}
    assert recorded["artifact:spectra"] == content_hash(upstream)


def test_an_edited_experiment_is_stale_for_every_module_that_read_it(cell, root):
    """The hash comparison. Experiments are append-only, so this is what a
    hand edit to a stored file looks like to the checker."""
    eis = write_eis(cell, root)
    write_hppc(cell, root)
    rederive(cell, root)

    meta = root / "experiments" / cell / eis / "meta.json"
    meta.write_text(meta.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    rows = list_stale(cell, root)
    assert set(_modules(rows)) == {"spectra", "cell_model"}
    (spectra,) = [r for r in rows if r["module"] == "spectra"]
    assert spectra["reason"] == "input hash changed"


# ---------------------------------------------------------------------------
# the fixed point
# ---------------------------------------------------------------------------

def test_a_rescan_catches_what_the_same_pass_made_stale(cell, root):
    """`cell_model` is current when the pass starts. Re-deriving an older
    spectrum makes it the newest `spectra` artifact, and only then is
    `cell_model` stale. One scan cannot see that, so `rederive` scans again."""
    write_eis(cell, root)
    write_eis(cell, root)
    write_hppc(cell, root)
    rederive(cell, root)

    newest = newest_artifact(cell, "spectra", "spectrum_*.json", root)
    (older,) = [p for p in newest.parent.glob("spectrum_*.json") if p != newest]
    older.unlink()
    assert _modules(list_stale(cell, root)) == ["spectra"]

    assert _modules(rederive(cell, root)) == ["spectra", "cell_model"]
    assert list_stale(cell, root) == []


def test_a_runner_that_does_not_record_its_inputs_is_refused(cell, root, monkeypatch):
    """Such a module would be stale again straight after every run. The loop
    raises instead of repeating the run until its round limit."""
    from monocell import rederive as rd
    from monocell.schema.artifacts import write_artifact

    def forgetful(ids, out_dir, root=None):
        return write_artifact("spectra", f"spectrum_{ids[0]}.json", {}, [], {}, CELL,
                              out_dir, root)

    monkeypatch.setitem(rd.RUNNERS, "spectra", forgetful)
    write_eis(cell, root)
    with pytest.raises(ValueError, match="not recording what it consumed"):
        rederive(cell, root)


def test_a_dry_run_names_the_jobs_and_writes_nothing(cell, root):
    write_eis(cell, root)
    write_hppc(cell, root)

    jobs = rederive(cell, root, dry_run=True)
    assert jobs and all(j["dry_run"] for j in jobs)
    assert not (root / "artifacts" / cell).exists()
    assert len(list_stale(cell, root)) == 3


def test_an_artifact_input_with_no_declared_glob_is_refused(cell, toy_graph, root):
    """An input that cannot be located cannot be hashed, and an unhashed input
    never marks anything stale. The graph is refused where the name is known."""
    toy_graph["cell_model"] = {**toy_graph["cell_model"], "artifact_input_globs": {}}
    write_eis(cell, root)
    write_hppc(cell, root)
    with pytest.raises(ValueError, match="no glob"):
        list_stale(cell, root)
