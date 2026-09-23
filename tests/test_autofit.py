"""Auto-derive on ingest: the scope, the failure path, and the CLI flag.

The load-bearing test here is `test_the_scope_leaves_other_modules_stale`. A
naive implementation ("after the ingest, call `rederive`") passes every other
test in this file, because on a clean store a full re-derivation and a scoped
one produce the same modules. They differ only when something ELSE is stale,
and that is what the test builds: an HPPC-only module that an EIS-driven pass
must leave alone. The test asserts both halves, so it cannot pass because
nothing else happened to be stale.

The scope tests run on `conftest.TOY_GRAPH`, because the shipped graph has one
module and a scope over one module has nothing to leave out.
"""

from __future__ import annotations

import pytest

from conftest import BUILD, EXAMPLES, register_cell, write_eis, write_hppc
from monocell.autofit import derive_for, modules_for, scope_for, summary_lines
from monocell.cli import main as cli_main
from monocell.rederive import MODULE_GRAPH, ModuleRunError, list_stale, rederive
from monocell.schema.manifest import list_experiments
from monocell.schema.tables import SPECS

CELL = "auto_cell"


@pytest.fixture
def cell(root):
    register_cell(CELL, BUILD, root)
    return CELL


def _modules(rows) -> set[str]:
    return {r["module"] for r in rows}


# ---------------------------------------------------------------------------
# the module map is the graph's, not a second copy of it
# ---------------------------------------------------------------------------

def test_modules_for_is_read_off_the_graph():
    """Every (module, type) edge in `MODULE_GRAPH` appears in `modules_for`,
    for every type, and no edge appears that the graph does not declare."""
    types = {t for spec in MODULE_GRAPH.values() for t in spec["input_experiments"]}
    assert types, "the graph declares no experiment types at all"
    for t in sorted(types):
        for module, spec in MODULE_GRAPH.items():
            assert (module in modules_for(t)) == (t in spec["input_experiments"]), (t, module)


def test_every_measured_type_feeds_the_parameter_file_and_misc_feeds_nothing():
    """`misc` is where a file the schema has no shape for lands. It unlocks
    nothing until it is promoted, and an unknown type is not an error."""
    for exp_type in SPECS:
        expected = () if exp_type == "misc" else ("parameters",)
        assert modules_for(exp_type) == expected, exp_type
    assert modules_for("no_such_type") == ()


# ---------------------------------------------------------------------------
# the scope
# ---------------------------------------------------------------------------

def test_the_scope_leaves_other_modules_stale(cell, root, toy_graph):
    """An EIS ingest derives `spectra` and `cell_model`, and leaves a stale
    `pulses` alone.

    Both halves are asserted: that `pulses` was NOT run, and that it WAS stale
    and a later full pass clears it.
    """
    write_hppc(cell, root)
    eis = write_eis(cell, root)
    assert _modules(list_stale(cell, root)) == {"spectra", "pulses", "cell_model"}, \
        "the fixture no longer builds the case"

    res = derive_for(cell, "eis", experiment_id=eis, root=root)
    assert {d["module"] for d in res["done"]} == {"spectra", "cell_model"}
    assert res["failed"] == [] and res["pending"] == []
    assert _modules(list_stale(cell, root)) == {"pulses"}, \
        "the eis-scoped pass ran a module the eis type does not feed"

    assert [d["module"] for d in rederive(cell, root)] == ["pulses"]
    assert list_stale(cell, root) == []


def test_experiment_id_none_covers_every_experiment_of_the_type(cell, root, toy_graph):
    """What a multi-file ingest wants: the writer's ids are not to hand, and
    scoping to one of them would leave the others underived."""
    write_hppc(cell, root)
    a, b = write_eis(cell, root), write_eis(cell, root)

    res = derive_for(cell, "eis", root=root)
    # one `spectra` row per experiment, plus ONE `cell_model` row for the cell
    assert res["wanted"] == 3
    spectra = [d for d in res["done"] if d["module"] == "spectra"]
    assert sorted(d["experiment_ids"][0] for d in spectra) == sorted([a, b])


def test_an_experiment_id_scopes_to_that_experiment_only(cell, root, toy_graph):
    """The other half of the same rule: a per-experiment module runs for the
    file that was just written and NOT for its neighbour."""
    hppc = write_hppc(cell, root)
    a, b = write_eis(cell, root), write_eis(cell, root)

    res = derive_for(cell, "eis", experiment_id=a, root=root)
    spectra = [d for d in res["done"] if d["module"] == "spectra"]
    assert [d["experiment_ids"] for d in spectra] == [[a]]
    # a cell-level module always runs over the full id list
    (model,) = [d for d in res["done"] if d["module"] == "cell_model"]
    assert sorted(model["experiment_ids"]) == sorted([a, b, hppc])


def test_scope_keeps_a_cell_level_row_with_no_new_ids():
    """The empty-id row is the case worth stating: it means the module went
    stale because a consumed ARTIFACT moved, which is what happens partway
    through a scoped pass (an upstream re-runs for the new file). Dropping it
    would leave the consumer stale right after the ingest that should have
    refreshed it."""
    accept = scope_for(("spectra", "cell_model"), "eis_1")
    assert accept({"module": "cell_model", "experiment_id": None, "experiment_ids": []})
    assert accept({"module": "cell_model", "experiment_id": None, "experiment_ids": ["eis_2"]})
    # a module the type does not feed is never accepted
    assert not accept({"module": "pulses", "experiment_id": "hppc_1"})
    # a per-experiment row is accepted only for the experiment in hand
    assert accept({"module": "spectra", "experiment_id": "eis_1"})
    assert not accept({"module": "spectra", "experiment_id": "eis_2"})


def test_nothing_to_do_is_reported_as_such(cell, root, toy_graph):
    write_hppc(cell, root)
    write_eis(cell, root)
    rederive(cell, root)
    res = derive_for(cell, "eis", root=root)
    assert res["wanted"] == 0 and res["done"] == [] and res["pending"] == []
    assert "nothing to derive" in summary_lines(res, "eis")[0]


# ---------------------------------------------------------------------------
# a failed derivation
# ---------------------------------------------------------------------------

def _break(monkeypatch, module: str, exc: Exception):
    """Make one module's runner raise, leaving every other one alone.

    Returns the real runner lookup, so a test can put it back without undoing
    the rest of its monkeypatching.
    """
    from monocell import rederive as rd

    real = rd._runner

    def runner(name):
        if name != module:
            return real(name)

        def die(ids, out_dir, root=None):
            raise exc

        return die

    monkeypatch.setattr(rd, "_runner", runner)
    return real


def test_a_failed_derivation_leaves_the_experiment_standing(cell, root, toy_graph, monkeypatch):
    """The measurement is written BEFORE any derivation starts, so a derivation
    that fails cannot cost the ingest its data. The failure is reported rather
    than raised."""
    from monocell import rederive as rd

    write_hppc(cell, root)
    eis = write_eis(cell, root)
    real = _break(monkeypatch, "spectra", ValueError("the spectrum has no low-frequency tail"))

    res = derive_for(cell, "eis", experiment_id=eis, root=root)

    assert [f["module"] for f in res["failed"]] == ["spectra"]
    assert "no low-frequency tail" in res["failed"][0]["error"]
    assert "spectra" in res["failed"][0]["error"]  # the module is named, not just the cause
    assert res["done"] == []
    # `cell_model` reads the spectra artifact, so it could not run either. The
    # report names the cause and the consequence.
    assert res["pending"] == ["cell_model", "spectra"]

    # the experiment itself is untouched
    d = root / "experiments" / cell / eis
    assert (d / "meta.json").exists() and (d / "quality.json").exists()
    assert [r["experiment_id"] for r in list_experiments(cell, "eis", root)] == [eis]

    # and it is re-runnable: with the cause fixed the same store derives clean
    monkeypatch.setattr(rd, "_runner", real)
    rederive(cell, root)
    assert list_stale(cell, root) == []


def test_the_report_says_what_is_left_and_that_it_is_re_runnable(cell, root, toy_graph,
                                                                  monkeypatch):
    write_hppc(cell, root)
    write_eis(cell, root)
    _break(monkeypatch, "spectra", RuntimeError("fit did not converge"))
    res = derive_for(cell, "eis", root=root)
    text = "\n".join(summary_lines(res, "eis"))
    assert "FAILED" in text and "fit did not converge" in text
    assert "still stale: cell_model, spectra" in text
    assert "stored" in text  # never reads as "the ingest failed"


def test_rederive_names_the_module_that_failed(cell, root, toy_graph, monkeypatch):
    """`rederive` raises, and the exception says WHICH module. A library
    message alone leaves nothing to act on."""
    write_eis(cell, root)
    _break(monkeypatch, "spectra", ZeroDivisionError("division by zero"))
    with pytest.raises(ModuleRunError) as ei:
        rederive(cell, root)
    assert ei.value.module == "spectra"
    assert isinstance(ei.value.__cause__, ZeroDivisionError)
    assert "spectra" in str(ei.value)


def test_work_finished_before_a_failure_is_still_reported(cell, root, toy_graph, monkeypatch):
    """`rederive` stops at the first failure but does NOT undo what ran before
    it. Reporting `done == []` next to rewritten artifacts on disk would be the
    report contradicting the store."""
    write_hppc(cell, root)
    write_eis(cell, root)
    _break(monkeypatch, "cell_model", ValueError("no reference block"))
    res = derive_for(cell, "eis", root=root)
    assert [d["module"] for d in res["done"]] == ["spectra"]
    assert [f["module"] for f in res["failed"]] == ["cell_model"]
    assert (root / "artifacts" / cell / "spectra").exists()


def test_the_exception_carries_the_work_already_done(cell, root, toy_graph, monkeypatch):
    """The same guarantee for a caller that only sees the exception."""
    write_hppc(cell, root)
    write_eis(cell, root)
    _break(monkeypatch, "cell_model", ValueError("no reference block"))
    with pytest.raises(ModuleRunError) as ei:
        rederive(cell, root)
    assert ei.value.module == "cell_model"
    assert [d["module"] for d in ei.value.done] == ["pulses", "spectra"]


# ---------------------------------------------------------------------------
# the writer stays clean
# ---------------------------------------------------------------------------

def test_the_ingest_itself_never_derives(cell, root):
    """`write_experiment` is on the path of every ingest and every fixture. A
    derivation inside it would let an ingest fail because a derivation failed.
    Nothing is derived until it is asked for, afterwards."""
    eis = write_eis(cell, root)
    assert not (root / "artifacts" / cell).exists()
    assert _modules(list_stale(cell, root)) == {"parameters"}
    assert (root / "experiments" / cell / eis / "quality.json").exists()


def test_derive_for_still_raises_for_an_unregistered_cell(root):
    """A typo'd cell id is the caller's problem, and must not be swallowed
    into an empty report that reads as success."""
    with pytest.raises(FileNotFoundError):
        derive_for("nobody", "eis", root=root)


# ---------------------------------------------------------------------------
# the CLI flag, on a bundled sample file
# ---------------------------------------------------------------------------

def _ingest_args(root, *extra) -> list[str]:
    return ["ingest", "--cell", CELL, "--type", "hppc",
            "--file", str(EXAMPLES / "cycler" / "hppc_soc_sweep.csv"),
            "--sidecar", str(EXAMPLES / "cycler" / "hppc_soc_sweep.sidecar.json"),
            "--data-root", str(root), *extra]


def test_cli_ingest_derive_writes_the_parameter_file(cell, root, capsys):
    assert cli_main(_ingest_args(root, "--derive")) == 0
    out = capsys.readouterr().out
    assert "auto-derive" in out and "parameters" in out
    assert list_stale(cell, root) == []
    assert list((root / "artifacts" / cell / "parameters").glob("parameter_file_*.json"))


def test_cli_ingest_without_the_flag_only_writes(cell, root, capsys):
    """Off by default on the CLI: a scripted import loop must not silently
    take on a derivation's runtime."""
    assert cli_main(_ingest_args(root)) == 0
    assert "auto-derive" not in capsys.readouterr().out
    assert _modules(list_stale(cell, root)) == {"parameters"}


def test_cli_ingest_derive_exits_zero_when_the_derivation_fails(cell, root, capsys, monkeypatch):
    """A failed derivation must not look like a failed ingest to a shell
    script: the data IS in the store, and the exit code is what a script
    branches on."""
    _break(monkeypatch, "parameters", ValueError("too few pulses"))
    assert cli_main(_ingest_args(root, "--derive")) == 0
    out = capsys.readouterr().out
    assert "FAILED" in out and "too few pulses" in out
    assert len(list_experiments(cell, "hppc", root)) == 1
