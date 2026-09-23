"""The shipped module graph, re-derivation over it, and the `rederive` command.

The general machinery (run order, artifact edges, the fixed point) is tested
on a toy graph in `test_rederive_graph.py`. This file is about the graph that
ships: one module, the abstracted parameter extraction, over every measured
type.
"""

from __future__ import annotations

import pytest

from conftest import BUILD, register_cell, write_eis, write_hppc
from monocell.cli import main
from monocell.rederive import MODULE_GRAPH, list_stale, rederive
from monocell.schema import write_artifact
from monocell.schema.artifacts import record_pattern
from monocell.schema.tables import SPECS


@pytest.fixture
def cell(root):
    register_cell("test_cell", BUILD, root)
    return "test_cell"


def test_the_graph_is_one_cell_level_module_over_every_measured_type():
    """Any subset of types is derivable: a type with no experiments yet is a
    gap in the parameter file, not a reason to skip the derivation."""
    assert set(MODULE_GRAPH) == {"parameters"}
    spec = MODULE_GRAPH["parameters"]
    assert set(spec["input_experiments"]) == set(SPECS) - {"misc"}
    assert spec["mode"] == "all_experiments"
    assert spec["require_all_types"] is False


def test_the_graph_and_the_store_layout_agree_on_the_record_file():
    """`schema.artifacts.MODULE_RECORDS` says which file a consumer reads;
    the graph says which files staleness is computed over. Two declarations
    of one fact, so they are checked against each other."""
    assert record_pattern("parameters") == MODULE_GRAPH["parameters"]["artifact_glob"]


def test_one_experiment_makes_the_parameter_file_stale(cell, root):
    eis = write_eis(cell, root)
    (row,) = list_stale(cell, root)
    assert row["module"] == "parameters"
    assert row["experiment_id"] is None and row["experiment_ids"] == [eis]


def test_rederive_clears_stale_and_writes_the_parameter_file(cell, root):
    write_eis(cell, root)
    write_hppc(cell, root)
    done = rederive(cell, root)
    assert [d["module"] for d in done] == ["parameters"]
    assert list((root / "artifacts" / cell / "parameters").glob("parameter_file_*.json"))
    assert list_stale(cell, root) == []


def test_dry_run_writes_nothing(cell, root):
    write_eis(cell, root)
    done = rederive(cell, root, dry_run=True)
    assert done[0]["dry_run"] is True
    assert not (root / "artifacts" / cell / "parameters").exists()
    assert len(list_stale(cell, root)) == 1


def test_an_edited_experiment_is_detected(cell, root):
    eis = write_eis(cell, root)
    rederive(cell, root)
    assert list_stale(cell, root) == []

    meta = root / "experiments" / cell / eis / "meta.json"
    meta.write_text(meta.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    (row,) = list_stale(cell, root)
    assert row["experiment_ids"] == [eis]

    rederive(cell, root)
    assert list_stale(cell, root) == []


def test_an_artifact_with_no_experiments_behind_it_is_not_flagged(cell, root):
    """A documented limit: staleness is computed over the cell's current
    experiments, so a cell with none has nothing to be stale about."""
    write_artifact("parameters", "parameter_file_test_cell_vghost.json", {}, [], {},
                   cell_id=cell, out_dir=root / "artifacts" / cell / "parameters", root=root)
    assert list_stale(cell, root) == []


def test_an_unregistered_cell_is_an_error_not_an_empty_answer(root):
    with pytest.raises(FileNotFoundError):
        list_stale("ghost", root)


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------

def test_cli_rederive_reports_and_runs(cell, root, capsys):
    eis = write_eis(cell, root)
    assert main(["rederive", "--cell", cell, "--data-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert f"cell {cell}: 1 stale artifact(s)" in out
    assert eis in out
    assert "re-derived parameters" in out
    assert list_stale(cell, root) == []


def test_cli_nothing_stale_message(cell, root, capsys):
    write_eis(cell, root)
    rederive(cell, root)
    assert main(["rederive", "--cell", cell, "--data-root", str(root)]) == 0
    assert "nothing stale" in capsys.readouterr().out


def test_cli_dry_run(cell, root, capsys):
    write_eis(cell, root)
    assert main(["rederive", "--cell", cell, "--data-root", str(root), "--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert len(list_stale(cell, root)) == 1


@pytest.mark.parametrize("command", ["rederive", "simulate"])
def test_cli_an_unregistered_cell_is_reported_in_one_line(root, capsys, command):
    """A typo'd cell is user input, not a bug: say so and exit non-zero."""
    assert main([command, "--cell", "ghost", "--data-root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "ghost is not registered" in out and "Traceback" not in out


def test_cli_requires_a_subcommand_and_rederive_requires_a_cell():
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["rederive"])
