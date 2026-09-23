"""A store whose index and files disagree, and what each reader does about it.

The manifest is a rebuildable query index over the data root; the files are
the truth. When the two part company (a directory removed by hand, a store
copied without its experiments, an ingest interrupted between `meta.json` and
`series.parquet`) every reader that trusts the index asks for a file that is
not there. Every test here damages a real store and asserts on the reader's
answer.
"""

from __future__ import annotations

import json
import shutil

import pytest

from conftest import BUILD, eis_meta, eis_series
from monocell.cells import register_cell
from monocell.schema import write_experiment
from monocell.schema.manifest import rebuild_manifest
from monocell.schema.write_experiment import ExperimentMissing


@pytest.fixture
def damaged(root):
    """A two-experiment cell, with the first experiment's directory deleted.

    The manifest still lists both. Returns `(root, cell, gone_id, kept_id)`.
    """
    register_cell("c1", BUILD, root)
    dirs = [write_experiment("c1", "eis", eis_meta(), eis_series(), root=root)
            for _ in range(2)]
    ids = [json.loads((d / "meta.json").read_text(encoding="utf-8"))["experiment_id"]
           for d in dirs]
    shutil.rmtree(dirs[0])
    return root, "c1", ids[0], ids[1]


def _two_parameter_files(root):
    """Two artifacts in one module directory, the second one corrupted."""
    from monocell.schema import write_artifact

    register_cell("c1", BUILD, root)
    out = root / "artifacts" / "c1" / "parameters"
    for name in ("parameter_file_c1_vgood.json", "parameter_file_c1_vbad.json"):
        write_artifact("parameters", name, {"x": 1}, inputs=[], params={}, cell_id="c1",
                       out_dir=out, root=root)
    bad = out / "parameter_file_c1_vbad.json"
    bad.write_text("{not json", encoding="utf-8")
    return bad


# ---------------------------------------------------------------------------
# the readers
# ---------------------------------------------------------------------------


def test_loading_a_gone_experiment_names_the_remedy(damaged):
    """The error says which experiment, and what to run. A `read_parquet` error
    says neither, and a caller cannot tell it from a bad path it was handed."""
    root, cell, gone, _ = damaged
    from monocell.schema.write_experiment import load_experiment

    with pytest.raises(ExperimentMissing) as e:
        load_experiment(gone, cell, root)
    assert gone in str(e.value) and "manifest rebuild" in str(e.value)


def test_missing_experiments_names_them(damaged):
    from monocell.rederive import missing_experiments

    root, cell, gone, _ = damaged
    assert missing_experiments(cell, root) == [gone]


def test_staleness_is_still_answerable(damaged):
    """`list_stale` hashes every input experiment. It skips the one it cannot
    hash rather than refusing the cell: the remaining data is still enough to
    say whether a module is behind."""
    from monocell.rederive import list_stale

    root, cell, _, kept = damaged
    (row,) = list_stale(cell, root)
    assert row["experiment_ids"] == [kept]


# ---------------------------------------------------------------------------
# the rebuild, which is the fix
# ---------------------------------------------------------------------------


def test_the_rebuild_drops_the_gone_row(damaged):
    from monocell.rederive import missing_experiments

    root, cell, _, _ = damaged
    rebuild_manifest(root)
    assert missing_experiments(cell, root) == []


def test_the_rebuild_reports_what_it_saw(damaged):
    """A rebuild that indexed a fifth of the store must not look like one that
    indexed all of it."""
    root, _, _, _ = damaged
    summary = rebuild_manifest(root)
    assert summary["experiment_files"] == 1 and summary["experiments_indexed"] == 1


def test_a_repeated_experiment_id_is_skipped_and_named(damaged):
    """A store copied under a second cell id repeats every experiment id. The
    scan indexes the first and reports the rest, instead of overwriting each
    row's path with the copy's and leaving no trace."""
    root, cell, _, _ = damaged
    src = root / "experiments" / cell
    shutil.copytree(src, root / "experiments" / "c1_copy")
    summary = rebuild_manifest(root)
    assert summary["experiment_files"] == 2
    assert summary["experiments_indexed"] == 1
    assert len(summary["duplicate_experiment_ids"]) == 1


def test_a_half_written_experiment_is_not_indexed(damaged):
    """An experiment is meta + series and `load_experiment` requires both.
    Indexing half of one only moves the failure to whoever reads it next."""
    root, cell, _, kept = damaged
    (root / "experiments" / cell / kept / "series.parquet").unlink()
    summary = rebuild_manifest(root)
    assert summary["experiments_indexed"] == 0
    assert kept in summary["incomplete"][0]


def test_an_unreadable_artifact_does_not_abandon_the_scan(root):
    """The rebuild is what a user runs BECAUSE the store is inconsistent, so a
    corrupt file in it must not be the one thing that stops the rebuild."""
    bad = _two_parameter_files(root)
    summary = rebuild_manifest(root)
    assert summary["artifacts_indexed"] == 1
    assert bad.name in summary["unreadable"][0]


def test_a_corrupt_artifact_sorts_oldest_rather_than_raising(root):
    """`artifact_order_key` runs while sorting a whole directory, so one file
    that will not parse would otherwise take down every reader that picks a
    newest artifact."""
    from monocell.schema.artifacts import artifact_order_key, newest_artifact

    bad = _two_parameter_files(root)
    assert artifact_order_key(bad)[0] == ""
    newest = newest_artifact("c1", "parameters", "parameter_file_*.json", root)
    assert newest.name == "parameter_file_c1_vgood.json"


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------


def test_manifest_status_names_the_break_then_agrees(damaged, capsys):
    """`status` is the diagnosis and `rebuild` is the fix, and the first has
    to be able to say the second worked."""
    from monocell.cli import main

    root, _, gone, _ = damaged
    assert main(["manifest", "status", "--data-root", str(root)]) == 1
    assert gone in capsys.readouterr().out

    assert main(["manifest", "rebuild", "--data-root", str(root)]) == 0
    capsys.readouterr()
    assert main(["manifest", "status", "--data-root", str(root)]) == 0
    assert "agree" in capsys.readouterr().out


def test_status_separates_a_gone_directory_from_a_half_written_one(damaged, capsys):
    """Two conditions with two different remedies. A rebuild drops a row whose
    directory is gone; it cannot mend a directory that is there and empty."""
    from monocell.cli import main

    root, cell, _, kept = damaged
    (root / "experiments" / cell / kept / "series.parquet").unlink()
    assert main(["manifest", "status", "--data-root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "directory is gone" in out and "without series.parquet" in out
    assert "re-ingest" in out
