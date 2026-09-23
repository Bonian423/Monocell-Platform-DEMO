"""Holding one index connection for an operation, and what may not change.

Opening the index file is the cost. `duckdb.connect` plus the `CREATE TABLE IF
NOT EXISTS` block measures about 11 ms on Windows and the queries it carries are
microseconds, so a reader that asks the manifest many small questions pays the
open once per question, and that dominates its run time.

A session is a per-thread, per-file connection held for one logical operation.
The tests below are in two halves. The first is that it is actually shared. The
second, and the one that matters, is that sharing changes no answer: the same
query returns the same thing inside a session and outside one, a write made in a
session is visible to a reader outside it, and nothing is held between
operations — DuckDB allows one read-write process per file, so a connection kept
past its operation would lock every CLI command out of the store.
"""

from __future__ import annotations

import json
import threading

import pytest

from monocell.cells import register_cell
from monocell.schema import write_experiment
from monocell.schema.manifest import (
    _SESSIONS,
    find_experiment,
    list_experiments,
    rebuild_manifest,
    session,
)


@pytest.fixture
def stocked(root):
    """A cell with three experiments, indexed."""
    from conftest import eis_meta, eis_series

    register_cell("c1", {"capacity_Ah": 5.0, "chemistry": "NMC"}, root)
    ids = []
    for _ in range(3):
        d = write_experiment("c1", "eis", eis_meta(), eis_series(), root=root)
        ids.append(json.loads((d / "meta.json").read_text(encoding="utf-8"))["experiment_id"])
    return root, ids


# ---------------------------------------------------------------------------
# it is shared
# ---------------------------------------------------------------------------


def test_one_session_is_one_connection(root, monkeypatch):
    from monocell.schema import manifest as M

    opens = {"n": 0}
    real = M._open  # noqa: SLF001

    def counted(*a, **k):
        opens["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(M, "_open", counted)
    with session(root):
        for _ in range(8):
            list_experiments("c1", None, root)
    assert opens["n"] == 1, f"a session opened the file {opens['n']} times"


def test_without_a_session_every_query_opens_its_own(root, monkeypatch):
    """The negative half. Without this the test above passes on a build where
    `_open` is memoised globally, which is the thing that must not happen."""
    from monocell.schema import manifest as M

    opens = {"n": 0}
    real = M._open  # noqa: SLF001

    def counted(*a, **k):
        opens["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(M, "_open", counted)
    for _ in range(8):
        list_experiments("c1", None, root)
    assert opens["n"] == 8


def test_a_nested_session_joins_the_outer_one(root, monkeypatch):
    """An operation that holds a session calls another that holds one too. The
    inner must not open a second connection, and must not close the outer one
    when it returns."""
    from monocell.schema import manifest as M

    opens = {"n": 0}
    real = M._open  # noqa: SLF001
    monkeypatch.setattr(M, "_open", lambda *a, **k: (opens.__setitem__("n", opens["n"] + 1),
                                                     real(*a, **k))[1])
    with session(root):
        with session(root):
            list_experiments("c1", None, root)
        # The outer session is still usable after the inner one exits.
        list_experiments("c1", None, root)
    assert opens["n"] == 1


def test_two_threads_do_not_share_a_connection(root):
    """A DuckDB connection is not safe to use from two threads at once."""
    seen: list[int] = []

    def work():
        with session(root) as con:
            seen.append(id(con))

    with session(root) as mine:
        t = threading.Thread(target=work)
        t.start()
        t.join()
        assert seen and seen[0] != id(mine)


def test_nothing_is_held_after_the_operation(root):
    with session(root):
        list_experiments("c1", None, root)
    assert not _SESSIONS, "a connection outlived its session"


# ---------------------------------------------------------------------------
# it changes no answer
# ---------------------------------------------------------------------------


def test_the_same_query_answers_the_same_in_and_out_of_a_session(stocked):
    root, ids = stocked
    outside = list_experiments("c1", None, root)
    with session(root):
        inside = list_experiments("c1", None, root)
    assert [r["experiment_id"] for r in outside] == [r["experiment_id"] for r in inside]
    assert sorted(r["experiment_id"] for r in inside) == sorted(ids)


def test_a_write_inside_a_session_is_visible_outside_it(stocked):
    """Sessions autocommit, like the connection-per-statement they replaced.
    A write that only landed when the session closed would make an ingest
    invisible to a reader that is waiting for it."""
    from conftest import eis_meta, eis_series

    root, ids = stocked
    with session(root):
        d = write_experiment("c1", "eis", eis_meta(), eis_series(), root=root)
        new = json.loads((d / "meta.json").read_text(encoding="utf-8"))["experiment_id"]
        assert find_experiment(new, root) is not None, "not visible within the session"
    assert find_experiment(new, root) is not None, "not visible after it"


def test_a_rebuild_replaces_the_index_rather_than_adding_to_it(stocked):
    """The rebuild drops and recreates the tables rather than deleting the
    file, because a session may be holding the file open and on Windows an open
    file cannot be unlinked, so the thing to check is that replacing still
    means replacing."""
    import shutil

    root, ids = stocked
    gone = root / "experiments" / "c1" / ids[0]
    shutil.rmtree(gone)
    rebuild_manifest(root)
    assert find_experiment(ids[0], root) is None, "the dropped row survived the rebuild"
    assert len(list_experiments("c1", None, root)) == 2


def test_a_rebuild_inside_a_session_still_works(stocked):
    """A rebuild can run inside whatever session the caller had open."""
    root, ids = stocked
    with session(root):
        summary = rebuild_manifest(root)
    assert summary["experiments_indexed"] == 3


def test_a_repeated_artifact_id_is_reported(root):
    """`artifact_id` is the artifacts table's primary key, so a second file
    claiming one would silently replace the first, and a store could lose most
    of its index without a word."""
    from monocell.schema import write_artifact

    register_cell("c1", {"capacity_Ah": 5.0, "chemistry": "NMC"}, root)
    out = root / "artifacts" / "c1" / "parameters"
    write_artifact("parameters", "parameter_file_c1_one.json", {"x": 1}, inputs=[], params={},
                   cell_id="c1", out_dir=out, root=root)
    twin = json.loads((out / "parameter_file_c1_one.json").read_text(encoding="utf-8"))
    (out / "parameter_file_c1_two.json").write_text(json.dumps(twin), encoding="utf-8")

    summary = rebuild_manifest(root)
    assert summary["artifact_files"] == 2
    assert summary["artifacts_indexed"] == 1
    assert len(summary["duplicate_artifact_ids"]) == 1


# ---------------------------------------------------------------------------
# the hash memo
# ---------------------------------------------------------------------------


def test_the_hash_memo_follows_the_bytes_not_the_call(stocked):
    """Keyed on every part's (path, mtime, size). A cache keyed on the id would
    hold a stale hash after a re-ingest, and this hash is what makes a derived
    result stale, so it would silently freeze the store in its current state."""
    import time

    from monocell.schema.write_experiment import experiment_hash

    root, ids = stocked
    before = experiment_hash("c1", ids[0], root)
    assert experiment_hash("c1", ids[0], root) == before

    meta = root / "experiments" / "c1" / ids[0] / "meta.json"
    time.sleep(0.01)
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["notes"] = "edited by hand"
    meta.write_text(json.dumps(doc), encoding="utf-8")

    assert experiment_hash("c1", ids[0], root) != before, "the memo served a stale hash"


def test_a_rewritten_series_moves_the_hash(stocked):
    """The memo keys on every part, not only on `meta.json`: a series rewritten
    under the same name is a different experiment as far as staleness goes."""
    import time

    import pandas as pd

    from monocell.schema.write_experiment import experiment_hash

    root, ids = stocked
    before = experiment_hash("c1", ids[0], root)
    series = root / "experiments" / "c1" / ids[0] / "series.parquet"
    frame = pd.read_parquet(series)
    frame["T_degC"] = 26.0
    time.sleep(0.01)
    frame.to_parquet(series, index=False)
    assert experiment_hash("c1", ids[0], root) != before


def test_reading_a_header_does_not_read_the_series(stocked):
    """`load_meta` is the header alone. Listing a store needs only the headers,
    and `load_experiment` reads the parquet unconditionally, so a listing built
    on it would parse every series only to discard it."""
    from monocell.schema.write_experiment import ExperimentMissing, load_meta

    root, ids = stocked
    # Patched on `pandas` itself, because the reader holds the module rather
    # than the function: `import pandas as pd` binds the same object, so this
    # is the same patch from the reader's side.
    import pandas

    reads = {"n": 0}
    real = pandas.read_parquet

    def counted(*a, **k):
        reads["n"] += 1
        return real(*a, **k)

    pandas.read_parquet = counted
    try:
        metas = [load_meta(eid, "c1", root) for eid in ids]
    finally:
        pandas.read_parquet = real
    assert [m["experiment_id"] for m in metas] == ids
    assert reads["n"] == 0, f"reading headers read {reads['n']} series"

    # ...and an experiment without its series is still not a readable one,
    # header or not, so a listing cannot disagree with every other reader
    (root / "experiments" / "c1" / ids[0] / "series.parquet").unlink()
    with pytest.raises(ExperimentMissing):
        load_meta(ids[0], "c1", root)
