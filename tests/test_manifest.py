"""DuckDB manifest: upserts, queries, content hashing, rebuild."""

from __future__ import annotations

import pytest

from monocell.schema.manifest import (
    content_hash,
    find_experiment,
    list_artifacts,
    list_experiments,
    manifest_path,
    rebuild_manifest,
    upsert_artifact,
    upsert_experiment,
)


def _exp_rec(eid, cell="c1", typ="eis"):
    return {
        "experiment_id": eid,
        "cell_id": cell,
        "exp_type": typ,
        "created_at": "2026-09-08T00:00:00Z",
        "path": f"experiments/{cell}/{eid}",
        "producer_kind": "real",
    }


def test_upsert_find_roundtrip(root):
    upsert_experiment(_exp_rec("e1"), root)
    rec = find_experiment("e1", root)
    assert rec["experiment_id"] == "e1" and rec["cell_id"] == "c1" and rec["producer_kind"] == "real"


def test_find_miss_returns_none(root):
    assert find_experiment("nope", root) is None


def test_upsert_replaces(root):
    upsert_experiment(_exp_rec("e1"), root)
    upsert_experiment(_exp_rec("e1", typ="hppc"), root)
    assert find_experiment("e1", root)["exp_type"] == "hppc"


def test_list_experiments_filters_by_cell_and_type(root):
    upsert_experiment(_exp_rec("e1", cell="c1"), root)
    upsert_experiment(_exp_rec("e2", cell="c1", typ="hppc"), root)
    upsert_experiment(_exp_rec("e3", cell="c2"), root)
    assert {r["experiment_id"] for r in list_experiments("c1", None, root)} == {"e1", "e2"}
    assert [r["experiment_id"] for r in list_experiments("c1", "hppc", root)] == ["e2"]


def test_list_artifacts_filters(root):
    for art_id, mod in [("parameters/x", "parameters"), ("simulation/y", "simulation")]:
        upsert_artifact(
            {"artifact_id": art_id, "cell_id": "c1", "module": mod, "created_at": "t", "path": art_id + ".json"},
            root,
        )
    assert [r["artifact_id"] for r in list_artifacts("c1", "simulation", root)] == ["simulation/y"]
    assert len(list_artifacts("c1", None, root)) == 2


def test_content_hash_stable_and_order_independent(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"x": 1}', encoding="utf-8")
    b.write_text('{"y": 2}', encoding="utf-8")
    assert content_hash(a, b) == content_hash(b, a)
    h_before = content_hash(a, b)
    b.write_text('{"y": 3}', encoding="utf-8")
    assert content_hash(a, b) != h_before  # any byte change flips the hash


def test_rebuild_manifest_from_disk(root):
    from monocell.schema import write_artifact, write_experiment
    from monocell.cells import register_cell
    from conftest import eis_meta, eis_series

    register_cell("c1", {"capacity_Ah": 5.0, "chemistry": "NMC"}, root)
    d = write_experiment("c1", "eis", eis_meta(), eis_series(), root=root)
    import json

    eid = json.loads((d / "meta.json").read_text(encoding="utf-8"))["experiment_id"]
    write_artifact("parameters", "parameter_file_c1_vx.json", {"x": 1}, inputs=[], params={},
                   cell_id="c1", out_dir=root / "artifacts" / "c1" / "parameters", root=root)

    manifest_path(root).unlink()
    rebuild_manifest(root)
    assert find_experiment(eid, root)["cell_id"] == "c1"
    assert [r["module"] for r in list_artifacts("c1", None, root)] == ["parameters"]


# ---------------------------------------------------------------------------
# the index, while somebody else is writing it
# ---------------------------------------------------------------------------

def test_opening_the_manifest_waits_for_a_file_another_process_is_holding(monkeypatch, root):
    """DuckDB is single-writer per file, and the index is read while another
    thread or process writes it. On Windows that surfaces as `IO Error: ...
    being used by another process` from CONNECT.

    Driven rather than raced: a real two-thread test would reproduce it
    sometimes, which is the property a regression test must not have.
    """
    import duckdb

    from monocell.schema import manifest as M

    real = duckdb.connect
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise duckdb.IOException("IO Error: being used by another process")
        return real(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", flaky)
    monkeypatch.setattr(M, "_OPEN_PAUSE_S", 0.001)

    with M._conn(root) as con:                                  # noqa: SLF001
        assert con.execute("SELECT 1").fetchone() == (1,)
    assert calls["n"] == 3, "it did not retry, or it retried the wrong number of times"


def test_a_file_that_never_frees_up_still_fails(monkeypatch, root):
    """Patience, not a hang. A genuinely stuck file has to fail while somebody
    is watching, and with the error the file gave rather than a timeout of our
    own invention."""
    import duckdb

    from monocell.schema import manifest as M

    def always(*args, **kwargs):
        raise duckdb.IOException("IO Error: being used by another process")

    monkeypatch.setattr(duckdb, "connect", always)
    monkeypatch.setattr(M, "_OPEN_PAUSE_S", 0.0)
    monkeypatch.setattr(M, "_OPEN_ATTEMPTS", 3)

    # Entered, not just called: `_conn` is a context manager now, so the open
    # it is patient about happens on `__enter__`. Every caller in the platform
    # uses `with`, which is why the change is invisible outside this test.
    with pytest.raises(duckdb.IOException, match="another process"):
        with M._conn(root):                                      # noqa: SLF001
            pass


# ---------------------------------------------------------------------------
# the derived-only flag cache
# ---------------------------------------------------------------------------


from conftest import BUILD  # noqa: E402
from monocell.cells import register_cell  # noqa: E402


def _flagged_experiment(root, cell, *, rate):
    """One RPT whose ingest rule fires (or does not), by its RPT rate.

    `rpt_rate_C20_min` flags anything slower than C/20, which is the cheapest
    real flag in the platform — real, so the cache is holding something a
    reader acts on rather than a string this test invented.
    """
    import numpy as np
    import pandas as pd

    from monocell.schema import write_experiment

    n = 12
    charging = np.arange(n) < n // 2
    series = pd.DataFrame({
        "t_s": np.arange(n, dtype=float),
        "I_A": np.where(charging, 0.5, -0.5),
        "V_V": np.linspace(3.0, 4.2, n),
        "T_degC": np.full(n, 25.0),
        "q_Ah": np.tile(np.linspace(0.0, 5.0, n // 2), 2),
        "seg_type": np.where(charging, "charge", "discharge"),
        "cycle_index": np.ones(n, dtype=np.int32),
    })
    meta = {"producer": {"kind": "real", "software": "t", "version": "0"},
            "protocol": {"description": "rpt"},
            "cell_state": {"rpt_index": 0, "capacity_ref_Ah": 5.0, "age_efc": 0.0,
                           "rpt_rate": rate},
            "instrument": {"sampling_rate_Hz": 1.0}}
    return write_experiment(cell, "rpt", meta, series, root=root).name


def test_the_flag_cache_is_written_by_the_rebuild_and_by_nothing_else(root):
    """The distinction the derived table turns on.

    A table a WRITER maintains breaks "plain files are the source of truth":
    the writer becomes responsible for a second copy of the flags, and the
    first one that forgets makes the index disagree with the files with nothing
    noticing. A table only the REBUILD writes cannot drift — it is a cache of a
    scan, and deleting it loses nothing.

    So: writing an experiment must leave the table untouched, and the rebuild
    must fill it.
    """
    from monocell.schema.manifest import _conn, rebuild_manifest

    register_cell("flagcache", BUILD, root)
    _flagged_experiment(root, "flagcache", rate=0.1)      # C/10 — flagged
    _flagged_experiment(root, "flagcache", rate=0.05)     # C/20 — clean

    with _conn(root) as con:
        assert con.execute("SELECT count(*) FROM experiment_flags").fetchone()[0] == 0, (
            "a writer populated the derived table, which is the one thing it must never do"
        )

    rebuild_manifest(root)
    with _conn(root) as con:
        rows = dict(con.execute("SELECT experiment_id, n_flags FROM experiment_flags").fetchall())
    assert len(rows) == 2
    assert sorted(rows.values()) == [0, 1], (
        "the scan did not distinguish the flagged run from the clean one"
    )


def test_a_search_answers_the_same_before_and_after_a_rebuild(root):
    """The cache is a cache. An experiment ingested since the last rebuild has
    no row and falls through to reading its own meta — which is not a fast path
    made slow, it is the only correct answer for a file the scan has never
    seen.

    Reading "no row" as "no flags" would quietly clear every warning on the
    newest data in the store, which is the data anybody is actually looking at.
    """
    from monocell.schema.manifest import rebuild_manifest, search_experiments

    register_cell("flagsame", BUILD, root)
    _flagged_experiment(root, "flagsame", rate=0.1)
    _flagged_experiment(root, "flagsame", rate=0.05)

    uncached = search_experiments(root, cells=["flagsame"], flagged=True)
    rebuild_manifest(root)
    cached = search_experiments(root, cells=["flagsame"], flagged=True)

    assert len(uncached) == 1
    assert [r["experiment_id"] for r in cached] == [r["experiment_id"] for r in uncached]
    assert [r["flags"] for r in cached] == [r["flags"] for r in uncached]

    # ...and one written AFTER the rebuild still reports its flags, from its
    # own meta, with the table holding nothing about it.
    late = _flagged_experiment(root, "flagsame", rate=0.1)
    after = search_experiments(root, cells=["flagsame"], flagged=True)
    assert late in {r["experiment_id"] for r in after}


def test_only_rebuild_manifest_writes_the_derived_table():
    """Enforced across the source rather than remembered. The whole safety of
    a derived-only table is that one function owns it; a second writer would
    reintroduce exactly the drift it exists to avoid, and would do it
    invisibly, because the table would still look right on a fresh store."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    text = (repo / "monocell" / "schema" / "manifest.py").read_text(encoding="utf-8")
    # The declaration and the one INSERT, and nothing else may name it as a
    # write target.
    writes = re.findall(r"(INSERT|UPDATE|DELETE)[^\n]*experiment_flags", text)
    assert len(writes) == 1, f"experiment_flags is written from {len(writes)} places: {writes}"

    offenders = [p.relative_to(repo).as_posix()
                 for p in (repo / "monocell").rglob("*.py")
                 if p.name != "manifest.py"
                 and "experiment_flags" in p.read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} name `experiment_flags`. It is derived-only: if a writer keeps a second "
        "copy of the flags, the index is free to disagree with the files, which is the drift "
        "the manifest exists to prevent"
    )


def test_the_cache_is_what_answers_when_it_has_the_row(root):
    """A behavioural proof rather than a timing one.

    The claim is that a scanned experiment's flags come from the table and not
    from a file read — that is the whole point of the table, and a timing test
    would measure this machine rather than the code. So the meta files are
    removed after the rebuild: without the cache `_flags_of` returns `[]` on an
    unreadable meta, and with it the flags are still there.
    """
    from monocell.schema.manifest import rebuild_manifest, search_experiments

    register_cell("flagproof", BUILD, root)
    _flagged_experiment(root, "flagproof", rate=0.1)
    rebuild_manifest(root)

    for meta in (root / "experiments" / "flagproof").glob("*/meta.json"):
        meta.unlink()

    rows = search_experiments(root, cells=["flagproof"])
    assert len(rows) == 1
    assert rows[0]["flags"], "the flags came from the file read, so the cache is not consulted"
