"""Searching the manifest: the person's question rather than the program's.

`list_experiments(cell, type)` answers "what does this cell have", never "where
is that run". `search_experiments` is the second question, and it is the one an
engineer with a long campaign behind them actually has.

The design decision worth testing is the SPLIT. What the index holds is filtered
in SQL; the quality flags are not in the index and are a second pass over the
survivors' own `meta.json`. Putting them in the index would make every writer
responsible for a second copy of them, which is the drift the manifest's design
avoids, so the two-pass shape is deliberate and the tests below pin what each
pass is responsible for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import BUILD, register_cell
from monocell.cli import main as cli_main
from monocell.schema import write_experiment
from monocell.schema.manifest import search_experiments


def _series(n: int = 12) -> pd.DataFrame:
    charging = np.arange(n) < n // 2
    return pd.DataFrame({
        "t_s": np.arange(n, dtype=float),
        "I_A": np.where(charging, 1e-3, -1e-3),
        "V_V": np.linspace(3.0, 4.2, n),
        "T_degC": np.full(n, 25.0),
        "q_Ah": np.linspace(0.0, 1e-3, n),
        "seg_type": np.where(charging, "charge", "discharge"),
        "cycle_index": np.ones(n, dtype=np.int32),
    })


def _meta(created_at: str, **cell_state) -> dict:
    return {
        "producer": {"kind": "real", "software": "test", "version": "0"},
        "protocol": {"description": "a run"},
        "cell_state": {"rpt_index": 0, **cell_state},
        "instrument": {"sampling_rate_Hz": 1.0},
        "created_at": created_at,
    }


@pytest.fixture
def store(root):
    """Two cells in a lot, a third outside it, and runs across two months.

    One of them has no `capacity_ref_Ah`, so ingest flags it — a real flag from
    a real rule, rather than a string poked into the file to give the filter
    something to find.
    """
    from monocell.batches import add_cell_to_batch, save_batch

    for cell in ("cell_a", "cell_b", "cell_out"):
        register_cell(cell, BUILD, root)
    save_batch("lot_1", {"capacity_Ah": 5.0, "chemistry": "NMC811/graphite",
                         "layer_count": 12}, root)
    add_cell_to_batch("lot_1", "cell_a", root, serial="S-1")
    add_cell_to_batch("lot_1", "cell_b", root, serial="S-2")

    write_experiment("cell_a", "cycling", _meta("2026-05-04T09:00:00+00:00",
                                                capacity_ref_Ah=1e-3), _series(), root=root)
    write_experiment("cell_b", "cycling", _meta("2026-06-11T09:00:00+00:00",
                                                capacity_ref_Ah=1e-3), _series(), root=root)
    write_experiment("cell_b", "rpt", _meta("2026-06-12T09:00:00+00:00",
                                            capacity_ref_Ah=1e-3, rpt_rate=0.05),
                     _series(), root=root)
    # no capacity reference: `cycling_soh` says so, and that is the flag below
    write_experiment("cell_out", "cycling", _meta("2026-06-13T09:00:00+00:00"),
                     _series(), root=root)
    return root


# ---------------------------------------------------------------------------
# what SQL answers
# ---------------------------------------------------------------------------

def test_the_newest_answer_comes_first(store):
    """An engineer searching is almost always looking for something recent, and
    a result set in insertion order makes them read to the bottom to find out."""
    rows = search_experiments(store)
    assert len(rows) == 4
    assert [r["created_at"] for r in rows] == sorted(
        (r["created_at"] for r in rows), reverse=True)


def test_a_date_bound_may_be_a_prefix_and_until_means_the_end_of_its_day(store):
    """`created_at` is an ISO string, so lexicographic comparison IS
    chronological comparison and `2026-06` is a legal bound. The `until` half
    is the one that bites: a bare date compared as-is means midnight, so
    `--until 2026-06-11` would exclude everything that happened ON the 11th —
    a filter nobody wants and everybody writes by accident.
    """
    assert len(search_experiments(store, since="2026-06")) == 3

    # `until 2026-05` means through the END of May, not before it — the same
    # rule one level up: a partial bound names a period, and the period is
    # included.
    assert len(search_experiments(store, until="2026-05")) == 1
    assert len(search_experiments(store, until="2026-04")) == 0

    on_the_day = search_experiments(store, until="2026-06-11")
    assert len(on_the_day) == 2, "a bare `until` date dropped the runs made that day"


def test_a_lot_searches_its_member_cells_and_a_cell_outside_it_is_not_found(store):
    """The join a lot filter needs is over JSON batch files, done in Python on
    purpose: a glob is the right index for tens of them, and a `cells` table
    would need every writer to upsert it."""
    rows = search_experiments(store, lot="lot_1")
    assert {r["cell_id"] for r in rows} == {"cell_a", "cell_b"}
    assert "cell_out" not in {r["cell_id"] for r in rows}

    # ...and a lot that does not exist finds nothing rather than everything,
    # which is the failure direction that matters: an ignored filter silently
    # widens a search
    assert search_experiments(store, lot="no_such_lot") == []


def test_filters_combine_rather_than_replace(store):
    """Each filter narrows. Worth its own test because the natural way to get
    this wrong — building the WHERE clause from the last argument set — passes
    every single-filter test there is."""
    assert len(search_experiments(store, lot="lot_1", exp_types=["cycling"])) == 2
    assert len(search_experiments(store, lot="lot_1", exp_types=["cycling"],
                                  since="2026-06")) == 1
    assert search_experiments(store, cells=["cell_out"], lot="lot_1") == [], \
        "a cell outside the lot survived both filters"


# ---------------------------------------------------------------------------
# what the second pass answers
# ---------------------------------------------------------------------------

def test_the_flag_filter_reads_the_experiments_own_meta(store):
    """Quality is not in the index, and this is the deliberate consequence: the
    flag filter is a second pass over files. Both directions, because either
    alone passes on a filter that is simply ignored."""
    flagged = search_experiments(store, flagged=True)
    clean = search_experiments(store, flagged=False)

    assert {r["cell_id"] for r in flagged} == {"cell_out"}
    assert any("capacity_ref_Ah" in f for f in flagged[0]["flags"])
    assert len(clean) == 3 and all(not r["flags"] for r in clean)
    assert len(flagged) + len(clean) == len(search_experiments(store))


def test_a_row_whose_files_are_gone_reads_as_unflagged_rather_than_raising(store):
    """A manifest row the disk does not have is a broken store, and the result
    of a search is the wrong place to find that out. `rebuild_manifest` is the
    documented fix; a search says what it can and keeps going."""
    import shutil

    row = search_experiments(store, cells=["cell_a"])[0]
    shutil.rmtree(row["path"])

    still = search_experiments(store, cells=["cell_a"])
    assert len(still) == 1 and still[0]["flags"] == []


# ---------------------------------------------------------------------------
# through the CLI
# ---------------------------------------------------------------------------

def test_cli_find_prints_a_line_per_match_and_says_when_there_are_none(store, capsys):
    """An empty result is an answer, so it exits 0 and says so in words. A
    search that exited non-zero on "no matches" would be unusable in a script
    and alarming in a terminal."""
    assert cli_main(["find", "--lot", "lot_1", "--type", "cycling",
                     "--data-root", str(store)]) == 0
    out = capsys.readouterr().out
    assert out.count("cycling_") == 2
    assert "2 experiment(s)" in out

    assert cli_main(["find", "--cell", "cell_a", "--since", "2027-01",
                     "--data-root", str(store)]) == 0
    assert "nothing matches" in capsys.readouterr().out


def test_cli_find_shows_the_flags_themselves_when_asked(store, capsys):
    """The count answers "is anything wrong here"; the text answers "what".
    Both are wanted at different moments and the default is the quiet one,
    because a flagged campaign would otherwise scroll off the screen."""
    cli_main(["find", "--flagged", "--data-root", str(store)])
    quiet = capsys.readouterr().out
    assert "1 flag(s)" in quiet and "capacity_ref_Ah" not in quiet

    cli_main(["find", "--flagged", "--show-flags", "--data-root", str(store)])
    assert "capacity_ref_Ah" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# what has happened here lately
# ---------------------------------------------------------------------------

def test_recent_activity_mixes_experiments_and_artifacts_in_one_order(store):
    """A file landing and a derivation being re-run are the same kind of news
    to somebody who has been away for a week, so they are ordered together.

    Unioned in SQL rather than merged from two ordered lists in Python, which
    is the version that quietly drops one table's rows whenever the other has
    more than `limit` of them — the bug this test exists to keep out.
    """
    from monocell.schema.artifacts import write_artifact
    from monocell.schema.manifest import recent_activity

    # An artifact written BETWEEN the experiment timestamps, so a merge that
    # concatenated the two tables rather than ordering them would show it in
    # the wrong place.
    write_artifact("parameters", "parameter_file_cell_a_v1.json", {"cell_id": "cell_a"}, [],
                   {}, "cell_a", store / "artifacts" / "cell_a" / "parameters", store)

    rows = recent_activity(store, limit=20)
    assert {r["kind"] for r in rows} == {"experiment", "artifact"}
    assert [r["created_at"] for r in rows] == sorted(
        (r["created_at"] for r in rows), reverse=True)


def test_recent_activity_respects_the_cells_it_is_given(store):
    """A caller scoped to a lot needs its activity scoped too: a feed of the
    whole store would answer about cells the caller did not ask about."""
    from monocell.schema.manifest import recent_activity

    rows = recent_activity(store, cells=["cell_b"])
    assert rows and {r["cell_id"] for r in rows} == {"cell_b"}

    # an empty cell list is "no cells", not "every cell" — the failure
    # direction that matters, since a dropped filter silently widens
    assert recent_activity(store, cells=[]) == []


def test_recent_activity_stops_at_its_limit(store):
    """A feed is a feed. The number is the caller's, and the query does the
    cutting — slicing in Python would read every row of both tables on a store
    that has years of them."""
    from monocell.schema.manifest import recent_activity

    assert len(recent_activity(store, limit=2)) == 2
