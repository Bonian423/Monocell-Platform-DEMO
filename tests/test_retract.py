"""Withdrawing an experiment without destroying it.

The store is append-only. An experiment that was wrongly ingested (the sidecar
named the wrong cell, the rig was mis-set, the file was somebody else's) is
still a record of what happened, and deleting it would destroy the evidence
that a correction was needed. Without a retraction, the only correction
available would be a NEW experiment with `derived_from`, which does not remove
the old one from a derivation that pools every experiment of a cell.

A retraction is a tombstone beside `meta.json`. The files stay; what changes is
what they count towards. The tests split three ways: the tombstone itself, what
stops seeing the experiment, and what goes stale because of it.
"""

from __future__ import annotations

import json

import pytest

from conftest import BUILD, eis_meta, eis_series, write_eis, write_hppc
from monocell.cells import register_cell
from monocell.rederive import list_stale, rederive
from monocell.schema import write_experiment
from monocell.schema.artifacts import load_artifact, newest_artifact
from monocell.schema.manifest import list_experiments
from monocell.schema.write_experiment import (
    ExperimentMissing,
    RETRACTED_FILENAME,
    retract,
    retraction_of,
    unretract,
)


@pytest.fixture
def stocked(root):
    register_cell("c1", BUILD, root)
    ids = []
    for _ in range(3):
        d = write_experiment("c1", "eis", eis_meta(), eis_series(), root=root)
        ids.append(json.loads((d / "meta.json").read_text(encoding="utf-8"))["experiment_id"])
    return root, ids


@pytest.fixture
def derived(root):
    """`(root, eis_ids, hppc_id)`: a settled store whose parameter file cites
    three EIS experiments and one HPPC."""
    register_cell("c1", BUILD, root)
    eis = [write_eis("c1", root) for _ in range(3)]
    hppc = write_hppc("c1", root)
    rederive("c1", root)
    assert list_stale("c1", root) == []
    return root, eis, hppc


def _cited(root) -> set[str]:
    path = newest_artifact("c1", "parameters", "parameter_file_*.json", root)
    return {r["id"] for r in load_artifact(path)["inputs"]}


# ---------------------------------------------------------------------------
# the tombstone
# ---------------------------------------------------------------------------


def test_the_data_stays_on_disk(stocked):
    """The whole design in one assertion. A retraction that deleted the series
    would leave the store unable to show why the correction was needed."""
    root, ids = stocked
    d = root / "experiments" / "c1" / ids[0]
    retract(ids[0], "c1", "the sidecar named the wrong cell", root)
    assert (d / "meta.json").exists() and (d / "series.parquet").exists()
    assert (d / RETRACTED_FILENAME).exists()


def test_a_reason_is_required(stocked):
    """A retraction with no reason cannot be told apart from a mistake in the
    retraction."""
    root, ids = stocked
    for empty in ("", "   ", None):
        with pytest.raises(ValueError, match="reason"):
            retract(ids[0], "c1", empty, root)
    assert retraction_of("c1", ids[0], root) is None


def test_the_reason_and_who_are_both_kept(stocked):
    root, ids = stocked
    retract(ids[0], "c1", "the rig was mis-set", root, by="cli")
    rec = retraction_of("c1", ids[0], root)
    assert rec["reason"] == "the rig was mis-set"
    assert rec["by"] == "cli" and rec["at"]


def test_retracting_an_experiment_that_is_not_there_says_which(stocked):
    root, _ = stocked
    with pytest.raises(ExperimentMissing):
        retract("nope", "c1", "because", root)


def test_a_second_retraction_corrects_the_reason(stocked):
    """Not an error: the second call is somebody fixing what the first said."""
    root, ids = stocked
    retract(ids[0], "c1", "first reason", root)
    retract(ids[0], "c1", "the actual reason", root)
    assert retraction_of("c1", ids[0], root)["reason"] == "the actual reason"


def test_an_unreadable_tombstone_still_retracts(stocked):
    """The tombstone says "do not use this". Failing to read the reason must
    not turn into treating the experiment as live."""
    root, ids = stocked
    retract(ids[0], "c1", "whatever", root)
    (root / "experiments" / "c1" / ids[0] / RETRACTED_FILENAME).write_text(
        "{not json", encoding="utf-8")
    assert retraction_of("c1", ids[0], root) is not None
    assert ids[0] not in {r["experiment_id"] for r in list_experiments("c1", None, root)}


def test_a_retraction_can_be_undone(stocked):
    """A retraction is a judgement, and judgements are revised."""
    root, ids = stocked
    retract(ids[0], "c1", "on reflection, no", root)
    assert unretract(ids[0], "c1", root) is True
    assert retraction_of("c1", ids[0], root) is None
    assert ids[0] in {r["experiment_id"] for r in list_experiments("c1", None, root)}
    assert unretract(ids[0], "c1", root) is False


# ---------------------------------------------------------------------------
# what stops seeing it
# ---------------------------------------------------------------------------


def test_the_listing_leaves_it_out(stocked):
    """`list_experiments` is the single point every reader goes through, which
    is what keeps a withdrawn run out of every derivation without each caller
    having to remember."""
    root, ids = stocked
    retract(ids[0], "c1", "duplicate ingest", root)
    live = {r["experiment_id"] for r in list_experiments("c1", None, root)}
    assert ids[0] not in live and len(live) == 2


def test_the_listing_can_be_asked_for_it(stocked):
    root, ids = stocked
    retract(ids[0], "c1", "duplicate ingest", root)
    every = {r["experiment_id"] for r in list_experiments("c1", None, root,
                                                          include_retracted=True)}
    assert every == set(ids)


def test_it_is_read_from_the_file_and_not_the_index(stocked):
    """A derived index table would be wrong here: an experiment retracted since
    the last rebuild would go on feeding every derivation until somebody
    re-indexed."""
    root, ids = stocked
    retract(ids[0], "c1", "duplicate ingest", root)
    # No rebuild between the retraction and the read. That is the point.
    assert ids[0] not in {r["experiment_id"] for r in list_experiments("c1", None, root)}


# ---------------------------------------------------------------------------
# what goes stale
# ---------------------------------------------------------------------------


def test_a_derivation_that_rests_on_it_goes_stale(derived):
    """The claim the feature turns on. Comparing only forwards (walking the
    live experiments and checking each has a current artifact) would let a
    parameter file still resting on a withdrawn experiment read as fresh,
    because a withdrawn experiment is not among the live ones."""
    root, eis, _ = derived
    retract(eis[0], "c1", "the sidecar named the wrong cell", root)

    (row,) = list_stale("c1", root)
    assert row["module"] == "parameters"
    assert eis[0] in row["reason"] and "no longer current" in row["reason"]


def test_re_deriving_drops_it_and_the_store_settles(derived):
    """Two claims in one run, and the second makes the first usable: the new
    parameter file does not cite it, AND nothing is stale afterwards. A check
    that could not be satisfied would make `rederive` refuse the store on its
    own fixed-point guard."""
    root, eis, hppc = derived
    retract(eis[0], "c1", "the sidecar named the wrong cell", root)
    rederive("c1", root)

    assert _cited(root) == {eis[1], eis[2], hppc}
    assert not list_stale("c1", root)


def test_putting_it_back_settles_the_store_again(derived):
    root, eis, hppc = derived
    retract(eis[0], "c1", "on reflection, no", root)
    rederive("c1", root)
    unretract(eis[0], "c1", root)
    assert list_stale("c1", root), "the experiment came back and nothing noticed"
    rederive("c1", root)
    assert not list_stale("c1", root)
    assert _cited(root) == {*eis, hppc}


def test_the_dropped_input_rule_ignores_artifact_refs_and_unknown_ids(derived):
    """The rule behind "cited but no longer current", condition by condition.

    * an id the manifest has never heard of is a typo in an old artifact, and
      not a reason to hold a module stale for the life of the store;
    * an experiment of a type the module does not declare is some other
      module's input, and staleness on that route is the artifact edge's job;
    * an `artifact:` id is not an experiment at all.
    """
    from monocell.rederive import _dropped_inputs

    root, eis, hppc = derived
    target = eis[0]
    cited = {"artifact:spectra", target, hppc, "rpt_never_ingested"}

    assert _dropped_inputs(("eis",), "c1", root, cited, {}) == [target]
    assert _dropped_inputs(("eis", "hppc"), "c1", root, cited, {}) == sorted([target, hppc])
    assert _dropped_inputs(("pressure",), "c1", root, cited, {}) == []
    # ...and nothing that IS current is reported, whatever its type
    assert _dropped_inputs(("eis",), "c1", root, cited, {target: "hash"}) == []
