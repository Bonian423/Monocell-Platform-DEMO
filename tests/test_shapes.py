"""Type proposals from a file's shape — and the discipline of not deciding.

The load-bearing test is `test_every_sample_file_proposes_its_own_type_first`.
The bundled sample campaign holds a file of each common type, written by a
simulator that has never heard of this module, so it is a real round trip
rather than a fixture written to match the rules: if a rule here encodes the
wrong idea of what an HPPC looks like, the HPPC file says so.

Everything else in this file is about what the module must REFUSE to do —
propose a type for a table that is not one, invent a C-rate when no capacity
was given, or pick a side when its two pieces of evidence disagree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from conftest import BUILD, EXAMPLE_FILES, EXAMPLES
from monocell.schema.shapes import TypeProposal, propose_type

CAP_AH = BUILD["capacity_Ah"]


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------

def test_every_sample_file_proposes_its_own_type_first():
    """A file of type X must propose X, for every sample file there is.

    Each file is read by the reader its type would use, and the SHAPE of what
    came back is then asked what it is. The loop collects every mismatch
    instead of stopping at the first.
    """
    from monocell.schema.ingest import read_for_ingest

    wrong = []
    for stem, exp_type, instrument in EXAMPLE_FILES:
        frame, _ = read_for_ingest(exp_type, EXAMPLES / f"{stem}.csv", instrument, None)
        proposals = propose_type(frame, capacity_Ah=CAP_AH)
        if not proposals:
            wrong.append(f"{stem}: proposed nothing at all")
        elif proposals[0].exp_type != exp_type:
            wrong.append(f"{stem}: proposed as {proposals[0].exp_type!r} "
                         f"— {proposals[0].evidence}")
    assert not wrong, "the sample files were misread: " + "; ".join(wrong)


# ---------------------------------------------------------------------------
# what it refuses to do
# ---------------------------------------------------------------------------

def test_a_table_that_is_no_type_proposes_nothing_rather_than_the_closest_one():
    """The answer "I do not recognise this" has to be available, or the ranking
    is worthless: a module that always names a best guess turns `misc` — the
    honest destination for an unrecognised file — into a thing nobody ever
    reaches."""
    assert propose_type(pd.DataFrame({"depth_m": [1.0, 2.0], "flux": [3.0, 4.0]})) == []

    # a cycler table whose current never leaves zero is not a measurement of
    # anything, and is not an RPT with a very small current
    at_rest = pd.DataFrame({"t_s": np.arange(100.0), "I_A": np.zeros(100),
                            "V_V": np.full(100, 3.7)})
    assert propose_type(at_rest, capacity_Ah=CAP_AH) == []


def test_a_decisive_column_wins_over_the_shape_of_the_current():
    """A three-electrode run is a cycler table with pulses in it, so the
    structural rules would have something to say about it. They must not get
    the chance: the reference electrode is a fact about the FILE, and the shape
    of the current is an inference about the protocol."""
    n = 300
    current = np.where((np.arange(n) % 100) < 10, -5.0, 0.0)
    frame = pd.DataFrame({
        "t_s": np.arange(float(n)), "I_A": current,
        "V_full_V": np.full(n, 3.7), "V_pos_V": np.full(n, 4.0),
        "V_neg_V": np.full(n, 0.3),
    })
    proposals = propose_type(frame, capacity_Ah=CAP_AH)
    assert [p.exp_type for p in proposals] == ["three_electrode_ts"]
    assert proposals[0].decisive


def test_without_a_capacity_the_rate_rules_go_quiet_and_say_so():
    """A C-rate needs a capacity and a GITT file does not contain one — it
    never completes a leg. Assuming one would put a rate on every file and be
    wrong by whatever the assumption was off by, which is the exact shape of
    the mA-read-as-A error the profiles exist to stop.

    So: the proposal is still made, from the pulse LENGTH, and the evidence
    names the missing input rather than quietly omitting it.
    """
    n = 4000
    # 30-minute pulses, 1-hour rests: GITT by duration at any current
    phase = (np.arange(n) // 500) % 2
    frame = pd.DataFrame({
        "t_s": np.arange(float(n)) * 3.6,
        "I_A": np.where(phase == 0, -CAP_AH / 20.0, 0.0),
        "V_V": np.full(n, 3.7),
    })

    blind = propose_type(frame)
    assert blind and blind[0].exp_type == "gitt"
    assert "no capacity" in blind[0].evidence, \
        "the missing input has to be named, not silently skipped"

    told = propose_type(frame, capacity_Ah=CAP_AH)
    assert told[0].exp_type == "gitt"
    assert "C/20" in told[0].evidence, "a C/20 pulse should be said to be one"


def test_evidence_that_disagrees_is_reported_as_disagreement():
    """The two pulse rules can contradict each other — a long pulse at a
    working current is a shape neither type has. Picking the stronger one and
    staying quiet is the only way this module could actively mislead somebody,
    so both are offered and each says the other disagrees.
    """
    n = 4000
    phase = (np.arange(n) // 500) % 2
    frame = pd.DataFrame({
        "t_s": np.arange(float(n)) * 3.6,           # 30-minute segments -> GITT
        "I_A": np.where(phase == 0, -CAP_AH, 0.0),  # ...at 1C           -> HPPC
        "V_V": np.full(n, 3.7),
    })
    proposals = propose_type(frame, capacity_Ah=CAP_AH)

    assert {p.exp_type for p in proposals} == {"gitt", "hppc"}
    assert proposals[0].exp_type == "hppc", "the rate is the stronger evidence and ranks first"
    for p in proposals:
        assert "disagree" in p.evidence


def test_the_same_shape_is_an_rpt_or_a_cycling_run_by_its_count_and_its_rate():
    """`rpt` and `cycling` are the same columns and the same shape. What makes
    one a check-up and the other the aging it checks up on is how many times it
    happens and how fast — which is exactly the distinction `rpt_rate_C20_min`
    exists to protect, so getting it wrong here would send fast data to the one
    module that cannot use it."""
    def legs(n_cycles: int, current_A: float) -> pd.DataFrame:
        per = 200
        rows = []
        for cycle in range(1, n_cycles + 1):
            for sign in (+1.0, -1.0):
                rows.append(pd.DataFrame({
                    "t_s": np.arange(per, dtype=float),
                    "I_A": np.full(per, sign * current_A),
                    "V_V": np.linspace(3.0, 4.2, per),
                    "cycle_index": np.full(per, cycle, dtype=np.int32),
                }))
        out = pd.concat(rows, ignore_index=True)
        out["t_s"] = np.arange(len(out), dtype=float)
        return out

    slow_once = propose_type(legs(1, CAP_AH / 10.0), capacity_Ah=CAP_AH)
    assert slow_once[0].exp_type == "rpt"

    slow_often = propose_type(legs(8, CAP_AH / 10.0), capacity_Ah=CAP_AH)
    assert slow_often[0].exp_type == "cycling"
    assert "8 distinct cycle_index" in slow_often[0].evidence

    # ...and one cycle at 1C is not a check-up either, because an RPT that fast
    # cannot do an RPT's job
    fast_once = propose_type(legs(1, CAP_AH), capacity_Ah=CAP_AH)
    assert fast_once[0].exp_type == "cycling"
    assert "Too fast for an RPT" in fast_once[0].evidence


def test_a_proposal_carries_a_sentence_and_not_a_score():
    """Deliberate design, asserted so it survives the first person who wants a
    number to sort by: what lets a person accept or reject a proposal is the
    observation behind it. A 0.72 would look like a probability, would not be
    one, and would be sorted on anyway."""
    assert not hasattr(TypeProposal("rpt", "because"), "score")
    assert not hasattr(TypeProposal("rpt", "because"), "confidence")


# ---------------------------------------------------------------------------
# probing a file on disk, and the CLI that prints it
# ---------------------------------------------------------------------------

def _landt_csv(path: Path, n: int = 40, cycles: int = 4) -> Path:
    """A Landt-shaped export of `cycles` charge/discharge cycles."""
    rows = ["TestTime/h,Current/mA,Voltage/V,Capacity/mAh,AuxTemp/dC.,"
            "Cycle-Index,Step-State,SpeCap/mAh/g"]
    t = 0
    for cycle in range(1, cycles + 1):
        for i in range(n):
            chg = i < n // 2
            t += 1
            rows.append(f"{t * 60 / 3600},{0.5 if chg else -0.5},"
                        f"{3.0 + 0.03 * i},{0.02 * i},25.0,{cycle},"
                        f"{'RateC' if chg else 'RateD'},{10.0 * i}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_probing_a_file_names_the_profile_the_type_and_what_was_not_read(tmp_path):
    """The three questions a person has about a file they did not create, in
    one call and with nothing written: which rig wrote it, what it is, and what
    the reader is going to leave behind."""
    from monocell.schema.shapes import probe_file

    probe = probe_file(_landt_csv(tmp_path / "run.csv"), capacity_Ah=0.0005)

    assert probe["profiles"] == ["landt"] and probe["profile"] == "landt"
    assert probe["types"] and probe["types"][0].exp_type == "cycling"
    assert "SpeCap/mAh/g" in probe["report"]["unplaced"]
    assert probe["error"] is None


def test_a_file_nothing_can_read_is_a_row_and_not_an_exception(tmp_path):
    """A directory of forty files can contain a photo. The row for it has to
    say so beside the other thirty-nine — a scan that raises on the first
    unreadable file is a scan nobody can run on a real directory."""
    from monocell.schema.shapes import probe_file

    junk = tmp_path / "notes.txt"
    junk.write_text("ran the cell overnight, check the fridge\n", encoding="utf-8")

    probe = probe_file(junk)
    assert probe["error"], "an unreadable file must SAY why, not look like an empty one"
    assert probe["types"] == [] and probe["rows"] == 0


def test_inspect_prints_a_row_per_file_and_survives_the_one_it_cannot_read(tmp_path, capsys):
    """The directory scan, and the property that makes it usable."""
    from monocell.cli import main as cli_main

    _landt_csv(tmp_path / "cell_a.csv")
    _landt_csv(tmp_path / "cell_b.csv")
    (tmp_path / "readme.txt").write_text("not data\n", encoding="utf-8")

    before = set(tmp_path.iterdir())
    assert cli_main(["inspect", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out

    assert out.count("cycling") >= 2, "both real files should have been proposed as cycling"
    assert "readme.txt" in out and "unreadable" in out
    assert set(tmp_path.iterdir()) == before, "inspect wrote something; it must write nothing"


def test_inspect_hands_back_the_command_that_would_act_on_its_proposal(tmp_path, capsys):
    """The payoff line. A proposal an engineer has to translate into a command
    is a proposal they will translate wrongly once — the profile and the type
    both have to survive the trip, and the profile is the half that is easy to
    forget."""
    from monocell.cli import main as cli_main

    csv = _landt_csv(tmp_path / "run.csv")
    assert cli_main(["inspect", "--file", str(csv)]) == 0
    out = capsys.readouterr().out

    assert "monocell ingest" in out
    assert "--type cycling" in out
    assert "--instrument landt" in out, \
        "the suggested command drops the profile, so running it would parse by guess"


def test_inspect_says_what_it_could_not_compute_without_a_cell(tmp_path, capsys):
    """A C-rate needs a capacity, and `--cell` is where the capacity comes from.
    Without it the rate rules go quiet, and the output has to say which
    evidence is missing rather than presenting a weaker answer as a full one."""
    from monocell.cli import main as cli_main

    cli_main(["inspect", "--file", str(_landt_csv(tmp_path / "run.csv"))])
    out = capsys.readouterr().out
    assert "no --cell given" in out
    assert "could not be computed" in out
