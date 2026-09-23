"""The `cycling` type: the aging DRIVER, and why it is not an `rpt`.

An RPT is a periodic check-up and is deliberately SLOW — `rpt_rate_C20_min`
enforces C/20 — because a slow check-up keeps the fine structure of the voltage
curve readable; at 1C polarisation smears it. Cycling data cannot do an RPT's
job, and filing it as one would either fail that gate forever or force the gate
to be weakened. The gate is right.

What cycling data CAN give is the fade curve at every cycle instead of at the
handful of RPT points.

The load-bearing test here is `test_a_file_that_mixes_rates_...`: a real aging
campaign often has a slow check-up at each end of its duty cycling, and a
first-vs-last retention then compares one check-up against the other.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import BUILD, register_cell
from monocell.schema import write_experiment
from monocell.schema.quality import cycling_soh

CELL = "cyc_cell"


def _cycle(index: int, cap_Ah: float, current_A: float, ce: float = 0.995) -> pd.DataFrame:
    """One charge/discharge cycle at a stated capacity and current."""
    n = 25
    chg_cap = cap_Ah / ce
    frames = []
    for seg, cap, sign in (("charge", chg_cap, +1.0), ("discharge", cap_Ah, -1.0)):
        frames.append(pd.DataFrame({
            "t_s": np.linspace(0, cap / current_A * 3600.0, n),
            "I_A": np.full(n, sign * current_A),
            "V_V": np.linspace(3.0, 4.2, n)[::int(sign)],
            "T_degC": np.full(n, 25.0),
            "q_Ah": np.linspace(0.0, cap, n),
            "seg_type": np.full(n, seg, dtype=object),
            "cycle_index": np.full(n, index, dtype=np.int32),
        }))
    return pd.concat(frames, ignore_index=True)


def _series(specs) -> pd.DataFrame:
    """`[(cycle_index, capacity_Ah, current_A), ...]` -> one series frame."""
    out = pd.concat([_cycle(*s) for s in specs], ignore_index=True)
    out["t_s"] = np.arange(len(out), dtype=float)   # monotone across the run
    return out


def _meta(**cell_state) -> dict:
    return {
        "producer": {"kind": "real", "software": "test", "version": "0"},
        "protocol": {"description": "1C cycling", "c_rate": 1.0},
        "cell_state": {"rpt_index": 0, **cell_state},
        "instrument": {"sampling_rate_Hz": 1.0},
    }


# ---------------------------------------------------------------------------
# the fade curve
# ---------------------------------------------------------------------------

def test_every_cycle_gets_its_capacity_and_coulombic_efficiency():
    """The point of the type: the trajectory, not a summary of it."""
    caps = [1.00e-3, 0.98e-3, 0.96e-3, 0.94e-3]
    q = cycling_soh(_series([(i + 1, c, 1e-3) for i, c in enumerate(caps)]), _meta())

    assert q["n_cycles"] == 4
    got = [r["discharge_Ah"] for r in q["per_cycle"]]
    assert got == pytest.approx(caps, rel=1e-6)
    for r in q["per_cycle"]:
        assert r["coulombic_efficiency"] == pytest.approx(0.995, rel=1e-6)
    assert q["capacity_retention"] == pytest.approx(0.94, rel=1e-6)


def test_capacity_soh_needs_a_reference_and_says_so_when_it_has_none():
    """`capacity_SOH` and `capacity_retention` answer different questions, and
    a reference the file does not carry must not be invented from the file's
    own first cycle — that is retention wearing SOH's name."""
    series = _series([(1, 1.0e-3, 1e-3), (2, 0.9e-3, 1e-3)])

    bare = cycling_soh(series, _meta())
    assert bare["capacity_SOH"] is None
    assert bare["capacity_retention"] == pytest.approx(0.9, rel=1e-6)
    assert any("no capacity_ref_Ah" in f for f in bare["flags"])

    known = cycling_soh(series, _meta(capacity_ref_Ah=2.0e-3))
    assert known["capacity_SOH"] == pytest.approx(0.45, rel=1e-6)


# ---------------------------------------------------------------------------
# the one that matters
# ---------------------------------------------------------------------------

def test_a_file_that_mixes_rates_reports_retention_within_one_rate():
    """A real aging campaign is often not one protocol.

    The shape below is a common one: a slow check-up, a run of duty cycles that
    fade, and a slow check-up at the end. Comparing the file's first cycle to
    its last compares CHECK-UP to CHECK-UP, and those barely move, so the
    answer would read as healthy for a cell whose duty capacity has clearly
    fallen, computed from data that contains the right answer.

    So retention is computed within the MAJORITY rate, and the minority cycles
    are named rather than averaged in.
    """
    duty, slow = 1e-3, 5e-5              # a factor of twenty, i.e. 1C vs C/20
    series = _series(
        [(1, 2.0e-3, slow)]                                        # check-up
        + [(i, 1.0e-3 - 0.002e-3 * i, duty) for i in range(2, 12)]  # the cycling
        + [(12, 1.98e-3, slow)])                                   # check-up

    q = cycling_soh(series, _meta())

    naive = 1.98e-3 / 2.0e-3
    assert q["capacity_retention"] != pytest.approx(naive, rel=1e-3), \
        "retention compared the two check-ups and reported the cycling as healthy"
    assert q["capacity_retention"] == pytest.approx((1.0e-3 - 0.002e-3 * 11)
                                                    / (1.0e-3 - 0.002e-3 * 2), rel=1e-6)
    assert q["duty_current_A"] == pytest.approx(duty, rel=1e-6)
    assert q["n_cycles"] == 10 and q["n_cycles_all_rates"] == 12
    assert q["first_cycle"] == 2 and q["last_cycle"] == 11

    flag = next(f for f in q["flags"] if "mixes rates" in f)
    assert "2 at" in flag, "the minority cycles must be counted in the flag"
    assert "own experiment" in flag, "and the reader told what they probably are"


def test_a_single_rate_file_raises_no_mixing_flag():
    """The flag has to mean something. A file that really is one protocol must
    not carry a warning about a second."""
    q = cycling_soh(_series([(i, 1e-3, 1e-3) for i in range(1, 6)]), _meta())
    assert not any("mixes rates" in f for f in q["flags"])


def test_a_sliced_export_says_where_its_retention_is_measured_from():
    """Real exports are often slices that open part-way through a run.
    Retention against the first cycle IN THE FILE is then not retention from
    BOL, and the reader cannot tell without being told.

    The sentence names the cycle the denominator came from rather than "the
    first cycle in the file", because when a file opens with a slow check-up
    those are different cycles.
    """
    q = cycling_soh(_series([(i, 1e-3, 1e-3) for i in range(11, 15)]), _meta())
    assert q["first_cycle"] == 11
    assert any("measured from cycle 11" in f and "duty rate" in f for f in q["flags"])


def test_a_cycle_whose_charge_landed_in_the_next_index_is_flagged():
    """A coulombic efficiency far below 1 at a protocol change is usually not a
    cell losing half its charge: it is a charge and its discharge landing either
    side of a cycle-index boundary, so the charge span of one index covers two
    protocols' worth.

    The number is still reported per cycle, because clipping a measurement to
    make it look sane is the one thing this platform must never do. What is
    added is the sentence saying which of those numbers are not measurements
    of the cell.

    Both halves are asserted. A flag that fires on ordinary data trains the
    reader to ignore it, and then it is worth less than nothing.
    """
    clean = _series([(i, 1e-3, 1e-3) for i in range(1, 5)])
    assert not any("coulombic efficiency" in f for f in cycling_soh(clean, _meta())["flags"])

    split = _series([(1, 1e-3, 1e-3), (2, 1e-3, 1e-3, 0.5), (3, 1e-3, 1e-3)])
    q = cycling_soh(split, _meta())
    flag = next(f for f in q["flags"] if "coulombic efficiency" in f)
    assert "1 of 3" in flag
    assert "cycle 2" in flag, "the flag has to name which cycle, or it cannot be checked"

    # ...and the number itself is untouched
    assert q["per_cycle"][1]["coulombic_efficiency"] == pytest.approx(0.5, rel=1e-6)
    # ...as is the capacity, which is one segment and still a measurement
    assert q["capacity_retention"] == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# the type, through the real writer
# ---------------------------------------------------------------------------

def test_cycling_is_not_held_to_the_rpt_rate_gate(root):
    """`rpt_rate_C20_min` exists because a check-up needs a slow sweep. Cycling
    data is the aging DRIVER and is fast on purpose, so the gate must not apply,
    and `rpt` must still carry it, or moving the data here would have been a
    way to dodge a check rather than to classify it."""
    from monocell.schema.tables import SPECS

    assert "rpt_rate_C20_min" not in SPECS["cycling"].rules
    assert "rpt_rate_C20_min" in SPECS["rpt"].rules

    register_cell(CELL, BUILD, root)
    d = write_experiment(CELL, "cycling", _meta(capacity_ref_Ah=1.0e-3),
                         _series([(i, 1e-3, 1e-3) for i in range(1, 4)]), root=root)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert not [f for f in meta["quality"]["flags"] if "C/20" in f]


def test_the_per_cycle_table_reaches_the_stored_quality_record(root):
    """The trajectory is the deliverable, so it has to survive the writer:
    `meta.quality` carries a fixed summary key set, and the full record lives
    in `quality.json` beside it."""
    register_cell(CELL, BUILD, root)
    d = write_experiment(CELL, "cycling", _meta(capacity_ref_Ah=1.0e-3),
                         _series([(1, 1.0e-3, 1e-3), (2, 0.9e-3, 1e-3)]), root=root)

    q = json.loads((d / "quality.json").read_text(encoding="utf-8"))
    assert [r["cycle_index"] for r in q["per_cycle"]] == [1, 2]
    assert q["capacity_retention"] == pytest.approx(0.9, rel=1e-6)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["quality"]["capacity_SOH"] == pytest.approx(0.9, rel=1e-6)


# ---------------------------------------------------------------------------
# the type's own headline number reaches the header
# ---------------------------------------------------------------------------


def test_retention_reaches_the_header_of_a_file_with_no_capacity_reference(root):
    """What `TableSpec.quality_summary` is for.

    `meta.quality` is projected out of the full quality record by each type's
    own declaration. A cycling export with no `capacity_ref_Ah` has
    `capacity_SOH = None` (correctly, there is nothing to reference it to), and
    without the declaration its retention would be computed, written to
    `quality.json`, and dropped before anything a reader sees.

    Both halves are asserted: the number the type declares arrives, and the one
    it cannot compute is still honestly absent.
    """
    from monocell.schema import load_experiment

    meta = _meta()                      # no capacity_ref_Ah
    register_cell(CELL, BUILD, root)
    d = write_experiment(CELL, "cycling", meta,
                         _series([(1, 1.0e-3, 1e-3), (2, 0.95e-3, 1e-3),
                                  (3, 0.9e-3, 1e-3)]), root=root)

    header = json.loads((d / "meta.json").read_text(encoding="utf-8"))["quality"]
    assert header["capacity_SOH"] is None
    assert header["capacity_retention"] == pytest.approx(0.9, rel=1e-6)
    assert header["n_cycles"] == 3
    # ...and the full record still carries what a header must not: the
    # per-cycle trajectory stays in quality.json, where the reader that wants
    # it already looks. A header that grew one row per cycle would stop being a
    # header, and it is parsed once per row of every experiment listing.
    full = load_experiment(d.name, CELL, root)["quality"]
    assert len(full["per_cycle"]) == 3
    assert "per_cycle" not in header


def test_a_types_declared_summary_keys_are_keys_its_quality_record_can_hold():
    """A declaration that names a key nothing produces is a column that is
    always empty, and it would read as "this file had no retention" rather than
    as "nobody wrote that key". Checked for every type that declares any,
    against the record `compute_quality` actually builds."""
    from monocell.schema.quality import compute_quality
    from monocell.schema.tables import SHARED_QUALITY_KEYS, SPECS

    declaring = {t: s for t, s in SPECS.items() if s.quality_summary}
    assert declaring, "no type declares a summary key, so this test proves nothing"
    for exp_type, spec in declaring.items():
        for key in spec.quality_summary:
            assert key not in SHARED_QUALITY_KEYS, (
                f"{exp_type} re-declares the shared key {key!r}, which every type carries anyway"
            )
        assert exp_type == "cycling", (
            f"{exp_type} declares summary keys and this test only builds a cycling frame — "
            "give it one, rather than leaving the declaration unchecked"
        )
        produced = compute_quality(exp_type, _series([(1, 1.0e-3, 1e-3), (2, 0.9e-3, 1e-3)]),
                                   _meta())
        missing = [k for k in spec.quality_summary if k not in produced]
        assert not missing, f"{exp_type} declares {missing}, which compute_quality never writes"
