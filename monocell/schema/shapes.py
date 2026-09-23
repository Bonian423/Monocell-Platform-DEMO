"""What shape is this file? — type proposals from structure, never a decision.

An engineer with forty files in a directory should not have to name the type of
each one by hand, and a platform that guesses for them is worse than one that
does not guess at all. So this module does the middle thing: it reads the
structure of a parsed table and returns the types it is *consistent with*, each
with the evidence, ranked. The caller shows the ranking and a person picks.

THE DISCIPLINE IS THE SAME ONE `read_misc` AND `instruments.detect` HAVE.
Proposing is cheap and reversible; deciding is neither, because the store is
append-only and an experiment filed as the wrong type is corrected by writing a
second one. An empty list is a legitimate answer and means "nothing here looks
like a type I know" — which is exactly what `misc` is for.

WHAT SEPARATES THE TYPES, AND WHAT IT COSTS TO KNOW IT
-----------------------------------------------------
Four types are decided by a column no other type has — a frequency axis is an
EIS spectrum and nothing else. Those are `decisive` and need no numbers.

The rest are all "a cycler table of time, current and voltage", and they are
separated by what the CURRENT does:

  * **pulse/relax** — short driven segments separated by long rests. `gitt` and
    `hppc` both look like this, and what tells them apart is the pulse: GITT
    titrates at C/20 for half an hour, HPPC pulses at ~1C for a minute. Rate is
    the stronger evidence and duration corroborates it; when they disagree,
    BOTH are proposed with both reasons rather than one being picked.
  * **full legs** — the current runs one way for a long time and then the other.
    `rpt` and `cycling` are the same shape, separated by how many times it
    happens and how fast: an RPT is one slow check-up (C/20), cycling is the
    aging run that check-up interrupts.

**A C-rate needs a capacity, and a file does not always contain one.** A GITT
train never completes a leg, so nothing in it says how big the cell is. The
capacity therefore comes from the cell's own build record when the caller has
one, and when it does not, the rate rules are simply unavailable and the
evidence string says so. A default capacity would put a number on every file and
be wrong on some of them, which is the failure mode this platform exists to
avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# |I| below this fraction of the run's largest current counts as "at rest".
# Relative, because the same file can be a coin cell at a milliamp or a large
# pouch at a hundred amps, and neither an absolute threshold nor a noise model
# would survive both.
_REST_FRAC = 0.02

# A pulse at or below this C-rate is a titration (a GITT pulse is typically
# C/20); at or above the second, it is an HPPC pulse. Between them the evidence
# does not separate them and both are proposed.
_GITT_MAX_C = 0.10
_HPPC_MIN_C = 0.30

# Likewise for pulse duration, the corroborating evidence: GITT's default pulse
# is 30 min and HPPC's is a minute, so the gap is wide and the boundaries do not
# have to be precise to be useful.
_GITT_MIN_PULSE_S = 300.0
_HPPC_MAX_PULSE_S = 180.0

# The rate above which a single charge/discharge pair is not a check-up. NOT
# C/20, even though `rpt_rate_C20_min` is: that rule is about the rate recorded
# in `cell_state.rpt_rate`, and the bundled sample RPTs drive their legs at C/10
# while recording C/20. A boundary at C/10 would sit exactly on that data and
# flip with the last bit of a float. C/4 is nowhere near either an RPT leg or a
# duty cycle.
_RPT_MAX_C = 0.25


@dataclass(frozen=True)
class TypeProposal:
    """One type this file could be, and why.

    `evidence` is a sentence, not a score. A number between 0 and 1 would look
    like a probability and would not be one; what a person needs in order to
    accept or reject a proposal is the observation behind it.
    """

    exp_type: str
    evidence: str
    decisive: bool = False


# ---------------------------------------------------------------------------
# the decisive columns — a type that owns a column owns the file
# ---------------------------------------------------------------------------

# Each entry: the columns that identify the type, and what they mean. Ordered,
# and checked in order, because `three_electrode_ts` is also a cycler table and
# would otherwise fall through to the structural rules.
_SIGNATURE_COLUMNS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("eis", ("f_Hz",),
     "a frequency axis (`f_Hz`), which no time-domain type has"),
    ("pressure", ("F_N",),
     "a force channel (`F_N`) — the stack-pressure rig's own measurement"),
    ("pressure", ("p_kPa",),
     "a pressure channel (`p_kPa`) — the stack-pressure rig's own measurement"),
    ("three_electrode_ts", ("V_pos_V", "V_neg_V"),
     "separate positive and negative potentials, so the cell has a reference "
     "electrode in it"),
    ("half_cell_ocp", ("V_vs_Li_V",),
     "a potential against lithium (`V_vs_Li_V`), which is a half cell"),
)


def _decisive(frame: pd.DataFrame) -> list[TypeProposal]:
    cols = set(frame.columns)
    found: list[TypeProposal] = []
    for exp_type, needed, why in _SIGNATURE_COLUMNS:
        if set(needed) <= cols and not any(p.exp_type == exp_type for p in found):
            found.append(TypeProposal(exp_type, f"this file carries {why}.", decisive=True))
    return found


# ---------------------------------------------------------------------------
# the structural rules — what the current does
# ---------------------------------------------------------------------------

def _runs(at_rest: np.ndarray) -> list[tuple[int, int, bool]]:
    """`[(start, stop, at_rest), ...]` — the boolean array as its runs."""
    if not len(at_rest):
        return []
    edges = np.flatnonzero(at_rest[1:] != at_rest[:-1]) + 1
    bounds = [0, *edges.tolist(), len(at_rest)]
    return [(a, b, bool(at_rest[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def _c_rate(current_A: float, capacity_Ah: float | None) -> float | None:
    if not capacity_Ah or capacity_Ah <= 0:
        return None
    return abs(current_A) / capacity_Ah


def _fmt_c(rate: float) -> str:
    """A C-rate the way an engineer says it: `C/20`, not `0.05 C`."""
    if rate <= 0:
        return "0 C"
    return f"{rate:.3g} C" if rate >= 0.5 else f"C/{1.0 / rate:.3g}"


def _pulse_family(frame: pd.DataFrame, driven: list[tuple[int, int, bool]],
                  capacity_Ah: float | None) -> list[TypeProposal]:
    """`gitt` or `hppc` — a pulse train, split by what the pulse is."""
    t = frame["t_s"].to_numpy(dtype=float)
    current = frame["I_A"].to_numpy(dtype=float)
    durations = [t[b - 1] - t[a] for a, b, _ in driven if b - a > 1]
    amplitudes = [float(np.nanmedian(np.abs(current[a:b]))) for a, b, _ in driven]
    if not durations or not amplitudes:
        return []
    pulse_s = float(np.median(durations))
    pulse_A = float(np.median(amplitudes))
    rate = _c_rate(pulse_A, capacity_Ah)

    shape = (f"{len(driven)} driven segment(s) of about {pulse_s:.0f} s separated by rests")
    by_rate: str | None = None
    if rate is not None:
        if rate <= _GITT_MAX_C:
            by_rate = "gitt"
        elif rate >= _HPPC_MIN_C:
            by_rate = "hppc"
    by_time = "gitt" if pulse_s >= _GITT_MIN_PULSE_S else (
        "hppc" if pulse_s <= _HPPC_MAX_PULSE_S else None)

    rate_note = (f"pulses at {_fmt_c(rate)}" if rate is not None
                 else "no capacity for this cell was given, so the pulse C-rate could not be "
                      "computed and only the duration is evidence")
    both = [c for c in (by_rate, by_time) if c]
    if not both:
        return [TypeProposal("hppc", f"{shape}; {rate_note}. That is a pulse train, but nothing "
                                     "in it separates an HPPC from a GITT titration.", False),
                TypeProposal("gitt", f"{shape}; {rate_note}. That is a pulse train, but nothing "
                                     "in it separates a GITT titration from an HPPC.", False)]
    if by_rate and by_time and by_rate != by_time:
        # Disagreement is reported as disagreement. Picking one and staying
        # quiet would be the only way this module could mislead anybody.
        return [TypeProposal(by_rate, f"{shape}; {rate_note} — an {by_rate.upper()} rate. The "
                                      f"pulse LENGTH says {by_time.upper()} instead, so these "
                                      "two files' worth of evidence disagree and both are "
                                      "offered.", False),
                TypeProposal(by_time, f"{shape}; the pulse length of {pulse_s:.0f} s is a "
                                      f"{by_time.upper()} pulse. The rate disagrees "
                                      f"({rate_note}).", False)]
    chosen = by_rate or by_time
    if chosen == "gitt":
        return [TypeProposal("gitt", f"{shape}; {rate_note}. A slow pulse followed by a long "
                                     "relaxation is a GITT titration.", False)]
    return [TypeProposal("hppc", f"{shape}; {rate_note}. Short pulses at a working current, "
                                 "each followed by a relaxation, is an HPPC.", False)]


def _leg_family(frame: pd.DataFrame, driven: list[tuple[int, int, bool]],
                capacity_Ah: float | None) -> list[TypeProposal]:
    """`rpt` or `cycling` — full charge/discharge legs, split by count and rate."""
    current = frame["I_A"].to_numpy(dtype=float)
    amplitudes = [float(np.nanmedian(np.abs(current[a:b]))) for a, b, _ in driven]
    leg_A = float(np.median(amplitudes)) if amplitudes else 0.0
    rate = _c_rate(leg_A, capacity_Ah)

    if "cycle_index" in frame.columns:
        n_cycles = int(pd.Series(frame["cycle_index"]).nunique())
        counted = f"{n_cycles} distinct cycle_index value(s)"
    else:
        # No cycle column: a cycle is a change of direction, and two changes
        # make a cycle. Named as the weaker evidence it is.
        signs = np.sign([float(np.nanmedian(current[a:b])) for a, b, _ in driven])
        n_cycles = max(1, int(np.sum(signs[1:] != signs[:-1])) // 2 + 1)
        counted = (f"no cycle_index column, so the {len(driven)} driven leg(s) were counted as "
                   f"about {n_cycles} cycle(s) by their direction changes")

    rate_note = (f"at {_fmt_c(rate)}" if rate is not None
                 else "at a C-rate that could not be computed (no capacity was given for "
                      "this cell)")

    if n_cycles >= 3:
        return [TypeProposal("cycling", f"full charge/discharge legs, {counted}, {rate_note}. "
                                        "That is an aging run, not a check-up.", False)]
    if rate is not None and rate > _RPT_MAX_C:
        return [TypeProposal(
            "cycling", f"full charge/discharge legs, {counted}, {rate_note}. Too fast for an "
                       "RPT, which must be C/20 or slower to keep the voltage curve's fine "
                       "structure readable — so it is filed as cycling even though it is only "
                       "a cycle or two.", False)]
    return [TypeProposal("rpt", f"full charge/discharge legs, {counted}, {rate_note}. One slow "
                                "charge/discharge pair is the check-up an RPT is.", False)]


def propose_type(frame: pd.DataFrame, capacity_Ah: float | None = None) -> list[TypeProposal]:
    """The types this table is consistent with, best first, each with its evidence.

    An empty list is the honest answer for a table that is not one of these —
    the caller's move then is `misc`, which stores it and unlocks nothing.

    `capacity_Ah` comes from the cell's build record. Without it every rule that
    needs a C-rate goes quiet and says so, rather than assuming a capacity and
    reporting a rate that is wrong by whatever factor the assumption was off.
    """
    decisive = _decisive(frame)
    if decisive:
        return decisive
    if not {"t_s", "I_A"} <= set(frame.columns):
        return []

    current = np.asarray(frame["I_A"], dtype=float)
    finite = current[np.isfinite(current)]
    if not len(finite) or not np.any(np.abs(finite) > 0):
        return []
    at_rest = np.abs(np.nan_to_num(current)) < _REST_FRAC * float(np.max(np.abs(finite)))
    runs = _runs(at_rest)
    driven = [r for r in runs if not r[2]]
    if not driven:
        return []

    resting = float(np.count_nonzero(at_rest)) / len(at_rest)
    # A pulse train is mostly rest; a charge/discharge run is mostly not.
    # Typical extremes are ~92% (GITT: 30 min on, 6 h off) and 0% (an RPT
    # sweep), so the boundary is nowhere near either and does not need care.
    if resting > 0.30 and len(driven) >= 3:
        return _pulse_family(frame, driven, capacity_Ah)
    return _leg_family(frame, driven, capacity_Ah)


# ---------------------------------------------------------------------------
# probing a file on disk — everything that can be said without being told
# ---------------------------------------------------------------------------

# What `probe_file` covers, and what it does not. It reads the CYCLER-TABLE
# family: anything an instrument profile can map, plus a plain cycler CSV the
# synonym list can read. That is deliberate rather than a shortfall — the reason
# folder ingest exists is an aging campaign of forty RPT files in one directory,
# and those are cycler tables. An EIS workbook or a pressure-rig export arrives
# one at a time and already has a reader that knows its shape; a file this
# cannot parse is reported as unparsed, with the reader's own error, which is a
# more useful answer than a guess at what it might have been.


def probe_file(path, root=None, capacity_Ah: float | None = None) -> dict:
    """`{profile, types, report, ...}` — what a file looks like, before any decision.

    Nothing is written and nothing is chosen. The caller gets the profiles whose
    signature matches, the type proposals with their evidence, and the parse
    report for whichever reader got the table — and picks.

    `error` is set and the rest is empty when nothing could read it. That is a
    normal outcome, not an exception: a directory of forty files can contain a
    photo, and the row for it should say so beside the other thirty-nine rather
    than stopping the scan.
    """
    from pathlib import Path

    from .ingest import read_cycler_csv, read_with_instrument
    from .instruments import detect, load_profiles

    path = Path(path)
    out: dict = {"path": path, "error": None, "profiles": [], "profile": None,
                 "report": None, "rows": 0, "columns": [], "types": []}
    library = load_profiles(root)
    out["profiles"] = [name for name, _ in detect(path, library)]

    frame = None
    if out["profiles"]:
        name = out["profiles"][0]
        try:
            # `rpt` only to give the report a type to check its columns
            # against; the type this file IS gets proposed below, from the
            # frame, and the report's own type-level fields are recomputed by
            # whatever ingest the caller actually asks for.
            frame, report = read_with_instrument("rpt", path, library[name])
            out["profile"], out["report"] = name, report
        except Exception as exc:  # noqa: BLE001 — an unreadable file is a row, not a crash
            out["error"] = f"profile {name!r} could not read it: {type(exc).__name__}: {exc}"
    if frame is None and out["error"] is None:
        try:
            frame = read_cycler_csv(path, exp_type="rpt")
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"

    if frame is None:
        return out
    out["rows"] = int(len(frame))
    out["columns"] = list(frame.columns)
    out["types"] = propose_type(frame, capacity_Ah=capacity_Ah)
    return out
