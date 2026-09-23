"""Instrument profiles — one parser, many cyclers.

WHY THIS IS A PROFILE AND NOT A READER PER VENDOR
-------------------------------------------------
The obvious response to "our lab runs several different cyclers" is a reader
per cycler. It is the wrong one. Three readers is a month of work that covers
three machines, goes stale the first time one of them changes its export, and
leaves the fourth machine — the one bought next year — as unusable as before.

What actually differs between a Landt export and a Neware one is *not*
structure. Both are a rectangular table of time, current, voltage, capacity and
a step label. What differs is **names, units, and junk**: `Current/mA` against
`Current(mA)`, hours against seconds, an all-null spacer column in the middle,
a step vocabulary of `RateC`/`RateD` against `CC_Chg`/`CC_DChg`. Those are
*data*, and data belongs in a record an engineer can read, edit and save — not
in a function.

So there is one parser (`read_with_profile`) and a library of profiles. Adding
a cycler is adding a `InstrumentProfile(...)` literal; adding a rig with a
custom export is cloning one and changing three lines, which is something the
person who owns the rig can do without touching Python.

VERIFIED AND UNVERIFIED PROFILES ARE DIFFERENT THINGS, AND THE RECORD SAYS SO
-----------------------------------------------------------------------------
Some profiles have been checked against a real export; others were written
from published column names and **have never met a file**. That is a real
difference: a header spelled `Test Time` where the profile says `TestTime`
fails, and the failure is silent if the parser drops what it cannot place.

So `verified` is a field, every profile carries the evidence behind its names,
and the ingest path reports an unverified profile as what it is: a starting
point that got you most of the way and needs one look.

NOTHING IS DROPPED SILENTLY
---------------------------
The rule this module exists to hold: a column the profile cannot place is
REPORTED, never discarded. `read_cycler_csv` raises only when the *time* column
is missing; every other column it fails to recognise simply does not arrive,
which is how a rig whose header is one character off loses its temperature
channel without anybody finding out. `read_with_profile` returns the unplaced
names alongside the frame, and `ingest` puts them in the experiment's own
quality flags.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# The closed vocabulary `ColumnRule.parse` draws on. A dict rather than a tuple
# plus a dispatch chain, so the set of legal values and the code that implements
# them cannot disagree — adding a key here is the whole of adding a form.
PARSERS: dict[str, str] = {
    "number": "float(cell), with the profile's null tokens becoming NaN",
    "duration_hms": "hh:mm:ss[.f] | mm:ss | a bare number of seconds -> seconds",
    "datetime": "an absolute timestamp -> seconds since the first row",
}


@dataclass(frozen=True)
class ColumnRule:
    """One source header -> one schema column, with the unit conversion.

    `scale` is what the SOURCE must be multiplied by to reach the schema's
    unit: hours -> seconds is 3600, mA -> A is 1e-3, mAh -> Ah is 1e-3. It is a
    number rather than a unit string on purpose — a unit parser is a second
    thing to be wrong about, and every conversion a cycler needs is a constant.
    """

    source: str
    target: str
    scale: float = 1.0
    offset: float = 0.0
    note: str = ""
    # HOW the source cell becomes a number, from a closed vocabulary — see
    # `PARSERS` below. The same argument as `scale`: a strptime pattern is a
    # second thing to be wrong about, and it would be wrong in a way that
    # silently produces NaN for a whole column. Three forms cover every export
    # this platform has met or is likely to:
    #
    #   number         `float(cell)`, the default and what every rule was
    #                  before this field existed
    #   duration_hms   `hh:mm:ss[.f]` (and `mm:ss`, and a bare number of
    #                  seconds) -> seconds. Neware writes `Test Time` this way
    #                  in some versions, which the `neware` profile's own
    #                  `evidence` string admitted it could not parse
    #   datetime       an absolute timestamp -> seconds since the FIRST row.
    #                  A wall-clock column is not an elapsed time, and the
    #                  subtraction is the whole of the conversion
    #
    # `scale` and `offset` still apply AFTER the parse, so a `duration_hms`
    # column that a rig writes in minutes is still expressible.
    parse: str = "number"

    def __post_init__(self) -> None:
        if self.parse not in PARSERS:
            raise ValueError(
                f"{self.source!r} -> {self.target!r}: parse {self.parse!r} is not one of "
                f"{tuple(PARSERS)}. The vocabulary is closed on purpose — a format string here "
                "would be a second thing to be wrong about, and wrong in a way that turns a "
                "whole column into NaN without saying so"
            )


@dataclass(frozen=True)
class InstrumentProfile:
    """How to read one cycler's export. Data, not code — see the module docstring."""

    name: str
    vendor: str
    columns: tuple[ColumnRule, ...]
    # Header values that identify this profile. Matched as a SUBSET of the
    # file's headers, so a profile stays matchable when a rig gains a column.
    signature: tuple[str, ...] = ()
    # The vendor's step vocabulary -> the schema's `seg_type`. Compared
    # case-insensitively after stripping, because exports are inconsistent
    # about both and neither carries meaning.
    seg_map: dict[str, str] = field(default_factory=dict)
    # Cell values meaning "nothing was recorded". A channel that was not
    # connected is not a zero.
    null_tokens: tuple[str, ...] = ("-", "--", "", "NaN", "null", "N/A")
    sheet: str | None = None          # xlsx: which sheet, None = the first
    header_row: int = 0
    decimal: str = "."
    encoding: str | None = None       # csv only
    verified: bool = False
    evidence: str = ""                # where the column names came from
    # Whether this rig's time column RESTARTS at each protocol step.
    #
    # Declared rather than detected, because the two cases it separates look
    # identical in the numbers: a `t_s` that goes 0, 1, 2, 0, 1, 2 is either a
    # per-step clock (fine, and it has to be accumulated) or a corrupted export
    # (not fine). Only the rig knows which, and the profile is where what the
    # rig does is written down.
    #
    # `read_with_profile` ACCUMULATES when this is set — each step's clock is
    # offset by the total elapsed time before it — so the frame that reaches
    # `write_experiment` has the monotone `t_s` the schema means. When it is not
    # set and the column is non-monotone, that is the corrupted case and it is
    # flagged.
    time_restarts_per_step: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InstrumentProfile:
        cols = tuple(ColumnRule(**c) for c in d.get("columns", ()))
        rest = {k: v for k, v in d.items() if k != "columns"}
        for key in ("signature", "null_tokens"):
            if key in rest and rest[key] is not None:
                rest[key] = tuple(rest[key])
        return cls(columns=cols, **rest)

    def clone(self, name: str, **changes: Any) -> InstrumentProfile:
        """A copy under a new name — how a rig-specific profile is made.

        `verified` resets to False and stays reset unless the caller says
        otherwise: a clone has not met a file, whatever its parent had met.
        The default `evidence` is overridable, because the usual reason to
        clone is that you HAVE a file and a correction to record.
        """
        edits: dict[str, Any] = {"verified": False,
                                 "evidence": f"cloned from {self.name!r}"}
        edits.update(changes)
        return replace(self, name=name, **edits)


# ---------------------------------------------------------------------------
# the built-in library
# ---------------------------------------------------------------------------

# Landt, `verified`.
#
# Quirks of the export format:
#   * units live in the header: `TestTime/h`, `Current/mA`, `Capacity/mAh`
#   * an all-null spacer column (`Unnamed: 13`) can sit between `SysTime` and
#     `Cycle-Index`, so the identifying columns come after a hole
#   * `AuxTemp/dC.` is present and entirely `-` when the probe is unplugged,
#     which is a string column of null tokens rather than a missing one
#   * an export can be a SLICE of a longer run: neither the record number nor
#     the test time starts at zero
LANDT = InstrumentProfile(
    name="landt",
    vendor="Landt",
    signature=("TestTime/h", "Current/mA", "Voltage/V", "Step-State"),
    columns=(
        ColumnRule("TestTime/h", "t_s", scale=3600.0, note="hours -> seconds"),
        ColumnRule("Current/mA", "I_A", scale=1e-3, note="mA -> A"),
        ColumnRule("Voltage/V", "V_V"),
        ColumnRule("Capacity/mAh", "q_Ah", scale=1e-3, note="mAh -> Ah"),
        ColumnRule("AuxTemp/dC.", "T_degC", note="often unconnected and all '-'"),
        ColumnRule("Cycle-Index", "cycle_index"),
        ColumnRule("Step-State", "seg_type"),
    ),
    seg_map={"r": "rest", "ratec": "charge", "rated": "discharge",
             "cc_chg": "charge", "cc_dchg": "discharge"},
    verified=True,
    evidence="checked against a real export",
)

# The three below have NEVER MET A FILE. Their column names come from the
# vendors' published export formats, which is a real source and a weaker one
# than a file in hand: a header spelled `Test Time` where this says `Test_Time`
# will not match, and the ingest path will say so rather than quietly dropping
# the column. Treat each as a starting point that needs one look, clone it, and
# send the correction back.
NEWARE = InstrumentProfile(
    name="neware",
    vendor="Neware",
    signature=("Voltage(V)", "Current(mA)"),
    columns=(
        ColumnRule("Test Time", "t_s", parse="duration_hms",
                   note="hh:mm:ss.f in some firmware versions and a decimal number of seconds "
                        "in others. `duration_hms` accepts BOTH, which is what makes this rule "
                        "safe to set on a profile no file has verified"),
        ColumnRule("Current(mA)", "I_A", scale=1e-3),
        ColumnRule("Voltage(V)", "V_V"),
        ColumnRule("Capacity(mAh)", "q_Ah", scale=1e-3),
        ColumnRule("Aux Temperature", "T_degC"),
        ColumnRule("Cycle Index", "cycle_index"),
        ColumnRule("Step Type", "seg_type"),
    ),
    seg_map={"cc_chg": "charge", "cc chg": "charge", "cccv_chg": "charge",
             "cc_dchg": "discharge", "cc dchg": "discharge", "rest": "rest"},
    verified=False,
    evidence="Neware BTS export column names; NOT checked against a file. `Test Time` now "
             "parses as `duration_hms`, which takes hh:mm:ss.f AND a bare number of seconds — "
             "so the version difference no longer decides whether the column arrives. "
             "`time_restarts_per_step` is left FALSE: Neware is reported to restart its step "
             "clock in some configurations, no export checked so far does, and setting it on "
             "a guess would accumulate a column that was already elapsed.",
)

MACCOR = InstrumentProfile(
    name="maccor",
    vendor="Maccor",
    signature=("Volts", "Amps"),
    columns=(
        ColumnRule("TestTime", "t_s"),
        ColumnRule("Amps", "I_A"),
        ColumnRule("Volts", "V_V"),
        ColumnRule("Amp-hr", "q_Ah"),
        ColumnRule("Temp 1", "T_degC"),
        ColumnRule("Cyc#", "cycle_index"),
        ColumnRule("State", "seg_type"),
    ),
    seg_map={"c": "charge", "d": "discharge", "r": "rest", "o": "rest"},
    verified=False,
    evidence="Maccor export column names; NOT checked against a file.",
)

ARBIN = InstrumentProfile(
    name="arbin",
    vendor="Arbin",
    signature=("Test_Time(s)", "Voltage(V)"),
    columns=(
        ColumnRule("Test_Time(s)", "t_s"),
        ColumnRule("Current(A)", "I_A"),
        ColumnRule("Voltage(V)", "V_V"),
        ColumnRule("Aux_Temperature_1(C)", "T_degC"),
        ColumnRule("Cycle_Index", "cycle_index"),
    ),
    verified=False,
    evidence="Arbin export column names; NOT checked against a file. Arbin splits capacity "
             "into `Charge_Capacity(Ah)` and `Discharge_Capacity(Ah)`, which one source "
             "column per target cannot express — so `q_Ah` is UNMAPPED here rather than "
             "mapped to a half-truth. Combining them needs a rule this record does not yet "
             "have, and inventing one for a format nobody has tested against would be "
             "guessing twice.",
)

BUILTIN: dict[str, InstrumentProfile] = {
    p.name: p for p in (LANDT, NEWARE, MACCOR, ARBIN)
}


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def raw_table(path: Path, profile: InstrumentProfile | None = None,
              nrows: int | None = None) -> pd.DataFrame:
    """The table as the file has it — headers and values, before any mapping.

    Public so a caller can show a file's own headers and first rows before any
    mapping, which is what a person correcting a profile needs to look at.
    `nrows` asks for a head, and the file-level quirks (`header_row`,
    `decimal`, `encoding`, `sheet`) still come from the profile because they
    decide what the headers even ARE.
    """
    profile = profile or InstrumentProfile(name="_raw", vendor="", columns=())
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=profile.sheet or 0,
                             header=profile.header_row, nrows=nrows)
    return pd.read_csv(path, header=profile.header_row, decimal=profile.decimal,
                       encoding=profile.encoding, nrows=nrows)


def accumulate_restarts(t: np.ndarray) -> tuple[np.ndarray, int]:
    """A per-step clock -> one elapsed time. Returns `(t_s, how many restarts)`.

    A rig whose time column restarts at each protocol step writes 0, 1, 2, 0,
    1, 2 — six samples over six seconds, and the schema's `t_s` means elapsed
    time from the start of the experiment. Every downstream reader assumes
    that: pulse windows, relaxation tails, per-cycle grouping, and every plot
    with time on the x axis.

    A restart is a DECREASE, which is the only evidence the column carries. It
    is not evidence of a clock that restarts — a corrupted export decreases too
    — which is why this runs only when the profile SAYS the rig does this.
    Guessing from the numbers is how a corrupted file gets silently repaired.

    The boundary sample keeps the previous step's last time rather than gaining
    an invented gap: two samples at one instant is what the file says, and
    inserting a sampling interval would be making data up to make a plot
    tidier.
    """
    out = np.asarray(t, dtype=float).copy()
    offset = 0.0
    restarts = 0
    for i in range(1, len(out)):
        if np.isfinite(t[i]) and np.isfinite(t[i - 1]) and t[i] < t[i - 1]:
            offset += float(t[i - 1])
            restarts += 1
        out[i] = t[i] + offset
    return out, restarts


def _nulled(col: pd.Series, nulls: tuple[str, ...]) -> pd.Series:
    """The column as stripped text, with the profile's null tokens blanked.

    Shared by all three parse forms, because "a channel that was not connected
    reads `-` in every row" is a fact about the RIG and not about the format its
    times are written in. Split out of `_numeric`, which used to own it.
    """
    lowered = {t.strip().lower() for t in nulls}
    s = col.astype(str).str.strip()
    return s.where(~s.str.lower().isin(lowered), other="")


def _parse_duration_hms(s: pd.Series) -> np.ndarray:
    """`hh:mm:ss[.f]`, `mm:ss`, or a bare number of seconds -> seconds.

    All three, because one rig writes all three: Neware's `Test Time` is
    `hh:mm:ss.f` in some firmware versions and a decimal number of seconds in
    others, and a step shorter than an hour can come back as `mm:ss`. Accepting
    the bare number is what makes the rule safe to set on a profile whose
    export format varies — the alternative is a column of NaN on half the files
    and a profile nobody dares change.

    Read RIGHT to left, so `mm:ss` and `hh:mm:ss` need no branch: the last
    field is always seconds, the one before it minutes, the one before that
    hours. A negative sign on the whole value is honoured, because an export
    with a pre-trigger window has one.
    """
    text = s.astype(str).str.strip()
    out = np.full(len(text), np.nan, dtype=float)
    for i, raw in enumerate(text):
        if not raw or raw.lower() in ("nan", "none"):
            continue
        sign = -1.0 if raw.startswith("-") else 1.0
        parts = raw.lstrip("+-").split(":")
        try:
            fields = [float(p) for p in parts]
        except ValueError:
            continue
        total = 0.0
        for unit, value in zip((1.0, 60.0, 3600.0, 86400.0), reversed(fields)):
            total += unit * value
        out[i] = sign * total
    return out


def _parse_datetime(s: pd.Series) -> np.ndarray:
    """An absolute timestamp column -> seconds since its FIRST row.

    A wall-clock column is not an elapsed time, and the subtraction is the
    whole of the conversion. Anchored on the first row rather than on the
    minimum, deliberately: a file whose clock steps backwards is a file with a
    problem, and anchoring on the minimum would hide it by making the earliest
    sample the origin whatever its position.
    """
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.notna().sum() == 0:
        return np.full(len(parsed), np.nan, dtype=float)
    first = parsed.iloc[parsed.notna().to_numpy().argmax()]
    return ((parsed - first).dt.total_seconds()).to_numpy(dtype=float)


def _numeric(col: pd.Series, nulls: tuple[str, ...]) -> np.ndarray:
    """A column as floats, with the profile's null tokens becoming NaN.

    A channel that was not connected reads `-` in every row, and `float("-")`
    raises. Coercing it to NaN keeps the column present and empty, which is the
    honest shape: the rig HAS a temperature channel and it recorded nothing.
    """
    lowered = {t.strip().lower() for t in nulls}
    s = col.astype(str).str.strip()
    s = s.where(~s.str.lower().isin(lowered), other=np.nan)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def read_with_profile(path: Path, profile: InstrumentProfile
                      ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """`(frame, report)` — the mapped table, and everything the profile did not place.

    The report is the point. It names:

      `unplaced`   headers in the file that no rule claimed. NOT an error — an
                   export carries columns the schema has no home for — but the
                   engineer has to SEE them, because a column that should have
                   been mapped and was not looks identical to one that should
                   not have been.
      `missing`    rules whose source header is absent. This is how a profile
                   that is one character off announces itself instead of
                   quietly producing a frame with no temperature.
      `empty`      columns that mapped but hold no number at all — the
                   unconnected-probe case, which is different from absent.
      `unmapped_segments`  step labels the `seg_map` has no entry for.
    """
    return map_frame(raw_table(path, profile), profile)


def map_frame(raw: pd.DataFrame, profile: InstrumentProfile
              ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """`read_with_profile`, on a table somebody already read.

    Split from the file read so a caller holding a head of the file can re-map
    it without reading the file again. The file half is `raw_table` and this is
    everything after it, so a preview and the writer run the same mapping over
    different numbers of rows.
    """
    headers = [str(c) for c in raw.columns]
    by_header = {str(c): c for c in raw.columns}

    out: dict[str, Any] = {}
    claimed: set[str] = set()
    missing: list[str] = []
    empty: list[str] = []
    unmapped_segments: list[str] = []

    for rule in profile.columns:
        if rule.source not in by_header:
            missing.append(rule.source)
            continue
        claimed.add(rule.source)
        col = raw[by_header[rule.source]]
        if rule.target == "seg_type":
            labels = col.astype(str).str.strip()
            mapped = labels.str.lower().map(profile.seg_map)
            unknown = sorted({lab for lab, m in zip(labels, mapped)
                              if isinstance(m, float) or m is None})
            unmapped_segments.extend(u for u in unknown if u and u.lower() != "nan")
            # An unmapped label keeps its OWN text rather than becoming a
            # guess: a `seg_type` the schema does not know is visible, and a
            # wrong one that looks known is not.
            out["seg_type"] = np.asarray(mapped.fillna(labels), dtype=object)
            continue
        if rule.parse == "duration_hms":
            raw_vals = _parse_duration_hms(_nulled(col, profile.null_tokens))
        elif rule.parse == "datetime":
            raw_vals = _parse_datetime(_nulled(col, profile.null_tokens))
        else:
            raw_vals = _numeric(col, profile.null_tokens)
        # `scale`/`offset` apply AFTER the parse, so a rig that writes hh:mm:ss
        # in minutes is still expressible without a fourth parse form.
        vals = raw_vals * rule.scale + rule.offset
        if not np.isfinite(vals).any():
            empty.append(rule.source)
        out[rule.target] = vals

    # An all-null column is a spacer, not data the profile forgot — Landt's
    # `Unnamed: 13` is exactly this. Reporting it as unplaced would train the
    # reader to ignore the report.
    spacers = [h for h in headers
               if h not in claimed and raw[by_header[h]].isna().all()]
    unplaced = [h for h in headers if h not in claimed and h not in spacers]

    # The per-step clock, accumulated — but only where the PROFILE says this rig
    # restarts it. See `accumulate_restarts` for why this is not inferred.
    restarts = 0
    if profile.time_restarts_per_step and "t_s" in out:
        out["t_s"], restarts = accumulate_restarts(out["t_s"])

    frame = pd.DataFrame(out)
    # What is left going backwards AFTER any accumulation. On a profile with the
    # flag set this should be zero and a non-zero is a real anomaly; on one
    # without it, this is the whole of the detection: `write_experiment` rejects
    # NaNs in `t_s` but not a time column that runs backwards.
    backwards = 0
    if "t_s" in frame.columns and len(frame) > 1:
        backwards = int((frame["t_s"].diff() < 0).sum())

    return frame, {
        "profile": profile.name,
        "verified": profile.verified,
        "rows": int(len(frame)),
        "unplaced": unplaced,
        "spacers": spacers,
        "missing": missing,
        "empty": empty,
        "unmapped_segments": sorted(set(unmapped_segments)),
        # Reported rather than only acted on: an engineer who set the flag
        # wants to see it fire, and one who did not wants to see that it would
        # have.
        "time_restarts": restarts,
        "t_s_backwards": backwards,
    }


def detect(path: Path, library: dict[str, InstrumentProfile] | None = None
           ) -> list[tuple[str, int]]:
    """Profiles whose signature is a subset of this file's headers, best first.

    PROPOSES, never decides — the same discipline `read_misc` has. The caller
    shows the ranking and the engineer picks; a detector that chose silently
    would be a fourth way to get a wrong number with a confident face.

    Ranked by how many signature columns matched, so a file matching two
    profiles (they share `Voltage(V)`) puts the more specific one first.
    """
    library = BUILTIN if library is None else library
    try:
        # Only the header row is needed, and these files are tens of MB.
        suffix = path.suffix.lower()
        head = (pd.read_excel(path, nrows=0) if suffix in (".xlsx", ".xls")
                else pd.read_csv(path, nrows=0))
    except Exception:  # noqa: BLE001 — an unreadable file detects as nothing
        return []
    headers = {str(c).strip() for c in head.columns}
    hits = [(p.name, sum(1 for s in p.signature if s in headers))
            for p in library.values() if p.signature]
    return sorted([h for h in hits if h[1] == len(library[h[0]].signature)],
                  key=lambda h: -h[1])


# ---------------------------------------------------------------------------
# stored profiles — a rig's mapping lives in the store, beside its data
# ---------------------------------------------------------------------------

def profiles_dir(root: Path) -> Path:
    return Path(root) / "instruments"


def save_profile(profile: InstrumentProfile, root: Path) -> Path:
    """Write a profile into the store. Store-scoped: a rig serves every cell."""
    d = profiles_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{profile.name}.json"
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


def load_profiles(root: Path | None) -> dict[str, InstrumentProfile]:
    """The built-ins, overlaid with whatever this store has saved.

    Store-first on purpose: an engineer who corrected `neware` for their own
    machine must get THEIR version, not the unverified one this module ships.
    """
    out = dict(BUILTIN)
    if root is None:
        return out
    d = profiles_dir(root)
    if not d.exists():
        return out
    for path in sorted(d.glob("*.json")):
        try:
            p = InstrumentProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — a corrupt profile must not break ingest
            continue
        out[p.name] = p
    return out
