"""Instrument profiles: one parser, many cyclers, and nothing dropped quietly.

The fixture below is a **synthetic Landt export** built to carry every quirk a
real one has:

  * units in the header — `TestTime/h`, `Current/mA`, `Capacity/mAh`
  * an all-null spacer column between `SysTime` and `Cycle-Index`
  * a temperature channel that is present and entirely `-`
  * a step vocabulary of `R` / `RateC` / `RateD`
  * columns the schema has no home for (`SpeCap/mAh/g`, `Energy/mWh`, …)
  * a slice of a longer run: neither the record number nor the test time
    starts at zero
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monocell.schema.instruments import (BUILTIN, LANDT, ColumnRule, InstrumentProfile,
                                         detect, load_profiles, map_frame,
                                         read_with_profile, save_profile)


@pytest.fixture
def landt_file(tmp_path) -> Path:
    """A Landt-shaped xlsx carrying every quirk a real export has."""
    n = 60
    t_h = np.linspace(3.5, 3.5 + n * 0.01, n)
    state = np.array(["R"] * 10 + ["RateC"] * 25 + ["RateD"] * 25, dtype=object)
    current_ma = np.where(state == "RateC", 0.5,
                          np.where(state == "RateD", -0.5, 0.0))
    df = pd.DataFrame({
        "Record": np.arange(1201, 1201 + n),
        "TestTime/h": t_h,
        "StepTime/h": np.zeros(n),
        "Current/mA": current_ma,
        "Capacity/mAh": np.abs(np.linspace(0.0, 1.2, n)),
        "SpeCap/mAh/g": np.linspace(0.0, 180.0, n),
        "SOC|DOD/%": np.linspace(0.0, 100.0, n),
        "Voltage/V": np.linspace(4.09, 2.50, n),
        "Energy/mWh": np.linspace(0.0, 4.6, n),
        "SpeEnergy/Wh/kg": np.linspace(0.0, 700.0, n),
        "AuxTemp/dC.": ["-"] * n,          # the probe was not connected
        "AuxVolt/V": ["-"] * n,
        "SysTime": pd.date_range("2026-01-15 09:00:00", periods=n, freq="s"),
        "Unnamed: 13": [np.nan] * n,        # the spacer
        "Cycle-Index": np.full(n, 4),
        "Step-Index": np.full(n, 17),
        "Step-State": state,
    })
    path = tmp_path / "landt.xlsx"
    df.to_excel(path, index=False)
    return path


# ---------------------------------------------------------------------------
# the mapping
# ---------------------------------------------------------------------------

def test_the_units_in_the_header_are_converted_not_assumed(landt_file):
    """Hours, milliamps and milliamp-hours become seconds, amps and amp-hours.

    The failure this prevents is not a crash: a `Current/mA` column taken at
    face value gives a cell that looks like it charged at 0.5 A instead of
    0.5 mA, and every C-rate, every SOC and every derived resistance downstream
    is wrong by 1000x while remaining perfectly plausible in isolation.
    """
    frame, _ = read_with_profile(landt_file, LANDT)

    assert frame["t_s"].iloc[0] == pytest.approx(3.5 * 3600.0)
    assert frame["I_A"].max() == pytest.approx(0.5e-3)
    assert frame["I_A"].min() == pytest.approx(-0.5e-3)
    assert frame["q_Ah"].max() == pytest.approx(1.2e-3)
    assert frame["V_V"].iloc[0] == pytest.approx(4.09)   # already in volts


def test_the_step_vocabulary_is_translated(landt_file):
    """`R`/`RateC`/`RateD` are Landt's words; `rest`/`charge`/`discharge` are the
    schema's. Nothing downstream should have to know the first set."""
    frame, report = read_with_profile(landt_file, LANDT)
    assert set(frame["seg_type"]) == {"rest", "charge", "discharge"}
    assert report["unmapped_segments"] == []


def test_a_label_the_profile_cannot_translate_keeps_its_own_text(landt_file):
    """An unknown step label must stay visible, not become a guess.

    Mapping it to `rest` because rest is the safe default would put a charging
    segment into the rest bucket, where an OCV estimator would read its
    polarised voltage as an equilibrium one. Keeping the vendor's own word
    makes the gap findable; the report names it too.
    """
    df = pd.read_excel(landt_file)
    df.loc[df.index[:5], "Step-State"] = "CCCV_Chg"      # not in LANDT's seg_map
    df.to_excel(landt_file, index=False)

    frame, report = read_with_profile(landt_file, LANDT)
    assert "CCCV_Chg" in set(frame["seg_type"])
    assert report["unmapped_segments"] == ["CCCV_Chg"]


# ---------------------------------------------------------------------------
# nothing is dropped quietly — the rule this module exists to hold
# ---------------------------------------------------------------------------

def test_every_column_the_profile_did_not_place_is_reported(landt_file):
    """`read_cycler_csv` raises only when the TIME column is missing; every
    other column it cannot recognise simply does not arrive. That is how a rig
    whose header is one character off loses its temperature channel without
    anybody finding out.

    Three outcomes here, and they are three different facts:
      unplaced  the file has it and the schema has no home for it
      spacers   an all-null column, which is furniture and not data
      missing   the profile expects it and the file does not have it
    """
    _, report = read_with_profile(landt_file, LANDT)

    assert "SpeCap/mAh/g" in report["unplaced"]
    assert "Step-Index" in report["unplaced"]
    assert report["spacers"] == ["Unnamed: 13"], \
        "an all-null spacer must not be reported as a column the profile forgot"
    assert report["missing"] == []
    # every header in the file is accounted for exactly once
    headers = {str(c) for c in pd.read_excel(landt_file, nrows=0).columns}
    placed = {r.source for r in LANDT.columns}
    assert headers == placed | set(report["unplaced"]) | set(report["spacers"])


def test_a_profile_whose_header_is_one_character_off_says_so(landt_file):
    """The silent failure this report exists for. A profile that expects
    `Current/mA` against a file spelling it `Current/ma` must announce a
    missing column, not return a frame with no current in it."""
    bent = LANDT.clone("landt-bent", columns=tuple(
        ColumnRule("Current/ma", "I_A", scale=1e-3) if r.source == "Current/mA" else r
        for r in LANDT.columns))

    frame, report = read_with_profile(landt_file, bent)
    assert report["missing"] == ["Current/ma"]
    assert "I_A" not in frame.columns, "a missing source must not fabricate a column"


def test_an_unconnected_channel_is_empty_rather_than_zero(landt_file):
    """`AuxTemp/dC.` is present and entirely `-` when the probe is unplugged.

    Three wrong answers are available: crash on `float('-')`, drop the column
    (so the cell looks like it has no temperature channel at all), or coerce to
    0 degC (a lie that will be averaged into something). The column arrives,
    holds NaN, and the report says it is empty.
    """
    frame, report = read_with_profile(landt_file, LANDT)
    assert "T_degC" in frame.columns
    assert frame["T_degC"].isna().all()
    assert report["empty"] == ["AuxTemp/dC."]


# ---------------------------------------------------------------------------
# detection proposes
# ---------------------------------------------------------------------------

def test_detection_matches_on_the_signature(landt_file):
    assert [n for n, _ in detect(landt_file)] == ["landt"]


def test_detection_returns_nothing_rather_than_guessing(tmp_path):
    """A file matching no signature must produce an empty list, never a
    best-effort pick. The caller shows a ranking and the engineer chooses; a
    detector that decided silently would be a fourth way to get a wrong number
    with a confident face."""
    p = tmp_path / "mystery.csv"
    p.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    assert detect(p) == []
    assert detect(tmp_path / "does_not_exist.csv") == []


# ---------------------------------------------------------------------------
# verified vs not, and the store
# ---------------------------------------------------------------------------

def test_only_the_profile_built_from_real_files_claims_to_be_verified():
    """`verified` is a fact about evidence, and the difference is real: a
    profile written from published column names has never met a file, and a
    header spelled differently will not match. Every profile carries where its
    names came from."""
    assert LANDT.verified is True
    assert [p.name for p in BUILTIN.values() if p.verified] == ["landt"]
    for p in BUILTIN.values():
        assert p.evidence, f"{p.name} does not say where its column names came from"
        if not p.verified:
            assert "NOT checked against a file" in p.evidence


def test_a_stored_profile_wins_over_the_builtin_of_the_same_name(tmp_path):
    """An engineer who corrected `neware` for their own machine must get THEIR
    version. A built-in that silently outranked the correction would make the
    fix look applied and change nothing."""
    root = tmp_path / "data"
    fixed = BUILTIN["neware"].clone("neware", evidence="corrected against a real export")
    save_profile(fixed, root)

    loaded = load_profiles(root)
    assert loaded["neware"].evidence == "corrected against a real export"
    assert loaded["landt"].verified is True, "the other built-ins are still there"
    assert load_profiles(None)["neware"].evidence != "corrected against a real export"


def test_a_corrupt_stored_profile_does_not_break_ingest(tmp_path):
    """A half-written JSON file must cost its own profile and nothing else —
    ingest is the one path that has to keep working."""
    root = tmp_path / "data"
    save_profile(LANDT.clone("mine"), root)
    (root / "instruments" / "broken.json").write_text("{not json", encoding="utf-8")

    loaded = load_profiles(root)
    assert "mine" in loaded and "landt" in loaded and "broken" not in loaded


def test_a_profile_survives_a_json_round_trip():
    """A profile is data an engineer can save, send and edit — so it has to
    come back identical, `ColumnRule`s and all."""
    back = InstrumentProfile.from_dict(json.loads(json.dumps(LANDT.to_dict())))
    assert back == LANDT


# ---------------------------------------------------------------------------
# through the ingest path — a profile is only worth having if it is reachable
# ---------------------------------------------------------------------------

def _cycling_sidecar() -> dict:
    return {"protocol": {"description": "1C cycling"},
            "cell_state": {"rpt_index": 0, "capacity_ref_Ah": 1.0e-3},
            "instrument": {"sampling_rate_Hz": 1.0}}


def test_what_the_profile_could_not_place_reaches_the_experiments_own_record(landt_file, root):
    """The module's promise, kept where it matters.

    `read_with_profile` reporting an unplaced column is worth nothing if the
    report dies inside the reader. It has to land on the experiment, because
    the question it answers ("why has this run no temperature?") is asked
    months later by somebody who did not run the ingest.
    """
    from conftest import BUILD
    from monocell.cells import register_cell
    from monocell.schema.ingest import ingest_data

    register_cell("landt_cell", BUILD, root)
    d = ingest_data("landt_cell", "cycling", landt_file, _cycling_sidecar(), root,
                    instrument="landt")

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["ingest_report"]["profile"] == "landt"
    assert "SpeCap/mAh/g" in meta["ingest_report"]["unplaced"]

    flags = meta["quality"]["flags"]
    assert any("SpeCap/mAh/g" in f for f in flags), \
        "a column the profile did not place is not on the experiment's record"
    assert any("AuxTemp" in f and "empty" in f for f in flags), \
        "the unconnected probe reads as empty and the record does not say so"


def test_an_unverified_profile_says_so_on_every_file_it_reads(landt_file, root):
    """`verified` is a fact about the PROFILE, and the place it has to be
    visible is the data it produced. A profile written from a published column
    list that happens to parse cleanly is the failure that looks like success,
    so the flag is attached on the way in rather than left in the library for
    someone to go and look up."""
    from conftest import BUILD
    from monocell.cells import register_cell
    from monocell.schema.ingest import ingest_data, preview_ingest

    guess = LANDT.clone("landt_guess")          # clone() resets `verified`
    save_profile(guess, root)
    assert not guess.verified

    prev = preview_ingest("cycling", landt_file, _cycling_sidecar(),
                          instrument="landt_guess", root=root)
    assert any("UNVERIFIED" in f for f in prev["quality"]["flags"])

    # ...and the preview's flags ARE the stored flags, because both come from
    # the same report through the same function. A preview that showed one
    # thing and stored another would be worse than no preview.
    register_cell("guess_cell", BUILD, root)
    d = ingest_data("guess_cell", "cycling", landt_file, _cycling_sidecar(), root,
                    instrument="landt_guess")
    stored = json.loads((d / "meta.json").read_text(encoding="utf-8"))["quality"]["flags"]
    assert stored == prev["quality"]["flags"]


def test_an_unknown_profile_name_refuses_rather_than_falling_back(landt_file, root):
    """The forgiving behaviour is the wrong one. A fallback to the built-in
    synonym reader would let `--instrument landtt` parse by guess and report
    itself as a clean ingest — the engineer asked for a mapping and would get
    one that was never applied."""
    from monocell.schema.ingest import read_for_ingest

    with pytest.raises(ValueError) as exc:
        read_for_ingest("cycling", landt_file, "landtt", root)
    assert "landtt" in str(exc.value) and "landt" in str(exc.value), \
        "the refusal must name what was asked for and what is available"


def test_the_report_names_what_the_type_needs_and_did_not_get(landt_file):
    """Two facts about the TYPE rather than the file, computed in the reader so
    the preview and the write agree: a required column the mapping did not
    produce, and a produced column the type does not declare. The refusal
    itself stays in `write_experiment` — one place decides what may be stored —
    but a preview that could not see this coming would be a preview of a
    different ingest."""
    from monocell.schema.ingest import read_with_instrument

    no_voltage = LANDT.clone(
        "landt_no_v",
        columns=tuple(c for c in LANDT.columns if c.target != "V_V"))
    _, report = read_with_instrument("cycling", landt_file, no_voltage)
    assert report["required_missing"] == ["V_V"]
    assert report["not_in_schema"] == []

    # ...and the same file read as a type whose schema has no `cycle_index`
    _, as_gitt = read_with_instrument("gitt", landt_file, LANDT)
    assert "cycle_index" in as_gitt["not_in_schema"]


def test_two_rigs_two_shapes_one_stored_series(tmp_path, root):
    """The claim the whole package exists to make.

    The same logical experiment, exported by two machines that agree on
    nothing: different header spellings, hours against seconds, mA against A,
    `RateC`/`RateD` against `CC_Chg`/`CC_DChg`, and a column each that the
    other does not have. Two profiles, one parser, and the STORED series must
    be the same measurement.

    "The same" here is to within floating point and not bitwise: `1.1 mA`
    times `1e-3` and `0.0011 A` are the same number and not the same float.
    The tolerance below is 1e-9 relative, which is six orders of magnitude
    tighter than the smallest error this test exists to catch — a unit slip is
    1000x and an hours-for-seconds slip is 3600x. Nothing that matters can hide
    under it.

    No real vendor file is needed to make this claim, which is the point: it is
    a claim about the mapping layer, and it holds the same on the day a fourth
    machine arrives.
    """
    from conftest import BUILD
    from monocell.cells import register_cell
    from monocell.schema.ingest import ingest_data

    n = 40
    t_s = np.arange(n, dtype=float) * 60.0
    current_a = np.where(np.arange(n) < 20, 1e-3, -1e-3)
    volts = np.linspace(3.0, 4.2, n)
    q_ah = np.linspace(0.0, 1e-3, n)
    charging = np.arange(n) < 20

    rig_a = tmp_path / "rig_a.csv"
    pd.DataFrame({
        "TestTime/h": t_s / 3600.0,
        "Current/mA": current_a * 1e3,
        "Voltage/V": volts,
        "Capacity/mAh": q_ah * 1e3,
        "AuxTemp/dC.": np.full(n, 25.0),
        "Cycle-Index": np.ones(n, dtype=int),
        "Step-State": np.where(charging, "RateC", "RateD"),
        "SpeCap/mAh/g": np.linspace(0.0, 210.0, n),      # only rig A writes this
    }).to_csv(rig_a, index=False)

    rig_b = tmp_path / "rig_b.csv"
    pd.DataFrame({
        "Test_Time(s)": t_s,
        "Current(A)": current_a,
        "Voltage(V)": volts,
        "Capacity(Ah)": q_ah,
        "Aux_Temperature_1(C)": np.full(n, 25.0),
        "Cycle_Index": np.ones(n, dtype=int),
        "Step Type": np.where(charging, "CC_Chg", "CC_DChg"),
        "DataPoint": np.arange(n),                        # only rig B writes this
    }).to_csv(rig_b, index=False)

    profile_b = InstrumentProfile(
        name="rig_b", vendor="other",
        columns=(ColumnRule("Test_Time(s)", "t_s"),
                 ColumnRule("Current(A)", "I_A"),
                 ColumnRule("Voltage(V)", "V_V"),
                 ColumnRule("Capacity(Ah)", "q_Ah"),
                 ColumnRule("Aux_Temperature_1(C)", "T_degC"),
                 ColumnRule("Cycle_Index", "cycle_index"),
                 ColumnRule("Step Type", "seg_type")),
        seg_map={"cc_chg": "charge", "cc_dchg": "discharge"})
    save_profile(profile_b, root)

    register_cell("two_rigs", BUILD, root)
    sidecar = _cycling_sidecar()
    dir_a = ingest_data("two_rigs", "cycling", rig_a, sidecar, root, instrument="landt")
    dir_b = ingest_data("two_rigs", "cycling", rig_b, sidecar, root, instrument="rig_b")

    series_a = pd.read_parquet(dir_a / "series.parquet")
    series_b = pd.read_parquet(dir_b / "series.parquet")
    pd.testing.assert_frame_equal(series_a, series_b, check_like=True, rtol=1e-9)

    # ...and so is the per-cycle record each one produced, which is the number
    # anybody actually reads off a cycling file
    q_a = json.loads((dir_a / "quality.json").read_text(encoding="utf-8"))
    q_b = json.loads((dir_b / "quality.json").read_text(encoding="utf-8"))
    assert [r["cycle_index"] for r in q_a["per_cycle"]]         == [r["cycle_index"] for r in q_b["per_cycle"]]
    for row_a, row_b in zip(q_a["per_cycle"], q_b["per_cycle"]):
        for key in ("discharge_Ah", "charge_Ah", "discharge_A", "coulombic_efficiency"):
            assert row_a[key] == pytest.approx(row_b[key], rel=1e-9),                 f"{key} differs between the two rigs at cycle {row_a['cycle_index']}"

    # The two FILES are not identical, and each report says which is which —
    # otherwise this test would also pass on a parser that quietly ignored
    # both extra columns.
    meta_a = json.loads((dir_a / "meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((dir_b / "meta.json").read_text(encoding="utf-8"))
    assert meta_a["ingest_report"]["unplaced"] == ["SpeCap/mAh/g"]
    assert meta_b["ingest_report"]["unplaced"] == ["DataPoint"]


# ---------------------------------------------------------------------------
# a profile can express a FORMAT, not only a unit
# ---------------------------------------------------------------------------


def test_a_duration_column_parses_in_all_three_shapes_one_rig_writes():
    """`hh:mm:ss[.f]`, `mm:ss`, and a bare number of seconds — because one rig
    writes all three. Neware's `Test Time` is `hh:mm:ss.f` in some firmware
    versions and a decimal number of seconds in others, and a step shorter than
    an hour comes back as `mm:ss`.

    Accepting all three is what makes the rule safe to set on a profile no file
    has verified: the alternative is a column of NaN on half the exports and a
    profile nobody dares change. Unparseable cells become NaN rather than
    raising, which is the same treatment an unconnected channel gets.
    """
    from monocell.schema.instruments import _parse_duration_hms

    got = _parse_duration_hms(pd.Series(["01:02:03.5", "02:03", "7.25", "-00:00:10", "", "x"]))
    assert got[0] == pytest.approx(3723.5)
    assert got[1] == pytest.approx(123.0)
    assert got[2] == pytest.approx(7.25)
    assert got[3] == pytest.approx(-10.0)
    assert np.isnan(got[4]) and np.isnan(got[5])


def test_a_wall_clock_column_becomes_seconds_from_the_first_row():
    """An absolute timestamp is not an elapsed time, and the subtraction is the
    whole of the conversion.

    Anchored on the FIRST row rather than on the minimum, deliberately: a file
    whose clock steps backwards has a problem, and anchoring on the minimum
    would hide it by making the earliest sample the origin whatever its
    position — which is the `t_s` monotonicity check's business, not this
    function's to paper over.
    """
    from monocell.schema.instruments import _parse_datetime

    got = _parse_datetime(pd.Series(["2026-01-01 00:00:00", "2026-01-01 00:01:30",
                                     "2026-01-01 00:00:30"]))
    assert list(got) == [0.0, 90.0, 30.0]


def test_a_format_string_is_not_expressible_and_that_is_the_point():
    """The same argument `scale` makes: a strptime pattern here would be a
    second thing to be wrong about, and wrong in a way that turns a whole
    column into NaN without saying so. The vocabulary is closed and the refusal
    is at construction, where the name is still in hand."""
    from monocell.schema.instruments import PARSERS

    assert set(PARSERS) == {"number", "duration_hms", "datetime"}
    with pytest.raises(ValueError, match="is not one of"):
        ColumnRule("Test Time", "t_s", parse="%H:%M:%S")


def test_scale_still_applies_after_the_parse():
    """So a rig that writes `hh:mm:ss` meaning minutes is expressible without a
    fourth parse form. The two fields answer different questions — HOW the cell
    becomes a number, and what that number must be multiplied by — and keeping
    them independent is what stops the vocabulary growing one entry per rig."""
    profile = InstrumentProfile(
        name="t", vendor="t",
        columns=(ColumnRule("T", "t_s", parse="duration_hms", scale=60.0),))
    frame, _ = map_frame(pd.DataFrame({"T": ["00:00:02"]}), profile)
    assert frame["t_s"].iloc[0] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# a clock that restarts at each step
# ---------------------------------------------------------------------------


def test_a_per_step_clock_is_accumulated_only_when_the_profile_says_so():
    """The two cases look IDENTICAL in the numbers.

    `0, 1, 2, 0, 1, 2` is either a per-step clock (fine, and it has to be
    accumulated) or a corrupted export (not fine, and repairing it silently is
    the worst possible response). Only the rig knows which, and the profile is
    where what the rig does is written down — so the accumulation runs on the
    declaration and never on the shape of the data.

    Both directions, because a test of only the accumulating half would pass on
    an implementation that inferred it from the decrease.
    """
    raw = pd.DataFrame({"T": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]})
    rule = (ColumnRule("T", "t_s"),)

    declared = InstrumentProfile(name="a", vendor="v", columns=rule,
                                 time_restarts_per_step=True)
    frame, report = map_frame(raw, declared)
    assert list(frame["t_s"]) == [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]
    assert report["time_restarts"] == 1
    assert report["t_s_backwards"] == 0

    undeclared = InstrumentProfile(name="b", vendor="v", columns=rule)
    frame, report = map_frame(raw, undeclared)
    assert list(frame["t_s"]) == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0], (
        "the column was repaired on a profile that never said this rig restarts its clock"
    )
    assert report["time_restarts"] == 0
    assert report["t_s_backwards"] == 1


def test_the_boundary_sample_gains_no_invented_gap():
    """Two samples at one instant is what the file says. Inserting a sampling
    interval to make the series strictly increasing would be making data up to
    make a plot tidier."""
    frame, _ = map_frame(
        pd.DataFrame({"T": [0.0, 10.0, 0.0, 5.0]}),
        InstrumentProfile(name="a", vendor="v", columns=(ColumnRule("T", "t_s"),),
                          time_restarts_per_step=True))
    assert list(frame["t_s"]) == [0.0, 10.0, 10.0, 15.0]


# ---------------------------------------------------------------------------
# saying so, at the moment it matters
# ---------------------------------------------------------------------------


def test_naming_an_unverified_profile_is_reported_at_ingest(tmp_path, capsys):
    """The higher-risk of the two paths. `--instrument` means this profile
    decides how every column is read, and a mapping written from published
    header names can be wrong in a way that produces NUMBERS rather than an
    error — which is the one failure the rest of this platform cannot catch.

    The unverified built-ins have never met a file, and without this the
    ingest would go ahead without a word about that.
    """
    from monocell.cli import _warn_if_unverified

    _warn_if_unverified("neware", tmp_path / "data")
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out and "neware" in out
    assert "NOT checked against a file" in out, "the evidence itself is not quoted"
    assert "inspect --file" in out, "no way out is offered"


def test_naming_the_verified_profile_says_nothing(tmp_path, capsys):
    """Silence is the correct output for a profile checked against a real file.
    A note on every ingest would be noise, and noise is how a real warning stops
    being read."""
    from monocell.cli import _warn_if_unverified

    _warn_if_unverified("landt", tmp_path / "data")
    assert capsys.readouterr().out == ""


def test_a_corrected_profile_of_the_engineers_own_is_still_flagged(tmp_path, capsys):
    """`verified` resets on edit and stays reset — an engineer correcting the
    mapping for their own machine has not thereby verified it against a file,
    and the platform must not treat their edit as evidence."""
    from monocell.cli import _warn_if_unverified

    root = tmp_path / "data"
    save_profile(BUILTIN["neware"].clone("neware", evidence="corrected for our own cycler"),
                 root)
    _warn_if_unverified("neware", root)
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out and "corrected for our own cycler" in out


def test_a_detected_profile_says_which_kind_it_is(tmp_path, capsys):
    """Naming the matching profiles without saying whether anybody has checked
    them invites the reader to assume the first.

    The fixture file is written from the profile's OWN column rules, so this
    tests the reporting rather than the detection — which has its own tests
    above, and would otherwise make this one silently vacuous the first time a
    header string changed.
    """
    from monocell.cli import _propose_instrument

    profile = BUILTIN["neware"]
    heads = [rule.source for rule in profile.columns]
    assert set(profile.signature) <= set(heads), "the fixture header misses the signature"
    csv = tmp_path / "export.csv"
    csv.write_text(",".join(heads) + "\n" + ",".join("0" for _ in heads) + "\n",
                   encoding="utf-8")

    _propose_instrument(csv, tmp_path / "data")
    out = capsys.readouterr().out
    assert "neware" in out, "the profile whose own columns these are was not detected"
    assert "UNVERIFIED" in out


def test_the_ingest_command_is_what_prints_it(tmp_path, capsys):
    """The wiring, not the function. `_warn_if_unverified` could be perfect and
    never called, and then the ingest path would go ahead without a word about
    an unchecked reader.

    The file is deliberately unreadable, because the note is printed BEFORE the
    read and the claim is that it reaches the user whatever the read then does.
    """
    from monocell.cells import register_cell
    from monocell.cli import main

    root = tmp_path / "data"
    register_cell("c1", {"capacity_Ah": 5.0, "chemistry": "NMC"}, root)
    bad = tmp_path / "export.csv"
    bad.write_text("not,a,cycler,file\n1,2,3,4\n", encoding="utf-8")

    code = main(["ingest", "--cell", "c1", "--type", "cycling", "--file", str(bad),
                 "--instrument", "neware", "--data-root", str(root)])
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out and "neware" in out, (
        "the ingest command did not report that the named profile is unchecked")
    assert code == 1, "this fixture file should not have read cleanly"
