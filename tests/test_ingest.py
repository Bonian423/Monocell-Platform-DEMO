"""Real-file ingestors: one pipeline, whatever wrote the file.

The round-trip: build a frame -> write it out in a real instrument format
(cycler CSV / Palmsense4 CSV / EIS XLSX / pressure-rig CSV, each with its own
header spellings and units) -> ingest it as producer kind "real" -> the stored
series must be the frame that went out, and the ingest-side derivations
(protocol, instrument block, segment labels) must be what the file implies.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from monocell.cli import main as cli_main
from monocell.schema.ingest import ingest
from monocell.schema.write_experiment import load_experiment
from conftest import BUILD, eis_spectrum, register_cell

CELL = "real_cell"


@pytest.fixture
def real_cell(root):
    register_cell(CELL, BUILD, root)
    return CELL


def _sidecar(root, tmp_path, **extra):
    p = tmp_path / "sidecar.json"
    p.write_text(json.dumps({
        "protocol": {"description": "real instrument run"},
        "instrument": {"sampling_rate_Hz": 1.0},
        "cell_state": {"capacity_ref_Ah": 5.0, "rpt_rate": 0.05, "age_efc": 0.0},
        **extra,
    }), encoding="utf-8")
    return p


def _stored(d) -> pd.DataFrame:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return load_experiment(meta["experiment_id"], meta["cell_id"], d.parents[2])["series"]


# ---------------------------------------------------------------------------
# frames the tests build, so each one says exactly what went out
# ---------------------------------------------------------------------------

def _hppc_frame(pulse_s=10.0, relax_s=40.0, rest_s=10.0, hz=10.0, current_A=5.0,
                socs=(0.8, 0.5, 0.2)) -> pd.DataFrame:
    """Rest, pulse, relax at each state of charge, on one continuous clock."""
    parts, t0, dt = [], 0.0, 1.0 / hz
    for pulse_id, soc in enumerate(socs, start=1):
        ocv = 3.4 + 0.6 * soc
        for seg, dur, i in (("rest", rest_s, 0.0), ("pulse", pulse_s, current_A),
                            ("relax", relax_s, 0.0)):
            t = t0 + np.arange(0.0, dur, dt)
            if seg == "pulse":
                v = ocv - 0.004 * i - 0.002 * (1.0 - np.exp(-(t - t0) / 5.0))
            else:
                v = np.full(len(t), ocv)
            parts.append(pd.DataFrame({"t_s": np.round(t, 6), "I_A": i, "V_full_V": v,
                                       "T_degC": 25.0, "soc": soc, "pulse_id": pulse_id,
                                       "seg_type": seg}))
            t0 = float(t[-1]) + dt
    return pd.concat(parts, ignore_index=True)


def _hppc_csv(tmp_path, frame: pd.DataFrame):
    """The frame under one rig's header spellings."""
    csv = tmp_path / "hppc_export.csv"
    frame.rename(columns={
        "t_s": "time/s", "I_A": "Current [A]", "V_full_V": "Voltage [V]",
        "T_degC": "Temperature [C]", "soc": "SOC", "seg_type": "step",
    }).to_csv(csv, index=False)
    return csv


def _gitt_frame(n_pulses=6, pulse_s=600.0, relax_s=1800.0, current_A=0.25) -> pd.DataFrame:
    """An opening rest, then `n_pulses` pulse/relax pairs, sampled every 10 s."""
    segs = [("relax", relax_s, 0.0, 0)]
    for k in range(1, n_pulses + 1):
        segs += [("pulse", pulse_s, current_A, k), ("relax", relax_s, 0.0, k)]
    parts, t0, v = [], 0.0, 3.9
    for seg, dur, i, pulse_id in segs:
        t = t0 + np.arange(0.0, dur, 10.0)
        if seg == "pulse":
            volts = v - 0.02 - 1e-5 * (t - t0)
            v -= 0.01
        else:
            volts = v - 0.005 * np.exp(-(t - t0) / 300.0)
        parts.append(pd.DataFrame({"t_s": t, "I_A": i, "V_V": volts, "T_degC": 25.0,
                                   "soc": 0.9 - 0.02 * pulse_id, "pulse_id": pulse_id,
                                   "seg_type": seg}))
        t0 = float(t[-1]) + 10.0
    return pd.concat(parts, ignore_index=True)


def _gitt_export(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={"t_s": "time/s", "I_A": "Current [A]", "V_V": "Voltage [V]",
                                 "T_degC": "Temperature [C]", "soc": "SOC"})


# ---------------------------------------------------------------------------
# cycler CSV
# ---------------------------------------------------------------------------

def test_cycler_csv_hppc_roundtrip(real_cell, root, tmp_path):
    """Vendor headers in, schema columns out, and the numbers untouched."""
    frame = _hppc_frame()
    sidecar = _sidecar(root, tmp_path, instrument={"sampling_rate_Hz": 10.0}, protocol={
        "description": "real instrument run", "pulse_current_A": 5.0, "pulse_duration_s": 10.0,
        "rest_rule": "5*tau2"}, cell_state={
        "capacity_ref_Ah": 5.0, "age_efc": 0.0, "soc_basis": "coulomb_measured"})
    d = ingest(CELL, "hppc", _hppc_csv(tmp_path, frame), sidecar, root)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["producer"]["kind"] == "real"

    stored = _stored(d)
    for col in ("t_s", "I_A", "V_full_V", "T_degC", "soc", "pulse_id"):
        np.testing.assert_allclose(stored[col].to_numpy(dtype=float),
                                   frame[col].to_numpy(dtype=float), rtol=1e-12, err_msg=col)
    assert list(stored["seg_type"]) == list(frame["seg_type"])
    assert not [f for f in meta["quality"]["flags"] if f.startswith("rule:")]


def test_cycler_csv_hppc_bare_sidecar_derives_protocol(real_cell, root, tmp_path):
    """The pulse current and duration live in the CSV, not the sidecar. A real
    ingest with a bare sidecar must still record them, because anything that
    analyses the pulses needs `protocol.pulse_current_A` and
    `protocol.pulse_duration_s`: ingest fills them from the pulse segments."""
    frame = _hppc_frame(pulse_s=10.0, current_A=5.0)
    d = ingest(CELL, "hppc", _hppc_csv(tmp_path, frame), _sidecar(root, tmp_path), root)
    proto = json.loads((d / "meta.json").read_text(encoding="utf-8"))["protocol"]
    assert proto["pulse_current_A"] == pytest.approx(5.0, rel=1e-9)
    assert proto["pulse_duration_s"] == pytest.approx(10.0, rel=1e-9)


def test_cli_ingest(real_cell, root, tmp_path, capsys):
    sidecar = _sidecar(root, tmp_path, protocol={
        "description": "real instrument run", "pulse_current_A": 5.0, "pulse_duration_s": 10.0,
    })
    rc = cli_main(["ingest", "--cell", CELL, "--type", "hppc",
                   "--file", str(_hppc_csv(tmp_path, _hppc_frame())),
                   "--sidecar", str(sidecar), "--data-root", str(root)])
    assert rc == 0
    assert "ingested hppc ->" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# EIS XLSX
# ---------------------------------------------------------------------------

def test_eis_xlsx_roundtrip(real_cell, root, tmp_path):
    """The -Z'' convention is undone on the way in, and the linear-KK check at
    ingest sees exactly the spectrum that went out."""
    from monocell.schema.quality import lin_kk

    f, z = eis_spectrum(noise=1e-3, seed=3)
    xlsx = tmp_path / "real_eis.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"freq/Hz": f, "Zre/Ohm": z.real, "-Zim/Ohm": -z.imag}).to_excel(
            w, sheet_name="spectrum", index=False)
        pd.DataFrame([("soc", 0.5), ("v_dc_v", 3.7), ("t_degc", 25.0),
                      ("electrode", "full")]).to_excel(w, sheet_name="meta", index=False)
    sidecar = _sidecar(root, tmp_path, instrument={
        "potentiostat": "potentiostat-01", "amplitude_mV": 5.0, "f_min_Hz": 0.01,
        "f_max_Hz": 1e4, "points_per_decade": 10})
    d = ingest(CELL, "eis", xlsx, sidecar, root)

    stored = _stored(d)
    # an XLSX cell holds a double, so the round trip is exact to ~15 digits
    np.testing.assert_allclose(stored["Z_re_Ohm"], z.real, rtol=1e-12)
    np.testing.assert_allclose(stored["Z_im_Ohm"], z.imag, rtol=1e-12)
    assert (stored["Z_im_Ohm"] < 0).all(), "the sign convention was not undone"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["quality"]["kk_residual_pct"] == pytest.approx(
        lin_kk(f, z.real, z.imag)["kk_residual_pct"], rel=1e-6)


# ---------------------------------------------------------------------------
# pressure-rig CSV
# ---------------------------------------------------------------------------

def test_pressure_csv_roundtrip(real_cell, root, tmp_path):
    n = 600
    t = np.arange(n, dtype=float)
    soc = 0.2 + 0.6 * t / n
    force = 350.0 + 40.0 * soc
    frame = pd.DataFrame({
        "t_s": t, "F_N": force, "p_kPa": force / 13.5, "thickness_um": 5000.0 + 60.0 * soc,
        "T_degC": 25.0, "V_V": 3.4 + 0.8 * soc, "I_A": np.where(t < 100, 0.0, 5.0),
        "soc": soc, "seg_type": np.where(t < 100, "rest", "charge"),
    })
    csv = tmp_path / "rig.csv"
    frame.rename(columns={"t_s": "time/s", "F_N": "Force [N]", "p_kPa": "Pressure [kPa]",
                          "T_degC": "Temperature [C]", "V_V": "Voltage [V]", "I_A": "Current [A]",
                          "seg_type": "step"}).to_csv(csv, index=False)
    sidecar = _sidecar(root, tmp_path, instrument={
        "load_cell_id": "LC-01", "calibration_date": "2026-01-15",
        "fixture_stiffness_N_per_um": 5.0, "creep_hold_protocol": "standard",
        "sampling_rate_Hz": 1.0, "ntp_synced": True})
    d = ingest(CELL, "pressure", csv, sidecar, root)

    stored = _stored(d)
    for col in ("t_s", "F_N", "p_kPa", "thickness_um", "T_degC", "V_V", "I_A", "soc"):
        np.testing.assert_allclose(stored[col].to_numpy(dtype=float),
                                   frame[col].to_numpy(dtype=float), rtol=1e-12, err_msg=col)
    assert list(stored["seg_type"]) == list(frame["seg_type"])


# gitt's TableSpec.instrument is a REQUIRED-key set, so a real sidecar must
# carry both of these.
_GITT_INSTRUMENT = {"sampling_rate_Hz": 0.1, "pulse_rate": 0.05,
                    "relax_criterion": "rest until dV/dt < 0.1 mV/h"}


# --- raw Palmsense4 CSV: the two real layouts -------------------------------
#
# An in-situ export has a mixed-mode GITT section and Edc on every block. A
# real file is not always that: a standalone EIS run has no mixed-mode section
# at all, and some exports omit the Edc column, leaving the GITT correlation as
# the only voltage source. This writer covers the matrix.

def _write_ps4(path, spectra, voltages, *, mixed_mode=True, with_edc=True,
               amplitude_mV=5.0, pulse_uA=-1000.0):
    """A minimal raw Palmsense4 CSV (UTF-16-LE, BOM) with switchable sections."""
    lines = ["Palmsense4 test export"]

    def fmt(x):
        return f"{float(x):.8g}"

    if mixed_mode:
        lines.append(",".join(["s,V,s,µA"] * (2 * len(spectra))))
        for row_t in (0.0, 60.0):
            cells = []
            for v in voltages:
                cells += [row_t, v, row_t, 0.0]                 # OCV step
                cells += [row_t, v, row_t, pulse_uA]            # negative pulse
            lines.append(",".join(fmt(x) for x in cells))
        lines.append("")

    for k, df in enumerate(spectra):
        lines.append(f"Measurement:,Impedance Spectroscopy [{k}]")
        lines.append(f"Eac: {amplitude_mV} mV")
        header = "freq / Hz,Z' / Ohm,Z'' / Ohm" + (",Edc / V" if with_edc else "")
        lines.append(header)
        for i in range(len(df)):
            row = [df["f_Hz"].iloc[i], df["Z_re_Ohm"].iloc[i], -df["Z_im_Ohm"].iloc[i]]
            if with_edc:
                row.append(voltages[k])
            lines.append(",".join(fmt(x) for x in row))
        lines.append("")

    path.write_bytes(("﻿" + "\r\n".join(lines) + "\r\n").encode("utf-16-le"))
    return path


def _ps4_frames(n=61, f_min=0.01, f_max=1e4, scale=1.0):
    """`n`-point RC-like spectrum frames (one per entry) for the writer above."""
    f = np.geomspace(f_min, f_max, n)
    w = 2.0 * np.pi * f
    z = scale * (2e-3 + 1e-3 / (1 + 1j * w * 1e-3) + 5e-4 / (1 + 1j * w * 3.0))
    return pd.DataFrame({"f_Hz": f, "Z_re_Ohm": z.real, "Z_im_Ohm": z.imag})


def test_palmsense4_standalone_csv_derives_instrument_and_mode(real_cell, root, tmp_path):
    """A standalone Palmsense4 file: no mixed-mode section, Edc on the block.

    The instrument block is derived from the file, the block's own Edc is the
    DC bias, and the stored frame is the spectrum that went out.
    """
    from monocell.schema.ingest import palmsense4_context, read_palmsense4_csv

    frame = _ps4_frames()
    csv = _write_ps4(tmp_path / "standalone.csv", [frame], [3.7], mixed_mode=False)
    back = read_palmsense4_csv(csv)
    assert len(back) == len(frame) == 61
    assert set(back["spectrum_id"]) == {"EIS_0"}  # the file's own block index
    assert back["V_dc_V"].iloc[0] == pytest.approx(3.7)  # Edc
    # the instrument's Z'' column is negated on the way in (the schema's -Z''
    # convention), so the frame round-trips with its sign intact
    assert back["Z_im_Ohm"].iloc[0] == pytest.approx(frame["Z_im_Ohm"].iloc[0], rel=1e-7)
    assert frame["Z_im_Ohm"].iloc[0] < 0.0
    assert back["Z_mag_Ohm"].iloc[0] == pytest.approx(
        np.hypot(back["Z_re_Ohm"].iloc[0], back["Z_im_Ohm"].iloc[0]))

    ctx = palmsense4_context(csv)
    assert ctx["eis_mode"] == "standalone"
    assert ctx["n_spectra"] == 1 and ctx["spectrum_ids"] == ["EIS_0"]
    assert "mixed_mode_steps" not in ctx and "correlations" not in ctx

    d = ingest(CELL, "eis", csv, None, root)  # a real CSV has no sidecar
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    inst, proto = meta["instrument"], meta["protocol"]
    assert proto["eis_mode"] == "standalone" and proto["spectrum_ids"] == ["EIS_0"]
    assert inst["potentiostat"] == "Palmsense4"      # from the format
    assert inst["amplitude_mV"] == pytest.approx(5.0)  # from the Eac line
    # the measured grid, straight off the data
    assert inst["f_min_Hz"] == pytest.approx(1e-2, rel=1e-6)
    assert inst["f_max_Hz"] == pytest.approx(1e4, rel=1e-6)
    assert inst["points_per_decade"] == pytest.approx(10.0, rel=1e-6)
    np.testing.assert_allclose(_stored(d)["Z_re_Ohm"], frame["Z_re_Ohm"], rtol=1e-7)


def test_palmsense4_correlation_supplies_v_dc_without_edc(real_cell, root, tmp_path):
    """The in-situ layout with NO Edc column: the correlation is the only
    voltage source, and it must still name and bias every spectrum."""
    from monocell.schema.ingest import palmsense4_context, read_palmsense4_csv

    volts = [3.0, 3.8]
    csv = _write_ps4(tmp_path / "insitu_no_edc.csv", [_ps4_frames(scale=1 + 0.1 * k) for k in range(2)],
                     volts, with_edc=False)
    back = read_palmsense4_csv(csv)
    assert sorted(set(back["spectrum_id"])) == ["EIS_3.0000V", "EIS_3.8000V"]
    by_id = {sid: float(grp["V_dc_V"].iloc[0]) for sid, grp in back.groupby("spectrum_id")}
    assert by_id == {"EIS_3.0000V": pytest.approx(3.0), "EIS_3.8000V": pytest.approx(3.8)}

    ctx = palmsense4_context(csv)
    assert ctx["eis_mode"] == "insitu"
    assert ctx["spectrum_ids"] == ["EIS_3.0000V", "EIS_3.8000V"]
    # the pairing: EIS k sits at the END voltage of the step before the k-th
    # negative-current step (here step 0 of each spectrum's pair)
    assert [c["step"] for c in ctx["correlations"]] == [0, 2]
    assert [c["eis_index"] for c in ctx["correlations"]] == ["0", "1"]
    steps = ctx["mixed_mode_steps"]
    assert [s["step"] for s in steps] == [0, 1, 2, 3]
    assert [s["i_mean_uA"] < 0 for s in steps] == [False, True, False, True]
    # ...and that step's end voltage IS the block's bias
    assert [steps[c["step"]]["v_end_V"] for c in ctx["correlations"]] == [3.0, 3.8]


def test_palmsense4_sidecar_wins_on_the_keys_it_provides(real_cell, root, tmp_path):
    """Derivation fills gaps; it never overrules the operator's sidecar."""
    csv = _write_ps4(tmp_path / "sidecar_wins.csv", [_ps4_frames()], [3.7], mixed_mode=False)
    sidecar = _sidecar(root, tmp_path, instrument={
        "potentiostat": "potentiostat-01", "amplitude_mV": 10.0, "f_min_Hz": 0.05,
    })
    d = ingest(CELL, "eis", csv, sidecar, root)
    inst = json.loads((d / "meta.json").read_text(encoding="utf-8"))["instrument"]
    assert inst["potentiostat"] == "potentiostat-01"   # provided -> kept
    assert inst["amplitude_mV"] == pytest.approx(10.0)
    assert inst["f_min_Hz"] == pytest.approx(0.05)    # even though the data says 0.01
    assert inst["f_max_Hz"] == pytest.approx(1e4, rel=1e-6)   # missing -> derived
    assert inst["points_per_decade"] == pytest.approx(10.0, rel=1e-6)


def test_palmsense4_sidecar_protocol_keys_survive_the_file_context_merge(real_cell, root,
                                                                         tmp_path):
    """The file-format context (eis_mode, spectrum ids) merges INTO the
    sidecar's protocol dict. Whatever the operator recorded there must still be
    there afterwards: a merge that replaced the dict would drop it silently."""
    csv = _write_ps4(tmp_path / "manual.csv", [_ps4_frames()], [3.7], mixed_mode=False)
    sidecar = _sidecar(root, tmp_path, protocol={
        "description": "real instrument run", "operator": "tech-01",
        "fixture": {"holder": "B", "torque_Nm": 0.4},
    })
    d = ingest(CELL, "eis", csv, sidecar, root)
    proto = json.loads((d / "meta.json").read_text(encoding="utf-8"))["protocol"]
    assert proto["operator"] == "tech-01"
    assert proto["fixture"] == {"holder": "B", "torque_Nm": 0.4}
    # ...and the file-derived context arrived beside them
    assert proto["eis_mode"] == "standalone"


# ---------------------------------------------------------------------------
# GITT (a cycler CSV of pulses and rests)
# ---------------------------------------------------------------------------

def test_gitt_csv_roundtrip_labelled(real_cell, root, tmp_path):
    """A cycler export with a real step column, round-tripped through ingest.

    The labeller must fold the instrument's own wording ("CC Discharge",
    "Rest") onto the schema's pulse|relax vocabulary, and the pulse index is
    derived from it: GITT's whole structure is which segment is which.
    """
    frame = _gitt_frame(n_pulses=6)
    export = _gitt_export(frame)
    labels = {"pulse": "CC Discharge", "relax": "Rest"}
    export["Step Type"] = [labels[s] for s in export["seg_type"]]
    csv = tmp_path / "real_gitt.csv"
    export.drop(columns=["seg_type", "pulse_id"]).to_csv(csv, index=False)

    d = ingest(CELL, "gitt", csv, _sidecar(root, tmp_path, instrument=_GITT_INSTRUMENT), root)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["producer"]["kind"] == "real"

    stored = _stored(d)
    assert list(stored["seg_type"]) == list(frame["seg_type"])
    assert list(stored["pulse_id"]) == list(frame["pulse_id"]), "the pulse index was not recovered"
    np.testing.assert_allclose(stored["V_V"], frame["V_V"], rtol=1e-12)


def test_gitt_csv_roundtrip_bare(real_cell, root, tmp_path):
    """No step column at all: the pulses must be inferred from current."""
    from monocell.schema.ingest import read_gitt

    frame = _gitt_frame(n_pulses=20)
    csv = tmp_path / "bare_gitt.csv"
    _gitt_export(frame).drop(columns=["seg_type", "pulse_id"]).to_csv(csv, index=False)
    df = read_gitt(csv)
    assert set(df["seg_type"]) == {"pulse", "relax"}
    assert df["pulse_id"].max() == 20
    assert (df.loc[df["seg_type"] == "pulse", "I_A"].abs() > 0).all()
    assert (df.loc[df["seg_type"] == "relax", "I_A"].abs() == 0).all()


def test_gitt_csv_rejects_an_unrecognised_step_label(real_cell, root, tmp_path):
    from monocell.schema.ingest import read_gitt

    export = _gitt_export(_gitt_frame(n_pulses=4))
    export["Step Type"] = "Constant Voltage Hold"  # neither a pulse nor a relax
    csv = tmp_path / "bad_gitt.csv"
    export.drop(columns=["seg_type", "pulse_id"]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="neither a pulse nor a relax"):
        read_gitt(csv)


# ---------------------------------------------------------------------------
# instrument profiles, from the command line
# ---------------------------------------------------------------------------

def _landt_csv(path, n=40, cycles=1):
    """A Landt-shaped export of `cycles` charge/discharge cycles."""
    charging = np.tile(np.arange(n) < n // 2, cycles)
    total = n * cycles
    pd.DataFrame({
        "TestTime/h": np.arange(total, dtype=float) * 60.0 / 3600.0,
        "Current/mA": np.where(charging, 0.5, -0.5),
        "Voltage/V": np.tile(np.linspace(3.0, 4.2, n), cycles),
        "Capacity/mAh": np.tile(np.linspace(0.0, 0.5, n), cycles),
        "AuxTemp/dC.": np.full(total, 25.0),
        "Cycle-Index": np.repeat(np.arange(1, cycles + 1), n),
        "Step-State": np.where(charging, "RateC", "RateD"),
        "SpeCap/mAh/g": np.tile(np.linspace(0.0, 180.0, n), cycles),
    }).to_csv(path, index=False)
    return path


def test_cli_ingest_through_an_instrument_profile(real_cell, root, tmp_path, capsys):
    """`--instrument` is the whole reason the profile library is reachable from
    a terminal, and the flags it produces are printed rather than left in
    `quality.json` for somebody to go and read. A flag nobody sees is a flag
    that does not exist."""
    csv = _landt_csv(tmp_path / "landt_cycling.csv")
    sidecar = _sidecar(root, tmp_path)

    rc = cli_main(["ingest", "--cell", CELL, "--type", "cycling", "--file", str(csv),
                   "--sidecar", str(sidecar), "--data-root", str(root),
                   "--instrument", "landt"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ingested cycling ->" in out
    assert "SpeCap/mAh/g" in out, \
        "the column the profile did not place has to reach the person who ran the ingest"

    # the units came from the profile, not from a guess: 0.5 mA, not 0.5 A
    d = sorted((root / "experiments" / CELL).glob("cycling_*"))[-1]
    series = pd.read_parquet(d / "series.parquet")
    assert series["I_A"].abs().max() == pytest.approx(0.5e-3, rel=1e-9)


def test_cli_ingest_without_a_profile_names_the_ones_that_match(real_cell, root, tmp_path,
                                                                capsys):
    """Detection PROPOSES and never decides.

    Without `--instrument` the read goes ahead by the built-in synonym list,
    and the file's matching profiles are named on the way past. A detector that
    switched readers for you would be one more way to get a wrong number with a
    confident face, and the engineer would have no way to tell which reader
    produced it.

    On a Landt export the synonym read then FAILS, which is worth asserting:
    `TestTime/h` is not in the synonym list, so there is no time column and
    `read_cycler_csv` raises. That is the right outcome (a file it cannot read
    is better than a file it half-reads), and it is why the note is printed
    BEFORE the read rather than after it. The sentence an engineer needs is
    then directly above the error that sent them looking for it.
    """
    csv = _landt_csv(tmp_path / "landt_cycling.csv")
    sidecar = _sidecar(root, tmp_path)

    assert cli_main(["ingest", "--cell", CELL, "--type", "cycling", "--file", str(csv),
                     "--sidecar", str(sidecar), "--data-root", str(root)]) == 1

    out = capsys.readouterr().out
    assert "no recognizable time column" in out
    assert "landt" in out and "--instrument landt" in out, \
        "the profile that would have read this file was not named"
    assert not list((root / "experiments" / CELL).glob("cycling_*")), \
        "detection proposed and the reader still refused — nothing should be stored"


def test_an_unknown_instrument_name_stops_the_ingest(real_cell, root, tmp_path, capsys):
    """Before the write, not after it. The store is append-only, so an
    experiment written under a reader the engineer did not ask for cannot be
    taken back — only superseded."""
    csv = _landt_csv(tmp_path / "landt_cycling.csv")
    sidecar = _sidecar(root, tmp_path)

    assert cli_main(["ingest", "--cell", CELL, "--type", "cycling", "--file", str(csv),
                     "--sidecar", str(sidecar), "--data-root", str(root),
                     "--instrument", "no_such_rig"]) == 1
    out = capsys.readouterr().out
    assert "no_such_rig" in out and "Traceback" not in out
    assert not list((root / "experiments" / CELL).glob("cycling_*"))


# ---------------------------------------------------------------------------
# folder ingest — forty files in a directory is what an aging campaign IS
# ---------------------------------------------------------------------------

def _campaign(d, n_files: int = 2, cycles: int = 4):
    """A directory of Landt-shaped cycling exports, plus one file that is not."""
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n_files):
        _landt_csv(d / f"cell_{k}.csv", cycles=cycles)
    (d / "notes.txt").write_text("check the fridge\n", encoding="utf-8")
    return d


def test_a_folder_ingest_prints_its_plan_and_writes_nothing_without_confirm(
        real_cell, root, tmp_path, capsys):
    """The deliberate inconsistency with single-file ingest, and the reason for it.

    The store is append-only: a wrong ingest is corrected by writing a second
    experiment that supersedes it, and that cost scales with the number of
    files. One mistake is a correction; forty is an afternoon. So the plan is
    printed and the confirmation is a word.
    """
    campaign = _campaign(tmp_path / "campaign")

    assert cli_main(["ingest", "--cell", CELL, "--dir", str(campaign),
                     "--data-root", str(root)]) == 0
    out = capsys.readouterr().out

    assert "plan for 3 file(s)" in out
    assert out.count("cycling") >= 2
    assert "--confirm" in out
    assert not list((root / "experiments" / CELL).glob("cycling_*")), \
        "the plan wrote experiments; it must write nothing"


def test_confirming_writes_one_experiment_per_readable_file_and_names_the_others(
        real_cell, root, tmp_path, capsys):
    """The point of a folder scan is to come back with an account of the WHOLE
    directory. A scan that stopped at the text file somebody left in it would be
    a scan nobody could run on a real directory."""
    campaign = _campaign(tmp_path / "campaign", n_files=3)

    assert cli_main(["ingest", "--cell", CELL, "--dir", str(campaign),
                     "--data-root", str(root), "--confirm"]) == 0
    out = capsys.readouterr().out

    written = sorted((root / "experiments" / CELL).glob("cycling_*"))
    assert len(written) == 3, f"expected one experiment per readable file, got {written}"
    assert "3 of 3 file(s) ingested" in out
    assert "SKIP  notes.txt" in out, "the file it could not read has to be named"

    # and each one went through the profile, not the synonym guess: 0.5 mA
    for d in written:
        series = pd.read_parquet(d / "series.parquet")
        assert series["I_A"].abs().max() == pytest.approx(0.5e-3, rel=1e-9)


def test_a_file_whose_shape_fits_two_types_is_skipped_rather_than_chosen_for(
        real_cell, root, tmp_path, capsys):
    """`propose_type` refuses to choose between a GITT pulse and an HPPC one
    when the evidence does not separate them. Choosing here instead would not
    remove that refusal, it would only move it somewhere nobody can see it — so
    the row is skipped, the ambiguity is printed, and `--type` is offered."""
    campaign = tmp_path / "ambiguous"
    campaign.mkdir()
    # Four 30-minute segments, so it reads as a pulse TRAIN and not as two
    # legs — the family has to be the pulse one for the two rules to be able
    # to disagree at all.
    n = 4000
    phase = (np.arange(n) // 500) % 2
    pd.DataFrame({
        "time_s": np.arange(float(n)) * 3.6,        # 30-minute segments -> GITT
        "current_A": np.where(phase == 0, -BUILD["capacity_Ah"], 0.0),   # ...at 1C -> HPPC
        "voltage_V": np.full(n, 3.7),
    }).to_csv(campaign / "puzzle.csv", index=False)

    assert cli_main(["ingest", "--cell", CELL, "--dir", str(campaign),
                     "--data-root", str(root), "--confirm"]) == 0
    out = capsys.readouterr().out

    assert "SKIP  puzzle.csv" in out
    assert "gitt" in out and "hppc" in out and "--type" in out
    assert not list((root / "experiments" / CELL).glob("*")), "an ambiguous file was ingested"

    # ...and naming the type settles it, which is what the message said to do.
    # Asserted on the PLAN rather than on a write: this file is four square
    # pulses and nothing else, so the writer would rightly refuse it for the
    # physical columns a GITT needs. What is under test here is which row the
    # planner skips, and `--type` is what stops it skipping this one.
    assert cli_main(["ingest", "--cell", CELL, "--dir", str(campaign), "--type", "gitt",
                     "--data-root", str(root)]) == 0
    planned = capsys.readouterr().out
    assert "SKIP" not in planned
    assert "gitt       puzzle.csv" in planned


def test_ingesting_one_named_file_still_needs_its_type(real_cell, root, tmp_path, capsys):
    """`--dir` can read the type off each file's shape. One named file has only
    the name, so `--type` stays required there — and the refusal points at the
    command that would answer the question."""
    csv = _landt_csv(tmp_path / "run.csv")

    assert cli_main(["ingest", "--cell", CELL, "--file", str(csv),
                     "--data-root", str(root)]) == 2
    out = capsys.readouterr().out
    assert "--type is required" in out and "monocell inspect" in out


def test_the_capacity_reference_comes_from_the_cell_and_not_from_a_literal(
        root, tmp_path):
    """A file that arrives without `capacity_ref_Ah` in its sidecar gets the
    cell's own capacity from its build record.

    A default sized for one cell would be wrong by orders of magnitude on
    another, and nothing about the result would LOOK wrong: the SOH would be a
    small number, small numbers are what a degraded cell has, and the
    arithmetic would be internally consistent.
    """
    from monocell.schema.ingest import ingest_data

    register_cell("coin", {"capacity_Ah": 5.0e-4, "chemistry": "NMC811/graphite (coin)"}, root)
    d = ingest_data("coin", "cycling", _landt_csv(tmp_path / "coin.csv"),
                    {"protocol": {"description": "1C cycling"},
                     "instrument": {"sampling_rate_Hz": 1.0}},
                    root, instrument="landt")

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["cell_state"]["capacity_ref_Ah"] == pytest.approx(5.0e-4), \
        "the reference is not this cell's capacity"
    assert 0.0 < meta["quality"]["capacity_SOH"] < 2.0, \
        "the SOH is not on the scale a state of health lives on"

    # ...and a sidecar that names one still wins, because somebody who measured
    # the reference knows more than the nameplate does
    d2 = ingest_data("coin", "cycling", _landt_csv(tmp_path / "coin2.csv"),
                     {"protocol": {"description": "1C cycling"},
                      "instrument": {"sampling_rate_Hz": 1.0},
                      "cell_state": {"capacity_ref_Ah": 4.6e-4}},
                     root, instrument="landt")
    meta2 = json.loads((d2 / "meta.json").read_text(encoding="utf-8"))
    assert meta2["cell_state"]["capacity_ref_Ah"] == pytest.approx(4.6e-4)


def test_an_unregistered_cell_gets_no_capacity_reference_at_all(tmp_path):
    """The other half: when there is no build record there is no number, and
    the platform says `capacity_SOH` is absent rather than inventing a
    reference. A missing fact must not be defaulted to a plausible one."""
    from monocell.schema.ingest import preview_ingest

    prev = preview_ingest("cycling", _landt_csv(tmp_path / "orphan.csv"),
                          {"protocol": {"description": "x"},
                           "instrument": {"sampling_rate_Hz": 1.0}},
                          instrument="landt")

    assert prev["quality"]["capacity_SOH"] is None
    assert any("no capacity_ref_Ah" in f for f in prev["quality"]["flags"])


# ---------------------------------------------------------------------------
# `misc` as the on-ramp
# ---------------------------------------------------------------------------

def test_a_misc_file_is_promoted_to_a_real_type_and_the_original_stays(
        real_cell, root, tmp_path):
    """The way out of `misc`, and the reason `misc` is worth having.

    `misc` takes a file the schema has no shape for, stores it exactly as it
    arrived and unlocks nothing, which makes it a good place to LAND a file and
    a bad place to leave one. Promotion is the other half: name the type, name
    the profile, and it becomes a real experiment.
    """
    from monocell.schema.ingest import ingest_data, promote

    csv = _landt_csv(tmp_path / "mystery.csv", cycles=3)
    landed = ingest_data(CELL, "misc", csv, {"protocol": {"description": "landed as misc"}},
                         root)

    promoted = promote(CELL, landed.name, "cycling", instrument="landt", root=root)
    meta = json.loads((promoted / "meta.json").read_text(encoding="utf-8"))

    assert meta["experiment_type"] == "cycling"
    assert meta["derived_from"] == landed.name, \
        "the promoted experiment does not say where it came from"
    assert (landed / "meta.json").exists(), \
        "promotion edited the original; the store is append-only and must add"

    # ...and it really went through the profile: 0.5 mA, not 0.5 A
    series = pd.read_parquet(promoted / "series.parquet")
    assert series["I_A"].abs().max() == pytest.approx(0.5e-3, rel=1e-9)
    assert meta["quality"]["capacity_SOH"] is not None, \
        "the promoted experiment did not get the per-cycle quality its type carries"


def test_promotion_reads_the_store_and_not_the_file_it_came_from(real_cell, root, tmp_path):
    """`raw/series.parquet` is the table as it arrived, headers and all, and it
    is in the store. So a promotion works months later on a machine that never
    saw the original export — which is the point of keeping the raw frame."""
    from monocell.schema.ingest import ingest_data, promote

    csv = _landt_csv(tmp_path / "gone.csv", cycles=2)
    landed = ingest_data(CELL, "misc", csv, {"protocol": {"description": "x"}}, root)
    csv.unlink()                                    # the export is off the machine

    promoted = promote(CELL, landed.name, "cycling", instrument="landt", root=root)
    assert len(pd.read_parquet(promoted / "series.parquet")) == 80


def test_promoting_to_the_type_it_already_is_is_refused(real_cell, root, tmp_path):
    """Not a no-op, and not harmless: it would write a second copy of the same
    measurement with a `derived_from` pointing at the first, and nothing
    downstream distinguishes two experiments of one type except by reading
    both."""
    from monocell.schema.ingest import ingest_data, promote

    landed = ingest_data(CELL, "misc", _landt_csv(tmp_path / "m.csv"),
                         {"protocol": {"description": "x"}}, root)

    with pytest.raises(ValueError, match="already a misc"):
        promote(CELL, landed.name, "misc", root=root)


def test_without_a_profile_promotion_uses_the_synonym_list_and_says_what_it_dropped(
        real_cell, root, tmp_path):
    """The second route, and it goes through the SAME mapping so that a column
    has one place to vanish rather than two. On a Landt file the synonym list
    knows almost nothing, and the report is what makes that visible instead of
    it arriving as a frame with three columns and no explanation."""
    from monocell.schema.ingest import ingest_data, promote

    landed = ingest_data(CELL, "misc", _landt_csv(tmp_path / "m.csv"),
                         {"protocol": {"description": "x"}}, root)

    with pytest.raises(ValueError, match="missing required series columns"):
        promote(CELL, landed.name, "cycling", root=root)

    # the same file, through the profile that describes the rig, goes in
    assert promote(CELL, landed.name, "cycling", instrument="landt", root=root).exists()


def test_cli_promote_names_the_original_it_derived_from(real_cell, root, tmp_path, capsys):
    """The one line an engineer needs afterwards: the store now has two
    experiments where it had one, and which is which."""
    from monocell.schema.ingest import ingest_data

    landed = ingest_data(CELL, "misc", _landt_csv(tmp_path / "m.csv", cycles=2),
                         {"protocol": {"description": "x"}}, root)

    assert cli_main(["promote", "--cell", CELL, "--experiment", landed.name,
                     "--type", "cycling", "--instrument", "landt",
                     "--data-root", str(root)]) == 0
    out = capsys.readouterr().out

    assert "promoted" in out and landed.name in out
    assert "derived_from" in out
    assert len(list((root / "experiments" / CELL).glob("cycling_*"))) == 1
    assert len(list((root / "experiments" / CELL).glob("misc_*"))) == 1
