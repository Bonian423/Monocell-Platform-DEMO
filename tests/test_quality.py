"""Quality checks at ingest: sum_check (drift, mutation) and lin_kk
(fidelity, noise scaling, artifact detection, input guards)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monocell.schema.quality import compute_quality, lin_kk, sum_check
from conftest import eis_series, eis_spectrum, hppc_series


def three_el_series(n=200, drift_mv=0.0, mismatch_mv=0.0) -> pd.DataFrame:
    """three_electrode_ts frame with a configurable reference story.

    A drifting reference means the MEASURED V_neg walks away from the true
    anode potential; V_full (full-cell, no reference) does not see it — so
    the sum residual V_full - (V_pos - V_neg) inherits the drift.
    """
    t = np.arange(n) * 0.1  # 20 s
    v_pos = 4.2 + 0.001 * np.sin(t)
    v_neg_measured = 0.1 + (drift_mv / 1e3) * np.arange(n) / (n - 1)
    v_full = v_pos - 0.1 + mismatch_mv / 1e3  # full-cell voltage, true anode 0.1 V
    return pd.DataFrame(
        {
            "t_s": t,
            "I_A": np.zeros(n),
            "V_full_V": v_full,
            "V_pos_V": v_pos,
            "V_neg_V": v_neg_measured,
            "T_degC": 25.0,
            "soc": 0.5,
            "seg_id": np.arange(n, dtype=np.int32),
            "seg_type": "rest",
        }
    )


# ---------------------------------------------------------------- sum_check

def test_sum_check_clean():
    res = sum_check(three_el_series())
    assert res["sum_check_ok"] is True
    assert res["sum_residual_max_mV"] < 1e-6
    assert res["reference_drift_mV"] == 0.0


def test_sum_check_flags_offset():
    res = sum_check(three_el_series(mismatch_mv=10.0))
    assert res["sum_check_ok"] is False
    assert res["sum_residual_max_mV"] == pytest.approx(10.0, abs=1e-6)
    assert any("sum_residual_max" in f for f in res["flags"])


def test_sum_check_flags_drift():
    res = sum_check(three_el_series(drift_mv=3.0))
    assert res["sum_check_ok"] is True  # constant offset is tiny
    assert res["reference_drift_mV"] == pytest.approx(3.0, abs=1e-3)
    assert any("reference drift" in f for f in res["flags"])


def test_sum_check_mutates_frame():
    df = three_el_series()
    sum_check(df)
    assert "sum_residual_mV" in df.columns
    np.testing.assert_allclose(df["sum_residual_mV"], 0.0, atol=1e-9)


# ------------------------------------------------------------------- lin_kk
#
# Spectra and artifacts are built here rather than imported, so each test says
# exactly what it put in. Noise is modulus-relative: at an SNR of `s` dB each
# component carries 10^(-s/20) of |Z|.

def _noisy(z: np.ndarray, snr_db: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = 10.0 ** (-snr_db / 20.0)
    return z + np.abs(z) * sigma * (rng.standard_normal(len(z)) + 1j * rng.standard_normal(len(z)))


def _bump(f: np.ndarray, z: np.ndarray, centre_hz=10.0, width_dec=0.15, gain=0.10) -> np.ndarray:
    """A local gain error over a narrow band: a bad contact, for one."""
    x = np.log10(f / centre_hz)
    return z * (1.0 + gain * np.exp(-0.5 * (x / width_dec) ** 2))


def _pickup(f: np.ndarray, z: np.ndarray, hz=50.0, gain=0.05) -> np.ndarray:
    """Mains pickup: one point near 50 Hz off by a few percent."""
    out = z.copy()
    out[int(np.argmin(np.abs(np.log10(f / hz))))] *= 1.0 + gain
    return out


def _drift(f: np.ndarray, z: np.ndarray, corner_hz=1.0, per_decade=0.03) -> np.ndarray:
    """The cell drifting during the slow end of the sweep: below the corner the
    real part grows with the time spent there. Smooth everywhere except at the
    corner, so a smoothness test would miss it; a causal fit cannot absorb it."""
    x = np.clip(np.log10(corner_hz / f), 0.0, None)
    return z + np.abs(z).max() * per_decade * x


def _kk(f, z):
    return lin_kk(f, z.real, z.imag)


def test_lin_kk_passes_noiseless_causal_spectrum():
    # the spectrum is itself an RC ladder, so the noiseless residual is only the
    # error of the fixed tau grid
    f, z = eis_spectrum()
    res = _kk(f, z)
    assert res["kk_ok"] is True
    assert res["kk_residual_pct"] < 0.5


def test_lin_kk_flags_injected_bump():
    f, z = eis_spectrum()
    res = _kk(f, _bump(f, z))
    assert res["kk_ok"] is False
    assert res["kk_residual_pct"] > 1.0


def test_lin_kk_noise_scaling():
    # at 40 dB the per-point noise (1% of |Z|) already exceeds the 1% gate, so a
    # CLEAN spectrum fails; at 60 dB it passes
    f, z = eis_spectrum()
    r40 = _kk(f, _noisy(z, 40.0))
    r60 = _kk(f, _noisy(z, 60.0))
    assert r40["kk_ok"] is False and r40["kk_residual_pct"] > 1.0
    assert r60["kk_ok"] is True and r60["kk_residual_pct"] < 1.0


def test_lin_kk_input_guards():
    f, z = eis_spectrum()
    with pytest.raises(ValueError, match="at least two"):
        lin_kk(f[:1], z.real[:1], z.imag[:1])
    with pytest.raises(ValueError, match="positive"):
        lin_kk(np.array([-1.0, 1.0]), z.real[:2], z.imag[:2])
    with pytest.raises(ValueError, match="strictly increasing"):
        lin_kk(np.array([2.0, 1.0]), z.real[:2], z.imag[:2])
    with pytest.raises(ValueError, match=r"\|Z\| = 0"):
        lin_kk(f, np.zeros_like(z.real), np.zeros_like(z.imag))


def test_lin_kk_flags_drift_and_pickup():
    # the artifact classes a KK check exists for, each on top of 60 dB noise
    # that passes on its own: a mid-sweep drift onset and a narrow mains pickup
    f, z = eis_spectrum()
    noisy = _noisy(z, 60.0)
    assert _kk(f, noisy)["kk_ok"] is True
    for name, injected in (("drift", _drift(f, noisy)), ("pickup", _pickup(f, noisy))):
        res = _kk(f, injected)
        assert res["kk_ok"] is False, name
        assert res["kk_residual_pct"] > 1.0, name


# --------------------------------------------------------- compute_quality

def test_compute_quality_hppc_skips_sum_check_without_reference():
    res = compute_quality("hppc", hppc_series(), {})
    assert res["sum_check_ok"] is None
    assert any("no reference electrode columns" in f for f in res["flags"])


def test_compute_quality_hppc_runs_sum_check_with_reference():
    s = hppc_series(with_vpos=True, with_vneg=True)
    res = compute_quality("hppc", s, {})
    assert res["sum_check_ok"] is True
    assert res["sum_residual_max_mV"] == pytest.approx(0.0, abs=1e-6)


def test_compute_quality_eis_runs_lin_kk_only():
    res = compute_quality("eis", eis_series(), {})
    assert res["kk_ok"] is True
    assert res["sum_check_ok"] is None
    assert res["flags"] == []


# ---------------------------------------------------------------------------
# `t_s` going backwards is a flag, on every type that has one
# ---------------------------------------------------------------------------


def test_a_time_column_that_runs_backwards_is_flagged_at_ingest():
    """`write_experiment` rejects NaNs in `t_s` but does not look at its ORDER,
    so without this check a column that runs backwards would pass ingest and
    every window built on it would be silently wrong: pulse extraction,
    relaxation tails, per-cycle grouping, every plot with time on the x axis.

    A FLAG and not a refusal. Three different causes produce it — a per-step
    clock read as an elapsed one, a corrupted export, two files concatenated —
    and refusing the ingest would leave the engineer with nothing to look at
    while deciding which. The flag names the fix for the most common cause.
    """
    series = hppc_series()
    series.loc[series.index[len(series) // 2], "t_s"] = -1.0

    flags = compute_quality("hppc", series, {"instrument": {"sampling_rate_Hz": 10.0}})["flags"]
    backwards = [f for f in flags if "t_s goes BACKWARDS" in f]
    assert len(backwards) == 1, flags
    assert "time_restarts_per_step" in backwards[0], (
        "the flag names the symptom and not the one thing that usually fixes it"
    )


def test_a_monotone_time_column_raises_no_flag():
    """The direction that makes the test above mean something: a check that
    fired on clean data would be a constant, and this one runs on every
    ingest of every type."""
    flags = compute_quality("hppc", hppc_series(),
                            {"instrument": {"sampling_rate_Hz": 10.0}})["flags"]
    assert not [f for f in flags if "BACKWARDS" in f]
