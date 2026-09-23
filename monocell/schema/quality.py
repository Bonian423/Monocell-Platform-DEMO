"""Quality checks computed at ingest, not later. Results land in `quality.json`
and the `meta.quality` summary block.

- `sum_check`: three-electrode reference check — V_full must equal
  V_pos - V_neg. Returns the per-row residual column and a monotonic drift
  measure (reference drift is a series, not a scalar).
- `lin_kk`: linear Kramers-Kronig validation (Schönleber et al. 2014 linKK)
  — fit a causal R0 + RC-ladder model and residualize; the residual % is the
  artifact detector for EIS.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

SUM_CHECK_MAX_MV = 5.0  # |V_full - (V_pos - V_neg)| beyond this fails the check
DRIFT_MAX_MV = 2.0  # fitted reference drift beyond this gets flagged
KK_MAX_PCT = 1.0  # lin-KK residual beyond this fails the check
# A cycle's coulombic efficiency outside this band is not the cell being
# strange, it is the bookkeeping being wrong — see `cycling_soh`. Wide on
# purpose: a real shuttle or a leaking cell sits just outside 1.0 and must stay
# visible as a measurement, while a charge counted into the wrong cycle lands
# near 0.5 or above 2.
CE_PLAUSIBLE = (0.9, 1.1)


def sum_check(df) -> dict:
    """Three-electrode reference sum-check on a series frame.

    Requires columns V_full_V, V_pos_V, V_neg_V. Returns a quality dict and
    (by mutating df) the per-row `sum_residual_mV` column.
    """
    resid_mv = (df["V_full_V"] - (df["V_pos_V"] - df["V_neg_V"]).to_numpy()) * 1e3
    if "sum_residual_mV" in df.columns:
        df["sum_residual_mV"] = resid_mv
    else:
        df.insert(df.shape[1], "sum_residual_mV", resid_mv)

    max_resid = float(np.max(np.abs(resid_mv)))
    # Monotonic drift: linear slope of the residual series vs time, scaled to
    # the experiment duration. Store the series (residuals are monotonic-ish),
    # report the scalar.
    t = df["t_s"].to_numpy()
    if t[-1] > t[0]:
        slope = np.polyfit(t, resid_mv, 1)[0]  # mV/s
        drift_mv = float(slope * (t[-1] - t[0]))
    else:
        drift_mv = 0.0
    ok = max_resid <= SUM_CHECK_MAX_MV
    flags = []
    if not ok:
        flags.append(f"sum_residual_max {max_resid:.3f} mV > {SUM_CHECK_MAX_MV} mV")
    if abs(drift_mv) > DRIFT_MAX_MV:
        flags.append(f"reference drift {drift_mv:.3f} mV > {DRIFT_MAX_MV} mV")
    return {
        "sum_check_ok": bool(ok),
        "sum_residual_max_mV": max_resid,
        "reference_drift_mV": drift_mv,
        "flags": flags,
    }


def lin_kk(f_Hz, Z_re_Ohm, Z_im_Ohm, n_basis: int = 48) -> dict:
    """Linear Kramers-Kronig validation (Schönleber et al., Electrochim. Acta
    131 (2014) 20-27, "linKK"): fit the causal model

        Zhat(w) = R0 + sum_k Rk / (1 + j w tau_k)

    with tau_k log-spaced over [1/w_max, 1/w_min], by complex least squares
    with Rk >= 0 (R0 free), and residualize per point as |Z - Zhat| / |Z|.
    A KK-consistent spectrum fits to the instrument noise level; a localized
    non-causal artifact (bad contact, drift, mains pickup) cannot be
    represented by the causal ladder and shows up as a residual spike.

    The reported residual is the MAX over the band. For white modulus-relative
    noise of level sigma the max residual concentrates near ~3*sigma (the max
    of ~2*n_f normal variates), so the 1% threshold means "artifacts above
    the noise" for sigma <= ~0.3% (SNR >= ~50 dB). A clean spectrum at 40 dB
    (sigma = 1%, max residual ~3%) fails the gate.

    n_basis = 8 RC elements per decade (48 over a 6-decade band), which keeps
    the ladder well below a typical data count (61 points). At one element per
    measured point the fit interpolates the noise and the check goes blind; a
    much coarser basis false-flags causal spectra.

    Returns {"kk_ok", "kk_residual_pct", "flags"}.
    """
    f = np.asarray(f_Hz, dtype=float)
    re = np.asarray(Z_re_Ohm, dtype=float)
    im = np.asarray(Z_im_Ohm, dtype=float)
    if f.size < 2:
        raise ValueError("lin-KK needs at least two frequency points")
    if f.min() <= 0:
        raise ValueError("frequencies must be positive")
    if np.any(np.diff(f) <= 0):
        raise ValueError("f_Hz must be strictly increasing")
    z_mod = np.hypot(re, im)
    if not np.all(z_mod > 0):
        raise ValueError("lin-KK is undefined where |Z| = 0")

    w = 2.0 * np.pi * f
    tau = np.geomspace(1.0 / w.max(), 1.0 / w.min(), n_basis)
    X = np.column_stack([np.ones_like(f, dtype=complex)] + [1.0 / (1.0 + 1j * w * t) for t in tau])
    A = np.vstack([X.real / z_mod[:, None], X.imag / z_mod[:, None]])
    b = np.concatenate([re / z_mod, im / z_mod])
    bounds = (np.concatenate([[-np.inf], np.zeros(n_basis)]), np.full(n_basis + 1, np.inf))
    coef = lsq_linear(A, b, bounds=bounds).x
    zhat = X @ coef
    resid_pct = float(100.0 * np.max(np.abs(1.0 - zhat / (re + 1j * im))))  # |Z - Zhat| / |Z|
    ok = resid_pct <= KK_MAX_PCT
    flags = [] if ok else [f"lin-KK residual {resid_pct:.2f}% > {KK_MAX_PCT}%"]
    return {"kk_ok": bool(ok), "kk_residual_pct": resid_pct, "flags": flags}


def rpt_soh(series, meta: dict) -> dict:
    """RPT state-of-health at ingest: capacity and resistance.

    capacity_SOH = discharge-leg q span / capacity_ref. resistance_SOH_Ohm from
    half the charge/discharge leg gap at mid-q (V_chg - V_dch = 2*R*|I|).
    """
    dis = series[series["seg_type"] == "discharge"]
    cap_ah = float(dis["q_Ah"].max() - dis["q_Ah"].min()) if len(dis) else 0.0
    ref = (meta.get("cell_state") or {}).get("capacity_ref_Ah")
    soh = cap_ah / ref if ref else None

    r_ohm = None
    chg = series[series["seg_type"] == "charge"]
    if len(dis) and len(chg):
        q_mid = 0.5 * (dis["q_Ah"].min() + dis["q_Ah"].max())
        v_d = float(np.interp(q_mid, dis["q_Ah"].to_numpy(), dis["V_V"].to_numpy()))
        v_c = float(np.interp(q_mid, chg["q_Ah"].to_numpy(), chg["V_V"].to_numpy()))
        i_abs = float(np.abs(chg["I_A"].iloc[0]))
        if i_abs > 0:
            r_ohm = (v_c - v_d) / (2.0 * i_abs)
    return {"capacity_SOH": soh, "resistance_SOH_Ohm": r_ohm}


def cycling_soh(series, meta: dict) -> dict:
    """Per-cycle capacity, coulombic efficiency and retention, at ingest.

    What cycling data is FOR. An RPT gives the fade curve at a handful of
    points; a hundred cycles give it at a hundred.

    Two different denominators, reported separately because they answer
    different questions and only one of them is usually available:

      `capacity_SOH`        against the cell's `capacity_ref_Ah` — absolute
                            state of health, and `None` when no reference is
                            recorded rather than silently against something else
      `capacity_retention`  last cycle / FIRST CYCLE IN THIS FILE. Not the same
                            as SOH and must never be read as it: a real export
                            is usually a slice, so the first cycle here is often
                            not cycle one and the retention is relative to
                            wherever the export happens to begin. `first_cycle`
                            is reported beside it so the reader can see that.

    Capacity per segment is a SPAN (`max - min`) rather than a final value,
    because a cycler's capacity column may reset per step or accumulate across
    the run and the span is right either way.
    """
    if "cycle_index" not in series.columns or "seg_type" not in series.columns:
        return {"capacity_SOH": None, "resistance_SOH_Ohm": None,
                "flags": ["no cycle_index/seg_type: per-cycle capacity not computed"]}

    rows: list[dict] = []
    for cyc, g in series.groupby("cycle_index", sort=True):
        dis = g[g["seg_type"] == "discharge"]
        chg = g[g["seg_type"] == "charge"]
        d_ah = float(dis["q_Ah"].max() - dis["q_Ah"].min()) if len(dis) else None
        c_ah = float(chg["q_Ah"].max() - chg["q_Ah"].min()) if len(chg) else None
        rows.append({
            "cycle_index": int(cyc),
            "discharge_Ah": d_ah,
            "charge_Ah": c_ah,
            # The current this cycle ran at. Recorded per cycle because a real
            # aging file is NOT one protocol — see `_retention_at_rate`.
            "discharge_A": (float(np.nanmean(np.abs(dis["I_A"]))) if len(dis) else None),
            # Never clipped and never dropped: a CE over 1 is a shuttle, a leak
            # or a miscounted step, and each of those is worth seeing. Which of
            # them it is, is the reader's call — `CE_PLAUSIBLE` only decides
            # whether the FILE gets a flag saying some of these are not
            # measurements of the cell.
            "coulombic_efficiency": (d_ah / c_ah) if (d_ah and c_ah) else None,
        })

    usable = [r for r in rows if r["discharge_Ah"] and r["discharge_A"]]
    flags: list[str] = []
    ref = (meta.get("cell_state") or {}).get("capacity_ref_Ah")
    if not usable:
        return {"capacity_SOH": None, "resistance_SOH_Ohm": None,
                "capacity_retention": None, "first_cycle": None, "n_cycles": 0,
                "per_cycle": rows,
                "flags": ["no cycle carried a usable discharge segment: nothing to measure"]}

    duty, groups = _rate_groups(usable)
    at_duty = groups[duty]
    first, last = at_duty[0], at_duty[-1]
    retention = last["discharge_Ah"] / first["discharge_Ah"]
    soh = (last["discharge_Ah"] / ref) if ref else None

    if len(groups) > 1:
        others = {r: len(g) for r, g in groups.items() if r != duty}
        flags.append(
            f"this file mixes rates: {len(at_duty)} cycle(s) at {duty * 1e3:.4g} mA and "
            + ", ".join(f"{n} at {r * 1e3:.4g} mA" for r, n in sorted(others.items()))
            + ". Capacity at two rates is two different measurements, so retention is "
            "computed WITHIN the majority rate only — the slow cycles are almost "
            "certainly check-ups (an RPT) embedded in the cycling and belong in their "
            "own experiment.")
    odd = [r for r in rows if r["coulombic_efficiency"] is not None
           and not (CE_PLAUSIBLE[0] <= r["coulombic_efficiency"] <= CE_PLAUSIBLE[1])]
    if odd:
        worst = max(odd, key=lambda r: abs(r["coulombic_efficiency"] - 1.0))
        flags.append(
            f"{len(odd)} of {len(rows)} cycle(s) have a coulombic efficiency outside "
            f"{CE_PLAUSIBLE[0]}-{CE_PLAUSIBLE[1]} (worst "
            f"{worst['coulombic_efficiency']:.3g} at cycle {worst['cycle_index']}). A charge "
            "and its discharge that land either side of a cycle-index boundary produce "
            "exactly this, and a protocol change part-way through a run is the usual cause "
            "— so for those cycles the CE is not a measurement of the cell, and their "
            "capacities are worth checking against the trajectory before they are used.")
    if first["cycle_index"] > 1:
        # "the first cycle at the duty rate", not "the first cycle in the file":
        # the two differ when a file opens with a slow check-up before the duty
        # cycling starts, and the number this sentence is about is the retention
        # DENOMINATOR, which is the latter.
        flags.append(
            f"capacity_retention is measured from cycle {first['cycle_index']}, the first "
            "cycle at the duty rate in this export, and not from the cell's first cycle")
    if soh is None:
        flags.append("no capacity_ref_Ah: capacity_SOH not computed — retention is against "
                     "this file's first cycle at the duty rate, not against BOL")

    return {
        "capacity_SOH": soh,
        "resistance_SOH_Ohm": None,   # a duty cycle has no charge/discharge leg pair at one q
        "capacity_retention": retention,
        "duty_current_A": duty,
        "first_cycle": first["cycle_index"],
        "last_cycle": last["cycle_index"],
        "n_cycles": len(at_duty),
        "n_cycles_all_rates": len(usable),
        "per_cycle": rows,
        "flags": flags,
    }


def _rate_groups(usable: list[dict], tol: float = 0.05) -> tuple[float, dict[float, list[dict]]]:
    """`(duty current, {current: cycles})` — the file's rates, and its main one.

    A real aging campaign is often not one protocol: duty cycling with a slow
    check-up at each end is common. Taking the first and last cycle of such a
    file compares one check-up against the other, and can report retention
    above 100% on a cell whose duty-rate capacity has clearly fallen. Grouping
    by rate and measuring retention within the majority rate avoids that.

    Grouped by CURRENT rather than by C-rate because a C-rate needs
    `capacity_ref_Ah` and a real export often has none, while the current is
    always in the file. `tol` is relative, so it separates a C/20 check from a
    1C duty by a factor of twenty and is blind to the ordinary drift of a
    current-controlled step.
    """
    groups: dict[float, list[dict]] = {}
    for r in usable:
        i = r["discharge_A"]
        for key in groups:
            if abs(i - key) <= tol * max(abs(key), abs(i)):
                groups[key].append(r)
                break
        else:
            groups[i] = [r]
    duty = max(groups, key=lambda k: len(groups[k]))
    return duty, groups


def instrument_flags(meta: dict) -> list[str]:
    """The profile parse report, as sentences on the experiment's own record.

    `instruments.read_with_profile` returns four different facts about what it
    could not place, and they stay four here rather than collapsing into "some
    columns were skipped". They are different problems:

      `missing`     the profile names a header this file does not have — the
                    usual cause is a rig whose export was renamed, and the
                    schema column is ABSENT, not zero
      `unplaced`    the file has a column the profile does not name. Often
                    correct (an export carries channels the schema has no home
                    for) and the engineer still has to see it, because a column
                    that SHOULD have been mapped looks identical to one that
                    should not
      `empty`       mapped, and every value a null token. The unconnected probe:
                    the channel exists and recorded nothing, which is neither
                    absent nor zero
      `unmapped_segments`  a step label the `seg_map` has no entry for. Those
                    rows kept their own text, so they are visible rather than
                    guessed into `rest`

    Plus the one fact about the profile rather than the file: an UNVERIFIED
    profile got its column names from a published format and has never met an
    export. That is a real difference and the flag says so — a profile that
    parsed cleanly against names it invented is exactly the failure that looks
    like success.

    Flags, not errors: nothing here decides whether the file may be stored.
    `write_experiment` does that, from the schema, and `required_missing` is
    the report's word for the same refusal seen early.
    """
    rep = meta.get("ingest_report") or {}
    if not rep:
        return []
    name = rep.get("profile", "?")
    flags: list[str] = []
    if not rep.get("verified"):
        flags.append(
            f"instrument profile {name!r} is UNVERIFIED — its column names come from a "
            "published export format, not from a file. It got you most of the way; "
            "check the mapping before trusting the numbers, then save the correction "
            "as a profile of your own.")
    if rep.get("required_missing"):
        flags.append(
            f"profile {name!r} produced no {rep['required_missing']} — the type requires "
            "those columns, so this file will not store until the mapping places them")
    if rep.get("missing"):
        flags.append(
            f"profile {name!r} expected headers this file does not have: {rep['missing']} "
            "— the schema columns they feed are absent, not zero")
    if rep.get("unplaced"):
        flags.append(
            f"columns the profile does not place, and which were therefore NOT read: "
            f"{rep['unplaced']} — nothing was dropped quietly, but nothing was read "
            "from them either")
    if rep.get("not_in_schema"):
        flags.append(
            f"profile {name!r} produced columns this type does not declare: "
            f"{rep['not_in_schema']} — it is probably the wrong type for this file")
    if rep.get("empty"):
        flags.append(
            f"mapped and entirely empty: {rep['empty']} — the channel is present and "
            "recorded nothing (an unconnected probe reads as null tokens, not as 0)")
    if rep.get("unmapped_segments"):
        flags.append(
            f"step labels the profile has no entry for: {rep['unmapped_segments']} — "
            "those rows kept their own text in seg_type rather than becoming a guess")
    return flags


def compute_quality(exp_type: str, series, meta: dict) -> dict:
    """Dispatch the ingest quality checks for an experiment type.

    `sum_check` applies to three_electrode_ts always, and to hppc only when
    the three-electrode columns are present (full-cell HPPC has no reference).
    `rpt_soh` applies to rpt (capacity/resistance at ingest, part of the schema
    contract).
    """
    from .tables import SHARED_QUALITY_KEYS, SPECS

    spec = SPECS[exp_type]
    # Built from the declaration rather than written out again: this record and
    # `write_experiment`'s projection of it are two views of one tuple, and a
    # literal in each is how a key came to be computed here and dropped there.
    result: dict = {k: None for k in SHARED_QUALITY_KEYS}
    result["flags"] = []
    if "sum_check" in spec.quality_checks:
        cols = series.columns
        if {"V_full_V", "V_pos_V", "V_neg_V"} <= set(cols):
            result.update(sum_check(series))
        else:
            result["flags"].append("no reference electrode columns — sum-check skipped")
    if "lin_kk" in spec.quality_checks:
        if "spectrum_id" in series.columns and series["spectrum_id"].nunique() > 1:
            # a multi-spectrum file (in-situ sweep) is NOT one continuous
            # spectrum — the concatenation jumps discontinuously, so lin-KK
            # runs per spectrum and the frame-level result is the worst one
            residuals, oks = [], []
            for _, grp in series.groupby("spectrum_id", sort=False):
                r = lin_kk(grp["f_Hz"], grp["Z_re_Ohm"], grp["Z_im_Ohm"])
                residuals.append(r["kk_residual_pct"])
                oks.append(bool(r["kk_ok"]))
            result.update({"kk_ok": bool(all(oks)), "kk_residual_pct": float(max(residuals))})
            if not all(oks):
                result["flags"].append(
                    f"lin-KK failed on {oks.count(False)} of {len(oks)} spectra "
                    f"(worst residual {max(residuals):.2f}%)")
        else:
            result.update(lin_kk(series["f_Hz"], series["Z_re_Ohm"], series["Z_im_Ohm"]))
    if exp_type == "rpt":
        result.update(rpt_soh(series, meta))
    if exp_type == "cycling":
        cyc = cycling_soh(series, meta)
        result["flags"].extend(cyc.pop("flags"))
        result.update(cyc)
    # Every type whose series is keyed on time. `write_experiment` rejects NaNs
    # in `t_s` but does not look at its ORDER, so a column that runs backwards
    # (a per-step clock read as an elapsed one, a corrupted export, two files
    # concatenated) would pass ingest, and every window built on it would be
    # silently wrong: pulse extraction, relaxation tails, per-cycle grouping,
    # every plot with time on the x axis.
    #
    # A FLAG and not a refusal. The frame is still the measurement, the rows
    # are still in the file's order, and refusing the ingest would leave the
    # engineer with nothing to look at while deciding which of the three causes
    # it is. `instruments.InstrumentProfile.time_restarts_per_step` is the fix
    # for the first cause, and it is named here because a reader who sees this
    # flag on a cycler export usually wants it.
    result["flags"] = _time_order_flags(series) + result["flags"]
    # First, and for every type: what the parse could not place is the reason
    # a later flag reads the way it does, so it belongs above it rather than
    # under it.
    result["flags"] = instrument_flags(meta) + result["flags"]
    return result


def _time_order_flags(series) -> list[str]:
    """`t_s` going backwards, as a sentence — or nothing when it does not."""
    if "t_s" not in getattr(series, "columns", ()) or len(series) < 2:
        return []
    diffs = series["t_s"].diff()
    back = int((diffs < 0).sum())
    if not back:
        return []
    worst = float(diffs.min())
    return [f"t_s goes BACKWARDS at {back} row(s) (worst step {worst:.3g} s): every window "
            "built on time — pulse extraction, relaxation tails, per-cycle grouping — reads "
            "the wrong rows. If this rig restarts its clock at each protocol step, set "
            "`time_restarts_per_step` on its instrument profile and re-ingest"]
