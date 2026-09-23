"""Per-type series column specs: the single source of truth for the schema.

Every experiment type the store accepts is declared here, with its columns,
the instrument metadata it must carry, the quality checks run at ingest, and
the ingest rules that turn missing protocol facts into flags.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # float64 | int32 | str
    required: bool = True  # column must be present (non-nullable)
    note: str = ""


@dataclass(frozen=True)
class TableSpec:
    type: str
    columns: tuple[Column, ...]
    instrument: dict[str, str]  # required instrument-block keys -> note
    quality_checks: tuple[str, ...] = ()  # which quality checks run at ingest
    # THE TYPE'S OWN HEADLINE NUMBERS, beyond the shared eight below.
    #
    # `compute_quality` writes the whole quality record to `quality.json` and
    # `write_experiment` projects a SUMMARY of it into `meta.quality` — the
    # small header every reader already opens, and the one the experiment table
    # and the manifest read. That projection used to be a literal tuple of
    # eight key names in `write_experiment`, so a type whose headline number is
    # not one of the eight had it computed, written to `quality.json`, and
    # dropped before anything a reader sees.
    #
    # `cycling` is the type that needs it: a hundred-cycle export with no
    # `capacity_ref_Ah` has `capacity_SOH = None` (correct, there is nothing
    # to reference it to), while `capacity_retention` IS the number the run was
    # done for, and without this field it would never reach the header.
    #
    # SCALARS ONLY, deliberately. `cycling`'s `per_cycle` is one row per cycle
    # and belongs in `quality.json`, where the reader that wants the trajectory
    # already looks. A header that grew a hundred rows would stop being a
    # header, and `meta.json` is parsed once per row of every experiment listing.
    quality_summary: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()  # ingest validation rules, see RULES
    # The ONE declared exemption from "every type has a declared schema": a
    # permissive spec accepts columns it never declared, keeps them as they
    # arrived, and records what it saw in `meta.observed_columns`. Exactly one
    # type sets this (`misc`), and `write_experiment` says so in the code path
    # that acts on it — an exemption is only safe where it can be counted.
    permissive: bool = False

# The quality keys EVERY type carries, in the order they are written. Declared
# here rather than in `quality.py` because `compute_quality` builds its base
# record from this tuple and `write_experiment` projects by it — two places
# that must agree about what "the shared quality record" is, and a literal in
# each was how they came to disagree.
#
# `flags` is last and is the only non-scalar: a list of sentences, which is the
# one thing in the header worth its bytes on every type.
SHARED_QUALITY_KEYS: tuple[str, ...] = (
    "sum_check_ok", "sum_residual_max_mV", "reference_drift_mV",
    "kk_ok", "kk_residual_pct", "capacity_SOH", "resistance_SOH_Ohm", "flags",
)


def summary_keys(exp_type: str) -> tuple[str, ...]:
    """The keys `meta.quality` carries for `exp_type` — shared, then its own.

    One function so the projection has one definition. A caller that wants to
    know what is in a header (a page, a test, the manifest's flag scan) asks
    here rather than reading `write_experiment` to find out.
    """
    extra = tuple(k for k in SPECS[exp_type].quality_summary if k not in SHARED_QUALITY_KEYS)
    return SHARED_QUALITY_KEYS + extra


# How a series' `soc` column was arrived at. A closed vocabulary, because the
# four of these are four different numbers, and most series specs carry a bare
# `soc` column with nothing in the file saying which one it is:
#
#   coulomb_nameplate  counted charge over the cell's NAMEPLATE capacity. The
#                      usual default, and wrong by the fade on an aged cell.
#   coulomb_measured   counted charge over a capacity this cell was measured at
#   ocv_lookup         read off an OCV table rather than counted at all
#   generator_map      a simulator's own state variable; true by
#                      construction and not a measurement of anything
SOC_BASES = ("coulomb_nameplate", "coulomb_measured", "ocv_lookup", "generator_map")

# Where `cell_state.capacity_ref_Ah` came from. Recorded rather than inferred:
# the number is the same shape whichever it is, and the difference decides
# whether an SOH computed against it means anything.
CAPACITY_REF_SOURCES = ("sidecar", "build_record", "measured")

# Ingest rules referenced by TableSpec.rules
RULES: dict[str, str] = {
    "sampling_ge_10Hz": "sampling_rate_Hz >= 10 (honest R0)",
    "rpt_rate_C20_min": "rpt_rate recorded and <= C/20; C/10, or no rate at all, is "
                        "flagged at ingest",
    "rest_rule_5tau2": "protocol must record the 5*tau2 rest rule and the tau2 used",
    "soc_basis_recorded": "cell_state.soc_basis must say how the `soc` column was arrived at "
                          "(" + " | ".join(SOC_BASES) + ") — a counted SOC over a nameplate "
                          "capacity and one read off an OCV table are different numbers",
    "capacity_ref_not_nameplate_when_aged": "this run is on an aged cell and its "
                                            "capacity_ref_Ah is the cell's NAMEPLATE — every "
                                            "SOC and SOH against it is referenced to a capacity "
                                            "the cell no longer has",
}


def c(name, required=True, note="", dtype="float64"):
    return Column(name, dtype, required, note)


SPECS: dict[str, TableSpec] = {
    "three_electrode_ts": TableSpec(
        type="three_electrode_ts",
        columns=(
            c("t_s"), c("I_A"), c("V_full_V"), c("V_pos_V"), c("V_neg_V"),
            c("sum_residual_mV", note="V_full - (V_pos - V_neg), per row — drift is a series, not a scalar"),
            c("T_degC"), c("soc"), c("seg_id", dtype="int32"),
            c("seg_type", dtype="str", note="rest | cc | anode_hold | pulse"),
        ),
        instrument={
            "reference_type": "the reference electrode used",
            "calibrated_at": "reference calibration date",
            "rinse_protocol": "rinse protocol for the reference-electrode cell",
            "sampling_rate_Hz": "time-series sampling rate",
        },
        quality_checks=("sum_check",),
    ),
    "half_cell_ocp": TableSpec(
        type="half_cell_ocp",
        columns=(
            c("t_s"), c("I_A"), c("V_vs_Li_V"), c("q_Ah"), c("soc"), c("T_degC"),
        ),
        instrument={
            "electrode": "pos | neg",
            "counter": "counter electrode",
            "mass_mg_active": "active mass",
            "electrolyte": "electrolyte composition",
        },
    ),
    "hppc": TableSpec(
        type="hppc",
        columns=(
            c("t_s"), c("I_A"), c("V_full_V"),
            c("V_pos_V", required=False, note="present for three-electrode HPPC"),
            c("V_neg_V", required=False, note="present for three-electrode HPPC"),
            c("T_degC"), c("soc"), c("pulse_id", dtype="int32"),
            c("seg_type", dtype="str", note="rest | pulse | relax"),
        ),
        instrument={"sampling_rate_Hz": ">= 10 Hz for honest R0"},
        quality_checks=("sum_check",),  # applies only when V_pos_V/V_neg_V present
        rules=("sampling_ge_10Hz", "rest_rule_5tau2"),
    ),
    "eis": TableSpec(
        type="eis",
        columns=(
            c("f_Hz"), c("Z_re_Ohm"), c("Z_im_Ohm"), c("Z_mag_Ohm"), c("Z_phase_deg"),
            c("soc"), c("V_dc_V"), c("T_degC"),
            c("electrode", dtype="str", note="full | neg | pos"),
            c("spectrum_id", required=False, dtype="str",
              note="per-spectrum identity for multi-spectrum files (in-situ sweeps)"),
        ),
        instrument={
            "potentiostat": "instrument id",
            "amplitude_mV": "excitation amplitude",
            "f_min_Hz": "lowest frequency",
            "f_max_Hz": "highest frequency",
            "points_per_decade": "log spacing",
        },
        quality_checks=("lin_kk",),
    ),
    "rpt": TableSpec(
        type="rpt",
        columns=(
            c("t_s"), c("I_A"), c("V_V"), c("T_degC"), c("q_Ah"),
            c("seg_type", dtype="str"), c("cycle_index", dtype="int32"),
        ),
        instrument={},
        rules=("rpt_rate_C20_min",),
    ),
    # The aging DRIVER, as opposed to the RPT's periodic check-up. Same columns,
    # and a different type for a physical reason rather than a bookkeeping one:
    #
    #   * an RPT is deliberately SLOW (C/20, and `rpt_rate_C20_min` enforces it)
    #     because a slow check-up keeps the fine structure of the voltage curve
    #     readable; at 1C polarisation smears it. Cycling data cannot do an RPT's
    #     job, so filing it as one would either fail that gate forever or force
    #     the gate to be weakened, and the gate is right.
    #   * what cycling data CAN give is the fade curve at every cycle rather
    #     than at the handful of RPT points: capacity, coulombic efficiency and
    #     retention per cycle.
    #
    # So: no C/20 rule, its own per-cycle quality block, and no claim to be a
    # check-up.
    "cycling": TableSpec(
        type="cycling",
        columns=(
            c("t_s"), c("I_A"), c("V_V"), c("T_degC"), c("q_Ah"),
            c("seg_type", dtype="str"), c("cycle_index", dtype="int32"),
        ),
        instrument={},
        # Retention is the number a cycling run is DONE for, and it is the one
        # that survives a file with no capacity reference — where `capacity_SOH`
        # is correctly `None`. The duty current and the cycle count are what say
        # whether the retention means anything: a hundred cycles at one rate and
        # a hundred at four are the same integer and not the same experiment.
        quality_summary=("capacity_retention", "duty_current_A", "n_cycles",
                         "n_cycles_all_rates"),
    ),
    "pressure": TableSpec(
        type="pressure",
        columns=(
            c("t_s"), c("F_N"), c("p_kPa"),
            c("thickness_um", required=False),
            c("T_degC"), c("V_V"), c("I_A"), c("soc"),
            c("seg_type", dtype="str", note="includes creep_hold"),
        ),
        instrument={
            "load_cell_id": "load-cell id",
            "calibration_date": "load-cell calibration date (also mirrored in the rig's build record)",
            "fixture_stiffness_N_per_um": "measured, not from the drawing",
            "creep_hold_protocol": "creep-hold definition",
            "sampling_rate_Hz": "sampling rate",
            "ntp_synced": "whether the acquisition clock was NTP-synced",
        },
    ),
    "gitt": TableSpec(
        type="gitt",
        columns=(
            c("t_s"), c("I_A"), c("V_V"), c("T_degC"), c("soc"),
            c("pulse_id", dtype="int32"),
            c("seg_type", dtype="str", note="pulse | relax"),
        ),
        instrument={
            "pulse_rate": "pulse current (C/20)",
            "relax_criterion": "relax-until-|dV/dt| threshold",
        },
    ),
    # The one permissive type. `misc` exists because real files arrive whose
    # shape is not yet worth a schema: a rig sweep, a one-off characterisation,
    # a colleague's export. Without it such a file has nowhere to go, and the
    # only options are to invent a type for it or to leave it outside the store.
    # `promote` re-reads a misc experiment as a real type once one fits.
    #
    # `t_s` is declared but NOT required: it is the one column the rest of the
    # platform assumes exists (`write_experiment` null-checks it), so a misc file
    # that has one gets the check and a file that does not is still accepted.
    # Nothing consumes misc (`autofit.modules_for("misc")` is `()`), so it never
    # enters the staleness graph.
    "misc": TableSpec(
        type="misc",
        columns=(c("t_s", required=False,
                   note="the one column the rest of the platform assumes, when the file has it"),),
        instrument={},
        permissive=True,
    ),
}

def _universal_rules(spec: TableSpec) -> tuple[str, ...]:
    """The rules that follow from a spec's SHAPE rather than from its name.

    Attached here rather than repeated in nine literals, because nine identical
    strings are nine places for one of them to be forgotten — and the two below
    are exactly the kind that get forgotten on the type added next year:

      * a type with a `soc` column has to say how that column was arrived at
      * a type that can record an age has to not reference its numbers to a
        capacity the cell no longer has

    `misc` is exempt from the second for the reason it is exempt from
    everything: it declares no columns and no `cell_state` expectations, so a
    rule about `cell_state` has nothing to hold it to.
    """
    out: list[str] = []
    if any(c.name == "soc" for c in spec.columns):
        out.append("soc_basis_recorded")
    if not spec.permissive:
        out.append("capacity_ref_not_nameplate_when_aged")
    return tuple(out)


SPECS = {name: replace(spec, rules=spec.rules + _universal_rules(spec))
         for name, spec in SPECS.items()}


# Registry of ingest rules beyond plain field presence. Instrument required
# keys (rinse_protocol, calibrated_at, ...) are validated separately.
_EXTRA_RULES = {
    "sampling_ge_10Hz": lambda meta, spec: (meta.get("instrument") or {}).get("sampling_rate_Hz", 0) >= 10,
    "rest_rule_5tau2": lambda meta, spec: "rest_rule" in (meta.get("protocol") or {}),
    # No default: a file that never states its rate must fail this check, which
    # is the one direction this gate must not get wrong. `sampling_ge_10Hz`
    # above has the same shape, defaulting to 0 so that a missing rate flags.
    "rpt_rate_C20_min": lambda meta, spec: (
        (r := (meta.get("cell_state") or {}).get("rpt_rate")) is not None and r <= 0.05),
    # Absent fails, and an unknown word fails too. The vocabulary is closed
    # because its whole value is that two readers mean the same thing by
    # `coulomb_measured`; a free-text field would be a note, not a basis.
    "soc_basis_recorded": lambda meta, spec: (
        (meta.get("cell_state") or {}).get("soc_basis") in SOC_BASES),
    # Only fires when BOTH are true: the cell has aged, and the reference is the
    # nameplate. A fresh cell's nameplate reference is correct, and an aged
    # cell's MEASURED reference is the thing to aim for — flagging either would
    # train the reader to ignore this.
    "capacity_ref_not_nameplate_when_aged": lambda meta, spec: not (
        ((meta.get("cell_state") or {}).get("age_efc") or 0) > 0
        and (meta.get("cell_state") or {}).get("capacity_ref_source") == "build_record"),
}


def check_rules(exp_type: str, meta: dict) -> list[str]:
    """Return the list of rule descriptions that FAIL for this meta. Empty = passes."""
    spec = SPECS[exp_type]
    return [RULES[r] for r in spec.rules if not _EXTRA_RULES[r](meta, spec)]
