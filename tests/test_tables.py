"""Schema specs: SPECS self-consistency and the ingest-rule checker."""

from __future__ import annotations

import json

import numpy as np

from monocell.schema.tables import SPECS, check_rules

ALL_TYPES = {"three_electrode_ts", "half_cell_ocp", "hppc", "eis", "rpt", "cycling",
             "pressure", "gitt", "misc"}


def test_every_type_is_defined():
    assert ALL_TYPES <= set(SPECS)


def test_specs_self_consistent():
    for name, spec in SPECS.items():
        assert spec.type == name
        names = [c.name for c in spec.columns]
        assert len(names) == len(set(names)), f"{name}: duplicate column names"
        if spec.permissive:
            # Flipped rather than dropped. "At least one required column" is
            # definitionally false for the one type that exists to accept
            # columns it never declared, so for that type the invariant becomes
            # its mirror image, which is a real contract: a permissive spec that
            # required anything would refuse a file for a missing column and
            # contradict its own flag.
            assert not any(c.required for c in spec.columns), \
                f"{name}: is permissive, so it must require nothing"
            continue
        assert any(c.required for c in spec.columns), f"{name}: no required columns"


def test_hppc_three_electrode_columns_optional():
    spec = SPECS["hppc"]
    by_name = {c.name: c for c in spec.columns}
    assert by_name["V_pos_V"].required is False
    assert by_name["V_neg_V"].required is False


def _hppc_meta(**over) -> dict:
    """An HPPC meta that passes every rule the type carries.

    Complete rather than minimal: these tests each assert about ONE rule, so the
    meta has to satisfy the others, or a rule added to the spec later breaks
    three tests for a reason none of them is about.
    """
    meta = {
        "protocol": {"rest_rule": "5*tau2"},
        "instrument": {"sampling_rate_Hz": 100.0},
        "cell_state": {"soc_basis": "coulomb_nameplate"},
    }
    meta.update(over)
    return meta


def test_check_rules_hppc_passes():
    assert check_rules("hppc", _hppc_meta()) == []


def test_check_rules_hppc_flags_low_sampling_rate():
    flags = check_rules("hppc", _hppc_meta(instrument={"sampling_rate_Hz": 5.0}))
    assert len(flags) == 1 and "sampling_rate_Hz >= 10" in flags[0]


def test_check_rules_hppc_flags_missing_rest_rule():
    flags = check_rules("hppc", _hppc_meta(protocol={}))
    assert len(flags) == 1 and "5*tau2" in flags[0]


def test_check_rules_rpt_rate():
    ok = {"protocol": {}, "instrument": {}, "cell_state": {"rpt_rate": 0.05}}
    bad = {"protocol": {}, "instrument": {}, "cell_state": {"rpt_rate": 0.1}}
    assert check_rules("rpt", ok) == []
    assert len(check_rules("rpt", bad)) == 1


def test_check_rules_absent_meta_blocks_default():
    """An empty meta fails every rule the type has, and each for its own reason:
    the sampling rate defaults to 0, the rest rule is absent, and so is the SOC
    basis. None of them defaults to its own passing value, which is the property
    this test is really about."""
    flags = check_rules("hppc", {})
    assert "sampling_rate_Hz >= 10 (honest R0)" in flags
    assert "protocol must record the 5*tau2 rest rule and the tau2 used" in flags
    assert any("soc_basis" in f for f in flags)
    assert len(flags) == len(SPECS["hppc"].rules) - 1, (
        "one rule passed on an empty meta, which means it defaults to passing: " + repr(flags))


def test_an_unrecorded_rpt_rate_is_flagged_rather_than_assumed_compliant():
    """The rule is "a C/10 RPT is flagged at ingest", and a gate that defaults
    to the passing value can never do that: a real cycler export that never
    states its rate would be certified compliant.

    `sampling_ge_10Hz` has the same shape: it defaults to 0 and flags. A
    missing field has to fail in the same direction for both gates.
    """
    from monocell.schema.tables import check_rules

    assert check_rules("rpt", {"cell_state": {"rpt_rate": 0.05}}) == []
    slow = check_rules("rpt", {"cell_state": {"rpt_rate": 0.1}})
    assert len(slow) == 1 and "C/20" in slow[0]

    for absent in ({"cell_state": {}}, {"cell_state": {"rpt_rate": None}}, {}):
        flags = check_rules("rpt", absent)
        assert len(flags) == 1, f"an unrecorded rate must flag: {absent}"


def test_ingest_does_not_fill_in_an_unrecorded_rpt_rate(cell, root, tmp_path):
    """...and ingest must not invent the field before the rule runs.

    The gate above is only reachable if `rpt_rate` is still missing by the time
    `check_rules` sees the meta. If ingest defaulted it, a file ingested through
    the real path would arrive already compliant however right the rule was.
    Both halves have to be right for either to matter, which is why this is a
    second test and not a second assertion.
    """
    import pandas as pd

    from monocell.schema.ingest import ingest_data

    n = 40
    csv = tmp_path / "rpt.csv"
    pd.DataFrame({
        "time_s": np.arange(n, dtype=float),
        "current_A": np.full(n, -0.5),
        "voltage_V": np.linspace(4.1, 3.4, n),
        "temperature_C": np.full(n, 25.0),
        "capacity_Ah": np.linspace(0.0, 0.5, n),
        "cycle_index": np.zeros(n, dtype=int),
    }).to_csv(csv, index=False)

    d = ingest_data(cell, "rpt", csv, sidecar=None, root=root)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert "rpt_rate" not in meta["cell_state"], (
        "ingest defaulted the rate, so the C/20 gate can never fire on a real file")
    assert any("C/20" in f for f in meta["quality"]["flags"]), (
        "an unrecorded rate must reach the experiment's own quality flags")
