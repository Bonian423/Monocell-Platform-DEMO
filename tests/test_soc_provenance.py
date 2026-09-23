"""Where a `soc` column and a capacity reference came from.

Most series specs carry a bare `soc` column, and four different numbers can be
in it: a charge counted over the NAMEPLATE capacity, a charge counted over a
measured one, a value read off an OCV table, and a simulator's own state
variable, which is exact by construction and is not a measurement. The file
alone cannot say which, so the sidecar has to.

The capacity reference is the same problem one level down. An SOH computed
against the nameplate on a cell that has lost capacity describes a cell that
no longer exists. Both facts are recorded, and both are FLAGGED when absent
rather than defaulted to a value that passes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import EXAMPLE_FILES, EXAMPLES, register_cell
from monocell.schema.tables import CAPACITY_REF_SOURCES, SOC_BASES, SPECS, check_rules


def _meta(**cell_state) -> dict:
    return {"producer": {"kind": "real", "software": "t", "version": "0"},
            "protocol": {"description": "x", "rest_rule": "5*tau2"},
            "instrument": {"sampling_rate_Hz": 100.0},
            "cell_state": cell_state}


# ---------------------------------------------------------------------------
# the SOC basis
# ---------------------------------------------------------------------------

def test_every_type_with_a_soc_column_has_to_say_where_it_came_from():
    """The rule follows from the SPEC's shape rather than from the type's name,
    so a type added later gets it without anybody remembering to add it."""
    for name, spec in SPECS.items():
        has_soc = any(c.name == "soc" for c in spec.columns)
        assert has_soc == ("soc_basis_recorded" in spec.rules), \
            f"{name}: soc column {has_soc}, rule {'soc_basis_recorded' in spec.rules}"


def test_an_absent_basis_is_flagged_and_an_invented_word_is_too():
    """The vocabulary is closed because its value is that two readers mean the
    same thing by `coulomb_measured`. A free-text field would be a note."""
    assert check_rules("hppc", _meta()) == [
        r for r in check_rules("hppc", _meta()) if "soc_basis" in r], \
        "something other than the basis is failing, so this test is not about what it says"
    assert any("soc_basis" in f for f in check_rules("hppc", _meta()))
    assert any("soc_basis" in f for f in check_rules("hppc", _meta(soc_basis="from the rig")))
    assert not any("soc_basis" in f
                   for f in check_rules("hppc", _meta(soc_basis="coulomb_nameplate")))


def test_the_bundled_samples_say_their_soc_is_a_simulators():
    """The sample files were written by a simulator, and a value that is exact
    by construction must not be labelled as counted. `generator_map` is the
    word for it."""
    wrong = []
    for stem, exp_type, _ in EXAMPLE_FILES:
        if not any(c.name == "soc" for c in SPECS[exp_type].columns):
            continue
        sidecar = json.loads((EXAMPLES / f"{stem}.sidecar.json").read_text(encoding="utf-8"))
        basis = sidecar.get("cell_state", {}).get("soc_basis")
        if basis != "generator_map":
            wrong.append(f"{stem}: {basis!r}")
    assert not wrong, wrong


# ---------------------------------------------------------------------------
# the capacity reference
# ---------------------------------------------------------------------------

def test_an_aged_run_referenced_to_the_nameplate_is_flagged():
    """The number has the same shape whichever it is, so only the recorded
    SOURCE can distinguish "this cell's measured capacity" from "what the
    datasheet said when it was new"."""
    flagged = check_rules("rpt", _meta(age_efc=200.0, capacity_ref_source="build_record",
                                       rpt_rate=0.05))
    assert any("NAMEPLATE" in f for f in flagged)

    # a FRESH cell's nameplate reference is correct, and flagging it would
    # teach the reader to ignore the flag
    fresh = check_rules("rpt", _meta(age_efc=0.0, capacity_ref_source="build_record",
                                     rpt_rate=0.05))
    assert not any("NAMEPLATE" in f for f in fresh)

    # ...and so is an aged cell referenced to a value somebody supplied
    decided = check_rules("rpt", _meta(age_efc=200.0, capacity_ref_source="sidecar",
                                       rpt_rate=0.05))
    assert not any("NAMEPLATE" in f for f in decided)


def test_ingest_records_where_the_reference_came_from(root, tmp_path):
    """Recorded at the moment it is decided, because nothing downstream can
    work it out afterwards: a nameplate capacity and a measured one that happen
    to be equal are indistinguishable in the file."""
    from monocell.schema.ingest import ingest_data

    register_cell("ref_cell", {"capacity_Ah": 5.0e-4, "chemistry": "NMC811/graphite (coin)"},
                  root)
    csv = tmp_path / "run.csv"
    n = 20
    charging = np.arange(n) < n // 2
    pd.DataFrame({
        "TestTime/h": np.arange(n, dtype=float) / 60.0,
        "Current/mA": np.where(charging, 0.5, -0.5),
        "Voltage/V": np.linspace(3.0, 4.2, n),
        "Capacity/mAh": np.tile(np.linspace(0.0, 0.5, n // 2), 2),
        "AuxTemp/dC.": np.full(n, 25.0),
        "Cycle-Index": np.ones(n, dtype=int),
        "Step-State": np.where(charging, "RateC", "RateD"),
    }).to_csv(csv, index=False)

    from_build = ingest_data("ref_cell", "cycling", csv,
                             {"protocol": {"description": "x"},
                              "instrument": {"sampling_rate_Hz": 1.0}},
                             root, instrument="landt")
    meta = json.loads((from_build / "meta.json").read_text(encoding="utf-8"))
    assert meta["cell_state"]["capacity_ref_source"] == "build_record"

    from_sidecar = ingest_data("ref_cell", "cycling", csv,
                               {"protocol": {"description": "x"},
                                "instrument": {"sampling_rate_Hz": 1.0},
                                "cell_state": {"capacity_ref_Ah": 4.6e-4}},
                               root, instrument="landt")
    meta = json.loads((from_sidecar / "meta.json").read_text(encoding="utf-8"))
    assert meta["cell_state"]["capacity_ref_source"] == "sidecar"
    assert meta["cell_state"]["capacity_ref_Ah"] == pytest.approx(4.6e-4)


def test_both_vocabularies_are_closed_and_named():
    """Both are read by rules and written by several callers. A word that is in
    one place and not the other is a rule that never fires."""
    assert "coulomb_nameplate" in SOC_BASES and "generator_map" in SOC_BASES
    assert set(CAPACITY_REF_SOURCES) == {"sidecar", "build_record", "measured"}
    assert len(set(SOC_BASES)) == len(SOC_BASES)
