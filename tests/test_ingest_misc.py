"""`misc`: the one permissive experiment type, and the one exemption it buys.

Every other type in the store declares its columns and REFUSES a frame that
carries anything else. `misc` declares none and accepts whatever arrives, which
is only safe if the exemption is exactly one type wide — so the negative is
tested here beside the positive, on the same frame.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import BUILD, register_cell
from monocell.autofit import modules_for
from monocell.schema.ingest import INGESTABLE, preview_ingest, read_any, read_misc
from monocell.schema.tables import SPECS
from monocell.schema.write_experiment import load_experiment, write_experiment


@pytest.fixture
def cell(root):
    register_cell("misc_cell", BUILD, root)
    return "misc_cell"


def _meta(**instrument) -> dict:
    return {"producer": {"kind": "real"}, "protocol": {}, "cell_state": {},
            "instrument": instrument}


def test_only_one_type_is_permissive():
    """The flag is a declared property of ONE spec, not a general escape hatch."""
    permissive = {name for name, spec in SPECS.items() if spec.permissive}
    assert permissive == {"misc"}
    # ...and a permissive spec must require nothing, or its permissiveness is a
    # claim the writer would contradict by refusing a file for a missing column.
    assert not any(c.required for c in SPECS["misc"].columns)
    assert "misc" in INGESTABLE


def test_misc_keeps_a_column_the_schema_never_declared(root, cell):
    """The positive: an undeclared column is STORED, and recorded in the meta.

    Recorded is the load-bearing half. For a permissive type the observed
    columns ARE the schema, so a misc file whose columns were only in the
    parquet would have no description a reader could find without opening it.
    """
    frame = pd.DataFrame({"t_s": [0.0, 1.0], "clamp_N": [120.0, 121.0],
                          "operator_note": ["baseline", "warm"]})
    d = write_experiment(cell, "misc", _meta(), frame, root=root)

    stored = load_experiment(d.name, cell, root)
    assert set(stored["series"].columns) == set(frame.columns)
    # The map is what the file contains, dtype and all — not the declared spec.
    assert set(stored["meta"]["observed_columns"]) == set(frame.columns)
    assert stored["meta"]["observed_columns"]["clamp_N"] == "float64"
    # A type nothing consumes is not an error, and it must not claim otherwise.
    assert modules_for("misc") == ()


def test_a_declared_type_still_refuses_that_same_column(root, cell):
    """The negative, on the identical frame — the exemption is one type wide.

    Without this, the relaxation in `write_experiment` could widen silently and
    every type would accept anything, which is the failure that would not show
    up until a consumer read a column it had never been promised.
    """
    frame = pd.DataFrame({"t_s": [0.0], "I_A": [1.0], "V_full_V": [3.0],
                          "T_degC": [25.0], "soc": [0.5], "pulse_id": [1],
                          "seg_type": ["rest"], "clamp_N": [120.0]})
    with pytest.raises(ValueError, match="unknown series columns for hppc"):
        write_experiment(cell, "hppc", _meta(sampling_rate_Hz=10.0), frame, root=root)


def test_misc_reads_a_real_file_and_previews_it(tmp_path):
    """A misc file goes in through the same ingest path every other type uses.

    Including the encoding, which is not hypothetical: the Palmsense4 exports
    this repo already reads are UTF-16-LE, and a `misc` file is exactly the file
    nobody has normalised yet.
    """
    csv = tmp_path / "rig_sweep.csv"
    csv.write_text("t_s,clamp_N,note\n0,120,baseline\n1,121,warm\n", encoding="utf-8")
    frame = read_misc(csv)
    assert list(frame.columns) == ["t_s", "clamp_N", "note"]

    utf16 = tmp_path / "rig_sweep_utf16.csv"
    utf16.write_bytes(csv.read_text(encoding="utf-8").encode("utf-16-le"))
    assert list(read_misc(utf16).columns) == ["t_s", "clamp_N", "note"]

    # `read_any` dispatches to it, so misc is ingestable rather than merely
    # writable, and the preview is the dry run of that same path.
    assert list(read_any("misc", csv).columns) == ["t_s", "clamp_N", "note"]
    preview = preview_ingest("misc", csv)
    assert preview["n_rows"] == 2
    assert preview["quality"]["flags"] == []  # no rules declared, so nothing to fail
