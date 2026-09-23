"""The PyBaMM bridge: a parameter file in, a simulation artifact out."""

from __future__ import annotations

import csv
import numbers

import pytest

from conftest import BUILD, register_cell, write_eis
from monocell.cli import main as cli_main
from monocell.rederive import rederive
from monocell.schema.artifacts import load_artifact
from monocell.schema.manifest import content_hash
from monocell.simulate import CURVE_POINTS, latest_parameter_file, parameter_values, simulate


@pytest.fixture
def derived_cell(root):
    """A registered cell with one experiment and its parameter file."""
    register_cell("sim_cell", BUILD, root)
    write_eis("sim_cell", root)
    rederive("sim_cell", root)
    return "sim_cell"


def test_parameter_values_is_the_base_set_with_the_overrides_applied():
    """Every scalar the record does not override is the base set's own."""
    import pybamm

    overrides = {"Nominal cell capacity [A.h]": 4.0, "Negative electrode porosity": 0.3}
    values = parameter_values({"base_parameter_set": "Chen2020", "pybamm_overrides": overrides})
    base = pybamm.ParameterValues("Chen2020")

    assert set(values.keys()) == set(base.keys())
    for key in base.keys():
        if key in overrides:
            assert values[key] == overrides[key], key
        elif isinstance(base[key], numbers.Number):
            assert values[key] == base[key], key


def test_a_key_the_base_set_does_not_define_is_refused():
    """PyBaMM would store it as a new parameter that no model reads, and the
    value would have no effect without any sign of it."""
    record = {"base_parameter_set": "Chen2020",
              "pybamm_overrides": {"Negative electrode porosity": 0.3,
                                   "Negative electrode porosty": 0.3}}
    with pytest.raises(ValueError, match="Negative electrode porosty"):
        parameter_values(record)


def test_a_record_that_is_not_a_parameter_file_is_refused():
    with pytest.raises(ValueError, match="not a parameter file"):
        parameter_values({"pybamm_overrides": {}})


def test_no_parameter_file_is_refused_with_the_command_that_makes_one(root):
    register_cell("bare", BUILD, root)
    with pytest.raises(FileNotFoundError, match="monocell rederive --cell bare"):
        simulate("bare", root=root)
    with pytest.raises(ValueError, match="c_rate"):
        simulate("bare", c_rate=0.0, root=root)


@pytest.mark.slow
def test_a_discharge_ends_at_the_cut_off_and_cites_its_parameter_file(derived_cell, root):
    out = simulate(derived_cell, c_rate=1.0, model="SPM", root=root)
    summary = out["summary"]
    assert summary["reached_cutoff"]
    assert summary["v_end_V"] <= summary["v_cutoff_V"] + 1e-3 < summary["v_start_V"]
    assert summary["capacity_Ah"] > 0 and summary["duration_s"] > 0
    assert out["parameter_source"] == "abstracted"

    curve = out["curve"]
    assert 2 < len(curve["t_s"]) <= CURVE_POINTS
    assert curve["V_V"][-1] == summary["v_end_V"], "the thinned curve dropped its last point"

    env = load_artifact(out["artifact"])
    parameter_file = latest_parameter_file(derived_cell, root)
    assert env["inputs"] == [{"id": "artifact:parameters", "hash": content_hash(parameter_file)}]
    assert env["params"]["parameter_file"] == load_artifact(parameter_file)["artifact_id"]


@pytest.mark.slow
def test_cli_simulate_prints_a_summary_and_writes_the_curve(derived_cell, root, tmp_path,
                                                            capsys):
    dest = tmp_path / "curve.csv"
    assert cli_main(["simulate", "--cell", derived_cell, "--model", "SPM", "--c-rate", "0.5",
                     "--out", str(dest), "--data-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "SPM discharge at 0.5C" in out and "abstracted" in out

    with dest.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["t_s", "V_V", "Q_Ah"]
    assert len(rows) > 2
