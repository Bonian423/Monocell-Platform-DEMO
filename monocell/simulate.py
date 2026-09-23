"""The PyBaMM bridge: a cell's parameter file -> a PyBaMM simulation.

A parameter file names a base parameter set that ships with PyBaMM and carries
scalar overrides keyed by PyBaMM's own parameter names. This module loads the
base set, applies the overrides, and runs a constant-current discharge on a
lithium-ion model. The result is written as an artifact whose inputs cite the
parameter file's content hash, so every simulation can be traced to the exact
file it ran from.

PyBaMM is imported inside the functions, so importing `monocell` (and every
ingest or query) stays free of the solver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .schema.artifacts import artifacts_dir, load_artifact, newest_artifact, write_artifact
from .schema.manifest import content_hash

MODULE = "simulation"
MODELS = ("SPM", "SPMe", "DFN")
# A stored curve is for looking at, not for re-analysis, so it is thinned to at
# most this many points. The summary numbers are computed from the full solution.
CURVE_POINTS = 400


def latest_parameter_file(cell_id: str, root: Path | None = None) -> Path | None:
    """The cell's current parameter file, by the store's one "newest" rule."""
    return newest_artifact(cell_id, "parameters", "parameter_file_*.json", root)


def parameter_values(record: dict[str, Any]):
    """`pybamm.ParameterValues` for a parameter-file record.

    The named base set, with the record's overrides applied. A key the base set
    does not define is refused rather than added: PyBaMM would store it as a new
    parameter that no model reads, and the value would silently have no effect.
    """
    import pybamm

    base = record.get("base_parameter_set")
    overrides = record.get("pybamm_overrides")
    if not base or overrides is None:
        raise ValueError("not a parameter file: it needs `base_parameter_set` and "
                         "`pybamm_overrides`")
    values = pybamm.ParameterValues(base)
    unknown = sorted(set(overrides) - set(values.keys()))
    if unknown:
        raise ValueError(f"the parameter file sets {len(unknown)} key(s) that {base} does not "
                         f"define, so no model would read them: {unknown}")
    values.update(dict(overrides))
    return values


def _model(name: str):
    import pybamm

    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose one of {MODELS}")
    return getattr(pybamm.lithium_ion, name)()


def _thin(n: int, points: int = CURVE_POINTS) -> np.ndarray:
    """Indices that keep at most `points` samples, always including the last one."""
    if n <= points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, points).round().astype(int))


def simulate(cell_id: str, *, c_rate: float = 1.0, model: str = "DFN",
             root: Path | None = None, parameter_file: Path | None = None) -> dict[str, Any]:
    """Discharge the cell at `c_rate` from full charge to its lower cut-off.

    Uses the cell's newest parameter file unless `parameter_file` names one.
    Returns `{"artifact", "parameter_file", "parameter_source", "summary",
    "curve"}` and writes the summary and curve as a `simulation/` artifact.
    """
    if c_rate <= 0:
        raise ValueError("c_rate must be positive")
    path = Path(parameter_file) if parameter_file else latest_parameter_file(cell_id, root)
    if path is None:
        raise FileNotFoundError(f"cell {cell_id} has no parameter file yet; "
                                f"run `monocell rederive --cell {cell_id}` first")

    import pybamm  # after the cheap refusals, so they do not pay for the import

    envelope = load_artifact(path)
    record = envelope["data"]
    values = parameter_values(record)
    v_min = float(values["Lower voltage cut-off [V]"])

    experiment = pybamm.Experiment([f"Discharge at {c_rate:g}C until {v_min:g} V"])
    sim = pybamm.Simulation(_model(model), parameter_values=values, experiment=experiment)
    solution = sim.solve(initial_soc=1.0)

    t = solution["Time [s]"].entries
    voltage = solution["Voltage [V]"].entries
    capacity = solution["Discharge capacity [A.h]"].entries
    current = solution["Current [A]"].entries
    power = voltage * current
    energy_Wh = float(np.sum(0.5 * (power[1:] + power[:-1]) * np.diff(t)) / 3600.0)
    summary = {
        "model": model,
        "c_rate": float(c_rate),
        "current_A": float(np.median(current)),
        "capacity_Ah": float(capacity[-1]),
        "energy_Wh": energy_Wh,
        "duration_s": float(t[-1] - t[0]),
        "v_start_V": float(voltage[0]),
        "v_end_V": float(voltage[-1]),
        "v_cutoff_V": v_min,
        "reached_cutoff": bool(voltage[-1] <= v_min + 1e-3),
    }
    keep = _thin(len(t))
    curve = {"t_s": t[keep].tolist(), "V_V": voltage[keep].tolist(),
             "Q_Ah": capacity[keep].tolist()}

    rate_tag = f"{c_rate:g}".replace(".", "p")
    artifact = write_artifact(
        MODULE, f"discharge_{cell_id}_{rate_tag}C_{model}.json",
        {"summary": summary, "curve": curve},
        [{"id": "artifact:parameters", "hash": content_hash(path)}],
        {"model": model, "c_rate": float(c_rate), "parameter_file": envelope["artifact_id"],
         "base_parameter_set": record["base_parameter_set"],
         "pybamm_version": pybamm.__version__},
        cell_id, artifacts_dir(cell_id, root) / MODULE, root)
    return {"artifact": artifact, "parameter_file": envelope["artifact_id"],
            "parameter_source": record.get("source"), "summary": summary, "curve": curve}
