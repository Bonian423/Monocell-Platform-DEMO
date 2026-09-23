"""The abstracted extraction step: a fixed parameter set behind the real interface.

`extract_parameters` stands in for the proprietary extraction and returns
published values. `run` is everything around it that is not proprietary: load
the consumed experiments, call the extraction, and write the result as a
versioned artifact that cites its inputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..cells import load_build
from ..schema.artifacts import write_artifact
from ..schema.write_experiment import experiment_hash, load_experiment

MODULE = "parameters"
BASE_PARAMETER_SET = "Chen2020"

# Scalars from PyBaMM's `Chen2020` set (Chen et al., J. Electrochem. Soc. 167
# (2020) 080534; an LG M50 21700 NMC811/graphite-SiOx cell). Written out as
# literals on purpose: this is the stand-in for the extraction step, and it
# returns the same values whatever the data says. `tests/test_engine_stub.py`
# checks every one against PyBaMM's own copy of the set.
STANDARD_PARAMETERS: dict[str, float] = {
    "Nominal cell capacity [A.h]": 5.0,
    "Lower voltage cut-off [V]": 2.5,
    "Upper voltage cut-off [V]": 4.2,
    "Electrode height [m]": 0.065,
    "Electrode width [m]": 1.58,
    "Negative electrode thickness [m]": 8.52e-05,
    "Separator thickness [m]": 1.2e-05,
    "Positive electrode thickness [m]": 7.56e-05,
    "Negative electrode porosity": 0.25,
    "Separator porosity": 0.47,
    "Positive electrode porosity": 0.335,
    "Negative electrode active material volume fraction": 0.75,
    "Positive electrode active material volume fraction": 0.665,
    "Negative particle radius [m]": 5.86e-06,
    "Positive particle radius [m]": 5.22e-06,
    "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
    "Maximum concentration in positive electrode [mol.m-3]": 63104.0,
    "Initial concentration in electrolyte [mol.m-3]": 1000.0,
}

NOTE = ("Parameter extraction is abstracted in this copy. These are published Chen2020 "
        "values and do not depend on the cell's data.")


def extract_parameters(experiments: list[dict[str, Any]], build: dict[str, Any]) -> dict[str, Any]:
    """A PyBaMM parameter set for a cell. ABSTRACTED: returns fixed literature values.

    `experiments` are loaded experiments (`meta`, `series`, `quality`) and
    `build` is the cell's build record. The full platform derives the parameter
    set from them. This stand-in reads neither and returns
    `STANDARD_PARAMETERS` over the `Chen2020` base set, so everything downstream
    of it (the artifact, staleness, the PyBaMM bridge) runs end to end.

    Returns `{base_parameter_set, pybamm_overrides, source, note}`. The keys of
    `pybamm_overrides` are PyBaMM's own parameter names, which is the contract
    `monocell.simulate` reads.
    """
    del experiments, build  # the stand-in reads neither
    return {
        "base_parameter_set": BASE_PARAMETER_SET,
        "pybamm_overrides": dict(STANDARD_PARAMETERS),
        "source": "abstracted",
        "note": NOTE,
    }


def run(experiment_ids: list[str], out_dir: Path, root: Path | None = None) -> Path:
    """The `rederive` runner: load the inputs, extract, write the parameter file.

    Every consumed experiment is cited with its content hash, which is what
    makes the file go stale when the cell gets new or withdrawn data. The
    filename carries a short hash of those citations, so each distinct input
    set is its own version and earlier versions stay on disk.
    """
    if not experiment_ids:
        raise ValueError("parameters: there are no experiments to derive from")
    loaded = [load_experiment(eid, None, root) for eid in sorted(experiment_ids)]
    cells = {exp["meta"]["cell_id"] for exp in loaded}
    if len(cells) != 1:
        raise ValueError(f"parameters: one parameter file describes one cell, got {sorted(cells)}")
    cell_id = cells.pop()

    inputs = [{"id": exp["meta"]["experiment_id"],
               "hash": experiment_hash(cell_id, exp["meta"]["experiment_id"], root)}
              for exp in loaded]
    version = hashlib.sha256("".join(f"{r['id']}:{r['hash']};" for r in inputs)
                             .encode("utf-8")).hexdigest()[:6]

    by_type: dict[str, list[str]] = {}
    for exp in loaded:
        by_type.setdefault(exp["meta"]["experiment_type"], []).append(exp["meta"]["experiment_id"])

    record = extract_parameters(loaded, load_build(cell_id, root))
    data = {"cell_id": cell_id, "version": f"v{version}", **record, "consumed": by_type}
    params = {"base_parameter_set": record["base_parameter_set"], "source": record["source"],
              "n_inputs": len(inputs)}
    return write_artifact(MODULE, f"parameter_file_{cell_id}_v{version}.json", data, inputs,
                          params, cell_id, out_dir, root)
