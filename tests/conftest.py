"""Shared fixtures. Every test writes into its own tmp data root — no test
touches the CWD-relative `data/` store or $MONOCELL_DATA.

**The BLAS thread pin at the top of this file is load-bearing, and it works
only because of WHERE it is.** PyBaMM's solves go through numpy, numpy loads
OpenBLAS, and OpenBLAS reads its thread count from the environment when the
shared library is initialised, which `import numpy` is what triggers. So the
pin has to run before that import, which is why it sits above them rather than
in a fixture or a `pytest_configure`: by the time either of those runs, numpy
has already been imported and the setting is inert.

It matters when the suite runs under xdist (`pytest -n 4`): every worker would
otherwise default to one BLAS thread per core, and OpenBLAS's thread pool
busy-waits between parallel regions, so oversubscription does not merely fail
to help. Measured on the solve-heavy files of the full platform's suite, eight
unpinned workers ran 12.7x slower than serial, and pinned serial equalled
unpinned serial, so the pin costs a single-process run nothing.

`setdefault`, so exporting OMP_NUM_THREADS yourself still wins.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import shutil
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import pytest

from monocell.cells import register_cell

BUILD = {"capacity_Ah": 5.0, "chemistry": "NMC811/graphite (synthetic)"}

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "lab_exports"

# Every bundled sample file (its path under `EXAMPLES`, without the suffix),
# with the type and profile it is ingested under. `examples/lab_exports/INGEST.md`
# does the same by hand: `cycler/` as a folder, the other two file by file.
EXAMPLE_FILES: tuple[tuple[str, str, str | None], ...] = (
    *((f"cycler/landt_rpt_{efc:04d}efc", "rpt", "landt") for efc in (0, 200, 400, 600, 800, 1000)),
    ("cycler/hppc_soc_sweep", "hppc", None),
    ("cycler/gitt_15C", "gitt", None),
    ("cycler/three_electrode_0p5C", "three_electrode_ts", None),
    ("cycler/three_electrode_1C", "three_electrode_ts", None),
    ("cycler/three_electrode_2C", "three_electrode_ts", None),
    ("eis/eis_fresh", "eis", None),
    ("pressure/pressure_cycle", "pressure", None),
)


@pytest.fixture(autouse=True)
def _default_store_is_private(tmp_path_factory, monkeypatch):
    """A command run without `--data-root` falls back to `$MONOCELL_DATA`, then
    to `data/` in the working directory. The CLI's journal writes there even
    for `inspect`, so every test gets a private fallback, outside its own
    `tmp_path` so a test can still check that a command wrote nothing there."""
    monkeypatch.setenv("MONOCELL_DATA", str(tmp_path_factory.mktemp("default_store")))


@pytest.fixture
def root(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def cell(root):
    register_cell("test_cell", BUILD, root)
    return "test_cell"


# ---------------------------------------------------------------------------
# The store cache — build a store ONCE per run, copy it per test.
#
# Store construction, not assertion, is where a store-backed test spends its
# time: ingesting the whole sample campaign takes seconds, and copying the
# built store takes a few milliseconds.
#
# It has to be a cache ON DISK rather than a session-scoped fixture, because
# under xdist every worker is its own process and a session fixture would build
# once per worker. `tmp_path_factory.getbasetemp()` is per-worker
# (`.../pytest-123/popen-gw3`); its PARENT is the run's own directory, shared by
# every worker of THAT run and by no other, which is xdist's documented handle
# for exactly this.
#
# Serially there is no `popen-gwN` level, so the parent would be
# `pytest-of-<user>`, shared across RUNS, which is a stale-cache generator.
# `worker_id` is what tells the two apart, and it is why the fixture below
# branches on it rather than always taking `.parent`.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def worker_id(request):
    """xdist's worker id, or "master" when the suite runs in one process.

    Defined here so the suite also runs without pytest-xdist installed. It
    reads the same `workerinput` xdist sets on each worker, so the two agree.
    """
    return getattr(request.config, "workerinput", {}).get("workerid", "master")


@pytest.fixture(scope="session")
def store_cache(tmp_path_factory, worker_id):
    """The directory built stores are published into. Fresh every run."""
    base = tmp_path_factory.getbasetemp()
    if worker_id != "master":
        base = base.parent
    d = base / "store_cache"
    d.mkdir(exist_ok=True)
    return d


def _publish(store_cache: Path, worker_id: str, key: str, build) -> Path:
    """Build `key` into the cache if it is not there yet; return the cached root."""
    try:
        src = inspect.getsource(build)
    except (OSError, TypeError):  # a lambda from an exec'd string, say
        src = repr(build)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]
    final = store_cache / f"{key}-{digest}"
    if not final.exists():
        scratch = store_cache / f".build-{key}-{digest}-{worker_id}"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        build(scratch / "data")
        try:
            scratch.rename(final)
        except OSError:
            # another worker published first; its bytes are as good as ours
            shutil.rmtree(scratch, ignore_errors=True)
    return final


def _copy_out(final: Path, dst: Path) -> Path:
    """A private copy of a cached store, with a usable manifest."""
    shutil.copytree(final / "data", dst)
    # A store is NOT relocatable: `manifest.duckdb` records each experiment's
    # absolute `path`, so a copied store's index still points at the directory
    # it was built in. Rebuilding is the store's own documented answer to that
    # (the manifest is a query index, rebuildable from the files), and it is
    # what makes the copy a real store rather than a directory holding the bytes.
    from monocell.schema.manifest import rebuild_manifest

    rebuild_manifest(dst)
    return dst


@pytest.fixture
def cached_store(store_cache, tmp_path, worker_id):
    """`cached_store(key, build) -> Path` — a PRIVATE copy of a store built once.

    `build(root)` takes the data root to populate and returns nothing; it runs
    at most once per run across all workers. Every caller gets its own
    `copytree` of the result, so a test may mutate what it receives freely.

    The key names the RECIPE, not the test: two files wanting the same store
    should pass the same key and share the entry. The builder's own source is
    hashed into it, so editing a recipe cannot silently serve the old bytes.

    The race between workers is settled by `Path.rename`, which refuses to
    replace an existing directory rather than clobbering it: the first worker
    to finish publishes, the rest discard their copy and read the winner's.
    """
    def _get(key: str, build) -> Path:
        return _copy_out(_publish(store_cache, worker_id, key, build), tmp_path / "data")

    return _get


def build_example_store(root: Path) -> None:
    """Register `demo` and ingest every bundled sample file into it."""
    from monocell.schema.ingest import ingest

    register_cell("demo", BUILD, root)
    for stem, exp_type, instrument in EXAMPLE_FILES:
        ingest("demo", exp_type, EXAMPLES / f"{stem}.csv", EXAMPLES / f"{stem}.sidecar.json",
               root, instrument=instrument)


@pytest.fixture
def example_store(cached_store) -> Path:
    """A private store holding the whole sample campaign for cell `demo`."""
    return cached_store("examples", build_example_store)


# ---------------------------------------------------------------------------
# A toy module graph, for the re-derive machinery
#
# The shipped graph holds one module, and one module cannot show a run order,
# an artifact edge or a scoped pass. These three can. Their runners compute
# nothing: each writes an artifact that records what it consumed, which is all
# the staleness checker reads.
# ---------------------------------------------------------------------------

TOY_GRAPH: dict[str, dict] = {
    # one artifact per EIS experiment
    "spectra": {"input_experiments": ("eis",), "artifact_glob": "spectrum_*.json",
                "mode": "per_experiment"},
    # one artifact per HPPC experiment
    "pulses": {"input_experiments": ("hppc",), "artifact_glob": "pulses_*.json",
               "mode": "per_experiment"},
    # one artifact for the cell, over every EIS and HPPC, which also reads the
    # newest `spectra` artifact. The name sorts BEFORE its upstream's, so an
    # alphabetical run order would be the wrong one.
    "cell_model": {"input_experiments": ("eis", "hppc"), "artifact_glob": "model_*.json",
                   "mode": "all_experiments", "input_artifacts": ("spectra",),
                   "artifact_input_globs": {"spectra": "spectrum_*.json"}},
}


def _toy_inputs(cell_id: str, ids, root) -> list[dict[str, str]]:
    from monocell.schema.write_experiment import experiment_hash

    return [{"id": eid, "hash": experiment_hash(cell_id, eid, root)} for eid in ids]


def _toy_per_experiment(module: str, prefix: str):
    def run(ids, out_dir, root=None):
        from monocell.schema.artifacts import write_artifact

        (eid,) = ids
        cell_id = Path(out_dir).parent.name
        return write_artifact(module, f"{prefix}_{eid}.json", {}, _toy_inputs(cell_id, ids, root),
                              {}, cell_id, Path(out_dir), root)

    return run


def _toy_cell_model(ids, out_dir, root=None):
    from monocell.schema.artifacts import artifact_input_env, write_artifact

    cell_id = Path(out_dir).parent.name
    inputs = _toy_inputs(cell_id, ids, root) + artifact_input_env(
        cell_id, root, [("spectra", "spectrum_*.json")])
    version = hashlib.sha256(repr(inputs).encode("utf-8")).hexdigest()[:6]
    return write_artifact("cell_model", f"model_{cell_id}_v{version}.json", {}, inputs, {},
                          cell_id, Path(out_dir), root)


@pytest.fixture
def toy_graph(monkeypatch) -> dict[str, dict]:
    """Swap `MODULE_GRAPH` and its runners for the toy graph, for one test.

    Returns the graph in use, a private copy a test may edit.
    """
    from monocell import autofit, rederive

    graph = {name: dict(spec) for name, spec in TOY_GRAPH.items()}
    monkeypatch.setattr(rederive, "MODULE_GRAPH", graph)
    monkeypatch.setattr(autofit, "MODULE_GRAPH", graph)
    monkeypatch.setattr(rederive, "RUNNERS", {
        "spectra": _toy_per_experiment("spectra", "spectrum"),
        "pulses": _toy_per_experiment("pulses", "pulses"),
        "cell_model": _toy_cell_model,
    })
    return graph


def write_eis(cell_id: str, root: Path) -> str:
    """Write one minimal EIS experiment; return its id."""
    from monocell.schema import write_experiment

    d = write_experiment(cell_id, "eis", eis_meta(), eis_series(), root=root)
    return d.name


def write_hppc(cell_id: str, root: Path) -> str:
    """Write one minimal HPPC experiment; return its id."""
    from monocell.schema import write_experiment

    d = write_experiment(cell_id, "hppc", hppc_meta(), hppc_series(), root=root)
    return d.name


# ---------------------------------------------------------------------------
# minimal valid frames, for tests that need a store but not a campaign
# ---------------------------------------------------------------------------

def hppc_meta(sampling_rate_hz=100.0, rest_rule="5*tau2") -> dict:
    """Minimal valid hppc meta (real producer — no sidecar/rng_seed needed)."""
    return {
        "producer": {"kind": "real", "software": "test", "version": "0"},
        "protocol": {
            "description": "test HPPC",
            "pulse_current_A": 5.0,
            "pulse_duration_s": 60.0,
            "relax_duration_s": 300.0,
            "rest_rule": rest_rule,
        },
        "cell_state": {"rpt_index": 0, "rpt_rate": 0.05},
        "instrument": {"sampling_rate_Hz": sampling_rate_hz},
    }


def hppc_series(n=12, with_vneg=False, with_vpos=False, seg="rest") -> pd.DataFrame:
    """Minimal valid hppc series frame."""
    data = {
        "t_s": np.arange(n, dtype=float) * 0.01,
        "I_A": np.zeros(n),
        "V_full_V": np.full(n, 3.7),
        "T_degC": 25.0,
        "soc": 0.5,
        "pulse_id": np.ones(n, dtype=np.int32),
        "seg_type": np.full(n, seg),
    }
    if with_vpos:
        data["V_pos_V"] = np.full(n, 4.2)
    if with_vneg:
        data["V_neg_V"] = np.full(n, 0.5)
    return pd.DataFrame(data)


def eis_meta() -> dict:
    """Minimal valid eis meta (real producer)."""
    return {
        "producer": {"kind": "real", "software": "test", "version": "0"},
        "protocol": {"description": "test EIS", "excitation": "sine", "soc_fixed": True},
        "cell_state": {"rpt_index": 0, "rpt_rate": 0.05},
        "instrument": {
            "potentiostat": "test_pstat",
            "amplitude_mV": 5.0,
            "f_min_Hz": 0.01,
            "f_max_Hz": 1e4,
            "points_per_decade": 10,
        },
    }


def eis_spectrum(n=61, noise=0.0, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """`(f_Hz, Z)`: an R0 + 3RC spectrum, optionally with modulus-relative noise."""
    f = np.geomspace(0.01, 1e4, n)
    w = 2.0 * np.pi * f
    z = 2.0e-3 + 1.0e-3 / (1 + 1j * w * 1e-3) + 0.8e-3 / (1 + 1j * w * 1e-1) + 0.5e-3 / (1 + 1j * w * 3.0)
    if noise:
        rng = np.random.default_rng(seed)
        z = z + np.abs(z) * noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return f, z


def eis_series(n=61) -> pd.DataFrame:
    """Noiseless R0+3RC spectrum as a valid eis frame (no t_s column)."""
    f, z = eis_spectrum(n)
    return pd.DataFrame(
        {
            "f_Hz": f,
            "Z_re_Ohm": z.real,
            "Z_im_Ohm": z.imag,
            "Z_mag_Ohm": np.abs(z),
            "Z_phase_deg": np.degrees(np.angle(z)),
            "soc": 0.5,
            "V_dc_V": 3.7,
            "T_degC": 25.0,
            "electrode": "full",
        }
    )
