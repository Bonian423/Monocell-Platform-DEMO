"""The one writer function: `write_experiment` is the contract.

Used by the file ingestors and by anything that produces data in-process (a
simulator, a test fixture). Real and synthetic data differ only in
`producer.kind` and the presence of a `ground_truth` sidecar. Append-only:
corrections are new experiment_ids with `derived_from`. Quality is computed at
ingest and lands in `quality.json` + `meta.quality`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from ..cells import data_root, load_build
from .tables import SPECS, check_rules, summary_keys
from .quality import compute_quality

SCHEMA_VERSION = "1.0.0"

# meta.json keys required on every experiment type
META_REQUIRED = ("producer", "protocol", "cell_state")


def experiments_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "experiments"


class ExperimentMissing(FileNotFoundError):
    """The manifest knows this experiment and the disk does not.

    A subclass of `FileNotFoundError` so that anything already catching that
    keeps working, and a distinct type so a caller can tell "this store is
    inconsistent" from "the path you passed is wrong". The store's contract is
    that `data/` is the truth and the manifest is a rebuildable index of it, so
    this condition means the index is stale: a directory was removed by hand, a
    store was copied without its experiments, or a disk filled during a write.

    Callers that iterate a cell's experiments skip the row and report it rather
    than failing the whole listing. `monocell manifest rebuild` is the fix and
    the message says so.
    """

    def __init__(self, experiment_id: str, cell_id: str | None, path: Path):
        self.experiment_id = experiment_id
        self.cell_id = cell_id
        self.path = path
        where = f" for cell {cell_id!r}" if cell_id else ""
        super().__init__(
            f"experiment {experiment_id!r}{where} is in the manifest and not on disk "
            f"({path}). Run `monocell manifest rebuild` to re-index the store."
        )


def _experiment_dir(cell_id: str, experiment_id: str, root: Path | None = None) -> Path:
    return experiments_dir(root) / cell_id / experiment_id


# A tombstone, not a delete. The store is append-only and the data stays on
# disk: an experiment that was wrongly ingested (the sidecar named the wrong
# cell, the rig was mis-set, the file was somebody else's) is still a record of
# what happened, and destroying it destroys the evidence that the correction was
# needed. What retraction changes is what the experiment COUNTS towards.
#
# Beside `meta.json` rather than inside it: `meta.json` is what arrived, and
# rewriting it would make the store's own account of an ingest depend on a
# later opinion about it.
RETRACTED_FILENAME = "retracted.json"


def retract(experiment_id: str, cell_id: str, reason: str, root: Path | None = None,
            by: str = "cli") -> Path:
    """Mark an experiment as not to be used, with a reason. Returns the tombstone.

    Refuses an empty reason. A retraction with no reason cannot be told from a
    mistaken retraction, and the next reader has to guess which.

    Idempotent in the sense that matters: retracting twice overwrites the
    tombstone rather than raising, because the second call is somebody
    correcting the reason.
    """
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("retract: a reason is required — a retracted experiment with no "
                         "reason cannot be told from a mistaken retraction")
    d = _experiment_dir(cell_id, experiment_id, root)
    if not (d / "meta.json").exists():
        raise ExperimentMissing(experiment_id, cell_id, d)
    path = d / RETRACTED_FILENAME
    path.write_text(json.dumps({
        "experiment_id": experiment_id, "cell_id": cell_id, "reason": reason, "by": by,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2), encoding="utf-8")
    return path


def unretract(experiment_id: str, cell_id: str, root: Path | None = None) -> bool:
    """Remove a tombstone. True if there was one.

    A retraction is a judgement and judgements are revised. Nothing else has to
    change: the next `list_experiments` sees the experiment again and the
    staleness scan notices that a consumed input came back.
    """
    path = _experiment_dir(cell_id, experiment_id, root) / RETRACTED_FILENAME
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def retraction_of(cell_id: str, experiment_id: str, root: Path | None = None) -> dict | None:
    """This experiment's tombstone, or None.

    Read from the FILE and never from the manifest. The quality flags are
    cached in a derived table the rebuild populates, and that is fine for a
    flag: an experiment ingested since the last rebuild reading as unflagged
    costs nothing. A retraction read the same way would let a withdrawn
    experiment go on feeding derivations until somebody re-indexed the store.
    (The table is named in `manifest.py` and nowhere else, on purpose;
    `tests/test_manifest.py` holds it to that.)
    """
    path = _experiment_dir(cell_id, experiment_id, root) / RETRACTED_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Unreadable, but present. It still says "do not use this": refusing to
        # read the reason must not turn into treating the experiment as live.
        return {"experiment_id": experiment_id, "cell_id": cell_id,
                "reason": "(the tombstone is unreadable)", "by": "", "at": ""}


def experiment_hash(cell_id: str, experiment_id: str, root: Path | None = None) -> str:
    """The ONE content hash of an experiment. Every consumer delegates here.

    Every derivation and `rederive` anchor staleness on this. If two consumers
    hashed an experiment differently, the store would read permanently stale
    (or never stale), so there is exactly one implementation.
    """
    from .manifest import content_hash

    d = _experiment_dir(cell_id, experiment_id, root)
    if not (d / "meta.json").exists() or not (d / "series.parquet").exists():
        raise ExperimentMissing(experiment_id, cell_id, d)
    parts = [d / "meta.json", d / "series.parquet"]

    # Memoised on what the files ARE, not on having been asked before. The key
    # is every part's (path, mtime_ns, size), so a re-written file misses the
    # cache and is hashed again, which it must be, because this hash is what
    # makes a derived result stale.
    #
    # Worth doing because the staleness scan asks the same question about the
    # same file many times in one pass, and each miss re-reads a parquet file.
    key = tuple((str(f), st.st_mtime_ns, st.st_size)
                for f in parts for st in (f.stat(),))
    hit = _HASH_CACHE.get(key)
    if hit is not None:
        return hit
    value = content_hash(*parts)
    if len(_HASH_CACHE) >= _HASH_CACHE_MAX:
        # Not an LRU: this is a scan cache, and a scan that overflows it is one
        # where nothing would have been re-asked anyway. Dropping the lot is
        # cheaper than tracking recency and cannot serve a wrong answer.
        _HASH_CACHE.clear()
    _HASH_CACHE[key] = value
    return value


# Keyed by file identity, so nothing here can outlive the bytes it describes.
# The bound is a memory guard, not a correctness one.
_HASH_CACHE: dict[tuple, str] = {}
_HASH_CACHE_MAX = 20_000


def _default_experiment_id(exp_type: str, cell_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{exp_type}_{cell_id}_{ts}"


def write_experiment(
    cell_id: str,
    exp_type: str,
    meta: dict[str, Any],
    series: pd.DataFrame,
    sidecar: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """Validate and write one experiment. Returns the experiment directory.

    meta must carry the shared header: experiment_id (auto-filled as
    {type}_{cell}_{ISOts} if absent), experiment_type, schema_version,
    producer, timing, protocol, cell_state, derived_from. Quality is computed
    here and written into both meta.quality and quality.json.
    """
    if exp_type not in SPECS:
        raise ValueError(f"unknown experiment_type {exp_type!r}; known: {sorted(SPECS)}")
    load_build(cell_id, root)  # experiments attach to registered cells only
    spec = SPECS[exp_type]

    meta = dict(meta)
    meta.setdefault("experiment_id", _default_experiment_id(exp_type, cell_id))
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    meta["cell_id"] = cell_id
    meta["experiment_type"] = exp_type
    meta["schema_version"] = SCHEMA_VERSION
    for key in META_REQUIRED:
        if key not in meta:
            raise ValueError(f"meta.{key} is required for {exp_type}")

    producer = meta["producer"]
    if not isinstance(producer, dict):
        raise ValueError("meta.producer must be a dict")
    if producer.get("kind") not in ("real", "synthetic"):
        raise ValueError("producer.kind must be 'real' or 'synthetic'")
    if producer["kind"] == "synthetic":
        if not sidecar or "ground_truth" not in sidecar:
            raise ValueError("synthetic data requires a sidecar with ground_truth")
        if "rng_seed" not in producer:
            raise ValueError("synthetic producer must record rng_seed")

    # series columns per spec
    if not isinstance(series, pd.DataFrame):
        raise TypeError("series must be a pandas DataFrame")
    missing = [col.name for col in spec.columns if col.required and col.name not in series.columns]
    if missing:
        raise ValueError(f"missing required series columns for {exp_type}: {missing}")
    # The ONE relaxation of "every type has a declared schema", and it is gated
    # on the spec's own declared flag rather than on the type's name. A declared
    # type still refuses an undeclared column; that negative is what keeps this
    # exemption from quietly becoming general, and it has its own test.
    unknown = set(series.columns) - {col.name for col in spec.columns}
    if unknown and not spec.permissive:
        raise ValueError(f"unknown series columns for {exp_type}: {sorted(unknown)}")
    if spec.permissive:
        # Recorded, not merely permitted: for a permissive type the columns ARE
        # the schema, so the observed map is the only description the file will
        # ever have. It goes in meta because meta is the shared header every
        # reader already opens, and sorted, so two ingests of the same file
        # produce identical meta and the same `experiment_hash`.
        meta["observed_columns"] = {str(name): str(series[name].dtype)
                                   for name in sorted(series.columns, key=str)}
    if series.empty:
        raise ValueError("series must be non-empty")
    if "t_s" in series.columns and bool(series["t_s"].isna().any()):
        raise ValueError("series must have a valid t_s column (no NaNs)")  # eis is keyed on f_Hz, not t_s

    # instrument block: required keys when the type requires them
    instrument = meta.get("instrument") or {}
    if not isinstance(instrument, dict):
        raise ValueError("meta.instrument must be a dict")
    miss_inst = [k for k in spec.instrument if k not in instrument]
    if miss_inst:
        raise ValueError(f"missing instrument fields for {exp_type}: {miss_inst}")

    # append-only: an experiment_id already on disk must not be silently
    # clobbered; corrections are new experiment_ids with derived_from
    experiment_id = meta["experiment_id"]
    d = _experiment_dir(cell_id, experiment_id, root)
    if (d / "meta.json").exists():
        raise ValueError(f"append-only: experiment {experiment_id} already exists for {cell_id}")

    # the raw frame, exactly as received, before any ingest-side mutation
    # (sum_check inserts a residual column, so raw/ preserves the original)
    raw_dir = d / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    series.to_parquet(raw_dir / "series.parquet", index=False)

    # ingest rules (sampling rate, RPT rate, rest rule) → flags, not errors
    rule_flags = [f"rule: {r}" for r in check_rules(exp_type, meta)]

    # quality at ingest
    quality = compute_quality(exp_type, series, meta)
    quality["flags"] = list(rule_flags) + list(quality["flags"])
    # The header's summary of the quality record, projected by the TYPE's own
    # declaration (`TableSpec.quality_summary`) rather than by a literal tuple,
    # so a type whose headline number is not one of the shared keys still gets
    # it into the header. `if k in quality` because a summary key is per type
    # and a check can return early (a cycling file with no usable discharge
    # does), and a KeyError at ingest would refuse a file over a missing
    # summary field.
    meta["quality"] = {k: quality[k] for k in summary_keys(exp_type) if k in quality}

    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    series.to_parquet(d / "series.parquet", index=False)
    (d / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    if sidecar:
        (d / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    from .manifest import upsert_experiment

    upsert_experiment(
        {
            "experiment_id": experiment_id,
            "cell_id": cell_id,
            "exp_type": exp_type,
            "created_at": meta["created_at"],
            "path": str(d),
            "producer_kind": producer["kind"],
        },
        root,
    )
    return d


def load_meta(experiment_id: str, cell_id: str | None = None,
              root: Path | None = None) -> dict[str, Any]:
    """An experiment's `meta.json` alone, without reading its series.

    Same resolution and same `ExperimentMissing` guard as `load_experiment`,
    and deliberately the same completeness check: an experiment without its
    series is not a readable experiment even if the caller only wants the
    header. Listing a store needs only the meta, and `load_experiment` reads
    the parquet unconditionally.
    """
    d = _resolve_experiment_dir(experiment_id, cell_id, root)
    if not (d / "meta.json").exists() or not (d / "series.parquet").exists():
        raise ExperimentMissing(experiment_id, cell_id, d)
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def _resolve_experiment_dir(experiment_id: str, cell_id: str | None,
                            root: Path | None) -> Path:
    """The directory for an experiment, by cell id or through the manifest."""
    from .manifest import find_experiment

    if cell_id is None:
        rec = find_experiment(experiment_id, root)
        if rec is None:
            raise FileNotFoundError(f"experiment {experiment_id} not in manifest")
        return Path(rec["path"])
    return _experiment_dir(cell_id, experiment_id, root)


def load_experiment(experiment_id: str, cell_id: str | None = None, root: Path | None = None) -> dict[str, Any]:
    """Load an experiment's meta, series, sidecar, quality by id.

    cell_id is optional; when omitted the manifest is used to find the dir.
    """
    d = _resolve_experiment_dir(experiment_id, cell_id, root)
    # Checked before reading rather than letting `read_text` raise, so the
    # error names the experiment and the remedy instead of a path.
    if not (d / "meta.json").exists() or not (d / "series.parquet").exists():
        raise ExperimentMissing(experiment_id, cell_id, d)
    out = {
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
        "series": pd.read_parquet(d / "series.parquet"),
    }
    q = d / "quality.json"
    out["quality"] = json.loads(q.read_text(encoding="utf-8")) if q.exists() else None
    s = d / "sidecar.json"
    out["sidecar"] = json.loads(s.read_text(encoding="utf-8")) if s.exists() else None
    return out
