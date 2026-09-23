"""Artifact writer: the provenance contract for every derived output.

Every artifact carries schema_version, producer_version, `inputs` (ids +
content hashes of what it consumed) and `params`. Artifacts are the only
cross-module data interface; the re-derivation checker compares stored input
hashes against the current ones.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .. import __version__
from ..cells import data_root
from .manifest import content_hash, upsert_artifact

SCHEMA_VERSION = "1.0.0"


def artifacts_dir(cell_id: str, root: Path | None = None) -> Path:
    return (root or data_root()) / "artifacts" / cell_id


def write_artifact(
    module: str,
    filename: str,
    artifact_data: dict[str, Any],
    inputs: list[dict[str, str]],
    params: dict[str, Any],
    cell_id: str,
    out_dir: Path,
    root: Path | None = None,
) -> Path:
    """Write one artifact with the standard provenance envelope.

    `inputs` is [{"id": ..., "hash": sha256-of-consumed-files}], the
    re-derivation anchor. Returns the file path written.
    """
    envelope = {
        "artifact_id": f"{module}/{filename.removesuffix('.json')}",
        "module": module,
        "cell_id": cell_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "producer_version": __version__,
        "inputs": inputs,
        "params": params,
        "data": artifact_data,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    upsert_artifact(
        {
            "artifact_id": envelope["artifact_id"],
            "cell_id": cell_id,
            "module": module,
            "created_at": envelope["created_at"],
            "path": str(path),
        },
        root,
    )
    return path


def load_artifact(path: Path) -> dict[str, Any]:
    """One artifact envelope, parsed. Raises on a missing or malformed file.

    The STRICT reader. It is what a caller that has just chosen a path wants:
    the file was found, so anything wrong with it is news.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def read_artifact_or_none(path: Path) -> dict[str, Any] | None:
    """One artifact envelope, or `None` when the file cannot be read.

    The TOLERANT reader, and the only place tolerant artifact reading is
    implemented, so which failures are survivable does not depend on which
    caller happened to read the file. An artifact truncated by a kill during a
    write must not take down a whole listing.

    `ValueError` is the catch that does the work: `json.JSONDecodeError` is a
    subclass of it, and truncation is what a store actually suffers. A missing
    file is the ordinary state of a cell nobody has derived anything for yet,
    not an error.

    Tolerance is a property of READING, not a policy of the caller. A caller
    that must fail loudly uses `load_artifact` instead and says so at its own
    call site.
    """
    try:
        return load_artifact(path)
    except (OSError, ValueError):
        return None


# --- "which artifact is current" — ONE rule, for the whole platform ----------
#
# `created_at` alone is not a total order: two artifacts written in the same
# microsecond tie, and a tie resolved by `glob` order is a tie resolved by the
# filesystem, so "newest" would be a property of the directory listing rather
# than of the store. The rule is `(created_at, name)`, and it lives here, next
# to the writer that stamps the `created_at` it sorts on, so every picker in the
# platform agrees about what "newest" means.


def envelope_order_key(name: str, envelope: dict[str, Any]) -> tuple[str, str]:
    """The artifact total order for an envelope already in memory.

    Split from `artifact_order_key` so a caller that has already read a whole
    module directory can sort by the same rule without re-reading every file.
    """
    return (envelope.get("created_at") or "", name)


def artifact_order_key(path: Path) -> tuple[str, str]:
    """The artifact total order for a path on disk: `(created_at, name)`.

    A file that will not parse sorts OLDEST (empty `created_at`) rather than
    raising. This function is called while sorting a whole directory, so one
    corrupt file would otherwise take down every reader that picks a newest
    artifact, and sorting it last is what makes it lose to any readable one.
    """
    env = read_artifact_or_none(path)
    return envelope_order_key(path.name, env or {})


def newest_artifact(cell_id: str, module: str, pattern: str,
                    root: Path | None = None) -> Path | None:
    """Newest artifact matching `pattern` under `cell_id`'s `module` dir.

    Resolved through `artifacts_dir`, so `root=None` means the configured data
    root exactly as it does for every writer. A reader and a writer that
    disagreed about where the store is would be exactly the class of bug this
    section exists to remove.
    """
    directory = artifacts_dir(cell_id, root) / module
    if not directory.exists():
        return None
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=artifact_order_key)


# The cell-level RECORD file each module writes: the one file a consumer reads
# to ask "what does this module currently say about this cell?". Declared in the
# layer that owns the store layout, because `rederive` needs it to decide what
# has gone stale and a module that consumes another module's record needs the
# same answer. A second copy of this map would be free to disagree with it.
MODULE_RECORDS: tuple[tuple[str, str], ...] = (
    ("parameters", "parameter_file_*.json"),
)


def record_pattern(module: str) -> str | None:
    """The declared record glob for `module`, or None if the module has none."""
    for name, pattern in MODULE_RECORDS:
        if name == module:
            return pattern
    return None


def artifact_input_env(cell_id: str, root: Path | None,
                       declarations: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """`[{"id": "artifact:<module>", "hash": ...}]` for the newest of each record.

    The `artifact:` prefix cannot collide with an experiment id, and the same
    records go into an envelope's `inputs` AND its version hash, so the two can
    never disagree about what a file was derived from.

    `declarations` is `(module, glob)` pairs, taken as an ARGUMENT rather than
    read from a module-level tuple, because which file of a module was consumed
    is the CALLER's knowledge, not the store's. Declaring coverage of a file the
    caller never opened would be worse than declaring none.

    A module with no matching file is SKIPPED rather than hashed as empty:
    absent is not the same as present-and-empty, and the caller's own staleness
    logic depends on the difference.
    """
    out: list[dict[str, str]] = []
    for module, pattern in declarations:
        path = newest_artifact(cell_id, module, pattern, root)
        if path is not None:
            out.append({"id": f"artifact:{module}", "hash": content_hash(path)})
    return out
