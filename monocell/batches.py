"""The batch record: the design a lot of cells share.

A **batch** is a physical lot: one design (footprint, stack, coatings, tabs,
the operating window the cell is built for) plus the cells made to it, each
with its own serial number. It answers the question the build record answers
only one cell at a time: *what cell is this, and what else came off the same
line?*

An engineer states the design once ("NMC811/graphite pouch, 12 electrode
pairs, 80 µm anode coating") and then registers twenty cells against it.

Storage::

    {root}/batches/{batch_id}/batch.json    the design spec + membership
    {root}/cells/{cell_id}/build.json       unchanged shape; gains the optional
                                            `serial` and `batch_id`

The build record keeps its **two required fields** (`capacity_Ah`,
`chemistry`), and every field this module adds is optional, so an existing
store's `build.json` files stay valid.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cells import data_root

BATCH_SCHEMA_VERSION = "1.0.0"

# A batch id becomes a directory name.
_ID_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# The design block. Every field is OPTIONAL except the two the platform's own
# C-rate anchor and chemistry label need; a batch registered with only those two
# is valid. Required-ness deliberately mirrors `cells.BUILD_FIELDS`.
DESIGN_FIELDS: dict[str, tuple[str, bool]] = {
    # --- the two the platform needs ---------------------------------------
    "capacity_Ah": ("nominal capacity (Ah). The C-rate anchor", True),
    "chemistry": ("e.g. 'NMC811/graphite', positive/negative", True),
    # --- electrical --------------------------------------------------------
    "n_series": ("cells connected in series in the battery", False),
    "lower_cutoff_V": ("discharge voltage cutoff (V)", False),
    "upper_cutoff_V": ("charge voltage cutoff (V)", False),
    # --- footprint (metres) ------------------------------------------------
    "height": ("pouch dimension along z (m); tabs sit on the z = height edge", False),
    "width": ("pouch dimension along y (m)", False),
    "thickness_total": ("outer pouch thickness (m), packaging included", False),
    # --- stack -------------------------------------------------------------
    "layer_count": ("electrode pairs in the stack", False),
    "coating_thicknesses_um": ("per-layer thicknesses in µm, keyed L_cn/L_n/L_s/L_p/L_cp "
                               "(the unit a coating is specified and measured in)", False),
    # --- tabs --------------------------------------------------------------
    "tab_width": ("tab width along y (m)", False),
    "neg_tab_y_centre": ("negative tab centre y-coordinate (m)", False),
    "pos_tab_y_centre": ("positive tab centre y-coordinate (m)", False),
    "tabs_format": ("tab geometry/format as built (free text, alongside the numbers)", False),
    # --- thermal -----------------------------------------------------------
    "ambient_temperature_K": ("ambient temperature (K)", False),
    "cooling_regions": ("cooling geometry: a list of region dicts", False),
    # --- as-built record ---------------------------------------------------
    "electrode_source": ("where the electrodes came from", False),
    "areal_loading_mAh_cm2": ("measured electrode areal loading (mAh/cm²)", False),
    "porosities": ("per-layer porosities", False),
    "np_ratio": ("negative-to-positive capacity ratio", False),
    "electrolyte_volume_per_Ah_ml": ("electrolyte volume per Ah", False),
    "separator_type": ("separator material/type", False),
    "formation_protocol": ("formation protocol description", False),
    "stack_pressure_kPa": ("applied stack pressure", False),
    "pat_rinse_protocol": ("rinse protocol for reference-electrode test cells", False),
    "rig": ("rig data: fixture stiffness (measured, not from the drawing), load-cell calibration", False),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def batches_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "batches"


def batch_path(batch_id: str, root: Path | None = None) -> Path:
    return batches_dir(root) / batch_id / "batch.json"


def _check_batch_id(batch_id: str) -> str:
    if not batch_id or not _ID_OK.match(batch_id):
        raise ValueError(
            f"invalid batch id {batch_id!r}: start with a letter or digit and use letters, "
            "digits, '.', '_' or '-' (it becomes a directory name)"
        )
    return batch_id


def _atomic_write(path: Path, text: str) -> None:
    """The platform's one atomic write, imported lazily so `import
    monocell.batches` stays cheap."""
    from ._fs import atomic_write_text

    atomic_write_text(path, text)


def _check_version(raw: dict[str, Any], where: str) -> None:
    """Refuse a record written by a NEWER schema, naming where it came from.

    Compared against THIS module's own constant: two independently versioned
    schemas compared against one shared version string would let a future bump
    of either silently admit a record the other cannot read.
    """
    v = str(raw.get("schema_version", BATCH_SCHEMA_VERSION))
    if v > BATCH_SCHEMA_VERSION:
        raise ValueError(
            f"{where} schema_version {v} is newer than this platform understands "
            f"({BATCH_SCHEMA_VERSION}); upgrade monocell rather than reading it half-way"
        )


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def save_batch(batch_id: str, design: dict[str, Any], root: Path | None = None, *,
               name: str | None = None, note: str | None = None,
               cells: list[dict[str, Any]] | None = None, update: bool = False) -> Path:
    """Write a batch record. Returns the path written.

    Re-saving an identical record is idempotent (a script may re-save into the
    same root). A DIFFERENT record is an identity change and raises unless the
    caller passes `update=True`: the deliberate-edit path, where the caller has
    seen the current record and means to replace it. This is `register_cell`'s
    rule, for the same reason: a design other results were derived against must
    not change without anyone saying so.
    """
    _check_batch_id(batch_id)
    unknown = set(design) - set(DESIGN_FIELDS)
    if unknown:
        raise ValueError(f"unknown design fields for batch {batch_id}: {sorted(unknown)}")
    missing = [f for f, (_, req) in DESIGN_FIELDS.items() if req and design.get(f) is None]
    if missing:
        raise ValueError(f"missing required design fields for batch {batch_id}: {missing}")
    if float(design["capacity_Ah"]) <= 0:
        raise ValueError("capacity_Ah must be positive")

    path = batch_path(batch_id, root)
    now = _now()
    record: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "name": name or batch_id,
        "created_at": now,
        "updated_at": now,
        "design": dict(design),
        "cells": list(cells or []),
        "note": note,
    }
    if path.exists():
        existing = load_batch(batch_id, root)
        created = existing.get("created_at", now)
        if name is None:
            record["name"] = existing.get("name", batch_id)
        if note is None:
            record["note"] = existing.get("note")
        if cells is None:
            record["cells"] = existing.get("cells", [])
        record["created_at"] = created
        comparable_new = {k: v for k, v in record.items() if k not in ("updated_at", "schema_version")}
        comparable_old = {k: v for k, v in existing.items() if k not in ("updated_at", "schema_version")}
        if comparable_new == comparable_old:
            # Identical: do not rewrite. `updated_at` means "when this record
            # last changed", and a save that changed nothing must not move it:
            # a rewrite for a timestamp looks like a change to every reader that
            # hashes the file.
            return path
        if not update:
            raise ValueError(
                f"batch {batch_id} already exists with a different record "
                f"(differing: {sorted(k for k in set(comparable_new) | set(comparable_old) if comparable_new.get(k) != comparable_old.get(k))})"
            )
        record["updated_at"] = now

    _atomic_write(path, json.dumps(record, indent=2))
    return path


def load_batch(batch_id: str, root: Path | None = None) -> dict[str, Any]:
    """Read a batch record. Raises FileNotFoundError if it does not exist.

    The returned dict is exactly what is on disk, with no injected bookkeeping
    keys. `save_batch` compares a freshly built record against this one to
    decide whether a re-save is a no-op, and an extra key here would make every
    save look like a change and every `update=False` call raise. Use
    `batch_path(batch_id, root)` when the path is wanted.
    """
    path = batch_path(batch_id, root)
    if not path.exists():
        raise FileNotFoundError(f"batch {batch_id!r} is not registered")
    raw = json.loads(path.read_text(encoding="utf-8"))
    _check_version(raw, f"batch {batch_id!r}")
    return raw


def list_batches(root: Path | None = None) -> list[str]:
    """Every registered batch id, sorted. A glob, not an index: a batch is one
    JSON file, and the DuckDB manifest deliberately carries no batches or cells
    table, because a table would be a second source of truth."""
    d = batches_dir(root)
    if not d.is_dir():
        return []
    return sorted(p.parent.name for p in d.glob("*/batch.json"))


def batch_of_cell(cell_id: str, root: Path | None = None) -> str | None:
    """The batch a cell belongs to, from the cell's own build record, or None.

    Read from `build.json` (not from the batch's membership list) so a cell
    whose batch record was deleted out from under it still reports where it
    came from.
    """
    from .cells import build_path

    p = build_path(cell_id, root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("batch_id")
    except (OSError, json.JSONDecodeError):
        return None


def delete_batch(batch_id: str, root: Path | None = None) -> bool:
    """Remove a batch record. Returns True when something was removed.

    Deliberately does NOT touch the cells that referenced it: the batch is a
    design document, the cells are measurements, and deleting a design must not
    delete data. `batch_of_cell` keeps reporting the (now missing) id: a
    dangling reference is recorded, never fatal.
    """
    path = batch_path(batch_id, root)
    if not path.exists():
        return False
    path.unlink()
    parent = path.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    return True


# ---------------------------------------------------------------------------
# membership + serials
# ---------------------------------------------------------------------------

def add_cell_to_batch(batch_id: str, cell_id: str, root: Path | None = None, *,
                      serial: str | None = None, note: str | None = None,
                      update: bool = False) -> dict[str, Any]:
    """Add a cell to a batch, stamping `batch_id` + the serial onto its build record.

    Both directions are written, on purpose: the batch lists its members (so a
    batch view is one read) and the cell names its batch (so a cell view needs
    no reverse lookup). The two records can disagree, which is why the reverse
    index is derived from `build.json` rather than duplicated into a database.

    Serial numbers are unique within a batch: two cells off one line with the
    same serial is a transcription error, and the cost of catching it here is
    one comparison.
    """
    batch = load_batch(batch_id, root)
    from .cells import load_build, register_cell

    build = load_build(cell_id, root)
    members = list(batch.get("cells", []))
    serial = serial if serial is not None else build.get("serial")

    if serial:
        clash = [m for m in members if m.get("serial") == serial and m.get("cell_id") != cell_id]
        if clash:
            raise ValueError(
                f"serial {serial!r} is already taken in batch {batch_id} by "
                f"{clash[0]['cell_id']} — serials must be unique within a batch"
            )

    existing = next((m for m in members if m.get("cell_id") == cell_id), None)
    entry = {
        "cell_id": cell_id,
        "serial": serial,
        "built_at": (existing or {}).get("built_at") or _now(),
        "note": note if note is not None else (existing or {}).get("note"),
    }
    if existing is None:
        members.append(entry)
    else:
        if existing == entry and build.get("batch_id") == batch_id:
            return batch  # already a member with this serial: idempotent
        members[members.index(existing)] = entry

    save_batch(batch_id, batch["design"], root, name=batch.get("name"),
               note=batch.get("note"), cells=members, update=True)

    # stamp the cell's own record; `update=True` because the caller has just
    # been shown (or supplied) the value being changed
    new_build = {**build, "batch_id": batch_id}
    if serial:
        new_build["serial"] = serial
    new_build.pop("registered_at", None)
    new_build.pop("updated_at", None)
    register_cell(cell_id, new_build, root, update=True)
    return load_batch(batch_id, root)


def remove_cell_from_batch(batch_id: str, cell_id: str, root: Path | None = None) -> bool:
    """Drop a cell from a batch's membership. Returns True when it was a member.

    Clears `batch_id` on the cell too, so the two records keep agreeing. The
    cell itself is never deleted; that is `cells.delete_cell`'s job, and it
    asks first.
    """
    batch = load_batch(batch_id, root)
    members = [m for m in batch.get("cells", []) if m.get("cell_id") != cell_id]
    if len(members) == len(batch.get("cells", [])):
        return False
    save_batch(batch_id, batch["design"], root, name=batch.get("name"),
               note=batch.get("note"), cells=members, update=True)
    from .cells import load_build, register_cell

    try:
        build = load_build(cell_id, root)
    except FileNotFoundError:
        return True
    if build.get("batch_id") == batch_id:
        clean = {k: v for k, v in build.items()
                 if k not in ("batch_id", "registered_at", "updated_at")}
        register_cell(cell_id, clean, root, update=True)
    return True


def list_batch_cells(batch_id: str, root: Path | None = None) -> list[dict[str, Any]]:
    """The batch's members, each with `registered` telling whether the cell's
    build record still exists. A member whose cell was deleted is shown, not
    silently dropped."""
    from .cells import build_path

    out: list[dict[str, Any]] = []
    for m in load_batch(batch_id, root).get("cells", []):
        out.append({**m, "registered": build_path(m["cell_id"], root).exists()})
    return out
