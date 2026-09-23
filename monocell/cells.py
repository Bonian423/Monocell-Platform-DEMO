"""Cell registry: one `build.json` per cell.

The build record is what every consumer reads for a cell's capacity, chemistry
and geometry. Only `capacity_Ah` and `chemistry` are required; the rest of the
field list is the contract for consumers that need geometry or stack details.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# {field: (description, required)}
BUILD_FIELDS: dict[str, tuple[str, bool]] = {
    # required
    "capacity_Ah": ("reference capacity (Ah), the C-rate anchor", True),
    "chemistry": ("e.g. 'NMC811/graphite'", True),
    # optional
    "electrode_source": ("where the electrodes came from", False),
    "areal_loading_mAh_cm2": ("electrode areal loading (mAh/cm²)", False),
    "coating_thicknesses_um": ("per-layer coating thicknesses", False),
    "porosities": ("per-layer porosities", False),
    "np_ratio": ("negative-to-positive capacity ratio", False),
    "layer_count": ("number of electrode layers in the stack", False),
    "electrolyte_volume_per_Ah_ml": ("electrolyte volume per Ah", False),
    "separator_type": ("separator material/type", False),
    "tabs_format": ("tab geometry and format", False),
    "formation_protocol": ("formation protocol description", False),
    "stack_pressure_kPa": ("applied stack pressure", False),
    "pat_rinse_protocol": ("rinse protocol for reference-electrode test cells", False),
    "rig": ("rig data: fixture stiffness (measured, not from the drawing), load-cell calibration date/history", False),
    # --- membership (optional, so a build.json without it stays valid) --------
    "batch_id": ("the batch (design lot) this cell was built to", False),
    "serial": ("the cell's serial number, unique within its batch", False),
}


def data_root() -> Path:
    """Platform data store, overridable via MONOCELL_DATA (tests use transient dirs)."""
    import os

    return Path(os.environ.get("MONOCELL_DATA", "data"))


def cells_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "cells"


def build_path(cell_id: str, root: Path | None = None) -> Path:
    return cells_dir(root) / cell_id / "build.json"


def register_cell(cell_id: str, build: dict[str, Any], root: Path | None = None, update: bool = False) -> Path:
    """Write a cell's build record. Returns the path written.

    Re-registering the same build is idempotent (a script may re-run into the
    same root); a DIFFERENT build is an identity change and raises unless the
    caller passes `update=True` — the deliberate-edit path (a corrected
    loading, an updated rig calibration) where the caller has seen the current
    record and means to replace it.
    """
    unknown = set(build) - set(BUILD_FIELDS)
    if unknown:
        raise ValueError(f"unknown build fields for {cell_id}: {sorted(unknown)}")
    missing = [f for f, (_, req) in BUILD_FIELDS.items() if req and f not in build]
    if missing:
        raise ValueError(f"missing required build fields for {cell_id}: {missing}")
    if build["capacity_Ah"] <= 0:
        raise ValueError("capacity_Ah must be positive")

    record = dict(build)
    path = build_path(cell_id, root)
    now = datetime.now(timezone.utc).isoformat()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        registered_at = existing.pop("registered_at", now)
        existing.pop("updated_at", None)
        different = {k: existing[k] for k in sorted(existing)} != {k: build[k] for k in sorted(build)}
        if different and not update:
            raise ValueError(f"cell {cell_id} is already registered with a different build")
        record["registered_at"] = registered_at
        if different:
            record["updated_at"] = now
    record.setdefault("registered_at", now)
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic: a half-written build record is a cell whose geometry every
    # consumer would read as defaults
    from ._fs import atomic_write_text

    atomic_write_text(path, json.dumps(record, indent=2))
    return path


def load_build(cell_id: str, root: Path | None = None) -> dict[str, Any]:
    """Read a cell's build record. Raises FileNotFoundError if the cell is not registered."""
    return json.loads(build_path(cell_id, root).read_text(encoding="utf-8"))


def list_cells(root: Path | None = None) -> list[str]:
    """Every registered cell id, sorted.

    A glob, not an index, the same as `batches.list_batches`: a cell is one JSON
    file, and a `cells` table in the DuckDB manifest would be a second source of
    truth for a file that already has one.
    """
    d = cells_dir(root)
    if not d.is_dir():
        return []
    return sorted(p.parent.name for p in d.glob("*/build.json"))


def delete_cell(cell_id: str, root: Path | None = None, *, purge: bool = False) -> bool:
    """Remove a cell. Returns True when something was removed.

    Without `purge` only the build record goes, and the call REFUSES if the
    cell still has experiments or artifacts. Those are measurements and derived
    evidence, and the store's contract is append-only: un-registering a cell
    must never silently destroy the data someone else's conclusion rests on.
    The caller can say `purge=True` and mean it.
    """
    import shutil

    path = build_path(cell_id, root)
    if not path.exists():
        return False
    base = root or data_root()
    owned = [base / "experiments" / cell_id, base / "artifacts" / cell_id]
    if not purge:
        holding = [d for d in owned if d.is_dir() and any(d.iterdir())]
        if holding:
            raise ValueError(
                f"cell {cell_id} still owns data ({', '.join(d.name for d in holding)}); "
                "un-registering it would leave measurements with no build record. "
                "Pass purge=True to delete the cell's experiments and artifacts too."
            )
        shutil.rmtree(cells_dir(root) / cell_id, ignore_errors=True)
        return True
    for d in owned:
        shutil.rmtree(d, ignore_errors=True)
    # ...and the build record itself, so a purged cell is not left registered
    # with nothing behind it.
    shutil.rmtree(cells_dir(root) / cell_id, ignore_errors=True)
    return True
