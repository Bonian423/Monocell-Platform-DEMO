"""The batch record and its membership.

The claims worth testing are structural rather than numeric:

* the build-record contract did NOT move: `capacity_Ah` + `chemistry` are
  still the only required fields, so an existing store's `build.json` files
  stay valid (`test_cells.py` guards the other half of that);
* membership is a fact about TWO records (the batch lists its members, the
  cell names its batch), and every operation keeps the two agreeing.
"""

from __future__ import annotations

import json

import pytest

from conftest import BUILD
from monocell.batches import (
    DESIGN_FIELDS,
    add_cell_to_batch,
    batch_of_cell,
    delete_batch,
    list_batch_cells,
    list_batches,
    load_batch,
    remove_cell_from_batch,
    save_batch,
)

DESIGN = {
    "capacity_Ah": 5.0,
    "chemistry": "NMC811/graphite",
    "layer_count": 12,
    "height": 0.12,
    "width": 0.08,
    "thickness_total": 0.005,
    "n_series": 1,
    "lower_cutoff_V": 2.5,
    "upper_cutoff_V": 4.2,
    "ambient_temperature_K": 298.15,
    "tab_width": 0.02,
    "neg_tab_y_centre": 0.02,
    "pos_tab_y_centre": 0.06,
    "coating_thicknesses_um": {"L_cn": 8.0, "L_n": 80.0, "L_s": 12.0,
                               "L_p": 70.0, "L_cp": 12.0},
}


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def test_batch_round_trips(root):
    save_batch("lot_A", DESIGN, root, name="Lot A", note="first build")
    b = load_batch("lot_A", root)
    assert b["batch_id"] == "lot_A"
    assert b["name"] == "Lot A"
    assert b["note"] == "first build"
    assert b["design"] == DESIGN
    assert b["cells"] == []
    assert b["schema_version"]
    assert b["created_at"] and b["updated_at"]


def test_resave_is_idempotent_and_a_change_needs_update(root):
    save_batch("lot_A", DESIGN, root)
    before = load_batch("lot_A", root)
    save_batch("lot_A", DESIGN, root)  # identical -> no-op
    assert load_batch("lot_A", root)["updated_at"] == before["updated_at"]

    changed = {**DESIGN, "capacity_Ah": 6.0}
    with pytest.raises(ValueError, match="already exists with a different record"):
        save_batch("lot_A", changed, root)
    save_batch("lot_A", changed, root, update=True)
    assert load_batch("lot_A", root)["design"]["capacity_Ah"] == 6.0
    assert load_batch("lot_A", root)["created_at"] == before["created_at"]


def test_unknown_design_field_refused(root):
    with pytest.raises(ValueError, match="unknown design fields"):
        save_batch("lot_A", {**DESIGN, "coating_thickness_nm": 5}, root)


def test_missing_required_design_field_refused(root):
    with pytest.raises(ValueError, match="missing required design fields"):
        save_batch("lot_A", {"capacity_Ah": 5.0}, root)


def test_required_design_fields_are_the_two_the_platform_needs():
    """The same contract `cells.BUILD_FIELDS` states, restated for designs."""
    required = {f for f, (_, req) in DESIGN_FIELDS.items() if req}
    assert required == {"capacity_Ah", "chemistry"}


def test_a_batch_id_becomes_a_directory_name_so_it_is_checked(root):
    for bad in ("", "../escape", "has space", "-leading-dash"):
        with pytest.raises(ValueError, match="invalid batch id"):
            save_batch(bad, DESIGN, root)


def test_a_newer_schema_version_is_refused(root):
    save_batch("lot_A", DESIGN, root)
    p = root / "batches" / "lot_A" / "batch.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["schema_version"] = "99.0.0"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="newer than this platform understands"):
        load_batch("lot_A", root)


def test_write_is_atomic_and_leaves_no_temp_file(root):
    save_batch("lot_A", DESIGN, root)
    assert list((root / "batches" / "lot_A").glob("*.tmp")) == []


def test_a_store_with_no_batches_dir_still_reads(root):
    """A glob over a missing directory is an empty list, never an error."""
    from monocell.cells import register_cell

    register_cell("old_cell", BUILD, root)
    assert list_batches(root) == []
    with pytest.raises(FileNotFoundError):
        load_batch("lot_A", root)
    assert batch_of_cell("old_cell", root) is None


# ---------------------------------------------------------------------------
# membership + serials
# ---------------------------------------------------------------------------

def test_membership_writes_both_records(root):
    save_batch("lot_A", DESIGN, root)
    from monocell.cells import load_build, register_cell

    register_cell("cell_01", BUILD, root)
    add_cell_to_batch("lot_A", "cell_01", root, serial="S-001")

    members = list_batch_cells("lot_A", root)
    assert [m["cell_id"] for m in members] == ["cell_01"]
    assert members[0]["serial"] == "S-001"
    assert members[0]["registered"] is True

    build = load_build("cell_01", root)
    assert build["batch_id"] == "lot_A"
    assert build["serial"] == "S-001"
    assert build["capacity_Ah"] == BUILD["capacity_Ah"]  # the rest of the record is untouched
    assert batch_of_cell("cell_01", root) == "lot_A"


def test_adding_the_same_cell_twice_is_idempotent(root):
    save_batch("lot_A", DESIGN, root)
    from monocell.cells import register_cell

    register_cell("cell_01", BUILD, root)
    add_cell_to_batch("lot_A", "cell_01", root, serial="S-001")
    add_cell_to_batch("lot_A", "cell_01", root, serial="S-001")
    assert len(list_batch_cells("lot_A", root)) == 1


def test_serial_uniqueness_within_a_batch(root):
    save_batch("lot_A", DESIGN, root)
    from monocell.cells import register_cell

    register_cell("cell_01", BUILD, root)
    register_cell("cell_02", BUILD, root)
    add_cell_to_batch("lot_A", "cell_01", root, serial="S-001")
    with pytest.raises(ValueError, match="serials must be unique"):
        add_cell_to_batch("lot_A", "cell_02", root, serial="S-001")
    add_cell_to_batch("lot_A", "cell_02", root, serial="S-002")  # a different one is fine
    assert [m["serial"] for m in list_batch_cells("lot_A", root)] == ["S-001", "S-002"]


def test_removing_a_member_clears_the_cell_side(root):
    save_batch("lot_A", DESIGN, root)
    from monocell.cells import load_build, register_cell

    register_cell("cell_01", BUILD, root)
    add_cell_to_batch("lot_A", "cell_01", root, serial="S-001")
    assert remove_cell_from_batch("lot_A", "cell_01", root) is True
    assert list_batch_cells("lot_A", root) == []
    assert "batch_id" not in load_build("cell_01", root)
    assert remove_cell_from_batch("lot_A", "cell_01", root) is False  # already gone


def test_a_member_whose_cell_vanished_is_still_listed(root):
    save_batch("lot_A", DESIGN, root)
    from monocell.cells import register_cell

    register_cell("cell_01", BUILD, root)
    add_cell_to_batch("lot_A", "cell_01", root, serial="S-001")
    (root / "cells" / "cell_01" / "build.json").unlink()
    members = list_batch_cells("lot_A", root)
    assert members[0]["registered"] is False


def test_deleting_a_batch_leaves_its_cells_alone(root):
    save_batch("lot_A", DESIGN, root)
    from monocell.cells import load_build, register_cell

    register_cell("cell_01", BUILD, root)
    add_cell_to_batch("lot_A", "cell_01", root, serial="S-001")
    assert delete_batch("lot_A", root) is True
    assert list_batches(root) == []
    assert load_build("cell_01", root)["capacity_Ah"] == BUILD["capacity_Ah"]
    assert batch_of_cell("cell_01", root) == "lot_A"  # dangling, and honest about it
    assert delete_batch("lot_A", root) is False


def test_membership_is_not_design():
    """`serial` and `batch_id` are the only `BUILD_FIELDS` the design spec does
    not carry: where a cell sits in a lot is not what the lot IS, and a design
    that could name a serial would be a design that could disagree with the
    batch's own member list."""
    from monocell.cells import BUILD_FIELDS

    assert set(BUILD_FIELDS) - set(DESIGN_FIELDS) == {"serial", "batch_id"}
