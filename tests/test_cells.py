"""Cell registry: register_cell / load_build validation and idempotency."""

from __future__ import annotations

import pytest

from monocell.cells import (BUILD_FIELDS, build_path, delete_cell, list_cells,
                            load_build, register_cell)


def test_register_load_roundtrip(root):
    p = register_cell("c1", {"capacity_Ah": 5.0, "chemistry": "NMC"}, root)
    assert p == build_path("c1", root)
    rec = load_build("c1", root)
    assert rec["capacity_Ah"] == 5.0
    assert rec["chemistry"] == "NMC"
    assert "registered_at" in rec


def test_optional_fields_stored(root):
    register_cell(
        "c1",
        {"capacity_Ah": 5.0, "chemistry": "NMC", "rig": "fixture A", "np_ratio": 1.1},
        root,
    )
    rec = load_build("c1", root)
    assert rec["rig"] == "fixture A"
    assert rec["np_ratio"] == 1.1


def test_unknown_field_raises(root):
    with pytest.raises(ValueError, match="unknown build fields"):
        register_cell("c1", {"capacity_Ah": 5.0, "chemistry": "NMC", "color": "blue"}, root)


def test_missing_required_field_raises(root):
    with pytest.raises(ValueError, match="missing required build fields"):
        register_cell("c1", {"capacity_Ah": 5.0}, root)


def test_nonpositive_capacity_raises(root):
    with pytest.raises(ValueError, match="capacity_Ah must be positive"):
        register_cell("c1", {"capacity_Ah": 0.0, "chemistry": "NMC"}, root)
    with pytest.raises(ValueError, match="capacity_Ah must be positive"):
        register_cell("c1", {"capacity_Ah": -1.0, "chemistry": "NMC"}, root)


def test_load_unknown_cell_raises(root):
    with pytest.raises(FileNotFoundError):
        load_build("ghost", root)


def test_duplicate_same_build_is_idempotent(root):
    build = {"capacity_Ah": 5.0, "chemistry": "NMC"}
    register_cell("c1", build, root)
    register_cell("c1", build, root)  # a script may re-run into the same root
    assert load_build("c1", root)["capacity_Ah"] == 5.0


def test_duplicate_different_build_raises(root):
    register_cell("c1", {"capacity_Ah": 5.0, "chemistry": "NMC"}, root)
    with pytest.raises(ValueError, match="different build"):
        register_cell("c1", {"capacity_Ah": 6.0, "chemistry": "NMC"}, root)


def test_build_fields_contract():
    # the build-record contract: exactly two required fields
    required = {f for f, (_, req) in BUILD_FIELDS.items() if req}
    assert required == {"capacity_Ah", "chemistry"}


def test_purging_a_cell_also_un_registers_it(root):
    """`purge=True` is the "and I mean it" branch of a destructive call, so it
    has to leave the store in a state a reader can describe: no data and no
    build record. A registered cell with no data would be worse than the
    refusal on the branch without `purge`."""
    from monocell.schema.artifacts import write_artifact

    register_cell("doomed", {"capacity_Ah": 5.0, "chemistry": "NMC"}, root)
    write_artifact("parameters", "parameter_file_doomed_v1.json", {"cell_id": "doomed"}, [], {},
                   "doomed", root / "artifacts" / "doomed" / "parameters", root)

    assert delete_cell("doomed", root, purge=True) is True
    assert not (root / "artifacts" / "doomed").exists()
    with pytest.raises(FileNotFoundError):
        load_build("doomed", root)
    assert "doomed" not in list_cells(root)


def test_unregistering_a_cell_that_owns_data_is_refused(root):
    from monocell.schema.artifacts import write_artifact

    register_cell("kept", {"capacity_Ah": 5.0, "chemistry": "NMC"}, root)
    write_artifact("parameters", "parameter_file_kept_v1.json", {"cell_id": "kept"}, [], {},
                   "kept", root / "artifacts" / "kept" / "parameters", root)
    with pytest.raises(ValueError, match="still owns data"):
        delete_cell("kept", root)
    assert load_build("kept", root)["capacity_Ah"] == 5.0
