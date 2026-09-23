"""Artifact provenance envelope: write/load round-trip, ids, manifest row, and
the store's one "newest" rule."""

from __future__ import annotations

import json

from monocell import __version__
from monocell.schema.artifacts import (SCHEMA_VERSION, load_artifact, newest_artifact,
                                       write_artifact)
from monocell.schema.manifest import list_artifacts


def test_envelope_fields(root):
    out = root / "artifacts" / "c1" / "parameters"
    p = write_artifact(
        "parameters",
        "parameter_file_c1_v1.json",
        artifact_data={"pybamm_overrides": {"Electrode height [m]": 0.065}},
        inputs=[{"id": "e1", "hash": "abc"}],
        params={"source": "test"},
        cell_id="c1",
        out_dir=out,
        root=root,
    )
    env = load_artifact(p)
    assert env["artifact_id"] == "parameters/parameter_file_c1_v1"
    assert env["module"] == "parameters"
    assert env["cell_id"] == "c1"
    assert env["schema_version"] == SCHEMA_VERSION
    assert env["producer_version"] == __version__
    assert env["inputs"] == [{"id": "e1", "hash": "abc"}]
    assert env["data"]["pybamm_overrides"]["Electrode height [m]"] == 0.065
    rows = list_artifacts("c1", "parameters", root)
    assert len(rows) == 1 and rows[0]["artifact_id"] == "parameters/parameter_file_c1_v1"


def test_rewrite_same_filename_updates(root):
    # re-derivation MUST overwrite the same filename; the manifest keeps one row
    out = root / "artifacts" / "c1" / "parameters"
    write_artifact("parameters", "x.json", {"v": 1}, [], {}, "c1", out, root=root)
    p = write_artifact("parameters", "x.json", {"v": 2}, [], {}, "c1", out, root=root)
    assert json.loads(p.read_text(encoding="utf-8"))["data"]["v"] == 2
    assert len(list_artifacts("c1", "parameters", root)) == 1


def test_newest_is_decided_by_created_at_and_a_corrupt_file_never_wins(root):
    """The filename is not the order: a content-addressed name sorts randomly.

    `a_...` is written LAST, so a picker that sorted by name would return the
    older `z_...`. A truncated file is present too and must lose to any
    readable one rather than take the whole pick down.
    """
    out = root / "artifacts" / "c1" / "parameters"
    write_artifact("parameters", "parameter_file_c1_vz.json", {"v": "old"}, [], {}, "c1", out,
                   root=root)
    write_artifact("parameters", "parameter_file_c1_va.json", {"v": "new"}, [], {}, "c1", out,
                   root=root)
    (out / "parameter_file_c1_vzz.json").write_text('{"artifact_id": "trunc', encoding="utf-8")

    newest = newest_artifact("c1", "parameters", "parameter_file_*.json", root)
    assert newest is not None and newest.name == "parameter_file_c1_va.json"
