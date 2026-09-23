"""What the CLI prints when the user is wrong, and what it prints when we are.

Two different audiences. A file that is not what it claims, a cell that was
never registered, a column the schema needs and the export does not have: each
of those already carries a message written for a person, and a 30-line
traceback above it says only that the program did not expect its own refusal.
A bug is the opposite: the traceback is the whole value of the output.

So the boundary is drawn by exception type, and the second test here is the one
that matters: a boundary that swallowed everything would be worse than none.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monocell import cli


def _raising(exc: BaseException):
    def handler(args):
        raise exc

    return handler


def _patch_find(monkeypatch, exc: BaseException) -> None:
    """Make `find` raise `exc`. The parser bound the handler when it was built,
    so the module attribute is patched before `main` builds it."""
    monkeypatch.setattr(cli, "_cmd_find", _raising(exc))


def test_a_users_error_prints_one_line(monkeypatch, capsys):
    _patch_find(monkeypatch, ValueError("no experiment named 'eis_c9' under cell c9"))
    assert cli.main(["find", "--cell", "c9"]) == 1
    out = capsys.readouterr().out
    assert out.strip() == "no experiment named 'eis_c9' under cell c9"
    assert "Traceback" not in out


def test_a_bug_still_raises(monkeypatch):
    """The boundary catches the exceptions a user causes. Anything else is a
    defect in this platform and must arrive with its stack intact."""
    _patch_find(monkeypatch, AttributeError("'NoneType' object has no attribute 'value'"))
    with pytest.raises(AttributeError):
        cli.main(["find", "--cell", "c9"])


def test_an_interrupt_is_not_a_crash(monkeypatch, capsys):
    _patch_find(monkeypatch, KeyboardInterrupt())
    assert cli.main(["find", "--cell", "c9"]) == 130
    assert "interrupted" in capsys.readouterr().out


def test_inspecting_a_directory_names_the_flag_that_does_it(tmp_path, capsys):
    """`--file` on a directory would otherwise raise `IsADirectoryError` from
    deep in the reader. The two flags differ by one word and the mistake is an
    easy one to make."""
    assert cli.main(["inspect", "--file", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "--dir" in out and "Traceback" not in out


def test_inspecting_a_file_that_is_not_there_says_so(tmp_path, capsys):
    assert cli.main(["inspect", "--file", str(tmp_path / "nope.csv")]) == 1
    assert "does not exist" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# `ingest --dir`: the sidecar beside each file
# ---------------------------------------------------------------------------

def _cycler_csv(path, *, rows: int = 40) -> None:
    """The smallest thing `probe_file` will call an rpt: one charge/discharge pair."""
    import numpy as np
    import pandas as pd

    half = rows // 2
    pd.DataFrame({
        "t_s": np.arange(float(rows)) * 60.0,
        "I_A": np.r_[np.full(half, 0.25), np.full(rows - half, -0.25)],
        "voltage_V": np.r_[np.linspace(3.0, 4.2, half), np.linspace(4.2, 3.0, rows - half)],
        "T_degC": np.full(rows, 25.0),
        "q_Ah": np.r_[np.linspace(0.0, 5.0, half), np.linspace(5.0, 0.0, rows - half)],
        "seg_type": np.r_[np.full(half, "charge"), np.full(rows - half, "discharge")],
        "cycle_index": np.ones(rows, dtype=int),
    }).to_csv(path, index=False)


def _ages(cell: str, root) -> list[float]:
    """Every stored RPT's `age_efc`, read off the meta the ingest wrote.

    Off disk rather than out of the manifest: the index carries the id, the type
    and the path, and what the sidecar decided lives in `meta.json`.
    """
    from monocell.schema import list_experiments

    return [json.loads((Path(e["path"]) / "meta.json").read_text(encoding="utf-8"))
            ["cell_state"]["age_efc"]
            for e in list_experiments(cell, "rpt", root)]


def test_ingest_dir_reads_the_sidecar_beside_each_file(tmp_path, capsys):
    """Two RPTs of different ages keep their own ages.

    In a campaign the per-file half of the sidecar (`rpt_index`, `age_efc`,
    the chamber) is exactly what differs from file to file. One sidecar applied
    to every file would not fail: it would write experiments that all claim the
    same age, and a fade curve built on them would be wrong with nothing to
    show for it.
    """
    from monocell.cells import register_cell

    data = tmp_path / "campaign"
    data.mkdir()
    for index, efc in enumerate((0.0, 500.0)):
        stem = f"rpt_{int(efc):04d}"
        _cycler_csv(data / f"{stem}.csv")
        (data / f"{stem}.sidecar.json").write_text(json.dumps(
            {"cell_state": {"rpt_index": index, "age_efc": efc}}), encoding="utf-8")

    root = tmp_path / "store"
    register_cell("dir_cell", {"capacity_Ah": 5.0, "chemistry": "NMC811/graphite"}, root)
    assert cli.main(["ingest", "--cell", "dir_cell", "--dir", str(data), "--type", "rpt",
                     "--confirm", "--data-root", str(root)]) == 0

    ages = sorted(_ages("dir_cell", root))
    assert ages == [0.0, 500.0], f"the two RPTs did not keep their own ages: {ages}"
    assert "sidecar rpt_0000.sidecar.json" in capsys.readouterr().out, (
        "the plan must name the sidecar it found, or the convention is invisible")


def test_an_explicit_sidecar_still_wins_over_the_one_beside_the_file(tmp_path):
    """`--sidecar` is how "this whole folder came off one run" is said."""
    from monocell.cells import register_cell

    data = tmp_path / "campaign"
    data.mkdir()
    _cycler_csv(data / "rpt_0000.csv")
    (data / "rpt_0000.sidecar.json").write_text(json.dumps(
        {"cell_state": {"age_efc": 0.0}}), encoding="utf-8")
    override = tmp_path / "shared.json"
    override.write_text(json.dumps({"cell_state": {"age_efc": 777.0}}), encoding="utf-8")

    root = tmp_path / "store"
    register_cell("dir_cell", {"capacity_Ah": 5.0, "chemistry": "NMC811/graphite"}, root)
    assert cli.main(["ingest", "--cell", "dir_cell", "--dir", str(data), "--type", "rpt",
                     "--confirm", "--sidecar", str(override), "--data-root", str(root)]) == 0
    ages = _ages("dir_cell", root)
    assert ages == [777.0], f"--sidecar did not win: {ages}"


def test_a_sidecar_is_found_for_a_name_that_has_a_dot_in_it(tmp_path):
    """`run.2026.csv` must look for `run.2026.sidecar.json`.

    `Path.with_suffix` replaces only the LAST suffix, so a lookup built from it
    would go looking for `run.sidecar.json` and quietly find nothing.
    """
    from monocell.cli import _sidecar_beside

    (tmp_path / "run.2026.csv").write_text("x", encoding="utf-8")
    assert _sidecar_beside(tmp_path / "run.2026.csv") is None
    (tmp_path / "run.2026.sidecar.json").write_text("{}", encoding="utf-8")
    assert _sidecar_beside(tmp_path / "run.2026.csv").name == "run.2026.sidecar.json"
