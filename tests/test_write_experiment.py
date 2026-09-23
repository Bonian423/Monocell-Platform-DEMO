"""The single writer contract: validation paths, disk layout, manifest row,
append-only guard, raw-frame preservation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from monocell.schema import load_experiment, write_experiment
from monocell.schema.manifest import find_experiment
from conftest import eis_meta, eis_series, hppc_meta, hppc_series


def test_happy_hppc_real(cell, root):
    d = write_experiment("test_cell", "hppc", hppc_meta(), hppc_series(), root=root)
    for name in ("meta.json", "series.parquet", "quality.json", "raw/series.parquet"):
        assert (d / name).exists(), name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["cell_id"] == "test_cell"
    assert meta["experiment_type"] == "hppc"
    assert meta["schema_version"] == "1.0.0"
    assert meta["experiment_id"].startswith("hppc_test_cell_")
    assert "created_at" in meta
    # hppc without reference columns: sum-check skipped, no kk
    assert meta["quality"]["sum_check_ok"] is None
    assert meta["quality"]["kk_ok"] is None
    assert any("no reference electrode columns" in f for f in meta["quality"]["flags"])
    rec = find_experiment(meta["experiment_id"], root)
    assert rec["cell_id"] == "test_cell" and rec["producer_kind"] == "real"


def test_happy_eis_real_no_t_s(cell, root):
    d = write_experiment("test_cell", "eis", eis_meta(), eis_series(), root=root)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["quality"]["kk_ok"] is True  # noiseless causal frame passes
    assert meta["quality"]["sum_check_ok"] is None
    assert "t_s" not in pd.read_parquet(d / "series.parquet").columns


def test_raw_frame_preserved_before_sum_check_mutation(cell, root):
    s = hppc_series(with_vpos=True, with_vneg=True)
    d = write_experiment("test_cell", "hppc", hppc_meta(), s, root=root)
    main = pd.read_parquet(d / "series.parquet")
    raw = pd.read_parquet(d / "raw" / "series.parquet")
    assert "sum_residual_mV" in main.columns  # sum_check mutated the stored frame
    assert "sum_residual_mV" not in raw.columns  # raw keeps the frame as received
    assert meta_quality_ok(d)


def meta_quality_ok(d) -> bool:
    q = json.loads((d / "quality.json").read_text(encoding="utf-8"))
    return q["sum_check_ok"] is True  # V_full == V_pos - V_neg by construction


def test_explicit_experiment_id_honored(cell, root):
    meta = hppc_meta()
    meta["experiment_id"] = "hppc_test_cell_custom"
    d = write_experiment("test_cell", "hppc", meta, hppc_series(), root=root)
    assert d.name == "hppc_test_cell_custom"


def test_duplicate_experiment_id_raises(cell, root):
    meta = hppc_meta()
    meta["experiment_id"] = "hppc_test_cell_dup"
    write_experiment("test_cell", "hppc", dict(meta), hppc_series(), root=root)
    with pytest.raises(ValueError, match="append-only"):
        write_experiment("test_cell", "hppc", dict(meta), hppc_series(), root=root)


def test_unknown_exp_type_raises(cell, root):
    with pytest.raises(ValueError, match="unknown experiment_type"):
        write_experiment("test_cell", "cyclic_voltammetry", hppc_meta(), hppc_series(), root=root)


def test_unregistered_cell_raises(root):
    with pytest.raises(FileNotFoundError):
        write_experiment("ghost", "hppc", hppc_meta(), hppc_series(), root=root)


@pytest.mark.parametrize("key", ["producer", "protocol", "cell_state"])
def test_missing_meta_block_raises(cell, root, key):
    meta = hppc_meta()
    del meta[key]
    with pytest.raises(ValueError, match=f"meta.{key} is required"):
        write_experiment("test_cell", "hppc", meta, hppc_series(), root=root)


def test_producer_not_dict_raises(cell, root):
    meta = hppc_meta()
    meta["producer"] = "synthetic"
    with pytest.raises(ValueError, match="meta.producer must be a dict"):
        write_experiment("test_cell", "hppc", meta, hppc_series(), root=root)


def test_producer_bad_kind_raises(cell, root):
    meta = hppc_meta()
    meta["producer"]["kind"] = "simulated"
    with pytest.raises(ValueError, match="producer.kind"):
        write_experiment("test_cell", "hppc", meta, hppc_series(), root=root)


def test_synthetic_requires_sidecar_ground_truth(cell, root):
    meta = hppc_meta()
    meta["producer"] = {"kind": "synthetic", "rng_seed": 1}
    with pytest.raises(ValueError, match="ground_truth"):
        write_experiment("test_cell", "hppc", meta, hppc_series(), root=root)


def test_synthetic_requires_rng_seed(cell, root):
    meta = hppc_meta()
    meta["producer"] = {"kind": "synthetic"}
    with pytest.raises(ValueError, match="rng_seed"):
        write_experiment("test_cell", "hppc", meta, hppc_series(), sidecar={"ground_truth": {}}, root=root)


def test_series_not_dataframe_raises(cell, root):
    with pytest.raises(TypeError, match="DataFrame"):
        write_experiment("test_cell", "hppc", hppc_meta(), [1, 2, 3], root=root)


def test_missing_required_column_raises(cell, root):
    s = hppc_series().drop(columns=["V_full_V"])
    with pytest.raises(ValueError, match="missing required series columns.*V_full_V"):
        write_experiment("test_cell", "hppc", hppc_meta(), s, root=root)


def test_unknown_column_raises(cell, root):
    s = hppc_series()
    s["mystery_col"] = 1.0
    with pytest.raises(ValueError, match="unknown series columns.*mystery_col"):
        write_experiment("test_cell", "hppc", hppc_meta(), s, root=root)


def test_empty_series_raises(cell, root):
    s = hppc_series().iloc[:0]
    with pytest.raises(ValueError, match="non-empty"):
        write_experiment("test_cell", "hppc", hppc_meta(), s, root=root)


def test_nan_t_s_raises(cell, root):
    s = hppc_series()
    s.loc[3, "t_s"] = np.nan
    with pytest.raises(ValueError, match="t_s"):
        write_experiment("test_cell", "hppc", hppc_meta(), s, root=root)


def test_missing_instrument_fields_raise(cell, root):
    meta = hppc_meta()
    del meta["instrument"]
    with pytest.raises(ValueError, match="missing instrument fields for hppc: \\['sampling_rate_Hz'\\]"):
        write_experiment("test_cell", "hppc", meta, hppc_series(), root=root)
    meta = eis_meta()
    del meta["instrument"]["amplitude_mV"]
    with pytest.raises(ValueError, match="missing instrument fields for eis"):
        write_experiment("test_cell", "eis", meta, eis_series(), root=root)


def test_instrument_not_dict_raises(cell, root):
    meta = hppc_meta()
    meta["instrument"] = "potentiostat-01"
    with pytest.raises(ValueError, match="meta.instrument must be a dict"):
        write_experiment("test_cell", "hppc", meta, hppc_series(), root=root)


def test_rule_flags_recorded_not_raised(cell, root):
    meta = hppc_meta(sampling_rate_hz=5.0)  # violates sampling_ge_10Hz
    d = write_experiment("test_cell", "hppc", meta, hppc_series(), root=root)
    q = json.loads((d / "quality.json").read_text(encoding="utf-8"))
    assert any(f.startswith("rule: sampling_rate_Hz") for f in q["flags"])


def test_load_experiment_roundtrip(cell, root):
    s = hppc_series()
    d = write_experiment("test_cell", "hppc", hppc_meta(), s, root=root)
    eid = json.loads((d / "meta.json").read_text(encoding="utf-8"))["experiment_id"]
    exp = load_experiment(eid, cell_id="test_cell", root=root)
    pd.testing.assert_frame_equal(exp["series"], pd.read_parquet(d / "series.parquet"))
    assert exp["sidecar"] is None
    assert exp["quality"]["kk_ok"] is None
    # manifest-only lookup (no cell_id)
    exp2 = load_experiment(eid, root=root)
    assert exp2["meta"]["cell_id"] == "test_cell"


def test_load_experiment_unknown_id_raises(root):
    with pytest.raises(FileNotFoundError, match="not in manifest"):
        load_experiment("nope", root=root)
