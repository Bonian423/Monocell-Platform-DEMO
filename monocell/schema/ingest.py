"""Real-file ingestors: parse instrument exports into the schema.

Each ingestor is a thin format adapter: parse the file into the spec's
columns, assemble the meta from a sidecar JSON (protocol/instrument/
cell_state — real rigs pair data files with metadata), and hand everything
to the one `write_experiment`. Everything downstream of ingest sees one shape,
whatever wrote the file.

- `read_cycler_csv`: CSV cycler exports (Biologic/Arbin-style) for hppc /
  rpt / three_electrode_ts — tolerant column-name mapper.
- `read_eis_xlsx`: Palmsense4-style XLSX, one sheet per spectrum, Z'/−Z''
  columns (the −Z'' convention — Z_im = −(−Z'')).
- `read_palmsense4_csv`: raw Palmsense4 CSV (UTF-16-LE): measurement blocks
  per spectrum plus, when a GITT ran on the same file, a mixed-mode section
  whose steps are correlated to the EIS spectra (each EIS sits at the end
  voltage of the step before a negative-current GITT step). The correlated
  voltage becomes the spectrum's V_dc.
- `read_pressure_csv`: stack-pressure rig CSV.

Instrument fields are derived from the data file where the sidecar does not
provide them (`_eis_from_file`): f_min/f_max and points_per_decade from the
measured grid, potentiostat/amplitude best-effort from the file's producer
markers, eis_mode and the mixed-mode context into the protocol.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import io
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from .write_experiment import write_experiment


def _load_sidecar(sidecar_path: Path | None) -> dict:
    if sidecar_path is None:
        return {}
    return json.loads(Path(sidecar_path).read_text(encoding="utf-8"))


def _real_meta(source_file: Path, sidecar: dict, capacity_ref_Ah: float | None = None) -> dict:
    """Producer/protocol/instrument/cell_state from the sidecar (real data)."""
    meta = {
        "producer": {
            "kind": "real",
            "software": sidecar.get("software") or "monocell-ingest",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source_file),
        },
        "timing": {"ntp_synced": bool(sidecar.get("ntp_synced", False)), "t_epoch_s": 0.0, "clock_offset_s": {}},
        "protocol": sidecar.get("protocol") or {"description": "real experiment (sidecar protocol missing)"},
        "instrument": sidecar.get("instrument") or {},
        "cell_state": sidecar.get("cell_state") or {},
    }
    cs = meta["cell_state"]
    # The capacity this run's SOC and SOH are referenced to. There is no literal
    # default: a default sized for one cell makes every other cell's SOH and
    # C-rates wrong by orders of magnitude, all of it arithmetically consistent
    # and none of it about the cell.
    #
    # The platform HAS the right number: the cell's own build record. So the
    # caller passes it and the sidecar still wins where it names one, which is
    # the case where somebody measured the reference rather than reading it off
    # a datasheet. Only `preview_ingest` without a cell has no reference at all.
    if "capacity_ref_Ah" in cs:
        # The sidecar named one, so somebody decided it. `measured` is the word
        # for a reference taken from this cell's own measured capacity, and a
        # sidecar cannot distinguish the two — so the honest label for a value
        # that arrived from outside is where it arrived from.
        cs.setdefault("capacity_ref_source", "sidecar")
    elif capacity_ref_Ah is not None:
        cs["capacity_ref_Ah"] = float(capacity_ref_Ah)
        # NAMEPLATE, and labelled as such. `capacity_ref_not_nameplate_when_aged`
        # reads this: a reference that is the nameplate is correct on a fresh
        # cell and is a capacity the cell no longer has on an aged one, and the
        # number looks identical either way.
        cs["capacity_ref_source"] = "build_record"
    # `rpt_rate` is deliberately NOT defaulted. A default equal to the value
    # the C/20 rule passes at would make every file whose sidecar omits the rate
    # look compliant, and the rule could never fire. A missing rate is a fact
    # about the file, and the rule flags it as unrecorded.
    #
    # Where the run sits in the cell's life: which check-up, and how aged
    # (`capacity_ref_not_nameplate_when_aged` reads `age_efc`). A sidecar that
    # omits them describes the first test of a fresh cell.
    cs.setdefault("rpt_index", 0)
    cs.setdefault("age_efc", 0.0)
    return meta


def _cell_capacity(cell_id: str | None, root) -> float | None:
    """The cell's nameplate capacity, or None if there is no readable record.

    None rather than a number, deliberately: an ingest into a cell the platform
    does not know about must not carry a capacity reference invented here. The
    quality record then reports `capacity_SOH` as absent and says why, which is
    the true state of affairs.
    """
    if not cell_id:
        return None
    from ..cells import load_build

    try:
        return float(load_build(cell_id, root)["capacity_Ah"])
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


INGESTABLE = ("hppc", "rpt", "cycling", "three_electrode_ts", "eis", "pressure", "gitt", "misc")


def read_misc(file_path: Path) -> pd.DataFrame:
    """A real file of no declared shape, taken exactly as it is.

    The one reader that maps nothing. Every other ingestor here is a format
    adapter: it renames an instrument's headers onto the spec's columns and
    refuses what it cannot place. This one has no columns to map ONTO, because
    `misc` declares none — so it reads the table and stops. Whatever it finds is
    written, and `write_experiment` records the observed column/dtype map into
    `meta.observed_columns`, which is the only schema such a file will have.

    Deliberately NOT tolerant of a missing header row and NOT a directory
    scanner: a misc ingest is one table from one file. Guessing at structure is
    what the named types are for; anything clever here would make a file that
    looks understood when it is only accepted.
    """
    if file_path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    if file_path.suffix.lower() not in (".csv", ".txt"):
        raise ValueError(f"misc reads CSV, TXT or XLSX; got {file_path.suffix!r}")

    # Real exports arrive in whichever encoding the instrument chose, and the
    # same two this repo's other readers meet: UTF-16-LE (Palmsense4) and UTF-8.
    #
    # Detected by signature and NOT by "try to decode, keep the first that
    # works", which is wrong in a way that fails silently: the UTF-16-LE bytes
    # of an ASCII CSV are ALSO valid UTF-8, because
    # every character is followed by a NUL. A strict UTF-8 decoder therefore
    # succeeds on a UTF-16 file and hands pandas a frame of NUL-separated
    # fragments with `Unnamed: 2` columns and no error anywhere. A BOM, or a NUL
    # in the first block, is the actual evidence.
    raw = file_path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif b"\x00" in raw[:4096]:
        text = raw.decode("utf-16-le")  # no BOM, but NULs: UTF-16-LE in the wild
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            # latin-1 rather than a lossy decode: it maps every byte, so a file
            # in some third codepage keeps its bytes and can be re-read once
            # someone says what it really was.
            text = raw.decode("latin-1")
    return pd.read_csv(io.StringIO(text))


def read_with_instrument(exp_type: str, file_path: Path, profile) -> tuple[pd.DataFrame, dict]:
    """Parse through an `InstrumentProfile`, returning the frame AND its report.

    The second half is the point and is why this is not folded into `read_any`:
    a profile-driven read knows which columns it could not place, which it
    expected and did not find, and which arrived empty — and a caller that
    cannot see those is back to the silent-drop behaviour the profiles exist to
    end. `read_any` keeps its one-return shape for the built-in readers.

    The report gains the two facts that are about the TYPE rather than about the
    file — which required schema columns the profile did not produce, and which
    produced columns the type does not declare. Both are computed here, so the
    preview and the write see the same answer; the *refusal* stays in
    `write_experiment`, which is the one place that decides what may be stored.
    """
    from .instruments import read_with_profile
    from .tables import SPECS

    frame, report = read_with_profile(file_path, profile)
    if "cycle_index" in frame.columns:
        # the schema wants int32; a cycler writes it as a float like everything
        # else in the table
        frame["cycle_index"] = frame["cycle_index"].fillna(0).astype("int32")
    report["exp_type"] = exp_type
    spec = SPECS[exp_type]
    declared = {col.name for col in spec.columns}
    report["required_missing"] = [col.name for col in spec.columns
                                  if col.required and col.name not in frame.columns]
    report["not_in_schema"] = ([] if spec.permissive
                               else sorted(set(frame.columns) - declared))
    return frame, report


def read_for_ingest(exp_type: str, file_path: Path, instrument: str | None,
                    root: Path | None) -> tuple[pd.DataFrame, dict | None]:
    """`(frame, report | None)` — one parse, whichever route it took.

    The single place the two readers meet, so the preview and the write cannot
    take different ones. `instrument=None` is the built-in reader and has no
    report; a named profile is resolved through the STORE first
    (`load_profiles`), because an engineer who corrected a profile for their own
    rig must get their version and not the one this package ships.

    An unknown name raises rather than falling back to the built-in reader. The
    fallback would be the more forgiving behaviour and the wrong one: a typo in
    `--instrument landtt` would then parse by synonym guess and look like it had
    used the profile.
    """
    if instrument is None:
        return read_any(exp_type, file_path), None
    from .instruments import load_profiles

    profiles = load_profiles(root)
    if instrument not in profiles:
        raise ValueError(f"no instrument profile named {instrument!r}; "
                         f"available: {sorted(profiles)}")
    return read_with_instrument(exp_type, file_path, profiles[instrument])


def read_any(exp_type: str, file_path: Path) -> pd.DataFrame:
    """Parse a file of the given type into the schema columns (no writing)."""
    if exp_type in ("hppc", "rpt", "cycling", "three_electrode_ts"):
        return read_cycler_csv(file_path, exp_type=exp_type)
    if exp_type == "eis":
        if file_path.suffix.lower() == ".csv":
            return read_palmsense4_csv(file_path)
        return read_eis_xlsx(file_path)
    if exp_type == "pressure":
        return read_pressure_csv(file_path)
    if exp_type == "gitt":
        return read_gitt(file_path)
    if exp_type == "misc":
        return read_misc(file_path)
    raise ValueError(f"no ingestor for experiment type {exp_type!r}; ingestable: {list(INGESTABLE)}")


def ingest(cell_id: str, exp_type: str, file_path: Path, sidecar_path: Path | None,
           root: Path | None, instrument: str | None = None) -> Path:
    """Dispatch by experiment type. Returns the experiment dir."""
    return ingest_data(cell_id, exp_type, file_path, _load_sidecar(sidecar_path), root,
                       instrument=instrument)


def _hppc_protocol_from_series(meta: dict, series: pd.DataFrame) -> None:
    """A real HPPC's pulse current and duration live in the CSV, not the sidecar;
    a pulse analysis needs `protocol.pulse_current_A` and
    `protocol.pulse_duration_s`, so fill them from the pulse segments when the
    sidecar didn't provide them (a real ingest must be derivable)."""
    proto = meta.get("protocol") or {}
    if "I_A" not in series or "t_s" not in series:
        return
    seg = series.get("seg_type")
    pulse = series if seg is None else series.loc[np.asarray(seg, dtype=str) == "pulse"]
    if not len(pulse):
        return
    if proto.get("pulse_current_A") is None:
        proto["pulse_current_A"] = float(pulse["I_A"].abs().mean())
    if proto.get("pulse_duration_s") is None:
        t = pulse["t_s"].to_numpy(dtype=float)
        dt = float(np.median(np.diff(t)))
        runs = np.split(t, np.flatnonzero(np.diff(t) > 1.5 * dt) + 1)
        # a run's window is its span plus one sample interval (a pulse sampled
        # as arange(0, pulse_s, dt) has its last sample one dt before the end)
        durations = [float(r[-1] - r[0] + dt) for r in runs if len(r) > 1]
        proto["pulse_duration_s"] = float(np.mean(durations)) if durations else float(dt)
    meta["protocol"] = proto


def ingest_data(cell_id: str, exp_type: str, file_path: Path, sidecar: dict | None,
                root: Path | None, instrument: str | None = None) -> Path:
    """Ingest with an in-memory sidecar dict (one built by the caller)."""
    series, report = read_for_ingest(exp_type, file_path, instrument, root)
    meta = _real_meta(file_path, sidecar or {}, _cell_capacity(cell_id, root))
    if report is not None:
        # Stored, not merely acted on. Which profile read this file, and what it
        # could not place, is provenance of exactly the kind the rest of the
        # platform records: six months later the question "why has this
        # experiment no temperature" is answerable from the experiment itself
        # rather than from whoever ran the ingest.
        meta["ingest_report"] = report
    if exp_type == "hppc":
        _hppc_protocol_from_series(meta, series)
    elif exp_type == "eis":
        _eis_from_file(meta, series, file_path)
    return write_experiment(cell_id, exp_type, meta, series, root=root)


def preview_ingest(exp_type: str, file_path: Path, sidecar: dict | None = None,
                   instrument: str | None = None, root: Path | None = None,
                   cell_id: str | None = None) -> dict:
    """Parse + quality-check WITHOUT writing — the ingest preview.

    Returns the mapped columns, the row count, the ingest quality record and
    the rule flags the writer would attach, plus the head of the frame.

    `instrument` takes the same route the write would take, which is the whole
    value of a preview: the flags shown here are the flags that will be stored,
    because both come from `compute_quality` reading the same `ingest_report`.

    `cell_id` is the same idea applied to the capacity reference. The write
    fills `capacity_ref_Ah` from the cell's build record when the sidecar omits
    it, so a preview given no cell would show a capacity-referenced number the
    write is about to compute differently. Given one, the two agree; given
    none, `capacity_SOH` is absent here exactly as it would be there.
    """
    from .tables import check_rules
    from .quality import compute_quality

    series, report = read_for_ingest(exp_type, file_path, instrument, root)
    meta = _real_meta(file_path, sidecar or {}, _cell_capacity(cell_id, root))
    if report is not None:
        meta["ingest_report"] = report
    if exp_type == "eis":
        _eis_from_file(meta, series, file_path)
    quality = compute_quality(exp_type, series, meta)
    quality["flags"] = [f"rule: {r}" for r in check_rules(exp_type, meta)] + list(quality["flags"])
    return {
        "columns": list(series.columns),
        "n_rows": int(len(series)),
        "quality": quality,
        "instrument": meta.get("instrument") or {},
        "protocol": meta.get("protocol") or {},
        "ingest_report": report,
        "head": series.head(8),
    }


# ---------------------------------------------------------------------------
# cycler CSV
# ---------------------------------------------------------------------------

_CYCLER_COLUMN_MAP = {
    "t_s": ("t_s", "time_s", "time/s", "time [s]", "time", "t[s]", "elapsed_time_s"),
    "I_A": ("i_a", "current_a", "current/a", "current [a]", "current", "i[a]", "i_ma"),
    "V_V": ("v_v", "voltage_v", "voltage/v", "voltage [v]", "voltage", "e_v", "ewe/v"),
    "V_full_V": ("v_full_v", "vcell_v"),
    "V_pos_V": ("v_pos_v", "vwe_v"),
    "V_neg_V": ("v_neg_v", "vref_v", "vce_v"),
    "T_degC": ("t_degc", "temperature_c", "temperature [c]", "temperature", "temperature/c", "temp/c"),
    "soc": ("soc", "state_of_charge", "soc_%"),
    "q_Ah": ("q_ah", "capacity_ah", "capacity/ah", "charge_ah", "q_charge"),
    # the space-separated spellings are appended LAST so that first-match-wins
    # keeps every existing parse identical; "Step Type" is a very common cycler
    # export header, and for GITT this column is what separates a pulse from a
    # relaxation, so missing it silently falls back to current-based inference
    "seg_type": ("seg_type", "segment", "step_type", "mode", "step",
                 "step type", "seg type", "segment type"),
    "pulse_id": ("pulse_id", "pulse", "pulse_number"),
    "cycle_index": ("cycle_index", "cycle", "cycle_number", "ncycle"),
}

_UNIT_CONVERSIONS = {"i_ma": 1e-3, "soc_%": 1e-2}


def read_cycler_csv(path: Path, exp_type: str = "hppc") -> pd.DataFrame:
    """Parse a cycler CSV into schema columns (tolerant header matching).

    The schema column for the full-cell voltage differs by type (hppc /
    three_electrode_ts use V_full_V, rpt uses V_V) — the "voltage" CSV column
    maps to the right one per exp_type.
    """
    df = pd.read_csv(path)
    raw = {c.strip().lower(): c for c in df.columns}
    out: dict[str, Any] = {}
    for target, synonyms in _CYCLER_COLUMN_MAP.items():
        for syn in synonyms:
            if syn in raw:
                col = raw[syn]
                if col == "i_ma":
                    out[target] = df[col].to_numpy(dtype=float) * 1e-3
                elif col == "soc_%":
                    out[target] = df[col].to_numpy(dtype=float) * 1e-2
                else:
                    out[target] = df[col].to_numpy()
                break
    if "t_s" not in out:
        raise ValueError(f"cycler CSV has no recognizable time column; found {sorted(raw)}")
    if "V_V" in out and exp_type in ("hppc", "three_electrode_ts"):
        out["V_full_V"] = out.pop("V_V")
    if "seg_type" not in out:
        out["seg_type"] = np.full(len(out["t_s"]), "rest")
    out["seg_type"] = np.asarray(out["seg_type"], dtype=str)
    if exp_type == "three_electrode_ts":
        # The reference-rig CSV carries only the three voltages; the schema
        # ALSO requires the two derived columns, and required-column
        # validation runs before compute_quality. `sum_residual_mV` is the
        # exact residual where the voltages allow it (sum_check recomputes it
        # at ingest anyway); `seg_id` is a run-length id over the segment
        # labels, which a step-log export carries implicitly in the label
        # column.
        n = len(out["t_s"])
        if {"V_full_V", "V_pos_V", "V_neg_V"} <= set(out):
            resid = (np.asarray(out["V_full_V"], dtype=float)
                     - (np.asarray(out["V_pos_V"], dtype=float) - np.asarray(out["V_neg_V"], dtype=float))) * 1e3
        else:
            resid = np.zeros(n)
        out.setdefault("sum_residual_mV", resid)
        if "seg_id" not in out:
            segs = np.asarray(out["seg_type"], dtype=str)
            out["seg_id"] = np.concatenate([[0], np.cumsum(segs[1:] != segs[:-1])]).astype(np.int32)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# GITT (a cycler CSV of pulses and rests)
# ---------------------------------------------------------------------------

# Cycler step-log labels, mapped onto the schema's exact `pulse | relax`
# vocabulary. A real GITT export says "CC Discharge"/"OCV"/"Rest" depending on
# the instrument; consumers dispatch on seg_type, so the mapping has to happen
# here rather than in every consumer.
_GITT_SEG_LABELS = {
    "pulse": ("pulse", "cc", "ccdischarge", "ccdis", "discharge", "charge",
              "cccharge", "current", "pulse_discharge", "pulse_charge"),
    "relax": ("relax", "rest", "ocv", "pause", "idle", "open_circuit", "relaxation"),
}


def _normalize_seg_label(s: str) -> str:
    """Fold an instrument's step label to a bare token: "CC Discharge" -> "ccdischarge"."""
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


# the normalized synonym sets, computed once (matching is on the folded form)
_GITT_SEG_NORM = {target: frozenset(_normalize_seg_label(s) for s in syns)
                  for target, syns in _GITT_SEG_LABELS.items()}


def _has_step_log(path: Path) -> bool:
    """Does the CSV carry a segment/step column at all? (header only, no parse)

    This cannot be answered from `read_cycler_csv`'s output: it fills a missing
    step column with "rest", which is itself a valid relax label — so a bare
    export and a genuinely all-rest file would look identical downstream. The
    two need different handling, so the header is peeked here.
    """
    head = pd.read_csv(path, nrows=0)
    raw = {str(c).strip().lower() for c in head.columns}
    return any(syn in raw for syn in _CYCLER_COLUMN_MAP["seg_type"])


def read_gitt(path: Path) -> pd.DataFrame:
    """Parse a GITT cycler CSV into the gitt schema columns.

    Reuses the tolerant cycler reader, then normalizes the segment vocabulary
    and derives `pulse_id` from the pulse/relax run-lengths when the export
    does not carry one. The initial rest is pulse_id 0 — the schema's way of
    supplying the pre-pulse OCP of step 1 without a third segment type.

    Two export styles are accepted: a step log naming each segment ("CC
    Discharge"/"OCV"/"Rest" — the vocabulary is normalized before matching, so
    spacing and case do not matter), and a bare time/current/voltage export
    with no step column, where the segments are derived from the current
    profile. Anything else is rejected rather than guessed at.
    """
    df = read_cycler_csv(path, exp_type="gitt")
    if _has_step_log(path):
        labels = df["seg_type"].map(_normalize_seg_label)
        seg = pd.Series(index=df.index, dtype=object)
        for target, syns in _GITT_SEG_NORM.items():
            seg[labels.isin(syns)] = target
        unknown = sorted(set(labels[seg.isna()]))
        if unknown:
            raise ValueError(
                f"gitt CSV has segment labels that are neither a pulse nor a relax: {unknown}; "
                f"expected one of {sorted(_GITT_SEG_LABELS['pulse'] + _GITT_SEG_LABELS['relax'])}")
    else:
        # no step log — a GITT is defined by its current profile, so read the
        # pulses straight off it
        if "I_A" not in df.columns or not np.any(np.asarray(df["I_A"], dtype=float)):
            raise ValueError("gitt CSV carries neither segment labels nor a non-zero current "
                             "column — cannot tell the pulses from the rests")
        seg = pd.Series(np.where(np.asarray(df["I_A"], dtype=float) != 0.0, "pulse", "relax"), index=df.index)
    df["seg_type"] = seg.to_numpy(dtype=object)
    if "pulse_id" not in df.columns:
        # a new pulse index opens at every relax -> pulse transition
        s = df["seg_type"].to_numpy()
        opens = (s == "pulse") & np.concatenate([[True], s[1:] != s[:-1]])
        df["pulse_id"] = np.cumsum(opens).astype(np.int32)
    return df


# ---------------------------------------------------------------------------
# EIS XLSX (Palmsense4-style, one sheet per spectrum)
# ---------------------------------------------------------------------------

_EIS_SHEET_VOLTAGE_RE = re.compile(r"^EIS_([-+]?\d+(?:\.\d+)?)V$", re.I)


def read_eis_xlsx(path: Path) -> pd.DataFrame:
    """Parse an EIS XLSX (one sheet per spectrum) into the eis schema columns.

    Each sheet carries a header block then a data table with the columns
    freq/Hz (or f), Zre/Ω, −Z''/Ω (the EIS export convention — Z_im is the
    NEGATIVE of the exported value). soc / V_dc / T come from a sheet named
    "meta" or from per-sheet cell rows; defaults 0.5 / 3.7 / 25.

    Every row gets a `spectrum_id` (the sheet name, conventionally
    `EIS_<index>` or `EIS_<voltage>V`), and a per-sheet "voltage (v)" column
    (some exports carry the Edc value there) overrides the meta default, as
    does a voltage spelled in the sheet name.
    """
    xls = pd.read_excel(path, sheet_name=None, header=None)
    meta = {}
    if "meta" in xls:
        meta = {str(r[0]).strip().lower(): r[1] for r in xls["meta"].itertuples(index=False) if len(r) >= 2}
    frames = []
    for sheet, raw in xls.items():
        if sheet == "meta":
            continue
        raw.columns = [str(c) for c in range(raw.shape[1])]
        # locate the data header row: contains a frequency column
        header_row = None
        for i in range(min(12, len(raw))):
            row = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
            if any("freq" in v or v in ("f", "f/hz") for v in row):
                header_row = i
                break
        if header_row is None:
            continue  # not a spectrum sheet
        names = [str(v).strip().lower() for v in raw.iloc[header_row].tolist()]
        data = raw.iloc[header_row + 1 :].copy()
        data.columns = names
        data = data.apply(pd.to_numeric, errors="coerce").dropna()
        f_col = next(n for n in names if "freq" in n or n in ("f", "f/hz"))
        # the remaining columns in order: Z' then -Z'' — position-aware, since
        # "re" is a substring of "freq" (a naive match grabs the frequency)
        rest = [n for n in names if n != f_col]
        zre_col = next(n for n in rest if "zre" in n or ("re" in n and "freq" not in n))
        zim_neg_col = next(n for n in rest if "zim" in n or ("im" in n and n != zre_col))
        f = data[f_col].to_numpy(dtype=float)
        zre = data[zre_col].to_numpy(dtype=float)
        zim = -data[zim_neg_col].to_numpy(dtype=float)  # -Z'' convention
        z = zre + 1j * zim
        # per-spectrum voltage: the sheet's own "voltage (v)" column, else a
        # voltage spelled in the sheet name, else meta
        v_dc = float(meta.get("v_dc_v", 3.7))
        if "voltage (v)" in names and len(data) and np.isfinite(data["voltage (v)"].iloc[0]):
            v_dc = float(data["voltage (v)"].iloc[0])
        else:
            m = _EIS_SHEET_VOLTAGE_RE.match(str(sheet).strip())
            if m:
                v_dc = float(m.group(1))
        frames.append(pd.DataFrame({
            "f_Hz": f,
            "Z_re_Ohm": zre,
            "Z_im_Ohm": zim,
            "Z_mag_Ohm": np.abs(z),
            "Z_phase_deg": np.degrees(np.angle(z)),
            "soc": float(meta.get("soc", 0.5)),
            "V_dc_V": v_dc,
            "T_degC": float(meta.get("t_degc", 25.0)),
            "electrode": str(meta.get("electrode", "full")),
            "spectrum_id": str(sheet),
        }))
    if not frames:
        raise ValueError(f"EIS XLSX has no spectrum sheets (no frequency header found)")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# EIS Palmsense4 CSV (raw export; in-situ when a mixed-mode GITT ran on it)
# ---------------------------------------------------------------------------

# a mixed-mode header line looks like "...,s,V,s,µA,s,V,s,µA,..." — the
# repeated 4-column (t, V, ?, I-µA) step layout of a GITT running EIS
_MIXED_MODE_RE = re.compile(r"s\s*,\s*V\s*,\s*s\s*,\s*µA")
_MEASUREMENT_RE = re.compile(r"Impedance Spectroscopy(?: \[(\d+)\])?")
_EAC_RE = re.compile(r"\b(?:amplitude|eac)\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(mV|V)", re.I)


@lru_cache(maxsize=8)
def _parse_palmsense4(path_str: str) -> dict[str, Any]:
    """The full Palmsense4 CSV parse, shared by the frame reader and the
    protocol-context builder (parsed once per path).

    Returns {"spectra": [(index, df)], "mixed_mode": {...}|None,
    "correlations": [(index, step, v_end_V)]}.
    """
    raw = Path(path_str).read_bytes()
    lines = None
    for enc in ("utf-16-le", "utf-8"):
        try:
            lines = raw.decode(enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    if lines is None:
        raise ValueError("Palmsense4 CSV is neither UTF-16-LE nor UTF-8")

    # measurement blocks: every `Measurement:,Impedance Spectroscopy [n]` line
    # opens a block that ends at the next measurement line (or EOF)
    measurement_info = []
    for i, line in enumerate(lines):
        if line.strip().startswith("Measurement:,Impedance Spectroscopy"):
            m = _MEASUREMENT_RE.search(line)
            measurement_info.append({"line": i, "index": (m.group(1) if m else None) or "0"})
    if not measurement_info:
        raise ValueError("no `Measurement:,Impedance Spectroscopy` blocks in the CSV")

    spectra: list[tuple[str, pd.DataFrame]] = []
    for idx, info in enumerate(measurement_info):
        end = measurement_info[idx + 1]["line"] if idx < len(measurement_info) - 1 else len(lines)
        df = _ps4_spectrum(lines[info["line"]:end])
        if df is not None:
            spectra.append((info["index"], df))

    # mixed-mode section lives BEFORE the first EIS measurement
    pre = lines[: measurement_info[0]["line"]]
    mixed_mode = None
    correlations: list[tuple[str, int, float]] = []
    header_idx = None
    for i, line in enumerate(pre):
        if _MIXED_MODE_RE.search(line) and line.count("s,V") > 1:
            header_idx = i
            break
    if header_idx is not None:
        table = pd.read_csv(io.StringIO("\n".join(pre[header_idx:])), low_memory=False)
        steps = []
        n_steps = len(table.columns) // 4
        for step in range(n_steps):
            step_df = table.iloc[:, step * 4:(step + 1) * 4]
            step_df.columns = [str(c).split(",")[-1].strip() if "," in str(c) else str(c).strip()
                               for c in table.columns[step * 4:(step + 1) * 4]]
            step_df = step_df.apply(pd.to_numeric, errors="coerce").dropna()
            if not len(step_df):
                continue
            v_end = float(step_df.iloc[:, 1].iloc[-1])
            i_mean = float(step_df.iloc[:, 3].mean())
            steps.append({"step": step, "v_end_V": v_end, "i_mean_uA": i_mean})
        mixed_mode = {"steps": steps}
        # the correlation: an EIS runs after each negative-current GITT step;
        # the measurement sits at the END VOLTAGE of the step before it, and
        # the k-th correlated step pairs with the k-th EIS block
        neg = [s["step"] for s in steps if s["i_mean_uA"] < 0]
        prev = [s - 1 for s in neg if s > 0]
        by_step = {s["step"]: s for s in steps}
        sorted_indices = sorted((idx for idx, _ in spectra), key=int)
        for k, step in enumerate(prev):
            if k < len(sorted_indices) and step in by_step:
                correlations.append((sorted_indices[k], step, by_step[step]["v_end_V"]))
    return {"spectra": spectra, "mixed_mode": mixed_mode, "correlations": correlations}


def _ps4_spectrum(section_lines: list[str]) -> pd.DataFrame | None:
    """One measurement block -> a clean (freq, Z', Z'', [Edc]) data frame."""
    for i, line in enumerate(section_lines):
        if not line.strip().startswith("freq / Hz"):
            continue
        data_lines = []
        for ln in section_lines[i + 1:]:
            s = ln.strip()
            if not s:
                continue
            if not re.match(r"^[-+0-9.Ee]", s):
                break  # metadata after the table
            data_lines.append(s)
        if not data_lines:
            return None
        return pd.read_csv(io.StringIO("\n".join([section_lines[i]] + data_lines)))
    return None


def read_palmsense4_csv(path: Path) -> pd.DataFrame:
    """Parse a raw Palmsense4 EIS CSV (UTF-16-LE) into the eis schema columns.

    One measurement block per spectrum; `spectrum_id` is `EIS_<voltage>V`
    when the mixed-mode GITT correlation assigned a voltage, else
    `EIS_<index>`. V_dc per spectrum: the block's own `Edc / V`
    value when present, else the correlated step end voltage, else 3.7.

    Recorded precedence, so it is a decision and not an accident: the two
    sources answer different questions. `Edc` is the block's own measured DC
    bias and wins for `V_dc`; the correlation is what IDENTIFIES a block (the
    k-th EIS belongs to the step before the k-th negative-current step) and so
    names it in mixed mode. For a standalone file the identity is the block's
    own index — `EIS_<index>` traces back to `Impedance Spectroscopy [<index>]`
    in the file, which an Edc-derived name would throw away.
    """
    parsed = _parse_palmsense4(str(path))
    corr = {i: v for i, _s, v in parsed["correlations"]}
    frames = []
    for eis_idx, df in parsed["spectra"]:
        f = df["freq / Hz"].to_numpy(dtype=float)
        zre = df["Z' / Ohm"].to_numpy(dtype=float)
        zim = -df["Z'' / Ohm"].to_numpy(dtype=float)  # -Z'' convention
        z = zre + 1j * zim
        if "Edc / V" in df.columns and np.isfinite(df["Edc / V"].iloc[0]):
            v_dc = float(df["Edc / V"].iloc[0])
        elif eis_idx in corr:
            v_dc = corr[eis_idx]
        else:
            v_dc = 3.7
        frames.append(pd.DataFrame({
            "f_Hz": f,
            "Z_re_Ohm": zre,
            "Z_im_Ohm": zim,
            "Z_mag_Ohm": np.abs(z),
            "Z_phase_deg": np.degrees(np.angle(z)),
            "soc": 0.5,
            "V_dc_V": v_dc,
            "T_degC": 25.0,
            "electrode": "full",
            "spectrum_id": f"EIS_{v_dc:.4f}V" if eis_idx in corr else f"EIS_{eis_idx}",
        }))
    if not frames:
        raise ValueError("Palmsense4 CSV has EIS measurement blocks but no parseable spectra")
    return pd.concat(frames, ignore_index=True)


def palmsense4_context(path: Path) -> dict[str, Any]:
    """The protocol context of a Palmsense4 CSV: eis_mode, spectrum list,
    the mixed-mode GITT step summary and the EIS↔step correlations."""
    parsed = _parse_palmsense4(str(path))
    corr = {i: v for i, _s, v in parsed["correlations"]}
    spectra = sorted((idx for idx, _ in parsed["spectra"]), key=int)
    ctx: dict[str, Any] = {
        "eis_mode": "insitu" if parsed["mixed_mode"] and corr else "standalone",
        "n_spectra": len(spectra),
        "spectrum_ids": [f"EIS_{corr[i]:.4f}V" if i in corr else f"EIS_{i}" for i in spectra],
    }
    if parsed["mixed_mode"]:
        ctx["mixed_mode_steps"] = parsed["mixed_mode"]["steps"]
        ctx["correlations"] = [{"spectrum_id": f"EIS_{corr[i]:.4f}V", "eis_index": i, "step": s}
                               for i, s, _ in parsed["correlations"]]
    return ctx


def _derive_eis_instrument(meta: dict, series: pd.DataFrame) -> None:
    """Fill the instrument block from the data where the sidecar didn't.

    f_min/f_max and points_per_decade come straight off the measured grid;
    the sidecar always wins on keys it provided. The amplifier markers live
    in the file header, not the frame, so `_eis_from_file` supplies those.
    """
    inst = meta.setdefault("instrument", {})
    f = series["f_Hz"].to_numpy(dtype=float)
    if inst.get("f_min_Hz") is None:
        inst["f_min_Hz"] = float(f.min())
    if inst.get("f_max_Hz") is None:
        inst["f_max_Hz"] = float(f.max())
    if inst.get("points_per_decade") is None:
        grids = []
        ids = series.get("spectrum_id") if "spectrum_id" in series.columns else None
        for _, grp in (series.groupby(ids, sort=False) if ids is not None else [(None, series)]):
            uf = np.unique(grp["f_Hz"].to_numpy(dtype=float))
            if len(uf) >= 2 and uf.max() > uf.min():
                grids.append((len(uf) - 1) / np.log10(uf.max() / uf.min()))
        inst["points_per_decade"] = round(float(np.mean(grids)), 3) if grids else None


def _eis_from_file(meta: dict, series: pd.DataFrame, file_path: Path) -> None:
    """The eis ingest-side derivation: instrument from the data, protocol
    context (eis_mode / mixed-mode correlation) from the file format."""
    _derive_eis_instrument(meta, series)
    inst = meta["instrument"]
    proto = meta.setdefault("protocol", {})
    if file_path.suffix.lower() == ".csv":
        if inst.get("potentiostat") is None:
            inst["potentiostat"] = "Palmsense4"
        if inst.get("amplitude_mV") is None:
            # best-effort header scan: the excitation amplitude is usually a
            # measurement-info line; never guess a unit or a magnitude
            for line in Path(file_path).read_bytes().decode("utf-16-le", errors="ignore").splitlines()[:200]:
                m = _EAC_RE.search(line)
                if m:
                    val = float(m.group(1))
                    inst["amplitude_mV"] = val if m.group(2).lower() == "mv" else val * 1e3
                    break
        proto.update(palmsense4_context(file_path))
    else:
        proto["eis_mode"] = "standalone"
        if "spectrum_id" in series.columns:
            proto["spectrum_ids"] = sorted(set(series["spectrum_id"].astype(str)), key=lambda s: (len(s), s))
    meta["instrument"] = inst
    meta["protocol"] = proto


# ---------------------------------------------------------------------------
# stack-pressure CSV
# ---------------------------------------------------------------------------

_PRESSURE_COLUMN_MAP = {
    "t_s": ("t_s", "time_s", "time/s", "time"),
    "F_N": ("f_n", "force_n", "force/n", "force [n]"),
    "p_kPa": ("p_kpa", "pressure_kpa", "pressure/kpa", "pressure [kpa]"),
    "thickness_um": ("thickness_um", "thickness/um", "gap_um"),
    "T_degC": ("t_degc", "temperature_c", "temperature [c]", "temperature", "temperature/c", "temp/c"),
    "V_V": ("v_v", "voltage_v", "voltage [v]", "voltage"),
    "I_A": ("i_a", "current_a", "current [a]", "current"),
    "soc": ("soc", "state_of_charge"),
    "seg_type": ("seg_type", "segment", "step", "step_type", "mode"),
}


def read_pressure_csv(path: Path) -> pd.DataFrame:
    """Parse a stack-pressure rig CSV into schema columns (tolerant headers)."""
    df = pd.read_csv(path)
    raw = {c.strip().lower(): c for c in df.columns}
    out: dict[str, Any] = {}
    for target, synonyms in _PRESSURE_COLUMN_MAP.items():
        for syn in synonyms:
            if syn in raw:
                out[target] = df[raw[syn]].to_numpy()
                break
    if "t_s" not in out or "p_kPa" not in out:
        raise ValueError(f"pressure CSV needs t_s and p_kPa; found {sorted(raw)}")
    if "seg_type" not in out:
        out["seg_type"] = np.full(len(out["t_s"]), "rest")
    out["seg_type"] = np.asarray(out["seg_type"], dtype=str)
    return pd.DataFrame(out)


def synonym_mapping(raw) -> dict:
    """`{source header: schema column or None}` from the built-in synonym list.

    The same table `read_cycler_csv` matches against, exposed as a PROPOSAL
    about a frame somebody already has, rather than a rule applied invisibly
    inside a reader. `None` for a header it does not know, which is the useful
    answer: `None` is the column a person has to look at.
    """
    synonyms = {syn: target for target, syns in _CYCLER_COLUMN_MAP.items() for syn in syns}
    return {str(c): synonyms.get(str(c).strip().lower()) for c in raw.columns}


def promote(cell_id: str, experiment_id: str, exp_type: str, instrument: str | None = None,
            root: Path | None = None) -> Path:
    """Re-read a stored experiment as a real type. A new experiment, `derived_from` the old.

    The way out of `misc`. `misc` accepts a file the schema has no shape for,
    stores it exactly as it arrived, and unlocks nothing, which makes it a good
    place to LAND a file and a bad place to leave one. Name the type, name the
    profile, and the file becomes a real experiment.

    Read from the store, not from the source file. `raw/series.parquet` is the
    table as it arrived, headers and all, and it is in the store, so a
    promotion works months later on a machine that never saw the original.

    APPEND-ONLY, SO THIS ADDS RATHER THAN EDITS. The `misc` experiment stays
    exactly where it is and the new one records `derived_from`, which is the
    schema's correction mechanism. Rewriting the original in place is the one
    thing a store like this must never do, because every artifact derived from
    it was derived from what it USED to say.
    """
    from .instruments import InstrumentProfile, load_profiles, map_frame
    from .tables import SPECS
    from .write_experiment import _experiment_dir

    d = _experiment_dir(cell_id, experiment_id, root)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no experiment {experiment_id!r} for cell {cell_id!r}")
    source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if source_meta.get("experiment_type") == exp_type:
        raise ValueError(f"{experiment_id} is already a {exp_type}; promotion is for changing "
                         "the type, and re-reading a file as what it already is would only "
                         "make a second copy of it")

    raw_path = d / "raw" / "series.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{experiment_id} has no raw/series.parquet, so there is nothing to re-read. "
            "Only experiments written by this platform's own writer carry one.")
    raw = pd.read_parquet(raw_path)

    if instrument:
        profiles = load_profiles(root)
        if instrument not in profiles:
            raise ValueError(f"no instrument profile named {instrument!r}; "
                             f"available: {sorted(profiles)}")
        profile = profiles[instrument]
    else:
        # No profile named: the synonym list, as a profile, so that BOTH routes
        # go through one mapping and produce one report. A promotion that read
        # its columns a second way would be a second place for a column to
        # vanish.
        from .instruments import ColumnRule

        profile = InstrumentProfile(
            name="synonyms", vendor="",
            columns=tuple(ColumnRule(source=src, target=tgt)
                          for src, tgt in synonym_mapping(raw).items() if tgt),
            evidence="the built-in synonym list, used because no profile was named")

    frame, report = map_frame(raw, profile)
    report["exp_type"] = exp_type
    spec_cols = {c.name for c in SPECS[exp_type].columns}
    report["required_missing"] = [c.name for c in SPECS[exp_type].columns
                                  if c.required and c.name not in frame.columns]
    report["not_in_schema"] = ([] if SPECS[exp_type].permissive
                               else sorted(set(frame.columns) - spec_cols))
    if "cycle_index" in frame.columns:
        frame["cycle_index"] = frame["cycle_index"].fillna(0).astype("int32")

    meta = {
        "producer": {**(source_meta.get("producer") or {}), "kind": "real",
                     "software": "monocell-promote", "version": __version__,
                     "generated_at": datetime.now(timezone.utc).isoformat()},
        "timing": source_meta.get("timing") or {"ntp_synced": False, "t_epoch_s": 0.0,
                                                "clock_offset_s": {}},
        "protocol": source_meta.get("protocol") or {"description": f"{exp_type} (promoted)"},
        "instrument": source_meta.get("instrument") or {},
        "cell_state": dict(source_meta.get("cell_state") or {}),
        # The whole point: the new experiment says where it came from, and the
        # old one is still there to be looked at.
        "derived_from": experiment_id,
        "ingest_report": report,
    }
    cap = _cell_capacity(cell_id, root)
    if cap is not None:
        meta["cell_state"].setdefault("capacity_ref_Ah", cap)
    return write_experiment(cell_id, exp_type, meta, frame, root=root)
