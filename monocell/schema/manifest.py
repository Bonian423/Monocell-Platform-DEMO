"""DuckDB manifest — a rebuildable query index over data/experiments and
data/artifacts. The parquet + JSON files are the source of truth; the
manifest can always be rebuilt from them (no migrations, no ORM).
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import duckdb

from ..cells import data_root

_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id VARCHAR PRIMARY KEY,
    cell_id VARCHAR,
    exp_type VARCHAR,
    created_at VARCHAR,
    path VARCHAR,
    producer_kind VARCHAR
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id VARCHAR PRIMARY KEY,
    cell_id VARCHAR,
    module VARCHAR,
    created_at VARCHAR,
    path VARCHAR
);
-- DERIVED ONLY. Populated exclusively inside `rebuild_manifest`, from the
-- `meta.json` files it already parses; NO writer upserts it, and
-- `write_experiment` does not know it exists.
--
-- The distinction this turns on is worth being exact about. The rule the
-- manifest lives by is "plain files
-- are the source of truth and the index is rebuildable from them". A table a
-- WRITER maintains breaks that rule: the writer becomes responsible for keeping
-- a second copy of the flags in step, and the first writer that forgets makes
-- the index disagree with the files without anything noticing. A table only the
-- REBUILD writes cannot drift, because there is nowhere for it to drift from —
-- it is a cache of a scan, and deleting it loses nothing.
--
-- `flags` is the JSON list verbatim rather than one row per flag, because every
-- reader wants the whole list and a row-per-flag table would need a GROUP BY to
-- answer the one question anybody asks. `n_flags` is what `flagged` filters on,
-- so the common query touches no JSON at all.
--
-- The PRESENCE of a row is the second thing this holds: an experiment with a
-- row and `n_flags = 0` was scanned and is clean, and one with no row has not
-- been scanned since it was written. Those are different, and conflating them
-- would make every experiment ingested since the last rebuild read as clean.
CREATE TABLE IF NOT EXISTS experiment_flags (
    experiment_id VARCHAR PRIMARY KEY,
    n_flags INTEGER,
    flags VARCHAR
);
"""


def manifest_path(root: Path | None = None) -> Path:
    return (root or data_root()) / "manifest.duckdb"


# How long a connection will wait for the file to be free before giving up.
# DuckDB is single-writer per file, and the index is read while it is being
# written whenever two things touch the store at once: a reader on one thread
# while `write_experiment` upserts from another, or two CLI commands. On Windows
# that shows up as `IO Error: ... being used by another process` from CONNECT,
# not from the query.
#
# Two seconds of patience, in short naps. Long enough to cover an upsert (which
# is one row and a commit), short enough that a genuinely stuck file still
# fails while somebody is watching. Not a lock and not a queue — the store is
# single-machine by design, and the contention is milliseconds wide.
_OPEN_ATTEMPTS = 40
_OPEN_PAUSE_S = 0.05


def _open(root: Path | None = None):
    """One connection to the index file, with the DDL applied."""
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for _ in range(_OPEN_ATTEMPTS):
        try:
            con = duckdb.connect(str(path))
            break
        except Exception as exc:  # noqa: BLE001 — re-raised below if it never clears
            last = exc
            time.sleep(_OPEN_PAUSE_S)
    else:
        raise last  # type: ignore[misc]
    con.execute(_DDL)
    return con


# Open connections, keyed by (thread, index file). Thread-keyed because a
# DuckDB connection is not safe to use from two threads at once.
_SESSIONS: dict[tuple[int, str], list] = {}


@contextlib.contextmanager
def session(root: Path | None = None):
    """Hold ONE connection open for the duration of a logical operation.

    Opening the file is the cost. `duckdb.connect` plus the `CREATE TABLE IF
    NOT EXISTS` block measures about 11 ms on Windows, and the queries it
    carries are microseconds, so an operation that asks fifty small questions
    can spend half its time connecting to the same file.

    **Scoped to an operation, never to a process.** DuckDB allows one
    read-write process per file, so a connection held for the life of a
    long-running process would lock every CLI command out of the store. An
    operation is typically under a couple of seconds, which is inside the
    patience `_OPEN_ATTEMPTS` already grants.

    Re-entrant: an inner `session()` for the same root and thread joins the
    outer one and leaves it open, so an outer operation can hold a session
    without its callees having to know whether one is already open.
    """
    key = (threading.get_ident(), str(manifest_path(root)))
    entry = _SESSIONS.get(key)
    if entry is not None:
        entry[1] += 1
        try:
            yield entry[0]
        finally:
            entry[1] -= 1
        return

    con = _open(root)
    _SESSIONS[key] = [con, 1]
    try:
        yield con
    finally:
        _SESSIONS.pop(key, None)
        con.close()


def in_session(root_at: int, root_name: str = "root"):
    """Run the wrapped function inside one index session.

    A decorator rather than a `with` inside each body, because the functions
    that want this are 60-line loops and wrapping them would reindent every
    line of a function whose logic is not changing. `root_at` is where the data
    root sits in the positional signature; `root_name` is its keyword.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if root_name in kwargs:
                root = kwargs[root_name]
            elif len(args) > root_at:
                root = args[root_at]
            else:
                root = None
            with session(Path(root) if root is not None else None):
                return fn(*args, **kwargs)
        return wrapper
    return deco


@contextlib.contextmanager
def _conn(root: Path | None = None):
    """The connection for one statement: the session's, or a fresh one.

    A caller outside a session gets exactly the old behaviour — open, use,
    close — so nothing has to know about sessions to keep working.
    """
    entry = _SESSIONS.get((threading.get_ident(), str(manifest_path(root))))
    if entry is not None:
        yield entry[0]
        return
    con = _open(root)
    try:
        yield con
    finally:
        con.close()


def content_hash(*paths: Path) -> str:
    """sha256 over the given files' bytes — the artifact invalidation anchor."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()


def upsert_experiment(rec: dict[str, Any], root: Path | None = None) -> None:
    with _conn(root) as con:
        con.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
            [rec["experiment_id"], rec["cell_id"], rec["exp_type"], rec["created_at"], rec["path"], rec["producer_kind"]],
        )


def upsert_artifact(rec: dict[str, Any], root: Path | None = None) -> None:
    with _conn(root) as con:
        con.execute(
            "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?)",
            [rec["artifact_id"], rec["cell_id"], rec["module"], rec["created_at"], rec["path"]],
        )


def find_experiment(experiment_id: str, root: Path | None = None) -> dict[str, Any] | None:
    with _conn(root) as con:
        row = con.execute(
            "SELECT experiment_id, cell_id, exp_type, created_at, path, producer_kind "
            "FROM experiments WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()
    if row is None:
        return None
    keys = ("experiment_id", "cell_id", "exp_type", "created_at", "path", "producer_kind")
    return dict(zip(keys, row))


def list_experiments(cell_id: str, exp_type: str | None = None, root: Path | None = None,
                     *, include_retracted: bool = False) -> list[dict[str, Any]]:
    """This cell's experiments. Retracted ones are left out unless asked for.

    The retraction is read off the DISK — one `stat` per row for a
    `retracted.json` beside the meta — and not out of the index. A derived
    table like `experiment_flags` would be wrong here: an experiment retracted
    since the last rebuild would go on feeding every fit until somebody
    remembered to re-index, which is the failure the feature exists to prevent.

    This is the single point that makes retraction mean something. Every reader
    in the platform asks the manifest for a cell's experiments through here, so
    filtering here is what keeps a withdrawn run out of the pooled analyses
    without eleven callers each having to remember.
    """
    q = "SELECT experiment_id, cell_id, exp_type, created_at, path, producer_kind FROM experiments WHERE cell_id = ?"
    args: list = [cell_id]
    if exp_type:
        q += " AND exp_type = ?"
        args.append(exp_type)
    with _conn(root) as con:
        rows = con.execute(q, args).fetchall()
    keys = ("experiment_id", "cell_id", "exp_type", "created_at", "path", "producer_kind")
    out = [dict(zip(keys, r)) for r in rows]
    if include_retracted:
        return out
    from .write_experiment import RETRACTED_FILENAME

    return [r for r in out if not (Path(r["path"]) / RETRACTED_FILENAME).exists()]


def list_artifacts(cell_id: str, module: str | None = None, root: Path | None = None) -> list[dict[str, Any]]:
    q = "SELECT artifact_id, cell_id, module, created_at, path FROM artifacts WHERE cell_id = ?"
    args: list = [cell_id]
    if module:
        q += " AND module = ?"
        args.append(module)
    with _conn(root) as con:
        rows = con.execute(q, args).fetchall()
    keys = ("artifact_id", "cell_id", "module", "created_at", "path")
    return [dict(zip(keys, r)) for r in rows]


def rebuild_manifest(root: Path | None = None) -> dict[str, Any]:
    """Re-scan data/experiments and data/artifacts into a fresh manifest.

    Returns a summary of what the scan saw: files read, rows indexed, and the
    files it could not use. Without these numbers a rebuild that indexed a
    fifth of the store would look exactly like one that indexed all of it, and
    the two commonest causes, a duplicate id and an unreadable file, are both
    silent by nature.

    Also the ONE place `experiment_flags` is written. It is a derived-only
    table (see the schema above): the rebuild already parses every `meta.json`,
    so the flags come free, and nothing else may touch it — which is what makes
    it a cache of a scan rather than a second source of truth.
    """
    root = root or data_root()
    seen_exp = seen_art = 0
    unreadable: list[str] = []
    incomplete: list[str] = []
    exp_ids: set[str] = set()
    art_ids: set[str] = set()
    duplicate_exp: list[str] = []
    duplicate_art: list[str] = []
    exp_rows: list[tuple] = []
    art_rows: list[tuple] = []
    flag_rows: list[tuple[str, int, str]] = []

    exp_root = root / "experiments"
    if exp_root.exists():
        for meta_f in exp_root.glob("*/*/meta.json"):
            seen_exp += 1
            try:
                meta = json.loads(meta_f.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                # One unreadable sidecar must not abandon the scan: the rebuild
                # is what a user runs BECAUSE the store is inconsistent.
                unreadable.append(f"{meta_f}: {type(exc).__name__}")
                continue
            if not (meta_f.parent / "series.parquet").exists():
                # An experiment is meta + series; `load_experiment` requires
                # both. Indexing half of one only moves the failure to whoever
                # reads it next, so the scan leaves it out and names it. An
                # interrupted ingest and a hand-deleted file both land here.
                incomplete.append(str(meta_f.parent))
                continue
            if meta.get("experiment_id") in exp_ids:
                duplicate_exp.append(str(meta_f))
                continue
            exp_ids.add(meta.get("experiment_id"))
            exp_rows.append((meta["experiment_id"], meta["cell_id"], meta["experiment_type"],
                             meta.get("created_at", ""), str(meta_f.parent),
                             meta["producer"]["kind"]))
            flags = [f for f in ((meta.get("quality") or {}).get("flags") or []) if f]
            flag_rows.append((meta["experiment_id"], len(flags), json.dumps(flags)))

    art_root = root / "artifacts"
    if art_root.exists():
        for art_f in art_root.glob("*/*/*.json"):
            seen_art += 1
            try:
                art = json.loads(art_f.read_text(encoding="utf-8"))
                art["artifact_id"], art["cell_id"], art["module"], art["created_at"]
            except (OSError, ValueError, KeyError) as exc:
                unreadable.append(f"{art_f}: {type(exc).__name__}")
                continue
            if art["artifact_id"] in art_ids:
                # `artifact_id` is the table's primary key, so a second file
                # claiming one silently replaced the first and the index kept
                # whichever the scan reached last. Reported for the same reason
                # the experiment case is: a store that loses most of its
                # artifacts to duplicate ids should not look like one that
                # indexed them all.
                duplicate_art.append(str(art_f))
                continue
            art_ids.add(art["artifact_id"])
            art_rows.append((art["artifact_id"], art["cell_id"], art["module"],
                             art["created_at"], str(art_f)))

    # One connection for the write, and one statement per table. Row-at-a-time
    # through `upsert_experiment` would open the file once per row, which
    # dominates the rebuild of a large store.
    with session(root) as con:
        # DROP rather than deleting the file: a session may be holding it open,
        # and on Windows an open file cannot be unlinked. This also still
        # rebuilds the schema, so a `_DDL` change lands on the next rebuild
        # exactly as it did before.
        con.execute("DROP TABLE IF EXISTS experiments; DROP TABLE IF EXISTS artifacts; "
                    "DROP TABLE IF EXISTS experiment_flags;")
        con.execute(_DDL)
        if exp_rows:
            con.executemany("INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
                            exp_rows)
        if flag_rows:
            con.executemany("INSERT OR REPLACE INTO experiment_flags VALUES (?, ?, ?)",
                            flag_rows)
        if art_rows:
            con.executemany("INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?)",
                            art_rows)
        n_exp = con.execute("SELECT count(*) FROM experiments").fetchone()[0]
        n_art = con.execute("SELECT count(*) FROM artifacts").fetchone()[0]
    return {
        "experiment_files": seen_exp, "experiments_indexed": int(n_exp),
        "artifact_files": seen_art, "artifacts_indexed": int(n_art),
        "duplicate_experiment_ids": duplicate_exp,
        "duplicate_artifact_ids": duplicate_art,
        "incomplete": incomplete,
        "unreadable": unreadable,
    }


# ---------------------------------------------------------------------------
# asking the index a question somebody typed
# ---------------------------------------------------------------------------

_EXP_KEYS = ("experiment_id", "cell_id", "exp_type", "created_at", "path", "producer_kind")


def search_experiments(root: Path | None = None, *, cells: list[str] | None = None,
                       exp_types: list[str] | None = None, producer: str | None = None,
                       since: str | None = None, until: str | None = None,
                       flagged: bool | None = None, lot: str | None = None,
                       limit: int | None = None) -> list[dict[str, Any]]:
    """Experiments matching every filter given, newest first.

    `list_experiments` takes one cell and one type, which answers "what does
    this cell have" and not "where is that run". This answers the second.

    TWO PASSES, AND THE SPLIT IS THE HONEST PART. Everything the index holds is
    filtered in SQL: cell, type, producer, and the date window (`created_at` is
    an ISO string, so lexicographic comparison IS chronological comparison, and
    a prefix like `2026-05` is a legal bound). `flagged` is not in the index —
    quality lives in each experiment's own `meta.json` — so it is a second pass
    over whatever the first pass left, and it reads files. Putting the flags in
    the index instead would make every writer responsible for keeping a second
    copy of them in step, which is the drift the manifest's design avoids.

    `lot` resolves to its member cells first. A batch is a JSON file and a glob
    is the right index for tens of them, so this is a join done in Python on
    purpose rather than a table that would have to be upserted.
    """
    where: list[str] = []
    args: list[Any] = []

    if lot:
        from ..batches import load_batch

        try:
            record = load_batch(lot, root)
        except FileNotFoundError:
            return []
        members = [m.get("cell_id") for m in record.get("cells", []) if m.get("cell_id")]
        if not members:
            return []
        cells = sorted(set(members) & set(cells)) if cells else members
        if not cells:
            return []
    if cells:
        where.append("cell_id IN (" + ", ".join("?" * len(cells)) + ")")
        args.extend(cells)
    if exp_types:
        where.append("exp_type IN (" + ", ".join("?" * len(exp_types)) + ")")
        args.extend(exp_types)
    if producer:
        where.append("producer_kind = ?")
        args.append(producer)
    if since:
        where.append("created_at >= ?")
        args.append(since)
    if until:
        # A partial bound names a PERIOD and the period is included: `2026-05`
        # means through the end of May and `2026-05-31` through the end of that
        # day. Compared as-is, a bare date means its midnight, so
        # `--until 2026-05-31` would drop everything that happened on the 31st
        # — a filter nobody wants and everybody writes by accident. The `T99`
        # sorts after any real time-of-day and after any longer date prefix,
        # which is what makes one line do both cases.
        args.append(until if len(until) > 10 else until + "T99")
        where.append("created_at <= ?")

    query = ("SELECT " + ", ".join(_EXP_KEYS) + " FROM experiments"
             + (" WHERE " + " AND ".join(where) if where else "")
             + " ORDER BY created_at DESC")
    with _conn(root) as con:
        rows = [dict(zip(_EXP_KEYS, r)) for r in con.execute(query, args).fetchall()]
        # The derived flag cache, for the rows the first pass left. An
        # experiment WITH a row was scanned by the last `rebuild_manifest`; one
        # without has been written since, and falls through to reading its own
        # meta below. That fallback is not a fast path made slow — it is the
        # only correct answer for a file the scan has never seen, and it is why
        # the cache can be absent, partial or deleted without any answer here
        # changing.
        cached = _cached_flags(con, [r["experiment_id"] for r in rows])

    out: list[dict[str, Any]] = []
    for row in rows:
        flags = cached.get(row["experiment_id"])
        row["flags"] = _flags_of(row) if flags is None else flags
        if flagged is not None and bool(row["flags"]) != flagged:
            continue
        out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def _cached_flags(con, experiment_ids: list[str]) -> dict[str, list[str]]:
    """`{experiment_id: flags}` for the ids the derived table has scanned.

    Absent from the dict means NOT SCANNED, which is different from scanned and
    clean — an experiment ingested since the last rebuild has no row, and
    reading that as "no flags" would quietly clear every warning on the newest
    data in the store, which is the data anybody is actually looking at.

    Chunked, because DuckDB's parameter list is not unbounded and a store with
    twenty thousand experiments is the case this table exists for.
    """
    out: dict[str, list[str]] = {}
    if not experiment_ids:
        return out
    chunk = 900
    for i in range(0, len(experiment_ids), chunk):
        batch = experiment_ids[i:i + chunk]
        placeholders = ", ".join("?" * len(batch))
        try:
            rows = con.execute(
                f"SELECT experiment_id, flags FROM experiment_flags "
                f"WHERE experiment_id IN ({placeholders})", batch).fetchall()
        except Exception:
            # A manifest written before this table existed has no such table.
            # Falling back is the whole point of the cache being derived: an
            # older store answers exactly as it did, one rebuild later it is
            # fast, and nothing in between is wrong.
            return out
        for exp_id, blob in rows:
            try:
                out[exp_id] = [f for f in json.loads(blob or "[]") if f]
            except ValueError:
                continue
    return out


def _flags_of(row: dict[str, Any]) -> list[str]:
    """This experiment's ingest quality flags, read from its own meta.

    An unreadable or absent meta reads as NO flags rather than raising: a row
    the index knows about and the disk does not is a broken store, and the
    answer to a search is the wrong place to discover that. `rebuild_manifest`
    is what fixes it, and it is documented as the fix.
    """
    try:
        meta = json.loads((Path(row["path"]) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [f for f in ((meta.get("quality") or {}).get("flags") or []) if f]


def recent_activity(root: Path | None = None, *, limit: int = 12,
                    cells: list[str] | None = None) -> list[dict[str, Any]]:
    """The store's last N events, newest first — experiments and artifacts together.

    Both tables carry `created_at`, and the question "what has happened here
    lately" does not care which of the two a row came from: a file landing and a
    fit being re-run are the same kind of news. So they are unioned in SQL and
    ordered once, rather than merged in Python from two ordered lists — which is
    the version that quietly drops one table's rows when the other has more than
    `limit` of them.

    NOT "since you last looked". That needs a per-viewer marker, and this
    platform has no per-viewer anything by design (no server, no users: the
    file store IS the collaboration mechanism). A marker would have to live
    somewhere, and every candidate is either a lie about who is looking or a
    new kind of state this store does not keep.
    """
    where = ""
    args: list[Any] = []
    if cells is not None:
        if not cells:
            return []
        placeholders = ", ".join("?" * len(cells))
        where = f" WHERE cell_id IN ({placeholders})"
        args = list(cells) + list(cells)

    query = (
        f"SELECT 'experiment' AS kind, experiment_id AS id, cell_id, exp_type AS what, "
        f"created_at, path FROM experiments{where} "
        "UNION ALL "
        f"SELECT 'artifact' AS kind, artifact_id AS id, cell_id, module AS what, "
        f"created_at, path FROM artifacts{where} "
        "ORDER BY created_at DESC LIMIT ?")
    with _conn(root) as con:
        rows = con.execute(query, [*args, int(limit)]).fetchall()
    keys = ("kind", "id", "cell_id", "what", "created_at", "path")
    return [dict(zip(keys, r)) for r in rows]
