"""What ran, when, how long it took, and what it raised.

Without a record, a re-derive that failed at 02:00 or an ingest that read the
wrong column leaves nothing behind once the process exits. The CLI prints to a
terminal somebody may have closed, and terminal output is not something you can
send to the person who wrote the code.

**The log lives in the data store**, at `<root>/logs/monocell.log`. That is the
one decision here worth arguing about, and it is deliberate: the store is what
somebody copies to a second machine, ships to a colleague, or attaches to a bug
report. A log in `~/.monocell` or beside the source would be the one file that
did not travel with the thing it describes.

Rotating, 1 MB across four files, so a long-running process cannot fill a
disk. Old lines are the ones to lose: this is a record of recent activity, not
an audit trail — the artifacts are the audit trail, and they are immutable and
hashed.

**Nothing here is configured at import time.** A module that installs a handler
when imported writes a file the moment anything touches the package, including
`--help`, and it writes it into whatever directory the process happened to start
in. `configure()` is called by the entry point: the CLI, once per invocation.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOG_DIRNAME = "logs"
LOG_FILENAME = "monocell.log"
MAX_BYTES = 1_000_000
BACKUPS = 3

_LOGGER_NAME = "monocell"
# The path the current handler is writing to, so `configure` can tell a repeat
# call from a change of store. `None` means nothing is installed yet.
_installed: Path | None = None


def log_path(root: Path | None = None) -> Path:
    from .cells import data_root

    return (root or data_root()) / LOG_DIRNAME / LOG_FILENAME


def configure(root: Path | None = None, *, level: int = logging.INFO) -> Path | None:
    """Point the log at this store. Idempotent; returns the file, or None.

    Returns None when the file could not be opened — a read-only store, a path
    that is not writable, a directory somebody deleted underneath us. **That is
    not an error here.** Refusing to run because the logging failed would mean
    the least important part of the platform could stop the rest of it, so the
    handler is simply not installed and the work goes ahead unrecorded.
    """
    global _installed

    path = log_path(root)
    if _installed == path:
        # A shortcut past closing and reopening the same file. Not what makes
        # this idempotent — the teardown below is, and it is what a repeat call
        # would otherwise rely on — so removing this line changes no answer.
        return path

    log = logging.getLogger(_LOGGER_NAME)
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    _installed = None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError:
        return None

    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(level)
    # Not to the root logger. The CLI writes to a terminal somebody is reading,
    # and duplicating every line there would bury the output they asked for.
    log.propagate = False
    _installed = path
    return path


def logger(name: str) -> logging.Logger:
    """The logger for one module. `name` is appended to `monocell`."""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


@contextmanager
def action(name: str, **fields: Any) -> Iterator[dict]:
    """Record one long action: what started, what ended, how long, what raised.

    Yields a dict the body may add to — a count, an artifact path, whatever the
    action learned while running — and those fields are written on the closing
    line. The closing line is what a reader wants, so it carries everything.

    **It re-raises.** A journal that swallowed an exception would be a journal
    that decided the outcome, and the caller's own error handling is what the
    user sees. The failure is logged on the way past, with the elapsed time,
    because "it failed after 40 minutes" and "it failed at once" are different
    problems.
    """
    log = logger(name)
    extra: dict[str, Any] = {}
    log.info("start %s%s", name, _fields(fields))
    started = time.perf_counter()
    try:
        yield extra
    except BaseException as exc:
        log.error("FAILED %s after %.1fs — %s: %s%s", name,
                  time.perf_counter() - started, type(exc).__name__, exc,
                  _fields({**fields, **extra}))
        raise
    log.info("done %s in %.1fs%s", name, time.perf_counter() - started,
             _fields({**fields, **extra}))


def _fields(fields: dict) -> str:
    """` key=value key=value`, or "". Values are truncated, not wrapped.

    A log line that runs to four hundred characters because somebody passed a
    DataFrame is a line nobody reads, and the field that mattered is at the far
    end of it.
    """
    if not fields:
        return ""
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value)
        parts.append(f"{key}={text[:120] + '…' if len(text) > 120 else text}")
    return (" " + " ".join(parts)) if parts else ""


def tail(root: Path | None = None, n: int = 200) -> list[str]:
    """The last `n` lines, oldest first. `[]` when there is no log yet.

    Reads the whole file rather than seeking backwards. It is capped at 1 MB by
    the rotation above, so the simple version is the honest one — a backwards
    seek is worth writing when the file can be a gigabyte, and this one cannot.
    """
    path = log_path(root)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:] if n > 0 else lines
