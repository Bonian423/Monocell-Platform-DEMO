"""The record of what ran, and the four ways it must not get in the way.

Without one, a re-derive that failed at 02:00 or an ingest that took nine
minutes leaves nothing behind once the process exits: the CLI prints to a
terminal somebody may have closed.

The interesting tests here are not that it writes lines. They are the
constraints: it writes into the STORE (so the log travels with the data it
describes), it writes nothing at import time, it re-raises rather than
swallowing, and a store it cannot write to costs nothing — the least important
part of the platform must not be able to stop the rest of it.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from monocell import journal


@pytest.fixture(autouse=True)
def _clean_handlers():
    """Each test starts with no handler installed and leaves none behind.

    `configure` is module state by design — one process, one log — so tests that
    did not reset it would silently share whichever file ran first.
    """
    yield
    log = logging.getLogger("monocell")
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    journal._installed = None  # noqa: SLF001


# ---------------------------------------------------------------------------
# where it goes
# ---------------------------------------------------------------------------


def test_the_log_lives_in_the_store(root):
    """The one decision worth arguing about. The store is what somebody copies
    to a second machine or attaches to a bug report, and a log in a home
    directory would be the one file that did not travel with it."""
    path = journal.configure(root)
    assert path == root / "logs" / "monocell.log"
    journal.logger("t").info("hello")
    assert "hello" in path.read_text(encoding="utf-8")


def test_changing_the_store_moves_the_log(root, tmp_path):
    """A long-running process can be pointed at a different store while it
    runs, and the log has to follow — otherwise the second store's activity
    lands in the first store's file."""
    second = tmp_path / "other"
    journal.configure(root)
    journal.logger("t").info("first store")
    journal.configure(second)
    journal.logger("t").info("second store")

    a = (root / "logs" / "monocell.log").read_text(encoding="utf-8")
    b = (second / "logs" / "monocell.log").read_text(encoding="utf-8")
    assert "first store" in a and "second store" not in a
    assert "second store" in b and "first store" not in b


def test_there_is_never_more_than_one_handler(root, tmp_path):
    """Every path through `configure` leaves exactly one, and the count is what
    is asserted rather than only the line count.

    Two handlers write every line twice, and the duplication looks exactly like
    the action having run twice — which is the reading that sends somebody
    looking for a bug in the action. Checked after a repeat call with the same
    store AND after a change of store, because those are different branches and
    only one of them tears the old handler down.
    """
    journal.configure(root)
    journal.configure(root)
    assert len(logging.getLogger("monocell").handlers) == 1
    journal.configure(tmp_path / "other")
    assert len(logging.getLogger("monocell").handlers) == 1

    journal.logger("t").info("once")
    text = (tmp_path / "other" / "logs" / "monocell.log").read_text(encoding="utf-8")
    assert text.count("once") == 1


def test_nothing_is_written_until_something_configures_it():
    """A module that installs a handler at import time writes a file the moment
    anything touches the package — including `--help` — and writes it into
    whatever directory the process started in."""
    code = ("import sys, monocell, monocell.journal as j; "
            "print(bool(j._installed), len(__import__('logging')"
            ".getLogger('monocell').handlers))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False 0"


def test_a_store_it_cannot_write_to_costs_nothing(tmp_path):
    """Returns None and installs nothing. Refusing to run because the LOGGING
    failed would let the least important part of the platform stop the rest."""
    obstacle = tmp_path / "logs"
    obstacle.write_text("this is a file, not a directory", encoding="utf-8")
    assert journal.configure(tmp_path) is None
    journal.logger("t").info("this must not raise")


# ---------------------------------------------------------------------------
# what one action leaves behind
# ---------------------------------------------------------------------------


def test_an_action_records_its_start_its_end_and_its_cost(root):
    journal.configure(root)
    with journal.action("rederive", cell="c1"):
        pass
    text = (root / "logs" / "monocell.log").read_text(encoding="utf-8")
    assert "start rederive cell=c1" in text
    assert "done rederive in" in text and "cell=c1" in text


def test_the_body_can_add_what_it_learned(root):
    """The closing line is what a reader wants, so it carries the count the
    action came back with rather than only the arguments it went in with."""
    journal.configure(root)
    with journal.action("rederive", cell="c1") as rec:
        rec["modules"] = 4
    closing = [ln for ln in journal.tail(root) if "done rederive" in ln][0]
    assert "modules=4" in closing


def test_a_failure_is_recorded_with_what_it_cost_and_then_re_raised(root):
    """Both halves. A journal that swallowed the exception would be a journal
    that decided the outcome; one that did not record the elapsed time would
    lose the difference between "failed at once" and "failed after 40 minutes"."""
    import time

    journal.configure(root)
    with pytest.raises(ValueError, match="no parameter file"):
        with journal.action("rederive", cell="c1"):
            time.sleep(0.15)
            raise ValueError("no parameter file for this cell")
    text = (root / "logs" / "monocell.log").read_text(encoding="utf-8")
    assert "FAILED rederive after" in text and "no parameter file" in text
    assert "done rederive" not in text
    # The elapsed figure is the real one. "failed at once" and "failed after 40
    # minutes" are different problems, and a constant would report the first
    # for both.
    assert "after 0.0s" not in text


def test_a_long_field_is_truncated(root):
    """A line that runs to four hundred characters because somebody passed a
    DataFrame is a line nobody reads, and the field that mattered is at the far
    end of it."""
    journal.configure(root)
    with journal.action("thing", payload="x" * 500):
        pass
    for line in journal.tail(root):
        assert len(line) < 300


def test_an_empty_field_is_left_out(root):
    journal.configure(root)
    with journal.action("thing", cell=None, count=0):
        pass
    start = [ln for ln in journal.tail(root) if "start thing" in ln][0]
    assert "cell=" not in start and "count=0" in start


# ---------------------------------------------------------------------------
# reading it back
# ---------------------------------------------------------------------------


def test_the_tail_is_oldest_first_and_bounded(root):
    journal.configure(root)
    for i in range(50):
        journal.logger("t").info("line %d", i)
    rows = journal.tail(root, 10)
    assert len(rows) == 10
    assert "line 40" in rows[0] and "line 49" in rows[-1]


def test_a_store_with_no_log_reads_as_empty_not_as_an_error(root):
    assert journal.tail(root) == []


def test_it_does_not_also_print_to_the_console(root, capsys):
    """The CLI writes to a terminal somebody is reading, and duplicating every
    log line there would bury the output they asked for.

    Asserted on the flag as well as on the output, and the flag is the part
    that bites: under pytest the root logger has no console handler, so a build
    that propagated would print nothing HERE and everything in production.
    """
    journal.configure(root)
    journal.logger("t").warning("something")
    captured = capsys.readouterr()
    assert "something" not in captured.out and "something" not in captured.err
    assert logging.getLogger("monocell").propagate is False


# ---------------------------------------------------------------------------
# through the entry point
# ---------------------------------------------------------------------------


def test_a_command_records_what_was_run_and_how_it_ended(root):
    from monocell.cells import register_cell
    from monocell.cli import main

    register_cell("c1", {"capacity_Ah": 5.0, "chemistry": "NMC"}, root)
    assert main(["simulate", "--cell", "c1", "--data-root", str(root)]) == 1
    text = (root / "logs" / "monocell.log").read_text(encoding="utf-8")
    assert "run simulate --cell c1" in text
    assert "refused: simulate" in text and "no parameter file" in text


def test_a_command_that_succeeds_records_its_status(root):
    from monocell.cli import main

    assert main(["manifest", "rebuild", "--data-root", str(root)]) == 0
    text = (root / "logs" / "monocell.log").read_text(encoding="utf-8")
    assert "exit manifest status 0" in text
