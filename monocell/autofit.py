"""Auto-derive on ingest: derive what the file that just landed feeds.

**This module computes nothing.** It is a scope over `rederive`, and that is
the whole design. `MODULE_GRAPH` already maps every experiment type to the
modules that consume it, `_topo_order` already sorts those modules so a
producer runs before its consumer, and `rederive` already rescans to a fixed
point. A dispatch table here would be a second copy of the graph, free to
disagree with the one `rederive` uses, so there isn't one. Auto-derive IS
re-derive, scoped to what the ingest touched.

Three rules, each of them load-bearing:

1. **Never inside `write_experiment`.** That function is on the path of every
   ingest and every test fixture. A derivation folded into it would give all
   of them a solver dependency, blow up their runtimes, and let an ingest FAIL
   BECAUSE A DERIVATION FAILED, on a store whose contract is append-only and
   whose experiments are already on disk by then. `ingest`'s behaviour is
   unchanged; this is a separate call made after it returns.

2. **A failed derivation leaves the experiment standing.** The measurement is
   written; what failed is a derivation, and a derivation is re-runnable.
   Every failure is caught, attributed to the module it came from, and
   reported — never raised, never allowed to take the ingest down with it.

3. **Always report what it produced.** A silent multi-minute compute is the
   failure mode to avoid, not the compute itself. `derive_for` returns a record
   the CLI prints, and `pending` is recomputed from `list_stale` afterwards
   rather than inferred from the run's own bookkeeping, so "it worked" is
   never the run's word for itself.

The scope is the only thing here that is not `rederive`'s, and it is expressed
as a predicate over stale ROWS (`rederive(accept=…)`) rather than over finished
jobs: a row is the only place where a per-experiment job (one experiment) is
distinguishable from an all-experiments job (the whole id list).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .rederive import MODULE_GRAPH, ModuleRunError, list_stale, rederive


def modules_for(exp_type: str) -> tuple[str, ...]:
    """The modules a type feeds, in `MODULE_GRAPH` order.

    Read off the graph rather than declared: a module that starts consuming a
    type appears here with no edit, and one that stops cannot be left behind
    claiming otherwise. A type nothing consumes (`misc`, for one) yields `()`,
    which is a legitimate answer and not an error.
    """
    return tuple(m for m, spec in MODULE_GRAPH.items() if exp_type in spec["input_experiments"])


def scope_for(wanted_modules: tuple[str, ...],
              experiment_id: str | None) -> Callable[[dict[str, Any]], bool]:
    """The `accept` predicate: which stale rows the ingest just created.

    Two shapes of row, and both branches matter:

    * **per-experiment** — one row per experiment. Keep the row for the
      experiment just written; with `experiment_id=None` keep them all, which
      is what a multi-file batch wants.
    * **all-experiments** — one row per module carrying the whole current id
      list. Keep it whenever the module consumes the ingested type,
      *including* the row whose id list is EMPTY.

    That last case is the one worth stating out loud. An empty id list means
    the module went stale because a consumed ARTIFACT moved, not because new
    data arrived, and that is exactly what happens two rounds into a scoped
    pass: an upstream module re-runs for the new file, which makes its consumer
    stale with no new experiments of its own. A filter that dropped empty rows
    would leave the consumer stale and the engineer re-deriving by hand to do
    the thing auto-derive was asked to do. So the rule is "every
    all-experiments module this type feeds", and the cost is at most a re-run
    of a module that was already stale.
    """
    def accept(row: dict[str, Any]) -> bool:
        if row["module"] not in wanted_modules:
            return False
        if row["experiment_id"] is not None:
            return experiment_id is None or row["experiment_id"] == experiment_id
        return True

    return accept


def derive_for(cell_id: str, exp_type: str, experiment_id: str | None = None,
               root: Path | None = None) -> dict[str, Any]:
    """Derive what the just-ingested experiment is an input to.

    Returns `{"modules", "wanted", "done", "failed", "pending"}`: the modules
    the type feeds, how many scoped jobs were stale, the jobs that ran, the ones
    that raised, and the modules still stale afterwards.

    Never raises for a module that failed. A missing build record or an
    unreadable store still raises — those are the ingest's own problems, and
    `list_stale` is deliberately strict about an unregistered cell id.
    """
    wanted = modules_for(exp_type)
    accept = scope_for(wanted, experiment_id)
    result: dict[str, Any] = {
        "modules": list(wanted), "wanted": 0, "done": [], "failed": [], "pending": [],
    }

    stale = [row for row in list_stale(cell_id, root) if accept(row)]
    result["wanted"] = len(stale)
    if not stale:
        return result

    try:
        result["done"] = rederive(cell_id, root, accept=accept)
    except ModuleRunError as exc:
        # The jobs that finished before the failure are on disk and current —
        # reporting them is the difference between "nothing ran" and "these
        # four ran, this one did not".
        result["done"] = exc.done
        result["failed"].append({"module": exc.module, "error": str(exc)})
    except Exception as exc:  # the convergence guard: a bug, but not the ingest's
        result["failed"].append({"module": "*", "error": f"{type(exc).__name__}: {exc}"})

    # Recomputed, not inferred. A runner that returned without recording its
    # inputs would otherwise be reported as done while `list_stale` says stale.
    still = [row for row in list_stale(cell_id, root) if accept(row)]
    result["pending"] = sorted({row["module"] for row in still})
    return result


def summary_lines(result: dict[str, Any], exp_type: str) -> list[str]:
    """The report, as lines a CLI or a notebook can print.

    Written here rather than in the CLI so two front ends never describe the
    same run differently.
    """
    if not result["modules"]:
        return [f"nothing derives from a {exp_type} yet — it is not an input to any module"]
    if result["wanted"] == 0:
        return [f"nothing to derive for this {exp_type}: "
                f"{', '.join(result['modules'])} already match their inputs"]

    lines = [f"auto-derive: ran {len(result['done'])} job(s) for this {exp_type}"]
    for rec in result["done"]:
        lines.append(f"  {rec['module']} ({', '.join(rec['experiment_ids'])}) -> {rec['artifact']}")
    for rec in result["failed"]:
        lines.append(f"  {rec['module']} FAILED: {rec['error']}")
    if result["pending"]:
        lines.append(f"  still stale: {', '.join(result['pending'])} — "
                     "the experiment is stored; re-run when you are ready "
                     "(monocell rederive)")
    return lines
