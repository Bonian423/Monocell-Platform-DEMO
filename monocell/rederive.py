"""The re-derivation checker: make-style, deterministic, no scheduler.

Every artifact records its `inputs` (ids + content hashes). New data marks
stale any artifact whose inputs have changed, and `monocell rederive --cell X`
lists and re-runs the stale ones. The dependency graph is `MODULE_GRAPH`. In
this copy it holds one module, the abstracted parameter extraction, but the
machinery below is general: per-experiment and all-experiments modules,
modules that consume other modules' artifacts, a topological run order and a
fixed-point loop. `tests/test_rederive_graph.py` drives it with a toy graph.

Append-only experiments can't change their own hashes, so in practice stale
means "an experiment with no current artifact derived from it", or an input
that was retracted. The hash comparison is kept because the general case (an
experiment that feeds several modules, a module that consumes several
experiments) rests on it.

Modules that consume OTHER artifacts declare them in `input_artifacts`; the
checker then also compares those, namespaced `artifact:{module}` so they can
never collide with experiment ids. Without this, a module that runs before its
upstream would consume the upstream's previous artifact, or miss it entirely,
and no later change would ever invalidate it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .cells import load_build
from .schema.artifacts import artifacts_dir, read_artifact_or_none, record_pattern
from .schema.manifest import in_session, list_experiments
from .schema.write_experiment import ExperimentMissing, _experiment_dir, experiment_hash

# A predicate over a `list_stale` ROW — see `_stale_jobs` for why the row, and
# not the finished job, is what a scope has to be expressed against.
Accept = Callable[[dict[str, Any]], bool]


class ModuleRunError(RuntimeError):
    """A module's runner raised. Carries WHICH module, for which inputs, and
    what the pass had already finished.

    Runners raise whatever their libraries raise — a `KeyError` from a pandas
    column, a `LinAlgError` from a fit, a `ValueError` from a solver. Those
    messages name a library's problem, not the platform's, and by the time one
    reaches the CLI or an ingest report the module that was running has been
    lost, which leaves nothing to act on. The original exception is kept as
    `__cause__`, so a traceback still shows it verbatim.

    `done` is the jobs that completed BEFORE this one. A pass stops at the
    first failure (its consumers cannot run against an input that was never
    written), but stopping is not the same as undoing: the artifacts already
    written are on disk and current. Without this, a caller that catches the
    error would report "nothing ran" next to a store where several artifacts
    had just been rewritten.
    """

    def __init__(self, module: str, experiment_ids: list[str], cause: BaseException,
                 done: list[dict[str, Any]] | None = None):
        super().__init__(f"{module} failed on {', '.join(experiment_ids) or '(no ids)'} — "
                         f"{type(cause).__name__}: {cause}")
        self.module = module
        self.experiment_ids = list(experiment_ids)
        self.cause = cause
        self.done = list(done or [])


# module -> (experiment types it consumes, artifact glob, mode)
#
# mode "per_experiment": one artifact per consumed experiment.
# mode "all_experiments": the artifact is a property of the CELL over ALL its
# experiments of the consumed types. Re-derivation re-runs the module once with
# the full current id list, never with a single new id (that would clobber the
# cell-level artifact with a one-run version).
#
# Optional keys: `input_artifacts` (upstream modules whose records this one
# consumes), `artifact_input_globs` (which file of an upstream module, where it
# is not the module's declared record), and `require_all_types` (default True:
# an all-experiments module is derivable only when every consumed type has at
# least one experiment).
MODULE_GRAPH: dict[str, dict[str, Any]] = {
    "parameters": {
        # The parameter file describes the cell over every experiment it has,
        # and any subset of types is derivable: a type with no experiments yet
        # is a gap in the file, not a reason to skip the derivation.
        "input_experiments": ("hppc", "rpt", "cycling", "three_electrode_ts", "eis",
                              "pressure", "gitt", "half_cell_ocp"),
        "artifact_glob": "parameter_file_*.json",
        "mode": "all_experiments",
        "require_all_types": False,
    },
}

# module -> runner(experiment_ids, out_dir, root=...) -> Path of the artifact
# written. Filled on first use, so importing `rederive` does not import the
# engine; a test replaces entries to drive the machinery with a toy graph.
RUNNERS: dict[str, Callable[..., Path]] = {}


def _runner(module: str) -> Callable[..., Path]:
    if not RUNNERS:
        from .engine import extract

        RUNNERS["parameters"] = extract.run
    return RUNNERS[module]


@in_session(1)
def missing_experiments(cell_id: str, root: Path | None = None) -> list[str]:
    """Experiments the manifest lists for this cell that are not on disk.

    Read by the CLI so an inconsistent store is reported once, in words,
    rather than as a traceback from whichever reader happened to touch it first.
    """
    out: list[str] = []
    for rec in list_experiments(cell_id, None, root):
        d = _experiment_dir(cell_id, rec["experiment_id"], root)
        if not (d / "meta.json").exists() or not (d / "series.parquet").exists():
            out.append(rec["experiment_id"])
    return sorted(out)


def _current_inputs_for_types(types: tuple[str, ...], cell_id: str, root: Path | None) -> dict[str, str]:
    """{experiment_id: hash} of everything of the given types.

    An experiment the manifest lists and the disk does not have is SKIPPED. It
    cannot be hashed, so it cannot take part in a staleness comparison, and
    raising here would stop a whole cell re-deriving because one directory was
    removed by hand. `missing_experiments` is what reports it.
    """
    out: dict[str, str] = {}
    for exp_type in types:
        for rec in list_experiments(cell_id, exp_type, root):
            try:
                out[rec["experiment_id"]] = experiment_hash(cell_id, rec["experiment_id"], root)
            except ExperimentMissing:
                continue
    return out


def _current_inputs(module: str, cell_id: str, root: Path | None) -> dict[str, str]:
    """{experiment_id: hash} of everything this module currently consumes."""
    return _current_inputs_for_types(MODULE_GRAPH[module]["input_experiments"], cell_id, root)


def _newest_first(paths: list[Path]) -> list[Path]:
    """Newest artifact first: the platform's one ordering rule, reversed.

    The tie-break rationale lives with the rule itself in `schema.artifacts`.
    Reversing a shared total order gives the same answer as writing a second
    one, and it cannot drift from it.
    """
    from .schema.artifacts import artifact_order_key

    return sorted(paths, key=artifact_order_key, reverse=True)


def _derived_artifacts(module: str, cell_id: str, root: Path | None) -> dict[str, Path]:
    """{experiment_id: artifact path}: the experiments each artifact consumed.

    When several artifacts consume the same experiment (versioned artifacts
    like the parameter file keep their history), the NEWEST one wins: the
    current version, not a stale one the filesystem glob happens to surface
    first.
    """
    derived: dict[str, Path] = {}
    art_dir = artifacts_dir(cell_id, root) / module
    if not art_dir.exists():
        return derived
    for art_path in _newest_first(list(art_dir.glob(MODULE_GRAPH[module]["artifact_glob"]))):
        env = read_artifact_or_none(art_path)
        if env is None:
            # A file that will not parse cannot say what it was derived from,
            # so it cannot answer "is this current?". Treating it as absent
            # makes its experiments read as underived, which is the honest
            # answer and is also the one that fixes it: the next re-derive
            # overwrites it.
            continue
        for rec in env["inputs"]:
            derived.setdefault(rec["id"], art_path)  # first (newest) wins
    return derived


def _newest_inputs(module: str, cell_id: str, root: Path | None) -> set[str]:
    """The input ids of this module's NEWEST artifact, and only that one.

    Not `_derived_artifacts`, which maps every id any artifact ever cited to
    the newest artifact citing it. That map is the right one for "has this
    experiment been derived", and the wrong one here: a retracted experiment
    goes on appearing in it forever through the historical artifact that
    consumed it, so a module re-derived without it would read as still stale
    and `rederive`'s fixed-point guard would refuse the store.
    """
    art_dir = artifacts_dir(cell_id, root) / module
    if not art_dir.exists():
        return set()
    newest = _newest_first(list(art_dir.glob(MODULE_GRAPH[module]["artifact_glob"])))
    for path in newest:
        env = read_artifact_or_none(path)
        if env is not None:
            return {rec["id"] for rec in env["inputs"]}
    return set()


def _dropped_inputs(types: tuple[str, ...], cell_id: str, root: Path | None,
                    cited, current: dict[str, str]) -> list[str]:
    """Experiments the newest artifact cites that this module no longer consumes.

    Two conditions decide it, and one is a shortcut:

    * **still known to the manifest, under this cell.** That is what separates
      "retracted" from "was never here": a typo'd id in an old artifact is not
      a reason to hold a module stale forever.
    * **of a type this module declares.** A record may cite an experiment it
      reached through another module's artifact, and staleness on that chain is
      already the artifact edge's job.

    The `artifact:` check is neither of those: an `artifact:`-shaped id is not
    an experiment, so the manifest lookup rejects it anyway. It is there to
    skip the query, and removing it changes no answer.
    """
    from .schema.manifest import find_experiment

    out: list[str] = []
    for eid in cited:
        if eid.startswith("artifact:") or eid in current:
            continue
        rec = find_experiment(eid, root)
        if rec and rec.get("cell_id") == cell_id and rec.get("exp_type") in types:
            out.append(eid)
    return sorted(out)


def _stale_id(exp_id: str, current: dict[str, str], derived: dict[str, Path]) -> bool:
    """True when `exp_id` has no current artifact, or its recorded hash is stale."""
    if exp_id not in derived:
        return True
    env = read_artifact_or_none(derived[exp_id])
    if env is None:
        return True
    stored = {rec["id"]: rec["hash"] for rec in env["inputs"]}
    return stored.get(exp_id) != current[exp_id]


def _artifact_inputs(module: str, cell_id: str, root: Path | None) -> dict[str, str]:
    """{artifact:{upstream}: hash} of the newest consumed artifacts.

    The per-upstream glob is the module graph's own override where it has one
    and `schema.artifacts.record_pattern` otherwise. A module's directory can
    hold more than its record file, so a bare `*.json` glob here would compare
    against the wrong file and make the consumer stale forever. The override is
    PARTIAL by design, which is why the default has to come from the same
    declaration the modules themselves read.
    """
    from .schema.artifacts import artifact_input_env

    declared = MODULE_GRAPH[module].get("input_artifacts", ())
    globs = MODULE_GRAPH[module].get("artifact_input_globs") or {}
    declarations = []
    for upstream in declared:
        pattern = globs.get(upstream) or record_pattern(upstream)
        if pattern is None:
            # Refused, not skipped. An unresolvable declaration means the
            # graph names an input the store cannot locate, and SKIPPING it
            # drops that input from the hash — so a change to it would stop
            # marking this module stale and the store would quietly serve a
            # result derived from a file that has since moved. That is the one
            # failure this whole checkpoint exists to catch, so it is raised
            # here where the name is still in hand rather than discovered
            # later as a wrong number.
            raise ValueError(
                f"rederive: {module} declares {upstream!r} as an artifact input, but no glob "
                f"is declared for it — neither in MODULE_GRAPH[{module!r}]"
                f"['artifact_input_globs'] nor in schema.artifacts.MODULE_RECORDS. An input that "
                f"cannot be located cannot be hashed, and an unhashed input is an input whose "
                f"changes never mark anything stale."
            )
        declarations.append((upstream, pattern))
    return {rec["id"]: rec["hash"]
            for rec in artifact_input_env(cell_id, root, declarations)}


def _stale_artifact_inputs(module: str, cell_id: str, root: Path | None) -> list[str]:
    """Artifact-input ids the module's own newest artifact fails to record at
    the current hash: the upstream-appears-later case, where nothing about the
    module's experiments changed, yet its newest artifact predates an upstream
    artifact it consumes. Newest-wins by the envelope's created_at, like every
    other picker in the store.

    Artifacts that record no `artifact:` inputs at all (a derived record that
    cites something else) are skipped: they were never the artifact-input
    record, and they must not make the module look stale.
    """
    current = _artifact_inputs(module, cell_id, root)
    if not current:
        return []
    art_dir = artifacts_dir(cell_id, root) / module
    if not art_dir.exists():
        return sorted(current)
    for art_path in _newest_first(list(art_dir.glob(MODULE_GRAPH[module]["artifact_glob"]))):
        env = read_artifact_or_none(art_path)
        if env is None:
            continue
        stored = {rec["id"]: rec["hash"] for rec in env["inputs"]}
        if not any(aid.startswith("artifact:") for aid in stored):
            continue  # not the artifact-input record — walk to the newest one that is
        return sorted(aid for aid, h in current.items() if stored.get(aid) != h)
    return sorted(current)


def _stale_artifact_inputs_per_experiment(module: str, cell_id: str, root: Path | None,
                                          derived: dict[str, Path]) -> list[dict[str, Any]]:
    """Per-experiment artifact-input staleness, for modules with `input_artifacts`.

    A `per_experiment` module's artifact is per experiment, so its upstream
    comparison must be per experiment too: each artifact is checked against the
    `artifact:{upstream}` hashes IT recorded. A module-level check (the one
    `all_experiments` modules use) cannot express this: for a module with one
    artifact per experiment it would pick an arbitrary one of them.

    Only artifacts that RECORDED an `artifact:` input are checked. An artifact
    that never read its upstream must not be invalidated by it changing:
    staleness follows what the artifact declared it consumed, which is the same
    rule the experiment-hash half uses.

    The entries are emitted in the existing per-experiment shape
    (`experiment_id` set). A module-level entry would route through
    `rederive`'s job builder into a multi-experiment call, and a
    `per_experiment` runner takes exactly one id.

    An upstream artifact that DISAPPEARED is not visible from here when the
    module has no artifact yet at all, but then the experiment-hash half
    already reports it, so there is nothing extra to say.
    """
    declared = MODULE_GRAPH[module].get("input_artifacts", ())
    if not declared:
        return []
    upstream = _artifact_inputs(module, cell_id, root)  # {artifact:{m}: hash}
    out: list[dict[str, Any]] = []
    for exp_id in sorted(derived):
        env = read_artifact_or_none(derived[exp_id])
        if env is None:
            continue
        stored = {rec["id"]: rec["hash"] for rec in env["inputs"]}
        recorded = {aid: h for aid, h in stored.items() if aid.startswith("artifact:")}
        if not recorded:
            continue
        bad = sorted(aid for aid, h in recorded.items() if upstream.get(aid) != h)
        if bad:
            reason = "consumed artifact changed: " + ", ".join(bad)
            out.append({"module": module, "experiment_id": exp_id, "artifact": None, "reason": reason})
    return out


def _topo_order(modules: list[str]) -> list[str]:
    """Kahn over the `input_artifacts` edges, alphabetical within the ready set.

    `sorted(jobs)` is not a valid run order: alphabetical order can put a
    consumer before the module it consumes, so the consumer would read its
    upstream's PREVIOUS artifact in the same pass and never be corrected.

    Alphabetical-within-ready keeps the order deterministic and identical to
    `sorted()` wherever no declared edge inverts it. A declared cycle raises:
    silently running a cycle in glob order would reintroduce exactly the bug
    this replaces.
    """
    mods = sorted(set(modules))
    deps = {m: {u for u in MODULE_GRAPH.get(m, {}).get("input_artifacts", ()) if u in mods}
            for m in mods}
    remaining = {m: set(d) for m, d in deps.items()}
    order: list[str] = []
    while True:
        ready = sorted(m for m in mods if m not in order and not remaining[m])
        if not ready:
            break
        for m in ready:
            order.append(m)
            for n in mods:
                remaining[n].discard(m)
    if len(order) != len(mods):
        raise ValueError(
            "cycle in MODULE_GRAPH input_artifacts among "
            f"{sorted(set(mods) - set(order))} — the dependency graph must be a DAG"
        )
    return order


@in_session(1)
def list_stale(cell_id: str, root: Path | None = None) -> list[dict[str, Any]]:
    """Stale modules: per-experiment modules report (module, experiment_id)
    pairs; all-experiments modules report ONE entry carrying `experiment_ids`.

    An all-experiments module is DERIVABLE only once every consumed type has
    at least one experiment, unless it declares `require_all_types=False`:
    a module that needs two types must not go stale on one of them with
    nothing to derive. Raises FileNotFoundError for an unregistered cell: a
    typo'd cell id must not masquerade as "nothing stale".
    """
    load_build(cell_id, root)
    stale: list[dict[str, Any]] = []
    for module, spec in MODULE_GRAPH.items():
        current = _current_inputs(module, cell_id, root)
        derived = _derived_artifacts(module, cell_id, root)
        if spec.get("mode") == "all_experiments":
            per_type = {t: _current_inputs_for_types((t,), cell_id, root) for t in spec["input_experiments"]}
            if spec.get("require_all_types", True):
                if not all(per_type[t] for t in spec["input_experiments"]):
                    continue  # a required type has no experiments -> nothing to derive
            elif not any(per_type[t] for t in spec["input_experiments"]):
                continue  # any subset is derivable; nothing at all is not
            # The CURRENT artifact's inputs, not every id any artifact ever
            # cited. An all-experiments module writes one file over the whole
            # set, so "has this been derived" means "is it in that file", and
            # the historical union answers yes for an experiment the newest
            # derivation left out, which is what an experiment RESTORED after
            # being withdrawn looks like.
            cited = _newest_inputs(module, cell_id, root)
            now = {eid: p for eid, p in derived.items() if eid in cited}
            bad = sorted(eid for eid in current if _stale_id(eid, current, now))
            # ...and the other direction: an input the artifact CITES which is
            # no longer current. Comparing only forwards would let a pooled
            # result that still rests on a withdrawn experiment read as fresh,
            # because the check walks the live experiments and a withdrawn one
            # is not among them. A hand-deleted directory lands here too.
            gone = _dropped_inputs(spec["input_experiments"], cell_id, root, cited, current)
            bad_artifacts: list[str] = []
            if spec.get("input_artifacts"):
                bad_artifacts = _stale_artifact_inputs(module, cell_id, root)
            if gone:
                bad = sorted(set(bad) | set(gone))
            if bad or bad_artifacts:
                if gone:
                    reason = (f"{len(gone)} consumed experiment(s) are no longer current "
                              f"(retracted or removed): {', '.join(gone)}")
                    if bad_artifacts:
                        reason = f"{reason}; {len(bad_artifacts)} consumed artifact(s) changed"
                    stale.append({"module": module, "experiment_id": None,
                                  "experiment_ids": bad, "artifact": None, "reason": reason})
                    continue
                if bad_artifacts:
                    reason = (f"{len(bad_artifacts)} consumed artifact(s) changed or missing "
                              f"({', '.join(bad_artifacts)})")
                    if bad:
                        reason = (f"{len(bad)} experiment(s) without a current {module} artifact; "
                                  + reason)
                else:
                    reason = f"{len(bad)} experiment(s) without a current {module} artifact"
                stale.append(
                    {
                        "module": module,
                        "experiment_id": None,
                        "experiment_ids": bad,
                        "artifact": None,
                        "reason": reason,
                    }
                )
            continue
        art_stale = {e["experiment_id"]: e["reason"] for e in
                     _stale_artifact_inputs_per_experiment(module, cell_id, root, derived)}
        for exp_id in sorted(current):
            hash_bad = _stale_id(exp_id, current, derived)
            if not (hash_bad or exp_id in art_stale):
                continue
            if hash_bad:
                reason = ("no artifact derived from this experiment"
                          if exp_id not in derived else "input hash changed")
                if exp_id in art_stale:
                    reason = f"{reason}; {art_stale[exp_id]}"
            else:
                reason = art_stale[exp_id]
            stale.append(
                {
                    "module": module,
                    "experiment_id": exp_id,
                    "artifact": None,
                    "reason": reason,
                }
            )
    return stale


def _stale_jobs(cell_id: str, root: Path | None, accept: Accept | None = None) -> list[tuple[str, list[str]]]:
    """The jobs one pass must run: the current stale set, ordered by the graph.

    per_experiment modules yield one job per stale experiment; all_experiments
    modules yield ONE job carrying the full current id list (deduped per
    module).

    `accept` narrows the stale ROWS before they become jobs, and it is applied
    on every rescan of the fixed-point loop rather than to a frozen list. That
    is what makes a scoped re-derivation (`autofit`) inherit the ordering, the
    aggregation and the convergence guard instead of reimplementing them: the
    filter can only ever remove jobs, so the relative order of what is left is
    still topological, and a scoped run that fails to converge is still a bug
    that raises.

    Filtering happens here rather than in `list_stale` because a stale row is
    the only place the two job shapes are distinguishable: a per-experiment row
    names ONE experiment, an all-experiments row carries the whole id list,
    and an all-experiments row with an EMPTY id list is a re-run caused by a
    consumed artifact moving, which is not something an ingest caused.
    """
    jobs: list[tuple[str, list[str]]] = []
    for s in list_stale(cell_id, root):
        if accept is not None and not accept(s):
            continue
        if s["experiment_id"] is not None:
            jobs.append((s["module"], [s["experiment_id"]]))
        else:
            current = _current_inputs(s["module"], cell_id, root)
            jobs.append((s["module"], sorted(current)))
    # dedupe, then order by the dependency graph
    seen: set[tuple[str, tuple[str, ...]]] = set()
    uniq: list[tuple[str, list[str]]] = []
    for module, ids in sorted(jobs):
        key = (module, tuple(ids))
        if key not in seen:
            seen.add(key)
            uniq.append((module, ids))
    rank = {m: i for i, m in enumerate(_topo_order([m for m, _ in uniq]))}
    uniq.sort(key=lambda job: (rank[job[0]], job[0]))

    # one runner call must never be handed more than one id unless the module
    # is an all_experiments module — the failure mode is a ValueError deep
    # inside a runner, which reads as a data problem rather than a graph one
    for module, ids in uniq:
        if len(ids) > 1 and MODULE_GRAPH[module].get("mode") != "all_experiments":
            raise ValueError(
                f"{module} is a per_experiment module but was handed {len(ids)} ids "
                f"({', '.join(ids)}); the staleness graph produced a module-level job"
            )
    return uniq


def rederive(cell_id: str, root: Path | None = None, dry_run: bool = False,
             accept: Accept | None = None) -> list[dict[str, Any]]:
    """Re-run every stale module, to a fixed point. Returns what was (re)derived.

    `_topo_order` orders the jobs found in ONE scan, and that is not enough on
    its own: a module becomes stale as a CONSEQUENCE of a job in the same pass.
    An upstream artifact rewritten in this pass makes its consumer's recorded
    `artifact:` hash wrong, but only once the upstream has actually run, which
    is after the stale set was computed. So a single pass would leave the
    consumer stale and the engineer would have to run it twice.

    Hence the loop: rescan, reorder, run, repeat until nothing is stale. Each
    round is a superset of the previous round's *order* guarantee, so a module
    still runs after everything it consumes.

    Two guards, because a fixed point that is not reached is a bug and must not
    look like success:
      * the same (module, ids) job must not come back — a runner that does not
        record what it consumed would otherwise be re-run for ever;
      * rounds are bounded by the module count.

    `accept` scopes the pass to a subset of the stale rows (see `_stale_jobs`).
    A scoped pass converges on the SAME fixed point as an unscoped one for the
    modules it accepts, and leaves everything else stale, which is the honest
    outcome: `list_stale` still reports what was left, so a partial pass cannot
    read as a complete one.
    """
    done: list[dict[str, Any]] = []
    ran: set[tuple[str, tuple[str, ...]]] = set()
    for _ in range(len(MODULE_GRAPH) + 1):
        uniq = _stale_jobs(cell_id, root, accept)
        if not uniq:
            break
        for module, ids in uniq:
            key = (module, tuple(ids))
            if key in ran:
                raise ValueError(
                    f"{module} is stale again for the same inputs ({', '.join(ids)}) immediately "
                    "after being re-derived — its runner is not recording what it consumed, so "
                    "re-deriving cannot converge"
                )
            ran.add(key)
            if dry_run:
                done.append({"module": module, "experiment_ids": ids, "dry_run": True})
                continue
            out_dir = artifacts_dir(cell_id, root) / module
            try:
                path = _runner(module)(ids, out_dir, root=root)
            except Exception as exc:
                raise ModuleRunError(module, ids, exc, done) from exc
            done.append({"module": module, "experiment_ids": ids, "artifact": str(path)})
        if dry_run:
            break  # a dry run changes nothing, so a second round would repeat it
    else:
        raise ValueError(
            f"re-deriving {cell_id} did not converge in {len(MODULE_GRAPH) + 1} rounds — the "
            "modules are still stale after every one of them ran"
        )
    return done
