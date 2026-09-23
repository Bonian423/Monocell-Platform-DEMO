"""The monocell CLI: register cells, ingest instrument files, query the store,
re-derive what is stale, and run the result through PyBaMM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _registered(cell_id: str, root: Path | None) -> bool:
    """True when the cell has a build record; otherwise say so in one line.

    A typo'd cell id is the most common way to reach a command with nothing to
    work on, and the reader's own error would be a path to a missing JSON file.
    """
    from .cells import build_path

    if build_path(cell_id, root).exists():
        return True
    print(f"cell {cell_id} is not registered (`monocell cell list` shows the ones that are)")
    return False


def _cmd_rederive(args) -> int:
    from .rederive import list_stale, rederive

    root = Path(args.data_root) if args.data_root else None
    if not _registered(args.cell, root):
        return 1
    stale = list_stale(args.cell, root)
    if not stale:
        print(f"cell {args.cell}: nothing stale — all artifacts are up to date")
        return 0
    print(f"cell {args.cell}: {len(stale)} stale artifact(s)")
    for s in stale:
        ids = s.get("experiment_id") or ", ".join(s.get("experiment_ids", []))
        print(f"  {s['module']} — {s['reason']} (experiment {ids})")
    if args.dry_run:
        print("dry run — not re-deriving")
        return 0
    done = rederive(args.cell, root)
    for d in done:
        print(f"  re-derived {d['module']} -> {d['artifact']}")
    return 0


def _cmd_ingest(args) -> int:
    """Ingest an instrument file through the one writer.

    `--derive` then runs the derivations this experiment is an input to. It is
    a SEPARATE call after the writer has returned, never folded into it: the
    measurement is on disk before any derivation starts, so one that fails
    cannot cost the ingest its data. The exit status stays 0 for a failed
    derivation, because what failed is re-runnable, and the report says so.
    """
    from .schema.ingest import ingest

    root = Path(args.data_root) if args.data_root else None
    if not _registered(args.cell, root):
        return 1
    if args.dir:
        if not Path(args.dir).is_dir():
            print(f"{args.dir} is not a directory")
            return 1
        return _cmd_ingest_folder(args, root)
    if not args.type:
        # `--dir` can read the type off each file's shape; one named file has
        # only the name, and naming the type is the thing `--type` is for.
        print("--type is required with --file (with --dir the shape proposes it). "
              "`monocell inspect --file ...` will tell you what this one looks like.")
        return 2
    path = Path(args.file)
    if args.instrument:
        _warn_if_unverified(args.instrument, root)
    else:
        _propose_instrument(path, root)
    d = ingest(args.cell, args.type, path, Path(args.sidecar) if args.sidecar else None,
               root, instrument=args.instrument)
    print(f"ingested {args.type} -> {d}")
    _print_ingest_flags(d)

    if args.derive:
        from .autofit import derive_for, summary_lines

        result = derive_for(args.cell, args.type, experiment_id=Path(d).name, root=root)
        for line in summary_lines(result, args.type):
            print(line)
    return 0


def _plan_folder(args, root) -> list[dict]:
    """One row per candidate file: what would be written, or why it would not be.

    The rules are the discipline `inspect` already has, applied to a decision:

      * a file whose type could not be proposed is SKIPPED, not guessed at
      * a file with more than one proposal is SKIPPED TOO, because choosing
        between `gitt` and `hppc` on the engineer's behalf is exactly the thing
        `propose_type` refuses to do, and doing it here would only move the
        refusal somewhere less visible
      * `--type` settles both cases at once, for the ordinary case of a
        directory that is forty of the same thing

    A skipped row is still a row. The point of a folder scan is to come back
    with an account of the whole directory, not to stop at the photo in it.
    """
    from .schema.shapes import probe_file

    d = Path(args.dir)
    paths = [p for p in sorted(d.iterdir())
             if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls", ".txt")]
    capacity = _cell_capacity_for(args.cell, root)

    rows: list[dict] = []
    for path in paths:
        probe = probe_file(path, root, capacity)
        row = {"path": path, "probe": probe, "exp_type": args.type,
               "instrument": args.instrument or probe["profile"],
               "sidecar": _sidecar_beside(path), "skip": None}
        if probe["error"]:
            row["skip"] = probe["error"]
        elif not args.type:
            types = probe["types"]
            if not types:
                row["skip"] = ("nothing here looks like a type this platform knows — pass "
                               "--type to say what it is, or ingest it as misc")
            elif len(types) > 1:
                row["skip"] = ("its shape fits " + " and ".join(t.exp_type for t in types)
                               + " equally well, and choosing for you is the one thing "
                                 "detection must not do — pass --type")
            else:
                row["exp_type"] = types[0].exp_type
        rows.append(row)
    return rows


def _sidecar_beside(path: Path) -> Path | None:
    """`foo.csv` -> `foo.sidecar.json`, when it is there.

    One `--sidecar` for a whole directory is the wrong shape for a campaign.
    The directory form exists for exactly that case (forty files off a rig),
    and in a campaign the per-file half of the sidecar is what differs:
    `rpt_index`, `age_efc`, the chamber temperature. Applying one file's
    `age_efc` to all six RPTs in a fade series does not fail; it writes six
    experiments that claim to be the same age, and the fade curve built on them
    is wrong with nothing to show for it.

    So the convention is read from the directory. `--sidecar` still wins where
    it is given, because "this whole folder came off one run" is a real case
    too, and the explicit flag is the one that should be able to say it.
    """
    # `path.stem`, not two `with_suffix` calls: `with_suffix` replaces the LAST
    # suffix, so `run.2026.csv` would look for `run.sidecar.json`.
    beside = path.parent / (path.stem + ".sidecar.json")
    return beside if beside.exists() else None


def _cell_capacity_for(cell_id, root) -> float | None:
    from .cells import load_build

    try:
        return float(load_build(cell_id, root)["capacity_Ah"])
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


def _cmd_ingest_folder(args, root) -> int:
    """Ingest a directory. Prints the plan first, and writes only on `--confirm`.

    The single-file `ingest` writes immediately and this does not, which is a
    deliberate inconsistency: the store is append-only, so a wrong ingest is
    corrected by writing a second experiment that supersedes it, and the cost of
    that scales with the number of files. One mistake is a correction; forty is
    an afternoon. So the plan is printed, and the confirmation is a word.
    """
    from .schema.ingest import ingest

    rows = _plan_folder(args, root)
    if not rows:
        print(f"no candidate files in {args.dir} (looking for .csv / .xlsx / .xls / .txt)")
        return 0

    doable = [r for r in rows if not r["skip"]]
    print(f"plan for {len(rows)} file(s) in {args.dir}, into cell {args.cell}:")
    for row in rows:
        name = row["path"].name
        if row["skip"]:
            print(f"  SKIP  {name}\n          {row['skip']}")
            continue
        via = f" via {row['instrument']}" if row["instrument"] else " via the synonym reader"
        dropped = (row["probe"]["report"] or {}).get("unplaced") or []
        note = f", {len(dropped)} column(s) not read" if dropped else ""
        if args.sidecar:
            note += ", --sidecar for every file"
        elif row["sidecar"]:
            note += f", sidecar {row['sidecar'].name}"
        print(f"  {row['exp_type']:<10} {name}{via} — {row['probe']['rows']} rows{note}")

    if not doable:
        print("\nnothing to ingest.")
        return 0
    if not args.confirm:
        print(f"\nnothing was written. Add --confirm to ingest the {len(doable)} file(s) above.")
        return 0

    print()
    override = Path(args.sidecar) if args.sidecar else None
    written = 0
    for row in doable:
        try:
            d = ingest(args.cell, row["exp_type"], row["path"],
                       override or row["sidecar"], root,
                       instrument=row["instrument"])
        except Exception as exc:  # noqa: BLE001 — one bad file must not cost the other 39
            print(f"  FAILED {row['path'].name}: {type(exc).__name__}: {exc}")
            continue
        written += 1
        print(f"  ingested {row['exp_type']} -> {d}")
        _print_ingest_flags(d)
    print(f"\n{written} of {len(doable)} file(s) ingested into {args.cell}.")
    return 0


def _propose_instrument(path: Path, root: Path | None) -> None:
    """Name the profiles whose signature this file matches. PROPOSES, never acts.

    Detection that chose for you would be one more way to get a wrong number
    with a confident face, which is why `detect` returns a ranking rather than
    a profile. So this prints, the ingest goes ahead by the built-in synonym
    reader, and the engineer decides whether to re-run with `--instrument`.
    """
    from .schema.instruments import detect, load_profiles

    profiles = load_profiles(root)
    hits = detect(path, profiles)
    if not hits:
        return
    # Marked here, not only in `inspect`. `verified` means somebody held the
    # profile against a real export from that rig; a profile written from
    # published column names has never met a file, so a header spelled
    # differently will not match. Naming the profile without saying which kind
    # it is invites the reader to assume the first.
    names = ", ".join(n + ("" if profiles[n].verified else " (UNVERIFIED)")
                      for n, _ in hits)
    print(f"note: this file's headers match instrument profile(s): {names}")
    print(f"      reading by the built-in synonym list instead. To use one: "
          f"--instrument {hits[0][0]}")


def _warn_if_unverified(name: str, root: Path | None) -> None:
    """Say so when the engineer NAMES a profile nobody has checked against a file.

    The higher-risk path of the two. `_propose_instrument` prints a note the
    reader may ignore because the ingest goes ahead by the synonym list either
    way; `--instrument` means this profile decides how every column is read, and
    a mapping written from published header names can be wrong in a way that
    produces numbers rather than an error.
    """
    from .schema.instruments import load_profiles

    profile = load_profiles(root).get(name)
    if profile is None or profile.verified:
        return
    print(f"note: instrument profile {name!r} is UNVERIFIED — {profile.evidence}")
    print("      check the first file's columns before trusting a campaign to it: "
          f"`monocell inspect --file <one file> --instrument {name}`")


def _print_ingest_flags(experiment_dir) -> None:
    """The stored quality flags, on stdout, right after the write.

    A flag that only exists in `quality.json` is a flag nobody reads. These are
    the parse's own account of what it could not place, and the moment they are
    worth seeing is the moment the file lands.
    """
    import json

    q = Path(experiment_dir) / "quality.json"
    if not q.exists():
        return
    for flag in json.loads(q.read_text(encoding="utf-8")).get("flags", []):
        print(f"  ! {flag}")


def _cmd_find(args) -> int:
    """Ask the manifest a question somebody typed.

    Every other read of the index is a program's read ("what does this cell
    have"). This is the person's: "where is that run?"

    Exit 0 on no matches. An empty result is an answer.
    """
    from .schema.manifest import search_experiments

    root = Path(args.data_root) if args.data_root else None
    flagged = True if args.flagged else (False if args.clean else None)
    rows = search_experiments(
        root, cells=args.cell or None, exp_types=args.type or None, lot=args.lot,
        producer=args.producer, since=args.since, until=args.until,
        flagged=flagged, limit=args.limit)

    if not rows:
        print("nothing matches.")
        return 0

    width = max(len(r["cell_id"]) for r in rows)
    type_width = max(len(r["exp_type"]) for r in rows)
    for r in rows:
        flags = r["flags"]
        mark = f"  {len(flags)} flag(s)" if flags else ""
        print(f"{r['created_at'][:19]}  {r['cell_id']:<{width}}  {r['exp_type']:<{type_width}}  "
              f"{r['experiment_id']}{mark}")
        if flags and args.show_flags:
            for f in flags:
                print(f"    ! {f}")
    print(f"\n{len(rows)} experiment(s).")
    return 0


def _cmd_promote(args) -> int:
    """Re-read a stored experiment as a real type: the way out of `misc`.

    Adds an experiment, never edits one. The original stays exactly where it is
    and the new one records `derived_from`, which is the schema's correction
    mechanism: everything derived from the original was derived from what it
    USED to say, so rewriting it in place is the one thing a store like this
    must not do.
    """
    from .schema.ingest import promote

    root = Path(args.data_root) if args.data_root else None
    d = promote(args.cell, args.experiment, args.type, instrument=args.instrument, root=root)
    print(f"promoted {args.experiment} -> {args.type} -> {d}")
    print(f"  the original is untouched; the new experiment records "
          f"derived_from = {args.experiment}")
    _print_ingest_flags(d)
    return 0


def _cmd_inspect(args) -> int:
    """Say what a file (or a directory of them) looks like, and write nothing.

    The dry run for `ingest`. Every line it prints is an observation or a
    proposal, and the last line of each entry is the command that would act on
    the proposal, so the engineer reads the evidence and then runs the thing,
    rather than the platform doing both and reporting afterwards.

    Exit status is 0 even when nothing could be read. `inspect` answers a
    question; not recognising a file is one of the answers, and a directory
    scan must not stop on the photo somebody left in it.
    """
    from .schema.shapes import probe_file

    root = Path(args.data_root) if args.data_root else None
    capacity = None
    if args.cell:
        from .cells import load_build

        try:
            capacity = float(load_build(args.cell, root)["capacity_Ah"])
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            print(f"note: cell {args.cell!r} has no readable build record, so no C-rate "
                  "can be computed and the type proposals lose their rate evidence")

    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"{d} is not a directory")
            return 1
        paths = [p for p in sorted(d.iterdir())
                 if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls", ".txt")]
        if not paths:
            print(f"no candidate files in {d} (looking for .csv / .xlsx / .xls / .txt)")
            return 0
    else:
        one = Path(args.file)
        if one.is_dir():
            print(f"{one} is a directory. Use --dir to scan the files in it:")
            print(f"  monocell inspect --dir {one}")
            return 1
        if not one.exists():
            print(f"{one} does not exist")
            return 1
        paths = [one]

    for path in paths:
        _print_probe(probe_file(path, root, capacity), args)
    if not args.cell:
        print("\n(no --cell given, so no C-rate was computed; pass one and the "
              "pulse/leg rules gain their strongest evidence)")
    return 0


def _print_probe(probe: dict, args) -> None:
    """One file's entry: what was read, what it looks like, what to run."""
    path = probe["path"]
    print(f"\n{path.name}")
    if probe["error"]:
        print(f"  unreadable   {probe['error']}")
        print("  it can still be ingested as `misc`, which stores it as it arrived "
              "and unlocks nothing")
        return

    if probe["profile"]:
        report = probe["report"] or {}
        verified = "verified" if report.get("verified") else "UNVERIFIED"
        print(f"  profile      {probe['profile']} ({verified})")
    else:
        print("  profile      none matched — read by the built-in synonym list, which drops "
              "what it cannot name")
    if len(probe["profiles"]) > 1:
        print(f"               (also matched: {', '.join(probe['profiles'][1:])})")
    print(f"  rows         {probe['rows']}")

    if not probe["types"]:
        print("  type         nothing here looks like a type this platform knows. `misc` "
              "stores it and unlocks nothing.")
    for i, proposal in enumerate(probe["types"]):
        label = "type" if i == 0 else "or"
        mark = " (decisive)" if proposal.decisive else ""
        print(f"  {label:<12} {proposal.exp_type}{mark} — {proposal.evidence}")

    report = probe["report"] or {}
    for key, caption in (("unplaced", "not read"), ("empty", "empty"),
                         ("missing", "absent"), ("unmapped_segments", "steps")):
        if report.get(key):
            print(f"  {caption:<12} {', '.join(str(x) for x in report[key])}")

    if probe["types"]:
        cell = args.cell or "<cell>"
        profile = f" --instrument {probe['profile']}" if probe["profile"] else ""
        rootarg = f" --data-root {args.data_root}" if args.data_root else ""
        print(f"  ingest with  monocell ingest --cell {cell} "
              f"--type {probe['types'][0].exp_type} --file \"{path}\"{profile}{rootarg}")


def _cmd_cell(args) -> int:
    """Register a cell, or list what is registered.

    A build record is a JSON file, so `--build` takes that file and
    `register_cell` validates it; the record can be version-controlled as it
    is. Two fields are required, `capacity_Ah` and `chemistry`; the rest are
    optional.
    """
    import json

    from .cells import BUILD_FIELDS, list_cells, load_build, register_cell

    root = Path(args.data_root) if args.data_root else None

    if args.action == "list":
        cells = list_cells(root)
        if not cells:
            print("no cells registered in this store")
            return 0
        for cell in cells:
            build = load_build(cell, root)
            print(f"{cell:24} {build.get('capacity_Ah')} Ah  {build.get('chemistry')}")
        return 0

    if args.action == "fields":
        for name, (description, required) in BUILD_FIELDS.items():
            mark = "required" if required else "optional"
            print(f"  {name:30} {mark:9} {description}")
        return 0

    if not args.cell:
        print("give --cell with `register`")
        return 1
    if args.build:
        build = json.loads(Path(args.build).read_text(encoding="utf-8"))
    elif args.capacity_Ah and args.chemistry:
        build = {"capacity_Ah": args.capacity_Ah, "chemistry": args.chemistry}
    else:
        print("give --build <file.json>, or both --capacity-Ah and --chemistry for the "
              "two required fields. `monocell cell fields` lists every field.")
        return 1

    path = register_cell(args.cell, build, root, update=args.update)
    print(f"registered {args.cell} -> {path}")
    absent = [f for f in BUILD_FIELDS if f not in build]
    if absent:
        print(f"  {len(absent)} optional build field(s) not given: {', '.join(sorted(absent))}")
        print("  add them with --build <file.json> --update")
    return 0


def _cmd_retract(args) -> int:
    """Withdraw an experiment from every derivation, without destroying it.

    The store is append-only and the files stay: an experiment that was wrongly
    ingested is still a record of what happened, and deleting it destroys the
    evidence that the correction was needed. What changes is what it counts
    towards: `list_experiments` leaves it out, so every derivation does too,
    and the modules that consumed it go stale.
    """
    from .rederive import list_stale
    from .schema.write_experiment import retract, retraction_of, unretract

    root = Path(args.data_root) if args.data_root else None

    if args.undo:
        if not unretract(args.experiment, args.cell, root):
            print(f"{args.experiment} was not retracted")
            return 1
        print(f"{args.experiment} is live again")
    else:
        path = retract(args.experiment, args.cell, args.reason or "", root, by="cli")
        record = retraction_of(args.cell, args.experiment, root) or {}
        print(f"retracted {args.experiment}")
        print(f"  reason: {record.get('reason')}")
        print(f"  tombstone: {path}")
        print("  the files are still on disk; what changed is what they count towards")

    stale = list_stale(args.cell, root)
    if stale:
        print(f"{len(stale)} module entry(ies) are now stale. "
              f"`monocell rederive --cell {args.cell}` brings them back in line:")
        for row in stale[:8]:
            print(f"  {row['module']}: {row['reason']}")
    else:
        print("nothing downstream consumed it, so nothing is stale")
    return 0


def _cmd_manifest(args) -> int:
    """Re-index the store from the files, or report where the two disagree.

    The manifest is a rebuildable query index over `data/`; the files are the
    truth. When the two part company (a directory moved by hand, a store copied
    without its experiments), every reader that trusts the index asks for a
    file that is not there. Several messages in this platform tell the user to
    rebuild the index; this is the command that does it.
    """
    from .cells import list_cells
    from .rederive import missing_experiments
    from .schema.manifest import list_experiments, rebuild_manifest, search_experiments

    root = Path(args.data_root) if args.data_root else None

    if args.action == "status":
        cells = list_cells(root)
        rows = len(search_experiments(root))
        print(f"{len(cells)} cell(s), {rows} experiment(s) indexed")
        missing = {c: gone for c in cells if (gone := missing_experiments(c, root))}
        if not missing:
            print("the index and the files agree")
            return 0
        # Two conditions with two different remedies, so they are reported
        # apart. A rebuild drops a row whose directory is gone; it cannot mend
        # a directory that is there and half-written.
        gone_rows: list[str] = []
        half_rows: list[str] = []
        for cell, ids in sorted(missing.items()):
            for eid in ids:
                d = Path(next(iter(
                    r["path"] for r in list_experiments(cell, None, root)
                    if r["experiment_id"] == eid)))
                (gone_rows if not d.exists() else half_rows).append(f"  {cell}: {eid}")
        if gone_rows:
            print(f"{len(gone_rows)} indexed experiment(s) whose directory is gone:")
            for line in gone_rows:
                print(line)
            print("  `monocell manifest rebuild` drops these from the index")
        if half_rows:
            print(f"{len(half_rows)} experiment directory(ies) without series.parquet:")
            for line in half_rows:
                print(line)
            print("  re-ingest the source file, or delete the directory and rebuild")
        return 1

    summary = rebuild_manifest(root)
    print(f"experiments: {summary['experiments_indexed']} indexed "
          f"of {summary['experiment_files']} file(s) seen")
    print(f"artifacts:   {summary['artifacts_indexed']} indexed "
          f"of {summary['artifact_files']} file(s) seen")
    for path in summary["duplicate_experiment_ids"]:
        print(f"  skipped, experiment_id already indexed: {path}")
    for path in summary["duplicate_artifact_ids"]:
        print(f"  skipped, artifact_id already indexed: {path}")
    for path in summary["incomplete"]:
        print(f"  skipped, no series.parquet: {path}")
    for line in summary["unreadable"]:
        print(f"  skipped, unreadable: {line}")
    skipped = (summary["duplicate_experiment_ids"] + summary["duplicate_artifact_ids"]
               + summary["incomplete"] + summary["unreadable"])
    if not skipped:
        print("every file seen was indexed")
    return 0


def _cmd_simulate(args) -> int:
    """Run the cell's newest parameter file through PyBaMM: a CC discharge."""
    from .simulate import simulate

    root = Path(args.data_root) if args.data_root else None
    if not _registered(args.cell, root):
        return 1
    out = simulate(args.cell, c_rate=args.c_rate, model=args.model, root=root)
    s = out["summary"]
    print(f"cell {args.cell}: {s['model']} discharge at {s['c_rate']:g}C, "
          f"parameters from {out['parameter_file']}")
    if out.get("parameter_source") == "abstracted":
        print("  (parameter extraction is abstracted in this copy: the values are "
              "published Chen2020 values)")
    print(f"  capacity   {s['capacity_Ah']:.4f} Ah")
    print(f"  energy     {s['energy_Wh']:.3f} Wh")
    print(f"  voltage    {s['v_start_V']:.3f} V -> {s['v_end_V']:.3f} V "
          f"(cut-off {s['v_cutoff_V']:g} V)")
    print(f"  duration   {s['duration_s'] / 60.0:.1f} min")
    print(f"  artifact   {out['artifact']}")
    if args.out:
        import csv

        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        curve = out["curve"]
        with dest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s", "V_V", "Q_Ah"])
            writer.writerows(zip(curve["t_s"], curve["V_V"], curve["Q_Ah"]))
        print(f"  curve      {dest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The whole command tree, built without running anything."""
    parser = argparse.ArgumentParser(
        prog="monocell",
        description="monocell — battery test data from instrument exports to a PyBaMM model")
    sub = parser.add_subparsers(dest="command", required=True)

    from .schema.ingest import INGESTABLE

    p = sub.add_parser("cell", help="register a cell, list what is registered, or list the "
                                    "build fields a record can carry")
    p.add_argument("action", choices=("register", "list", "fields"))
    p.add_argument("--cell", default=None, help="the cell id to register")
    p.add_argument("--build", default=None,
                   help="a JSON file holding the build record — the fuller route, and the one "
                        "that can be version-controlled")
    p.add_argument("--capacity-Ah", dest="capacity_Ah", type=float, default=None,
                   help="shorthand for a minimal record, with --chemistry")
    p.add_argument("--chemistry", default=None, help="shorthand for a minimal record")
    p.add_argument("--update", action="store_true",
                   help="replace an existing record. Without it, re-registering a DIFFERENT "
                        "build is refused — an identity change is not a re-run")
    p.add_argument("--data-root", default=None, help="data store root (default: data/ or $MONOCELL_DATA)")
    p.set_defaults(func=_cmd_cell)

    p = sub.add_parser("inspect", help="say what a file looks like — matching instrument "
                                      "profiles, the type its shape proposes, and what a "
                                      "read would not place. Writes nothing.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="one data file")
    g.add_argument("--dir", help="a directory of them (an aging campaign is forty files)")
    p.add_argument("--cell", default=None,
                   help="the cell this data is for. Only its CAPACITY is read, and only to "
                        "turn currents into C-rates — which is the evidence that separates a "
                        "GITT pulse from an HPPC one and an RPT from a cycling run.")
    p.add_argument("--data-root", default=None, help="data store root")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("ingest", help="ingest an instrument file (CSV/XLSX) through the standard writer")
    p.add_argument("--cell", required=True, help="cell id (must be registered)")
    p.add_argument("--type", choices=list(INGESTABLE),
                   help="experiment type. Required for --file; optional for --dir, where the "
                        "shape of each file proposes it and a file whose proposal is absent "
                        "or ambiguous is skipped rather than guessed at.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="data file (cycler CSV / EIS CSV or XLSX / pressure CSV)")
    src.add_argument("--dir", help="a directory of them — the plan is printed and nothing is "
                                   "written until --confirm")
    p.add_argument("--confirm", action="store_true",
                   help="--dir only: actually write. Without it the plan is printed and the "
                        "store is untouched.")
    p.add_argument("--sidecar", default=None, help="metadata JSON (protocol/instrument/cell_state)")
    p.add_argument("--instrument", default=None,
                   help="read the file through a named instrument profile (`landt`, `neware`, "
                        "... or one saved in this store) instead of the built-in synonym list. "
                        "Without it, matching profiles are named and not used.")
    p.add_argument("--derive", action="store_true",
                   help="after writing the file, derive what this experiment feeds (the "
                        "modules `rederive` would run, scoped to this ingest)")
    p.add_argument("--data-root", default=None, help="data store root")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("find", help="search the manifest: cell, lot, type, date window, "
                                   "producer, whether ingest flagged it")
    p.add_argument("--cell", action="append", default=[],
                   help="cell id; repeatable")
    p.add_argument("--type", action="append", default=[], choices=list(INGESTABLE) + ["half_cell_ocp"],
                   help="experiment type; repeatable")
    p.add_argument("--lot", default=None, help="batch id — searches its member cells")
    p.add_argument("--producer", default=None, choices=["real", "synthetic"],
                   help="where the data came from")
    p.add_argument("--since", default=None,
                   help="ISO date or prefix (2026-05, 2026-05-12T09). Inclusive.")
    p.add_argument("--until", default=None,
                   help="ISO date or prefix. Inclusive, and a bare date means the END of it.")
    flag = p.add_mutually_exclusive_group()
    flag.add_argument("--flagged", action="store_true", help="only what ingest flagged")
    flag.add_argument("--clean", action="store_true", help="only what it did not")
    p.add_argument("--show-flags", action="store_true", help="print each flag, not just a count")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--data-root", default=None, help="data store root")
    p.set_defaults(func=_cmd_find)

    p = sub.add_parser("promote", help="re-read a stored experiment as a real type — the way "
                                      "out of `misc`. Adds an experiment recording "
                                      "`derived_from`; the original is untouched.")
    p.add_argument("--cell", required=True, help="cell id")
    p.add_argument("--experiment", required=True, help="the experiment_id to re-read")
    p.add_argument("--type", required=True, choices=list(INGESTABLE),
                   help="the type it should have been")
    p.add_argument("--instrument", default=None,
                   help="read its stored raw table through a named instrument profile. Without "
                        "one the built-in synonym list is used, which is a guess.")
    p.add_argument("--data-root", default=None, help="data store root")
    p.set_defaults(func=_cmd_promote)

    p = sub.add_parser("retract", help="withdraw an experiment from every derivation without "
                                       "deleting it — the files stay, derivations stop using them")
    p.add_argument("--cell", required=True)
    p.add_argument("--experiment", required=True, help="the experiment_id to withdraw")
    p.add_argument("--reason", default=None,
                   help="why. Required, because a retraction with no reason cannot be told "
                        "from a mistaken one")
    p.add_argument("--undo", action="store_true", help="remove the tombstone instead")
    p.add_argument("--data-root", default=None, help="data store root")
    p.set_defaults(func=_cmd_retract)

    p = sub.add_parser("manifest", help="re-index the store from the files on disk, or "
                                        "report where the index and the files disagree")
    p.add_argument("action", choices=("rebuild", "status"),
                   help="rebuild: re-scan the files and replace the index. "
                        "status: name the indexed experiments the disk does not have")
    p.add_argument("--data-root", default=None, help="data store root")
    p.set_defaults(func=_cmd_manifest)

    p = sub.add_parser("rederive", help="list and re-run stale artifacts")
    p.add_argument("--cell", required=True, help="cell id")
    p.add_argument("--data-root", default=None, help="data store root")
    p.add_argument("--dry-run", action="store_true", help="list stale artifacts without re-running")
    p.set_defaults(func=_cmd_rederive)

    p = sub.add_parser("simulate", help="run the cell's newest parameter file through PyBaMM: "
                                       "a constant-current discharge from full charge to the "
                                       "lower cut-off")
    p.add_argument("--cell", required=True, help="cell id")
    p.add_argument("--c-rate", dest="c_rate", type=float, default=1.0, help="discharge C-rate")
    p.add_argument("--model", default="DFN", choices=("SPM", "SPMe", "DFN"),
                   help="PyBaMM lithium-ion model")
    p.add_argument("--out", default=None, help="also write the voltage curve as CSV here")
    p.add_argument("--data-root", default=None, help="data store root")
    p.set_defaults(func=_cmd_simulate)

    return parser


def main(argv=None) -> int:
    # legacy Windows consoles default to cp932/cp1252 — messages contain
    # em-dashes; print them lossily rather than crash
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)

    # After parsing, so `--help` and a bad flag write nothing, and keyed to the
    # store this invocation is about rather than to the directory it was run in.
    from . import journal

    journal.configure(Path(args.data_root) if getattr(args, "data_root", None) else None)
    log = journal.logger("cli")
    log.info("run %s", " ".join(argv if argv is not None else sys.argv[1:]))

    try:
        code = args.func(args)
    except KeyboardInterrupt:
        log.warning("interrupted: %s", args.command)
        print("\ninterrupted")
        return 130
    except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
        # The errors a user causes: a file that is not what it claims, a cell
        # that is not registered, a column the schema needs and the export does
        # not have. Each one already carries a message written for a person, and
        # a traceback in front of it says only that the program did not expect
        # its own refusal. Anything else propagates — a traceback for a bug is
        # the right output.
        log.error("refused: %s — %s: %s", args.command, type(exc).__name__, exc)
        print(f"{type(exc).__name__}: {exc}" if not str(exc) else str(exc))
        return 1
    except BaseException as exc:  # logged, then re-raised with its stack intact
        log.exception("crashed: %s — %s: %s", args.command, type(exc).__name__, exc)
        raise
    log.info("exit %s status %s", args.command, code)
    return code


if __name__ == "__main__":
    sys.exit(main())
