"""The layer boundaries, checked by reading the imports.

* `monocell.schema` (readers, profiles, QC, the store) depends on nothing that
  derives or simulates. Ingest works without the engine or PyBaMM.
* `monocell.engine` reads the store through `schema` and `cells` only. It is
  the one replaceable part, so it must not reach into anything else.
* Importing the package, the CLI or the ingest path does not import PyBaMM.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import monocell

PKG = Path(monocell.__file__).resolve().parent


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PKG.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path) -> set[str]:
    """Absolute names of everything `path` imports, relative imports resolved."""
    name = _module_name(path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            out.add(module)
            out.update(f"{module}.{alias.name}" for alias in node.names)
    return out


def _within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _files(subpackage: str) -> list[Path]:
    files = sorted((PKG / subpackage).rglob("*.py"))
    assert files, f"no files under monocell/{subpackage}"
    return files


def test_the_schema_layer_imports_nothing_that_derives_or_simulates():
    downstream = ("monocell.engine", "monocell.simulate", "monocell.rederive",
                  "monocell.autofit", "monocell.cli")
    offending = {
        _module_name(f): sorted(i for i in _imports(f) if any(_within(i, d) for d in downstream))
        for f in _files("schema")
    }
    assert not {k: v for k, v in offending.items() if v}


def test_the_engine_reads_the_store_through_schema_and_cells_only():
    allowed = ("monocell.engine", "monocell.schema", "monocell.cells", "monocell._fs")
    offending = {
        _module_name(f): sorted(i for i in _imports(f)
                                if _within(i, "monocell") and i != "monocell"
                                and not any(_within(i, a) for a in allowed))
        for f in _files("engine")
    }
    assert not {k: v for k, v in offending.items() if v}


def test_no_module_imports_pybamm_at_import_time():
    """PyBaMM is imported inside the functions that solve, so every ingest,
    query and re-derive runs without it. Checked in a fresh interpreter,
    because this test process may already have imported it."""
    code = ("import sys\n"
            "import monocell, monocell.cli, monocell.rederive, monocell.autofit\n"
            "import monocell.schema.ingest, monocell.engine.extract, monocell.simulate\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'pybamm'))\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=True, cwd=PKG.parent)
    assert result.stdout.strip() == "[]"
