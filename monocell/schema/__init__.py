"""The experiment data layer: the common schema.

One writer function (`write_experiment`) is the contract every ingestor goes
through. Append-only. Parquet + JSON sidecars under `data/` are the source of
truth; the DuckDB manifest is a rebuildable query index.
"""

from .tables import SPECS, TableSpec
from .write_experiment import (ExperimentMissing, load_experiment, load_meta,
                               write_experiment)
from .quality import lin_kk, sum_check
from .manifest import content_hash, find_experiment, list_artifacts, list_experiments, rebuild_manifest, upsert_artifact, upsert_experiment
from .artifacts import (artifact_input_env, artifact_order_key,
                        envelope_order_key, load_artifact, newest_artifact,
                        read_artifact_or_none, record_pattern, write_artifact)
from .ingest import ingest, read_cycler_csv, read_eis_xlsx, read_pressure_csv

__all__ = [
    "ExperimentMissing",
    "SPECS",
    "TableSpec",
    "load_experiment",
    "load_meta",
    "write_experiment",
    "lin_kk",
    "sum_check",
    "content_hash",
    "find_experiment",
    "list_artifacts",
    "list_experiments",
    "rebuild_manifest",
    "upsert_artifact",
    "upsert_experiment",
    "write_artifact",
    "load_artifact",
    "read_artifact_or_none",
    "newest_artifact",
    "artifact_order_key",
    "envelope_order_key",
    "artifact_input_env",
    "record_pattern",
    "ingest",
    "read_cycler_csv",
    "read_eis_xlsx",
    "read_pressure_csv",
]
