"""Parameter extraction (abstracted in this copy).

In the full platform, this step analyses each test type and assembles a
cell-specific PyBaMM parameter set with per-parameter provenance. That code is
proprietary and is not included here.

The interface is kept so the rest of the pipeline runs unchanged:
`extract.run` is the runner `rederive` calls, it writes a versioned
parameter-file artifact through the same provenance envelope every derived
result uses, and the file records the content hash of every experiment it
consumed. The values themselves are published literature values that do not
depend on the data (see `extract.STANDARD_PARAMETERS`).
"""
