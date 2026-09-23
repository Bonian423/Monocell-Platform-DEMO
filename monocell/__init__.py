"""monocell: battery test data, from instrument exports to a PyBaMM model.

Instrument exports are read through one schema, checked at ingest, and stored
append-only. Derived results are artifacts that record the content hashes of
their inputs, so new data marks them stale and a re-derive brings them current.
The parameter-extraction step is abstracted in this copy (`monocell.engine`).
"""

__version__ = "0.2.0"
