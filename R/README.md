# R loader for SOLETE

A minimal R loader for the SOLETE `.h5` files shipped at the repo root
(`SOLETE_short.h5`, `SOLETE_Pombo_60min.h5`), for R users who want the raw
data without going through Python.

**Column names and semantics are documented in
[`../DATA_DICTIONARY.md`](../DATA_DICTIONARY.md), not duplicated here.**
`load_solete()` returns the same columns, in the same order, as the Python
loader (`pandas.read_hdf(path, key="DATA")`) — read the data dictionary for
what each one means, its units, and known data-quality caveats.

## Usage

```r
source("r/load_solete.R")
df <- load_solete("SOLETE_Pombo_60min.h5")
head(df)
```

`df` is a plain data.frame: one row per timestamp (chronological, matching
on-disk order), a `datetime` column (POSIXct, UTC), and one column per SOLETE
variable with the exact same name as the Python loader (e.g.
`` `P_Solar[kW]` ``, `` `TEMPERATURE[degC]` `` — backtick these in R since
they contain `[`/`]`).

## Dependencies

CRAN packages `hdf5r` and `bit64`. On Debian/Ubuntu these are apt-installable
without needing CRAN or Bioconductor network access:

```sh
sudo apt-get install r-cran-hdf5r r-cran-bit64
```

Or from within R:

```r
install.packages(c("hdf5r", "bit64"))
```

**Why `hdf5r` and not `rhdf5`** (the package suggested when this session was
scoped): `rhdf5` is a Bioconductor package, which pulls in Bioconductor's own
package-management infrastructure. `hdf5r` is plain CRAN, is
apt-installable on its own, and this repo has no other Bioconductor
dependency to justify pulling that infrastructure in for one loader. `rhdf5`
would work too if a downstream project already depends on Bioconductor.

## Testing status

**Actually tested against real data**, not just written by hand. An R
environment was available (`r-base-core` + `r-cran-hdf5r` + `r-cran-bit64`,
installed via `apt-get`), so this was verified rather than shipped untested:

- `load_solete()` was run against both real files in this repo
  (`SOLETE_short.h5`, 24 rows, and `SOLETE_Pombo_60min.h5`, 10,969 rows).
- Row count, column count, column names/order, and the `datetime` index were
  checked against `pandas.read_hdf()` on the same files.
- **Numeric values were verified bit-for-bit identical** to the raw bytes in
  the HDF5 file (compared directly against `h5py`'s read of
  `DATA/block0_values`, via a raw binary dump — not a CSV/text comparison).
  A first pass using a CSV-based comparison showed ~1e-13-magnitude
  differences on some columns; those turned out to be a text round-trip
  artifact of writing/reading through CSV with finite decimal precision, not
  a real discrepancy — ruled out by comparing raw doubles directly instead of
  through text.

## Known limitations

- Both real files in this repo are written by `pandas.DataFrame.to_hdf(...,
  format="fixed")` as a **single** numeric block (every column is
  `float64`). `load_solete()` assumes this and will raise an informative
  error if it ever encounters a file with a second block (`block1_values`,
  which pandas would write if a column had a different dtype, e.g. a string
  QC column stored directly in the `.h5`). As of Phase 7, QC flags are
  computed at load time in Python (`Functions.py`'s QC-rule functions), not
  stored in the `.h5` files themselves, so this hasn't come up — but if that
  ever changes, this loader will need extending rather than silently
  guessing how to merge blocks.
- No QC-flag computation is reproduced here — this loader only reads the raw
  columns. If you need QC flags in R, either port the relevant logic from
  `Functions.py`/`QC_SCHEMA.md`, or compute them in Python and merge in R.
- Timestamps are read as UTC by default (`tz` argument on `load_solete()`).
  The source data has no timezone offset recorded in either language's
  loader — this is a labeling choice, not a conversion, and matches what the
  Python side does (a naive `datetime64[ns]` index).
