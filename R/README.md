# R loader

`load_solete.R` provides `load_solete(path)`, a minimal R function that reads
one of SOLETE's real HDF5 files (`SOLETE_short.h5`, `SOLETE_Pombo_60min.h5`,
or any sibling `SOLETE_Pombo_<resolution>.h5` produced the same way) into a
plain R `data.frame`, using the [`hdf5r`](https://cran.r-project.org/package=hdf5r)
package.

**Scope:** this mirrors the *raw* loading step only — Python's
`pd.read_hdf(name)` call, i.e. the start of `import_SOLETE_data()`'s `'Build'`
branch in `../Functions.py` — not the full pipeline. It does not reimplement
QC-flagging, `ExpandSOLETE()`, or the PV/thermodynamic models; those stay
Python-only. Column names, units, and semantics are exactly the raw file's
columns and are **not** re-described here — see
[`../DATA_DICTIONARY.md`](../DATA_DICTIONARY.md) for what each one means, and
`../QC_SCHEMA.md` if you need the QC-flag columns (which this loader does not
compute).

## Install

On Debian/Ubuntu, both `hdf5r` and its `bit64` dependency are packaged
directly — no CRAN/Bioconductor network access needed:

```sh
sudo apt-get install r-cran-hdf5r
```

(Elsewhere, `install.packages("hdf5r")` from CRAN.)

## Use

```r
source("load_solete.R")
df <- load_solete("../SOLETE_Pombo_60min.h5")
head(df)
```

Returns a `data.frame` with a `datetime` column (POSIXct, UTC) first, followed
by every raw column from the file in its original order.

## Tested

Run and verified against both real files in this repo (`SOLETE_short.h5` and
`SOLETE_Pombo_60min.h5`): every value and every timestamp matches Python's
`pandas.read_hdf()` output exactly (bit-for-bit on the underlying float64
data, verified via raw binary comparison rather than a text/CSV round trip,
which loses precision on its own and would give a false negative). Not an
untested/hand-written-only script.
