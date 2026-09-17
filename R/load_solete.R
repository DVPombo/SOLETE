# load_solete.R — minimal R loader for the SOLETE HDF5 files
#
# Reads the `.h5` files shipped at the repo root (`SOLETE_short.h5`,
# `SOLETE_Pombo_60min.h5`) and returns a data.frame with the same columns,
# in the same order, and the same values as `pandas.read_hdf(path, key="DATA")`
# in the Python loader. Column names and semantics are NOT redefined here —
# see `../DATA_DICTIONARY.md` for what each column means.
#
# Tested (Session 5, Phase 7): verified against both real files in this repo,
# see r/README.md for how and what was checked.
#
# Dependencies (CRAN, apt-installable on Ubuntu as r-cran-hdf5r / r-cran-bit64):
#   install.packages(c("hdf5r", "bit64"))
# `hdf5r` was chosen over the Bioconductor `rhdf5` package (the option named in
# the original task prompt) because it's plain CRAN and this repo has no other
# Bioconductor dependency — `rhdf5` remains a fine alternative if a project
# already depends on Bioconductor infrastructure.

#' Load a SOLETE HDF5 file into a data.frame
#'
#' @param path Path to a SOLETE `.h5` file (e.g. "SOLETE_Pombo_60min.h5" or
#'   "SOLETE_short.h5"), written by `pandas.DataFrame.to_hdf(path, key="DATA")`
#'   in fixed format (a single `axis0`/`axis1`/`block0_items`/`block0_values`
#'   layout — this loader assumes exactly one numeric block, which is true of
#'   both real files shipped in this repo as of Phase 7. If a future version
#'   of the dataset adds a second dtype (e.g. a string column), `pandas`
#'   writes a second `block1_*` group, which this loader does not currently
#'   handle — see the "Known limitations" note in r/README.md.)
#' @param key HDF5 group name the frame was written under. Default "DATA",
#'   matching every real file in this repo.
#' @param tz Timezone to attach to the returned POSIXct index. SOLETE
#'   timestamps are naive (no timezone offset stored) in both the Python and R
#'   loaders; default "UTC" avoids silently applying the local machine's
#'   timezone. Pass tz = "" if you specifically want local-time interpretation.
#'
#' @return A data.frame with one row per timestamp, one column per SOLETE
#'   variable (identical names/order to the Python loader), plus a
#'   `datetime` column (POSIXct) holding what pandas exposes as the
#'   DatetimeIndex. Row order matches the file's on-disk order (chronological
#'   in both real files).
#'
#' @examples
#' \dontrun{
#' df <- load_solete("SOLETE_Pombo_60min.h5")
#' head(df)
#' }
load_solete <- function(path, key = "DATA", tz = "UTC") {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("Package 'hdf5r' is required. Install with install.packages('hdf5r') ",
         "or, on Debian/Ubuntu, apt-get install r-cran-hdf5r.")
  }
  if (!requireNamespace("bit64", quietly = TRUE)) {
    stop("Package 'bit64' is required (the SOLETE index is stored as ",
         "nanosecond int64 timestamps, which lose precision as a plain R ",
         "double). Install with install.packages('bit64') or, on ",
         "Debian/Ubuntu, apt-get install r-cran-bit64.")
  }
  if (!file.exists(path)) {
    stop("File not found: ", path)
  }

  f <- hdf5r::H5File$new(path, mode = "r")
  on.exit(f$close_all(), add = TRUE)

  if (!f$exists(key)) {
    stop("Group '", key, "' not found in ", path, ". Available top-level ",
         "names: ", paste(names(f), collapse = ", "))
  }
  grp <- f[[key]]

  if (!grp$exists("block0_values") || !grp$exists("axis0") || !grp$exists("axis1")) {
    stop("'", key, "' does not look like a pandas fixed-format frame ",
         "(expected axis0/axis1/block0_values). This loader only handles ",
         "the single-numeric-block layout used by every real file in this ",
         "repo as of Phase 7 — see the 'Known limitations' note in r/README.md.")
  }
  if (grp$exists("block1_values")) {
    stop("This file has a second data block ('block1_values'), which means ",
         "at least one column has a dtype different from the rest (e.g. a ",
         "string column) — this loader does not currently merge multiple ",
         "blocks. Every real SOLETE file shipped in this repo as of Phase 7 ",
         "is single-block (all-float64); if this file isn't, the loader ",
         "needs extending rather than guessing at a merge order.")
  }

  # Column names. axis0 holds the column labels for a single-block frame;
  # for a single-block frame this is identical to block0_items (verified
  # against both real files) but axis0 is the authoritative one to read since
  # it's the frame's actual column index, not just this block's subset.
  col_names <- grp[["axis0"]]$read()

  # Values. HDF5 stores the 2-D dataset in the same (n_rows, n_cols)
  # row-major layout pandas/numpy wrote it in, but hdf5r reports (and fills)
  # array dimensions without transposing the raw bytes — so what comes back
  # from $read() has dim (n_cols, n_rows), i.e. transposed relative to the
  # pandas frame. Confirmed against both real files: element [j, i] here
  # equals the Python `block0_values[i, j]` (row i, column j). Transpose to
  # get the (n_rows, n_cols) orientation pandas exposes.
  raw_vals <- grp[["block0_values"]]$read()
  vals <- t(raw_vals)  # now (n_rows, n_cols), matching pandas orientation

  # Index. Stored as int64 nanoseconds since the Unix epoch. This exceeds
  # what a double can represent exactly (raw values are ~1.5e18; doubles are
  # only exact up to 2^53 ~= 9.007e15), so do the ns -> s conversion in
  # integer64 arithmetic (bit64) before ever touching a double.
  idx_ns <- grp[["axis1"]]$read()  # integer64, via bit64
  idx_s <- bit64::as.integer64(idx_ns) %/% bit64::as.integer64(1000000000L)
  datetime <- as.POSIXct(as.numeric(idx_s), origin = "1970-01-01", tz = tz)

  df <- as.data.frame(vals)
  names(df) <- col_names
  df <- cbind(datetime = datetime, df)
  rownames(df) <- NULL
  df
}

if (sys.nframe() == 0 && !interactive()) {
  # Allow `Rscript load_solete.R <path-to-h5>` as a quick smoke test.
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1) {
    df <- load_solete(args[1])
    cat("Loaded", nrow(df), "rows x", ncol(df), "columns from", args[1], "\n")
    print(utils::head(df))
  }
}
