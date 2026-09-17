# load_solete.R
#
# Minimal R loader for the raw SOLETE HDF5 files (SOLETE_short.h5,
# SOLETE_Pombo_60min.h5, and any sibling SOLETE_Pombo_<resolution>.h5 built
# the same way). Mirrors Python's raw-loading path -- i.e. the plain
# `pd.read_hdf(name)` call at the top of `import_SOLETE_data()`'s 'Build'
# branch in Functions.py -- not the full pipeline. It does not reimplement
# QC-flagging (`apply_qc_flags`), `ExpandSOLETE()`, or the PV/thermodynamic
# models; those stay Python-only. Column names and semantics are exactly the
# raw file's columns -- see ../DATA_DICTIONARY.md for what each one means.
#
# Requires: hdf5r (CRAN; pulls in bit64 as a dependency). On Debian/Ubuntu,
# both are packaged directly -- no CRAN network access needed:
#   sudo apt-get install r-cran-hdf5r
#
# Package choice: the original brief suggested `rhdf5` (Bioconductor), but
# rhdf5 requires BiocManager/Bioconductor's own installer rather than a
# straight apt/CRAN install, and SOLETE has no other Bioconductor
# dependency to justify pulling that ecosystem in. `hdf5r` is a
# general-purpose CRAN package (not tied to Bioconductor), is available as
# a plain Debian/Ubuntu package (`r-cran-hdf5r`), and was the one actually
# installed and tested against the real files below -- so it's the current,
# verified choice for this repo rather than an untested guess.
#
# WHY THE MANUAL BLOCK-LEVEL READING BELOW:
# These files are pandas DataFrames written via `DataFrame.to_hdf(..., 
# format="fixed")` (the pytables "fixed" layout, not pytables "table"
# format). That layout is pandas' own internal BlockManager serialized to
# HDF5, not a generic table -- there is no single "the data" dataset to
# just read. Concretely, under the top-level `DATA` group:
#   - axis0            : column names, in DataFrame column order
#   - axis1             : the DataFrame index, as int64 nanoseconds-since-
#                         epoch (both real SOLETE files have a plain
#                         DatetimeIndex, no MultiIndex)
#   - block<N>_items    : column names belonging to block N (pandas splits
#                         columns of different dtypes into separate
#                         blocks; both real SOLETE files happen to be a
#                         single all-float64 block, but the code below
#                         does not assume that stays true)
#   - block<N>_values   : the block's data. Verified against known values
#                         from `pd.read_hdf()` on SOLETE_short.h5: hdf5r
#                         reads this with dimensions (n_items, n_rows) --
#                         i.e. *features first, then time* -- the reverse
#                         of the (n_rows, n_items) shape h5py/numpy report
#                         for the exact same dataset. This is an HDF5
#                         row-major vs. R column-major storage-order
#                         artifact, not a difference in the actual data --
#                         confirmed by checking specific known cell values,
#                         not assumed. Transposing block<N>_values (or
#                         indexing it as [item_row, time_col], as done
#                         below) recovers the standard time-as-rows shape.
#
# Nanosecond int64 timestamps are deliberately converted to seconds using
# integer64 (`bit64`) arithmetic *before* going to double: a raw
# nanoseconds-since-1970 value is ~1.5e18, well past a double's 53-bit
# (~9e15) exact-integer range, so converting straight to double first would
# silently round to the nearest ~256ns. Dividing down to seconds in
# integer64 first (SOLETE's timestamps are always whole seconds -- hourly
# resolution here, sub-hourly resolutions elsewhere in the project are
# still whole seconds) keeps the conversion exact.

#' Load a raw SOLETE HDF5 file into a data.frame
#'
#' @param path Path to a SOLETE_*.h5 file (e.g. "SOLETE_Pombo_60min.h5").
#' @return A data.frame with a `datetime` column (POSIXct, UTC) first,
#'   followed by every raw column from the file, in the file's original
#'   column order. Column names match the Python loader's `df.columns`
#'   exactly (see ../DATA_DICTIONARY.md) -- no renaming is done.
load_solete <- function(path) {
  if (!file.exists(path)) {
    stop("SOLETE file not found: ", path)
  }

  f <- hdf5r::H5File$new(path, mode = "r")
  on.exit(f$close_all(), add = TRUE)

  if (!f$link_exists("DATA")) {
    stop(
      "'", path, "' doesn't have the expected top-level 'DATA' group -- ",
      "this loader only handles pandas to_hdf(..., format='fixed') files ",
      "like the real SOLETE_*.h5 files, not pytables 'table'-format HDF5."
    )
  }
  grp <- f[["DATA"]]

  col_order <- grp[["axis0"]]$read()
  ts_ns <- grp[["axis1"]]$read() # integer64, ns since epoch

  # Find every block<N>_items / block<N>_values pair actually present.
  member_names <- grp$names
  item_members <- grep("^block[0-9]+_items$", member_names, value = TRUE)
  block_ids <- sub("_items$", "", item_members)
  if (length(block_ids) == 0) {
    stop(
      "'", path, "' has a 'DATA' group but no block*_items datasets -- ",
      "unexpected pandas fixed-format layout, refusing to guess."
    )
  }

  columns <- list()
  for (block_id in block_ids) {
    items <- grp[[paste0(block_id, "_items")]]$read()
    values <- grp[[paste0(block_id, "_values")]]$read()

    # A block holding exactly one column comes back 1-D, not a matrix --
    # reshape so the indexing below is uniform regardless of block width.
    if (is.null(dim(values))) {
      values <- matrix(values, nrow = length(items))
    }

    for (i in seq_along(items)) {
      columns[[items[i]]] <- values[i, ]
    }
  }

  missing <- setdiff(col_order, names(columns))
  if (length(missing) > 0) {
    stop(
      "Column(s) listed in axis0 but not found in any block: ",
      paste(missing, collapse = ", ")
    )
  }

  ts_sec <- as.double(ts_ns %/% bit64::as.integer64(1e9))
  datetime <- as.POSIXct(ts_sec, origin = "1970-01-01", tz = "UTC")

  df <- data.frame(datetime = datetime, check.names = FALSE)
  for (name in col_order) {
    df[[name]] <- columns[[name]]
  }

  df
}

# Minimal CLI smoke test: `Rscript load_solete.R path/to/SOLETE_*.h5`
if (identical(environment(), globalenv()) && sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1) {
    df <- load_solete(args[1])
    cat(sprintf("Loaded %d rows x %d cols from %s\n", nrow(df), ncol(df), args[1]))
    cat("Columns:", paste(names(df), collapse = ", "), "\n")
    cat("Range:", format(min(df$datetime)), "..", format(max(df$datetime)), "\n")
    print(utils::head(df, 3))
  }
}
