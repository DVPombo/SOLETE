# -*- coding: utf-8 -*-
"""
solete/paths.py -- the single place that knows where files live.

Nothing else in the repository should build a data path by hand, and nothing
should depend on the current working directory. Scripts, notebooks, tests and
the dataset-cleaning pipeline all resolve files through this module, so the
project runs the same from Spyder, a terminal, Colab, Docker or pytest.

THE DATA FOLDER
---------------
The data files are NOT in git. Download them from figshare and unzip so that
the folder structure is preserved:

    <repo>/data/
        hdf5/       SOLETE_Pombo_1sec.h5   SOLETE_clean_1sec.h5   SOLETE_clean_1min.h5
                    SOLETE_clean_5min.h5   SOLETE_clean_1h.h5     (+ v3 files, see below)
        parquet/    SOLETE_clean_1sec.parquet ... SOLETE_clean_1h.parquet
        derived/    created by the code: expanded caches (never edit by hand)

That is exactly the layout of the figshare upload, so "drag the contents of
the download into data/" is the whole installation. The files may also sit
flat in data/ (no hdf5/ or parquet/ subfolder); both are found.

To keep the data elsewhere (an external drive, a cluster scratch folder), set
the environment variable SOLETE_DATA_DIR to that folder; the same
hdf5/ parquet/ structure is expected inside it.

FILE VERSIONS
-------------
    "v4"  SOLETE_clean_<res>.h5 / .parquet     res in 1sec, 1min, 5min, 1h
          (cleaned, with <column>_qc flags -- the figshare version 4 files)
    "v3"  SOLETE_Pombo_<res>.h5                res in 1sec, 1min, 5min, 60min
          (the originals; 1h in v4 is called 60min in v3)

The forecasting platform and the benchmarks in this repository were built on
the v3 hourly file and still read it by default (version="v3"). Reading the v4
files through the platform needs the QC-flag reconciliation described in
docs/RESTRUCTURE_NOTES.md first -- see that file before switching.
"""

from __future__ import annotations

import os
from pathlib import Path

FIGSHARE_DOI = "10.11583/DTU.17040767"
FIGSHARE_URL = "https://doi.org/" + FIGSHARE_DOI

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("SOLETE_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()
HDF5_DIR = DATA_DIR / "hdf5"
PARQUET_DIR = DATA_DIR / "parquet"
DERIVED_DIR = DATA_DIR / "derived"          # generated caches, safe to delete

OUTPUT_DIR = REPO_ROOT / "outputs"          # trained models, result files, figures
EXAMPLES_DIR = REPO_ROOT / "examples"       # tiny tracked sample files used by notebooks
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
SPLITS_DIR = BENCHMARKS_DIR / "splits"
RESULTS_DIR = BENCHMARKS_DIR / "results"

RESOLUTIONS = ("1sec", "1min", "5min", "60min")
_ALIASES = {"1h": "60min", "60min": "60min", "1hour": "60min",
            "1sec": "1sec", "1s": "1sec", "1min": "1min", "5min": "5min"}
_V4_RES = {"1sec": "1sec", "1min": "1min", "5min": "5min", "60min": "1h"}
_V3_RES = {"1sec": "1sec", "1min": "1min", "5min": "5min", "60min": "60min"}


def normalize_resolution(resolution: str) -> str:
    """Return the canonical resolution key ('1sec','1min','5min','60min').
    Accepts '1h' as an alias for '60min' (v4 naming)."""
    try:
        return _ALIASES[str(resolution).lower()]
    except KeyError:
        raise ValueError(
            f"Unknown resolution {resolution!r}; expected one of "
            f"{sorted(set(_ALIASES))}."
        ) from None


def data_filename(resolution: str, version: str = "v3", fmt: str = "hdf5") -> str:
    """File name of the SOLETE file for a resolution / version / format."""
    res = normalize_resolution(resolution)
    if version == "v4":
        stem = f"SOLETE_clean_{_V4_RES[res]}"
    elif version == "v3":
        stem = f"SOLETE_Pombo_{_V3_RES[res]}"
        if fmt != "hdf5":
            raise ValueError("Version 3 files exist only as HDF5.")
    else:
        raise ValueError(f"version must be 'v3' or 'v4', got {version!r}")
    return stem + (".parquet" if fmt == "parquet" else ".h5")


def _search_dirs(fmt: str):
    primary = PARQUET_DIR if fmt == "parquet" else HDF5_DIR
    return [primary, DATA_DIR]


class DataFileNotFoundError(FileNotFoundError):
    """A SOLETE data file is missing from the data folder (a FileNotFoundError,
    so existing `except FileNotFoundError` handling keeps working). The message
    says where to download the files and where to put them. The test suite turns
    this into a skip, so a fresh clone without the data runs green."""


def _not_found(what: str, searched) -> DataFileNotFoundError:
    lines = "\n".join(f"    {d}" for d in searched)
    return DataFileNotFoundError(
        f"Could not find {what}.\nLooked in:\n{lines}\n\n"
        f"The SOLETE data files are not part of the repository. Download them from\n"
        f"    {FIGSHARE_URL}\n"
        f"and unzip them into {DATA_DIR} keeping the hdf5/ and parquet/ folders\n"
        f"(or set the SOLETE_DATA_DIR environment variable to where you keep them).\n"
        f"See data/README.md."
    )


def find_data_file(resolution: str, version: str = "v3", fmt: str = "hdf5") -> Path:
    """Locate a SOLETE data file by resolution; raise a helpful error if absent."""
    name = data_filename(resolution, version, fmt)
    dirs = _search_dirs(fmt)
    for d in dirs:
        p = d / name
        if p.is_file():
            return p
    raise _not_found(name, dirs)


def resolve_input(path_or_name) -> Path:
    """Resolve a user-supplied input file.

    An existing path (absolute, or relative to the current directory) is used
    as given. A bare file name is looked up in data/hdf5, data/parquet, data/
    and examples/. Used by every command-line script so that
        python dataset/pipeline/clean_solete_1sec.py SOLETE_Pombo_1sec.h5
    works from anywhere once the files are in data/.
    """
    p = Path(path_or_name).expanduser()
    if p.exists():
        return p.resolve()
    if p.name == str(path_or_name):                      # bare name
        for d in (HDF5_DIR, PARQUET_DIR, DATA_DIR, EXAMPLES_DIR):
            cand = d / p.name
            if cand.is_file():
                return cand
    raise _not_found(str(path_or_name), (Path.cwd(), HDF5_DIR, PARQUET_DIR, DATA_DIR, EXAMPLES_DIR))


def resolve_sample(path_or_name) -> Path:
    """Resolve a notebook sample file (e.g. 'SOLETE_sample.h5'): as given,
    else inside examples/, else in the data folder."""
    p = Path(path_or_name).expanduser()
    if p.exists():
        return p.resolve()
    for d in (EXAMPLES_DIR, HDF5_DIR, DATA_DIR):
        cand = d / p.name
        if cand.is_file():
            return cand
    raise _not_found(str(path_or_name), (Path.cwd(), EXAMPLES_DIR, HDF5_DIR, DATA_DIR))


def resolve_output_prefix(prefix: str, kind: str = "hdf5") -> str:
    """Where a pipeline script should write '<prefix>.h5' (or '<prefix>_<rule>.h5').

    A prefix that already contains a directory is used as given. A bare prefix
    (the default, e.g. 'SOLETE_clean_1sec') goes into data/hdf5 so the outputs
    land exactly where the figshare layout expects them. Creates the folder."""
    p = Path(prefix)
    if p.parent != Path("."):
        return str(p)
    folder = {"hdf5": HDF5_DIR, "parquet": PARQUET_DIR}[kind]
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / p.name)


def derived_path(name: str) -> Path:
    """Path for a generated cache file (data/derived/<name>); folder is created."""
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    return DERIVED_DIR / name


def output_path(name: str) -> Path:
    """Path for a generated model / result / figure (outputs/<name>); folder is created."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def describe() -> str:
    """Human-readable summary of what the code will use and what is present."""
    rows = [f"repo root : {REPO_ROOT}",
            f"data dir  : {DATA_DIR}" + ("" if DATA_DIR.is_dir() else "   (missing)")]
    for label, d in (("hdf5", HDF5_DIR), ("parquet", PARQUET_DIR)):
        files = sorted(p.name for p in d.glob("*") if p.is_file()) if d.is_dir() else []
        rows.append(f"  {label:8s}: " + (", ".join(files) if files else "(empty)"))
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
