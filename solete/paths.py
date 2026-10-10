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
        hdf5/       SOLETE_Pombo_1sec_original_v4.h5
                    SOLETE_Pombo_1sec_v4.h5  SOLETE_Pombo_1min_v4.h5
                    SOLETE_Pombo_5min_v4.h5  SOLETE_Pombo_60min_v4.h5
                    (+ the v3 files, see below; the raw v3 1 s file is SOLETE_Pombo_1sec.h5)
        parquet/    the same five stems with .parquet
        derived/    created by the code: expanded caches (never edit by hand)

That is exactly the layout of the figshare upload, so "drag the contents of
the download into data/" is the whole installation. The files may also sit
flat in data/ (no hdf5/ or parquet/ subfolder); both are found.

To keep the data elsewhere (an external drive, a cluster scratch folder), set
the environment variable SOLETE_DATA_DIR to that folder; the same
hdf5/ parquet/ structure is expected inside it.

FILE VERSIONS
-------------
    "v4"  SOLETE_Pombo_<res>_v4.h5 / .parquet   res in 1sec, 1min, 5min, 60min
          (cleaned, with <column>_qc flags and the model columns -- the figshare version 4 files)
          SOLETE_Pombo_1sec_original_v4.h5 / .parquet   (original=True: the raw 1 s data,
          sorted, nine measured columns only, nothing cleaned)
    "v3"  SOLETE_Pombo_<res>.h5                 res in 1sec, 1min, 5min, 60min
          (the originals, unchanged; the 1 s one is the raw input of the build)

"1h" is accepted as an alias of "60min" in every resolution argument (input only: no file is named 1h).

The forecasting platform and the benchmarks in this repository were built on
the v3 hourly file and still read it by default (version="v3"). Reading the v4
files through the platform works (`Control_Var["data_version"] = "v4"`): their flags are read
as they are, the model columns are computed by solete/expansion.py (docs/RESTRUCTURE_NOTES.md §2).
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
RELEASE_VERSION_TAG = "v4"


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


def data_filename(resolution: str, version: str = "v3", fmt: str = "hdf5", original: bool = False) -> str:
    """File name of the SOLETE file for a resolution / version / format.

    v3  SOLETE_Pombo_<res>.h5            (HDF5 only)
    v4  SOLETE_Pombo_<res>_v4.<ext>      (the scheme is implemented here and nowhere else)
    v4, original=True (1 s only)  SOLETE_Pombo_1sec_original_v4.<ext>
    """
    res = normalize_resolution(resolution)
    if original and (version != "v4" or res != "1sec"):
        raise ValueError("The _original file exists for version 'v4' at 1 s only.")
    if version == "v4":
        stem = f"SOLETE_Pombo_{res}" + ("_original" if original else "") + f"_{RELEASE_VERSION_TAG}"
    elif version == "v3":
        stem = f"SOLETE_Pombo_{res}"
        if fmt != "hdf5":
            raise ValueError("Version 3 files exist only as HDF5.")
    else:
        raise ValueError(f"version must be 'v3' or 'v4', got {version!r}")
    return stem + (".parquet" if fmt == "parquet" else ".h5")


def release_stems() -> list:
    """The five file stems of the v4 release, in build order (original first)."""
    return [data_filename("1sec", "v4", original=True)[:-3]] + [data_filename(r, "v4")[:-3] for r in RESOLUTIONS]


def release_path(resolution: str, fmt: str = "hdf5", original: bool = False, folder=None) -> Path:
    """Where a v4 release file is (to be) written: <data>/hdf5 or <data>/parquet, or `folder`.
    No existence check (see find_data_file for that); no folder is created."""
    name = data_filename(resolution, "v4", fmt, original=original)
    return (Path(folder) if folder else (PARQUET_DIR if fmt == "parquet" else HDF5_DIR)) / name


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


def find_data_file(resolution: str, version: str = "v3", fmt: str = "hdf5", original: bool = False) -> Path:
    """Locate a SOLETE data file by resolution; raise a helpful error if absent."""
    name = data_filename(resolution, version, fmt, original=original)
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
    (the default, e.g. 'SOLETE_Pombo_1sec_v4') goes into data/hdf5 so the outputs
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
