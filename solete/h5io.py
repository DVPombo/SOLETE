# -*- coding: utf-8 -*-
"""
solete/h5io.py -- read and write SOLETE HDF5 files in slices, so no stage has to hold the 1-second frame.

Why this exists: `pd.read_hdf` can only slice (`start`/`stop`) a file stored in the PyTables `table` format.
The published v3 files (and the resampled files) are pandas `fixed` format, which pandas can only load
whole. `read_rows` gives the same row-range read for both formats; for `fixed` it reads the arrays
with PyTables directly (the layout pandas writes: `axis0` columns, `axis1` index, one `blockN_values`
array of shape (rows, items) per dtype block). It is checked against `pd.read_hdf` in
tests/test_h5io.py and refuses anything it does not understand instead of guessing.

Pure pandas + PyTables; no solete imports, so it can be used from anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_KEY = "DATA"
COMPLEVEL = 1
COMPLIB = "zlib"


def _storer_info(path, key):
    with pd.HDFStore(str(path), mode="r") as store:
        storer = store.get_storer(key)
        fmt = storer.format_type
        if fmt == "table":
            return {"format": "table", "nrows": int(storer.nrows),
                    "columns": list(storer.non_index_axes[0][1]) if storer.non_index_axes else None}
        return {"format": fmt, "nrows": None, "columns": None}


def _fixed_layout(path, key):
    """(columns, nrows, [(items, node_name), ...]) of a fixed-format frame, or raise."""
    import tables
    with tables.open_file(str(path), "r") as f:
        g = f.get_node("/" + key)
        attrs = g._v_attrs
        if str(getattr(attrs, "pandas_type", "")) != "frame":
            raise ValueError(f"{path}:{key} is not a pandas fixed-format DataFrame")
        nblocks = int(attrs.nblocks)
        axis1 = g.axis1
        if str(axis1.attrs.index_class) != "datetime":
            raise ValueError(f"{path}:{key}: index is not a DatetimeIndex ({axis1.attrs.index_class!r})")
        if not bool(getattr(axis1.attrs, "transposed", True)):
            raise ValueError("unsupported fixed layout (index not transposed)")
        columns = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in g.axis0.read()]
        blocks = []
        for b in range(nblocks):
            items = [c.decode("utf-8") if isinstance(c, bytes) else str(c)
                     for c in f.get_node(g, f"block{b}_items").read()]
            blocks.append((items, f"block{b}_values"))
        return columns, int(axis1.shape[0]), blocks


def h5_info(path, key=DEFAULT_KEY) -> dict:
    """{'format': 'table'|'fixed', 'nrows': int, 'columns': [...]} without loading the data."""
    info = _storer_info(path, key)
    if info["format"] == "fixed":
        cols, n, _ = _fixed_layout(path, key)
        info.update(nrows=n, columns=cols)
    elif info["columns"] is None:
        info["columns"] = list(pd.read_hdf(str(path), key=key, start=0, stop=1).columns)
    return info


def read_index(path, key=DEFAULT_KEY) -> pd.DatetimeIndex:
    """The full time index only (about 8 bytes per row: 316 MB for the 39.5 M-row file)."""
    info = _storer_info(path, key)
    if info["format"] == "table":
        with pd.HDFStore(str(path), mode="r") as store:
            idx = store.select_column(key, "index")
        return pd.DatetimeIndex(idx, name=None)
    import tables
    with tables.open_file(str(path), "r") as f:
        ax = f.get_node("/" + key).axis1
        kind = str(ax.attrs.kind)
        if not kind.startswith("datetime64"):
            raise ValueError(f"unsupported index kind {kind!r}")
        unit = kind[kind.index("[") + 1:-1] if "[" in kind else "ns"
        return pd.DatetimeIndex(ax.read().astype(np.int64, copy=False).view(f"datetime64[{unit}]"))


def read_rows(path, key=DEFAULT_KEY, start=0, stop=None, columns=None) -> pd.DataFrame:
    """Rows [start, stop) of a table- or fixed-format frame (all columns, or `columns`)."""
    info = _storer_info(path, key)
    if info["format"] == "table":
        return pd.read_hdf(str(path), key=key, start=start, stop=stop, columns=columns)
    import tables
    cols, n, blocks = _fixed_layout(path, key)
    stop = n if stop is None else min(int(stop), n)
    start = max(int(start), 0)
    with tables.open_file(str(path), "r") as f:
        g = f.get_node("/" + key)
        ax = g.axis1
        kind = str(ax.attrs.kind)
        unit = kind[kind.index("[") + 1:-1] if "[" in kind else "ns"
        idx = pd.DatetimeIndex(ax[start:stop].astype(np.int64, copy=False).view(f"datetime64[{unit}]"))
        data = {}
        for items, node in blocks:
            want = [i for i, name in enumerate(items) if columns is None or name in columns]
            if not want:
                continue
            arr = f.get_node(g, node)[start:stop, :]
            for i in want:
                data[items[i]] = np.ascontiguousarray(arr[:, i])
    order = [c for c in (columns if columns is not None else cols) if c in data]
    return pd.DataFrame({c: data[c] for c in order}, index=idx)


class TableWriter:
    """Append DataFrames to an HDF5 `table` (key DATA, zlib level 1, no per-column indexes), writing to
    '<path>.tmp' and renaming on a successful close, so a crash never leaves a half file under the final
    name. Refuses to replace an existing file unless overwrite=True.

        with TableWriter(path) as w:
            w.append(block)
    """

    def __init__(self, path, key=DEFAULT_KEY, overwrite=False):
        self.path = Path(path)
        self.key = key
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; pass overwrite=True / --overwrite to replace it.")
        self.tmp = Path(str(self.path) + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp.exists():
            self.tmp.unlink()
        self.store = pd.HDFStore(str(self.tmp), mode="w", complevel=COMPLEVEL, complib=COMPLIB)
        self.rows = 0
        self._cols = None

    def append(self, df: pd.DataFrame):
        if self._cols is None:
            self._cols = list(df.columns)
        elif list(df.columns) != self._cols:
            raise ValueError("TableWriter.append: columns differ from the first block")
        self.store.append(self.key, df, format="table", index=False)
        self.rows += len(df)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.store.close()
        if exc_type is None:
            os.replace(self.tmp, self.path)
        elif self.tmp.exists():
            self.tmp.unlink()
        return False


def write_fixed(df: pd.DataFrame, path, key=DEFAULT_KEY, overwrite=False):
    """Write a (small) frame as a fixed-format file, via '<path>.tmp' + rename."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True / --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        df.to_hdf(str(tmp), key=key, mode="w", complevel=COMPLEVEL, complib=COMPLIB)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
