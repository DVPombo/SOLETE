"""
Shared helper: prints a clearly-delimited, JSON-formatted block to the
console, so every script's results can be read by people and parsed by tools. Handles numpy/
pandas types that plain json.dumps chokes on.

Import this in every other script rather than ad-hoc print()'ing results,
so every script's output is consistently parseable regardless of which
one produced it.
"""
import json
import datetime

import numpy as np
import pandas as pd


def _sanitize(obj):
    """
    Recursively walks the payload BEFORE json.dumps ever sees it. This is
    necessary (not just belt-and-suspenders) because Python's json module
    handles native `float('nan')`/`float('inf')` itself -- emitting the
    literal (invalid-JSON) tokens NaN/Infinity -- WITHOUT ever calling a
    custom `default=` function, since it considers plain floats already
    "handled". A NaN anywhere in a payload (e.g. a stat computed on an
    all-empty column) would otherwise silently produce a block that looks
    like JSON but isn't, and would fail to parse. Sanitizing everything to
    plain Python types + None here, rather than relying on `default=`
    alone, closes that gap.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (isinstance(obj, float) and np.isnan(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def print_report(name: str, payload) -> None:
    """
    Prints:
        =====BEGIN SOLETE REPORT: <name>=====
        { ... valid JSON, NaN-free (nulls instead) ... }
        =====END SOLETE REPORT: <name>=====

    Copy everything between (and including) the BEGIN/END lines when
    reporting back -- multiple blocks from the same or different scripts
    can be pasted together in one message, the markers keep them separable.
    """
    clean_payload = _sanitize(payload)
    print()
    print(f"=====BEGIN SOLETE REPORT: {name}=====")
    print(json.dumps(clean_payload, indent=2, allow_nan=False))
    print(f"=====END SOLETE REPORT: {name}=====")
    print()
