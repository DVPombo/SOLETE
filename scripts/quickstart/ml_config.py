# -*- coding: utf-8 -*-
"""
CLI/config-file loader for scripts/quickstart/MLForecasting.py.

Builds the Control_Var dict that used to be a hardcoded literal at the top of
MLForecasting.py. Precedence, lowest to highest:

    scripts/quickstart/configs/default_control_var.json  (shipped defaults)
        < --config path/to/custom.json                   (optional full/partial override)
            < explicit CLI flags                         (--resolution, --mltype, ...)

Calling get_control_var() with no arguments (e.g. from Spyder's "Run file", where
sys.argv is just the script path) reproduces the shipped defaults exactly -- the
CLI is opt-in, never required. See scripts/quickstart/configs/README.md for what
each Control_Var key means.
"""

import argparse
import copy
import json
from pathlib import Path

from solete.postprocess import error_msg

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_control_var.json"

# Sentinel so we can tell "flag not passed" apart from "flag passed with a falsy value".
_UNSET = argparse.SUPPRESS


def _read_json(path):
    try:
        # utf-8-sig tolerates a BOM, which Notepad/PowerShell commonly add when a
        # non-expert user hand-edits one of these JSON files on Windows.
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except FileNotFoundError:
        error_msg(key="config_file_not_found")
    except json.JSONDecodeError as exc:
        error_msg(key="invalid_config_json")


def load_default_control_var():
    """Reads the shipped default config (the single source of truth for defaults)."""
    return _read_json(DEFAULT_CONFIG_PATH)


def load_config_file(path):
    """Reads a user-supplied --config JSON file."""
    data = _read_json(path)
    if not isinstance(data, dict):
        error_msg(key="invalid_config_json")
    return data


def deep_merge(base, override):
    """Recursively merges override onto a copy of base. Dicts are merged key-by-key;
    anything else (including lists, e.g. PossibleFeatures/Neurons) is replaced wholesale."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Train/evaluate a SOLETE forecasting model. Run with no arguments to use "
                     "scripts/quickstart/configs/default_control_var.json unchanged (Spyder-safe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=None,
                     help="JSON file (same shape as configs/default_control_var.json) merged on "
                          "top of the shipped defaults before any CLI flags below are applied")
    ap.add_argument("--resolution", choices=["1sec", "1min", "5min", "60min"], default=_UNSET)
    ap.add_argument("--mltype", dest="MLtype", choices=["RF", "SVM", "LSTM", "CNN", "CNN_LSTM"],
                     default=_UNSET, help="which ML model to train/import")
    ap.add_argument("-H", "--horizon", type=int, default=_UNSET, dest="H",
                     help="forecast horizon length, in samples")
    ap.add_argument("--pre", "--lookback", type=int, default=_UNSET, dest="PRE",
                     help="number of previous samples used as predictors")
    ap.add_argument("--train-val-test", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                     default=_UNSET, help="train/val/test split percentages, must sum to 100")
    ap.add_argument("--scaler", dest="Scaler", choices=["MinMax01", "MinMax11", "Standard"],
                     default=_UNSET)
    ap.add_argument("--random-seed", type=int, default=_UNSET,
                     help="seeds Python/NumPy/Keras for reproducible ANN runs (omit for random)")

    train_group = ap.add_mutually_exclusive_group()
    train_group.add_argument("--train", dest="trainVSimport", action="store_true", default=_UNSET,
                              help="train a new model (Control_Var['trainVSimport'] = True)")
    train_group.add_argument("--import-model", dest="trainVSimport", action="store_false",
                              default=_UNSET, help="import a previously saved model instead of training")

    save_group = ap.add_mutually_exclusive_group()
    save_group.add_argument("--save-model", dest="saveMLmodel", action="store_true", default=_UNSET)
    save_group.add_argument("--no-save-model", dest="saveMLmodel", action="store_false", default=_UNSET)

    build_group = ap.add_mutually_exclusive_group()
    build_group.add_argument("--build", dest="SOLETE_builvsimport", action="store_const",
                              const="Build", default=_UNSET,
                              help="expand the raw dataset (Control_Var['SOLETE_builvsimport'] = 'Build')")
    build_group.add_argument("--import-solete", dest="SOLETE_builvsimport", action="store_const",
                              const="Import", default=_UNSET,
                              help="import a previously built-and-saved expansion")

    val_group = ap.add_mutually_exclusive_group()
    val_group.add_argument("--use-val-as-train", dest="RF_SVM_use_val_as_train",
                            action="store_true", default=_UNSET)
    val_group.add_argument("--no-use-val-as-train", dest="RF_SVM_use_val_as_train",
                            action="store_false", default=_UNSET)

    return ap


def apply_cli_overrides(control_var, args):
    """Applies only the flags the user actually passed (argparse.SUPPRESS means "not passed",
    so those never end up as attributes on args)."""
    control_var = copy.deepcopy(control_var)
    overrides = vars(args)
    if "train_val_test" in overrides:
        control_var["Train_Val_Test"] = list(overrides.pop("train_val_test"))
    overrides.pop("config", None)
    for key, value in overrides.items():
        control_var[key] = value
    return control_var


def _validate(control_var):
    if sum(control_var["Train_Val_Test"]) != 100:
        error_msg(key="invalid_train_val_test")
    return control_var


def get_control_var(argv=None):
    """Builds the final Control_Var dict: shipped defaults < --config file < explicit CLI flags."""
    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args(argv)

    control_var = load_default_control_var()
    if getattr(args, "config", None):
        control_var = deep_merge(control_var, load_config_file(args.config))
    control_var = apply_cli_overrides(control_var, args)

    return _validate(control_var)
