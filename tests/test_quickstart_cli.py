# -*- coding: utf-8 -*-
"""
Unit tests for scripts/quickstart/ml_config.py -- the CLI/config-file loader
that builds MLForecasting.py's Control_Var dict.

Pure dict-building logic, no data files, no keras/tensorflow import needed.
Run with: pytest tests/test_quickstart_cli.py -v  (from the repo root)
"""
import json
import sys
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "quickstart"))

import ml_config


def test_no_args_matches_shipped_default_json():
    """Running with no CLI args (e.g. Spyder's "Run file", where sys.argv has no flags)
    must reproduce scripts/quickstart/configs/default_control_var.json exactly."""
    with open(ml_config.DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        expected = json.load(fh)

    assert ml_config.get_control_var([]) == expected


def test_cli_flag_overrides_default():
    control = ml_config.get_control_var(["--mltype", "RF", "--horizon", "7", "--no-save-model"])
    assert control["MLtype"] == "RF"
    assert control["H"] == 7
    assert control["saveMLmodel"] is False
    # Untouched keys/nested dicts stay at their default value.
    assert control["PRE"] == 5
    assert control["LSTM"]["epo_num"] == 1000


def test_config_file_overrides_default_and_merges_nested_dicts(tmp_path):
    config_path = tmp_path / "custom.json"
    config_path.write_text(json.dumps({"H": 20, "LSTM": {"Neurons": [5, 5]}}), encoding="utf-8")

    control = ml_config.get_control_var(["--config", str(config_path)])
    assert control["H"] == 20
    assert control["LSTM"]["Neurons"] == [5, 5]
    # Keys not present in the user config file keep the shipped default.
    assert control["LSTM"]["epo_num"] == 1000
    assert control["MLtype"] == "CNN"


def test_cli_flag_wins_over_config_file(tmp_path):
    config_path = tmp_path / "custom.json"
    config_path.write_text(json.dumps({"H": 20}), encoding="utf-8")

    control = ml_config.get_control_var(["--config", str(config_path), "--horizon", "99"])
    assert control["H"] == 99


def test_invalid_train_val_test_exits():
    with pytest.raises(SystemExit):
        ml_config.get_control_var(["--train-val-test", "50", "20", "10"])


def test_missing_config_file_exits():
    with pytest.raises(SystemExit):
        ml_config.get_control_var(["--config", "does_not_exist.json"])


def test_malformed_config_json_exits(tmp_path):
    config_path = tmp_path / "broken.json"
    config_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SystemExit):
        ml_config.get_control_var(["--config", str(config_path)])
