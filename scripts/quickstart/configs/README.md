# `scripts/quickstart/configs/`

`default_control_var.json` is the single source of truth for `MLForecasting.py`'s default
configuration (what used to be the hardcoded `Control_Var` dict literal + the five per-model
hyperparameter dicts). Running `scripts/quickstart/MLForecasting.py` with no arguments (including
from Spyder's "Run file") loads this file unchanged, so default behavior is identical to before
the CLI/config loader existed.

You can configure a run in three equivalent ways (see `python scripts/quickstart/MLForecasting.py -h`):
1. Edit `default_control_var.json` directly -- same affordance as editing the old dict literal.
2. Copy it, edit the copy, and pass `--config path/to/your_copy.json` -- only the keys you put in
   the copy are read; anything you want to keep must be in the copy (the config file is merged
   on top of `default_control_var.json`, not deep-merged with your intentions guessed).
3. Pass one of the top-level CLI flags for the handful of scalar options a newcomer is most
   likely to want to change (`--resolution`, `--mltype`, `-H/--horizon`, `--pre/--lookback`,
   `--train-val-test`, `--scaler`, `--train/--import-model`, `--save-model/--no-save-model`,
   `--random-seed`, `--build/--import-solete`, `--use-val-as-train/--no-use-val-as-train`).
   CLI flags win over `--config`, which wins over the shipped defaults.

## Field reference

Top-level:
- `resolution`: `1sec`, `1min`, `5min`, or `60min`. Each has its own expanded-dataset cache file.
- `SOLETE_builvsimport`: `Build` expands the raw dataset (slow, once per resolution+feature set);
  `Import` loads a previously built-and-saved expansion.
- `SOLETE_save`: save the expansion to disk after building it (only used when `Build`).
- `trainVSimport`: `true` trains a new ML model; `false` imports a previously saved one.
- `saveMLmodel`: save the trained model to `outputs/` (only used when `trainVSimport` is `true`).
- `Train_Val_Test`: `[train, val, test]` percentages of the data, must sum to 100.
- `RF_SVM_use_val_as_train`: RF/SVM don't early-stop on a validation set the way the ANN models
  do. `true` trains RF/SVM on train+val (e.g. 90%), only holding out the test split -- more data,
  closer to how you'd deploy them, but then they see more training data than LSTM/CNN/CNN_LSTM.
  `false` trains RF/SVM on train only (e.g. 70%), same split boundaries as the ANN models, useful
  for comparing model types on strictly identical training data.
- `random_seed`: `int` or `null`. `null` (default) is fully random each run, same as before this
  option existed. An int (e.g. `42`) seeds Python/NumPy/Keras for reproducible ANN weight
  init/shuffling. RF's own `random_state` below is independent of this. Not a GPU-determinism
  guarantee -- some backend ops stay nondeterministic regardless of seed.
- `Scaler`: `MinMax01`, `MinMax11`, or `Standard` -- see `solete/preprocessing.py:PreProcessDataset`.
- `IntrinsicFeature`: the column to be predicted.
- `PossibleFeatures`: ordered list of feature columns fed to the model -- order matters, it is the
  literal column order passed to `.fit`/`.predict`. `TempModule_RP` is deliberately left out by
  default: it requires a slow thermodynamic model run during expansion.
- `MLtype`: `RF`, `SVM`, `LSTM`, `CNN`, or `CNN_LSTM`.
- `H`: forecast horizon length, in samples.
- `PRE`: number of previous samples used as predictors (lookback).

Per-model dicts (`RF`, `SVM`, `LSTM`, `CNN`, `CNN_LSTM`): hyperparameters for that model type. Be
careful with the ANN ones (LSTM/CNN/CNN_LSTM): `Neurons`/`Dense` are lists describing actual
network topology (e.g. `Neurons: [15, 15, 15]` makes three layers of 15 neurons each), not scalar
knobs -- there's no CLI flag for these, edit the JSON directly. `EarlyStopping.enabled` stops
training once `patience` epochs pass with no `val_loss` improvement (instead of always running all
`epo_num` epochs); `restore_best_weights` rolls back to the best epoch's weights when stopping,
instead of keeping the last (possibly already-overfit) ones.
