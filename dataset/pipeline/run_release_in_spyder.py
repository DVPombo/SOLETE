# -*- coding: utf-8 -*-
"""
run_release_in_spyder.py -- press Run in Spyder to build the whole SOLETE v4 release.

1. Edit the SETTINGS block below (paths and options), nothing else.
2. First run with DRY_RUN = True: it prints the plan, the disk estimate and refuses if the disks are too small.
3. Set DRY_RUN = False and run again. The build runs the stages as subprocesses of the same Python and prints
   everything; the final summary table is also saved as <DATA_DIR>/build_summary_v4.json.

If it stops (power cut, Ctrl+C, an error), fix the cause and run again: SKIP_EXISTING = True resumes at the stage level.
Equivalent terminal command: python dataset/pipeline/build_release.py --raw <RAW> --skip-existing ...
"""
# ----------------------------------------------------------------------------- SETTINGS (edit these)
DATA_DIR = ""
RAW = ""
SLICE_DAYS = 31                                      # 31 is about 1.8 GB peak; use 7 for a machine with little RAM
REBUILD_CHECK = "sample"                             # "sample" (default), "full" (second full build: do it once before publishing), "none"
KEEP_INTERMEDIATE = False                            # True keeps the scratch folder (several GB) after a successful build
SKIP_EXISTING = True                                 # resume: skip a stage whose outputs already exist
OVERWRITE = False                                    # True replaces existing outputs (needed to redo a stage)
DRY_RUN = True                                       # True: plan + disk estimate only, nothing is written
STAGES = "all"                                       # or a comma list, e.g. "manifest" or "verify,manifest" (stages: original, clean, resample, expand, parquet, verify, manifest)
# -----------------------------------------------------------------------------------------------------
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))             # repo root

if DATA_DIR:
    os.environ["SOLETE_DATA_DIR"] = DATA_DIR         # must be set BEFORE solete.paths is imported
else:
    os.environ.pop("SOLETE_DATA_DIR", None)          # "" = the repo's data/ folder; Spyder's kernel would otherwise keep an earlier value
for name in [m for m in sys.modules if m == "solete" or m.startswith("solete.") or m == "build_release"]:
    del sys.modules[name]                            # Spyder keeps modules between runs: drop stale paths

import build_release  # noqa: E402

args = ["--slice-days", str(SLICE_DAYS), "--rebuild-check", REBUILD_CHECK, "--stages", STAGES]
if RAW:
    args += ["--raw", RAW]
for flag, on in (("--keep-intermediate", KEEP_INTERMEDIATE), ("--skip-existing", SKIP_EXISTING),
                 ("--overwrite", OVERWRITE), ("--dry-run", DRY_RUN)):
    if on:
        args.append(flag)
try:
    build_release.main(args)
except SystemExit as e:                               # build_release exits with 0 / 2 (refused) / 3 (failed)
    print(f"\nbuild_release finished with exit code {e.code}"
          + ("  <- NOT successful: read the messages above" if e.code not in (0, None) else ""))
