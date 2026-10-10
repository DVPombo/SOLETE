# -*- coding: utf-8 -*-
"""
expansion_checks.py -- two measurements for solete/expansion.py.

    python scripts/expansion_checks.py memory [--start 2019-06-01 --days 30] [--workdir DIR]
        Peak memory of expand_physical on a month of 1-second rows, measured in a fresh process
        (VmHWM from /proc/self/status) after the table has been loaded from disk, so the figure is
        not polluted by the synthetic generator. Compares chunk sizes.

    python scripts/expansion_checks.py effect [--days 7] [--input FILE.h5 --key DATA]
        The resolution effect (docs: dataset/docs/METHODOLOGY.md): for each model column, the
        difference between  expand(resampled inputs)  and  resample(expand(1 s inputs))  at
        1 min / 5 min / 60 min, and how often the substitution flag disagrees.
        Without --input the 1-second table is SYNTHETIC (solete/synthetic.py): the numbers then
        describe the generator, not the dataset. With --input FILE (a cleaned 1-second file or a slice
        of one, v4 columns) the same table is produced from real data.

Prints a Markdown table and a JSON block. Never writes into data/.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: `import solete` from any cwd / Spyder

import numpy as np
import pandas as pd

from solete.expansion import compute_physical, expand_physical
from solete.params import import_PV_WT_data
from solete.qc_codes import QC_MODEL_SUBSTITUTED
from solete.synthetic import synthetic_solete

MEASURED = ['TEMPERATURE[degC]', 'HUMIDITY[%]', 'WIND_SPEED[m1s]', 'WIND_DIR[deg]', 'GHI[kW1m2]',
            'POA Irr[kW1m2]', 'P_Gaia[kW]', 'P_Solar[kW]', 'Pressure[mbar]']
MODEL_FLOAT = ['Pac', 'Pdc', 'TempModule', 'TempCell', 'P_Solar_clean[kW]', 'P_hybrid[kW]']


def _vmhwm_mb():
    for line in open('/proc/self/status'):
        if line.startswith(('VmHWM', 'VmRSS')):
            yield line.split(':')[0], int(line.split()[1]) / 1024.0


def _rss():
    return dict(_vmhwm_mb())


# ---------------------------------------------------------------------------
def cmd_memory(a):
    work = Path(a.workdir or tempfile.mkdtemp(prefix='solete_mem_'))
    work.mkdir(parents=True, exist_ok=True)
    f = work / 'month_1s.h5'
    if not f.exists():
        print(f"generating a synthetic {a.days}-day 1-second table -> {f}")
        synthetic_solete(a.start, periods=a.days * 86400).to_hdf(f, key='DATA', mode='w')
    out = []
    for chunk in a.chunks:
        r = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), 'memory-run', '--file', str(f),
                            '--chunk', str(chunk)], capture_output=True, text=True, check=True)
        out.append(json.loads(r.stdout.strip().splitlines()[-1]))
    print("\n| chunk_rows | rows | RSS after load (MB) | peak RSS (MB) | peak above loaded table (MB) | seconds |")
    print("|---|---|---|---|---|---|")
    for o in out:
        print(f"| {o['chunk_rows']:,} | {o['rows']:,} | {o['rss_loaded_mb']:.0f} | {o['peak_mb']:.0f} | "
              f"{o['peak_mb'] - o['rss_loaded_mb']:.0f} | {o['seconds']:.1f} |")
    print(json.dumps(out))


def cmd_memory_run(a):
    df = pd.read_hdf(a.file, key='DATA')
    base = _rss()
    PV, _ = import_PV_WT_data()
    t = time.time()
    expand_physical(df, PV, chunk_rows=a.chunk)
    dt = time.time() - t
    peak = _rss()
    print(json.dumps({'chunk_rows': a.chunk, 'rows': len(df), 'rss_loaded_mb': base['VmRSS'],
                      'peak_mb': peak['VmHWM'], 'seconds': dt, 'columns_added': len(df.columns)}))


# ---------------------------------------------------------------------------
def _flag_stats(flag_res, flag_1s_frac):
    """flag_res: bool array computed at the coarse step; flag_1s_frac: fraction of 1 s rows flagged in each bucket."""
    maj = flag_1s_frac >= 0.5
    anyf = flag_1s_frac > 0
    return {
        'flagged_rate_1s_rows': float(np.nanmean(flag_1s_frac)),
        'flagged_rate_at_resolution': float(flag_res.mean()),
        'disagree_vs_majority_of_1s': float((flag_res != maj).mean()),
        'disagree_vs_any_1s': float((flag_res != anyf).mean()),
    }


def effect_table(df1s, rules=('1min', '5min', '60min')):
    PV, _ = import_PV_WT_data()
    phys1 = compute_physical(df1s, PV)
    rows, flags = [], {}
    for rule in rules:
        res = df1s[MEASURED].resample(rule, label='left', closed='left').mean()          # measured: from 1 s
        direct = compute_physical(res.dropna(how='any'), PV)                                # model: at this resolution
        res = res.loc[direct.index]
        agg = pd.concat([phys1[MODEL_FLOAT], (phys1[['P_Solar[kW]_qc']] == QC_MODEL_SUBSTITUTED).astype(float).rename(columns={'P_Solar[kW]_qc': 'subst'})], axis=1) \
            .resample(rule, label='left', closed='left').mean().loc[direct.index]
        for col in MODEL_FLOAT:
            d = direct[col].to_numpy() - agg[col].to_numpy()
            ref_mean = float(np.nanmean(np.abs(agg[col].to_numpy())))
            rows.append({'resolution': rule, 'column': col, 'mean_resampled_1s_model': float(np.nanmean(agg[col])),
                         'mean_model_at_resolution': float(np.nanmean(direct[col])),
                         'mean_diff': float(np.nanmean(d)), 'mean_abs_diff': float(np.nanmean(np.abs(d))),
                         'rel_mean_abs_diff_pct': 100 * float(np.nanmean(np.abs(d))) / ref_mean if ref_mean else float('nan'),
                         'max_abs_diff': float(np.nanmax(np.abs(d)))})
        flags[rule] = _flag_stats((direct['P_Solar[kW]_qc'] == QC_MODEL_SUBSTITUTED).to_numpy(),
                                  agg['subst'].to_numpy())
        flags[rule]['n_buckets'] = int(len(direct))
    return pd.DataFrame(rows), flags


def cmd_effect(a):
    if a.input:
        df = pd.read_hdf(a.input, key=a.key, start=a.start_row, stop=a.stop_row)
        source = f"REAL file {a.input} rows {a.start_row}:{a.stop_row}"
    else:
        df = synthetic_solete(a.start, periods=a.days * 86400, seed=a.seed)
        source = f"SYNTHETIC ({a.days} days of 1 s, seed {a.seed}) -- describes the generator, not the dataset"
    df = df.dropna(subset=MEASURED)      # model columns need all inputs; a gap would only confuse the comparison
    print(f"source: {source}; {len(df):,} one-second rows")
    table, flags = effect_table(df)
    pd.set_option('display.width', 200)
    print("\n### Model columns: expand(resampled inputs) vs resample(expand(1 s inputs))\n")
    print("| resolution | column | mean of resampled 1 s model | mean of model at resolution | mean diff | mean abs diff | rel. mean abs diff % | max abs diff |")
    print("|---|---|---|---|---|---|---|---|")
    for r in table.itertuples():
        print(f"| {r.resolution} | `{r.column}` | {r.mean_resampled_1s_model:.4f} | {r.mean_model_at_resolution:.4f} | "
              f"{r.mean_diff:+.4f} | {r.mean_abs_diff:.4f} | {r.rel_mean_abs_diff_pct:.2f} | {r.max_abs_diff:.3f} |")
    print("\n### Substitution flag (code 6: Pac >= 1.5 * P_Solar and Pac > 0)\n")
    print("| resolution | buckets | flagged at 1 s (share of 1 s rows) | flagged at resolution | disagrees w/ majority-of-1 s | disagrees w/ any-1 s |")
    print("|---|---|---|---|---|---|")
    for rule, s in flags.items():
        pc = lambda k: f"{100*s[k]:.2f}%"
        print(f"| {rule} | {s['n_buckets']:,} | {pc('flagged_rate_1s_rows')} | {pc('flagged_rate_at_resolution')} | "
              f"{pc('disagree_vs_majority_of_1s')} | {pc('disagree_vs_any_1s')} |")
    print("\n" + json.dumps({'source': source, 'table': table.to_dict('records'), 'flags': flags}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('memory')
    m.add_argument('--start', default='2019-06-01'); m.add_argument('--days', type=int, default=30)
    m.add_argument('--workdir', default=None)
    m.add_argument('--chunks', type=int, nargs='+', default=[250_000, 1_000_000, 0])
    m.set_defaults(fn=cmd_memory)
    r = sub.add_parser('memory-run'); r.add_argument('--file'); r.add_argument('--chunk', type=int, default=0)
    r.set_defaults(fn=cmd_memory_run)
    e = sub.add_parser('effect')
    e.add_argument('--start', default='2019-06-01'); e.add_argument('--days', type=int, default=7)
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--input', default=None); e.add_argument('--key', default='DATA')
    e.add_argument('--start-row', type=int, default=None); e.add_argument('--stop-row', type=int, default=None)
    e.set_defaults(fn=cmd_effect)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
