# SOLETE

**Please take the latest available release on the right hand side.**

Author: **Daniel Vázquez Pombo** - Contact: daniel.vazquez.pombo@gmail.com<br/>
LinkedIn: https://www.linkedin.com/in/dvp/<br/>
ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo   
ORCID: https://orcid.org/0000-0001-5664-9421

See [CHANGELOG.md](CHANGELOG.md) for release history, including the v3.0 corrigendum.

**This repository is the one-stop home of the whole SOLETE project**: the cleaning and quality-control pipeline behind version 4 (v4) of the dataset, and the forecasting platform and benchmarks built on it. (The data files themselves live on figshare, see below.)

This repository used to be complementary material to its twin "Data in Brief" article [1], and a series of papers covering Solar PV power forecasting [2, 3, 4]. The objective is to increase the transparency of my work, which is one of the main limitations of Machine Learning in general.
However, as it sometimes happens, the project has grown life by itself and has now become a platform to experiment on time-series forecasting based on Machine Learning.
I included a number of functions that can be used by beginners to kickstart their projects with solar power, machine learning, forecasting, or simply python.

Long Live Open Science!

The papers were developed under the PhD thesis Operation and Planning of Isolated Hybrid Power Systems at the Technical University of Denmark (DTU).
Version v1.0 was released during the PhD thus, Copyright 2021 Technical University of Denmark.
Version v2.0 was released months after finalising my employment at DTU, therefore, Copyright belongs to me (yeah baby!).
Version v3.0 was motivated by the discovery of some nasty bugs. Thank you very much to Pierre Pinson for letting me know about his suspicion. 
Version v4.0 is motivated by the desire of facilitating access to data scientist. The community has spoken and I have done my best to address their concerns and more.

# What is where

| I want to… | Go to |
|---|---|
| **get the data** | figshare: <https://doi.org/10.11583/DTU.17040767> → unzip into [`data/`](data/README.md) |
| understand the columns and quality flags | [`dataset/docs/DATA_DICTIONARY.md`](dataset/docs/DATA_DICTIONARY.md), [`dataset/docs/QC_SCHEMA.md`](dataset/docs/QC_SCHEMA.md) |
| see how the raw data became version 4 | [`dataset/`](dataset/README.md): `pipeline/` (the code), `docs/CLEANING_DECISIONS.md` (every rule and why), `diagnostics/` (the investigations) |
| train forecasting models | [`solete/`](solete/) (the package) and [`scripts/quickstart/MLForecasting.py`](scripts/quickstart/MLForecasting.py) |
| reproduce or extend the benchmarks | [`benchmarks/`](benchmarks/BENCHMARKS.md) |
| learn by example | the notebooks in [`examples/`](examples/) (they run on a tiny sample, no additional download needed) |

```
SOLETE/
├── data/          <- put the figshare files here (git-ignored; see data/README.md)
│   ├── hdf5/
│   └── parquet/
├── dataset/       cleaning + quality-control pipeline, its documentation and diagnostics
├── solete/        the importable package: paths, io, qc, physics, preprocessing, modeling, metrics, ...
├── benchmarks/    baselines and benchmark scripts, fixed splits, stored results
├── scripts/       quickstart/RunMe.py, quickstart/MLForecasting.py, dataset inspection tools
├── examples/      notebooks + tiny sample files
├── matlab/  R/    loaders for MATLAB and R users
├── tests/  docs/
```

# Dependencies
The pinned dependency versions live in `requirements.txt`. Target interpreter: Python 3.13 (the newest Python that every pinned package currently ships wheels for). The dataset-cleaning scripts need less: see `dataset/requirements.txt`.

## Setup
```
pip install -r requirements.txt
pip install -e .          # optional: makes `import solete` work from any folder
```
Without `pip install -e .` everything still works: every script adds the repository root to the Python path itself, so you can press F5 in Spyder from wherever the script is.

# Where the code looks for the data

Everything goes through [`solete/paths.py`](solete/paths.py); no script depends on the folder you run it from.

1. Download the figshare files and unzip them into `data/`, keeping the `hdf5/` and `parquet/` sub-folders (the layout of the figshare upload — no renaming).
2. Check what the code sees: `python -m solete.paths`
3. Prefer a different location (external drive, cluster)? Set the environment variable `SOLETE_DATA_DIR`.

```python
from solete.paths import find_data_file
import pandas as pd
df = pd.read_parquet(find_data_file("60min", version="v4", fmt="parquet"))   # cleaned data with quality flags
```

# Examples for Beginners
New to SOLETE? The notebooks in `examples/` walk through the dataset on a small sample file, so you can get a feel for it without downloading the full dataset first. Each has an "Open in Colab" badge to run it straight in the browser.

- [`examples/01_dataset_overview.ipynb`](examples/01_dataset_overview.ipynb) — first look at SOLETE: loading the sample data and exploring its columns and structure.
- [`examples/02_data_quality.ipynb`](examples/02_data_quality.ipynb) — walks through the QC flag layer and the data-quality issues it catches.
- [`examples/03_pv_forecasting.ipynb`](examples/03_pv_forecasting.ipynb) — a small persistence-vs-Random-Forest forecasting demo for solar PV power.
- [`examples/04_wind_forecasting.ipynb`](examples/04_wind_forecasting.ipynb) — the same forecasting demo for wind power, including a data note on the turbine's near-zero output in this dataset.
- [`examples/05_hybrid_forecasting.ipynb`](examples/05_hybrid_forecasting.ipynb) — hybrid wind+solar forecasting (`P_hybrid[kW]`). Honestly scoped: this dataset's wind record is too sparse to demonstrate wind-solar complementarity, so the notebook documents the infrastructure and methodology (derived column, joint-vs-independent comparison, ramp-rate tooling) for reuse once better-populated wind data is available, rather than a positive complementarity claim. Needs the full v3 hourly file in `data/hdf5/`. See `benchmarks/BENCHMARKS.md`'s hybrid section and `KNOWN_ISSUES.md` #10.
  
# How to Use the Forecasting Platform
1. Put the SOLETE data in `data/` (see above). The first script works without it, on the small sample in `examples/`.
2. Open **scripts/quickstart/RunMe.py**. This allows you to load SOLETE and sneak a peek at its contents.
3. Open **scripts/quickstart/MLForecasting.py**. This allows you to configure Random Forest (RF), Support Vector Machine (SVM), and three kinds of Artificial Neuronal Networks: Convolutional Neuronal Network (CNN), Long-Short Term Memory (LSTM), and a Hybrid (CNN-LSTM).
   - The file itself contains notes explaining how to use it.
   - The main objective is to introduce the SOLETE dataset and help people learning basics of time series forecasting based on Machine Learning
   - You can basically replicate most of the methodology from [2, 3, 4] and build on top.
   - I included some error messages to debug what I expect are the most common errors when running stuff.
   - Trained models and result files are written to `outputs/` (git-ignored).
   - Let me know if you like it or what needs to be fixed.
4. Have Fun!

You can of course use your own dataset, you will have to adapt things here and there, but you will be able to reuse most of the code.

Note that the dataset includes a 1sec resolution version. The file is quite large, which might meant that you PC is not able to open it. Please consider only reading part of it if you really want to play with that resolution. Alternatively, drop it in an HPC and enjoy yourself. :D

### Reproducing the version 4 cleaning
See [`dataset/README.md`](dataset/README.md). In short, from the repository root:
```
pip install -r dataset/requirements.txt
python dataset/pipeline/build_release.py --raw SOLETE_Pombo_1sec.h5 --skip-existing    # all ten v4 files + checksums; --dry-run first
```

### Notes for _MATLAB_ and _R_ users ###
I have been reached out by several people complaining that hdf5 can't be imported in MATLAB. That is not true, they weren't doing properly. Nevertheless, worry not dear user. Your peers have asked and I answer:
1. Open the file **matlab/RunMe_matlab.m** in MATLAB and hit F5. That will import SOLETE as a _table_ (it finds `data/hdf5/` by itself).
2. Alternatively, you can run the Python scripts from MATLAB, which I find a bit weird... but hey! You do you baby!

3. I also included and _R_ script, same reason as for MATLAB.

*I coded this using 2021b, so anything newer should work, but I haven't actually checked with older versions.

# How to cite this:
Technically, you should cite the repository itself, however I don't get those citations captured where it matters, so please cite [1] like this:

@article{pombo2022solete,
  title={SOLETE, a 15-month long holistic dataset including: Meteorology, co-located wind and solar PV power from Denmark with various resolutions},
  author={Pombo, Daniel Vazquez and Gehrke, Oliver and Bindner, Henrik W},
  journal={Data in Brief},
  volume={42},
  pages={108046},
  year={2022},
  publisher={Elsevier}
}


Nontheless, here is the citation for the git itself:
D. V. Pombo, The SOLETE platform (March, 2023).doi:10.11583/DTU.17040626.URL https://data.dtu.dk/articles/software/TheSOLETEplatform/17040626

@article{SOLETE2021Code,
author = "Daniel Vazquez Pombo",
title = "{The SOLETE platform}",
year = "2023",
month = "Mar,",
url = "https://data.dtu.dk/articles/software/The_SOLETE_platform/17040626",
doi = "10.11583/DTU.17040626"
} 




# References
    [1] Pombo, D. V., Gehrke, O., & Bindner, H. W. (2022). SOLETE, a 15-month long holistic dataset including: Meteorology, co-located wind and solar PV power from Denmark with various resolutions. Data in Brief, 42, 108046.
        
    [2] Pombo, D. V., Bindner, H. W., Spataru, S. V., Sørensen, P. E., & Bacher, P. (2022). Increasing the accuracy of hourly multi-output solar power forecast with physics-informed machine learning. Sensors, 22(3), 749.
    
    [3] Pombo, D. V., Bacher, P., Ziras, C., Bindner, H. W., Spataru, S. V., & Sørensen, P. E. (2022). Benchmarking physics-informed machine learning-based short term PV-power forecasting tools. Energy Reports, 8, 6512-6520.
    
    [4] Pombo, D. V., Rincón, M. J., Bacher, P., Bindner, H. W., Spataru, S. V., & Sørensen, P. E. (2022). Assessing stacked physics-informed machine learning models for co-located wind–solar power forecasting. Sustainable Energy, Grids and Networks, 32, 100943.

# Corrigendum
Versions of the SOLETE Platform up to and including v2.3 contained two major bugs.
1. When spliting the data in training, validation, and testing sets. This was directly affecting accuracy. 
2. In the postprocessing of results, when calculating RMSE. This was affecting evaluation quality.

I can only apologize for these mistakes, which have been corrected in versions v3.0 and upwards. 