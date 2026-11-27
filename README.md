# calliopy

Keller lab core code for data management, preprocessing and analysis: a Python port of calliope.

## Install

From a clone of this repository, with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

This creates `.venv` with calliopy (editable) and the dev tools. To use it in another environment:

```
pip install git+https://github.com/KellerLabFMI/calliopy
```

## Develop

```
uv run pytest        # tests
uv run ruff check .  # lint
```

The version is taken from git tags (e.g. tag `v0.1.0` gives version `0.1.0`).

## Porting progress

Coverage of calliope (MATLAB, revision r2003) by calliopy, counted in `.m` files and computed
from `docs/matlab_map.csv`. Last updated: 2026-11-26.

**Overall: 225 of 382 MATLAB files (58.9 %).**

| Folder | MATLAB files | Ported | Elsewhere | Deferred | Not ported | Not started | Coverage |
|---|---|---|---|---|---|---|---|
| `01_DataManagement` | 30 | 9 | 1 | 2 | 18 | 0 | 33.3 % |
| `02_ExpLog` | 21 | 14 | 2 | 3 | 2 | 0 | 76.2 % |
| `03_Network` | 2 | 1 | 0 | 0 | 1 | 0 | 50.0 % |
| `04_Calliope` | 21 | 10 | 0 | 9 | 2 | 0 | 47.6 % |
| `05_LoadData` | 32 | 25 | 0 | 3 | 4 | 0 | 78.1 % |
| `06_Library` | 21 | 8 | 3 | 2 | 8 | 0 | 52.4 % |
| `07_ViewStack` | 3 | 2 | 0 | 1 | 0 | 0 | 66.7 % |
| `08_RoiQuest` | 90 | 26 | 5 | 53 | 6 | 0 | 34.4 % |
| `09_Plugins` | 4 | 0 | 0 | 0 | 0 | 4 | 0.0 % |
| `10_Behavior` | 4 | 4 | 0 | 0 | 0 | 0 | 100.0 % |
| `11_TwoPhoton` | 8 | 5 | 0 | 3 | 0 | 0 | 62.5 % |
| `12_Widefield` | 40 | 26 | 3 | 7 | 4 | 0 | 72.5 % |
| `13_OII` | 5 | 4 | 0 | 0 | 1 | 0 | 80.0 % |
| `20_MetaData` | 10 | 4 | 0 | 5 | 1 | 0 | 40.0 % |
| `21_Analysis` | 45 | 41 | 1 | 2 | 1 | 0 | 93.3 % |
| `22_Plotting` | 25 | 10 | 9 | 0 | 6 | 0 | 76.0 % |
| `23_Prefs` | 4 | 0 | 0 | 0 | 4 | 0 | 0.0 % |
| `24_ProjMeta` | 15 | 12 | 0 | 2 | 1 | 0 | 80.0 % |
| repository root | 2 | 0 | 0 | 0 | 1 | 1 | 0.0 % |
| **Total** | 382 | 201 | 24 | 92 | 60 | 5 | 58.9 % |

- *Ported*: the computation of the file is available in calliopy. Parts that need ExpLog,
  the database or writing adata files may still be deferred; `docs/matlab_map.csv` says which.
- *Elsewhere*: covered by another calliopy function or by numpy/scipy.
- *Deferred*: waits for something calliopy does not have yet. *Not ported*: obsolete,
  duplicate or third-party files that will not be ported. *Not started*: no row in the map yet.
- Coverage = (ported + elsewhere) / MATLAB files. It counts files, not lines, and says
  nothing about verification against MATLAB output: see the `verified_against_r2003` column
  of the map.
