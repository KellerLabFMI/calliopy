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
from `docs/matlab_map.csv`. Last updated: 2026-11-25.

**Overall: 71 of 382 MATLAB files (18.6 %).**

| Folder | MATLAB files | Ported | Elsewhere | Deferred | Not ported | Not started | Coverage |
|---|---|---|---|---|---|---|---|
| `01_DataManagement` | 30 | 4 | 0 | 1 | 0 | 25 | 13.3 % |
| `02_ExpLog` | 21 | 2 | 1 | 0 | 0 | 18 | 14.3 % |
| `03_Network` | 2 | 0 | 0 | 0 | 0 | 2 | 0.0 % |
| `04_Calliope` | 21 | 1 | 0 | 1 | 0 | 19 | 4.8 % |
| `05_LoadData` | 32 | 17 | 0 | 2 | 0 | 13 | 53.1 % |
| `06_Library` | 21 | 3 | 1 | 0 | 0 | 17 | 19.0 % |
| `07_ViewStack` | 3 | 2 | 0 | 1 | 0 | 0 | 66.7 % |
| `08_RoiQuest` | 90 | 1 | 0 | 0 | 0 | 89 | 1.1 % |
| `09_Plugins` | 4 | 0 | 0 | 0 | 0 | 4 | 0.0 % |
| `10_Behavior` | 4 | 0 | 0 | 0 | 0 | 4 | 0.0 % |
| `11_TwoPhoton` | 8 | 0 | 0 | 0 | 0 | 8 | 0.0 % |
| `12_Widefield` | 40 | 26 | 3 | 7 | 4 | 0 | 72.5 % |
| `13_OII` | 5 | 0 | 0 | 0 | 0 | 5 | 0.0 % |
| `20_MetaData` | 10 | 1 | 0 | 0 | 0 | 9 | 10.0 % |
| `21_Analysis` | 45 | 4 | 0 | 0 | 0 | 41 | 8.9 % |
| `22_Plotting` | 25 | 5 | 0 | 0 | 0 | 20 | 20.0 % |
| `23_Prefs` | 4 | 0 | 0 | 0 | 0 | 4 | 0.0 % |
| `24_ProjMeta` | 15 | 0 | 0 | 0 | 0 | 15 | 0.0 % |
| repository root | 2 | 0 | 0 | 0 | 0 | 2 | 0.0 % |
| **Total** | 382 | 66 | 5 | 12 | 4 | 295 | 18.6 % |

- *Ported*: the computation of the file is available in calliopy. Parts that need ExpLog,
  the database or writing adata files may still be deferred; `docs/matlab_map.csv` says which.
- *Elsewhere*: covered by another calliopy function or by numpy/scipy.
- *Deferred*: waits for something calliopy does not have yet. *Not ported*: obsolete,
  duplicate or third-party files that will not be ported. *Not started*: no row in the map yet.
- Coverage = (ported + elsewhere) / MATLAB files. It counts files, not lines, and says
  nothing about verification against MATLAB output: see the `verified_against_r2003` column
  of the map.
