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

**Overall: 52 of 382 MATLAB files (13.6 %).**

| Folder | MATLAB files | Ported | Elsewhere | Deferred | Not ported | Not started | Coverage |
|---|---|---|---|---|---|---|---|
| `01_DataManagement` | 30 | 2 | 0 | 0 | 0 | 28 | 6.7 % |
| `02_ExpLog` | 21 | 0 | 0 | 0 | 0 | 21 | 0.0 % |
| `03_Network` | 2 | 0 | 0 | 0 | 0 | 2 | 0.0 % |
| `04_Calliope` | 21 | 0 | 0 | 0 | 0 | 21 | 0.0 % |
| `05_LoadData` | 32 | 15 | 0 | 1 | 0 | 16 | 46.9 % |
| `06_Library` | 21 | 2 | 1 | 0 | 0 | 18 | 14.3 % |
| `07_ViewStack` | 3 | 0 | 0 | 0 | 0 | 3 | 0.0 % |
| `08_RoiQuest` | 90 | 1 | 0 | 0 | 0 | 89 | 1.1 % |
| `09_Plugins` | 4 | 0 | 0 | 0 | 0 | 4 | 0.0 % |
| `10_Behavior` | 4 | 0 | 0 | 0 | 0 | 4 | 0.0 % |
| `11_TwoPhoton` | 8 | 0 | 0 | 0 | 0 | 8 | 0.0 % |
| `12_Widefield` | 40 | 26 | 3 | 7 | 4 | 0 | 72.5 % |
| `13_OII` | 5 | 0 | 0 | 0 | 0 | 5 | 0.0 % |
| `20_MetaData` | 10 | 0 | 0 | 0 | 0 | 10 | 0.0 % |
| `21_Analysis` | 45 | 1 | 0 | 0 | 0 | 44 | 2.2 % |
| `22_Plotting` | 25 | 1 | 0 | 0 | 0 | 24 | 4.0 % |
| `23_Prefs` | 4 | 0 | 0 | 0 | 0 | 4 | 0.0 % |
| `24_ProjMeta` | 15 | 0 | 0 | 0 | 0 | 15 | 0.0 % |
| repository root | 2 | 0 | 0 | 0 | 0 | 2 | 0.0 % |
| **Total** | 382 | 48 | 4 | 8 | 4 | 318 | 13.6 % |

- *Ported*: the computation of the file is available in calliopy. Parts that need ExpLog,
  the database or writing adata files may still be deferred; `docs/matlab_map.csv` says which.
- *Elsewhere*: covered by another calliopy function or by numpy/scipy.
- *Deferred*: waits for something calliopy does not have yet. *Not ported*: obsolete,
  duplicate or third-party files that will not be ported. *Not started*: no row in the map yet.
- Coverage = (ported + elsewhere) / MATLAB files. It counts files, not lines, and says
  nothing about verification against MATLAB output: see the `verified_against_r2003` column
  of the map.
