# Porting calliope (MATLAB) to calliopy

Reference version: **calliope r2003** (`svn checkout -r 2003 https://svn.code.sf.net/p/iris-scanning/calliope`).
Every ported function is checked against what r2003 returns. `docs/matlab_map.csv` lists every
calliope function touched so far, with its Python name and status
(`ported`, `planned`, `deferred`, `not_ported`, `elsewhere`).

## Conventions

1. **Shapes as in MATLAB.** Arrays keep MATLAB's orientation, so ported analysis code lines up
   with the MATLAB original: `act` is `[nUnits x nFrames]`, `rd` is `[nLayers x nExpIDs]`,
   a 1xN vector stays `(1, N)`. Exception: 1x1 numbers from MAT-files become numpy scalars.
2. **Stored values are never changed.** MATLAB index fields (ROI `indices`, frame numbers) stay
   1-based. Converting them is the job of a helper, never of a loader.
3. **Readers take a file path.** Lookups by ExpID (`load_exp`, `get_data_path`, ...) come later,
   with `calliopy.config` and ExpLog access.
4. **Return values instead of side effects.** No writing into the caller's workspace (`ws`),
   no `disp`/progress bars (use `logging`), no `status = -1`: errors raise exceptions.
5. **Several MATLAB outputs -> a `NamedTuple`** with the MATLAB output names in snake_case, so
   `data, scan_rate, *_ = load_lvd(path)` and `load_lvd(path).data` both work.
6. **Large raw data is memory-mapped** (`numpy.memmap`) unless the caller asks for a copy.
7. **Dates are `datetime.datetime`**, not MATLAB datenums.
8. **MATLAB quirks are fixed, not copied**, when they are bugs (e.g. `readini` evaluating
   `2024-01-05` as arithmetic, or silently dropping a line it cannot parse). Every such
   difference is listed in the function's docstring under *Differences from MATLAB* and in the
   `notes` column of `matlab_map.csv`.

## Translating MATLAB code

| MATLAB | numpy |
|---|---|
| `x(1)`, `x(end)` | `x[0]`, `x[-1]` |
| `x(a:b)` (inclusive) | `x[a-1:b]` |
| `reshape(x, m, n)` (column-major) | `x.reshape(m, n, order="F")` |
| `x(:)` | `x.ravel(order="F")` |
| `fread(fid, n, 'int16')` | `np.fromfile(f, dtype="<i2", count=n)` (check the byte order MATLAB used in `fopen`) |
| `size(x, 1)` | `x.shape[0]` |
| `isempty(x)` | `x.size == 0` (arrays), `not x` (str, list) |

Linear indices stored by MATLAB are column-major and 1-based:
`np.unravel_index(idx - 1, shape, order="F")`.

## Code layout

- Public functions are imported from the subpackage: `from calliopy.io import load_lvd`.
- Implementation files are private and grouped by kind of data (`io/_mat.py`, `io/_aux.py`,
  `io/_imaging.py`, `io/_eeg.py`). A file is never named after a function it holds.
- `io/__init__.py` re-exports the public names and lists them in `__all__`.
- numpy-style docstrings; `ruff check` must pass.
- Dependencies are added with `uv add` (heavy ones as optional extras), never `pip install`.

## Tests

- **Synthetic files** (always run, also in CI): tests build small files in the format the
  MATLAB reader expects. `tests/mat73_writer.py` writes files with the MATLAB v7.3 layout.
- **Real files and MATLAB outputs** (run when `CALLIOPY_TESTDATA` is set; kept outside this
  public repository):

  ```
  CALLIOPY_TESTDATA/
      samples/     one small file per format
      reference/   outputs of tests/matlab/make_reference.m at r2003
  ```

  Generate the references once in MATLAB, with calliope r2003 on the path:
  `make_reference('<CALLIOPY_TESTDATA>/samples', '<CALLIOPY_TESTDATA>/reference')`.
  Each reader group has its own `tests/matlab/ref_<group>.m`; outputs are saved with
  `save_ref` as `ref_<reader>.mat` holding `input_file`, `out1`, `out2`, ... in MATLAB's
  output order. Tests load them with the `matlab_ref` fixture.
- Run everything with `uv run pytest`; a reader counts as ported when its reference test passes,
  and `verified_against_r2003` in `matlab_map.csv` is then set to the date.

## Still to verify against real MATLAB files

- How MATLAB stores the dimensions of empty arrays (order of the stored size vector).
- Struct arrays with one element, and empty struct arrays.
