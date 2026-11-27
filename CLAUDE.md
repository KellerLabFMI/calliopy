# CLAUDE.md: porting calliope (MATLAB) to calliopy (Python)

calliopy is the Python port of calliope, the Keller lab MATLAB code base. Read
`docs/porting_guide.md` before porting anything: it holds the conventions for shapes, stored
values, return types, code layout and tests. This file adds the rules an agent must follow.
Where the two disagree, stop and ask.

## Reference version (MATLAB)

| | |
|---|---|
| Repository | `https://svn.code.sf.net/p/iris-scanning/calliope` |
| Repository UUID | `be57830b-97b9-400b-bf4f-2ef472044a1e` |
| Handover revision | **r2003** |
| Last commit in r2003 | 2026-11-20, `12_Widefield/wid_show_stack_drift.m` |

This revision is the MATLAB source for every port. Folder and file names in this file and in
`docs/matlab_map.csv` are relative to the repository root. The repository address above is the
only location that is written down: where a working copy sits on a machine is never recorded
(rule 1 below).

## Rules that are never broken

1. **No paths. Ever.** No absolute path, drive letter, network share, user name, machine name or
   lab folder name appears in code, tests, docstrings, comments, docs, examples or this file.
   A function gets every location as an argument. A MATLAB default such as `set_lab_paths`,
   a temp-data drive or a code folder becomes a required argument, never a Python default.
   Tests use `tmp_path` or the `CALLIOPY_TESTDATA` fixtures. Files shipped with the package are
   reached with `importlib.resources`.
2. **uv only.** `uv sync`, `uv run pytest`, `uv run ruff check .`, `uv add`. No pip, no conda,
   no `requirements.txt`, no hand edits of `uv.lock`. If uv cannot run in the session, say so
   and check the work with an existing interpreter, but do not write a second workflow down.
3. **No new dependency without the user's yes.** The runtime dependencies are numpy, scipy,
   matplotlib, h5py, tifffile and openpyxl (xlsx score sheets only, imported lazily). See
   "Dependencies" below for how to decide.
4. **The existing public API does not change.** Add names; do not rename, move or change the
   behaviour of what `calliopy.io` (or any released subpackage) already exports.
5. **Do not translate code the lab does not own.** Check the header of every `.m` file. Copies of
   MathWorks functions, File Exchange code, code under a non-commercial or share-alike licence
   and bundled third-party packages are not ported line by line into this MIT repository.
   Use the numpy/scipy equivalent, or implement the published algorithm from its paper, and
   record that in `docs/matlab_map.csv`. If neither is possible, ask.
6. **Never write "verified" for something that was not compared with MATLAB output.** An agent
   cannot run MATLAB. The user runs `tests/matlab/make_reference.m`; until a reference test
   passes, the function is `ported` with an empty `verified_against_r2003`.
7. **Do not commit, push or tag** unless the user asks.

## Dependencies

Decide in this order, and stop at the first that works:

1. The Python standard library (`json`, `csv`, `logging`, `pathlib`, `datetime`, `struct`).
2. A numpy or scipy function. Prefer it over hand-written code, even when the MATLAB original
   was hand-written (a manual convolution loop becomes `scipy.signal` or `scipy.ndimage`).
3. A private helper of a few dozen lines of numpy, with its own test, when numpy/scipy has no
   equivalent (for example `normxcorr2`).
4. Otherwise stop and ask the user, naming the MATLAB function, what it would take to write it
   with numpy/scipy, and the candidate package. Never run `uv add` on your own.

A package that does one thing (bootstrap resampling, one filter, one colormap) is never the
answer. pandas, scikit-image, scikit-learn, OpenCV and joblib are not dependencies: `readtable`
becomes `csv`/numpy, `parfor` becomes a loop or vectorised code. Plotting uses matplotlib only.
Interactive windows use tkinter (standard library) around a matplotlib canvas, as in
`calliopy.roiquest`. openpyxl is the one approved exception (xlsx reading/writing, imported
lazily inside the functions that need it). FISSA, CASCADE and rastermap are not dependencies:
the MATLAB wrappers around them are `not_ported` by user decision.

## How to port a folder

1. **List and triage every `.m` file** before writing code. Each file gets one status in
   `docs/matlab_map.csv`, with a one-line reason when it is not `ported`:
   - `ported`: computation that works on arrays or files.
   - `deferred`: needs something calliopy does not have yet (database writes, lookups by
     ExpID, lab site configuration such as data locations or project defaults). Port its
     computational core now if it has one, and defer only the rest. Keep the note neutral:
     no internal details of the lab infrastructure in this public repository.
   - `not_ported`: obsolete or duplicate versions (`*_BU`, `*NEW`, "OBSOLETE" in the header,
     `.asv` autosaves), rig-side scripts tied to one machine.
   - `elsewhere`: already covered by an existing calliopy function. Check `calliopy.io` first.
2. **Find what the folder leans on.** List the functions it calls in other calliope folders and
   who calls it. Port a needed outside function as a private helper and give it its own row.
   Some library files shadow MATLAB built-ins (`prctile`, `nanmean`): map those to numpy.
3. **Separate computation from everything else.** One MATLAB function usually mixes a lookup by
   ExpID, file loading, the computation, printing, plotting and saving. In Python:
   - the core takes arrays and returns values;
   - a thin wrapper takes file paths and calls `calliopy.io` readers and the core;
   - nothing looks up an ExpID and nothing prints. The core never writes files: what MATLAB
     would have saved is returned as a `MatStruct` with the MATLAB variable names;
   - writing into adata files is done only by separate, explicit save functions that take the
     adata path and the core's result and write MATLAB's variable names and layout through
     `calliopy.io` (`WritableAdata`, `save_adata`, `update_mat`, `delete_mat_variables`);
     interactive tools (RoiQuest, viewers) back the file up before their first write;
   - plots are separate functions that take the computed result and return a matplotlib
     `Figure`. Interactive input (`ginput`, `input`, dialogs) becomes an argument.
4. **Collapse copy-paste.** When MATLAB repeats a block per channel or per mode, write it once.
   The outputs must stay the same.
5. **Handle bugs openly.** A MATLAB bug that crashes or cannot be intended is fixed and listed
   under *Differences from MATLAB* in the docstring and in the `notes` column. A suspected bug
   that changes numbers for valid input is not fixed silently: keep MATLAB's result, and tell
   the user.
6. **Test, then record.** Add synthetic tests, a `tests/matlab/ref_<group>.m` section that saves
   the MATLAB output for the reference test, and the `matlab_map.csv` rows. Run
   `uv run pytest` and `uv run ruff check .`.

## Translation traps

Indices and layout:

- MATLAB is 1-based and column-major. Use `order="F"` for `reshape`, `ravel` and linear indices.
- Values that MATLAB stores in adata files or metafiles (ROI `indices`, frame numbers,
  `frame_times`) keep the MATLAB value, 1-based, also when Python computes them, so files made
  by either language are interchangeable.
- Every argument that selects or locates something (which ROIs, which frames, a crop rectangle)
  is 0-based, as are indices used inside a function. Two exceptions: options that `calliopy.io`
  already exposes do not change, and point coordinates of transforms follow the next item.
  Every index-valued argument and output states its base in the docstring.
- Image coordinates: MATLAB geometric functions (`imwarp`, `ginput`, `fitgeotform2d`) use
  (x, y) = (column, row), 1-based, pixel centres at integers. scipy.ndimage uses (row, column),
  0-based. Convert in one place and test with an asymmetric image.

Numbers:

- `round` rounds halves away from zero; `np.round` rounds halves to even.
- Integer types saturate and round in MATLAB; they wrap and truncate in numpy. Convert to float
  before arithmetic where MATLAB did (`single(...)`, `double(...)`), and keep `single` results
  as `float32`.
- `std` and `var` normalise by N-1 (`ddof=1`).
- Random numbers (`rand`, `randi`, `randperm`, `rng`) cannot be reproduced. Take a
  `numpy.random.Generator` argument and test properties, not values.
- Compare with MATLAB at `rtol=1e-12` for float64 and `1e-5` for float32, exactly for integers
  and indices. A looser tolerance needs a reason in the test.

Equivalents to start from (each use still needs a test):

| MATLAB | Python |
|---|---|
| `prctile(x, p)` | `np.percentile(x, p, method="hazen")` (`np.nanpercentile` when NaNs occur) |
| `nanmean`, `nanstd`, `nanmedian` | `np.nanmean`, `np.nanstd(ddof=1)`, `np.nanmedian` |
| `corr2(a, b)`, `corrcoef` | `np.corrcoef(a.ravel(), b.ravel())[0, 1]` |
| `conv2(a, k, 'same')` | `scipy.signal.convolve2d(a, k, mode="full")`, then slice from `k.shape // 2`; scipy's `"same"` is shifted by one sample for even-sized kernels |
| `filter2(k, a)`, `imfilter(a, k)` | correlation with zero padding: `conv2` with `k` rotated by 180 degrees, same caution for even-sized kernels |
| `imgaussfilt(a, s)` | `scipy.ndimage.gaussian_filter(a, s, mode="nearest", truncate=ceil(2*s)/s)` |
| `interp1(x, y, xq)` | `np.interp`, then NaN outside `[x[0], x[-1]]` (MATLAB does not clamp) |
| `ifft(Y, 'symmetric')` | `np.fft.ifft(Y).real` |
| `filtfilt(b, a, x)` | `scipy.signal.filtfilt(b, a, x, padlen=3*(max(len(a), len(b))-1))` |
| `xcorr`, `resample` | `scipy.signal.correlate`, `scipy.signal.resample_poly` |
| `ismember`, `unique`, `histcounts` | `np.isin`, `np.unique`, `np.histogram` |
| `ttest`, `ttest2`, `ranksum`, `bootstrp` | `scipy.stats` (`ttest_1samp`, `ttest_ind`, `ranksums`, `bootstrap`) |
| `imwarp`, `fitgeotform2d` | `scipy.ndimage.affine_transform` with a 3x3 matrix; see image coordinates above |
| `imresize` | no exact equivalent (antialiasing, pixel-centre convention): ask or document the difference |
| `imregtform` | no equivalent in numpy/scipy: ask |
| `jsondecode`, `readtable` | `json`, `csv` or `np.loadtxt` |
| `datenum`, `datestr`, `now` | `datetime.datetime` |

Language:

| MATLAB | Python |
|---|---|
| `inputParser`, `varargin`, `nargin`, `exist('x','var')` | keyword-only arguments with defaults, MATLAB names in snake_case |
| `disp`, `fprintf` | `logging` (never `print`) |
| `warning(...)` | `warnings.warn` |
| `error(...)`, `return` after a message, `status = -1` | raise an exception |
| `evalin`, `assignin`, `global`, `persistent` | return values and arguments |
| struct array, `[s.field]`, `s(end+1)` | object array of `MatStruct` shaped as in MATLAB; build a list, convert once |
| `[]` in a struct field | empty array of shape `(0, 0)` |
| several outputs | `NamedTuple` with the MATLAB output names |

## Layout

- One subpackage per kind of work (`calliopy.io`, `calliopy.widefield`, ...). Implementation
  files are private (`_name.py`), grouped by topic, never named after one function. The
  subpackage `__init__.py` re-exports the public names and lists them in `__all__`.
- Helpers that reproduce a MATLAB built-in live once, in `calliopy/_matlab.py`. Look there
  before writing one.
- Every public function has a numpy-style docstring that names the MATLAB function and folder
  it replaces, gives shapes and units, and lists *Differences from MATLAB*.

## Tests

- Synthetic tests always run and need no lab data: analytic cases, a cross-check against an
  independent scipy route, and the edge cases MATLAB handles in a particular way (empty input,
  NaN, a single frame, integer types).
- Reference tests compare with MATLAB output through the `matlab_ref` fixture and are skipped
  when `CALLIOPY_TESTDATA` is not set. A file that MATLAB already produced (a template, an
  adata file) is a valid reference: use it when the user provides one.
- Tests write only under `tmp_path`, use no network and no fixed seed-dependent values from
  MATLAB.

## Porting progress in README.md

`README.md` ends with the section `## Porting progress`. Bring it up to date as the last step
before every commit, also when the user makes the commit himself: hand the work over with the
section already updated.

- It is always the last section of `README.md`. Replace it as a whole; change nothing above it.
- `docs/matlab_map.csv` is the only source. Give every file you ported or triaged its row
  first, then recompute the table from the map. Never adjust a number by hand.
- One row per top-level folder of calliope r2003, one for the repository root, and a total.
- *MATLAB files* is the number of `.m` files in the folder at r2003, subfolders included.
  These numbers are fixed; they change only if the reference revision changes.
- *Ported*, *Elsewhere*, *Deferred* and *Not ported* count the map rows with that status, by
  the top-level folder of `matlab_folder`. Each `.m` file counts once; rows that are not `.m`
  files (autosaves) are left out. *Not started* is *MATLAB files* minus those four.
- *Coverage* = (ported + elsewhere) / MATLAB files, in percent with one decimal. The line
  above the table gives the same for all folders together.
- Write the date of the update. Keep the explanatory notes under the table.

## Working with sub-agents

Give each sub-agent files nobody else writes. Shared helpers (`_matlab.py`, anything two
tracks import) are written and tested first, before the parallel tracks start. Each track
reports the functions ported, deferred and not ported, every difference from MATLAB, and what
remains unverified.

## Done means

- every `.m` file of the folder has a row in `docs/matlab_map.csv`;
- the `## Porting progress` section of `README.md` is recomputed from the map;
- `uv run pytest` and `uv run ruff check .` pass, and the result is quoted in the report;
- the report lists what was not compared with MATLAB and every decision left to the user.
