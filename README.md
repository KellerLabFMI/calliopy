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
