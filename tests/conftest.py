"""Shared test fixtures.

Tests against real lab files and MATLAB outputs need a test-data folder, set with
the environment variable CALLIOPY_TESTDATA (kept outside this public repository):

    CALLIOPY_TESTDATA/
        samples/     one small file per format (.ini, .lvd, .bin, ...)
        reference/   outputs of tests/matlab/make_reference.m at calliope r2003

Without it, those tests are skipped; tests on synthetic files always run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def testdata() -> Path:
    root = os.environ.get("CALLIOPY_TESTDATA")
    if not root or not Path(root).is_dir():
        pytest.skip("CALLIOPY_TESTDATA is not set")
    return Path(root)


@pytest.fixture(scope="session")
def sample_file(testdata):
    """sample_file(".lvd") -> first sample file with that extension, or skip."""

    def find(ext: str) -> Path:
        hits = sorted((testdata / "samples").glob(f"*{ext}"))
        if not hits:
            pytest.skip(f"no {ext} sample in {testdata / 'samples'}")
        return hits[0]

    return find


@pytest.fixture(scope="session")
def matlab_ref(testdata):
    """matlab_ref("load_lvd") -> the MATLAB r2003 output saved by make_reference.m, or skip."""
    from calliopy.io import load_mat

    def get(name: str):
        path = testdata / "reference" / f"ref_{name}.mat"
        if not path.is_file():
            pytest.skip(f"no MATLAB reference {path.name}")
        return load_mat(path)

    return get
