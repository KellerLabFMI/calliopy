"""Tests for the joke ports: wtf (06_Library) and why_keller (08_RoiQuest/+utils)."""

from __future__ import annotations

import numpy as np
import pytest

from calliopy.analysis._jokes import WTF_ANSWERS, wtf
from calliopy.roiquest import DEFAULT_WHY_SUBJECTS, why_keller


def test_wtf_returns_each_of_17_answers():
    rng = np.random.default_rng(0)
    seen = {wtf(rng=rng) for _ in range(2000)}
    assert len(WTF_ANSWERS) == 17
    assert seen == set(WTF_ANSWERS)


def test_why_keller_shapes_and_capitalisation():
    rng = np.random.default_rng(1)
    for _ in range(500):
        s = why_keller(rng=rng)
        assert isinstance(s, str) and s
        assert s[0] == s[0].upper()
        assert s.endswith(".")


def test_why_keller_uses_given_names_only():
    rng = np.random.default_rng(2)
    texts = [why_keller(["Alice", "Bob"], rng=rng) for _ in range(500)]
    joined = " ".join(texts)
    assert "Alice" in joined and "Bob" in joined
    assert not any(n in joined for n in DEFAULT_WHY_SUBJECTS if n != "the PI")


def test_why_keller_special_answer_is_always_the_first_case():
    rng = np.random.default_rng(3)
    texts = {why_keller(rng=rng) for _ in range(2000)}
    assert "Because someone is still eating meat." in texts
    assert "It's trivial." not in texts and "Go and figure it out." not in texts


def test_why_keller_reproducible_with_generator():
    a = [why_keller(rng=np.random.default_rng(7)) for _ in range(3)]
    assert len(set(a)) == 1


def test_why_keller_rejects_empty_names():
    with pytest.raises(ValueError):
        why_keller([])
