"""Tests for ``core.llm.scorecard.cli`` — the user-facing CLI for
inspecting and maintaining the scorecard.

These exercise the rendering + filter logic over a seeded scorecard.
They don't shell out — we drive the parsed argparse Namespace
directly via ``cmd_*`` for fast feedback. End-to-end shim
invocation is covered by a small smoke test below.
"""

from __future__ import annotations

import datetime as _dt
import io
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.llm.scorecard import EventType, ModelScorecard
from core.llm.scorecard import cli as cli_mod


# ---------------------------------------------------------------------------
# Fixture: a richly-populated scorecard
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_scorecard(tmp_path):
    """Three cells across two models so the renderers have
    something representative to work with: trustworthy, learning,
    and fall-through."""
    path = tmp_path / "sc.json"
    sc = ModelScorecard(path)

    # trustworthy
    for _ in range(100):
        sc.record_event(
            "codeql:py/sql-injection", "claude-haiku-4-5",
            EventType.CHEAP_SHORT_CIRCUIT, "correct",
        )
    # learning (n<10)
    for _ in range(5):
        sc.record_event(
            "codeql:cpp/uncontrolled-format", "claude-haiku-4-5",
            EventType.CHEAP_SHORT_CIRCUIT, "correct",
        )
    # fall-through (high miss rate) + samples
    for _ in range(20):
        sc.record_event(
            "codeql:js/path-injection", "claude-haiku-4-5",
            EventType.CHEAP_SHORT_CIRCUIT, "correct",
        )
    for i in range(5):
        sc.record_event(
            "codeql:js/path-injection", "claude-haiku-4-5",
            EventType.CHEAP_SHORT_CIRCUIT, "incorrect",
            sample={
                "this_reasoning": f"cheap thought FP {i}",
                "other_reasoning": f"full found real bug {i}",
            },
        )
    # second model
    for _ in range(50):
        sc.record_event(
            "codeql:py/sql-injection", "gemini-2.5-flash-lite",
            EventType.CHEAP_SHORT_CIRCUIT, "correct",
        )
    return path


def _make_args(**kwargs):
    """Build a Namespace with the union of all defaults the CLI
    handlers expect; tests override specific fields."""
    base = dict(
        path=None, by_savings=False, by_miss_rate=False,
        untrusted=False, learning=False, consumer=None, since=None,
        model_a=None, model_b=None,
        decision_class=None, model=None, as_=None,
        older_than_days=None, all=False,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _capture(handler, args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = handler(args)
    return rc, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_default_shows_all_cells(seeded_scorecard):
    rc, out, _ = _capture(
        cli_mod.cmd_list, _make_args(path=seeded_scorecard),
    )
    assert rc == 0
    # All three decision_classes appear.
    assert "codeql:py/sql-injection" in out
    assert "codeql:cpp/uncontrolled-format" in out
    assert "codeql:js/path-injection" in out


def test_list_by_savings_sorts_descending(seeded_scorecard):
    rc, out, _ = _capture(
        cli_mod.cmd_list,
        _make_args(path=seeded_scorecard, by_savings=True),
    )
    lines = [l for l in out.splitlines() if "codeql:" in l]
    # Highest calls_saved (py/sql-injection on haiku, 100) should
    # appear before js/path-injection (20).
    py_idx = next(i for i, l in enumerate(lines) if "py/sql-injection" in l and "claude" in l)
    js_idx = next(i for i, l in enumerate(lines) if "js/path-injection" in l)
    assert py_idx < js_idx


def test_list_untrusted_filters_to_fall_through_only(seeded_scorecard):
    rc, out, _ = _capture(
        cli_mod.cmd_list,
        _make_args(path=seeded_scorecard, untrusted=True),
    )
    # Only the js/path-injection cell on Haiku is fall-through.
    assert "js/path-injection" in out
    # py/sql-injection (haiku) is short-circuit, must not appear.
    haiku_py_lines = [
        l for l in out.splitlines()
        if "py/sql-injection" in l and "claude-haiku-4-5" in l
    ]
    assert haiku_py_lines == []


def test_list_learning_filters_to_below_floor_only(seeded_scorecard):
    rc, out, _ = _capture(
        cli_mod.cmd_list,
        _make_args(path=seeded_scorecard, learning=True),
    )
    assert "cpp/uncontrolled-format" in out
    assert "py/sql-injection" not in out


def test_list_consumer_filter_prefix_matches(seeded_scorecard):
    """Cells starting with ``codeql:`` match ``--consumer codeql``.
    Defends the operator's expectation that prefix-filtering works
    without trailing colons."""
    rc, out, _ = _capture(
        cli_mod.cmd_list,
        _make_args(path=seeded_scorecard, consumer="codeql"),
    )
    assert "codeql:py/sql-injection" in out


def test_list_consumer_filter_excludes_other_prefixes(seeded_scorecard):
    """A scorecard with both codeql and (synthetic) sca cells →
    --consumer codeql shows only codeql."""
    sc = ModelScorecard(seeded_scorecard)
    sc.record_event(
        "sca:major_bump:PyPI", "claude-haiku-4-5",
        EventType.CHEAP_SHORT_CIRCUIT, "correct",
    )
    rc, out, _ = _capture(
        cli_mod.cmd_list,
        _make_args(path=seeded_scorecard, consumer="codeql"),
    )
    assert "sca:major_bump" not in out
    assert "codeql:py/sql-injection" in out


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_shows_overlap_only(seeded_scorecard):
    """``compare`` only shows decision_classes BOTH models have
    seen — comparing where there's no shared evidence is
    misleading."""
    rc, out, _ = _capture(
        cli_mod.cmd_compare,
        _make_args(
            path=seeded_scorecard,
            model_a="claude-haiku-4-5",
            model_b="gemini-2.5-flash-lite",
        ),
    )
    assert "codeql:py/sql-injection" in out
    # cpp/uncontrolled-format and js/path-injection were only seen
    # by haiku — must not appear in the comparison output.
    assert "cpp/uncontrolled-format" not in out
    assert "js/path-injection" not in out


def test_compare_no_overlap_returns_helpful_message(seeded_scorecard):
    rc, out, _ = _capture(
        cli_mod.cmd_compare,
        _make_args(
            path=seeded_scorecard,
            model_a="claude-haiku-4-5",
            model_b="some-model-no-data",
        ),
    )
    assert "no decision_classes" in out


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------


def test_samples_renders_disagreements(seeded_scorecard):
    rc, out, _ = _capture(
        cli_mod.cmd_samples,
        _make_args(
            path=seeded_scorecard,
            decision_class="codeql:js/path-injection",
        ),
    )
    assert rc == 0
    assert "Sample 1" in out
    assert "cheap thought FP" in out
    assert "full found real bug" in out


def test_samples_unknown_class_returns_nonzero(seeded_scorecard):
    rc, out, err = _capture(
        cli_mod.cmd_samples,
        _make_args(path=seeded_scorecard, decision_class="nope:nope"),
    )
    assert rc == 1


# ---------------------------------------------------------------------------
# pin / unpin
# ---------------------------------------------------------------------------


def test_pin_sets_policy_override(seeded_scorecard):
    args = _make_args(
        path=seeded_scorecard,
        decision_class="codeql:py/sql-injection",
        model="claude-haiku-4-5",
        as_="force_fall_through",
    )
    _capture(cli_mod.cmd_pin, args)
    sc = ModelScorecard(seeded_scorecard)
    stat = sc.get_stat("codeql:py/sql-injection", "claude-haiku-4-5")
    assert stat.policy_override == "force_fall_through"


def test_unpin_releases_to_auto(seeded_scorecard):
    sc = ModelScorecard(seeded_scorecard)
    sc.set_policy_override(
        "codeql:py/sql-injection", "claude-haiku-4-5",
        "force_short_circuit",
    )
    args = _make_args(
        path=seeded_scorecard,
        decision_class="codeql:py/sql-injection",
        model="claude-haiku-4-5",
    )
    _capture(cli_mod.cmd_unpin, args)
    stat = sc.get_stat("codeql:py/sql-injection", "claude-haiku-4-5")
    assert stat.policy_override == "auto"


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_single_decision_class(seeded_scorecard):
    args = _make_args(
        path=seeded_scorecard,
        decision_class="codeql:py/sql-injection",
    )
    rc, _, err = _capture(cli_mod.cmd_reset, args)
    assert rc == 0
    sc = ModelScorecard(seeded_scorecard)
    remaining = {s.decision_class for s in sc.get_stats()}
    assert "codeql:py/sql-injection" not in remaining


def test_reset_by_model(seeded_scorecard):
    args = _make_args(path=seeded_scorecard, model="claude-haiku-4-5")
    rc, _, _ = _capture(cli_mod.cmd_reset, args)
    sc = ModelScorecard(seeded_scorecard)
    remaining = {(s.model, s.decision_class) for s in sc.get_stats()}
    assert all(m != "claude-haiku-4-5" for (m, _) in remaining)


def test_reset_all(seeded_scorecard):
    args = _make_args(path=seeded_scorecard, all=True)
    _capture(cli_mod.cmd_reset, args)
    sc = ModelScorecard(seeded_scorecard)
    assert sc.get_stats() == []


# ---------------------------------------------------------------------------
# Smoke test: the shim actually executes
# ---------------------------------------------------------------------------


def test_libexec_shim_runs(tmp_path):
    """End-to-end: the shim is executable and dispatches to the CLI.
    Empty scorecard → "(no scorecard data)" message.

    Sets ``_RAPTOR_TRUSTED=1`` to bypass the inline trust-marker
    check the shim shares with every libexec script — that gate is
    designed to refuse bare-shell invocation but allow tests."""
    repo_root = Path(__file__).resolve().parents[4]
    shim = repo_root / "libexec" / "raptor-llm-scorecard"
    if not shim.exists():
        pytest.skip("shim not present (running outside the repo)")
    sc_path = tmp_path / "empty.json"
    env = {**os.environ, "_RAPTOR_TRUSTED": "1"}
    out = subprocess.run(
        [str(shim), "--path", str(sc_path), "list"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    assert out.returncode == 0, out.stderr
    assert "no scorecard data" in out.stdout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_since_durations():
    assert cli_mod._parse_since("7d") == _dt.timedelta(days=7)
    assert cli_mod._parse_since("12h") == _dt.timedelta(hours=12)
    assert cli_mod._parse_since("30m") == _dt.timedelta(minutes=30)
    assert cli_mod._parse_since("90s") == _dt.timedelta(seconds=90)


def test_parse_since_rejects_garbage():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        cli_mod._parse_since("abc")


def test_humanise_age_recent():
    now = _dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=_dt.timezone.utc)
    ten_min_ago = (now - _dt.timedelta(minutes=10)).isoformat()
    assert cli_mod._humanise_age(ten_min_ago, now=now) == "10m ago"


def test_humanise_age_days():
    now = _dt.datetime(2026, 5, 6, 12, 0, 0, tzinfo=_dt.timezone.utc)
    three_days_ago = (now - _dt.timedelta(days=3, hours=2)).isoformat()
    assert cli_mod._humanise_age(three_days_ago, now=now) == "3d ago"
