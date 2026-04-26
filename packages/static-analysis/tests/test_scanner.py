"""Tests for packages/static-analysis/scanner.py."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

# static-analysis has a hyphen — load via importlib
_SCANNER_PATH = Path(__file__).parent.parent / "scanner.py"
_spec = importlib.util.spec_from_file_location("static_analysis_scanner", _SCANNER_PATH)
_scanner_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
_spec.loader.exec_module(_scanner_mod)

run_codeql = _scanner_mod.run_codeql


# ---------------------------------------------------------------------------
# run_codeql()
# ---------------------------------------------------------------------------

class TestRunCodeql:

    def test_returns_empty_list_when_codeql_not_installed(self, tmp_path):
        with patch("shutil.which", return_value=None):
            result = run_codeql(tmp_path, tmp_path / "out", ["python"])
        assert result == []

    @patch("shutil.which", return_value="/usr/bin/codeql")
    @patch.object(_scanner_mod, "run")
    def test_creates_output_dir(self, mock_run, mock_which, tmp_path):
        mock_run.return_value = (1, "", "db create failed")
        out_dir = tmp_path / "codeql_out"
        run_codeql(tmp_path, out_dir, ["python"])
        assert out_dir.exists()

    @patch("shutil.which", return_value="/usr/bin/codeql")
    @patch.object(_scanner_mod, "run")
    def test_skips_language_if_db_create_fails(self, mock_run, mock_which, tmp_path):
        mock_run.return_value = (1, "", "database create error")
        result = run_codeql(tmp_path, tmp_path / "out", ["python", "java"])
        assert result == []

    @patch("shutil.which", return_value="/usr/bin/codeql")
    @patch.object(_scanner_mod, "run")
    def test_uses_list_based_args(self, mock_run, mock_which, tmp_path):
        """run() must be called with list args, never shell strings."""
        mock_run.return_value = (1, "", "")
        run_codeql(tmp_path, tmp_path / "out", ["python"])
        for c in mock_run.call_args_list:
            cmd_arg = c.args[0] if c.args else c.kwargs.get("cmd", [])
            assert isinstance(cmd_arg, list), "Command must be a list (no shell injection)"

    @patch("shutil.which", return_value="/usr/bin/codeql")
    @patch.object(_scanner_mod, "run")
    def test_empty_languages_returns_empty(self, mock_run, mock_which, tmp_path):
        result = run_codeql(tmp_path, tmp_path / "out", [])
        assert result == []
        mock_run.assert_not_called()

