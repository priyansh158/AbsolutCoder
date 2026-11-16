"""The absolut console script prints help and checks provider API keys."""

import json
import os
import subprocess
import sys
from pathlib import Path


def _executable() -> tuple[Path, Path]:
    scripts = Path(sys.executable).parent
    executable = scripts / "absolut.exe"
    if not executable.exists():
        executable = scripts / "absolut"
    return scripts, executable


def test_absolut_help_exits_zero():
    scripts, executable = _executable()
    env = os.environ.copy()
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    env.pop("OPENAI_API_KEY", None)
    completed = subprocess.run(
        [str(executable), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    assert "absolut" in completed.stdout.lower()


def _run_absolut(tmp_path, main_model, subagent_model, drop_keys):
    scripts, executable = _executable()
    payload = {"main_model": main_model, "subagent_model": subagent_model}
    payload = {key: value for key, value in payload.items() if value is not None}
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    env = os.environ.copy()  # keeps PYTHONPATH so the child imports this worktree
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    env["ABSOLUTCODER_HOME"] = str(tmp_path)
    for key in drop_keys:
        env.pop(key, None)
    return subprocess.run(
        [str(executable), "hello"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def test_cli_reports_missing_anthropic_key(tmp_path):
    completed = _run_absolut(
        tmp_path,
        "anthropic:claude-x",
        "anthropic:claude-x",
        ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
    )
    assert completed.returncode == 1
    assert "ANTHROPIC_API_KEY" in completed.stderr
    assert "OPENAI_API_KEY" not in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_reports_missing_openai_key_by_default(tmp_path):
    completed = _run_absolut(tmp_path, "openai:gpt-x", "openai:gpt-x", ["OPENAI_API_KEY"])
    assert completed.returncode == 1
    assert "OPENAI_API_KEY is not set" in completed.stderr


def test_cli_reports_bad_model_id(tmp_path):
    completed = _run_absolut(tmp_path, "a b:x", "openai:gpt-x", ["OPENAI_API_KEY"])
    assert completed.returncode == 1
    assert "Config error" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_reports_incomplete_config(tmp_path):
    completed = _run_absolut(tmp_path, None, "openai:x", ["OPENAI_API_KEY"])
    assert completed.returncode == 1
    assert "Config error" in completed.stderr
    assert "main_model" in completed.stderr
    assert "Traceback" not in completed.stderr
