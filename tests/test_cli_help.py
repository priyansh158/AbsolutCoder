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


def test_cli_reports_non_object_providers(tmp_path):
    scripts, executable = _executable()
    payload = {"main_model": "openai:a", "subagent_model": "openai:b", "providers": []}
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    env["ABSOLUTCODER_HOME"] = str(tmp_path)
    completed = subprocess.run(
        [str(executable), "hello"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    assert completed.returncode == 1
    assert "Config error" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_until_flag_sets_exit_code(tmp_path, monkeypatch):
    from absolutcoder.cli import build_parser
    from absolutcoder.config import Config
    from absolutcoder.loops import VerifyOutcome
    from absolutcoder.repl import run_cli

    monkeypatch.chdir(tmp_path)
    seen: list = []

    def factory_returning(outcome):
        class FakeRepl:
            def __init__(self, repo, config):
                seen.append(config)
                self.pending_attachments = []

            def run_task(self, text, paths=None):
                seen.append((text, paths))
                return outcome

        return FakeRepl

    config = Config(main_model="openai:a", subagent_model="openai:b", verify_command="make", max_rounds=5)
    args = build_parser().parse_args(["fix it", "--until", "pytest -q", "--max-rounds", "3"])
    assert run_cli(args, config, repl_factory=factory_returning(VerifyOutcome(False, 3, "boom"))) == 2
    assert seen[0].verify_command == "pytest -q"
    assert seen[0].max_rounds == 3
    assert seen[1] == ("fix it", [])
    assert config.verify_command == "make"
    assert run_cli(args, config, repl_factory=factory_returning(VerifyOutcome(True, 1, "ok"))) == 0
    plain = build_parser().parse_args(["fix it"])
    assert run_cli(plain, config, repl_factory=factory_returning(None)) == 0
    assert seen[-2].verify_command == "make"


def test_until_without_task_is_an_error(capsys):
    import pytest

    from absolutcoder.cli import main

    with pytest.raises(SystemExit) as raised:
        main(["--until", "x"])
    assert raised.value.code == 2
    assert "--until needs a task" in capsys.readouterr().err
    with pytest.raises(SystemExit) as raised:
        main(["do it", "--max-rounds", "0"])
    assert raised.value.code == 2
