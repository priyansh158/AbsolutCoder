"""One-shot exit codes, non-terminal output and the --auto flag."""

import io
import json

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

from absolutcoder.cli import build_parser, utf8_when_piped
from absolutcoder.config import Config, config_path, save_config
from absolutcoder.loops import VerifyOutcome
from absolutcoder.render import StreamRenderer
from absolutcoder.repl import Repl, run_cli
from tests.support import role_models, scripted_model

_CONFIG = Config(main_model="openai:a", subagent_model="openai:b")


def test_piped_cp1252_stream_prints_symbols():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        Console(file=stream, force_terminal=False).print("✓ done →")
        stream.flush()
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    utf8_when_piped(stream)
    Console(file=stream, force_terminal=False).print("✓ done →")
    stream.flush()
    assert raw.getvalue().decode("utf-8").strip() == "✓ done →"
    utf8_when_piped(object())  # a stream without reconfigure is left alone


def test_long_line_is_not_wrapped_when_not_a_terminal():
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=False, width=80))
    text = " ".join(["word"] * 60)
    renderer.finish(type("Snap", (), {"values": {"messages": [AIMessage(content=text, id="m1")]}})())
    assert output.getvalue().splitlines() == [text]


def _fake(**state):
    class FakeRepl:
        def __init__(self, repo, config):
            self.pending_attachments = []
            self.config = config
            self.last_turn_interrupted = False
            self.last_turn_failed = False

        def run_task(self, text, paths=None):
            if state.get("raise_interrupt"):
                raise KeyboardInterrupt
            self.last_turn_interrupted = state.get("interrupted", False)
            self.last_turn_failed = state.get("failed", False)
            return state.get("outcome")

    return FakeRepl


@pytest.mark.parametrize(
    ("state", "code"),
    [
        ({}, 0),
        ({"failed": True}, 1),
        ({"interrupted": True}, 130),
        ({"raise_interrupt": True}, 130),
        ({"outcome": VerifyOutcome(False, 3, "boom")}, 2),
        ({"outcome": VerifyOutcome(True, 1, "ok"), "failed": True}, 0),
    ],
)
def test_one_shot_exit_codes(tmp_path, monkeypatch, state, code):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["fix it"])
    assert run_cli(args, _CONFIG, repl_factory=_fake(**state)) == code


def _real_factory(made: list, **repl_kwargs):
    worker = scripted_model(lambda _m: AIMessage(content="w"))

    def factory(repo_path, config, background=True):
        repl = Repl(
            repo_path,
            config,
            chat_model=scripted_model(lambda _m: AIMessage(content="ok")),
            model_factory=role_models({}, default=worker),
            console=Console(file=io.StringIO(), force_terminal=False, width=200),
            background=background,
            **repl_kwargs,
        )
        made.append(repl)
        return repl

    return factory


def test_one_shot_error_line_goes_to_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    made: list = []
    real = _real_factory(made)

    def factory(repo_path, config, background=True):
        repl = real(repo_path, config, background)

        def broken(*_a, **_k):
            raise RuntimeError("stream broke")

        repl.agent.stream = broken
        return repl

    code = run_cli(build_parser().parse_args(["go"]), _CONFIG, repl_factory=factory)
    assert code == 1
    assert "Error: stream broke" in capsys.readouterr().err
    assert "Error:" not in made[0].console.file.getvalue()


def test_auto_flag_is_for_this_run_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_config(config_path(), _CONFIG)
    before = config_path().read_text(encoding="utf-8")
    seen = []

    class FakeRepl:
        def __init__(self, repo, config):
            seen.append(config)
            self.pending_attachments = []

        def run_task(self, text, paths=None):
            return None

    assert run_cli(build_parser().parse_args(["go", "--auto"]), None, repl_factory=FakeRepl) == 0
    assert seen[0].auto_run is True
    assert config_path().read_text(encoding="utf-8") == before
    assert json.loads(before)["auto_run"] is False
    assert run_cli(build_parser().parse_args(["go"]), None, repl_factory=FakeRepl) == 0
    assert seen[1].auto_run is False


def test_auto_flag_is_not_saved_by_later_setting_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_config(config_path(), _CONFIG)
    made: list = []
    run_cli(
        build_parser().parse_args(["--auto"]),
        None,
        repl_factory=_real_factory(made, input_fn=lambda _p="": "/exit"),
    )
    repl = made[0]
    assert repl.config.auto_run is True
    repl.handle_line("/verify make test")
    saved = json.loads(config_path().read_text(encoding="utf-8"))
    assert saved["auto_run"] is False
    assert saved["verify_command"] == "make test"
    assert repl.config.auto_run is True


def test_until_timeout_overrides_for_one_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = []

    class FakeRepl:
        def __init__(self, repo, config):
            seen.append(config)
            self.pending_attachments = []

        def run_task(self, text, paths=None):
            return None

    config = Config(main_model="openai:a", subagent_model="openai:b", verify_timeout=1800)
    args = build_parser().parse_args(["go", "--until", "make", "--until-timeout", "42"])
    run_cli(args, config, repl_factory=FakeRepl)
    assert seen[0].verify_timeout == 42
    run_cli(build_parser().parse_args(["go"]), config, repl_factory=FakeRepl)
    assert seen[1].verify_timeout == 1800
    with pytest.raises(SystemExit):
        build_parser().parse_args(["go", "--until-timeout", "0"])


def test_until_timeout_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_config(config_path(), _CONFIG)
    made: list = []
    args = build_parser().parse_args(["--until-timeout", "42"])
    run_cli(args, None, repl_factory=_real_factory(made, input_fn=lambda _p="": "/exit"))
    repl = made[0]
    assert repl.config.verify_timeout == 42
    repl.handle_line("/verify make test")
    assert json.loads(config_path().read_text(encoding="utf-8"))["verify_timeout"] == 1800


def test_help_epilog_names_commands_config_and_flags(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    text = capsys.readouterr().out
    for expected in ("/model", "/verify", "/quit", "config.json", "provider:model", "anthropic:", "--until", "--auto"):
        assert expected in text, expected
