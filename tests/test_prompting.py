"""Terminal input: prompt_toolkit on a real terminal, the builtin input otherwise."""

import builtins
import io
import sys

from rich.console import Console

from absolutcoder.prompting import background_output, make_input_fn


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, no_color=True, width=100)


def _pretend_terminal(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)


def test_non_tty_uses_builtin_input():
    console = _console()
    assert make_input_fn(console) is builtins.input
    with background_output(console) as entered:
        assert entered is None
    with background_output(console):
        pass  # usable more than once


def test_falls_back_when_prompt_toolkit_is_missing(monkeypatch):
    _pretend_terminal(monkeypatch)
    for name in [key for key in sys.modules if key == "prompt_toolkit" or key.startswith("prompt_toolkit.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "prompt_toolkit", None)  # the import now raises ImportError
    console = _console()
    assert make_input_fn(console) is builtins.input
    with background_output(console):
        pass


def test_falls_back_when_session_creation_fails(monkeypatch):
    import prompt_toolkit

    def broken(*_args, **_kwargs):
        raise RuntimeError("no console here")

    _pretend_terminal(monkeypatch)
    monkeypatch.setattr(prompt_toolkit, "PromptSession", broken)
    assert make_input_fn(_console()) is builtins.input


def test_rule_is_one_column_short_of_the_terminal():
    from absolutcoder import prompting

    assert prompting.rule_text(40) == "─" * 39
    assert prompting.rule_text(0) == "─"


def test_entered_line_is_echoed_with_the_prompt_mark():
    from absolutcoder import prompting

    assert prompting.echo_text("fix the bug").plain == "> fix the bug"


def _framed_run(keys):
    import io

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    from absolutcoder import prompting

    buffer = io.StringIO()
    output = Vt100_Output(buffer, lambda: Size(rows=24, columns=30), term="xterm")
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        pipe.send_text(keys)
        return prompting.framed_app(InMemoryHistory()).run(), buffer.getvalue()


def test_framed_prompt_draws_a_rule_above_and_below_and_returns_the_line():
    line, drawn = _framed_run("one two\x1b[D\x1b[DX\r")
    assert line == "one tXwo"
    first_frame = drawn.split("> ")
    assert "─" * 29 in first_frame[0]
    assert "─" * 29 in first_frame[1]


def test_framed_prompt_keeps_ctrl_c_and_ctrl_d():
    import pytest

    with pytest.raises(KeyboardInterrupt):
        _framed_run("\x03")
    with pytest.raises(EOFError):
        _framed_run("\x04")
