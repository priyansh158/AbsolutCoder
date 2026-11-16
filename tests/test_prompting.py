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
