"""Terminal input that survives lines printed by other threads.

On a real terminal the prompt is read with ``prompt_toolkit``, and while the
loop runs, other threads' output is drawn above the prompt instead of through
what the user is typing. Anywhere else (pipes, tests) the builtin ``input`` is
used and nothing is patched.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

from rich.console import Console


def make_input_fn(console: Console) -> Callable[[str], str]:
    """A line reader for the loop: a ``prompt_toolkit`` session on a terminal, else ``input``.

    History is kept in memory only. Any failure to set the session up falls
    back to ``input``. ``console`` is the console the loop prints to.
    """

    if not _interactive():
        return input
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory

        session = PromptSession(history=InMemoryHistory())
    except Exception:  # noqa: BLE001 - no usable terminal session: plain input still works
        return input

    def read(prompt: str = "") -> str:
        return session.prompt(prompt)

    return read


def background_output(console: Console) -> AbstractContextManager[None]:
    """While entered, lines other threads print appear above the prompt; a no-op off a terminal."""

    if not _interactive():
        return nullcontext()
    try:
        from prompt_toolkit.patch_stdout import patch_stdout
    except Exception:  # noqa: BLE001 - without prompt_toolkit, input is the builtin one anyway
        return nullcontext()
    return patch_stdout(raw=True)


def _interactive() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):  # a replaced or closed stream
        return False
