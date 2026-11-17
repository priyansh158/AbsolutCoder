"""Terminal input that survives lines printed by other threads.

On a real terminal the prompt is read with ``prompt_toolkit``, and while the
loop runs, other threads' output is drawn above the prompt instead of through
what the user is typing. The loop's main prompt is drawn between two
horizontal rules; once a line is entered the rules are erased and the line is
echoed, so everything the tool prints stays above the next prompt. Anywhere
else (pipes, tests) the builtin ``input`` is used and nothing is patched.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

from rich.console import Console
from rich.text import Text

MAIN_PROMPT = "absolut> "  # the loop's own prompt; only this one is framed
FRAMED_PROMPT = "> "
RULE = "─"
RULE_STYLE = "#6b6b6b"
ACCENT = "#ff5b1f"


def rule_text(width: int) -> str:
    """A horizontal rule one column short of ``width``, so it never wraps."""

    return RULE * max(width - 1, 1)


def echo_text(line: str) -> Text:
    """How an entered line is kept in the scrollback once the frame is erased."""

    text = Text(FRAMED_PROMPT, style=ACCENT)
    text.append(line, style="bold")
    return text


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

        history = InMemoryHistory()
        session = PromptSession(history=history)
        framed_app(history)  # fail here, not at the first prompt, if the layout cannot be built
    except Exception:  # noqa: BLE001 - no usable terminal session: plain input still works
        return input

    def read(prompt: str = "") -> str:
        if prompt != MAIN_PROMPT:
            return session.prompt(prompt)
        line = framed_app(history).run()
        if line.strip():
            console.print(echo_text(line), markup=False, highlight=False)
        return line

    return read


def framed_app(history):  # noqa: ANN001, ANN201 - prompt_toolkit types, imported lazily
    """A one-line editor between two rules; ``run()`` returns the entered line.

    The frame is erased when the line is entered. Ctrl+C raises
    ``KeyboardInterrupt`` and Ctrl+D on an empty line raises ``EOFError``, as
    a plain prompt does.
    """

    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.styles import Style

    def accept(buffer: Buffer) -> bool:
        get_app().exit(result=buffer.text)
        return True  # keep the text until the frame is erased

    buffer = Buffer(history=history, multiline=False, accept_handler=accept)

    def rule() -> list[tuple[str, str]]:
        try:
            width = get_app().output.get_size().columns
        except Exception:  # noqa: BLE001 - an odd terminal: a sane default width
            width = 80
        return [("class:rule", rule_text(width))]

    def rule_window() -> Window:
        return Window(FormattedTextControl(rule), height=1, dont_extend_height=True)

    bindings = KeyBindings()

    @bindings.add("c-c")
    def _interrupt(event) -> None:  # noqa: ANN001
        event.app.exit(exception=KeyboardInterrupt())

    @bindings.add("c-d", filter=Condition(lambda: not buffer.text))
    def _end(event) -> None:  # noqa: ANN001
        event.app.exit(exception=EOFError())

    layout = Layout(
        HSplit(
            [
                rule_window(),
                VSplit(
                    [
                        Window(
                            FormattedTextControl([("class:mark", FRAMED_PROMPT)]),
                            width=len(FRAMED_PROMPT),
                            dont_extend_width=True,
                            dont_extend_height=True,
                        ),
                        Window(BufferControl(buffer), wrap_lines=True, dont_extend_height=True),
                    ]
                ),
                rule_window(),
            ]
        ),
        focused_element=buffer,
    )
    return Application(
        layout=layout,
        key_bindings=bindings,
        style=Style.from_dict({"rule": RULE_STYLE, "mark": f"{ACCENT} bold"}),
        erase_when_done=True,
        full_screen=False,
    )


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
