"""The AbsolutCoder terminal header: frames, layout and how it is printed."""

import io
import math
import re
import threading

from rich.console import Console

from absolutcoder import banner
from absolutcoder.banner import H, INDENT, LETTER_DUR, LETTER_START, MARK, STAGGER, TOTAL, WIDTH
from absolutcoder.render import output_lock

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_HINT = "  Type a task, or /help for commands."
_RESTING = banner.frame_lines(math.inf)
# Derived from the glyph tables by the layout rules (the brief's string had the
# "s" one column right and a five-cell second "o"; the tables give this).
_BOTTOM = "  ███   ██   ██ █████  ████   ████  ██  █████   ███  ██████  ████   █████  █████ ██"


class FakeTime:
    def __init__(self, interrupt_on: int | None = None) -> None:
        self.now = 0.0
        self.calls: list[float] = []
        self.interrupt_on = interrupt_on

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.interrupt_on is not None and len(self.calls) == self.interrupt_on:
            raise KeyboardInterrupt
        self.now += seconds


def _console(*, terminal: bool, width: int = 120, **kwargs) -> tuple[Console, io.StringIO]:
    output = io.StringIO()
    console = Console(
        file=output, force_terminal=terminal, width=width, legacy_windows=False, **kwargs
    )
    return console, output


def _plain_lines(output: io.StringIO) -> list[str]:
    return _ANSI.sub("", output.getvalue()).split("\n")


def _rows_before_footer(lines: list[str]) -> list[str]:
    """The six rows drawn last: above the blank line, the facts line and the hint."""

    hint = max(i for i, line in enumerate(lines) if line == _HINT)
    return lines[hint - 2 - H : hint - 2]


def test_width_and_total():
    assert WIDTH == 92
    assert TOTAL == 1210


def test_resting_frame_shape():
    assert len(_RESTING) == H
    for r, row in enumerate(_RESTING):
        assert len(row) <= WIDTH
        assert row == row.rstrip()
        lead = INDENT + (H - 1 - r)
        assert row[:lead] == " " * lead
        assert row[lead : lead + 3] == "███"
        assert row[lead + 3] == " "
    assert _RESTING[-1] == _BOTTOM


def test_resting_cells_are_ink_except_the_mark():
    cells = banner.frame_cells(math.inf)
    accent = {(r, c) for r in range(H) for c in range(WIDTH) if cells[r][c] == "accent"}
    mark = {(r, INDENT + (H - 1 - r) + c) for r in range(H) for c in range(len(MARK[0]))}
    assert accent == mark
    assert len(accent) == 18
    letters = [cell for row in cells for cell in row if cell is not None and cell != "accent"]
    assert letters and set(letters) == {"ink"}


def test_time_zero_shows_only_the_mark_sliding_in():
    cells = banner.frame_cells(0)
    filled = [(r, c, cells[r][c]) for r in range(H) for c in range(WIDTH) if cells[r][c]]
    assert filled
    for r, c, paint in filled:
        assert paint == "accent"
        assert c < INDENT + (H - 1 - r) + len(MARK[0])


def _frame_with_word(monkeypatch, word: str, t: float):
    with monkeypatch.context() as patch:
        patch.setattr(banner, "WORD", word)
        return banner.frame_cells(t)


def test_letters_arrive_in_order(monkeypatch):
    t = LETTER_START + 3 * STAGGER + 1
    full = banner.frame_cells(t)
    assert full == _frame_with_word(monkeypatch, banner.WORD[:4], t)  # nothing from letter five on
    assert full != _frame_with_word(monkeypatch, banner.WORD[:3], t)  # the fourth letter is there
    assert "ink" not in {cell for row in full for cell in row}  # all four still moving
    # Once "A" has landed it is ink while the letters after it are still accent.
    later = LETTER_START + LETTER_DUR + 1
    cells = banner.frame_cells(later)
    only_a = _frame_with_word(monkeypatch, "A", later)
    ink = {(r, c) for r in range(H) for c in range(WIDTH) if cells[r][c] == "ink"}
    assert ink
    assert all(only_a[r][c] == "ink" for r, c in ink)
    assert cells != _frame_with_word(monkeypatch, "A", later)


def test_frames_converge():
    assert banner.frame_lines(TOTAL) == _RESTING
    assert banner.frame_lines(TOTAL + 5000) == _RESTING


def test_js_rounding_rounds_halves_up():
    assert banner.js_round(0.5) == 1
    assert banner.js_round(2.5) == 3
    assert banner.js_round(-0.5) == 0
    assert banner.js_round(-1.5) == -1


def test_ascii_mode_uses_hash():
    assert all("█" not in row for row in banner.frame_lines(math.inf, block="#"))
    for width in (120, 60):
        console, output = _console(terminal=False, width=width)
        banner.show(console, version="1", animate=False, ascii_only=True, environ={})
        text = output.getvalue()
        assert "█" not in text and "▰" not in text
        assert ("#" in text) == (width == 120)
    assert "// AbsolutCoder" in text


def test_facts_line():
    assert banner.facts_line("0.1.0", "openai:gpt-5", "~/code/app") == (
        "  v0.1.0  ·  model openai:gpt-5  ·  ~/code/app"
    )
    assert banner.facts_line("0.1.0", None, "~/code/app") == "  v0.1.0  ·  ~/code/app"
    assert banner.facts_line(None, None, None) is None


def test_static_show_prints_header_facts_and_hint():
    console, output = _console(terminal=False)
    banner.show(console, version="0.1.0", model="openai:m", cwd="~/r", animate=False, environ={})
    text = output.getvalue()
    lines = text.split("\n")
    for row in _RESTING:
        assert row in lines
    assert "  v0.1.0  ·  model openai:m  ·  ~/r" in lines
    assert _HINT in lines
    assert lines[0] == ""  # a blank line precedes the wordmark
    assert "\x1b" not in text and "\r" not in text


def test_narrow_console_gets_the_one_line_form():
    console, output = _console(terminal=False, width=60)
    banner.show(console, version="0.1.0", animate=False, environ={})
    lines = output.getvalue().split("\n")
    assert "  ▰ AbsolutCoder" in lines
    assert all(row not in lines for row in _RESTING)
    assert "█" not in output.getvalue()
    assert _HINT in lines


def test_animated_show_ends_on_the_resting_frame():
    console, output = _console(terminal=True, color_system="truecolor")
    fake = FakeTime()
    banner.show(console, version="0.1.0", model="m", environ={}, sleep=fake.sleep, clock=fake.clock)
    assert fake.calls and set(fake.calls) == {banner.FRAME / 1000}
    raw = output.getvalue()
    assert raw.index("\x1b[?25l") < raw.rindex("\x1b[?25h")
    assert "\x1b[6A" in raw  # moved up over the reserved rows
    assert "38;2;255;91;31" in raw  # the accent colour
    assert _rows_before_footer(_plain_lines(output)) == _RESTING
    assert fake.now * 1000 >= TOTAL


def test_no_anim_switch_is_static():
    console, output = _console(terminal=True)
    fake = FakeTime()
    banner.show(console, version="0.1.0", environ={"AC_NO_ANIM": "1"}, sleep=fake.sleep, clock=fake.clock)
    assert fake.calls == []
    assert "\x1b[?25l" not in output.getvalue()
    assert _rows_before_footer(_plain_lines(output)) == _RESTING


def test_no_color_switch_emits_no_colour_codes():
    console, output = _console(terminal=True, color_system="truecolor")
    banner.show(console, version="0.1.0", animate=False, environ={"NO_COLOR": "1"})
    text = output.getvalue()
    assert "\x1b[" not in text
    assert _RESTING[-1] in text.split("\n")


def test_ctrl_c_during_animation_restores_cursor_and_returns():
    console, output = _console(terminal=True)
    fake = FakeTime(interrupt_on=3)
    banner.show(console, version="0.1.0", environ={}, sleep=fake.sleep, clock=fake.clock)
    assert len(fake.calls) == 3
    raw = output.getvalue()
    assert raw.rindex("\x1b[?25h") > raw.index("\x1b[?25l")
    assert _rows_before_footer(_plain_lines(output)) == _RESTING


def test_terminal_error_falls_back_to_one_line():
    class Broken(Console):
        def control(self, *control):
            raise OSError("no terminal")

    output = io.StringIO()
    console = Broken(file=output, force_terminal=True, width=120, legacy_windows=False)
    banner.show(console, version="0.1.0", environ={}, sleep=lambda _s: None)
    plain = _plain_lines(output)
    assert "  ▰ AbsolutCoder" in plain
    assert _HINT in plain


def test_show_holds_the_output_lock():
    console, _output = _console(terminal=True)
    lock = output_lock(console)
    seen: list[bool] = []

    def sleep(_seconds: float) -> None:
        free: list[bool] = []

        def probe() -> None:
            got = lock.acquire(blocking=False)
            if got:
                lock.release()
            free.append(got)

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        seen.append(not free[0])
        fake.sleep(_seconds)

    fake = FakeTime()
    banner.show(console, environ={}, sleep=sleep, clock=fake.clock)
    assert seen and all(seen)
