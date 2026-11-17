"""The AbsolutCoder terminal header: a slanted block-letter wordmark that builds in.

A port of the design system's Node reference (``terminal/absolutcoder-banner.js``).
The glyph tables and timing constants are copied from it verbatim.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping

from rich.console import Console
from rich.control import Control
from rich.segment import ControlType
from rich.text import Text

from absolutcoder.render import output_lock

H = 6
RAW = {
    "A": [" ##### ", "##   ##", "##   ##", "#######", "##   ##", "##   ##"],
    "b": ["##    ", "##    ", "##### ", "##  ##", "##  ##", "##### "],
    "s": ["     ", "     ", " ####", "###  ", "  ###", "#### "],
    "o": ["      ", "      ", " #### ", "##  ##", "##  ##", " #### "],
    "l": ["##", "##", "##", "##", "##", "##"],
    "u": ["      ", "      ", "##  ##", "##  ##", "##  ##", " #####"],
    "t": ["     ", " ##  ", "#####", " ##  ", " ##  ", "  ###"],
    "C": [" ######", "##     ", "##     ", "##     ", "##     ", " ######"],
    "d": ["    ##", "    ##", " #####", "##  ##", "##  ##", " #####"],
    "e": ["      ", "      ", " #### ", "######", "##    ", " #####"],
    "r": ["     ", "     ", "## ##", "#### ", "##   ", "##   "],
}
MARK = ["###"] * 6  # the slanted accent block
WORD = "AbsolutCoder"
GAP = 1  # columns between letters
MARK_GAP = 3  # columns between the mark and "A"
LEAN = 1  # italic: each row up shifts this many columns right
INDENT = 2  # left margin
FRAME = 22  # ms between frames
STAGGER = 70  # ms between letter starts
LETTER_START = 140  # ms before "A" starts
LETTER_DUR = 300  # ms a letter takes to land
MARK_DUR = 240  # ms the mark takes to slide in
SLIDE = 6  # a letter starts this many columns left of its slot
EXTRA_LEAN = 3  # and leans this many extra columns at the top, straightening as it lands

ACCENT = "#ff5b1f"
HINT_BEFORE = "  Type a task, or "
HINT_COMMAND = "/help"
HINT_AFTER = " for commands."
NAME = "AbsolutCoder"


def _width(glyph: list[str]) -> int:
    return max(len(row) for row in glyph)


def _slots() -> tuple[list[int], int]:
    x = INDENT + len(MARK[0]) + MARK_GAP
    slots = []
    for letter in WORD:
        slots.append(x)
        x += _width(RAW[letter]) + GAP
    return slots, x


SLOTS, _END = _slots()
WIDTH: int = _END - GAP + LEAN * (H - 1) + 1
TOTAL: int = LETTER_START + (len(WORD) - 1) * STAGGER + LETTER_DUR


def js_round(value: float) -> int:
    """JavaScript's ``Math.round``: halves round toward positive infinity."""

    return math.floor(value + 0.5)


def ease_out(p: float) -> float:
    return 1 - (1 - min(max(p, 0.0), 1.0)) ** 3


def _stamp(cells: list[list[str | None]], glyph: list[str], x0: int, extra_lean: int, paint: str) -> None:
    for r, row in enumerate(glyph):
        row_from_bottom = H - 1 - r
        shift = LEAN * row_from_bottom + js_round(extra_lean * row_from_bottom / (H - 1))
        for c, char in enumerate(row):
            if char != "#":
                continue
            column = x0 + c + shift
            if 0 <= column < WIDTH:
                cells[r][column] = paint


def frame_cells(t: float) -> list[list[str | None]]:
    """H rows x WIDTH columns at ``t`` ms; each cell is None, "accent" or "ink"."""

    cells: list[list[str | None]] = [[None] * WIDTH for _ in range(H)]
    mark_x = INDENT - js_round((1 - ease_out(t / MARK_DUR)) * SLIDE)
    _stamp(cells, MARK, mark_x, 0, "accent")
    for i, letter in enumerate(WORD):
        start = LETTER_START + i * STAGGER
        if t < start:
            continue
        p = (t - start) / LETTER_DUR
        e = ease_out(p)
        x0 = SLOTS[i] - js_round((1 - e) * SLIDE)
        extra = js_round((1 - e) * EXTRA_LEAN)
        _stamp(cells, RAW[letter], x0, extra, "accent" if p < 1 else "ink")
    return cells


def frame_lines(t: float, *, block: str = "█") -> list[str]:
    """Plain text rows of the frame at ``t`` ms, trailing spaces removed."""

    return ["".join(block if cell else " " for cell in row).rstrip() for row in frame_cells(t)]


# The mid-size wordmark, for windows too narrow for the full one. Each glyph is
# drawn on a grid of square pixels, two pixel rows to a text row (half blocks).
MID_PIXEL_ROWS = 6
MID_ROWS = MID_PIXEL_ROWS // 2
MID_RAW = {
    "A": [".#.", "#.#", "#.#", "###", "#.#", "#.#"],
    "b": ["#..", "#..", "##.", "#.#", "#.#", "##."],
    "s": ["...", "...", ".##", "##.", ".##", "##."],
    "o": ["...", "...", ".#.", "#.#", "#.#", ".#."],
    "l": ["#", "#", "#", "#", "#", "#"],
    "u": ["...", "...", "#.#", "#.#", "#.#", ".##"],
    "t": ["...", ".#.", "###", ".#.", ".#.", ".##"],
    "C": [".##", "#..", "#..", "#..", "#..", ".##"],
    "d": ["..#", "..#", ".##", "#.#", "#.#", ".##"],
    "e": ["...", "...", ".##", "#.#", "##.", ".##"],
    "r": ["...", "...", "#.#", "##.", "#..", "#.."],
}
MID_MARK = ["##"] * MID_PIXEL_ROWS
MID_MARK_GAP = 2
MID_SLIDE = 4


def _mid_slots() -> tuple[list[int], int]:
    x = INDENT + len(MID_MARK[0]) + MID_MARK_GAP
    slots = []
    for letter in WORD:
        slots.append(x)
        x += _width(MID_RAW[letter]) + GAP
    return slots, x


MID_SLOTS, _MID_END = _mid_slots()
MID_WIDTH: int = _MID_END - GAP + LEAN * (MID_ROWS - 1) + 1


def _mid_stamp(pixels: list[list[str | None]], glyph: list[str], x0: int, paint: str) -> None:
    for r, row in enumerate(glyph):
        shift = LEAN * (MID_ROWS - 1 - r // 2)  # lean by text row, so each cell's two pixels align
        for c, char in enumerate(row):
            column = x0 + c + shift
            if char == "#" and 0 <= column < MID_WIDTH:
                pixels[r][column] = paint


def mid_frame_cells(t: float) -> list[list[tuple[str, str] | None]]:
    """MID_ROWS x MID_WIDTH at ``t`` ms; a cell is None or (half-block char, paint)."""

    pixels: list[list[str | None]] = [[None] * MID_WIDTH for _ in range(MID_PIXEL_ROWS)]
    mark_x = INDENT - js_round((1 - ease_out(t / MARK_DUR)) * MID_SLIDE)
    _mid_stamp(pixels, MID_MARK, mark_x, "accent")
    for i, letter in enumerate(WORD):
        start = LETTER_START + i * STAGGER
        if t < start:
            continue
        p = (t - start) / LETTER_DUR
        x0 = MID_SLOTS[i] - js_round((1 - ease_out(p)) * MID_SLIDE)
        _mid_stamp(pixels, MID_RAW[letter], x0, "accent" if p < 1 else "ink")
    cells: list[list[tuple[str, str] | None]] = []
    for r in range(MID_ROWS):
        top, bottom = pixels[2 * r], pixels[2 * r + 1]
        row: list[tuple[str, str] | None] = []
        for upper, lower in zip(top, bottom, strict=True):
            if upper and lower:
                row.append(("█", lower))
            elif upper:
                row.append(("▀", upper))
            elif lower:
                row.append(("▄", lower))
            else:
                row.append(None)
        cells.append(row)
    return cells


def mid_frame_lines(t: float) -> list[str]:
    """Plain text rows of the mid-size frame at ``t`` ms, trailing spaces removed."""

    return ["".join(cell[0] if cell else " " for cell in row).rstrip() for row in mid_frame_cells(t)]


def facts_line(version: str | None, model: str | None, cwd: str | None) -> str | None:
    parts = []
    if version:
        parts.append(f"v{version}")
    if model:
        parts.append(f"model {model}")
    if cwd:
        parts.append(cwd)
    if not parts:
        return None
    return "  " + "  ·  ".join(parts)


def _is_set(environ: Mapping[str, str], name: str) -> bool:
    return bool(environ.get(name))


class _Painter:
    """Builds styled rich text; with ``color`` off every style is left out."""

    def __init__(self, color: bool, block: str) -> None:
        self.color = color
        self.block = block

    def style(self, name: str | None) -> str:
        return (name or "") if self.color else ""

    def row(self, cells: list[str | None]) -> Text:
        text = Text()
        last = max((i for i, cell in enumerate(cells) if cell), default=-1)
        start = 0
        while start <= last:  # one append per run of equal cells
            paint = cells[start]
            end = start
            while end + 1 <= last and cells[end + 1] == paint:
                end += 1
            count = end - start + 1
            if paint is None:
                text.append(" " * count)
            else:
                text.append(self.block * count, style=self.style(ACCENT if paint == "accent" else "bold"))
            start = end + 1
        return text

    def mid_row(self, cells: list[tuple[str, str] | None]) -> Text:
        text = Text()
        last = max((i for i, cell in enumerate(cells) if cell), default=-1)
        for cell in cells[: last + 1]:
            if cell is None:
                text.append(" ")
            else:
                text.append(cell[0], style=self.style(ACCENT if cell[1] == "accent" else "bold"))
        return text

    def compact(self, typed: int, *, mark: str, done: bool) -> Text:
        text = Text("  ")
        text.append(mark, style=self.style(ACCENT))
        text.append(" ")
        name = NAME[:typed]
        if done or not name:
            text.append(name, style=self.style("bold italic"))
        else:
            text.append(name[:-1], style=self.style("bold italic"))
            text.append(name[-1], style=self.style(f"bold italic {ACCENT}"))
        return text

    def hint(self) -> Text:
        text = Text(HINT_BEFORE, style=self.style("dim"))
        text.append(HINT_COMMAND, style=self.style(ACCENT))
        text.append(HINT_AFTER, style=self.style("dim"))
        return text


_PRINT = {"markup": False, "highlight": False, "soft_wrap": True}
_ERASE_TO_END = Control((ControlType.ERASE_IN_LINE, 0))


def show(
    console: Console,
    *,
    version: str | None = None,
    model: str | None = None,
    cwd: str | None = None,
    animate: bool | None = None,
    ascii_only: bool | None = None,
    compact: bool | None = None,
    environ: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Print the header through ``console``; never raises because of the terminal."""

    env = os.environ if environ is None else environ
    if ascii_only is None:
        ascii_only = _is_set(env, "AC_ASCII")
    with output_lock(console):
        painter = _Painter(not _is_set(env, "NO_COLOR"), "#" if ascii_only else "█")
        mark = "//" if ascii_only else "▰"
        facts = facts_line(version, model, cwd)
        try:
            # Half blocks need a font with block characters, so ASCII mode skips the mid size.
            # The small wordmark is the default; AC_BIG asks for the full six-row one.
            big = (ascii_only or _is_set(env, "AC_BIG")) and console.width >= WIDTH + 1
            mid = compact is None and not ascii_only and not big and console.width >= MID_WIDTH + 1
            if compact is None:
                compact = not big and not mid
            if animate is None:
                animate = (
                    console.is_terminal
                    and not console.legacy_windows
                    and not _is_set(env, "AC_NO_ANIM")
                    and not console.is_dumb_terminal
                )
            console.print(**_PRINT)
            if mid:

                def mid_rows(t: float) -> list[Text]:
                    return [painter.mid_row(cells) for cells in mid_frame_cells(t)]

                if animate:
                    _animated(console, sleep, clock, mid_rows, MID_ROWS)
                else:
                    for row in mid_rows(math.inf):
                        console.print(row, **_PRINT)
            elif compact:
                _compact(console, painter, mark, animate, sleep)
            elif animate:
                _animated(console, sleep, clock, lambda t: [painter.row(cells) for cells in frame_cells(t)], H)
            else:
                for cells in frame_cells(math.inf):
                    console.print(painter.row(cells), **_PRINT)
            _footer(console, painter, facts)
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001 - the header is decoration: fall back to one plain line
            try:
                console.print(painter.compact(len(NAME), mark=mark, done=True), **_PRINT)
                _footer(console, painter, facts)
            except Exception:  # noqa: BLE001 - the console itself is broken; nothing more to show
                pass


def _footer(console: Console, painter: _Painter, facts: str | None) -> None:
    console.print(**_PRINT)
    if facts is not None:
        console.print(Text(facts, style=painter.style("dim")), **_PRINT)
    console.print(painter.hint(), **_PRINT)
    console.print(**_PRINT)


def _animated(
    console: Console,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    rows: Callable[[float], list[Text]],
    height: int,
) -> None:
    """Reserve ``height`` rows, then redraw them every FRAME ms until the resting frame."""

    below = 0  # rows the cursor is below the first reserved row

    def draw(t: float) -> None:
        nonlocal below
        if below:
            console.control(Control.move_to_column(0, -below))
        below = 0
        for row in rows(t):
            console.print(row, end="", **_PRINT)
            console.control(_ERASE_TO_END)
            console.print(**_PRINT)
            below += 1

    console.control(Control.show_cursor(False))
    try:
        for _ in range(height):
            console.print(**_PRINT)
        below = height
        start = clock()
        try:
            while True:
                t = (clock() - start) * 1000
                if t >= TOTAL:
                    draw(math.inf)
                    break
                draw(t)
                sleep(FRAME / 1000)
        except KeyboardInterrupt:
            draw(math.inf)
    finally:
        console.control(Control.show_cursor(True))


def _compact(console: Console, painter: _Painter, mark: str, animate: bool, sleep: Callable[[float], None]) -> None:
    """The one-line form; animated, it is typed one letter per STAGGER ms."""

    if not animate:
        console.print(painter.compact(len(NAME), mark=mark, done=True), **_PRINT)
        return
    carriage_return = Control(ControlType.CARRIAGE_RETURN)
    console.control(Control.show_cursor(False))
    try:
        try:
            for typed in range(1, len(NAME) + 1):
                console.control(carriage_return)
                console.print(painter.compact(typed, mark=mark, done=False), end="", **_PRINT)
                console.control(_ERASE_TO_END)
                sleep(STAGGER / 1000)
        except KeyboardInterrupt:
            pass
        console.control(carriage_return)
        console.print(painter.compact(len(NAME), mark=mark, done=True), end="", **_PRINT)
        console.control(_ERASE_TO_END)
        console.print(**_PRINT)
    finally:
        console.control(Control.show_cursor(True))
