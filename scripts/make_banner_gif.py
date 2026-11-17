"""Draw docs/banner.gif: the full AbsolutCoder wordmark building in, as the terminal draws it.

The frames come from ``absolutcoder.banner``, so the GIF matches the header.
Needs Pillow (not a project dependency):

    pip install pillow
    python scripts/make_banner_gif.py [output.gif]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from absolutcoder import banner  # noqa: E402

CELL_W, CELL_H = 12, 24  # one terminal cell, in pixels
PAD_X, PAD_TOP, PAD_BOTTOM = 36, 44, 40
HINT_GAP = 30  # pixels between the wordmark and the hint line
BACKGROUND = (17, 17, 17)
INK = (244, 241, 234)
ACCENT = (255, 91, 31)  # banner.ACCENT, #ff5b1f
DIM = (138, 138, 138)

STEP = 30  # ms per frame while the letters land
TYPE_STEP = 30  # ms per typed hint character
BLINK = 520  # ms the cursor stays on or off
BLINKS = 4  # on/off cycles before the loop restarts
HINT = "Type a task, or /help for commands."
HINT_ACCENT = "/help"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/CascadiaMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


FONT = _font(18)
WIDTH = PAD_X * 2 + banner.WIDTH * CELL_W
HINT_Y = PAD_TOP + banner.H * CELL_H + HINT_GAP
HEIGHT = HINT_Y + 22 + PAD_BOTTOM
HINT_X = PAD_X + banner.INDENT * CELL_W


SCALE = 3  # drawn this many times larger, then reduced, for smooth edges


def _wordmark(t: float) -> Image.Image:
    """The wordmark at ``t`` ms. The terminal leans one cell per row in steps;
    here each cell is a parallelogram, so the same lean reads as a smooth italic."""

    w, h = CELL_W * SCALE, CELL_H * SCALE
    big = Image.new("RGB", (WIDTH * SCALE, (PAD_TOP + banner.H * CELL_H + 8) * SCALE), BACKGROUND)
    draw = ImageDraw.Draw(big)
    for r, row in enumerate(banner.frame_cells(t)):
        y0 = (PAD_TOP + r * CELL_H) * SCALE
        c = 0
        while c < len(row):
            cell = row[c]
            if not cell:
                c += 1
                continue
            end = c
            while end + 1 < len(row) and row[end + 1] == cell:
                end += 1
            x0, x1 = (PAD_X * SCALE) + c * w, (PAD_X * SCALE) + (end + 1) * w
            half = w / 2  # top edge half a cell right, bottom edge half a cell left
            draw.polygon(
                [(x0 + half, y0), (x1 + half, y0), (x1 - half, y0 + h), (x0 - half, y0 + h)],
                fill=ACCENT if cell == "accent" else INK,
            )
            c = end + 1
    return big.resize((WIDTH, big.height // SCALE), Image.Resampling.LANCZOS)


def _canvas(t: float) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    image.paste(_wordmark(t), (0, 0))
    return image, ImageDraw.Draw(image)


def _hint(draw: ImageDraw.ImageDraw, typed: int, cursor: bool) -> None:
    x = HINT_X
    start = HINT.index(HINT_ACCENT)
    for i, char in enumerate(HINT[:typed]):
        colour = ACCENT if start <= i < start + len(HINT_ACCENT) else DIM
        draw.text((x, HINT_Y), char, font=FONT, fill=colour)
        x += draw.textlength(char, font=FONT)
    if cursor:
        draw.rectangle([x + 2, HINT_Y + 1, x + 11, HINT_Y + 21], fill=ACCENT)


def frames() -> tuple[list[Image.Image], list[int]]:
    images: list[Image.Image] = []
    durations: list[int] = []

    def add(image: Image.Image, ms: int) -> None:
        images.append(image.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        durations.append(ms)

    add(_canvas(0)[0], 200)
    t = 0
    while t < banner.TOTAL:
        add(_canvas(t)[0], STEP)
        t += STEP
    for typed in range(1, len(HINT) + 1):
        image, draw = _canvas(math.inf)
        _hint(draw, typed, cursor=True)
        add(image, TYPE_STEP)
    for _ in range(BLINKS):
        for cursor in (False, True):
            image, draw = _canvas(math.inf)
            _hint(draw, len(HINT), cursor=cursor)
            add(image, BLINK)
    return images, durations


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "banner.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    images, durations = frames()
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print(f"wrote {out} ({len(images)} frames, {out.stat().st_size // 1024} KB, {WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
